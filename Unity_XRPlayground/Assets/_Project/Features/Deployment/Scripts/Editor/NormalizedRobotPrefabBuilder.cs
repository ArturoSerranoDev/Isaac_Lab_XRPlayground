using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    public static class NormalizedRobotPrefabBuilder
    {
        [MenuItem("XRPlayground/Deployment/Normalize Selected URDF Robot")]
        static void NormalizeSelected()
        {
            GameObject selected = Selection.activeGameObject;
            if (selected == null)
                throw new InvalidOperationException("Select an imported URDF robot root first");
            RobotDefinition definition = selected.GetComponent<ArticulationRobotDriver>()?.definition ??
                Selection.objects.OfType<RobotDefinition>().SingleOrDefault();
            if (definition == null)
                throw new InvalidOperationException(
                    "Assign a RobotDefinition to an ArticulationRobotDriver on the selected root, " +
                    "or select exactly one RobotDefinition asset with the imported root");
            if (!StationCatalog.TryLoad(out StationCatalog catalog, out string catalogError))
                throw new InvalidOperationException(catalogError);
            StationCatalogRobotAsset robotAsset = catalog.FindRobotAsset(definition.robotId) ??
                throw new InvalidOperationException(
                    $"Robot definition '{definition.robotId}' is absent from deployment/stations.json");
            if (!robotAsset.redistribution_verified)
                throw new InvalidOperationException(
                    $"Robot '{definition.robotId}' cannot be normalized until redistribution provenance is verified");
            if (string.IsNullOrWhiteSpace(definition.definitionSha256))
                throw new InvalidOperationException(
                    "Robot definition must record the canonical live Isaac physics SHA-256");
            string path = EditorUtility.SaveFilePanelInProject(
                "Save normalized robot prefab", selected.name + "_Normalized", "prefab",
                "Choose the committed normalized prefab location");
            if (string.IsNullOrEmpty(path))
                return;
            GameObject clone = UnityEngine.Object.Instantiate(selected);
            clone.name = selected.name;
            try
            {
                foreach (MonoBehaviour component in clone.GetComponentsInChildren<MonoBehaviour>(true))
                {
                    if (component == null)
                        continue;
                    string typeName = component.GetType().FullName ?? string.Empty;
                    if (typeName.StartsWith("Unity.Robotics.UrdfImporter", StringComparison.Ordinal))
                        UnityEngine.Object.DestroyImmediate(component);
                }
                if (clone.GetComponentInChildren<ArticulationBody>(true) == null)
                    throw new InvalidOperationException(
                        "The imported robot has no ArticulationBody hierarchy; import it as an articulation");
                ArticulationRobotDriver driver = clone.GetComponent<ArticulationRobotDriver>();
                if (driver == null)
                    driver = clone.AddComponent<ArticulationRobotDriver>();
                driver.definition = definition;
                driver.articulationRoot = FindArticulationRoot(clone);
                driver.configureFromDefinitionOnAwake = true;
                if (!driver.TryValidateHierarchy(out string hierarchyError))
                    throw new InvalidOperationException(hierarchyError);
                OfflineRigIdentity identity = clone.GetComponent<OfflineRigIdentity>();
                if (identity == null)
                    identity = clone.AddComponent<OfflineRigIdentity>();
                identity.robotId = definition.robotId;
                identity.definitionSha256 = definition.definitionSha256;
                identity.projectAuthoredProcedural = false;
                identity.robotDefinition = definition;
                identity.articulationDriver = driver;
                PrefabUtility.SaveAsPrefabAsset(clone, path);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(clone);
            }
        }

        [MenuItem("XRPlayground/Deployment/Install Selected Normalized Robot Rig")]
        static void InstallSelectedNormalizedRig()
        {
            GameObject prefab = Selection.activeObject as GameObject;
            string prefabPath = prefab != null ? AssetDatabase.GetAssetPath(prefab) : string.Empty;
            if (prefab == null || string.IsNullOrWhiteSpace(prefabPath) ||
                PrefabUtility.GetPrefabAssetType(prefab) == PrefabAssetType.NotAPrefab)
                throw new InvalidOperationException("Select a normalized robot prefab asset first");
            InstallNormalizedRig(prefab);
        }

        public static IReadOnlyList<GameObject> InstallNormalizedRig(GameObject prefab)
        {
            string prefabPath = prefab != null ? AssetDatabase.GetAssetPath(prefab) : string.Empty;
            if (prefab == null || string.IsNullOrWhiteSpace(prefabPath) ||
                PrefabUtility.GetPrefabAssetType(prefab) == PrefabAssetType.NotAPrefab)
                throw new InvalidOperationException("A normalized robot prefab asset is required");
            OfflineRigIdentity sourceIdentity = prefab.GetComponent<OfflineRigIdentity>() ??
                throw new InvalidOperationException("Selected prefab has no OfflineRigIdentity");
            if (!StationCatalog.TryLoad(out StationCatalog catalog, out string error))
                throw new InvalidOperationException(error);
            StationCatalogEntry[] destinations = catalog.stations
                .Where(station => station.robot_id == sourceIdentity.robotId).ToArray();
            if (destinations.Length == 0)
                throw new InvalidOperationException(
                    $"No catalog station uses robot '{sourceIdentity.robotId}'");
            var installed = new List<GameObject>();
            foreach (StationCatalogEntry station in destinations)
            {
                if (!sourceIdentity.ValidateForStation(catalog, station, false, out error))
                    throw new InvalidOperationException(error);
                GameObject shell = GameObject.Find($"Deployment Runtime/{station.station_id}") ??
                    throw new InvalidOperationException(
                        $"Station shell '{station.station_id}' is absent; build V2 runtime shells first");
                StationRuntime runtime = shell.GetComponent<StationRuntime>() ??
                    throw new InvalidOperationException(
                        $"Station shell '{station.station_id}' has no StationRuntime");
                Transform authored = DeploymentSceneSetup.FindAuthoredOfflineRigForStation(shell.transform);
                if (authored != null)
                {
                    OfflineRigIdentity existingIdentity = authored.GetComponent<OfflineRigIdentity>();
                    if (existingIdentity != null &&
                        existingIdentity.robotId == sourceIdentity.robotId &&
                        existingIdentity.definitionSha256 == sourceIdentity.definitionSha256)
                    {
                        installed.Add(authored.gameObject);
                        continue;
                    }
                    throw new InvalidOperationException(
                        $"Station '{station.station_id}' already owns authored rig '{authored.name}'");
                }
                Transform pending = shell.transform.Find("OfflineRig_PENDING_NORMALIZED_PREFAB");
                if (pending != null)
                    UnityEngine.Object.DestroyImmediate(pending.gameObject);
                Transform anchor = DeploymentSceneSetup.FindAnchorForStation(station.station_id);
                var wrapper = new GameObject($"OfflineRig_{sourceIdentity.robotId}");
                wrapper.transform.SetParent(shell.transform, false);
                wrapper.transform.SetPositionAndRotation(anchor.position, anchor.rotation);
                GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, wrapper.transform);
                instance.name = "Robot";
                Vector3 offset = DeploymentFrameConverter.IsaacToUnity(
                    DeploymentFrameConverter.Vector3From(
                        station.offline_bindings.robot_position_isaac));
                instance.transform.localPosition = offset;
                instance.transform.localRotation = Quaternion.identity;
                OfflineRigIdentity childIdentity = instance.GetComponent<OfflineRigIdentity>();
                OfflineRigIdentity wrapperIdentity = wrapper.AddComponent<OfflineRigIdentity>();
                wrapperIdentity.robotId = childIdentity.robotId;
                wrapperIdentity.definitionSha256 = childIdentity.definitionSha256;
                wrapperIdentity.projectAuthoredProcedural = false;
                wrapperIdentity.robotDefinition = childIdentity.robotDefinition;
                wrapperIdentity.articulationDriver = childIdentity.articulationDriver;
                UnityEngine.Object.DestroyImmediate(childIdentity);
                wrapper.SetActive(false);
                runtime.offlineRig = wrapper;
                EditorUtility.SetDirty(runtime);
                installed.Add(wrapper);
            }
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            EditorSceneManager.SaveOpenScenes();
            Debug.Log(
                $"Installed normalized '{sourceIdentity.robotId}' articulation in " +
                $"{destinations.Length} station shell(s). Task adapters and physical task objects " +
                "must still be authored before OfflinePolicy can activate.");
            return installed;
        }

        static ArticulationBody FindArticulationRoot(GameObject root)
        {
            ArticulationBody[] bodies = root.GetComponentsInChildren<ArticulationBody>(true);
            foreach (ArticulationBody body in bodies)
                if (body.isRoot)
                    return body;
            throw new InvalidOperationException("Imported articulation has no reduced-coordinate root body");
        }
    }
}
