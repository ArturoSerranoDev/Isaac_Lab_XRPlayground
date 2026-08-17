using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    public static class DeploymentSceneSetup
    {
        sealed class SceneStation
        {
            public string Id;
            public string PolicyId;
            public string VisualPath;
            public string AnchorPath;
            public string[] ObjectPaths;
            public string[] ObjectIds;
            public bool OwnsVisual = true;
        }

        static readonly SceneStation[] Stations =
        {
            new()
            {
                Id = "ball_catch", PolicyId = "ball_catch.throw",
                VisualPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Kinova_Jaco2_j2n7s300",
                AnchorPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Kinova_Jaco2_j2n7s300",
                ObjectPaths = new[] { "Interactables/Ball" }, ObjectIds = new[] { "ball" },
            },
            new()
            {
                Id = "conveyor_color", PolicyId = "conveyor_color.sort",
                VisualPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_B_Conveyor/UR10e_Robotiq",
                AnchorPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_B_Conveyor",
                ObjectPaths = Enumerable.Range(0, 4).Select(index =>
                    $"Environment/Lab Zones/01_Manipulation/Robot Stations/Station_B_Conveyor/ObjectSlots/Object_{index}").ToArray(),
                ObjectIds = Enumerable.Range(0, 4).Select(index => $"object_{index}").ToArray(),
            },
            new()
            {
                Id = "pick_place_table", PolicyId = "pick_place_table.place",
                VisualPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_C_PickPlace/Agibot_A2D",
                AnchorPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_C_PickPlace",
                ObjectPaths = new[] { "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_C_PickPlace/Piece_0" },
                ObjectIds = new[] { "piece_0" },
            },
            new()
            {
                Id = "balance_bot", PolicyId = "balance_bot.two_ball",
                VisualPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_D_BalanceBot/BalanceBot_Tray",
                AnchorPath = "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_D_BalanceBot",
                ObjectPaths = new[]
                {
                    "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_D_BalanceBot/Ball_0",
                    "Environment/Lab Zones/01_Manipulation/Robot Stations/Station_D_BalanceBot/Ball_1",
                },
                ObjectIds = new[] { "ball_0", "ball_1" },
            },
            new()
            {
                Id = "spot_loco", PolicyId = "spot.locomotion",
                VisualPath = "Environment/Lab Zones/02_Locomotion/Robot Stations/Station_E_Spot/Spot_BD",
                AnchorPath = "Environment/Lab Zones/02_Locomotion/Robot Stations/Station_E_Spot",
                ObjectPaths = Array.Empty<string>(), ObjectIds = Array.Empty<string>(), OwnsVisual = false,
            },
            new()
            {
                Id = "spot_follow", PolicyId = "spot.follow",
                VisualPath = "Environment/Lab Zones/02_Locomotion/Robot Stations/Station_E_Spot/Spot_BD",
                AnchorPath = "Environment/Lab Zones/02_Locomotion/Robot Stations/Station_E_Spot",
                ObjectPaths = Array.Empty<string>(), ObjectIds = Array.Empty<string>(),
            },
        };

        static readonly HashSet<string> LegacyComponentNames = new()
        {
            "RosTcpClient", "RosHeartbeatPublisher", "OnnxPolicyRunner", "OfflineJointDriver",
            "BallCatchOfflinePolicyController", "ConveyorOfflinePolicyController",
            "PickPlaceOfflinePolicyController", "BalanceBotOfflinePolicyController",
            "SpotOfflinePolicyController", "KinovaLinkPoseFollower", "RobotLinkPoseFollower",
            "AgibotLinkPoseFollower", "BalanceBotLinkPoseFollower", "SpotLinkPoseFollower",
        };

        [MenuItem("XRPlayground/Deployment/Build V2 Runtime Shells")]
        public static void Build()
        {
            if (!StationCatalog.TryLoad(out StationCatalog catalog, out string error))
                throw new InvalidOperationException(error);
            GameObject root = GameObject.Find("Deployment Runtime");
            if (root == null)
                root = new GameObject("Deployment Runtime");
            if (root.GetComponent<DeploymentCatalogBinder>() == null)
                root.AddComponent<DeploymentCatalogBinder>();
            DeploymentModeController modeController = GetOrAdd<DeploymentModeController>(root);
            modeController.binder = root.GetComponent<DeploymentCatalogBinder>();

            foreach (SceneStation source in Stations)
            {
                StationCatalogEntry station = catalog.Find(source.Id) ??
                    throw new InvalidOperationException($"Catalog station missing: {source.Id}");
                Transform visual = Find(source.VisualPath);
                Transform anchor = Find(source.AnchorPath);
                Transform shell = root.transform.Find(source.Id);
                if (shell == null)
                {
                    var shellObject = new GameObject(source.Id);
                    shellObject.transform.SetParent(root.transform, false);
                    shell = shellObject.transform;
                }
                StationRuntime runtime = GetOrAdd<StationRuntime>(shell.gameObject);
                PolicyRuntime policy = GetOrAdd<PolicyRuntime>(shell.gameObject);
                GoldenTraceRecorder recorder = GetOrAdd<GoldenTraceRecorder>(shell.gameObject);
                UnityScenarioEvaluationRunner evaluation =
                    GetOrAdd<UnityScenarioEvaluationRunner>(shell.gameObject);
                OpenLoopActionPlayer openLoop = GetOrAdd<OpenLoopActionPlayer>(shell.gameObject);
                policy.expectedPolicyId = source.PolicyId;
                policy.expectedStationId = source.Id;
                recorder.station = runtime;
                evaluation.station = runtime;
                openLoop.station = runtime;
                openLoop.recorder = recorder;

                Transform mirrorNode = shell.Find("MirrorRig");
                if (mirrorNode == null)
                {
                    var mirrorObject = new GameObject("MirrorRig");
                    mirrorObject.transform.SetParent(shell, false);
                    mirrorNode = mirrorObject.transform;
                }
                LengthPrefixedJsonClient transport = GetOrAdd<LengthPrefixedJsonClient>(mirrorNode.gameObject);
                BridgeV2Client bridge = GetOrAdd<BridgeV2Client>(mirrorNode.gameObject);
                MirrorRig mirror = GetOrAdd<MirrorRig>(mirrorNode.gameObject);
                transport.host = catalog.bridge.host;
                transport.port = station.port;
                transport.maxMessageBytes = catalog.bridge.max_message_bytes;
                bridge.transport = transport;
                bridge.stationId = source.Id;
                bridge.staleAfterSeconds = catalog.bridge.stale_after_seconds;
                bridge.disconnectAfterSeconds = catalog.bridge.disconnect_after_seconds;
                mirror.bridge = bridge;
                mirror.environmentAnchor = anchor;
                mirror.links = LinkBindings(visual);
                mirror.objects = ObjectBindings(source.ObjectPaths, source.ObjectIds);
                mirror.visualRoots = source.OwnsVisual ? new[] { visual.gameObject } : Array.Empty<GameObject>();

                Transform offline = FindAuthoredOfflineRig(shell);
                if (offline == null)
                {
                    offline = shell.Find("OfflineRig_PENDING_NORMALIZED_PREFAB");
                    if (offline == null)
                    {
                        var offlineObject = new GameObject("OfflineRig_PENDING_NORMALIZED_PREFAB");
                        offlineObject.transform.SetParent(shell, false);
                        offline = offlineObject.transform;
                    }
                }
                offline.gameObject.SetActive(false);
                runtime.stationId = source.Id;
                runtime.policy = policy;
                runtime.mirrorRig = mirror;
                runtime.offlineRig = offline.gameObject;
                runtime.mode = source.Id == "spot_loco" ? StationMode.Disabled : StationMode.Mirror;
            }
            DisableLegacyRuntime();
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            EditorSceneManager.SaveOpenScenes();
            Debug.Log("Built catalog-driven bridge-v2 runtime shells. Offline rigs remain fail-closed until normalized physical prefabs and promoted bundles are assigned.");
        }

        [MenuItem("XRPlayground/Deployment/Audit Runtime Readiness")]
        public static void Audit()
        {
            if (!StationCatalog.TryLoad(out StationCatalog catalog, out string catalogError))
                throw new InvalidOperationException(catalogError);
            GameObject root = GameObject.Find("Deployment Runtime");
            if (root == null)
            {
                Debug.LogWarning("Deployment Runtime is absent. Run Build V2 Runtime Shells.");
                return;
            }
            foreach (SceneStation station in Stations)
            {
                Transform shell = root.transform.Find(station.Id);
                StationRuntime runtime = shell != null ? shell.GetComponent<StationRuntime>() : null;
                OfflineRigIdentity identity = runtime?.offlineRig != null
                    ? runtime.offlineRig.GetComponent<OfflineRigIdentity>()
                    : null;
                StationCatalogEntry catalogStation = catalog.Find(station.Id);
                StationCatalogRobotAsset robotAsset = catalog.FindRobotAsset(
                    catalogStation.robot_id);
                string rigError = robotAsset != null && !robotAsset.redistribution_verified
                    ? $"robot '{catalogStation.robot_id}' provenance is unverified"
                    : "missing rig identity";
                bool physical = identity != null &&
                    identity.ValidateForStation(catalog, catalogStation, false, out rigError);
                if (!physical &&
                    !UsdArticulationPrefabBuilder.CanBuild(
                        catalogStation.robot_id, out string prefabBlocker))
                    rigError = prefabBlocker;
                bool adapter = runtime?.adapterBehaviour is IStationAdapter;
                bool bundle = runtime?.policy != null && runtime.policy.modelAsset != null &&
                    runtime.policy.policyContractAsset != null;
                Debug.Log($"Deployment {station.Id}: mirror={(runtime?.mirrorRig != null)}, " +
                          $"physical={physical}, adapter={adapter}, bundle={bundle}" +
                          (physical ? string.Empty : $", rig_error={rigError}"));
            }
        }

        static T GetOrAdd<T>(GameObject target) where T : Component
        {
            T existing = target.GetComponent<T>();
            return existing != null ? existing : target.AddComponent<T>();
        }

        static Transform FindAuthoredOfflineRig(Transform shell)
        {
            foreach (Transform child in shell)
                if (child.name.StartsWith("OfflineRig_", StringComparison.Ordinal) &&
                    child.name != "OfflineRig_PENDING_NORMALIZED_PREFAB")
                    return child;
            return null;
        }

        internal static Transform FindAuthoredOfflineRigForStation(Transform shell) =>
            FindAuthoredOfflineRig(shell);

        internal static Transform FindAnchorForStation(string stationId)
        {
            SceneStation station = Stations.FirstOrDefault(item => item.Id == stationId) ??
                throw new InvalidOperationException($"Unknown scene station '{stationId}'");
            return Find(station.AnchorPath);
        }

        static Transform Find(string path) => GameObject.Find(path)?.transform ??
            throw new InvalidOperationException($"Scene object not found: {path}");

        static NamedTransformBinding[] LinkBindings(Transform visual)
        {
            var names = new HashSet<string>();
            var result = new List<NamedTransformBinding>();
            foreach (Transform child in visual.GetComponentsInChildren<Transform>(true))
                if (names.Add(child.name))
                    result.Add(new NamedTransformBinding { id = child.name, target = child });
            return result.ToArray();
        }

        static NamedTransformBinding[] ObjectBindings(string[] paths, string[] ids)
        {
            var result = new List<NamedTransformBinding>();
            for (int index = 0; index < paths.Length; index++)
            {
                GameObject target = GameObject.Find(paths[index]);
                if (target != null)
                    result.Add(new NamedTransformBinding { id = ids[index], target = target.transform });
            }
            return result.ToArray();
        }

        static void DisableLegacyRuntime()
        {
            foreach (MonoBehaviour behaviour in UnityEngine.Object.FindObjectsByType<MonoBehaviour>(
                         FindObjectsInactive.Include))
                if (behaviour != null && LegacyComponentNames.Contains(behaviour.GetType().Name))
                {
                    Undo.RecordObject(behaviour, "Disable legacy deployment runtime");
                    behaviour.enabled = false;
                    EditorUtility.SetDirty(behaviour);
                }
        }
    }
}
