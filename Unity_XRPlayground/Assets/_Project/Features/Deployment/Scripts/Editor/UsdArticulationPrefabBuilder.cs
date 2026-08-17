using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    /// <summary>
    /// Converts the checked-in Isaac USD render/collision hierarchy into a
    /// reduced-coordinate Unity articulation using only exported Isaac topology.
    /// </summary>
    public static class UsdArticulationPrefabBuilder
    {
        sealed class SourceSpec
        {
            public string robotId;
            public string usdPath;
            public string prefabPath;
            public bool fixedBase;
            public bool permitVisualMeshColliders;
            public bool augmentPreparedRobotiq;
            public string exactLinkParentName;
            public bool installInStations;
            public bool auxiliaryLoopAdapterVerified;
            public float loopProxyMass = 0.1f;
        }

        public sealed class TopologySummary
        {
            public string rootLink;
            public string[] externalLoopJointNames;
            public string[] auxiliaryTreeJointNames;
        }

        sealed class TopologyPlan
        {
            public string rootLink;
            public readonly HashSet<string> treeJoints = new();
            public readonly HashSet<string> treeFixedJoints = new();
            public readonly HashSet<string> treeAuxiliaryJoints = new();
            public readonly List<RobotJointDefinition> externalJoints = new();
            public readonly List<RobotAuxiliaryJointDefinition> externalAuxiliaryJoints = new();
        }

        static readonly SourceSpec[] Sources =
        {
            new SourceSpec
            {
                robotId = "kinova_jaco2_n7s300",
                usdPath = "Assets/_Project/Features/Robots/KinovaJaco2/USD/j2n7s300_instanceable.usd",
                prefabPath = "Assets/_Project/Features/Robots/KinovaJaco2/Prefabs/KinovaJaco2_Normalized.prefab",
                fixedBase = true,
                permitVisualMeshColliders = false,
                installInStations = true,
            },
            new SourceSpec
            {
                robotId = "ur10e_robotiq_2f85",
                usdPath = "Assets/_Project/Features/Robots/UR10e/USD/configuration/ur10e_base.usd",
                prefabPath = "Assets/_Project/Features/Robots/UR10e/Prefabs/UR10eRobotiq2F85_Normalized.prefab",
                fixedBase = true,
                permitVisualMeshColliders = true,
                augmentPreparedRobotiq = true,
                exactLinkParentName = "ur10e",
                installInStations = true,
            },
            new SourceSpec
            {
                robotId = "spot",
                usdPath = "Assets/_Project/Features/Robots/Spot/USD/spot.usd",
                prefabPath = "Assets/_Project/Features/Robots/Spot/Prefabs/Spot_Normalized.prefab",
                fixedBase = false,
                // The checked-in Isaac Spot USD uses one reduced mesh for both
                // display and collision on the 13 colliding bodies.
                permitVisualMeshColliders = true,
                installInStations = true,
            },
            new SourceSpec
            {
                robotId = "agibot_a2d",
                usdPath = "Assets/_Project/Features/Robots/AgibotA2D/USD/A2D_physics.usd",
                prefabPath = "Assets/_Project/Features/Robots/AgibotA2D/Prefabs/AgibotA2D_Normalized.prefab",
                fixedBase = true,
                permitVisualMeshColliders = false,
                installInStations = false,
                // The live definition records four revolute four-bar closures.
                // The tree planner keeps the *_2 edges in the articulation and
                // routes the four passive duplicate-child edges through proxies.
                auxiliaryLoopAdapterVerified = true,
                loopProxyMass = 0.1f,
            },
        };

        [MenuItem("XRPlayground/Deployment/Build And Install USD Articulation Prefabs")]
        public static void BuildAndInstallAll()
        {
            RobotDefinitionImporter.ImportAllDefinitions();
            foreach (SourceSpec source in Sources)
            {
                if (!CanBuild(source.robotId, out string blocker))
                {
                    Debug.LogWarning(
                        $"Skipped normalized '{source.robotId}' prefab: {blocker}");
                    continue;
                }
                GameObject prefab = Build(source);
                if (source.installInStations)
                {
                    foreach (GameObject rig in NormalizedRobotPrefabBuilder.InstallNormalizedRig(prefab))
                        OfflineStationTaskBuilder.Configure(rig);
                }
                else
                {
                    Debug.Log(
                        $"Built provenance-gated '{source.robotId}' prefab but did not install it. " +
                        "Record and verify its redistribution license in deployment/stations.json first.");
                }
            }
            AssetDatabase.SaveAssets();
            Debug.Log(
                "Built and installed all eligible definition-bound articulations. " +
                "OfflinePolicy remains fail-closed until calibration and policy promotion pass.");
        }

        public static bool CanBuild(string robotId, out string blocker)
        {
            SourceSpec source = Sources.SingleOrDefault(item => item.robotId == robotId);
            if (source == null)
            {
                blocker = $"no checked-in USD source is configured for '{robotId}'";
                return false;
            }
            string definitionPath =
                $"Assets/_Project/Features/Deployment/RobotDefinitions/{robotId}/robot.definition.asset";
            RobotDefinition definition = AssetDatabase.LoadAssetAtPath<RobotDefinition>(definitionPath);
            var blockers = new List<string>();
            if (definition == null)
                blockers.Add($"missing RobotDefinition '{definitionPath}'");
            if (AssetDatabase.LoadAssetAtPath<GameObject>(source.usdPath) == null)
                blockers.Add($"USD '{source.usdPath}' is not imported as a GameObject root");
            if (!StationCatalog.TryLoad(out StationCatalog catalog, out string catalogError))
                blockers.Add(catalogError);
            else
            {
                StationCatalogRobotAsset asset = catalog.FindRobotAsset(robotId);
                if (asset == null)
                    blockers.Add("robot is absent from the authored catalog provenance table");
                else if (!asset.redistribution_verified)
                    blockers.Add("redistribution provenance is unverified");
            }
            if (definition != null)
            {
                try
                {
                    TopologyPlan topology = BuildTopologyPlan(definition);
                    int loopCount = topology.externalJoints.Count +
                        topology.externalAuxiliaryJoints.Count;
                    if (loopCount > 0 && !source.auxiliaryLoopAdapterVerified)
                        blockers.Add(
                            $"{loopCount} loop closures require a verified Unity loop adapter");
                }
                catch (InvalidOperationException exception)
                {
                    blockers.Add(exception.Message);
                }
            }
            blocker = blockers.Count == 0 ? null : string.Join("; ", blockers);
            return blockers.Count == 0;
        }

        public static GameObject Build(string robotId)
        {
            SourceSpec source = Sources.SingleOrDefault(item => item.robotId == robotId) ??
                throw new InvalidOperationException(
                    $"No checked-in USD articulation source is configured for '{robotId}'");
            RobotDefinitionImporter.ImportAllDefinitions();
            if (!CanBuild(robotId, out string blocker))
                throw new InvalidOperationException(
                    $"Robot '{robotId}' normalized prefab is gated: {blocker}");
            return Build(source);
        }

        static GameObject Build(SourceSpec source)
        {
            string definitionPath =
                $"Assets/_Project/Features/Deployment/RobotDefinitions/{source.robotId}/robot.definition.asset";
            RobotDefinition definition = AssetDatabase.LoadAssetAtPath<RobotDefinition>(definitionPath) ??
                throw new InvalidOperationException($"Missing RobotDefinition '{definitionPath}'");
            GameObject usdAsset = AssetDatabase.LoadAssetAtPath<GameObject>(source.usdPath) ??
                throw new InvalidOperationException(
                    $"USD '{source.usdPath}' is not imported as a GameObject root");
            EnsureFolder(System.IO.Path.GetDirectoryName(source.prefabPath).Replace('\\', '/'));

            GameObject clone = (GameObject)PrefabUtility.InstantiatePrefab(usdAsset);
            clone.name = source.robotId + "_Normalized";
            if (PrefabUtility.IsPartOfPrefabInstance(clone))
                PrefabUtility.UnpackPrefabInstance(
                    clone, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
            try
            {
                StripImporterComponents(clone);
                HashSet<string> synthesizedLinks = source.augmentPreparedRobotiq
                    ? AugmentPreparedRobotiq(clone, definition)
                    : new HashSet<string>();
                Dictionary<string, Transform> links = FindExactLinks(
                    clone, definition, source.exactLinkParentName, synthesizedLinks);
                TopologyPlan topology = BuildTopologyPlan(definition);
                ApplyTopology(links, definition, topology, synthesizedLinks);
                AddArticulationBodies(
                    links, definition, topology, source.fixedBase);
                AddLoopClosureBinding(
                    clone, links, topology, source.loopProxyMass);
                AddCollisionMeshes(
                    links, definition, source.permitVisualMeshColliders);

                if (definition.joints == null || definition.joints.Length == 0)
                    throw new InvalidOperationException(
                        $"Definition '{definition.robotId}' contains no actuated joints");
                string rootName = topology.rootLink;
                ArticulationBody articulationRoot = links[rootName].GetComponent<ArticulationBody>();
                ArticulationRobotDriver driver = clone.GetComponent<ArticulationRobotDriver>();
                if (driver == null)
                    driver = clone.AddComponent<ArticulationRobotDriver>();
                driver.definition = definition;
                driver.articulationRoot = articulationRoot;
                driver.configureFromDefinitionOnAwake = true;
                if (!driver.TryBind(out string bindError))
                    throw new InvalidOperationException(bindError);

                OfflineRigIdentity identity = clone.GetComponent<OfflineRigIdentity>();
                if (identity == null)
                    identity = clone.AddComponent<OfflineRigIdentity>();
                identity.robotId = definition.robotId;
                identity.definitionSha256 = definition.definitionSha256;
                identity.projectAuthoredProcedural = false;
                identity.robotDefinition = definition;
                identity.articulationDriver = driver;
                GameObject prefab = PrefabUtility.SaveAsPrefabAsset(clone, source.prefabPath);
                return prefab ?? throw new InvalidOperationException(
                    $"Could not save normalized prefab '{source.prefabPath}'");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(clone);
            }
        }

        static Dictionary<string, Transform> FindExactLinks(
            GameObject root,
            RobotDefinition definition,
            string requiredParentName,
            ISet<string> synthesizedLinks)
        {
            Transform[] all = root.GetComponentsInChildren<Transform>(true);
            var result = new Dictionary<string, Transform>();
            foreach (RobotLinkDefinition link in definition.links)
            {
                Transform[] matches = all.Where(item =>
                    item.name == link.name &&
                    (synthesizedLinks.Contains(link.name) ||
                     string.IsNullOrWhiteSpace(requiredParentName) ||
                     (item.parent != null && item.parent.name == requiredParentName))).ToArray();
                if (matches.Length != 1)
                    throw new InvalidOperationException(
                        $"USD '{root.name}' has {matches.Length} exact transforms for link '{link.name}'");
                result.Add(link.name, matches[0]);
            }
            return result;
        }

        static void ApplyTopology(
            IReadOnlyDictionary<string, Transform> links,
            RobotDefinition definition,
            TopologyPlan topology,
            ISet<string> synthesizedLinks)
        {
            foreach (RobotJointDefinition joint in definition.joints)
                if (topology.treeJoints.Contains(joint.name))
                    ApplyTopologyEdge(
                        links, joint.parentLink, joint.childLink,
                        joint.parentAnchorPosition, joint.parentAnchorRotation,
                        joint.anchorPosition, joint.anchorRotation, synthesizedLinks);
            foreach (RobotFixedJointDefinition joint in definition.fixedJoints)
                if (topology.treeFixedJoints.Contains(joint.name))
                    ApplyTopologyEdge(
                        links, joint.parentLink, joint.childLink,
                        joint.parentAnchorPosition, joint.parentAnchorRotation,
                        joint.anchorPosition, joint.anchorRotation, synthesizedLinks);
            foreach (RobotAuxiliaryJointDefinition joint in definition.auxiliaryJoints)
                if (topology.treeAuxiliaryJoints.Contains(joint.name))
                    ApplyTopologyEdge(
                        links, joint.parentLink, joint.childLink,
                        joint.parentAnchorPosition, joint.parentAnchorRotation,
                        joint.anchorPosition, joint.anchorRotation, synthesizedLinks);
        }

        static void ApplyTopologyEdge(
            IReadOnlyDictionary<string, Transform> links,
            string parentName,
            string childName,
            Vector3 parentAnchorPosition,
            Quaternion parentAnchorRotation,
            Vector3 anchorPosition,
            Quaternion anchorRotation,
            ISet<string> synthesizedLinks)
        {
            Transform child = links[childName];
            Transform parent = links[parentName];
            if (!synthesizedLinks.Contains(childName))
            {
                child.SetParent(parent, true);
                return;
            }
            child.SetParent(parent, false);
            Quaternion rotation = parentAnchorRotation * Quaternion.Inverse(anchorRotation);
            child.localRotation = rotation;
            child.localPosition = parentAnchorPosition - rotation * anchorPosition;
        }

        static HashSet<string> AugmentPreparedRobotiq(
            GameObject root, RobotDefinition definition)
        {
            const string meshRoot =
                "Assets/_Project/Features/Robots/UR10e/URDF/Meshes/Robotiq";
            string[] gripperLinks =
            {
                "base_link_0",
                "left_outer_knuckle", "right_outer_knuckle",
                "left_outer_finger", "right_outer_finger",
                "left_inner_finger", "right_inner_finger",
                "left_inner_knuckle", "right_inner_knuckle",
            };
            string manifest =
                "Assets/_Project/Features/Robots/UR10e/URDF/source.manifest.json";
            if (AssetDatabase.LoadAssetAtPath<TextAsset>(manifest) == null)
                throw new InvalidOperationException(
                    "Prepared Robotiq source is absent. Run " +
                    "Isaac_XRPlayground/scripts/deployment/prepare_unity_robot_sources.py first.");
            var result = new HashSet<string>(gripperLinks);
            foreach (string linkName in gripperLinks)
            {
                if (root.GetComponentsInChildren<Transform>(true).Any(item => item.name == linkName))
                    throw new InvalidOperationException(
                        $"UR source unexpectedly already contains gripper link '{linkName}'");
                var link = new GameObject(linkName);
                link.transform.SetParent(root.transform, false);
                string meshName = linkName == "base_link_0"
                    ? "base_link"
                    : linkName.StartsWith("left_", StringComparison.Ordinal)
                        ? linkName.Substring("left_".Length)
                        : linkName.Substring("right_".Length);
                string visualPath =
                    $"{meshRoot}/visual/robotiq_arg2f_85_{meshName}.dae";
                InstantiatePreparedMesh(visualPath, link.transform, "visuals", true);

                if (linkName == "base_link_0")
                {
                    CreateCombinedCollisionMesh(
                        visualPath, link.transform,
                        "Assets/_Project/Features/Robots/UR10e/URDF/Generated/RobotiqBaseCollision.asset");
                }
                else
                {
                    string collisionPath =
                        $"{meshRoot}/collision/robotiq_arg2f_85_{meshName}.dae";
                    InstantiatePreparedMesh(collisionPath, link.transform, "collisions", false);
                }
                if (linkName == "left_inner_finger" || linkName == "right_inner_finger")
                {
                    GameObject pad = InstantiatePreparedMesh(
                        $"{meshRoot}/visual/robotiq_arg2f_85_pad.dae",
                        link.transform, "collisions", false);
                    pad.name = "finger_pad_collision";
                    pad.transform.localPosition = new Vector3(0f, 0.03242f, -0.0220203447f);
                }
            }
            return result;
        }

        static GameObject InstantiatePreparedMesh(
            string assetPath, Transform link, string groupName, bool render)
        {
            GameObject asset = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath) ??
                throw new InvalidOperationException($"Prepared mesh '{assetPath}' is not imported");
            Transform group = link.Find(groupName);
            if (group == null)
            {
                var groupObject = new GameObject(groupName);
                groupObject.transform.SetParent(link, false);
                group = groupObject.transform;
            }
            GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(asset, group);
            instance.name = System.IO.Path.GetFileNameWithoutExtension(assetPath);
            instance.transform.localPosition = Vector3.zero;
            instance.transform.localRotation = Quaternion.identity;
            instance.transform.localScale = Vector3.one * 0.001f;
            foreach (Renderer renderer in instance.GetComponentsInChildren<Renderer>(true))
                renderer.enabled = render;
            return instance;
        }

        static void CreateCombinedCollisionMesh(
            string sourcePath, Transform link, string meshAssetPath)
        {
            GameObject source = InstantiatePreparedMesh(sourcePath, link, "collisions", false);
            Transform group = source.transform.parent;
            MeshFilter[] filters = source.GetComponentsInChildren<MeshFilter>(true);
            if (filters.Length == 0 || filters.Any(item => item.sharedMesh == null))
                throw new InvalidOperationException(
                    $"Prepared collision source '{sourcePath}' contains no complete meshes");
            var combines = new CombineInstance[filters.Length];
            for (int index = 0; index < filters.Length; index++)
            {
                combines[index] = new CombineInstance
                {
                    mesh = filters[index].sharedMesh,
                    transform = group.worldToLocalMatrix * filters[index].transform.localToWorldMatrix,
                };
            }
            var combined = new Mesh { name = "RobotiqBaseCollision" };
            combined.CombineMeshes(combines, true, true, false);
            EnsureFolder(System.IO.Path.GetDirectoryName(meshAssetPath).Replace('\\', '/'));
            Mesh persisted = AssetDatabase.LoadAssetAtPath<Mesh>(meshAssetPath);
            if (persisted == null)
            {
                AssetDatabase.CreateAsset(combined, meshAssetPath);
                persisted = combined;
            }
            else
            {
                EditorUtility.CopySerialized(combined, persisted);
                UnityEngine.Object.DestroyImmediate(combined);
                EditorUtility.SetDirty(persisted);
            }
            UnityEngine.Object.DestroyImmediate(source);
            var node = new GameObject("mesh_0");
            node.transform.SetParent(group, false);
            node.AddComponent<MeshFilter>().sharedMesh = persisted;
        }

        public static TopologySummary AnalyzeTopology(string robotId)
        {
            string definitionPath =
                $"Assets/_Project/Features/Deployment/RobotDefinitions/{robotId}/robot.definition.asset";
            RobotDefinition definition = AssetDatabase.LoadAssetAtPath<RobotDefinition>(definitionPath);
            if (definition == null)
                throw new InvalidOperationException($"Missing RobotDefinition '{definitionPath}'");
            TopologyPlan plan = BuildTopologyPlan(definition);
            return new TopologySummary
            {
                rootLink = plan.rootLink,
                externalLoopJointNames = plan.externalJoints
                    .Select(item => item.name)
                    .Concat(plan.externalAuxiliaryJoints.Select(item => item.name))
                    .ToArray(),
                auxiliaryTreeJointNames = plan.treeAuxiliaryJoints.OrderBy(item => item).ToArray(),
            };
        }

        static TopologyPlan BuildTopologyPlan(RobotDefinition definition)
        {
            var plan = new TopologyPlan();
            var links = new HashSet<string>(definition.links.Select(item => item.name));
            var assignedChildren = new HashSet<string>();
            var parents = links.ToDictionary(item => item, item => item);

            string Find(string link)
            {
                while (parents[link] != link)
                {
                    parents[link] = parents[parents[link]];
                    link = parents[link];
                }
                return link;
            }

            bool AcceptTreeEdge(string parent, string child)
            {
                if (string.IsNullOrWhiteSpace(parent) || !links.Contains(parent) ||
                    !links.Contains(child) || assignedChildren.Contains(child))
                    return false;
                string parentRoot = Find(parent);
                string childRoot = Find(child);
                if (parentRoot == childRoot)
                    return false;
                parents[childRoot] = parentRoot;
                assignedChildren.Add(child);
                return true;
            }

            foreach (RobotJointDefinition joint in definition.joints)
            {
                if (AcceptTreeEdge(joint.parentLink, joint.childLink))
                    plan.treeJoints.Add(joint.name);
                else if (joint.jointType == ArticulationJointType.RevoluteJoint &&
                         Mathf.Approximately(joint.stiffness, 0f) &&
                         Mathf.Approximately(joint.damping, 0f))
                    plan.externalJoints.Add(joint);
                else
                    throw new InvalidOperationException(
                        $"Robot '{definition.robotId}' requires unsupported driven or non-revolute " +
                        $"external joint '{joint.name}' ({joint.jointType}, " +
                        $"stiffness={joint.stiffness}, damping={joint.damping})");
            }
            foreach (RobotFixedJointDefinition joint in definition.fixedJoints)
            {
                if (!AcceptTreeEdge(joint.parentLink, joint.childLink))
                    throw new InvalidOperationException(
                        $"Robot '{definition.robotId}' fixed joint '{joint.name}' " +
                        "cannot be placed in the articulation tree");
                plan.treeFixedJoints.Add(joint.name);
            }
            foreach (RobotAuxiliaryJointDefinition joint in
                     definition.auxiliaryJoints ?? Array.Empty<RobotAuxiliaryJointDefinition>())
            {
                if (AcceptTreeEdge(joint.parentLink, joint.childLink))
                    plan.treeAuxiliaryJoints.Add(joint.name);
                else if (joint.jointType == "RevoluteJoint")
                    plan.externalAuxiliaryJoints.Add(joint);
                else
                    throw new InvalidOperationException(
                        $"Robot '{definition.robotId}' requires unsupported external " +
                        $"{joint.jointType} auxiliary joint '{joint.name}'");
            }
            string[] roots = links.Where(name => !assignedChildren.Contains(name)).ToArray();
            if (roots.Length != 1)
                throw new InvalidOperationException(
                    $"Robot '{definition.robotId}' topology has {roots.Length} tree roots");
            plan.rootLink = roots[0];
            return plan;
        }

        static void AddArticulationBodies(
            IReadOnlyDictionary<string, Transform> links,
            RobotDefinition definition,
            TopologyPlan topology,
            bool fixedBase)
        {
            string rootName = topology.rootLink;
            foreach (RobotLinkDefinition link in definition.links)
            {
                ArticulationBody body = links[link.name].GetComponent<ArticulationBody>();
                if (body == null)
                    body = links[link.name].gameObject.AddComponent<ArticulationBody>();
                if (body == null)
                    throw new InvalidOperationException(
                        $"Unity refused to add ArticulationBody to link '{link.name}'");
                body.useGravity = true;
                // Unity permits the property to be assigned only on the root,
                // even assigning false to a child reports an editor error.
                if (link.name == rootName)
                    body.immovable = fixedBase;
                body.linearDamping = 0f;
                body.angularDamping = 0f;
            }
            foreach (RobotFixedJointDefinition joint in definition.fixedJoints)
            {
                if (!topology.treeFixedJoints.Contains(joint.name))
                    continue;
                ArticulationBody body = links[joint.childLink].GetComponent<ArticulationBody>();
                body.jointType = ArticulationJointType.FixedJoint;
                body.parentAnchorPosition = joint.parentAnchorPosition;
                body.parentAnchorRotation = joint.parentAnchorRotation;
                body.anchorPosition = joint.anchorPosition;
                body.anchorRotation = joint.anchorRotation;
            }
            foreach (RobotAuxiliaryJointDefinition joint in
                     definition.auxiliaryJoints ?? Array.Empty<RobotAuxiliaryJointDefinition>())
            {
                if (!topology.treeAuxiliaryJoints.Contains(joint.name))
                    continue;
                if (joint.jointType != "RevoluteJoint")
                    throw new InvalidOperationException(
                        $"Auxiliary tree joint '{joint.name}' is not revolute");
                ArticulationBody body = links[joint.childLink].GetComponent<ArticulationBody>();
                body.jointType = ArticulationJointType.RevoluteJoint;
                body.twistLock = ArticulationDofLock.FreeMotion;
                Quaternion alignment = Quaternion.FromToRotation(
                    Vector3.right, joint.axis.normalized);
                body.parentAnchorPosition = joint.parentAnchorPosition;
                body.parentAnchorRotation = joint.parentAnchorRotation * alignment;
                body.anchorPosition = joint.anchorPosition;
                body.anchorRotation = joint.anchorRotation * alignment;
                body.xDrive = new ArticulationDrive
                {
                    lowerLimit = -360f,
                    upperLimit = 360f,
                    stiffness = 0f,
                    damping = 0f,
                    forceLimit = float.MaxValue,
                    target = 0f,
                };
            }
        }

        static void AddLoopClosureBinding(
            GameObject root,
            IReadOnlyDictionary<string, Transform> links,
            TopologyPlan topology,
            float proxyMass)
        {
            int count = topology.externalJoints.Count + topology.externalAuxiliaryJoints.Count;
            ArticulationLoopClosureBinding binding =
                root.GetComponent<ArticulationLoopClosureBinding>();
            if (count == 0)
            {
                if (binding != null)
                    UnityEngine.Object.DestroyImmediate(binding);
                return;
            }
            if (binding == null)
                binding = root.AddComponent<ArticulationLoopClosureBinding>();
            var entries = new List<ArticulationLoopClosureEntry>(count);
            foreach (RobotJointDefinition joint in topology.externalJoints)
                entries.Add(LoopEntry(
                    joint.name, joint.parentLink, joint.childLink,
                    joint.parentAnchorPosition, joint.parentAnchorRotation,
                    joint.anchorPosition, joint.anchorRotation, joint.axis,
                    links, proxyMass));
            foreach (RobotAuxiliaryJointDefinition joint in topology.externalAuxiliaryJoints)
                entries.Add(LoopEntry(
                    joint.name, joint.parentLink, joint.childLink,
                    joint.parentAnchorPosition, joint.parentAnchorRotation,
                    joint.anchorPosition, joint.anchorRotation, joint.axis,
                    links, proxyMass));
            binding.entries = entries.ToArray();
        }

        static ArticulationLoopClosureEntry LoopEntry(
            string name,
            string parentLink,
            string childLink,
            Vector3 parentAnchorPosition,
            Quaternion parentAnchorRotation,
            Vector3 childAnchorPosition,
            Quaternion childAnchorRotation,
            Vector3 axis,
            IReadOnlyDictionary<string, Transform> links,
            float proxyMass) => new ArticulationLoopClosureEntry
        {
            constraintName = name,
            parentBody = links[parentLink].GetComponent<ArticulationBody>(),
            childBody = links[childLink].GetComponent<ArticulationBody>(),
            parentAnchorPosition = parentAnchorPosition,
            parentAnchorRotation = parentAnchorRotation,
            childAnchorPosition = childAnchorPosition,
            childAnchorRotation = childAnchorRotation,
            axis = axis,
            proxyMass = proxyMass,
        };

        static void AddCollisionMeshes(
            IReadOnlyDictionary<string, Transform> links,
            RobotDefinition definition,
            bool permitVisualFallback)
        {
            var linkTransforms = new HashSet<Transform>(links.Values);
            foreach (Collider collider in links.Values.First().root.GetComponentsInChildren<Collider>(true))
                UnityEngine.Object.DestroyImmediate(collider);

            foreach (RobotLinkDefinition link in definition.links)
            {
                Transform linkTransform = links[link.name];
                List<MeshFilter> owned = OwnedMeshFilters(linkTransform, linkTransforms);
                List<MeshFilter> candidates = owned.Where(
                    item => HasAncestorNamed(item.transform, linkTransform, "collisions")).ToList();
                RobotCollisionShapeDefinition[] descriptors = link.collisionShapes ??
                    Array.Empty<RobotCollisionShapeDefinition>();
                int primitiveCount = descriptors.Count(item => item.shapeType != "mesh");
                int requiredMeshCount = descriptors.Length == link.collisionShapeCount
                    ? link.collisionShapeCount - primitiveCount
                    : link.collisionShapeCount;
                if (candidates.Count == 0 && permitVisualFallback && requiredMeshCount > 0)
                    candidates = owned.Where(
                        item => HasAncestorNamed(item.transform, linkTransform, "visuals")).ToList();
                candidates = candidates.Where(item => item.sharedMesh != null).ToList();
                if (candidates.Count != requiredMeshCount)
                    throw new InvalidOperationException(
                        $"Link '{link.name}' exposes {candidates.Count} collision meshes; " +
                        $"Isaac requires {requiredMeshCount}");
                foreach (MeshFilter filter in candidates)
                {
                    var collider = filter.gameObject.AddComponent<MeshCollider>();
                    collider.sharedMesh = filter.sharedMesh;
                    collider.convex = true;
                    if (HasAncestorNamed(filter.transform, linkTransform, "collisions"))
                    {
                        Renderer renderer = filter.GetComponent<Renderer>();
                        if (renderer != null)
                            renderer.enabled = false;
                    }
                }
                if (descriptors.Length == link.collisionShapeCount)
                    foreach (RobotCollisionShapeDefinition descriptor in descriptors)
                        if (descriptor.shapeType != "mesh")
                            AddPrimitiveCollider(linkTransform, descriptor);
            }
        }

        static void AddPrimitiveCollider(
            Transform link, RobotCollisionShapeDefinition descriptor)
        {
            var node = new GameObject($"IsaacCollision_{descriptor.shapeType}");
            node.transform.SetParent(link, false);
            node.transform.localPosition = descriptor.localPosition;
            node.transform.localRotation = descriptor.localRotation;
            switch (descriptor.shapeType)
            {
                case "sphere":
                    if (descriptor.radius <= 0f)
                        throw new InvalidOperationException("Isaac sphere collider has invalid radius");
                    node.AddComponent<SphereCollider>().radius = descriptor.radius;
                    break;
                case "box":
                    if (descriptor.size.x <= 0f || descriptor.size.y <= 0f || descriptor.size.z <= 0f)
                        throw new InvalidOperationException("Isaac box collider has invalid size");
                    node.AddComponent<BoxCollider>().size = descriptor.size;
                    break;
                case "capsule":
                    if (descriptor.radius <= 0f || descriptor.height < 0f)
                        throw new InvalidOperationException("Isaac capsule collider has invalid dimensions");
                    CapsuleCollider capsule = node.AddComponent<CapsuleCollider>();
                    capsule.radius = descriptor.radius;
                    capsule.height = descriptor.height + 2f * descriptor.radius;
                    capsule.direction = DominantAxis(descriptor.axis);
                    break;
                case "cylinder":
                    throw new InvalidOperationException(
                        "Cylinder collision export requires a committed convex mesh; no approximation is allowed");
                default:
                    throw new InvalidOperationException(
                        $"Unsupported Isaac collision primitive '{descriptor.shapeType}'");
            }
        }

        static int DominantAxis(Vector3 axis)
        {
            axis = new Vector3(Mathf.Abs(axis.x), Mathf.Abs(axis.y), Mathf.Abs(axis.z));
            if (axis.x >= axis.y && axis.x >= axis.z)
                return 0;
            return axis.y >= axis.z ? 1 : 2;
        }

        static List<MeshFilter> OwnedMeshFilters(
            Transform link, HashSet<Transform> linkTransforms)
        {
            var result = new List<MeshFilter>();
            foreach (MeshFilter filter in link.GetComponentsInChildren<MeshFilter>(true))
            {
                Transform current = filter.transform;
                while (current != null && !linkTransforms.Contains(current))
                    current = current.parent;
                if (current == link)
                    result.Add(filter);
            }
            return result;
        }

        static bool HasAncestorNamed(Transform item, Transform stop, string name)
        {
            Transform current = item.parent;
            while (current != null && current != stop)
            {
                if (current.name.Equals(name, StringComparison.OrdinalIgnoreCase))
                    return true;
                current = current.parent;
            }
            return false;
        }

        static void StripImporterComponents(GameObject root)
        {
            foreach (MonoBehaviour component in root.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (component == null)
                    continue;
                string typeName = component.GetType().FullName ?? string.Empty;
                if (typeName.StartsWith("Unity.Importer", StringComparison.Ordinal) ||
                    typeName.StartsWith("Unity.Formats.USD", StringComparison.Ordinal) ||
                    typeName.StartsWith("Unity.Robotics.UrdfImporter", StringComparison.Ordinal))
                    UnityEngine.Object.DestroyImmediate(component);
            }
        }

        static void EnsureFolder(string folder)
        {
            string[] parts = folder.Split('/');
            string current = parts[0];
            for (int index = 1; index < parts.Length; index++)
            {
                string next = current + "/" + parts[index];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[index]);
                current = next;
            }
        }
    }
}
