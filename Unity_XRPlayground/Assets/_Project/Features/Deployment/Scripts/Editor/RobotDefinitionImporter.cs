using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace XRPlayground.Deployment.Editor
{
    public sealed class RobotDefinitionImporter : AssetPostprocessor
    {
        const string DeploymentRoot = "Assets/_Project/Features/Deployment";

        [Serializable]
        sealed class Document
        {
            public int schema_version;
            public string robot_id;
            public string source_asset;
            public string source_sha256;
            public string definition_sha256;
            public string provenance;
            public string redistribution_license;
            public Link[] links;
            public Joint[] joints;
            public FixedJoint[] fixed_joints;
            public AuxiliaryJoint[] auxiliary_joints;
        }

        [Serializable]
        sealed class Link
        {
            public string name;
            public float mass;
            public float[] center_of_mass;
            public float[] inertia_tensor;
            public float[] inertia_tensor_rotation;
            public int collision_shape_count;
            public bool collision_enabled;
            public float static_friction;
            public float dynamic_friction;
            public float restitution;
            public float contact_offset;
            public float rest_offset;
            public CollisionShape[] collision_shapes;
        }

        [Serializable]
        sealed class CollisionShape
        {
            public string type;
            public float[] local_position;
            public float[] local_rotation;
            public float[] scale;
            public float[] size;
            public float radius;
            public float height;
            public float[] axis;
            public string source_prim_path;
        }

        [Serializable]
        sealed class Joint
        {
            public string name;
            public string parent_link;
            public string child_link;
            public string joint_type;
            public string position_unit;
            public string limit_mode;
            public float[] axis;
            public float[] parent_anchor_position;
            public float[] parent_anchor_rotation;
            public float[] anchor_position;
            public float[] anchor_rotation;
            public float lower_limit;
            public float upper_limit;
            public float default_position;
            public float stiffness;
            public float damping;
            public float force_limit;
            public float max_velocity;
            public string actuator_group;
            public string actuator_model;
            public bool actuator_is_implicit;
            public int actuator_delay_min_steps;
            public int actuator_delay_max_steps;
            public int actuator_nominal_delay_steps;
            public EffortSample[] effort_limit_curve;
        }

        [Serializable]
        sealed class EffortSample
        {
            public float position;
            public float max_effort;
        }

        [Serializable]
        sealed class FixedJoint
        {
            public string name;
            public string parent_link;
            public string child_link;
            public float[] parent_anchor_position;
            public float[] parent_anchor_rotation;
            public float[] anchor_position;
            public float[] anchor_rotation;
        }

        [Serializable]
        sealed class AuxiliaryJoint
        {
            public string name;
            public string source_prim_path;
            public string source_joint_type;
            public string joint_type;
            public string topology_role;
            public string parent_link;
            public string child_link;
            public float[] axis;
            public float[] parent_anchor_position;
            public float[] parent_anchor_rotation;
            public float[] anchor_position;
            public float[] anchor_rotation;
        }

        static void OnPostprocessAllAssets(
            string[] importedAssets,
            string[] deletedAssets,
            string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            foreach (string path in importedAssets)
                if (path.EndsWith("robot.definition.json", StringComparison.OrdinalIgnoreCase))
                    Import(path);
        }

        [InitializeOnLoadMethod]
        static void ScheduleInitialImport()
        {
            // The exporter may stage definitions while Unity is closed, or in the same
            // refresh that compiles this postprocessor. Scan once after every domain load
            // so those definitions cannot remain as unimported JSON sidecars.
            EditorApplication.delayCall += () => ImportAllDefinitions();
        }

        [MenuItem("XRPlayground/Deployment/Import All Robot Definitions")]
        static void ImportAllDefinitionsFromMenu() => ImportAllDefinitions();

        public static int ImportAllDefinitions()
        {
            string[] guids = AssetDatabase.FindAssets(
                "robot.definition t:TextAsset", new[] { DeploymentRoot });
            var paths = new List<string>();
            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (path.EndsWith("robot.definition.json", StringComparison.OrdinalIgnoreCase))
                    paths.Add(path);
            }
            paths.Sort(StringComparer.Ordinal);
            foreach (string path in paths)
                Import(path);
            return paths.Count;
        }

        [MenuItem("XRPlayground/Deployment/Import Selected Robot Definition")]
        static void ImportSelected()
        {
            string path = AssetDatabase.GetAssetPath(Selection.activeObject);
            if (!path.EndsWith("robot.definition.json", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Select a robot.definition.json TextAsset first");
            Import(path);
        }

        public static void Import(string jsonAssetPath)
        {
            Document source = JsonUtility.FromJson<Document>(File.ReadAllText(jsonAssetPath));
            if (source == null || source.schema_version != 6 ||
                string.IsNullOrWhiteSpace(source.robot_id) ||
                !IsSha256(source.definition_sha256))
                throw new InvalidDataException($"Invalid robot definition: {jsonAssetPath}");
            string assetPath = jsonAssetPath.Substring(0, jsonAssetPath.Length - ".json".Length) + ".asset";
            RobotDefinition target = AssetDatabase.LoadAssetAtPath<RobotDefinition>(assetPath);
            if (target == null)
            {
                target = ScriptableObject.CreateInstance<RobotDefinition>();
                AssetDatabase.CreateAsset(target, assetPath);
            }
            target.robotId = source.robot_id;
            target.sourceAsset = source.source_asset;
            target.sourceSha256 = source.source_sha256;
            target.definitionSha256 = source.definition_sha256;
            target.provenance = source.provenance;
            target.redistributionLicense = source.redistribution_license;
            target.links = ConvertLinks(source.links);
            target.joints = ConvertJoints(source.joints);
            target.fixedJoints = ConvertFixedJoints(source.fixed_joints);
            target.auxiliaryJoints = ConvertAuxiliaryJoints(source.auxiliary_joints);
            ValidateTopology(target);
            EditorUtility.SetDirty(target);
            AssetDatabase.SaveAssetIfDirty(target);
        }

        static bool IsSha256(string value)
        {
            if (string.IsNullOrWhiteSpace(value) || value.Length != 64)
                return false;
            foreach (char character in value)
                if (!Uri.IsHexDigit(character))
                    return false;
            return true;
        }

        static RobotLinkDefinition[] ConvertLinks(Link[] source)
        {
            if (source == null || source.Length == 0)
                throw new InvalidDataException("Robot definition has no links");
            var names = new HashSet<string>();
            var result = new RobotLinkDefinition[source.Length];
            for (int i = 0; i < source.Length; i++)
            {
                Link item = source[i];
                Vector3 centerOfMass = ToVector3(item?.center_of_mass, "center_of_mass");
                Vector3 inertiaTensor = ToVector3(item?.inertia_tensor, "inertia_tensor");
                Quaternion inertiaRotation = ToQuaternion(
                    item?.inertia_tensor_rotation, "inertia_tensor_rotation");
                if (item == null || !names.Add(item.name) || item.mass <= 0f ||
                    inertiaTensor.x <= 0f || inertiaTensor.y <= 0f || inertiaTensor.z <= 0f ||
                    item.collision_shape_count < 0 ||
                    (item.collision_shape_count > 0 &&
                     (item.static_friction < 0f || item.dynamic_friction < 0f || item.restitution < 0f)))
                    throw new InvalidDataException("Robot definition contains an invalid or duplicate link");
                RobotCollisionShapeDefinition[] collisionShapes =
                    ConvertCollisionShapes(item.collision_shapes);
                if (item.collision_shapes != null &&
                    collisionShapes.Length != item.collision_shape_count)
                    throw new InvalidDataException(
                        $"Link '{item.name}' collision descriptor count differs from PhysX");
                result[i] = new RobotLinkDefinition
                {
                    name = item.name,
                    mass = item.mass,
                    centerOfMass = centerOfMass,
                    inertiaTensor = inertiaTensor,
                    inertiaTensorRotation = inertiaRotation,
                    collisionShapeCount = item.collision_shape_count,
                    collisionEnabled = item.collision_enabled,
                    staticFriction = item.static_friction,
                    dynamicFriction = item.dynamic_friction,
                    restitution = item.restitution,
                    contactOffset = item.contact_offset,
                    restOffset = item.rest_offset,
                    collisionShapes = collisionShapes,
                };
            }
            return result;
        }

        static RobotCollisionShapeDefinition[] ConvertCollisionShapes(CollisionShape[] source)
        {
            if (source == null || source.Length == 0)
                return Array.Empty<RobotCollisionShapeDefinition>();
            var result = new RobotCollisionShapeDefinition[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                CollisionShape item = source[index];
                if (item == null || (item.type != "mesh" && item.type != "box" &&
                    item.type != "sphere" && item.type != "capsule" && item.type != "cylinder"))
                    throw new InvalidDataException("Robot definition contains an invalid collision shape");
                Vector3 scale = ToVector3(item.scale, "collision.scale");
                Vector3 size = item.size == null
                    ? Vector3.zero
                    : ToVector3(item.size, "collision.size");
                Vector3 axis = item.axis == null
                    ? Vector3.up
                    : ToVector3(item.axis, "collision.axis");
                if (scale.x <= 0f || scale.y <= 0f || scale.z <= 0f ||
                    (item.type == "box" && (size.x <= 0f || size.y <= 0f || size.z <= 0f)) ||
                    ((item.type == "sphere" || item.type == "capsule" || item.type == "cylinder") &&
                     (!float.IsFinite(item.radius) || item.radius <= 0f)) ||
                    ((item.type == "capsule" || item.type == "cylinder") &&
                     (!float.IsFinite(item.height) || item.height < 0f ||
                      !RobotDefinitionValidation.IsFiniteUnitAxis(axis))))
                    throw new InvalidDataException(
                        "Robot definition contains invalid collision geometry dimensions");
                result[index] = new RobotCollisionShapeDefinition
                {
                    shapeType = item.type,
                    localPosition = ToVector3(item.local_position, "collision.local_position"),
                    localRotation = ToQuaternion(item.local_rotation, "collision.local_rotation"),
                    scale = scale,
                    size = size,
                    radius = item.radius,
                    height = item.height,
                    axis = axis,
                    sourcePrimPath = item.source_prim_path,
                };
            }
            return result;
        }

        static RobotJointDefinition[] ConvertJoints(Joint[] source)
        {
            if (source == null || source.Length == 0)
                throw new InvalidDataException("Robot definition has no joints");
            var names = new HashSet<string>();
            var result = new RobotJointDefinition[source.Length];
            for (int i = 0; i < source.Length; i++)
            {
                Joint item = source[i];
                Vector3 axis = ToVector3(item?.axis, "axis");
                Vector3 parentAnchorPosition = ToVector3(
                    item?.parent_anchor_position, "parent_anchor_position");
                Quaternion parentAnchorRotation = ToQuaternion(
                    item?.parent_anchor_rotation, "parent_anchor_rotation");
                Vector3 anchorPosition = ToVector3(item?.anchor_position, "anchor_position");
                Quaternion anchorRotation = ToQuaternion(
                    item?.anchor_rotation, "anchor_rotation");
                if (item == null || !names.Add(item.name) ||
                    !Enum.TryParse(item.joint_type, out ArticulationJointType jointType) ||
                    !ValidPositionUnit(jointType, item.position_unit) ||
                    !ValidLimitMode(jointType, item.limit_mode) ||
                    !RobotDefinitionValidation.IsFiniteUnitAxis(axis) ||
                    string.IsNullOrWhiteSpace(item.actuator_model) ||
                    item.actuator_delay_min_steps < 0 ||
                    item.actuator_delay_max_steps < item.actuator_delay_min_steps ||
                    item.actuator_nominal_delay_steps < item.actuator_delay_min_steps ||
                    item.actuator_nominal_delay_steps > item.actuator_delay_max_steps ||
                    !ValidEffortCurve(item.effort_limit_curve))
                    throw new InvalidDataException("Robot definition contains an invalid or duplicate joint");
                result[i] = new RobotJointDefinition
                {
                    name = item.name,
                    parentLink = item.parent_link,
                    childLink = item.child_link,
                    jointType = jointType,
                    positionUnit = item.position_unit,
                    limitMode = item.limit_mode,
                    axis = axis,
                    parentAnchorPosition = parentAnchorPosition,
                    parentAnchorRotation = parentAnchorRotation,
                    anchorPosition = anchorPosition,
                    anchorRotation = anchorRotation,
                    lowerLimit = item.lower_limit,
                    upperLimit = item.upper_limit,
                    defaultPosition = item.default_position,
                    stiffness = item.stiffness,
                    damping = item.damping,
                    forceLimit = item.force_limit,
                    maxVelocity = item.max_velocity,
                    actuatorGroup = item.actuator_group,
                    actuatorModel = item.actuator_model,
                    actuatorIsImplicit = item.actuator_is_implicit,
                    actuatorDelayMinSteps = item.actuator_delay_min_steps,
                    actuatorDelayMaxSteps = item.actuator_delay_max_steps,
                    actuatorNominalDelaySteps = item.actuator_nominal_delay_steps,
                    effortLimitCurve = ConvertEffortCurve(item.effort_limit_curve),
                };
            }
            return result;
        }

        static RobotFixedJointDefinition[] ConvertFixedJoints(FixedJoint[] source)
        {
            if (source == null)
                throw new InvalidDataException("Robot definition has no fixed_joints array");
            var names = new HashSet<string>();
            var children = new HashSet<string>();
            var result = new RobotFixedJointDefinition[source.Length];
            for (int i = 0; i < source.Length; i++)
            {
                FixedJoint item = source[i];
                if (item == null || string.IsNullOrWhiteSpace(item.name) ||
                    string.IsNullOrWhiteSpace(item.parent_link) ||
                    string.IsNullOrWhiteSpace(item.child_link) ||
                    !names.Add(item.name) || !children.Add(item.child_link))
                    throw new InvalidDataException(
                        "Robot definition contains an invalid or duplicate fixed joint");
                result[i] = new RobotFixedJointDefinition
                {
                    name = item.name,
                    parentLink = item.parent_link,
                    childLink = item.child_link,
                    parentAnchorPosition = ToVector3(
                        item.parent_anchor_position, "fixed.parent_anchor_position"),
                    parentAnchorRotation = ToQuaternion(
                        item.parent_anchor_rotation, "fixed.parent_anchor_rotation"),
                    anchorPosition = ToVector3(item.anchor_position, "fixed.anchor_position"),
                    anchorRotation = ToQuaternion(item.anchor_rotation, "fixed.anchor_rotation"),
                };
            }
            return result;
        }

        static RobotAuxiliaryJointDefinition[] ConvertAuxiliaryJoints(AuxiliaryJoint[] source)
        {
            if (source == null || source.Length == 0)
                return Array.Empty<RobotAuxiliaryJointDefinition>();
            var names = new HashSet<string>();
            var result = new RobotAuxiliaryJointDefinition[source.Length];
            for (int i = 0; i < source.Length; i++)
            {
                AuxiliaryJoint item = source[i];
                Vector3 axis = ToVector3(item?.axis, "auxiliary.axis");
                if (item == null || string.IsNullOrWhiteSpace(item.name) ||
                    string.IsNullOrWhiteSpace(item.source_prim_path) ||
                    string.IsNullOrWhiteSpace(item.source_joint_type) ||
                    string.IsNullOrWhiteSpace(item.parent_link) ||
                    string.IsNullOrWhiteSpace(item.child_link) ||
                    item.parent_link == item.child_link || !names.Add(item.name) ||
                    (item.joint_type != "FixedJoint" &&
                     item.joint_type != "RevoluteJoint" &&
                     item.joint_type != "PrismaticJoint" &&
                     item.joint_type != "GenericJoint") ||
                    (item.topology_role != "tree_connector" &&
                     item.topology_role != "loop_closure") ||
                    !RobotDefinitionValidation.IsFiniteUnitAxis(axis))
                    throw new InvalidDataException(
                        "Robot definition contains an invalid or duplicate auxiliary joint");
                result[i] = new RobotAuxiliaryJointDefinition
                {
                    name = item.name,
                    sourcePrimPath = item.source_prim_path,
                    sourceJointType = item.source_joint_type,
                    jointType = item.joint_type,
                    topologyRole = item.topology_role,
                    parentLink = item.parent_link,
                    childLink = item.child_link,
                    axis = axis,
                    parentAnchorPosition = ToVector3(
                        item.parent_anchor_position, "auxiliary.parent_anchor_position"),
                    parentAnchorRotation = ToQuaternion(
                        item.parent_anchor_rotation, "auxiliary.parent_anchor_rotation"),
                    anchorPosition = ToVector3(
                        item.anchor_position, "auxiliary.anchor_position"),
                    anchorRotation = ToQuaternion(
                        item.anchor_rotation, "auxiliary.anchor_rotation"),
                };
            }
            return result;
        }

        static void ValidateTopology(RobotDefinition definition)
        {
            var links = new HashSet<string>();
            foreach (RobotLinkDefinition link in definition.links)
                links.Add(link.name);
            var parents = new Dictionary<string, string>();
            foreach (string link in links)
                parents[link] = link;

            string Find(string link)
            {
                while (parents[link] != link)
                {
                    parents[link] = parents[parents[link]];
                    link = parents[link];
                }
                return link;
            }

            bool Union(string first, string second)
            {
                string firstRoot = Find(first);
                string secondRoot = Find(second);
                if (firstRoot == secondRoot)
                    return false;
                parents[secondRoot] = firstRoot;
                return true;
            }

            foreach (RobotJointDefinition joint in definition.joints)
            {
                if (!links.Contains(joint.childLink) ||
                    (!string.IsNullOrEmpty(joint.parentLink) && !links.Contains(joint.parentLink)))
                    throw new InvalidDataException(
                        $"Joint '{joint.name}' references a missing link");
                if (!string.IsNullOrEmpty(joint.parentLink))
                    Union(joint.parentLink, joint.childLink);
            }
            foreach (RobotFixedJointDefinition joint in definition.fixedJoints)
            {
                if (!links.Contains(joint.parentLink) || !links.Contains(joint.childLink))
                    throw new InvalidDataException(
                        $"Fixed joint '{joint.name}' references a missing link");
                Union(joint.parentLink, joint.childLink);
            }
            foreach (RobotAuxiliaryJointDefinition joint in definition.auxiliaryJoints)
            {
                if (!links.Contains(joint.parentLink) || !links.Contains(joint.childLink))
                    throw new InvalidDataException(
                        $"Auxiliary joint '{joint.name}' references a missing link");
                string expectedRole = Union(joint.parentLink, joint.childLink)
                    ? "tree_connector"
                    : "loop_closure";
                if (joint.topologyRole != expectedRole)
                    throw new InvalidDataException(
                        $"Auxiliary joint '{joint.name}' is mislabeled; expected {expectedRole}");
            }
        }

        static bool ValidEffortCurve(EffortSample[] source)
        {
            if (source == null || source.Length == 0)
                return true;
            float previous = float.NegativeInfinity;
            foreach (EffortSample sample in source)
            {
                if (sample == null || !float.IsFinite(sample.position) ||
                    !float.IsFinite(sample.max_effort) || sample.max_effort <= 0f ||
                    sample.position <= previous)
                    return false;
                previous = sample.position;
            }
            return true;
        }

        static RobotJointEffortLimitSample[] ConvertEffortCurve(EffortSample[] source)
        {
            if (source == null || source.Length == 0)
                return Array.Empty<RobotJointEffortLimitSample>();
            var result = new RobotJointEffortLimitSample[source.Length];
            for (int i = 0; i < source.Length; i++)
                result[i] = new RobotJointEffortLimitSample
                {
                    position = source[i].position,
                    maxEffort = source[i].max_effort,
                };
            return result;
        }

        static Vector3 ToVector3(float[] source, string field)
        {
            if (source == null || source.Length != 3 ||
                !float.IsFinite(source[0]) || !float.IsFinite(source[1]) ||
                !float.IsFinite(source[2]))
                throw new InvalidDataException($"Robot definition field '{field}' is not a finite vec3");
            return new Vector3(source[0], source[1], source[2]);
        }

        static Quaternion ToQuaternion(float[] source, string field)
        {
            if (source == null || source.Length != 4 ||
                !float.IsFinite(source[0]) || !float.IsFinite(source[1]) ||
                !float.IsFinite(source[2]) || !float.IsFinite(source[3]))
                throw new InvalidDataException(
                    $"Robot definition field '{field}' is not a finite quaternion");
            var result = new Quaternion(source[0], source[1], source[2], source[3]);
            float magnitudeSquared = Quaternion.Dot(result, result);
            if (magnitudeSquared < 0.999f || magnitudeSquared > 1.001f)
                throw new InvalidDataException(
                    $"Robot definition field '{field}' is not a unit quaternion");
            return result;
        }

        static bool ValidPositionUnit(ArticulationJointType jointType, string unit) =>
            (jointType == ArticulationJointType.RevoluteJoint && unit == "radian") ||
            (jointType == ArticulationJointType.PrismaticJoint && unit == "meter") ||
            (jointType == ArticulationJointType.FixedJoint && unit == "fixed");

        static bool ValidLimitMode(ArticulationJointType jointType, string mode) =>
            (jointType == ArticulationJointType.RevoluteJoint &&
             (mode == "limited" || mode == "continuous")) ||
            (jointType == ArticulationJointType.PrismaticJoint && mode == "limited") ||
            (jointType == ArticulationJointType.FixedJoint && mode == "fixed");
    }

    [CustomEditor(typeof(ArticulationRobotDriver))]
    public sealed class ArticulationRobotDriverEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            if (!GUILayout.Button("Bind And Apply Isaac Physics"))
                return;
            var driver = (ArticulationRobotDriver)target;
            Undo.RecordObject(driver, "Apply Isaac Robot Physics");
            if (!driver.TryBind(out string error))
                throw new InvalidOperationException(error);
            foreach (ArticulationBody body in driver.GetComponentsInChildren<ArticulationBody>(true))
                EditorUtility.SetDirty(body);
        }
    }
}
