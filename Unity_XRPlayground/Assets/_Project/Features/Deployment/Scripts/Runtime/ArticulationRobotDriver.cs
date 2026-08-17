using System;
using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DefaultExecutionOrder(100)]
    [DisallowMultipleComponent]
    public sealed class ArticulationRobotDriver : MonoBehaviour
    {
        public RobotDefinition definition;
        public ArticulationBody articulationRoot;
        public bool configureFromDefinitionOnAwake = true;
        [Tooltip("Calibration offset applied to each actuator's exported nominal delay, clamped to its Isaac range.")]
        public int actuatorDelayStepOffset;

        readonly Dictionary<string, ArticulationBody> _joints = new();
        readonly Dictionary<string, ArticulationLoopClosureProxy> _loopJoints = new();
        readonly Dictionary<string, float> _targets = new();
        readonly Dictionary<string, Queue<float>> _targetDelayLines = new();
        readonly List<PhysicsMaterial> _runtimeMaterials = new();

        public bool IsBound { get; private set; }
        public bool IsHealthy => TryValidateHealth(out _);

        public bool TryValidateHealth(out string error)
        {
            if (!IsBound || definition?.joints == null)
                return FailHealth("Articulation driver is not bound", out error);
            ArticulationLoopClosureBinding loopBinding =
                GetComponent<ArticulationLoopClosureBinding>();
            if (loopBinding != null && Application.isPlaying &&
                !loopBinding.TryValidate(out error))
                return false;
            foreach (RobotJointDefinition joint in definition.joints)
            {
                if (!TryRead(joint.name, out float position, out float velocity))
                    return FailHealth(
                        $"Joint '{joint.name}' has no readable degree of freedom",
                        out error);
                if (!float.IsFinite(position) || !float.IsFinite(velocity))
                    return FailHealth(
                        $"Joint '{joint.name}' contains non-finite state",
                        out error);
            }
            error = null;
            return true;
        }

        static bool FailHealth(string message, out string error)
        {
            error = message;
            return false;
        }

        void Awake()
        {
            if (articulationRoot == null)
                articulationRoot = GetComponentInChildren<ArticulationBody>();
            TryBind(out string error);
            if (!string.IsNullOrEmpty(error))
                Debug.LogError(error, this);
        }

        void OnDestroy() => ClearRuntimeMaterials();

        public bool TryBind(out string error)
        {
            _joints.Clear();
            _loopJoints.Clear();
            _targets.Clear();
            _targetDelayLines.Clear();
            ClearRuntimeMaterials();
            IsBound = false;
            if (definition == null || articulationRoot == null)
            {
                error = "ArticulationRobotDriver requires a RobotDefinition and root ArticulationBody";
                return false;
            }
            if (definition.links == null || definition.links.Length == 0 ||
                definition.joints == null || definition.joints.Length == 0)
            {
                error = $"Robot '{definition.robotId}' definition has no links or joints";
                return false;
            }
            if (!TryValidateHierarchy(out error))
                return false;
            var bodiesByName = new Dictionary<string, ArticulationBody>();
            foreach (ArticulationBody body in articulationRoot.GetComponentsInChildren<ArticulationBody>(true))
                bodiesByName[body.name] = body;
            ArticulationLoopClosureBinding loopBinding =
                GetComponent<ArticulationLoopClosureBinding>();
            if (Application.isPlaying && loopBinding != null &&
                !loopBinding.TryValidate(out error))
                return false;
            foreach (RobotJointDefinition joint in definition.joints)
            {
                bool isExternalLoop = loopBinding != null && loopBinding.Contains(joint.name);
                if (isExternalLoop)
                {
                    if (Application.isPlaying)
                    {
                        if (!loopBinding.TryGetProxy(
                                joint.name, out ArticulationLoopClosureProxy proxy))
                        {
                            error = $"Robot '{definition.robotId}' loop joint " +
                                $"'{joint.name}' has no active proxy";
                            return false;
                        }
                        _loopJoints.Add(joint.name, proxy);
                        if (configureFromDefinitionOnAwake)
                            Configure(proxy.parentHinge, joint);
                    }
                    AddTargetState(joint);
                    continue;
                }
                if (joint == null || !bodiesByName.TryGetValue(joint.childLink, out ArticulationBody body))
                {
                    error = $"Robot '{definition.robotId}' is missing child link for joint '{joint?.name}'";
                    return false;
                }
                _joints.Add(joint.name, body);
                AddTargetState(joint);
                if (configureFromDefinitionOnAwake)
                    Configure(body, joint);
            }
            foreach (RobotLinkDefinition link in definition.links)
            {
                if (link == null || !bodiesByName.TryGetValue(link.name, out ArticulationBody body))
                {
                    error = $"Robot '{definition.robotId}' is missing link '{link?.name}'";
                    return false;
                }
                body.mass = Mathf.Max(1e-5f, link.mass);
                body.centerOfMass = link.centerOfMass;
                body.inertiaTensor = new Vector3(
                    Mathf.Max(1e-7f, link.inertiaTensor.x),
                    Mathf.Max(1e-7f, link.inertiaTensor.y),
                    Mathf.Max(1e-7f, link.inertiaTensor.z));
                body.inertiaTensorRotation = link.inertiaTensorRotation;
                if (!ConfigureCollisions(body, link, out error))
                    return false;
            }
            IsBound = true;
            error = null;
            return true;
        }

        public bool TryValidateHierarchy(out string error)
        {
            if (definition == null || articulationRoot == null)
            {
                error = "ArticulationRobotDriver requires a RobotDefinition and root ArticulationBody";
                return false;
            }
            if (definition.links == null || definition.links.Length == 0 ||
                definition.joints == null || definition.joints.Length == 0)
            {
                error = $"Robot '{definition.robotId}' definition has no links or joints";
                return false;
            }
            var bodiesByName = new Dictionary<string, ArticulationBody>();
            foreach (ArticulationBody body in articulationRoot.GetComponentsInChildren<ArticulationBody>(true))
            {
                if (!bodiesByName.TryAdd(body.name, body))
                {
                    error = $"Articulation hierarchy contains duplicate link name '{body.name}'";
                    return false;
                }
            }
            foreach (RobotLinkDefinition link in definition.links)
            {
                if (link == null || !bodiesByName.TryGetValue(link.name, out ArticulationBody body))
                {
                    error = $"Robot '{definition.robotId}' is missing link '{link?.name}'";
                    return false;
                }
                int colliderCount = 0;
                foreach (Collider collider in body.GetComponentsInChildren<Collider>(true))
                    if (FindOwningArticulationBody(collider.transform) == body)
                        colliderCount++;
                if (colliderCount != link.collisionShapeCount)
                {
                    error = $"Link '{link.name}' collision count {colliderCount} != Isaac {link.collisionShapeCount}";
                    return false;
                }
            }
            var jointNames = new HashSet<string>();
            ArticulationLoopClosureBinding loopBinding =
                GetComponent<ArticulationLoopClosureBinding>();
            foreach (RobotJointDefinition joint in definition.joints)
            {
                if (joint == null || string.IsNullOrWhiteSpace(joint.name) ||
                    !jointNames.Add(joint.name) ||
                    (!bodiesByName.ContainsKey(joint.childLink) &&
                     (loopBinding == null || !loopBinding.Contains(joint.name))) ||
                    !RobotDefinitionValidation.IsFiniteUnitAxis(joint.axis) ||
                    !HasExpectedUnit(joint) ||
                    !HasExpectedLimitMode(joint) ||
                    string.IsNullOrWhiteSpace(joint.actuatorModel) ||
                    joint.stiffness < 0f || joint.damping < 0f || joint.forceLimit <= 0f ||
                    joint.actuatorDelayMinSteps < 0 ||
                    joint.actuatorDelayMaxSteps < joint.actuatorDelayMinSteps ||
                    joint.actuatorNominalDelaySteps < joint.actuatorDelayMinSteps ||
                    joint.actuatorNominalDelaySteps > joint.actuatorDelayMaxSteps ||
                    (joint.jointType != ArticulationJointType.FixedJoint && joint.maxVelocity <= 0f))
                {
                    error = $"Robot '{definition.robotId}' has an invalid or unbound joint '{joint?.name}'";
                    return false;
                }
            }
            error = null;
            return true;
        }

        public bool TryRead(string jointName, out float position, out float velocity)
        {
            position = 0f;
            velocity = 0f;
            if (_joints.TryGetValue(jointName, out ArticulationBody body) &&
                body.jointPosition.dofCount >= 1)
            {
                position = body.jointPosition[0];
                velocity = body.jointVelocity[0];
                return true;
            }
            if (_loopJoints.TryGetValue(jointName, out ArticulationLoopClosureProxy proxy) &&
                proxy.parentHinge != null)
            {
                position = proxy.JointPositionRadians;
                velocity = proxy.JointVelocityRadiansPerSecond;
                return float.IsFinite(position) && float.IsFinite(velocity);
            }
            return false;
        }

        public JointStateTelemetry CaptureJointState()
        {
            if (!IsBound || definition?.joints == null)
                throw new InvalidOperationException("ArticulationRobotDriver is not bound");
            var result = new JointStateTelemetry
            {
                names = new string[definition.joints.Length],
                position_units = new string[definition.joints.Length],
                velocity_units = new string[definition.joints.Length],
                position = new float[definition.joints.Length],
                velocity = new float[definition.joints.Length],
            };
            for (int i = 0; i < definition.joints.Length; i++)
            {
                string name = definition.joints[i].name;
                if (!TryRead(name, out float position, out float velocity))
                    throw new InvalidOperationException($"Could not read telemetry joint '{name}'");
                result.names[i] = name;
                result.position_units[i] = definition.joints[i].positionUnit;
                result.velocity_units[i] = definition.joints[i].VelocityUnit;
                result.position[i] = position;
                result.velocity[i] = velocity;
            }
            return result;
        }

        public bool HasJointLimitViolation(float tolerance = 1e-3f)
        {
            if (!IsBound || definition?.joints == null)
                return true;
            foreach (RobotJointDefinition joint in definition.joints)
            {
                if (!TryRead(joint.name, out float position, out _))
                    return true;
                if (joint.jointType == ArticulationJointType.FixedJoint)
                    continue;
                if (joint.HasLimits &&
                    (position < joint.lowerLimit - tolerance ||
                     position > joint.upperLimit + tolerance))
                    return true;
            }
            return false;
        }

        public bool ApplyTerm(PolicyTerm term, float[] action, ref int cursor, float policyDeltaTime)
        {
            if (term == null || term.names == null)
                return false;
            int actionCount = term.FlatSize;
            if (actionCount <= 0 || cursor + actionCount > action.Length)
                return false;
            for (int index = 0; index < term.names.Length; index++)
            {
                string jointName = term.names[index];
                if (!_joints.ContainsKey(jointName) && !_loopJoints.ContainsKey(jointName))
                    return false;
                int actionIndex = actionCount == 1 ? 0 : index;
                float value = action[cursor + actionIndex];
                float scale = term.ScaleAt(actionIndex);
                float target;
                switch (term.integration)
                {
                    case "absolute_lerp":
                        if (term.target_range == null || term.target_range.Length != 2)
                            return false;
                        float alpha = 0.5f * (value + 1f);
                        target = Mathf.Lerp(term.target_range[0], term.target_range[1], alpha);
                        break;
                    case "delta":
                    case "angular_velocity":
                        target = _targets[jointName] + value * scale * policyDeltaTime;
                        break;
                    case "velocity_scaled_delta":
                        target = _targets[jointName] + value * scale *
                                 term.VelocityScaleAt(index) * policyDeltaTime;
                        break;
                    default:
                        target = value * scale + term.OffsetAt(actionIndex);
                        break;
                }
                RobotJointDefinition definitionJoint = definition.FindJoint(jointName);
                if (definitionJoint != null && definitionJoint.HasLimits)
                    target = Mathf.Clamp(target, definitionJoint.lowerLimit, definitionJoint.upperLimit);
                _targets[jointName] = target;
            }
            cursor += actionCount;
            return true;
        }

        public void ResetToDefaults()
        {
            if (!IsBound)
                return;
            foreach (RobotJointDefinition joint in definition.joints)
            {
                _targets[joint.name] = joint.defaultPosition;
                Queue<float> delayLine = _targetDelayLines[joint.name];
                delayLine.Clear();
                for (int i = 0; i < EffectiveDelaySteps(joint); i++)
                    delayLine.Enqueue(joint.defaultPosition);
                if (_joints.TryGetValue(joint.name, out ArticulationBody body))
                {
                    ArticulationDrive drive = body.xDrive;
                    drive.target = joint.ToUnityDriveUnits(joint.defaultPosition);
                    body.xDrive = drive;
                    // A reset must restore simulation state, not only the PD
                    // target. Otherwise a freshly enabled offline rig begins
                    // in the imported zero pose and can fall before its drives
                    // converge to the Isaac initial configuration.
                    body.jointPosition = new ArticulationReducedSpace(joint.defaultPosition);
                    body.jointVelocity = new ArticulationReducedSpace(0f);
                }
                else if (_loopJoints.TryGetValue(
                             joint.name, out ArticulationLoopClosureProxy proxy))
                    Configure(proxy.parentHinge, joint);
            }
        }

        public void ApplyDriveDampingScale(float scale)
        {
            if (!IsBound || definition?.joints == null)
                return;
            scale = Mathf.Max(0f, scale);
            foreach (RobotJointDefinition joint in definition.joints)
            {
                if (_joints.TryGetValue(joint.name, out ArticulationBody body))
                {
                    ArticulationDrive drive = body.xDrive;
                    drive.damping = joint.damping * scale;
                    body.xDrive = drive;
                }
                else if (_loopJoints.TryGetValue(
                             joint.name, out ArticulationLoopClosureProxy proxy) &&
                         proxy.parentHinge != null)
                {
                    JointSpring spring = proxy.parentHinge.spring;
                    spring.damper = joint.damping * scale;
                    proxy.parentHinge.spring = spring;
                }
            }
        }

        int EffectiveDelaySteps(RobotJointDefinition joint) => Mathf.Clamp(
            joint.actuatorNominalDelaySteps + actuatorDelayStepOffset,
            joint.actuatorDelayMinSteps,
            joint.actuatorDelayMaxSteps);

        void FixedUpdate()
        {
            if (!IsBound || definition?.joints == null)
                return;
            foreach (RobotJointDefinition joint in definition.joints)
            {
                if (!_targetDelayLines.TryGetValue(joint.name, out Queue<float> delayLine))
                    continue;
                delayLine.Enqueue(_targets[joint.name]);
                float delayedTarget = delayLine.Dequeue();
                if (_loopJoints.TryGetValue(
                        joint.name, out ArticulationLoopClosureProxy proxy))
                {
                    JointSpring spring = proxy.parentHinge.spring;
                    spring.targetPosition = joint.ToUnityDriveUnits(delayedTarget);
                    proxy.parentHinge.spring = spring;
                    continue;
                }
                if (!_joints.TryGetValue(joint.name, out ArticulationBody body))
                    continue;
                ArticulationDrive drive = body.xDrive;
                drive.target = joint.ToUnityDriveUnits(delayedTarget);
                float position = body.jointPosition.dofCount > 0
                    ? body.jointPosition[0]
                    : joint.defaultPosition;
                drive.forceLimit = joint.EffortLimitAt(position);
                body.xDrive = drive;
            }
        }

        void AddTargetState(RobotJointDefinition joint)
        {
            _targets.Add(joint.name, joint.defaultPosition);
            var delayLine = new Queue<float>();
            for (int i = 0; i < EffectiveDelaySteps(joint); i++)
                delayLine.Enqueue(joint.defaultPosition);
            _targetDelayLines.Add(joint.name, delayLine);
        }

        static void Configure(HingeJoint hinge, RobotJointDefinition joint)
        {
            if (hinge == null)
                return;
            hinge.useLimits = joint.HasLimits;
            if (joint.HasLimits)
            {
                JointLimits limits = hinge.limits;
                limits.min = joint.ToUnityDriveUnits(joint.lowerLimit);
                limits.max = joint.ToUnityDriveUnits(joint.upperLimit);
                hinge.limits = limits;
            }
            JointSpring spring = hinge.spring;
            spring.spring = joint.stiffness;
            spring.damper = joint.damping;
            spring.targetPosition = joint.ToUnityDriveUnits(joint.defaultPosition);
            hinge.spring = spring;
            hinge.useSpring = joint.stiffness > 0f || joint.damping > 0f;
        }

        static void Configure(ArticulationBody body, RobotJointDefinition joint)
        {
            body.jointType = joint.jointType;
            if (joint.jointType == ArticulationJointType.RevoluteJoint)
                body.twistLock = joint.HasLimits
                    ? ArticulationDofLock.LimitedMotion
                    : ArticulationDofLock.FreeMotion;
            body.anchorPosition = joint.anchorPosition;
            Quaternion alignment = Quaternion.FromToRotation(Vector3.right, joint.axis.normalized);
            body.anchorRotation = joint.anchorRotation * alignment;
            body.parentAnchorPosition = joint.parentAnchorPosition;
            body.parentAnchorRotation = joint.parentAnchorRotation * alignment;
            ArticulationDrive drive = body.xDrive;
            if (joint.HasLimits)
            {
                drive.lowerLimit = joint.ToUnityDriveUnits(joint.lowerLimit);
                drive.upperLimit = joint.ToUnityDriveUnits(joint.upperLimit);
            }
            else
            {
                drive.lowerLimit = -360f;
                drive.upperLimit = 360f;
            }
            drive.stiffness = joint.stiffness;
            drive.damping = joint.damping;
            drive.forceLimit = joint.forceLimit;
            drive.target = joint.ToUnityDriveUnits(joint.defaultPosition);
            body.xDrive = drive;
            if (joint.jointType != ArticulationJointType.FixedJoint)
                body.maxJointVelocity = joint.maxVelocity;
        }

        static bool HasExpectedUnit(RobotJointDefinition joint) =>
            (joint.jointType == ArticulationJointType.RevoluteJoint &&
             joint.positionUnit == "radian") ||
            (joint.jointType == ArticulationJointType.PrismaticJoint &&
             joint.positionUnit == "meter") ||
            (joint.jointType == ArticulationJointType.FixedJoint &&
             joint.positionUnit == "fixed");

        static bool HasExpectedLimitMode(RobotJointDefinition joint) =>
            (joint.jointType == ArticulationJointType.RevoluteJoint &&
             (joint.limitMode == "limited" || joint.limitMode == "continuous")) ||
            (joint.jointType == ArticulationJointType.PrismaticJoint &&
             joint.limitMode == "limited") ||
            (joint.jointType == ArticulationJointType.FixedJoint &&
             joint.limitMode == "fixed");

        bool ConfigureCollisions(
            ArticulationBody body,
            RobotLinkDefinition link,
            out string error)
        {
            var owned = new List<Collider>();
            foreach (Collider collider in body.GetComponentsInChildren<Collider>(true))
                if (FindOwningArticulationBody(collider.transform) == body)
                    owned.Add(collider);
            if (owned.Count != link.collisionShapeCount)
            {
                error = $"Link '{link.name}' collision count {owned.Count} != Isaac {link.collisionShapeCount}";
                return false;
            }
            if (!float.IsFinite(link.contactOffset) || !float.IsFinite(link.restOffset))
            {
                error = $"Link '{link.name}' contains non-finite collision offsets";
                return false;
            }
            // Unity Collider exposes a contact offset but no per-shape rest offset.
            // Isaac's sentinel (< 0) and an explicit zero are representable. Any
            // non-zero rest offset must be resolved during geometry preparation or
            // nominal calibration instead of being silently discarded.
            if (link.restOffset > 1e-6f)
            {
                error = $"Link '{link.name}' Isaac rest offset {link.restOffset} m " +
                    "cannot be represented by Unity Collider";
                return false;
            }
            PhysicsMaterial material = null;
            if (link.collisionShapeCount > 0)
            {
                if (link.staticFriction < 0f || link.dynamicFriction < 0f || link.restitution < 0f)
                {
                    error = $"Link '{link.name}' has invalid Isaac material properties";
                    return false;
                }
                material = new PhysicsMaterial($"{definition.robotId}_{link.name}_Isaac")
                {
                    staticFriction = link.staticFriction,
                    dynamicFriction = link.dynamicFriction,
                    bounciness = link.restitution,
                    hideFlags = HideFlags.HideAndDontSave,
                };
                _runtimeMaterials.Add(material);
            }
            foreach (Collider collider in owned)
            {
                collider.enabled = link.collisionEnabled;
                if (link.contactOffset > 0f)
                    collider.contactOffset = link.contactOffset;
                collider.sharedMaterial = material;
            }
            error = null;
            return true;
        }

        static ArticulationBody FindOwningArticulationBody(Transform item)
        {
            // GetComponentInParent returns null for components on prefab assets in
            // some Unity editor contexts. Walking the authored hierarchy keeps
            // validation identical for prefab assets and instantiated rigs.
            Transform current = item;
            while (current != null)
            {
                ArticulationBody body = current.GetComponent<ArticulationBody>();
                if (body != null)
                    return body;
                current = current.parent;
            }
            return null;
        }

        void ClearRuntimeMaterials()
        {
            foreach (PhysicsMaterial material in _runtimeMaterials)
            {
                if (material == null)
                    continue;
                if (Application.isPlaying)
                    Destroy(material);
                else
                    DestroyImmediate(material);
            }
            _runtimeMaterials.Clear();
        }
    }
}
