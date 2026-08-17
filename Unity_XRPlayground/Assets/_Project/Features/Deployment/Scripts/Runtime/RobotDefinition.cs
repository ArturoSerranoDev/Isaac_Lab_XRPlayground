using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class RobotJointEffortLimitSample
    {
        public float position;
        public float maxEffort;
    }

    [Serializable]
    public sealed class RobotLinkDefinition
    {
        public string name;
        public float mass = 1f;
        public Vector3 centerOfMass;
        public Vector3 inertiaTensor = Vector3.one;
        public Quaternion inertiaTensorRotation = Quaternion.identity;
        public int collisionShapeCount;
        public bool collisionEnabled;
        public float staticFriction = -1f;
        public float dynamicFriction = -1f;
        public float restitution = -1f;
        public float contactOffset = -1f;
        public float restOffset = -1f;
        public RobotCollisionShapeDefinition[] collisionShapes;
    }

    [Serializable]
    public sealed class RobotCollisionShapeDefinition
    {
        public string shapeType;
        public Vector3 localPosition;
        public Quaternion localRotation = Quaternion.identity;
        public Vector3 scale = Vector3.one;
        public Vector3 size;
        public float radius;
        public float height;
        public Vector3 axis = Vector3.up;
        public string sourcePrimPath;
    }

    [Serializable]
    public sealed class RobotJointDefinition
    {
        public string name;
        public string parentLink;
        public string childLink;
        public ArticulationJointType jointType = ArticulationJointType.RevoluteJoint;
        public string positionUnit = "radian";
        public string limitMode = "limited";
        public Vector3 axis = Vector3.right;
        public Vector3 parentAnchorPosition;
        public Quaternion parentAnchorRotation = Quaternion.identity;
        public Vector3 anchorPosition;
        public Quaternion anchorRotation = Quaternion.identity;
        public float lowerLimit = -3.1415927f;
        public float upperLimit = 3.1415927f;
        public float defaultPosition;
        public float stiffness = 1000f;
        public float damping = 100f;
        public float forceLimit = 1000f;
        public float maxVelocity = 1f;
        public string actuatorGroup;
        public string actuatorModel = "PhysXDrive";
        public bool actuatorIsImplicit = true;
        public int actuatorDelayMinSteps;
        public int actuatorDelayMaxSteps;
        public int actuatorNominalDelaySteps;
        public RobotJointEffortLimitSample[] effortLimitCurve;

        public bool IsAngular => jointType == ArticulationJointType.RevoluteJoint;
        public bool HasLimits => limitMode == "limited";
        public string VelocityUnit => jointType switch
        {
            ArticulationJointType.RevoluteJoint => "radian_per_second",
            ArticulationJointType.PrismaticJoint => "meter_per_second",
            _ => "fixed",
        };
        public float ToUnityDriveUnits(float value) => IsAngular ? value * Mathf.Rad2Deg : value;

        public float EffortLimitAt(float position)
        {
            if (effortLimitCurve == null || effortLimitCurve.Length == 0)
                return forceLimit;
            if (position <= effortLimitCurve[0].position)
                return effortLimitCurve[0].maxEffort;
            int last = effortLimitCurve.Length - 1;
            if (position >= effortLimitCurve[last].position)
                return effortLimitCurve[last].maxEffort;
            for (int i = 1; i < effortLimitCurve.Length; i++)
            {
                RobotJointEffortLimitSample upper = effortLimitCurve[i];
                if (position > upper.position)
                    continue;
                RobotJointEffortLimitSample lower = effortLimitCurve[i - 1];
                return Mathf.Lerp(
                    lower.maxEffort,
                    upper.maxEffort,
                    Mathf.InverseLerp(lower.position, upper.position, position));
            }
            return forceLimit;
        }
    }

    public static class RobotDefinitionValidation
    {
        public static bool IsFiniteUnitAxis(Vector3 axis) =>
            float.IsFinite(axis.x) && float.IsFinite(axis.y) && float.IsFinite(axis.z) &&
            Mathf.Abs(axis.sqrMagnitude - 1f) <= 1e-3f;
    }

    [Serializable]
    public sealed class RobotFixedJointDefinition
    {
        public string name;
        public string parentLink;
        public string childLink;
        public Vector3 parentAnchorPosition;
        public Quaternion parentAnchorRotation = Quaternion.identity;
        public Vector3 anchorPosition;
        public Quaternion anchorRotation = Quaternion.identity;
    }

    [Serializable]
    public sealed class RobotAuxiliaryJointDefinition
    {
        public string name;
        public string sourcePrimPath;
        public string sourceJointType;
        public string jointType;
        public string topologyRole;
        public string parentLink;
        public string childLink;
        public Vector3 axis = Vector3.right;
        public Vector3 parentAnchorPosition;
        public Quaternion parentAnchorRotation = Quaternion.identity;
        public Vector3 anchorPosition;
        public Quaternion anchorRotation = Quaternion.identity;

        public bool IsLoopClosure => topologyRole == "loop_closure";
    }

    [CreateAssetMenu(menuName = "XRPlayground/Deployment/Robot Definition")]
    public sealed class RobotDefinition : ScriptableObject
    {
        public string robotId;
        public string sourceAsset;
        public string sourceSha256;
        public string definitionSha256;
        public string provenance;
        public string redistributionLicense;
        public RobotLinkDefinition[] links;
        public RobotJointDefinition[] joints;
        public RobotFixedJointDefinition[] fixedJoints;
        public RobotAuxiliaryJointDefinition[] auxiliaryJoints;

        public RobotJointDefinition FindJoint(string jointName)
        {
            if (joints == null)
                return null;
            foreach (RobotJointDefinition joint in joints)
                if (joint != null && joint.name == jointName)
                    return joint;
            return null;
        }
    }
}
