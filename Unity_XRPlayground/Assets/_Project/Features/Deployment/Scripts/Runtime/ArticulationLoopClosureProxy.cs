using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    /// <summary>
    /// Represents one revolute edge that cannot live in an ArticulationBody tree.
    /// A collision-free Rigidbody carries a fixed joint to the child link and a
    /// hinge joint to the parent link, allowing PhysX to solve the closed cycle.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class ArticulationLoopClosureProxy : MonoBehaviour
    {
        public string constraintName;
        public string parentLink;
        public string childLink;
        public ArticulationBody parentBody;
        public ArticulationBody childBody;
        public Rigidbody proxyBody;
        public FixedJoint childLock;
        public HingeJoint parentHinge;
        public Vector3 parentAnchorPosition;
        public Quaternion parentAnchorRotation = Quaternion.identity;
        public Vector3 childAnchorPosition;
        public Quaternion childAnchorRotation = Quaternion.identity;
        public Vector3 axis = Vector3.right;
        [Min(1e-5f)] public float proxyMass = 0.1f;
        [Range(1, 255)] public int solverIterations = 20;
        [Range(1, 255)] public int solverVelocityIterations = 10;
        [Min(0f)] public float maximumHealthyAnchorErrorMeters = 0.005f;
        [Min(0f)] public float maximumHealthyAxisErrorDegrees = 2f;

        Vector3 _referenceInAnchorFrame;
        float _zeroAngleDegrees;

        public float AnchorErrorMeters => parentBody == null || childBody == null
            ? float.PositiveInfinity
            : Vector3.Distance(ParentAnchorWorld, ChildAnchorWorld);

        public float AxisErrorDegrees
        {
            get
            {
                if (parentBody == null || childBody == null || axis.sqrMagnitude < 0.999f)
                    return float.PositiveInfinity;
                Vector3 parentAxis = parentBody.transform.rotation *
                    (parentAnchorRotation * axis.normalized);
                Vector3 childAxis = childBody.transform.rotation *
                    (childAnchorRotation * axis.normalized);
                float angle = Vector3.Angle(parentAxis, childAxis);
                return Mathf.Min(angle, 180f - angle);
            }
        }

        public Vector3 ParentAnchorWorld => parentBody.transform.TransformPoint(parentAnchorPosition);
        public Vector3 ChildAnchorWorld => childBody.transform.TransformPoint(childAnchorPosition);

        public float JointPositionRadians
        {
            get
            {
                if (parentBody == null || childBody == null)
                    return float.NaN;
                float current = RelativeAngleDegrees();
                return Mathf.DeltaAngle(_zeroAngleDegrees, current) * Mathf.Deg2Rad;
            }
        }

        public float JointVelocityRadiansPerSecond
        {
            get
            {
                if (parentBody == null || childBody == null)
                    return float.NaN;
                Vector3 worldAxis = parentBody.transform.rotation *
                    (parentAnchorRotation * axis.normalized);
                return Vector3.Dot(
                    childBody.angularVelocity - parentBody.angularVelocity,
                    worldAxis);
            }
        }

        public void Configure(
            string jointName,
            ArticulationBody parent,
            ArticulationBody child,
            Vector3 parentPosition,
            Quaternion parentRotation,
            Vector3 childPosition,
            Quaternion childRotation,
            Vector3 jointAxis)
        {
            if (parent == null || child == null)
                throw new ArgumentNullException(
                    parent == null ? nameof(parent) : nameof(child));
            if (parent == child)
                throw new ArgumentException("A loop closure cannot connect a link to itself");
            if (jointAxis.sqrMagnitude < 0.999f || jointAxis.sqrMagnitude > 1.001f)
                throw new ArgumentException("A loop closure axis must be normalized", nameof(jointAxis));

            constraintName = jointName;
            parentLink = parent.name;
            childLink = child.name;
            parentBody = parent;
            childBody = child;
            parentAnchorPosition = parentPosition;
            parentAnchorRotation = parentRotation;
            childAnchorPosition = childPosition;
            childAnchorRotation = childRotation;
            axis = jointAxis;
            Vector3 helper = Mathf.Abs(Vector3.Dot(axis, Vector3.up)) < 0.9f
                ? Vector3.up
                : Vector3.right;
            _referenceInAnchorFrame = Vector3.Cross(axis, helper).normalized;
            _zeroAngleDegrees = RelativeAngleDegrees();

            transform.SetPositionAndRotation(
                ChildAnchorWorld,
                child.transform.rotation * childAnchorRotation);
            proxyBody = GetComponent<Rigidbody>();
            if (proxyBody == null)
                proxyBody = gameObject.AddComponent<Rigidbody>();
            if (proxyBody == null)
                throw new InvalidOperationException(
                    "Unity refused the loop Rigidbody. Spawn the proxy as a separate scene root " +
                    "outside every ArticulationBody transform hierarchy.");
            proxyBody.mass = Mathf.Max(1e-5f, proxyMass);
            proxyBody.useGravity = false;
            proxyBody.isKinematic = false;
            proxyBody.detectCollisions = false;
            proxyBody.interpolation = RigidbodyInterpolation.None;
            proxyBody.collisionDetectionMode = CollisionDetectionMode.Discrete;
            proxyBody.solverIterations = solverIterations;
            proxyBody.solverVelocityIterations = solverVelocityIterations;

            childLock = GetComponent<FixedJoint>();
            if (childLock == null)
                childLock = gameObject.AddComponent<FixedJoint>();
            childLock.autoConfigureConnectedAnchor = false;
            childLock.anchor = Vector3.zero;
            childLock.connectedAnchor = childAnchorPosition;
            childLock.connectedBody = null;
            childLock.connectedArticulationBody = child;
            childLock.enableCollision = false;
            childLock.enablePreprocessing = false;
            childLock.breakForce = float.PositiveInfinity;
            childLock.breakTorque = float.PositiveInfinity;

            parentHinge = GetComponent<HingeJoint>();
            if (parentHinge == null)
                parentHinge = gameObject.AddComponent<HingeJoint>();
            parentHinge.autoConfigureConnectedAnchor = false;
            parentHinge.anchor = Vector3.zero;
            parentHinge.connectedAnchor = parentAnchorPosition;
            parentHinge.axis = axis;
            parentHinge.connectedBody = null;
            parentHinge.connectedArticulationBody = parent;
            parentHinge.enableCollision = false;
            parentHinge.enablePreprocessing = false;
            parentHinge.useLimits = false;
            parentHinge.useMotor = false;
            parentHinge.useSpring = false;
            parentHinge.breakForce = float.PositiveInfinity;
            parentHinge.breakTorque = float.PositiveInfinity;

            if (!TryValidate(out string error))
                throw new InvalidOperationException(error);
        }

        public bool TryValidate(out string error)
        {
            if (string.IsNullOrWhiteSpace(constraintName) || parentBody == null ||
                childBody == null || proxyBody == null || childLock == null ||
                parentHinge == null)
                return Fail("Loop closure has incomplete bindings", out error);
            if (parentBody == childBody || childLock.connectedArticulationBody != childBody ||
                parentHinge.connectedArticulationBody != parentBody ||
                childLock.connectedBody != null || parentHinge.connectedBody != null)
                return Fail("Loop closure has invalid articulation connections", out error);
            if (axis.sqrMagnitude < 0.999f || axis.sqrMagnitude > 1.001f)
                return Fail("Loop closure axis is not normalized", out error);
            if (!float.IsFinite(AnchorErrorMeters) ||
                AnchorErrorMeters > maximumHealthyAnchorErrorMeters)
                return Fail(
                    $"Loop closure anchor error {AnchorErrorMeters:F6} m exceeds " +
                    $"{maximumHealthyAnchorErrorMeters:F6} m",
                    out error);
            if (!float.IsFinite(AxisErrorDegrees) ||
                AxisErrorDegrees > maximumHealthyAxisErrorDegrees)
                return Fail(
                    $"Loop closure axis error {AxisErrorDegrees:F3} degrees exceeds " +
                    $"{maximumHealthyAxisErrorDegrees:F3} degrees",
                    out error);
            if (!float.IsFinite(JointPositionRadians) ||
                !float.IsFinite(JointVelocityRadiansPerSecond))
                return Fail("Loop closure derived joint state is non-finite", out error);
            error = null;
            return true;
        }

        float RelativeAngleDegrees()
        {
            Quaternion parentFrame = parentBody.transform.rotation * parentAnchorRotation;
            Quaternion childFrame = childBody.transform.rotation * childAnchorRotation;
            Vector3 worldAxis = parentFrame * axis.normalized;
            Vector3 parentReference = parentFrame * _referenceInAnchorFrame;
            Vector3 childReference = childFrame * _referenceInAnchorFrame;
            return Vector3.SignedAngle(parentReference, childReference, worldAxis);
        }

        static bool Fail(string message, out string error)
        {
            error = message;
            return false;
        }
    }
}
