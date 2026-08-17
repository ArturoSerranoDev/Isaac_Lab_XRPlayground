using UnityEngine;

namespace XRPlayground.Deployment
{
    [DefaultExecutionOrder(50)]
    [DisallowMultipleComponent]
    public sealed class SpotFollowStationAdapter : StationAdapterBase
    {
        public SpotLocomotionStationAdapter locomotion;
        public PolicyRuntime locomotionPolicy;
        public ArticulationBody rootBody;
        public Transform target;
        [Tooltip("Optional body-heading reference. XR Origin is discovered from the camera hierarchy when omitted.")]
        public Transform headingTarget;
        public Transform environmentAnchor;
        [Min(0f)] public float followDistanceMeters = 1.25f;
        [Min(0.1f)] public float maxPlanarCommandMeters = 2.5f;
        [Tooltip("Unity locomotion safety derating applied after the high-level ONNX policy. Keep at 1 for parity diagnostics.")]
        [Range(0.1f, 1f)] public float unityCommandScale = 0.4f;
        [Tooltip("Blend of low-speed geometric steering used to keep cross-engine navigation convergent. Set to zero for pure ONNX parity diagnostics.")]
        [Range(0f, 1f)] public float unitySteeringAssistBlend = 1f;
        [Min(0.05f)] public float assistedForwardSpeed = 0.2f;
        [Tooltip("Optional lateral command. Keep at zero for the stable Unity profile; enable only after lateral gait calibration.")]
        [Min(0f)] public float assistedLateralSpeed = 0f;
        [Tooltip("Command-space braking lead used to fade the Unity gait before it reaches the requested follow point.")]
        [Min(0f)] public float assistedStopLeadMeters = 0.75f;
        [Tooltip("Optional physical yaw rate. Keep at zero for the stable Unity profile; use non-zero values only while calibrating a turn-capable locomotion bundle.")]
        [Min(0f)] public float assistedYawRate = 0f;
        [Tooltip("Orient the articulation toward the current player target as part of an explicit offline reset. This changes no root position and is never applied while the policy is running.")]
        public bool faceTargetOnReset = true;

        float[] _lowObservations;
        float[] _lowActions;
        Vector3 _command;
        Vector2 _goalPlanarIsaac;
        float _goalHeadingError;
        int _physicsSteps;
        bool _active;
        bool _evaluationFell;
        float _evaluationDistanceSum;
        float _evaluationHeadingSum;
        int _evaluationSamples;
        float _evaluationScore;
        Transform _resolvedHeadingTarget;

        public override string AdapterId => "spot_follow";
        public override int ObservationDimension => 10;
        public override int ActionDimension => 3;
        public override float TaskScore => _evaluationScore;
        public override bool HasJointLimitViolation =>
            locomotion == null || locomotion.HasJointLimitViolation;

        public override void BindContract(PolicyContract contract)
        {
            base.BindContract(contract);
            if (locomotion == null)
                throw new System.InvalidOperationException("Spot locomotion adapter is not assigned");
            string error = null;
            if (locomotionPolicy == null || !locomotionPolicy.TryLoad(out error))
            {
                error ??= "locomotion PolicyRuntime is not assigned";
                throw new System.InvalidOperationException($"Spot locomotion dependency failed: {error}");
            }
            if (locomotionPolicy.Contract.policy_id != "spot.locomotion" ||
                locomotionPolicy.Contract.ObservationDimension != 48 ||
                locomotionPolicy.Contract.ActionDimension != 12 ||
                locomotionPolicy.Contract.timing.policy_hz != 50)
                throw new System.InvalidOperationException(
                    "Spot Follow requires the validated 48 -> 12 locomotion policy at 50 Hz");
            PolicyDependency expectedDependency = null;
            if (contract.dependencies != null)
                foreach (PolicyDependency dependency in contract.dependencies)
                    if (dependency != null && dependency.policy_id == "spot.locomotion")
                        expectedDependency = dependency;
            if (expectedDependency == null ||
                expectedDependency.model_sha256 != locomotionPolicy.Contract.model_sha256 ||
                expectedDependency.torchscript_sha256 != locomotionPolicy.Contract.torchscript_sha256)
                throw new System.InvalidOperationException(
                    "Spot locomotion bundle does not match the dependency hashes in the Follow contract");
            locomotion.BindContract(locomotionPolicy.Contract);
            _lowObservations = new float[locomotionPolicy.Contract.ObservationDimension];
            _lowActions = new float[locomotionPolicy.Contract.ActionDimension];
        }

        public override void ResetStation(int seed)
        {
            if (target == null && Camera.main != null)
                target = Camera.main.transform;
            _resolvedHeadingTarget = ResolveHeadingTarget();
            Healthy = locomotion != null && locomotionPolicy != null && rootBody != null && target != null;
            _command = Vector3.zero;
            _physicsSteps = 0;
            _active = true;
            locomotionPolicy?.ResetState();
            locomotion?.ResetStation(seed);
            if (faceTargetOnReset)
                FaceTargetAtReset();
            Healthy &= locomotion != null && locomotion.IsHealthy;
        }

        void OnDisable() => _active = false;

        void FixedUpdate()
        {
            if (!_active || locomotionPolicy?.Contract == null || _lowObservations == null)
                return;
            int cadence = Mathf.Max(1, locomotionPolicy.Contract.timing.deployment_decimation);
            if ((_physicsSteps++ % cadence) != 0)
                return;
            locomotion.SetVelocityCommand(_command);
            locomotion.BuildObservation(_lowObservations);
            if (locomotionPolicy.TryInfer(_lowObservations, _lowActions, out _))
                locomotion.ApplyAction(_lowActions, locomotionPolicy.Contract.timing.policy_dt);
            else
                Healthy = false;
        }

        public override void BuildObservation(float[] destination)
        {
            int cursor = 0;
            Vector3 linearBodyUnity = rootBody.transform.InverseTransformDirection(rootBody.linearVelocity);
            Vector3 gravityBodyUnity = rootBody.transform.InverseTransformDirection(Physics.gravity.normalized);
            Vector3 linearBody = DeploymentFrameConverter.UnityToIsaac(linearBodyUnity);
            Vector3 projectedGravity = DeploymentFrameConverter.UnityToIsaac(gravityBodyUnity);

            FollowGoal(out Vector3 goalPosition, out Vector3 goalForward);
            Vector3 deltaWorld = goalPosition - rootBody.transform.position;
            YawFrame(out Vector3 isaacXWorld, out Vector3 isaacYWorld, out Vector3 up);
            Vector3 deltaIsaac = new(
                Vector3.Dot(deltaWorld, isaacXWorld),
                Vector3.Dot(deltaWorld, isaacYWorld),
                Vector3.Dot(deltaWorld, up));
            // UniformPose2dCommand is planar and keeps the goal at the robot's
            // nominal root height. Never feed HMD eye height into its z term.
            // Keep distant XR targets inside the square command envelope used
            // for training. The real target is retained and this window is
            // recomputed at 5 Hz, so Spot continues walking toward it without
            // exposing the policy to arbitrary scene-scale distances.
            deltaIsaac.x = Mathf.Clamp(
                deltaIsaac.x, -maxPlanarCommandMeters, maxPlanarCommandMeters);
            deltaIsaac.y = Mathf.Clamp(
                deltaIsaac.y, -maxPlanarCommandMeters, maxPlanarCommandMeters);
            deltaIsaac.z = 0f;
            float heading = Mathf.Atan2(
                Vector3.Dot(goalForward, isaacYWorld),
                Vector3.Dot(goalForward, isaacXWorld));
            _goalPlanarIsaac = new Vector2(deltaIsaac.x, deltaIsaac.y);
            _goalHeadingError = Mathf.DeltaAngle(0f, heading * Mathf.Rad2Deg) * Mathf.Deg2Rad;
            foreach (PolicyTerm term in Contract.observations)
            {
                string name = term.name.ToLowerInvariant();
                if (name.Contains("base_lin_vel"))
                    WriteTermVector(destination, ref cursor, term, linearBody);
                else if (name.Contains("projected_gravity"))
                    WriteTermVector(destination, ref cursor, term, projectedGravity);
                else if (name.Contains("pose_command") || name == "generated_commands")
                {
                    WriteTermValue(destination, ref cursor, term, 0, deltaIsaac.x);
                    WriteTermValue(destination, ref cursor, term, 1, deltaIsaac.y);
                    WriteTermValue(destination, ref cursor, term, 2, deltaIsaac.z);
                    WriteTermValue(destination, ref cursor, term, 3, heading);
                }
                else throw new System.InvalidOperationException(
                    $"Unsupported Spot Follow observation '{term.name}'");
            }
            VerifyCursor(cursor, destination.Length);
        }

        public override void ApplyAction(float[] action, float policyDeltaTime)
        {
            PolicyTerm term = Contract.actions[0];
            _command = new Vector3(
                action[0] * term.ScaleAt(0) + term.OffsetAt(0),
                action[1] * term.ScaleAt(1) + term.OffsetAt(1),
                action[2] * term.ScaleAt(2) + term.OffsetAt(2));
            _command *= unityCommandScale;
            Vector3 steering = BuildSteeringAssist();
            _command = Vector3.Lerp(_command, steering, unitySteeringAssistBlend);
            // Keep the low-level locomotion policy inside the velocity range
            // used for training even when the high-level network overshoots.
            _command.x = Mathf.Clamp(_command.x, -2f, 3f);
            _command.y = Mathf.Clamp(_command.y, -1.5f, 1.5f);
            _command.z = Mathf.Clamp(_command.z, -2f, 2f);
        }

        Vector3 BuildSteeringAssist()
        {
            float distance = _goalPlanarIsaac.magnitude;
            float bearing = distance > 1e-4f
                ? Mathf.Atan2(_goalPlanarIsaac.y, _goalPlanarIsaac.x)
                : 0f;
            // Use signed proportional forward speed so crossing the follow
            // point produces a braking/reverse command instead of letting the
            // learned gait coast while the robot turns through 180 degrees.
            // Unity's handedness means an Isaac-positive yaw command turns the
            // physical articulation opposite the positive bearing computed in
            // this Unity world frame, hence the explicit sign below.
            float forwardError = Mathf.Sign(_goalPlanarIsaac.x) * Mathf.Max(
                0f, Mathf.Abs(_goalPlanarIsaac.x) - assistedStopLeadMeters);
            float forward = Mathf.Abs(forwardError) > 0.05f
                ? Mathf.Clamp(forwardError * 0.25f,
                    -assistedForwardSpeed, assistedForwardSpeed)
                : 0f;
            float lateral = assistedLateralSpeed > 0f &&
                            Mathf.Abs(_goalPlanarIsaac.y) > 0.15f
                ? -Mathf.Clamp(_goalPlanarIsaac.y * 0.25f,
                    -assistedLateralSpeed, assistedLateralSpeed)
                : 0f;
            // The current Unity articulation transfers the learned forward
            // and lateral gaits reliably, while sustained learned yaw can
            // accumulate a fall. Preserve yaw as an explicit opt-in
            // calibration knob and keep the playable profile translation-only.
            float yawError = distance > 0.45f ? bearing : _goalHeadingError;
            float yaw = assistedYawRate > 0f
                ? -Mathf.Clamp(yawError * 0.8f, -assistedYawRate, assistedYawRate)
                : 0f;
            return new Vector3(forward, lateral, yaw);
        }

        void FaceTargetAtReset()
        {
            if (rootBody == null || target == null)
                return;
            Vector3 up = environmentAnchor != null ? environmentAnchor.up : Vector3.up;
            Vector3 desired = Vector3.ProjectOnPlane(
                target.position - rootBody.transform.position, up).normalized;
            Vector3 current = Vector3.ProjectOnPlane(rootBody.transform.right, up).normalized;
            if (desired.sqrMagnitude < 1e-6f || current.sqrMagnitude < 1e-6f)
                return;
            Quaternion rotation = Quaternion.FromToRotation(current, desired) *
                                  rootBody.transform.rotation;
            rootBody.TeleportRoot(rootBody.transform.position, rotation);
            rootBody.linearVelocity = Vector3.zero;
            rootBody.angularVelocity = Vector3.zero;
        }

        public override StationTelemetrySnapshot CaptureTelemetry() => Capture(
            locomotion != null ? locomotion.robot : null,
            rootBody,
            environmentAnchor);

        void FollowGoal(out Vector3 position, out Vector3 forward)
        {
            Vector3 up = environmentAnchor != null ? environmentAnchor.up : Vector3.up;
            Transform heading = headingTarget != null
                ? headingTarget
                : _resolvedHeadingTarget != null ? _resolvedHeadingTarget : ResolveHeadingTarget();
            forward = Vector3.ProjectOnPlane(heading.forward, up).normalized;
            if (forward.sqrMagnitude < 1e-6f)
                forward = Vector3.ProjectOnPlane(target.up, up).normalized;
            if (forward.sqrMagnitude < 1e-6f)
                forward = Vector3.ProjectOnPlane(rootBody.transform.forward, up).normalized;
            position = target.position - forward * followDistanceMeters;
            Vector3 rootToGoal = position - rootBody.transform.position;
            position -= up * Vector3.Dot(rootToGoal, up);
        }

        Transform ResolveHeadingTarget()
        {
            if (headingTarget != null)
                return headingTarget;
            for (Transform current = target; current != null; current = current.parent)
                if (current.name.ToLowerInvariant().Contains("xr origin"))
                    return current;
            return target;
        }

        void YawFrame(out Vector3 isaacXWorld, out Vector3 isaacYWorld, out Vector3 up)
        {
            up = environmentAnchor != null ? environmentAnchor.up : Vector3.up;
            // DeploymentFrameConverter maps Unity local +X/+Z to Isaac +X/+Y.
            // UniformPose2dCommand removes roll and pitch with yaw_quat(), so
            // project the robot's Isaac +X axis onto the environment plane and
            // reconstruct an orthonormal yaw-only frame around its up axis.
            isaacXWorld = Vector3.ProjectOnPlane(rootBody.transform.right, up).normalized;
            if (isaacXWorld.sqrMagnitude < 1e-6f)
                isaacXWorld = Vector3.ProjectOnPlane(rootBody.transform.forward, up).normalized;
            isaacYWorld = Vector3.Cross(isaacXWorld, up).normalized;
        }

        public override void BeginEvaluationScenario(int seed)
        {
            base.BeginEvaluationScenario(seed);
            _evaluationFell = false;
            _evaluationDistanceSum = 0f;
            _evaluationHeadingSum = 0f;
            _evaluationSamples = 0;
            _evaluationScore = 0f;
        }

        public override void SampleEvaluationScenario(float policyDeltaTime)
        {
            base.SampleEvaluationScenario(policyDeltaTime);
            if (rootBody == null || target == null)
            {
                _evaluationFell = true;
                return;
            }
            FollowGoal(out Vector3 goalPosition, out Vector3 goalForward);
            Vector3 deltaWorld = goalPosition - rootBody.transform.position;
            YawFrame(out Vector3 isaacXWorld, out Vector3 isaacYWorld, out _);
            Vector3 deltaIsaac = new(
                Vector3.Dot(deltaWorld, isaacXWorld),
                Vector3.Dot(deltaWorld, isaacYWorld),
                0f);
            _evaluationDistanceSum += new Vector2(deltaIsaac.x, deltaIsaac.y).magnitude;
            _evaluationHeadingSum += Mathf.Abs(Mathf.Atan2(
                Vector3.Dot(goalForward, isaacYWorld),
                Vector3.Dot(goalForward, isaacXWorld)));
            _evaluationSamples++;

            Vector3 up = environmentAnchor != null ? environmentAnchor.up : Vector3.up;
            float height = IsaacLocalPosition(environmentAnchor, rootBody.transform.position).z;
            float fallTilt = locomotion != null ? locomotion.fallTiltDegrees : 60f;
            float minimumHeight = locomotion != null ? locomotion.minimumRootHeightIsaac : 0.2f;
            if (Vector3.Angle(rootBody.transform.up, up) >= fallTilt || height < minimumHeight)
                _evaluationFell = true;
        }

        public override StationEvaluationSummary CompleteEvaluationScenario()
        {
            EvaluationMetrics.follow_observation_dim = ObservationDimension;
            EvaluationMetrics.high_level_policy_hz = 5f;
            EvaluationMetrics.low_level_policy_hz = 50f;
            EvaluationMetrics.target_distance_error_m = _evaluationSamples > 0
                ? _evaluationDistanceSum / _evaluationSamples
                : float.MaxValue;
            EvaluationMetrics.heading_error_rad = _evaluationSamples > 0
                ? _evaluationHeadingSum / _evaluationSamples
                : Mathf.PI;
            EvaluationMetrics.fall_rate = _evaluationFell ? 1f : 0f;
            _evaluationScore = _evaluationFell
                ? 0f
                : 1f / (1f + EvaluationMetrics.target_distance_error_m +
                         EvaluationMetrics.heading_error_rad);
            return base.CompleteEvaluationScenario();
        }
    }
}
