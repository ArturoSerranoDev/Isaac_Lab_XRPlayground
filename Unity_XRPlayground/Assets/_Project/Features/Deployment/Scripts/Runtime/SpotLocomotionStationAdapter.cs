using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class SpotLocomotionStationAdapter : StationAdapterBase
    {
        public ArticulationRobotDriver robot;
        public ArticulationBody rootBody;
        public Transform environmentAnchor;
        public string[] jointNames = new string[12];
        public Vector3 velocityCommandIsaac;
        public ContactSensor[] footContactSensors = new ContactSensor[4];
        public float fallTiltDegrees = 60f;
        public float minimumRootHeightIsaac = 0.2f;
        [Tooltip("Unity drive-response calibration applied only to physical joint excursion. Raw ONNX actions remain unchanged in observations and telemetry.")]
        [Range(0.05f, 1f)] public float unityActionScale = 1f;
        [Tooltip("Seconds used to blend from the reset stance into ONNX joint excursions.")]
        [Min(0f)] public float unityActionRampSeconds = 2f;
        [Tooltip("Unity ArticulationDrive damping multiplier. The exported Isaac value is preserved in RobotDefinition; this is a deployment calibration.")]
        [Min(0f)] public float unityDriveDampingScale = 6.666667f;
        [Tooltip("Command magnitude that applies the full learned joint excursion. Smaller follow commands fade toward the trained default stance so Spot can settle instead of coasting past a stationary player.")]
        [Min(0.01f)] public float unityFullMotionCommand = 0.2f;
        [Tooltip("Seconds used to blend physical joint excursion when the navigation command starts or stops. Raw ONNX actions remain unchanged.")]
        [Min(0.01f)] public float unityMotionBlendSeconds = 0.35f;

        readonly float[] _lastActions = new float[12];
        readonly float[] _unityActions = new float[12];
        bool _resetPoseCached;
        float _unityActionBlend;
        float _commandActionBlend;
        float _targetCommandActionBlend;
        Vector3 _resetWorldPosition;
        Quaternion _resetWorldRotation;
        bool _evaluationFell;
        float _evaluationTrackingSum;
        int _evaluationValidFootSamples;
        int _evaluationSamples;
        float _evaluationScore;

        public override string AdapterId => "spot_loco";
        public override int ObservationDimension => 48;
        public override int ActionDimension => 12;
        public override float TaskScore => _evaluationScore;
        public override bool HasJointLimitViolation =>
            robot == null || robot.HasJointLimitViolation();

        public override void BindContract(PolicyContract contract)
        {
            base.BindContract(contract);
            if (contract.actions == null || contract.actions.Length != 1 ||
                contract.actions[0].names == null || contract.actions[0].names.Length != 12)
                throw new System.InvalidOperationException(
                    "Spot locomotion contract must provide 12 ordered joint names");
            jointNames = (string[])contract.actions[0].names.Clone();
        }

        public void SetVelocityCommand(Vector3 commandIsaac)
        {
            velocityCommandIsaac = commandIsaac;
            float commandMagnitude = Mathf.Max(
                Mathf.Abs(commandIsaac.x),
                Mathf.Abs(commandIsaac.y),
                Mathf.Abs(commandIsaac.z));
            _targetCommandActionBlend = Mathf.Clamp01(
                commandMagnitude / Mathf.Max(0.01f, unityFullMotionCommand));
        }

        public override void ResetStation(int seed)
        {
            Healthy = robot != null && robot.IsHealthy && rootBody != null && jointNames?.Length == 12;
            if (rootBody != null)
            {
                if (!_resetPoseCached)
                {
                    _resetWorldPosition = rootBody.transform.position;
                    _resetWorldRotation = rootBody.transform.rotation;
                    _resetPoseCached = true;
                }
                rootBody.TeleportRoot(_resetWorldPosition, _resetWorldRotation);
                rootBody.linearVelocity = Vector3.zero;
                rootBody.angularVelocity = Vector3.zero;
            }
            robot?.ResetToDefaults();
            robot?.ApplyDriveDampingScale(unityDriveDampingScale);
            System.Array.Clear(_lastActions, 0, _lastActions.Length);
            _unityActionBlend = unityActionRampSeconds <= 0f ? 1f : 0f;
            _commandActionBlend = 0f;
            _targetCommandActionBlend = 0f;
            velocityCommandIsaac = Vector3.zero;
        }

        public override void BuildObservation(float[] destination)
        {
            int cursor = 0;
            Vector3 linearBodyUnity = rootBody.transform.InverseTransformDirection(rootBody.linearVelocity);
            Vector3 angularBodyUnity = rootBody.transform.InverseTransformDirection(rootBody.angularVelocity);
            Vector3 gravityBodyUnity = rootBody.transform.InverseTransformDirection(Physics.gravity.normalized);
            Vector3 linearBody = DeploymentFrameConverter.UnityToIsaac(linearBodyUnity);
            Vector3 angularBody = DeploymentFrameConverter.UnityAngularToIsaac(angularBodyUnity);
            Vector3 projectedGravity = DeploymentFrameConverter.UnityToIsaac(gravityBodyUnity);

            foreach (PolicyTerm term in Contract.observations)
            {
                string name = term.name.ToLowerInvariant();
                if (name.Contains("base_lin_vel")) WriteTermVector(destination, ref cursor, term, linearBody);
                else if (name.Contains("base_ang_vel")) WriteTermVector(destination, ref cursor, term, angularBody);
                else if (name.Contains("projected_gravity")) WriteTermVector(destination, ref cursor, term, projectedGravity);
                else if (name.Contains("command")) WriteTermVector(destination, ref cursor, term, velocityCommandIsaac);
                else if (name.Contains("joint_pos")) WriteJointVector(destination, ref cursor, term, false, true);
                else if (name.Contains("joint_vel")) WriteJointVector(destination, ref cursor, term, true, false);
                else if (name.Contains("last_action"))
                    for (int i = 0; i < _lastActions.Length; i++)
                        WriteTermValue(destination, ref cursor, term, i, _lastActions[i]);
                else throw new System.InvalidOperationException($"Unsupported Spot observation '{term.name}'");
            }
            VerifyCursor(cursor, destination.Length);
        }

        public override void ApplyAction(float[] action, float policyDeltaTime)
        {
            if (action.Length != 12)
                throw new System.ArgumentException("Spot locomotion requires 12 actions");
            PolicyTerm term = Contract.actions[0];
            if (_unityActionBlend < 1f)
                _unityActionBlend = Mathf.Min(
                    1f,
                    _unityActionBlend + policyDeltaTime / Mathf.Max(0.001f, unityActionRampSeconds));
            _commandActionBlend = Mathf.MoveTowards(
                _commandActionBlend,
                _targetCommandActionBlend,
                policyDeltaTime / Mathf.Max(0.01f, unityMotionBlendSeconds));
            for (int i = 0; i < _unityActions.Length; i++)
                _unityActions[i] = action[i] * unityActionScale * _unityActionBlend *
                    _commandActionBlend;
            int cursor = 0;
            if (!robot.ApplyTerm(term, _unityActions, ref cursor, policyDeltaTime))
                throw new System.InvalidOperationException("Could not apply Spot locomotion joint action");
            System.Array.Copy(action, _lastActions, 12);
            VerifyCursor(cursor, action.Length);
        }

        public override StationTelemetrySnapshot CaptureTelemetry() => Capture(
            robot, rootBody, environmentAnchor);

        public override void BeginEvaluationScenario(int seed)
        {
            base.BeginEvaluationScenario(seed);
            _evaluationFell = false;
            _evaluationTrackingSum = 0f;
            _evaluationValidFootSamples = 0;
            _evaluationSamples = 0;
            _evaluationScore = 0f;
        }

        public override void SampleEvaluationScenario(float policyDeltaTime)
        {
            base.SampleEvaluationScenario(policyDeltaTime);
            if (rootBody == null)
            {
                _evaluationFell = true;
                return;
            }
            Vector3 actualLinear = DeploymentFrameConverter.UnityToIsaac(
                rootBody.transform.InverseTransformDirection(rootBody.linearVelocity));
            Vector3 actualAngular = DeploymentFrameConverter.UnityAngularToIsaac(
                rootBody.transform.InverseTransformDirection(rootBody.angularVelocity));
            float trackingError = Mathf.Abs(actualLinear.x - velocityCommandIsaac.x) +
                                  Mathf.Abs(actualLinear.y - velocityCommandIsaac.y) +
                                  Mathf.Abs(actualAngular.z - velocityCommandIsaac.z);
            _evaluationTrackingSum += 1f / (1f + trackingError);
            int contacts = 0;
            if (footContactSensors != null)
                foreach (ContactSensor foot in footContactSensors)
                    if (foot != null && foot.IsTouching)
                        contacts++;
            if (contacts >= 2)
                _evaluationValidFootSamples++;
            _evaluationSamples++;

            Vector3 up = environmentAnchor != null ? environmentAnchor.up : Vector3.up;
            float height = IsaacLocalPosition(environmentAnchor, rootBody.transform.position).z;
            if (Vector3.Angle(rootBody.transform.up, up) >= fallTiltDegrees ||
                height < minimumRootHeightIsaac)
                _evaluationFell = true;
        }

        public override StationEvaluationSummary CompleteEvaluationScenario()
        {
            EvaluationMetrics.ten_second_stand_rate =
                EvaluationElapsed >= 9.99f && !_evaluationFell ? 1f : 0f;
            EvaluationMetrics.command_tracking_score = _evaluationSamples > 0
                ? _evaluationTrackingSum / _evaluationSamples
                : 0f;
            EvaluationMetrics.foot_contact_valid_rate = _evaluationSamples > 0
                ? (float)_evaluationValidFootSamples / _evaluationSamples
                : 0f;
            EvaluationMetrics.fall_rate = _evaluationFell ? 1f : 0f;
            _evaluationScore = Mathf.Clamp01((
                EvaluationMetrics.ten_second_stand_rate +
                EvaluationMetrics.command_tracking_score +
                EvaluationMetrics.foot_contact_valid_rate) / 3f);
            return base.CompleteEvaluationScenario();
        }

        void WriteJointVector(float[] destination, ref int cursor, PolicyTerm term, bool velocity, bool relative)
        {
            for (int index = 0; index < jointNames.Length; index++)
            {
                string name = jointNames[index];
                if (!robot.TryRead(name, out float position, out float speed))
                {
                    Healthy = false;
                    throw new System.InvalidOperationException(
                        $"Could not read Spot joint '{name}'");
                }
                if (relative)
                    position -= robot.definition.FindJoint(name)?.defaultPosition ?? 0f;
                WriteTermValue(destination, ref cursor, term, index, velocity ? speed : position);
            }
        }
    }
}
