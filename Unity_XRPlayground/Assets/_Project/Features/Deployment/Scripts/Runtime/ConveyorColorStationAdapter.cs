using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class ConveyorColorStationAdapter : StationAdapterBase
    {
        public ArticulationRobotDriver robot;
        public Transform environmentAnchor;
        public Transform endEffector;
        public Transform targetBin;
        public Transform rejectBin;
        public PhysicalObjectSlot[] objectSlots = new PhysicalObjectSlot[4];
        [Range(0, 2)] public int targetColor;
        public ContactGripAssist gripAssist;
        [Range(0f, 1f)] public float normalizedScore;
        public float binAcceptanceRadiusM = 0.2f;
        public float rejectAcceptanceRadiusM = 0.16f;
        public Vector2 acceptanceHeightIsaac = new(0.38f, 0.48f);
        public bool autoSpawnEnabled = true;
        public float spawnIntervalSeconds = 2.5f;

        System.Random _spawnRandom;
        float _spawnTimer;
        bool _simulationActive;

        public override string AdapterId => "conveyor_color";
        public override int ObservationDimension => 66;
        public override int ActionDimension => 7;
        public override float TaskScore => normalizedScore;
        public override bool HasJointLimitViolation =>
            robot == null || robot.HasJointLimitViolation();

        public override void ResetStation(int seed)
        {
            Healthy = robot != null && robot.IsHealthy && endEffector != null && targetBin != null &&
                      rejectBin != null &&
                      objectSlots?.Length == 4;
            robot?.ResetToDefaults();
            gripAssist?.Release(false);
            _spawnRandom = new System.Random(seed);
            _spawnTimer = 0f;
            targetColor = _spawnRandom.Next(0, 3);
            foreach (PhysicalObjectSlot slot in objectSlots)
            {
                if (slot == null)
                    continue;
                slot.active = false;
                slot.grasped = false;
                slot.Reset(environmentAnchor, _spawnRandom);
            }
            if (autoSpawnEnabled)
                ActivateSlot(0, _spawnRandom.NextDouble() < 0.5
                    ? targetColor
                    : _spawnRandom.Next(0, 3));
            _simulationActive = Healthy;
        }

        void OnDisable() => _simulationActive = false;

        void FixedUpdate()
        {
            if (!_simulationActive || objectSlots == null)
                return;
            Rigidbody held = gripAssist != null && gripAssist.ConstraintActive
                ? gripAssist.HeldBody
                : null;
            foreach (PhysicalObjectSlot slot in objectSlots)
                if (slot != null)
                    slot.grasped = slot.body != null && slot.body == held;
            if (!autoSpawnEnabled)
                return;
            _spawnTimer += Time.fixedDeltaTime;
            if (_spawnTimer < spawnIntervalSeconds)
                return;
            _spawnTimer = 0f;
            for (int index = 0; index < objectSlots.Length; index++)
                if (objectSlots[index] != null && !objectSlots[index].active)
                {
                    ActivateSlot(index, _spawnRandom.Next(0, 3));
                    break;
                }
        }

        void ActivateSlot(int index, int color)
        {
            if (index < 0 || index >= objectSlots.Length || objectSlots[index] == null)
                return;
            PhysicalObjectSlot slot = objectSlots[index];
            slot.color = Mathf.Clamp(color, 0, 2);
            slot.active = true;
            slot.grasped = false;
            slot.Reset(environmentAnchor, _spawnRandom);
        }

        public override void BuildObservation(float[] destination)
        {
            int cursor = 0;
            Vector3 ee = IsaacLocalPosition(environmentAnchor, endEffector.position);
            Vector3 nearest = Vector3.zero;
            float best = float.PositiveInfinity;
            for (int i = 0; i < objectSlots.Length; i++)
            {
                PhysicalObjectSlot slot = objectSlots[i];
                if (slot == null || !slot.active || slot.color != targetColor || slot.body == null)
                    continue;
                Vector3 delta = IsaacLocalPosition(environmentAnchor, slot.body.position) - ee;
                if (delta.sqrMagnitude < best)
                {
                    best = delta.sqrMagnitude;
                    nearest = delta;
                }
            }
            foreach (PolicyTerm term in Contract.observations)
            {
                if (term.name == "arm_joint_position") WriteJoints(destination, ref cursor, term, false);
                else if (term.name == "arm_joint_velocity") WriteJoints(destination, ref cursor, term, true);
                else if (term.name == "gripper_position") WriteJoints(destination, ref cursor, term, false);
                else if (term.name == "gripper_velocity") WriteJoints(destination, ref cursor, term, true);
                else if (term.name == "target_color_one_hot") WriteOneHot(destination, ref cursor, targetColor, 3, true);
                else if (term.name == "end_effector_position") Write(destination, ref cursor, ee);
                else if (term.name == "nearest_target_delta") Write(destination, ref cursor, nearest);
                else if (term.name == "bin_delta") Write(destination, ref cursor, IsaacLocalPosition(environmentAnchor, targetBin.position) - ee);
                else if (term.name.StartsWith("object_")) WriteObjectTerm(destination, ref cursor, term);
                else throw new System.InvalidOperationException($"Unsupported Conveyor observation '{term.name}'");
            }
            VerifyCursor(cursor, destination.Length);
        }

        public override void ApplyAction(float[] action, float policyDeltaTime)
        {
            int cursor = 0;
            foreach (PolicyTerm term in Contract.actions)
                if (!robot.ApplyTerm(term, action, ref cursor, policyDeltaTime))
                    throw new System.InvalidOperationException($"Could not apply Conveyor action '{term.name}'");
            gripAssist?.SetClosingCommand(action[^1] > 0f);
            VerifyCursor(cursor, action.Length);
        }

        public override StationTelemetrySnapshot CaptureTelemetry() => Capture(
            robot,
            robot != null ? robot.articulationRoot : null,
            environmentAnchor,
            Bindings(objectSlots));

        public override void BeginEvaluationScenario(int seed)
        {
            base.BeginEvaluationScenario(seed);
            normalizedScore = 0f;
            gripAssist?.ResetTelemetryCounters();
        }

        public override StationEvaluationSummary CompleteEvaluationScenario()
        {
            int targets = 0;
            int correct = 0;
            int rejects = 0;
            int rejected = 0;
            int wrong = 0;
            Vector3 bin = IsaacLocalPosition(environmentAnchor, targetBin.position);
            Vector3 reject = IsaacLocalPosition(environmentAnchor, rejectBin.position);
            foreach (PhysicalObjectSlot slot in objectSlots)
            {
                if (slot == null || !slot.active || slot.body == null)
                    continue;
                Vector3 position = IsaacLocalPosition(environmentAnchor, slot.body.position);
                bool inside = InZone(position, bin, binAcceptanceRadiusM);
                bool insideReject = InZone(position, reject, rejectAcceptanceRadiusM);
                if (slot.color == targetColor)
                {
                    targets++;
                    if (inside)
                        correct++;
                }
                else
                {
                    rejects++;
                    if (inside)
                        wrong++;
                    else if (insideReject)
                        rejected++;
                }
            }
            EvaluationMetrics.correct_sort_rate = targets > 0 ? (float)correct / targets : 0f;
            EvaluationMetrics.wrong_bin_rate = rejects > 0 ? (float)wrong / rejects : 0f;
            EvaluationMetrics.reject_rate = rejects > 0 ? (float)rejected / rejects : 0f;
            normalizedScore = Mathf.Clamp01(
                0.7f * EvaluationMetrics.correct_sort_rate +
                0.3f * EvaluationMetrics.reject_rate -
                0.5f * EvaluationMetrics.wrong_bin_rate);
            return base.CompleteEvaluationScenario();
        }

        bool InZone(Vector3 position, Vector3 center, float radius)
        {
            Vector2 delta = new(position.x - center.x, position.y - center.y);
            return delta.magnitude <= radius &&
                   position.z >= acceptanceHeightIsaac.x &&
                   position.z < acceptanceHeightIsaac.y;
        }

        void WriteObjectTerm(float[] destination, ref int cursor, PolicyTerm term)
        {
            string[] parts = term.name.Split('_');
            int index = int.Parse(parts[1]);
            PhysicalObjectSlot slot = objectSlots[index];
            bool active = slot != null && slot.active && slot.body != null && slot.body.gameObject.activeInHierarchy;
            if (term.name.EndsWith("_position"))
                Write(destination, ref cursor, active ? IsaacLocalPosition(environmentAnchor, slot.body.position) : Vector3.zero);
            else if (term.name.EndsWith("_linear_velocity"))
                Write(destination, ref cursor, active ? IsaacDirection(environmentAnchor, slot.body.linearVelocity) : Vector3.zero);
            else if (term.name.EndsWith("_color_one_hot"))
                WriteOneHot(destination, ref cursor, active ? slot.color : -1, 3, active);
            else if (term.name.EndsWith("_active"))
                Write(destination, ref cursor, active ? 1f : 0f);
        }

        void WriteJoints(float[] destination, ref int cursor, PolicyTerm term, bool velocity)
        {
            for (int index = 0; index < term.names.Length; index++)
            {
                string name = term.names[index];
                if (!robot.TryRead(name, out float position, out float speed))
                {
                    Healthy = false;
                    throw new System.InvalidOperationException(
                        $"Could not read Conveyor joint '{name}'");
                }
                WriteTermValue(destination, ref cursor, term, index, velocity ? speed : position);
            }
        }

        static void WriteOneHot(float[] destination, ref int cursor, int index, int count, bool active)
        {
            for (int i = 0; i < count; i++)
                Write(destination, ref cursor, active && i == index ? 1f : 0f);
        }
    }
}
