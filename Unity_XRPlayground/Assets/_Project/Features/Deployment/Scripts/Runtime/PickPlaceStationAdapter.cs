using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class PickPlaceStationAdapter : StationAdapterBase
    {
        public ArticulationRobotDriver robot;
        public Transform environmentAnchor;
        public Transform endEffector;
        public Transform bucketTarget;
        public PhysicalObjectSlot[] pieces = new PhysicalObjectSlot[4];
        public ContactGripAssist gripAssist;
        public float liftHeightIsaac = 0.52f;
        [Range(0f, 1f)] public float normalizedScore;
        public float placementRadiusM = 0.11f;
        public float stableSpeedMps = 0.1f;

        Rigidbody _evaluationHeldPiece;
        Rigidbody _evaluationReleasedPiece;
        bool _evaluationGrasped;
        bool _evaluationLifted;
        bool _evaluationReleased;
        bool _evaluationStable;

        public override string AdapterId => "pick_place_table";
        public override int ObservationDimension => 30;
        public override int ActionDimension => 8;
        public override float TaskScore => normalizedScore;
        public override bool HasJointLimitViolation =>
            robot == null || robot.HasJointLimitViolation();

        public override void ResetStation(int seed)
        {
            Healthy = robot != null && robot.IsHealthy && endEffector != null && bucketTarget != null &&
                      pieces?.Length == 4;
            robot?.ResetToDefaults();
            gripAssist?.Release(false);
            var random = new System.Random(seed);
            int activeIndex = pieces != null && pieces.Length > 0 ? random.Next(pieces.Length) : -1;
            for (int index = 0; index < pieces.Length; index++)
            {
                PhysicalObjectSlot piece = pieces[index];
                if (piece == null)
                    continue;
                piece.active = index == activeIndex;
                piece.grasped = false;
                piece.Reset(environmentAnchor, random);
            }
        }

        public override void BuildObservation(float[] destination)
        {
            int cursor = 0;
            Vector3 ee = IsaacLocalPosition(environmentAnchor, endEffector.position);
            PhysicalObjectSlot piece = NearestPiece(ee);
            bool active = piece?.body != null;
            Vector3 position = active ? IsaacLocalPosition(environmentAnchor, piece.body.position) : Vector3.zero;
            Vector3 velocity = active ? IsaacDirection(environmentAnchor, piece.body.linearVelocity) : Vector3.zero;
            foreach (PolicyTerm term in Contract.observations)
            {
                switch (term.name)
                {
                    case "arm_joint_position": WriteJoints(destination, ref cursor, term, false); break;
                    case "arm_joint_velocity": WriteJoints(destination, ref cursor, term, true); break;
                    case "gripper_position": WriteJoints(destination, ref cursor, term, false); break;
                    case "gripper_velocity": WriteJoints(destination, ref cursor, term, true); break;
                    case "piece_position": Write(destination, ref cursor, position); break;
                    case "piece_linear_velocity": Write(destination, ref cursor, velocity); break;
                    case "end_effector_to_piece": Write(destination, ref cursor, position - ee); break;
                    case "piece_to_bucket": Write(destination, ref cursor, IsaacLocalPosition(environmentAnchor, bucketTarget.position) - position); break;
                    case "piece_grasped": Write(destination, ref cursor, active && piece.grasped ? 1f : 0f); break;
                    case "piece_lifted": Write(destination, ref cursor, active && position.z > liftHeightIsaac ? 1f : 0f); break;
                    default: throw new System.InvalidOperationException($"Unsupported Pick observation '{term.name}'");
                }
            }
            VerifyCursor(cursor, destination.Length);
        }

        public override void ApplyAction(float[] action, float policyDeltaTime)
        {
            int cursor = 0;
            foreach (PolicyTerm term in Contract.actions)
                if (!robot.ApplyTerm(term, action, ref cursor, policyDeltaTime))
                    throw new System.InvalidOperationException($"Could not apply Pick action '{term.name}'");
            gripAssist?.SetClosingCommand(action[^1] < 0f);
            VerifyCursor(cursor, action.Length);
        }

        public override StationTelemetrySnapshot CaptureTelemetry() => Capture(
            robot,
            robot != null ? robot.articulationRoot : null,
            environmentAnchor,
            Bindings(pieces));

        public override void BeginEvaluationScenario(int seed)
        {
            base.BeginEvaluationScenario(seed);
            normalizedScore = 0f;
            _evaluationHeldPiece = null;
            _evaluationReleasedPiece = null;
            _evaluationGrasped = false;
            _evaluationLifted = false;
            _evaluationReleased = false;
            _evaluationStable = false;
            gripAssist?.ResetTelemetryCounters();
        }

        public override void SampleEvaluationScenario(float policyDeltaTime)
        {
            base.SampleEvaluationScenario(policyDeltaTime);
            Rigidbody current = gripAssist != null && gripAssist.ConstraintActive
                ? gripAssist.HeldBody
                : null;
            if (current != null && IsPiece(current))
            {
                _evaluationGrasped = true;
                _evaluationHeldPiece = current;
                float height = IsaacLocalPosition(environmentAnchor, current.position).z;
                if (height > liftHeightIsaac)
                    _evaluationLifted = true;
            }
            else if (_evaluationHeldPiece != null)
            {
                _evaluationReleased = true;
                _evaluationReleasedPiece = _evaluationHeldPiece;
                _evaluationHeldPiece = null;
            }

            if (_evaluationReleasedPiece != null && bucketTarget != null)
            {
                Vector3 piece = IsaacLocalPosition(environmentAnchor, _evaluationReleasedPiece.position);
                Vector3 targetPosition = IsaacLocalPosition(environmentAnchor, bucketTarget.position);
                _evaluationStable |= (piece - targetPosition).magnitude <= placementRadiusM &&
                                     _evaluationReleasedPiece.linearVelocity.magnitude <= stableSpeedMps;
            }
        }

        public override StationEvaluationSummary CompleteEvaluationScenario()
        {
            EvaluationMetrics.contact_grasp_rate = _evaluationGrasped ? 1f : 0f;
            EvaluationMetrics.lift_rate = _evaluationLifted ? 1f : 0f;
            EvaluationMetrics.release_rate = _evaluationReleased ? 1f : 0f;
            EvaluationMetrics.stable_placement_rate = _evaluationStable ? 1f : 0f;
            normalizedScore = 0.25f * (
                EvaluationMetrics.contact_grasp_rate + EvaluationMetrics.lift_rate +
                EvaluationMetrics.release_rate + EvaluationMetrics.stable_placement_rate);
            return base.CompleteEvaluationScenario();
        }

        bool IsPiece(Rigidbody body)
        {
            foreach (PhysicalObjectSlot piece in pieces)
                if (piece?.body == body)
                    return true;
            return false;
        }

        PhysicalObjectSlot NearestPiece(Vector3 ee)
        {
            PhysicalObjectSlot nearest = null;
            float distance = float.PositiveInfinity;
            foreach (PhysicalObjectSlot piece in pieces)
            {
                if (piece == null || !piece.active || piece.body == null || !piece.body.gameObject.activeInHierarchy)
                    continue;
                float value = (IsaacLocalPosition(environmentAnchor, piece.body.position) - ee).sqrMagnitude;
                if (value < distance)
                {
                    distance = value;
                    nearest = piece;
                }
            }
            return nearest;
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
                        $"Could not read Pick joint '{name}'");
                }
                WriteTermValue(destination, ref cursor, term, index, velocity ? speed : position);
            }
        }
    }
}
