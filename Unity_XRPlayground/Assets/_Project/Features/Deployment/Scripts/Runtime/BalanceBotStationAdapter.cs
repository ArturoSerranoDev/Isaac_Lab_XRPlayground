using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class BalanceBotStationAdapter : StationAdapterBase
    {
        public Transform environmentAnchor;
        public Transform tray;
        public Rigidbody trayBody;
        public Rigidbody[] balls = new Rigidbody[2];
        public bool[] ballActive = { true, true };
        public Vector3 trayCenterIsaac = new(0f, 0f, 0.75f);
        public float trayThicknessM = 0.02f;
        public float ballRadiusM = 0.035f;
        public float spawnHeightAboveTrayM = 0.08f;
        public float spawnXYHalfM = 0.12f;
        [Range(0, 3)] public int curriculumStage = 3;
        [Range(0f, 1f)] public float normalizedScore;
        public float maximumTiltRad = 0.35f;
        public float trayHalfExtentM = 0.27f;
        public float dropHeightBelowTrayM = 0.4f;

        float _roll;
        float _pitch;
        float _rollVelocity;
        float _pitchVelocity;
        bool _evaluationDropped;

        public override string AdapterId => "balance_bot";
        public override int ObservationDimension => 20;
        public override int ActionDimension => 2;
        public override float TaskScore => normalizedScore;
        public override bool HasJointLimitViolation =>
            Mathf.Abs(_roll) > maximumTiltRad + 1e-3f ||
            Mathf.Abs(_pitch) > maximumTiltRad + 1e-3f;

        public override void ResetStation(int seed)
        {
            _roll = _pitch = _rollVelocity = _pitchVelocity = 0f;
            Healthy = tray != null && balls != null && balls.Length == 2;
            ApplyTrayRotation();
            var random = new System.Random(seed);
            for (int i = 0; i < balls.Length; i++)
            {
                if (balls[i] == null)
                    continue;
                Vector2 offset = i == 0 ? Vector2.zero : new Vector2(0.09f, 0.06f);
                float jitter = spawnXYHalfM * 0.35f;
                Vector3 pIsaac = trayCenterIsaac + new Vector3(
                    offset.x + (float)(2.0 * random.NextDouble() - 1.0) * jitter,
                    offset.y + (float)(2.0 * random.NextDouble() - 1.0) * jitter,
                    0.5f * trayThicknessM + ballRadiusM + spawnHeightAboveTrayM);
                Vector3 p = DeploymentFrameConverter.IsaacToUnity(pIsaac);
                balls[i].position = environmentAnchor != null ? environmentAnchor.TransformPoint(p) : p;
                balls[i].linearVelocity = Vector3.zero;
                balls[i].angularVelocity = Vector3.zero;
                balls[i].gameObject.SetActive(ballActive[i]);
            }
        }

        public override void BuildObservation(float[] destination)
        {
            int cursor = 0;
            WriteTermValue(destination, ref cursor, Contract.observations[0], 0, _roll);
            WriteTermValue(destination, ref cursor, Contract.observations[1], 0, _pitch);
            WriteTermValue(destination, ref cursor, Contract.observations[2], 0, _rollVelocity);
            WriteTermValue(destination, ref cursor, Contract.observations[3], 0, _pitchVelocity);
            for (int i = 0; i < 2; i++)
            {
                bool active = ballActive[i] && balls[i] != null && balls[i].gameObject.activeInHierarchy;
                Vector3 p = active ? IsaacLocalPosition(environmentAnchor, balls[i].position) - trayCenterIsaac : Vector3.zero;
                Vector3 v = active ? IsaacDirection(environmentAnchor, balls[i].linearVelocity) : Vector3.zero;
                Write(destination, ref cursor, p);
                Write(destination, ref cursor, v);
                Write(destination, ref cursor, active ? 1f : 0f);
            }
            Write(destination, ref cursor, (ballActive[0] ? 1f : 0f) + (ballActive[1] ? 1f : 0f));
            Write(destination, ref cursor, curriculumStage);
            VerifyCursor(cursor, destination.Length);
        }

        public override void ApplyAction(float[] action, float policyDeltaTime)
        {
            PolicyTerm term = Contract.actions[0];
            _rollVelocity = action[0] * term.ScaleAt(0) + term.OffsetAt(0);
            _pitchVelocity = action[1] * term.ScaleAt(1) + term.OffsetAt(1);
            _roll = Mathf.Clamp(_roll + _rollVelocity * policyDeltaTime, -maximumTiltRad, maximumTiltRad);
            _pitch = Mathf.Clamp(_pitch + _pitchVelocity * policyDeltaTime, -maximumTiltRad, maximumTiltRad);
            ApplyTrayRotation();
        }

        public override StationTelemetrySnapshot CaptureTelemetry()
        {
            var objects = new ObjectTelemetryBinding[balls?.Length ?? 0];
            for (int i = 0; i < objects.Length; i++)
                objects[i] = new ObjectTelemetryBinding(
                    $"ball_{i}", balls[i], ballActive != null && i < ballActive.Length && ballActive[i]);
            return new StationTelemetrySnapshot
            {
                joint_state = new JointStateTelemetry
                {
                    names = new[] { "roll_joint", "pitch_joint" },
                    position_units = new[] { "radian", "radian" },
                    velocity_units = new[] { "radian_per_second", "radian_per_second" },
                    position = new[] { _roll, _pitch },
                    velocity = new[] { _rollVelocity, _pitchVelocity },
                },
                root_state = tray != null
                    ? DeploymentTelemetryCapture.Body(
                        tray,
                        trayBody != null ? trayBody.linearVelocity : Vector3.zero,
                        trayBody != null ? trayBody.angularVelocity :
                            new Vector3(_rollVelocity, 0f, _pitchVelocity),
                        environmentAnchor)
                    : new BodyStateTelemetry(),
                object_state = DeploymentTelemetryCapture.Objects(objects, environmentAnchor),
            };
        }

        public override void BeginEvaluationScenario(int seed)
        {
            base.BeginEvaluationScenario(seed);
            normalizedScore = 0f;
            _evaluationDropped = false;
        }

        public override void SampleEvaluationScenario(float policyDeltaTime)
        {
            base.SampleEvaluationScenario(policyDeltaTime);
            for (int i = 0; i < balls.Length; i++)
            {
                bool active = ballActive != null && i < ballActive.Length && ballActive[i] &&
                              balls[i] != null && balls[i].gameObject.activeInHierarchy;
                if (!active)
                {
                    _evaluationDropped = true;
                    continue;
                }
                Vector3 position = IsaacLocalPosition(environmentAnchor, balls[i].position);
                if (position.z < trayCenterIsaac.z - dropHeightBelowTrayM ||
                    Mathf.Abs(position.x - trayCenterIsaac.x) > trayHalfExtentM ||
                    Mathf.Abs(position.y - trayCenterIsaac.y) > trayHalfExtentM)
                    _evaluationDropped = true;
            }
        }

        public override StationEvaluationSummary CompleteEvaluationScenario()
        {
            EvaluationMetrics.full_episode_hold_rate = _evaluationDropped ? 0f : 1f;
            EvaluationMetrics.drop_rate = _evaluationDropped ? 1f : 0f;
            normalizedScore = EvaluationMetrics.full_episode_hold_rate;
            return base.CompleteEvaluationScenario();
        }

        void ApplyTrayRotation()
        {
            if (tray == null)
                return;
            Quaternion local = Quaternion.AngleAxis(_roll * Mathf.Rad2Deg, Vector3.right) *
                               Quaternion.AngleAxis(_pitch * Mathf.Rad2Deg, Vector3.forward);
            Quaternion world =
                (environmentAnchor != null ? environmentAnchor.rotation : Quaternion.identity) * local;
            if (trayBody != null && trayBody.isKinematic && Application.isPlaying)
                trayBody.MoveRotation(world);
            else
                tray.rotation = world;
        }
    }
}
