using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class BallCatchStationAdapter : StationAdapterBase
    {
        [System.Serializable]
        sealed class LaunchCommand
        {
            public float[] position;
            public float[] velocity;
        }
        public ArticulationRobotDriver robot;
        public Transform environmentAnchor;
        public Transform endEffector;
        public Transform[] fingertips;
        public Rigidbody ball;
        public ContactGripAssist gripAssist;
        public Vector3 resetBallPositionIsaac = new(0.55f, 0f, 0.95f);
        [Range(0f, 1f)] public float ballGravityScale = 0.55f;
        public Vector2 throwFrontOffsetM = new(0.16f, 0.32f);
        public Vector2 throwSideOffsetM = new(-0.14f, 0.14f);
        public Vector2 throwBelowOffsetM = new(0.02f, 0.05f);
        public Vector2 throwHeightBoostM = new(0.06f, 0.14f);
        public Vector2 throwFlightTimeS = new(0.60f, 0.95f);
        public float throwAimJitterXYM = 0.035f;
        public float throwAimJitterZM = 0.028f;

        bool _evaluationContact;
        bool _evaluationHeld;
        float _evaluationScore;

        public override string AdapterId => "ball_catch";
        public override int ObservationDimension => 30;
        public override int ActionDimension => 8;
        public override float TaskScore => _evaluationScore;
        public override bool HasJointLimitViolation =>
            robot == null || robot.HasJointLimitViolation();

        public override void ResetStation(int seed)
        {
            Healthy = robot != null && robot.IsHealthy && ball != null && endEffector != null;
            robot?.ResetToDefaults();
            if (ball != null)
            {
                Vector3 unity = DeploymentFrameConverter.IsaacToUnity(resetBallPositionIsaac);
                ball.position = environmentAnchor != null ? environmentAnchor.TransformPoint(unity) : unity;
                ball.rotation = Quaternion.identity;
                ball.linearVelocity = Vector3.zero;
                ball.angularVelocity = Vector3.zero;
            }
            gripAssist?.Release(false);
        }

        public void LaunchBall(Vector3 positionIsaac, Vector3 velocityIsaac)
        {
            Vector3 p = DeploymentFrameConverter.IsaacToUnity(positionIsaac);
            Vector3 v = DeploymentFrameConverter.IsaacToUnity(velocityIsaac);
            ball.position = environmentAnchor != null ? environmentAnchor.TransformPoint(p) : p;
            ball.linearVelocity = environmentAnchor != null ? environmentAnchor.TransformDirection(v) : v;
            ball.angularVelocity = Vector3.zero;
        }

        void FixedUpdate()
        {
            if (ball != null && ball.useGravity && !Mathf.Approximately(ballGravityScale, 1f))
                ball.AddForce(Physics.gravity * (ballGravityScale - 1f), ForceMode.Acceleration);
        }

        public override bool TryHandleCommand(string command, string payloadJson, out string error)
        {
            if (command != "launch_ball")
                return base.TryHandleCommand(command, payloadJson, out error);
            LaunchCommand payload = JsonUtility.FromJson<LaunchCommand>(payloadJson ?? "{}");
            if (payload?.position == null || payload.position.Length != 3 ||
                payload.velocity == null || payload.velocity.Length != 3 || ball == null)
            {
                error = "launch_ball requires three-value position and velocity arrays";
                return false;
            }
            LaunchBall(DeploymentFrameConverter.Vector3From(payload.position),
                DeploymentFrameConverter.Vector3From(payload.velocity));
            error = null;
            return true;
        }

        public override void BuildObservation(float[] destination)
        {
            int cursor = 0;
            Vector3 ballPosition = IsaacLocalPosition(environmentAnchor, ball.position);
            Vector3 ballVelocity = IsaacDirection(environmentAnchor, ball.linearVelocity);
            Vector3 eePosition = IsaacLocalPosition(environmentAnchor, endEffector.position);
            Vector3 tipPosition = IsaacLocalPosition(environmentAnchor, MeanPosition(fingertips));
            Vector3 toBall = ballPosition - eePosition;
            Vector3 tipToBall = ballPosition - tipPosition;
            Vector3 axis = (tipPosition - eePosition).normalized;
            float distance = Mathf.Max(1e-6f, toBall.magnitude);
            float alignment = Vector3.Dot(toBall / distance, axis);
            float along = Vector3.Dot(toBall, axis);
            float radial = (toBall - along * axis).magnitude;

            foreach (PolicyTerm term in Contract.observations)
            {
                switch (term.name)
                {
                    case "arm_joint_position": WriteJoints(destination, ref cursor, term, false); break;
                    case "arm_joint_velocity": WriteJoints(destination, ref cursor, term, true); break;
                    case "gripper_position_mean": WriteJointMean(destination, ref cursor, term, false); break;
                    case "gripper_velocity_mean": WriteJointMean(destination, ref cursor, term, true); break;
                    case "ball_position": WriteTermVector(destination, ref cursor, term, ballPosition); break;
                    case "ball_linear_velocity": WriteTermVector(destination, ref cursor, term, ballVelocity); break;
                    case "end_effector_to_ball": WriteTermVector(destination, ref cursor, term, toBall); break;
                    case "fingertip_center_to_ball": WriteTermVector(destination, ref cursor, term, tipToBall); break;
                    case "grasp_alignment": WriteTermValue(destination, ref cursor, term, 0, alignment); break;
                    case "ball_radial_error": WriteTermValue(destination, ref cursor, term, 0, radial); break;
                    default: throw new System.InvalidOperationException($"Unsupported Ball observation '{term.name}'");
                }
            }
            VerifyCursor(cursor, destination.Length);
        }

        public override void ApplyAction(float[] action, float policyDeltaTime)
        {
            int cursor = 0;
            foreach (PolicyTerm term in Contract.actions)
                if (!robot.ApplyTerm(term, action, ref cursor, policyDeltaTime))
                    throw new System.InvalidOperationException($"Could not apply action term '{term.name}'");
            gripAssist?.SetClosingCommand(action.Length > 0 && action[^1] > 0f);
            VerifyCursor(cursor, action.Length);
        }

        public override StationTelemetrySnapshot CaptureTelemetry() => Capture(
            robot,
            robot != null ? robot.articulationRoot : null,
            environmentAnchor,
            new ObjectTelemetryBinding("ball", ball));

        public override void BeginEvaluationScenario(int seed)
        {
            base.BeginEvaluationScenario(seed);
            _evaluationContact = false;
            _evaluationHeld = false;
            _evaluationScore = 0f;
            gripAssist?.ResetTelemetryCounters();
            LaunchSeededThrow(seed);
        }

        public override void SampleEvaluationScenario(float policyDeltaTime)
        {
            base.SampleEvaluationScenario(policyDeltaTime);
            bool contact = gripAssist != null && gripAssist.HasRequiredContact(ball);
            bool held = gripAssist != null && gripAssist.ConstraintActive &&
                        gripAssist.HeldBody == ball;
            _evaluationContact |= contact;
            _evaluationHeld |= held;
        }

        public override StationEvaluationSummary CompleteEvaluationScenario()
        {
            bool retained = gripAssist != null && gripAssist.ConstraintActive &&
                            gripAssist.HeldBody == ball;
            EvaluationMetrics.catch_rate = _evaluationContact && _evaluationHeld ? 1f : 0f;
            EvaluationMetrics.retained_grasp_rate = retained ? 1f : 0f;
            EvaluationMetrics.pre_contact_assist_events =
                gripAssist != null ? gripAssist.PreContactActivationCount : 0f;
            _evaluationScore = 0.5f * (
                EvaluationMetrics.catch_rate + EvaluationMetrics.retained_grasp_rate);
            return base.CompleteEvaluationScenario();
        }

        void LaunchSeededThrow(int seed)
        {
            if (ball == null || endEffector == null)
                return;
            var random = new System.Random(seed);
            Vector3 tip = IsaacLocalPosition(environmentAnchor, MeanPosition(fingertips));
            Vector3 ee = IsaacLocalPosition(environmentAnchor, endEffector.position);
            Vector3 cup = 0.55f * tip + 0.45f * ee;
            Vector3 target = cup + new Vector3(
                Sample(random, -throwAimJitterXYM, throwAimJitterXYM),
                Sample(random, -throwAimJitterXYM, throwAimJitterXYM),
                Sample(random, -throwAimJitterZM, throwAimJitterZM));
            Vector3 release = cup + new Vector3(
                Sample(random, throwFrontOffsetM.x, throwFrontOffsetM.y),
                Sample(random, throwSideOffsetM.x, throwSideOffsetM.y),
                -Sample(random, throwBelowOffsetM.x, throwBelowOffsetM.y));
            float flight = Sample(random, throwFlightTimeS.x, throwFlightTimeS.y);
            float midHeight = 0.5f * (release.z + target.z) +
                              Sample(random, throwHeightBoostM.x, throwHeightBoostM.y);
            Vector3 displacement = target - release;
            float linearMidHeight = release.z + 0.5f * displacement.z;
            float loft = 2f * (midHeight - linearMidHeight) / flight;
            Vector3 gravity = new(0f, 0f, -Physics.gravity.magnitude * ballGravityScale);
            Vector3 velocity = displacement / flight - 0.5f * gravity * flight;
            velocity.z += loft;
            LaunchBall(release, velocity);
        }

        static float Sample(System.Random random, float minimum, float maximum) =>
            minimum + (maximum - minimum) * (float)random.NextDouble();

        void WriteJoints(float[] destination, ref int cursor, PolicyTerm term, bool velocity)
        {
            for (int index = 0; index < term.names.Length; index++)
            {
                string name = term.names[index];
                if (!robot.TryRead(name, out float position, out float speed))
                {
                    Healthy = false;
                    throw new System.InvalidOperationException(
                        $"Could not read Ball joint '{name}'");
                }
                WriteTermValue(destination, ref cursor, term, index, velocity ? speed : position);
            }
        }

        void WriteJointMean(float[] destination, ref int cursor, PolicyTerm term, bool velocity)
        {
            float sum = 0f;
            int count = 0;
            foreach (string name in term.names)
            {
                if (robot.TryRead(name, out float position, out float speed))
                {
                    sum += velocity ? speed : position;
                    count++;
                }
                else
                {
                    Healthy = false;
                    throw new System.InvalidOperationException(
                        $"Could not read Ball joint '{name}'");
                }
            }
            WriteTermValue(destination, ref cursor, term, 0, count > 0 ? sum / count : 0f);
        }
    }
}
