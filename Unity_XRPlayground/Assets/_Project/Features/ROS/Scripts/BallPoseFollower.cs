using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Applies Isaac-sourced /xr/ball_state onto the Unity ball (mirror + post-throw catch).
    /// Ignores Unity-sourced packets so await_throw ownership does not fight itself.
    /// </summary>
    public sealed class BallPoseFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public Transform ballRoot;
        public Rigidbody ballBody;

        [Tooltip("Unity transform for Isaac env origin (usually Kinova root).")]
        public Transform envAnchor;

        [Tooltip("Legacy position-only offset if envAnchor is null.")]
        public Vector3 rootOffset;

        [Tooltip("When true, drive the ball from Isaac packets.")]
        public bool followingEnabled = true;

        public bool applyVelocity = true;

        void OnEnable()
        {
            if (ballRoot == null)
                ballRoot = transform;
            if (ballBody == null)
                ballBody = GetComponent<Rigidbody>();
            if (client != null)
                client.MessageReceived += OnMessage;
        }

        void OnDisable()
        {
            if (client != null)
                client.MessageReceived -= OnMessage;
        }

        void OnMessage(string json)
        {
            if (!followingEnabled)
                return;
            if (!RosJson.TryParseTopic(json, out var topic) || topic != RosTopics.BallState)
                return;
            if (!RosJson.TryParseBallState(json, out var state) || state == null)
                return;
            // Only follow Isaac authority (ignore Unity echo / player publishes)
            if (!string.IsNullOrEmpty(state.source) && state.source != "isaac")
                return;
            Apply(state);
        }

        public void Apply(BallStateData state)
        {
            if (ballRoot == null || state == null)
                return;

            Vector3 pLocal = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(state.position));
            Quaternion qLocal = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(state.orientation_xyzw));

            Vector3 worldPos;
            Quaternion worldRot;
            if (envAnchor != null)
            {
                worldPos = envAnchor.TransformPoint(pLocal);
                worldRot = envAnchor.rotation * qLocal;
            }
            else
            {
                worldPos = pLocal + rootOffset;
                worldRot = qLocal;
            }

            if (ballBody != null)
            {
                ballBody.position = worldPos;
                ballBody.rotation = worldRot;
                if (applyVelocity)
                {
                    Vector3 lin = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(state.linear_velocity));
                    Vector3 ang = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(state.angular_velocity));
                    if (envAnchor != null)
                    {
                        lin = envAnchor.TransformVector(lin);
                        ang = envAnchor.TransformVector(ang);
                    }
                    ballBody.linearVelocity = lin;
                    ballBody.angularVelocity = ang;
                }
                else
                {
                    ballBody.linearVelocity = Vector3.zero;
                    ballBody.angularVelocity = Vector3.zero;
                }
            }
            else
            {
                ballRoot.SetPositionAndRotation(worldPos, worldRot);
            }
        }

        public void SetKinematic(bool kinematic)
        {
            if (ballBody == null)
                return;
            ballBody.isKinematic = kinematic;
            if (kinematic)
            {
                ballBody.linearVelocity = Vector3.zero;
                ballBody.angularVelocity = Vector3.zero;
            }
        }
    }
}
