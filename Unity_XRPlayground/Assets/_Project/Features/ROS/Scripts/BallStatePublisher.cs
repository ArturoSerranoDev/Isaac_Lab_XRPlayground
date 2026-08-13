using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Interactables;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Publishes /xr/ball_state to Isaac. Used in await_throw while the player holds/throws.
    /// </summary>
    public sealed class BallStatePublisher : MonoBehaviour
    {
        public RosTcpClient client;
        public Transform ballRoot;
        public Rigidbody ballBody;
        public bool publishWhileHeld = true;
        public float publishHz = 30f;

        [Tooltip("Unity transform for Isaac env origin (usually Kinova root).")]
        public Transform envAnchor;

        [Tooltip("Legacy position-only offset if envAnchor is null.")]
        public Vector3 isaacRootOffset;

        [Tooltip("When false, no ball messages are sent (mirror mode / Isaac-owned phases).")]
        public bool publishingEnabled = true;

        XRGrabInteractable _grab;
        float _nextPublish;
        bool _held;

        public bool IsHeld => _held;

        void Awake()
        {
            if (ballRoot == null)
                ballRoot = transform;
            if (ballBody == null)
                ballBody = GetComponent<Rigidbody>();
            _grab = GetComponent<XRGrabInteractable>();
            if (_grab != null)
            {
                _grab.selectEntered.AddListener(_ => _held = true);
                _grab.selectExited.AddListener(_ =>
                {
                    _held = false;
                    Publish(throwEvent: true);
                });
            }
        }

        void OnDestroy()
        {
            if (_grab == null)
                return;
            _grab.selectEntered.RemoveAllListeners();
            _grab.selectExited.RemoveAllListeners();
        }

        void Update()
        {
            if (!publishingEnabled || client == null || !client.IsConnected)
                return;
            if (!publishWhileHeld && _held)
                return;
            if (Time.time < _nextPublish)
                return;
            _nextPublish = Time.time + 1f / Mathf.Max(1f, publishHz);
            Publish(throwEvent: false);
        }

        public void Publish(bool throwEvent)
        {
            if (!publishingEnabled || client == null || ballRoot == null)
                return;

            Vector3 unityWorld = ballRoot.position;
            Quaternion unityRot = ballRoot.rotation;
            Vector3 unityLin = ballBody != null ? ballBody.linearVelocity : Vector3.zero;
            Vector3 unityAng = ballBody != null ? ballBody.angularVelocity : Vector3.zero;

            Vector3 unityLocal;
            Quaternion unityLocalRot;
            Vector3 unityLinLocal = unityLin;
            Vector3 unityAngLocal = unityAng;
            if (envAnchor != null)
            {
                unityLocal = envAnchor.InverseTransformPoint(unityWorld);
                unityLocalRot = Quaternion.Inverse(envAnchor.rotation) * unityRot;
                unityLinLocal = envAnchor.InverseTransformVector(unityLin);
                unityAngLocal = envAnchor.InverseTransformVector(unityAng);
            }
            else
            {
                unityLocal = unityWorld - isaacRootOffset;
                unityLocalRot = unityRot;
            }

            Vector3 isaacPos = XrFrameConverter.UnityPosToIsaac(unityLocal);
            Quaternion isaacRot = XrFrameConverter.UnityQuatToIsaac(unityLocalRot);
            Vector3 isaacLin = XrFrameConverter.UnityPosToIsaac(unityLinLocal);
            Vector3 isaacAng = XrFrameConverter.UnityPosToIsaac(unityAngLocal);

            var data = new BallStateData
            {
                position = XrFrameConverter.ToArray(isaacPos),
                orientation_xyzw = XrFrameConverter.ToXyzw(isaacRot),
                linear_velocity = XrFrameConverter.ToArray(isaacLin),
                angular_velocity = XrFrameConverter.ToArray(isaacAng),
                grasped = _held,
                throw_event = throwEvent,
                source = "unity"
            };
            client.PublishJson(RosJson.SerializeBall(RosTopics.BallState, data));
        }
    }
}
