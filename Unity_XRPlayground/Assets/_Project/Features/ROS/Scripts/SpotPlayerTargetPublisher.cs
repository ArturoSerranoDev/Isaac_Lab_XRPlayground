using UnityEngine;
using UnityEngine.XR;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Publishes XR center-eye (or main camera) pose in Isaac env-local frame on /xr/spot/player_pose.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SpotPlayerTargetPublisher : MonoBehaviour
    {
        public RosTcpClient client;
        public Transform envAnchor;
        [Tooltip("Optional override; otherwise XR center-eye or Camera.main.")]
        public Transform playerTarget;
        public string topic = RosTopics.SpotPlayerPose;
        public float publishHz = 30f;
        public bool publishingEnabled = true;

        float _accum;

        void Awake()
        {
            if (envAnchor == null)
                envAnchor = transform;
        }

        void Update()
        {
            if (!publishingEnabled || client == null || !client.IsConnected)
                return;

            _accum += Time.deltaTime;
            float period = 1f / Mathf.Max(publishHz, 1f);
            if (_accum < period)
                return;
            _accum = 0f;

            if (!TryGetPlayerUnityWorld(out Vector3 worldPos, out Quaternion worldRot))
                return;

            XrFrameConverter.UnityWorldToIsaacLocal(
                envAnchor, worldPos, worldRot, out Vector3 isaacPos, out Quaternion isaacRot);

            client.PublishJson(RosJson.SerializePose(topic, isaacPos, isaacRot));
        }

        bool TryGetPlayerUnityWorld(out Vector3 pos, out Quaternion rot)
        {
            if (playerTarget != null)
            {
                pos = playerTarget.position;
                rot = playerTarget.rotation;
                return true;
            }

            var devices = new System.Collections.Generic.List<InputDevice>();
            InputDevices.GetDevicesAtXRNode(XRNode.CenterEye, devices);
            if (devices.Count > 0)
            {
                var d = devices[0];
                if (d.TryGetFeatureValue(CommonUsages.centerEyePosition, out Vector3 p) &&
                    d.TryGetFeatureValue(CommonUsages.centerEyeRotation, out Quaternion r))
                {
                    pos = p;
                    rot = r;
                    return true;
                }
            }

            if (Camera.main != null)
            {
                pos = Camera.main.transform.position;
                rot = Camera.main.transform.rotation;
                return true;
            }

            pos = Vector3.zero;
            rot = Quaternion.identity;
            return false;
        }
    }
}
