using System.Collections.Generic;
using UnityEngine;
using XRPlayground.Robots;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Applies /xr/pick_place/robot_state link poses onto Agibot A2D USD hierarchy.
    /// </summary>
    public sealed class AgibotLinkPoseFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public AgibotLinkMap linkMap;
        public Transform envAnchor;
        public string robotStateTopic = RosTopics.PickPlaceRobotState;
        public bool followingEnabled = true;
        public bool logFirstApply = true;
        public bool calibrateVisualFrames = false;

        readonly Dictionary<string, Quaternion> _visualCorrection = new Dictionary<string, Quaternion>();
        readonly Dictionary<string, Quaternion> _unityBindLocalRot = new Dictionary<string, Quaternion>();
        RobotStateData _pending;
        bool _bindCaptured;
        bool _calibrated;
        bool _loggedFirst;

        void Awake()
        {
            if (linkMap == null)
                linkMap = GetComponent<AgibotLinkMap>();
            if (envAnchor == null)
                envAnchor = transform;
            CaptureUnityBindPose();
        }

        void OnEnable()
        {
            if (linkMap == null)
                linkMap = GetComponent<AgibotLinkMap>();
            linkMap?.Rebuild();
            if (client != null)
                client.MessageReceived += OnMessage;
        }

        void OnDisable()
        {
            if (client != null)
                client.MessageReceived -= OnMessage;
            _pending = null;
        }

        void LateUpdate()
        {
            if (!followingEnabled || _pending == null)
                return;
            var state = _pending;
            _pending = null;
            Apply(state);
        }

        [ContextMenu("Recapture Unity Bind Pose")]
        public void CaptureUnityBindPose()
        {
            _unityBindLocalRot.Clear();
            _visualCorrection.Clear();
            _calibrated = false;
            _bindCaptured = false;
            if (linkMap == null)
                return;
            linkMap.Rebuild();
            Transform anchor = envAnchor != null ? envAnchor : transform;
            foreach (var name in linkMap.expectedLinks)
            {
                if (!linkMap.TryGet(name, out var t) || t == null)
                    continue;
                _unityBindLocalRot[name] = Quaternion.Inverse(anchor.rotation) * t.rotation;
            }
            _bindCaptured = _unityBindLocalRot.Count > 0;
        }

        void OnMessage(string json)
        {
            if (!followingEnabled)
                return;
            if (!RosJson.TryParseTopic(json, out var topic) || topic != robotStateTopic)
                return;
            if (!RosJson.TryParseRobotState(json, out var state, out _))
                return;
            _pending = state;
        }

        public void Apply(RobotStateData state)
        {
            if (state?.links == null || linkMap == null)
                return;
            Transform anchor = envAnchor != null ? envAnchor : transform;
            if (calibrateVisualFrames && !_calibrated)
                CalibrateFromPacket(state, anchor);

            int applied = 0;
            foreach (var link in state.links)
            {
                if (link == null || string.IsNullOrEmpty(link.name))
                    continue;
                if (!linkMap.TryGet(link.name, out var t) || t == null)
                    continue;
                ToUnityPose(link.position, link.orientation_xyzw, out var pLocal, out var qIsaacUnity);
                if (!_visualCorrection.TryGetValue(link.name, out var correction))
                    correction = Quaternion.identity;
                t.SetPositionAndRotation(
                    anchor.TransformPoint(pLocal),
                    anchor.rotation * (qIsaacUnity * correction));
                applied++;
            }

            if (logFirstApply && !_loggedFirst)
            {
                _loggedFirst = true;
                Debug.Log($"[AgibotLinkPoseFollower] Applied {applied}/{state.links.Length} links.", this);
            }
        }

        void CalibrateFromPacket(RobotStateData state, Transform anchor)
        {
            int n = 0;
            foreach (var link in state.links)
            {
                if (link == null || !_unityBindLocalRot.TryGetValue(link.name, out var unityBind))
                    continue;
                ToUnityPose(link.position, link.orientation_xyzw, out _, out var isaacBind);
                _visualCorrection[link.name] = Quaternion.Inverse(isaacBind) * unityBind;
                n++;
            }
            _calibrated = n > 0;
        }

        static void ToUnityPose(float[] pos, float[] quatXyzw, out Vector3 pUnity, out Quaternion qUnity)
        {
            pUnity = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(pos));
            qUnity = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(quatXyzw));
        }
    }
}
