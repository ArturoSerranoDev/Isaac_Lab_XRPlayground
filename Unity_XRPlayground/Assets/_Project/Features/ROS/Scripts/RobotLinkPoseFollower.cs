using System.Collections.Generic;
using UnityEngine;
using XRPlayground.Robots;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Applies /xr/conveyor/robot_state link poses onto a UR10e USD hierarchy.
    /// Same Isaac Z-up → Unity Y-up convention as <see cref="KinovaLinkPoseFollower"/> /
    /// ball-throw bridge: pos (x,z,y), quat (x,z,y,-w), then envAnchor frame.
    /// Applies in LateUpdate so USD import drivers cannot wipe poses the same frame.
    /// </summary>
    public sealed class RobotLinkPoseFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public RobotLinkMap linkMap;

        [Tooltip("Unity transform for the Isaac env origin (Station_B for conveyor).")]
        public Transform envAnchor;

        public string robotStateTopic = RosTopics.ConveyorRobotState;
        public bool followingEnabled = true;

        [Tooltip("Log how many links were applied from the first robot_state packet.")]
        public bool logFirstApply = true;

        [Tooltip("Optional: compute Isaac→USD visual corrections on the first packet (same bind pose).")]
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
                linkMap = GetComponent<RobotLinkMap>();
            if (envAnchor == null)
                envAnchor = transform;
            CaptureUnityBindPose();
        }

        void OnEnable()
        {
            if (linkMap == null)
                linkMap = GetComponent<RobotLinkMap>();
            if (envAnchor == null)
                envAnchor = transform;
            if (linkMap != null)
                linkMap.Rebuild();
            if (!_bindCaptured)
                CaptureUnityBindPose();
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
                linkMap = GetComponent<RobotLinkMap>();
            if (linkMap == null)
                return;

            linkMap.Rebuild();
            Transform anchor = envAnchor != null ? envAnchor : transform;
            foreach (var name in linkMap.expectedLinks)
            {
                if (!linkMap.TryGet(name, out var t) || t == null)
                    continue;
                Quaternion localRot = Quaternion.Inverse(anchor.rotation) * t.rotation;
                _unityBindLocalRot[name] = localRot;
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
            int missing = 0;
            foreach (var link in state.links)
            {
                if (link == null || string.IsNullOrEmpty(link.name))
                    continue;
                if (!linkMap.TryGet(link.name, out var t) || t == null)
                {
                    missing++;
                    continue;
                }

                ToUnityPose(link.position, link.orientation_xyzw, out var pLocal, out var qIsaacUnity);
                if (!_visualCorrection.TryGetValue(link.name, out var correction))
                    correction = Quaternion.identity;

                // Same as Kinova / ball-throw: Isaac body in Unity env frame, then USD visual correction.
                Quaternion qLocal = qIsaacUnity * correction;
                Vector3 worldPos = anchor.TransformPoint(pLocal);
                Quaternion worldRot = anchor.rotation * qLocal;
                t.SetPositionAndRotation(worldPos, worldRot);
                applied++;
            }

            if (logFirstApply && !_loggedFirst)
            {
                _loggedFirst = true;
                Debug.Log(
                    $"[RobotLinkPoseFollower] Applied {applied}/{state.links.Length} links " +
                    $"(missingMap={missing}, calibrated={_calibrated}, anchor={anchor.name}, " +
                    $"mapSize={linkMap.Map.Count}).",
                    this);
                if (applied == 0)
                {
                    Debug.LogWarning(
                        "[RobotLinkPoseFollower] 0 links matched. Run XRPlayground → Audit UR10e Hierarchy. " +
                        "Unity USD names must match Isaac body names (base_link, shoulder_link, …).",
                        this);
                }
            }
        }

        void CalibrateFromPacket(RobotStateData state, Transform anchor)
        {
            int n = 0;
            foreach (var link in state.links)
            {
                if (link == null || string.IsNullOrEmpty(link.name))
                    continue;
                if (!_unityBindLocalRot.TryGetValue(link.name, out var unityBind))
                    continue;

                ToUnityPose(link.position, link.orientation_xyzw, out _, out var isaacBind);
                _visualCorrection[link.name] = Quaternion.Inverse(isaacBind) * unityBind;
                n++;
            }

            _calibrated = n > 0;
            if (logFirstApply)
            {
                Debug.Log(
                    $"[RobotLinkPoseFollower] Visual-frame calibration for {n} links.",
                    this);
            }
        }

        static void ToUnityPose(float[] pos, float[] quatXyzw, out Vector3 pUnity, out Quaternion qUnity)
        {
            // Wire = Isaac Z-up. Match Kinova / XrFrameConverter / Python frame_math.
            pUnity = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(pos));
            qUnity = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(quatXyzw));
        }
    }
}
