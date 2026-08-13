using System.Collections.Generic;
using UnityEngine;
using XRPlayground.Robots;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Applies /xr/robot_state link poses onto a flat Unity robot hierarchy.
    ///
    /// Isaac publishes env-local Z-up poses. This converts them to Unity Y-up and places
    /// each link in the frame of <see cref="envAnchor"/> (typically the robot root that
    /// marks the Isaac env origin in the XR lab).
    /// </summary>
    public sealed class KinovaLinkPoseFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public KinovaLinkMap linkMap;

        [Tooltip("Unity transform that represents the Isaac env origin (position + rotation). Defaults to this transform.")]
        public Transform envAnchor;

        [Tooltip("Deprecated fallback if envAnchor is null: world translation only.")]
        public Vector3 rootOffset;

        public bool applyEeOnly;
        public Transform eeDebug;

        [Tooltip("If true, on the first packet compute per-link visual corrections. Only enable when Unity bind pose and Isaac pose match the same joint configuration.")]
        public bool calibrateVisualFrames = false;

        [Tooltip("Log how many links were applied from the first robot_state packet.")]
        public bool logFirstApply = true;

        readonly Dictionary<string, Quaternion> _visualCorrection = new Dictionary<string, Quaternion>();
        readonly Dictionary<string, Quaternion> _unityBindLocalRot = new Dictionary<string, Quaternion>();
        bool _bindCaptured;
        bool _calibrated;
        bool _loggedFirst;

        void Awake()
        {
            if (linkMap == null)
                linkMap = GetComponent<KinovaLinkMap>();
            if (envAnchor == null)
                envAnchor = transform;
            CaptureUnityBindPose();
        }

        void OnEnable()
        {
            if (client != null)
                client.MessageReceived += OnMessage;
            if (linkMap == null)
                linkMap = GetComponent<KinovaLinkMap>();
            if (envAnchor == null)
                envAnchor = transform;
            if (!_bindCaptured)
                CaptureUnityBindPose();
        }

        void OnDisable()
        {
            if (client != null)
                client.MessageReceived -= OnMessage;
        }

        [ContextMenu("Recapture Unity Bind Pose")]
        public void CaptureUnityBindPose()
        {
            _unityBindLocalRot.Clear();
            _visualCorrection.Clear();
            _calibrated = false;
            _bindCaptured = false;

            if (linkMap == null)
                linkMap = GetComponent<KinovaLinkMap>();
            if (linkMap == null)
                return;

            linkMap.Rebuild();
            Transform anchor = envAnchor != null ? envAnchor : transform;
            foreach (var name in linkMap.expectedLinks)
            {
                if (!linkMap.TryGet(name, out var t) || t == null)
                    continue;
                // Rotation of the link expressed in the env-anchor frame (Unity import / bind).
                Quaternion localRot = Quaternion.Inverse(anchor.rotation) * t.rotation;
                _unityBindLocalRot[name] = localRot;
            }

            _bindCaptured = _unityBindLocalRot.Count > 0;
        }

        void OnMessage(string json)
        {
            if (!RosJson.TryParseTopic(json, out var topic) || topic != RosTopics.RobotState)
                return;
            if (!RosJson.TryParseRobotState(json, out var state, out _))
                return;
            Apply(state);
        }

        public void Apply(RobotStateData state)
        {
            if (state == null)
                return;

            Transform anchor = envAnchor != null ? envAnchor : transform;

            if (state.ee != null && eeDebug != null)
            {
                ToUnityPose(state.ee.position, state.ee.orientation_xyzw, out var p, out var q);
                eeDebug.SetPositionAndRotation(anchor.TransformPoint(p), anchor.rotation * q);
            }

            if (applyEeOnly || state.links == null || linkMap == null)
                return;

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

                // Isaac body orientation in Unity env frame, then into the USD visual frame.
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
                    $"[KinovaLinkPoseFollower] Applied {applied}/{state.links.Length} links " +
                    $"(calibrated={_calibrated}, anchor={anchor.name}).",
                    this);
            }
        }

        void CalibrateFromPacket(RobotStateData state, Transform anchor)
        {
            // Assumes the first streamed configuration matches the Unity USD bind pose
            // (Isaac default_joint_pos ≈ imported USD rest pose).
            int n = 0;
            foreach (var link in state.links)
            {
                if (link == null || string.IsNullOrEmpty(link.name))
                    continue;
                if (!_unityBindLocalRot.TryGetValue(link.name, out var unityBind))
                    continue;

                ToUnityPose(link.position, link.orientation_xyzw, out _, out var isaacBind);
                // C = inv(R_isaac) * R_unity_visual  so  R_isaac * C = R_unity_visual
                _visualCorrection[link.name] = Quaternion.Inverse(isaacBind) * unityBind;
                n++;
            }

            _calibrated = n > 0;
            if (logFirstApply)
            {
                Debug.Log(
                    $"[KinovaLinkPoseFollower] Visual-frame calibration for {n} links " +
                    "(Isaac body → Unity USD mesh).",
                    this);
            }
        }

        static void ToUnityPose(float[] pos, float[] quatXyzw, out Vector3 pUnity, out Quaternion qUnity)
        {
            pUnity = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(pos));
            qUnity = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(quatXyzw));
        }

#if UNITY_EDITOR
        void OnValidate()
        {
            if (envAnchor == null)
                envAnchor = transform;
            // Keep legacy rootOffset in sync for older scenes / ball publisher setup.
            if (envAnchor != null)
                rootOffset = envAnchor.position;
        }
#endif
    }
}
