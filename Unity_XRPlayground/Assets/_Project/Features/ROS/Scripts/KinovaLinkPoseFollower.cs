using UnityEngine;
using XRPlayground.Robots;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Applies /xr/robot_state link poses onto the flat Unity Kinova USD hierarchy.
    /// </summary>
    public sealed class KinovaLinkPoseFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public KinovaLinkMap linkMap;
        [Tooltip("World offset of the Unity robot root relative to Isaac env origin (Unity Y-up).")]
        public Vector3 rootOffset;
        public bool applyEeOnly;
        public Transform eeDebug;

        void OnEnable()
        {
            if (client != null)
                client.MessageReceived += OnMessage;
            if (linkMap == null)
                linkMap = GetComponent<KinovaLinkMap>();
        }

        void OnDisable()
        {
            if (client != null)
                client.MessageReceived -= OnMessage;
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

            if (state.ee != null && eeDebug != null)
            {
                var p = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(state.ee.position));
                var q = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(state.ee.orientation_xyzw));
                eeDebug.SetPositionAndRotation(p + rootOffset, q);
            }

            if (applyEeOnly || state.links == null || linkMap == null)
                return;

            foreach (var link in state.links)
            {
                if (link == null || string.IsNullOrEmpty(link.name))
                    continue;
                if (!linkMap.TryGet(link.name, out var t) || t == null)
                    continue;
                var p = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(link.position));
                var q = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(link.orientation_xyzw));
                t.SetPositionAndRotation(p + rootOffset, q);
            }
        }
    }
}
