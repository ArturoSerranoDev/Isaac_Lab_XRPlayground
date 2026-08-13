using UnityEngine;
using XRPlayground.Robots;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Applies /xr/conveyor/robot_state (or custom topic) link poses onto a robot USD hierarchy.
    /// </summary>
    public sealed class RobotLinkPoseFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public RobotLinkMap linkMap;
        public Transform envAnchor;
        public string robotStateTopic = RosTopics.ConveyorRobotState;
        public bool followingEnabled = true;

        void OnEnable()
        {
            if (linkMap == null)
                linkMap = GetComponent<RobotLinkMap>();
            if (envAnchor == null)
                envAnchor = transform;
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
            if (!RosJson.TryParseTopic(json, out var topic) || topic != robotStateTopic)
                return;
            if (!RosJson.TryParseRobotState(json, out var state, out _))
                return;
            Apply(state);
        }

        public void Apply(RobotStateData state)
        {
            if (state?.links == null || linkMap == null)
                return;
            Transform anchor = envAnchor != null ? envAnchor : transform;
            foreach (var link in state.links)
            {
                if (link == null || string.IsNullOrEmpty(link.name))
                    continue;
                if (!linkMap.TryGet(link.name, out var t) || t == null)
                    continue;
                var p = XrFrameConverter.IsaacPosToUnity(XrFrameConverter.FromArray3(link.position));
                var q = XrFrameConverter.IsaacQuatToUnity(XrFrameConverter.FromXyzw(link.orientation_xyzw));
                t.SetPositionAndRotation(anchor.TransformPoint(p), anchor.rotation * q);
            }
        }
    }
}
