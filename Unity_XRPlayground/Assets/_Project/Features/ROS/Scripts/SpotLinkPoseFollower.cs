using UnityEngine;
using XRPlayground.Robots;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Applies /xr/spot/robot_state link poses onto the Spot hierarchy.
    /// </summary>
    public sealed class SpotLinkPoseFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public SpotLinkMap linkMap;
        public Transform envAnchor;
        public string robotStateTopic = RosTopics.SpotRobotState;
        public bool followingEnabled = true;
        public bool logFirstApply = true;

        RobotStateData _pending;
        bool _loggedFirst;

        void Awake()
        {
            if (linkMap == null)
                linkMap = GetComponent<SpotLinkMap>();
            if (envAnchor == null)
                envAnchor = transform;
        }

        void OnEnable()
        {
            if (linkMap != null)
                linkMap.Rebuild();
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

        void OnMessage(string json)
        {
            if (!followingEnabled)
                return;
            if (!RosJson.TryParseTopic(json, out var topic) || topic != robotStateTopic)
                return;
            if (!RosJson.TryParseRobotState(json, out var state, out _) || state == null)
                return;
            _pending = state;
        }

        public void Apply(RobotStateData state)
        {
            if (linkMap == null || envAnchor == null || state?.links == null)
                return;

            int applied = 0;
            foreach (var link in state.links)
            {
                if (link == null || string.IsNullOrEmpty(link.name))
                    continue;
                if (!linkMap.TryGet(link.name, out var t) || t == null)
                    continue;
                if (link.position == null || link.position.Length < 3)
                    continue;
                if (link.orientation_xyzw == null || link.orientation_xyzw.Length < 4)
                    continue;

                var pos = new Vector3(link.position[0], link.position[1], link.position[2]);
                var quat = new Quaternion(
                    link.orientation_xyzw[0],
                    link.orientation_xyzw[1],
                    link.orientation_xyzw[2],
                    link.orientation_xyzw[3]);
                XrFrameConverter.ApplyIsaacBody(t, envAnchor, pos, quat);
                applied++;
            }

            if (logFirstApply && !_loggedFirst)
            {
                _loggedFirst = true;
                Debug.Log($"SpotLinkPoseFollower: applied {applied}/{state.links.Length} links.", this);
            }
        }
    }
}
