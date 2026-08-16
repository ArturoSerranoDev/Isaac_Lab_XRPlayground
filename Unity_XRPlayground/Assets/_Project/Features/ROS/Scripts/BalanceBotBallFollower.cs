using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Mirrors /xr/balance_bot/balls_state onto Ball_0 / Ball_1 transforms.
    /// </summary>
    public sealed class BalanceBotBallFollower : MonoBehaviour
    {
        public RosTcpClient client;
        public Transform envAnchor;
        public Transform[] ballSlots = new Transform[2];
        public string ballsTopic = RosTopics.BalanceBotBallsState;
        public bool followingEnabled = true;

        BalanceBotBallsStateData _pending;

        void OnEnable()
        {
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
            Apply(_pending);
            _pending = null;
        }

        void OnMessage(string json)
        {
            if (!followingEnabled)
                return;
            if (!RosJson.TryParseTopic(json, out var topic) || topic != ballsTopic)
                return;
            if (!RosJson.TryParseBalanceBotBallsState(json, out var state) || state == null)
                return;
            _pending = state;
        }

        public void Apply(BalanceBotBallsStateData state)
        {
            if (envAnchor == null || state?.balls == null || ballSlots == null)
                return;

            foreach (var b in state.balls)
            {
                if (b == null || b.id < 0 || b.id >= ballSlots.Length)
                    continue;
                var slot = ballSlots[b.id];
                if (slot == null)
                    continue;

                if (!b.active)
                {
                    slot.gameObject.SetActive(false);
                    continue;
                }

                slot.gameObject.SetActive(true);
                if (b.position == null || b.position.Length < 3)
                    continue;
                if (b.orientation_xyzw == null || b.orientation_xyzw.Length < 4)
                    continue;

                var pos = new Vector3(b.position[0], b.position[1], b.position[2]);
                var quat = new Quaternion(
                    b.orientation_xyzw[0],
                    b.orientation_xyzw[1],
                    b.orientation_xyzw[2],
                    b.orientation_xyzw[3]);
                XrFrameConverter.ApplyIsaacBody(slot, envAnchor, pos, quat);
            }
        }
    }
}
