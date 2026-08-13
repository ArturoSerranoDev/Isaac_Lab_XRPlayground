using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>Sends periodic /xr/heartbeat to Isaac.</summary>
    public sealed class RosHeartbeatPublisher : MonoBehaviour
    {
        public RosTcpClient client;
        public float intervalSeconds = 1f;
        float _next;

        void Update()
        {
            if (client == null || !client.IsConnected)
                return;
            if (Time.time < _next)
                return;
            _next = Time.time + intervalSeconds;
            client.PublishJson(RosJson.SerializeHeartbeat("unity"));
        }
    }
}
