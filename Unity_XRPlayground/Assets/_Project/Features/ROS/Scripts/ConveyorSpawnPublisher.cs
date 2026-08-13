using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Sends /xr/conveyor/spawn so Isaac activates a colored cube on the belt.
    /// </summary>
    public sealed class ConveyorSpawnPublisher : MonoBehaviour
    {
        public RosTcpClient client;
        public Transform envAnchor;
        public bool publishingEnabled = true;

        /// <summary>Spawn at default belt entry (Isaac env-local, on belt surface).</summary>
        public void SpawnColor(int color)
        {
            // Matches conveyor_color_env_cfg spawn band; Isaac re-pins Z to belt surface.
            SpawnColorAt(color, new Vector3(0.55f, -0.50f, 0.445f));
        }

        public void SpawnColorAt(int color, Vector3 isaacLocalPos)
        {
            if (!publishingEnabled || client == null || !client.IsConnected)
            {
                Debug.LogWarning("[ConveyorSpawn] Not connected / publishing disabled.");
                return;
            }
            client.PublishJson(RosJson.SerializeConveyorSpawn(color % 3, XrFrameConverter.ToArray(isaacLocalPos)));
        }

        public void SpawnRed() => SpawnColor(0);
        public void SpawnGreen() => SpawnColor(1);
        public void SpawnBlue() => SpawnColor(2);
    }
}
