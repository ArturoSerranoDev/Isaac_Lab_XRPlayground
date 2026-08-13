using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Robots
{
    /// <summary>
    /// Link-name → Transform cache for UR10e + Robotiq (Isaac Lab body names).
    /// Names must match Isaac <c>names_ur10e.UR10E_LINK_NAMES</c> / bridge publish.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class RobotLinkMap : MonoBehaviour
    {
        [Tooltip("Isaac Lab body names present on ur10e_base.usd visuals (arm). Robotiq links appear after Gripper USD is imported.")]
        public string[] expectedLinks =
        {
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        };

        readonly Dictionary<string, Transform> _map = new Dictionary<string, Transform>();

        public IReadOnlyDictionary<string, Transform> Map => _map;

        void Awake() => Rebuild();

        [ContextMenu("Rebuild Link Map")]
        public void Rebuild()
        {
            _map.Clear();
            foreach (var t in GetComponentsInChildren<Transform>(true))
            {
                if (!_map.ContainsKey(t.name))
                    _map[t.name] = t;
            }

            int missing = 0;
            foreach (var name in expectedLinks)
            {
                if (!_map.ContainsKey(name))
                {
                    missing++;
                    Debug.LogWarning($"RobotLinkMap: missing '{name}' under {gameObject.name}", this);
                }
            }

            if (missing == 0)
                Debug.Log($"RobotLinkMap: bound {expectedLinks.Length} expected links.", this);
            else
                Debug.Log(
                    $"RobotLinkMap: {expectedLinks.Length - missing}/{expectedLinks.Length} links found " +
                    $"(total mapped={_map.Count}).",
                    this);
        }

        public bool TryGet(string linkName, out Transform t) => _map.TryGetValue(linkName, out t);
    }
}
