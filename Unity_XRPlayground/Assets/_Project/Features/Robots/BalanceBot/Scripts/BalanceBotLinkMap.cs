using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Robots
{
    /// <summary>
    /// Link-name cache for Balance Bot tray. Names match Isaac balance_bot_env_cfg + names_balance_bot.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class BalanceBotLinkMap : MonoBehaviour
    {
        [Tooltip("Isaac / Unity shared link names for the 2-DOF tray.")]
        public string[] expectedLinks =
        {
            "base_link",
            "roll_link",
            "tray_link",
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
                if (t == transform)
                    continue;
                if (!_map.ContainsKey(t.name))
                    _map[t.name] = t;
            }

            int missing = 0;
            foreach (var name in expectedLinks)
            {
                if (!_map.ContainsKey(name))
                {
                    missing++;
                    Debug.LogWarning($"BalanceBotLinkMap: missing link '{name}'", this);
                }
            }

            if (missing == 0)
                Debug.Log($"BalanceBotLinkMap: bound {expectedLinks.Length}/{expectedLinks.Length} expected links.", this);
        }

        public bool TryGet(string name, out Transform t) => _map.TryGetValue(name, out t);
    }
}
