using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Robots
{
    /// <summary>
    /// Link-name cache for Boston Dynamics Spot. Names match Nucleus spot.usd + names_spot.py.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SpotLinkMap : MonoBehaviour
    {
        [Tooltip("Isaac / Unity shared link names (17 bodies).")]
        public string[] expectedLinks =
        {
            "body",
            "fl_hip", "fl_uleg", "fl_lleg", "fl_foot",
            "fr_hip", "fr_uleg", "fr_lleg", "fr_foot",
            "hl_hip", "hl_uleg", "hl_lleg", "hl_foot",
            "hr_hip", "hr_uleg", "hr_lleg", "hr_foot",
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
                    Debug.LogWarning($"SpotLinkMap: missing link '{name}'", this);
                }
            }

            if (missing == 0)
                Debug.Log($"SpotLinkMap: bound {expectedLinks.Length}/{expectedLinks.Length} expected links.", this);
            else
                Debug.LogWarning($"SpotLinkMap: bound {expectedLinks.Length - missing}/{expectedLinks.Length} links.", this);
        }

        public bool TryGet(string name, out Transform t) => _map.TryGetValue(name, out t);
    }
}
