using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Robots
{
    /// <summary>
    /// Caches Kinova USD link Transforms by Isaac Lab body name for pose retarget.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class KinovaLinkMap : MonoBehaviour
    {
        [Tooltip("Link names that must exist under this robot root (Isaac Lab body names).")]
        public string[] expectedLinks =
        {
            "j2n7s300_link_base",
            "j2n7s300_link_1",
            "j2n7s300_link_2",
            "j2n7s300_link_3",
            "j2n7s300_link_4",
            "j2n7s300_link_5",
            "j2n7s300_link_6",
            "j2n7s300_link_7",
            "j2n7s300_end_effector",
            "j2n7s300_link_finger_1",
            "j2n7s300_link_finger_2",
            "j2n7s300_link_finger_3",
            "j2n7s300_link_finger_tip_1",
            "j2n7s300_link_finger_tip_2",
            "j2n7s300_link_finger_tip_3",
        };

        readonly Dictionary<string, Transform> _map = new Dictionary<string, Transform>();

        public IReadOnlyDictionary<string, Transform> Map => _map;

        void Awake() => Rebuild();

        [ContextMenu("Rebuild Link Map")]
        public void Rebuild()
        {
            _map.Clear();
            var transforms = GetComponentsInChildren<Transform>(true);
            foreach (var t in transforms)
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
                    Debug.LogWarning($"KinovaLinkMap: missing link '{name}' under {name}", this);
                }
            }

            if (missing == 0)
                Debug.Log($"KinovaLinkMap: bound {expectedLinks.Length} expected links.", this);
        }

        public bool TryGet(string linkName, out Transform t) => _map.TryGetValue(linkName, out t);
    }
}
