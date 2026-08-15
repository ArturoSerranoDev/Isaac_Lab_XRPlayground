using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Robots
{
    /// <summary>
    /// Link-name cache for Agibot A2D (Isaac Lab body names). Prefer articulated subtree over visuals/.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class AgibotLinkMap : MonoBehaviour
    {
        [Tooltip("Isaac Lab Agibot A2D bodies (torso + right arm + gripper).")]
        public string[] expectedLinks =
        {
            "base_link",
            "body_link",
            "head_link",
            "right_arm_link1",
            "right_arm_link2",
            "right_arm_link3",
            "right_arm_link4",
            "right_arm_link5",
            "right_arm_link6",
            "right_arm_link7",
            "right_gripper_base",
            "right_gripper_center",
            "right_Left_Pad_Link",
            "right_Right_Pad_Link",
        };

        public string preferredSubtree = "A2D";
        public string[] ignoredSubtreeRoots = { "visuals", "meshes", "colliders", "Looks", "materials" };

        readonly Dictionary<string, Transform> _map = new Dictionary<string, Transform>();

        public IReadOnlyDictionary<string, Transform> Map => _map;

        void Awake() => Rebuild();

        [ContextMenu("Rebuild Link Map")]
        public void Rebuild()
        {
            _map.Clear();
            Transform searchRoot = ResolveSearchRoot();
            var candidates = new Dictionary<string, List<Transform>>();

            foreach (var t in searchRoot.GetComponentsInChildren<Transform>(true))
            {
                if (t == searchRoot || IsUnderIgnoredBranch(t, searchRoot))
                    continue;
                if (!candidates.TryGetValue(t.name, out var list))
                {
                    list = new List<Transform>(2);
                    candidates[t.name] = list;
                }
                list.Add(t);
            }

            foreach (var kv in candidates)
                _map[kv.Key] = PickBest(kv.Value, searchRoot);
        }

        public bool TryGet(string name, out Transform t) => _map.TryGetValue(name, out t);

        Transform ResolveSearchRoot()
        {
            if (!string.IsNullOrEmpty(preferredSubtree))
            {
                var sub = transform.Find(preferredSubtree);
                if (sub != null)
                    return sub;
                foreach (var t in GetComponentsInChildren<Transform>(true))
                {
                    if (t.name == preferredSubtree)
                        return t;
                }
            }
            return transform;
        }

        static bool IsUnderIgnoredBranch(Transform t, Transform root)
        {
            while (t != null && t != root)
            {
                if (t.name is "visuals" or "meshes" or "colliders" or "Looks" or "materials")
                    return true;
                t = t.parent;
            }
            return false;
        }

        static Transform PickBest(List<Transform> list, Transform root)
        {
            if (list.Count == 1)
                return list[0];
            Transform best = list[0];
            int bestDepth = int.MaxValue;
            foreach (var t in list)
            {
                int d = Depth(t, root);
                if (d < bestDepth)
                {
                    bestDepth = d;
                    best = t;
                }
            }
            return best;
        }

        static int Depth(Transform t, Transform root)
        {
            int d = 0;
            while (t != null && t != root)
            {
                d++;
                t = t.parent;
            }
            return d;
        }
    }
}
