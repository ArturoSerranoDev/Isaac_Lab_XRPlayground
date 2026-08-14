using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Robots
{
    /// <summary>
    /// Link-name → Transform cache for UR10e + Robotiq (Isaac Lab body names).
    /// The USD import often duplicates link names under <c>visuals/</c> (payload) and
    /// <c>ur10e/</c> (the articulated mesh). We prefer <see cref="preferredSubtree"/>.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class RobotLinkMap : MonoBehaviour
    {
        [Tooltip("Isaac Lab body names (arm + Robotiq on Gripper=Robotiq_2f_85).")]
        public string[] expectedLinks =
        {
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "base_link_0",
            "left_outer_knuckle",
            "left_outer_finger",
            "left_inner_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_outer_finger",
            "right_inner_finger",
            "right_inner_knuckle",
        };

        [Tooltip("Child under the robot root that owns the visible articulated mesh (not visuals/).")]
        public string preferredSubtree = "ur10e";

        [Tooltip("Top-level branches to ignore when a preferred subtree is present.")]
        public string[] ignoredSubtreeRoots =
        {
            "visuals",
            "meshes",
            "colliders",
            "Looks",
            "materials",
        };

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
                if (t == searchRoot)
                    continue;
                if (IsUnderIgnoredBranch(t, searchRoot))
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

            // Leaf aliases if importer used path-like names
            var extras = new List<KeyValuePair<string, Transform>>();
            foreach (var kv in _map)
            {
                string n = kv.Key;
                int slash = n.LastIndexOf('/');
                if (slash >= 0 && slash < n.Length - 1)
                {
                    string leaf = n.Substring(slash + 1);
                    if (!_map.ContainsKey(leaf))
                        extras.Add(new KeyValuePair<string, Transform>(leaf, kv.Value));
                }
            }
            foreach (var e in extras)
                _map[e.Key] = e.Value;

            int missing = 0;
            foreach (var name in expectedLinks)
            {
                if (!_map.ContainsKey(name))
                {
                    missing++;
                    Debug.LogWarning($"RobotLinkMap: missing '{name}' under {searchRoot.name}", this);
                }
            }

            string rootLabel = searchRoot == transform ? transform.name : $"{transform.name}/{searchRoot.name}";
            if (missing == 0)
                Debug.Log($"RobotLinkMap: bound {expectedLinks.Length} expected links under '{rootLabel}'.", this);
            else
                Debug.Log(
                    $"RobotLinkMap: {expectedLinks.Length - missing}/{expectedLinks.Length} links under '{rootLabel}' " +
                    $"(total mapped={_map.Count}).",
                    this);
        }

        Transform ResolveSearchRoot()
        {
            if (!string.IsNullOrEmpty(preferredSubtree))
            {
                // Direct child first (UR10e_Robotiq/ur10e)
                var direct = transform.Find(preferredSubtree);
                if (direct != null)
                    return direct;

                // Nested: …/ur10e anywhere under root
                foreach (var t in GetComponentsInChildren<Transform>(true))
                {
                    if (t != transform && t.name == preferredSubtree)
                        return t;
                }
            }

            return transform;
        }

        bool IsUnderIgnoredBranch(Transform t, Transform searchRoot)
        {
            // When searching the whole robot, skip known payload branches.
            if (searchRoot != transform)
                return false;

            Transform cur = t;
            while (cur != null && cur != transform)
            {
                if (cur.parent == transform)
                {
                    foreach (var ignore in ignoredSubtreeRoots)
                    {
                        if (!string.IsNullOrEmpty(ignore) && cur.name == ignore)
                            return true;
                    }
                    break;
                }
                cur = cur.parent;
            }
            return false;
        }

        static Transform PickBest(List<Transform> list, Transform searchRoot)
        {
            if (list.Count == 1)
                return list[0];

            // Prefer the shallowest transform under searchRoot that is not a pure mesh leaf
            // named "mesh" / "visuals" child of the link.
            Transform best = list[0];
            int bestScore = int.MinValue;
            foreach (var t in list)
            {
                int score = 0;
                // Prefer nodes that have children (link frames) over terminal mesh GOs
                if (t.childCount > 0)
                    score += 10;
                if (t.GetComponent<MeshFilter>() == null)
                    score += 5;
                // Prefer closer to searchRoot
                score -= DepthBelow(t, searchRoot);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = t;
                }
            }
            return best;
        }

        static int DepthBelow(Transform t, Transform root)
        {
            int d = 0;
            while (t != null && t != root)
            {
                d++;
                t = t.parent;
            }
            return d;
        }

        public bool TryGet(string linkName, out Transform t) => _map.TryGetValue(linkName, out t);
    }
}
