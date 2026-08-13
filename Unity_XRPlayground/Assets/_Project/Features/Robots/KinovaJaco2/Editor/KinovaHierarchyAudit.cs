#if UNITY_EDITOR
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace XRPlayground.Robots.Editor
{
    /// <summary>
    /// Compares scene Kinova link names to the Isaac Lab expected list.
    /// Menu: XRPlayground / Audit Kinova Hierarchy
    /// </summary>
    public static class KinovaHierarchyAudit
    {
        public const string RobotName = "Kinova_Jaco2_j2n7s300";

        public static readonly string[] ExpectedLinks =
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

        [MenuItem("XRPlayground/Audit Kinova Hierarchy")]
        public static void Audit()
        {
            var root = GameObject.Find(RobotName);
            if (root == null)
            {
                Debug.LogError($"XRPlayground: '{RobotName}' not found in the active scene.");
                return;
            }

            var found = new HashSet<string>();
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                found.Add(t.name);

            var missing = new List<string>();
            foreach (var name in ExpectedLinks)
            {
                if (!found.Contains(name))
                    missing.Add(name);
            }

            var joints = new List<string>();
            foreach (var name in found)
            {
                if (name.Contains("joint"))
                    joints.Add(name);
            }

            var sb = new StringBuilder();
            sb.AppendLine($"Kinova hierarchy audit for '{RobotName}':");
            sb.AppendLine($"  Expected links: {ExpectedLinks.Length}");
            sb.AppendLine($"  Missing links: {missing.Count}");
            foreach (var m in missing)
                sb.AppendLine($"    - {m}");
            sb.AppendLine($"  Joint-named transforms: {joints.Count} (Isaac has joints; Unity USD is flat links)");
            foreach (var j in joints)
                sb.AppendLine($"    - {j}");

            if (missing.Count == 0)
                Debug.Log(sb.ToString());
            else
                Debug.LogWarning(sb.ToString());
        }
    }
}
#endif
