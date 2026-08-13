#if UNITY_EDITOR
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace XRPlayground.Robots.Editor
{
    /// <summary>
    /// Compares scene UR10e + Robotiq link names to the Isaac Lab expected list.
    /// Menu: XRPlayground / Audit UR10e Hierarchy
    /// </summary>
    public static class Ur10eHierarchyAudit
    {
        public const string RobotName = "UR10e_Robotiq";

        /// <summary>Must stay in sync with Isaac names_ur10e.UR10E_LINK_NAMES.</summary>
        public static readonly string[] ExpectedLinks =
        {
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "robotiq_base_link",
            "left_outer_knuckle",
            "left_outer_finger",
            "left_inner_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_outer_finger",
            "right_inner_finger",
            "right_inner_knuckle",
        };

        public static readonly string[] ExpectedJoints =
        {
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "finger_joint",
        };

        [MenuItem("XRPlayground/Audit UR10e Hierarchy")]
        public static void Audit()
        {
            var root = GameObject.Find(RobotName);
            if (root == null)
            {
                Debug.LogError(
                    $"XRPlayground: '{RobotName}' not found. Run 'Setup Conveyor Color Station' first.");
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

            var sb = new StringBuilder();
            sb.AppendLine($"UR10e hierarchy audit for '{RobotName}':");
            sb.AppendLine($"  Expected Isaac links: {ExpectedLinks.Length}");
            sb.AppendLine($"  Missing links: {missing.Count}");
            foreach (var m in missing)
                sb.AppendLine($"    - {m}");

            sb.AppendLine("  Isaac joints (published by bridge; Unity follows link poses, not joint DOFs):");
            foreach (var j in ExpectedJoints)
                sb.AppendLine($"    - {j}");

            var map = root.GetComponent<RobotLinkMap>();
            if (map != null)
            {
                map.Rebuild();
                sb.AppendLine($"  RobotLinkMap mapped transforms: {map.Map.Count}");
            }

            if (missing.Count == 0)
                Debug.Log(sb.ToString());
            else
                Debug.LogWarning(sb.ToString());
        }
    }
}
#endif
