#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XRPlayground.Robots.Editor
{
    /// <summary>
    /// Creates URP Lit materials for the Isaac USD UR10e and assigns them in-scene.
    /// The Nucleus USD uses OmniPBR; Unity importer leaves MeshRenderer slots empty.
    /// Menu: XRPlayground / Apply UR10e Materials
    /// </summary>
    public static class Ur10eMaterialSetup
    {
        public const string RobotName = "UR10e_Robotiq";
        const string MaterialsFolder = "Assets/_Project/Features/Robots/UR10e/Materials";

        // Approximate Universal Robots look (graphite joints + light arm shells).
        static readonly Color BodyColor = new Color(0.88f, 0.89f, 0.90f, 1f);
        static readonly Color DarkMetal = new Color(0.16f, 0.17f, 0.18f, 1f);
        static readonly Color AccentBlue = new Color(0.10f, 0.45f, 0.78f, 1f);
        static readonly Color GripperPad = new Color(0.07f, 0.07f, 0.08f, 1f);
        static readonly Color JointCap = new Color(0.05f, 0.05f, 0.055f, 1f);

        [MenuItem("XRPlayground/Apply UR10e Materials")]
        public static void Apply()
        {
            var root = GameObject.Find(RobotName);
            if (root == null)
            {
                Debug.LogError($"XRPlayground: '{RobotName}' not found in the active scene.");
                return;
            }

            ApplyTo(root);
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        /// <summary>Used by Station B setup after instantiating the USD.</summary>
        public static void ApplyTo(GameObject root)
        {
            if (root == null)
                return;

            EnsureMaterialAssets();

            var body = LoadMat("UR10e_Body");
            var dark = LoadMat("UR10e_DarkMetal");
            var accent = LoadMat("UR10e_Accent");
            var pad = LoadMat("UR10e_GripperPad");
            var joint = LoadMat("UR10e_JointCap");

            int assigned = 0;
            int hidden = 0;
            foreach (var renderer in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                string path = GetHierarchyPath(renderer.transform, root.name).ToLowerInvariant();

                // Collision meshes + USD mesh library payloads (duplicated under visuals).
                bool hide =
                    path.Contains("/colliders") ||
                    path.Contains("/collider") ||
                    path.Contains("/collision") ||
                    IsMeshLibraryOnly(path);
                if (hide)
                {
                    renderer.enabled = false;
                    hidden++;
                    continue;
                }

                Material mat = ChooseMaterial(path, body, dark, accent, pad, joint);
                var mats = renderer.sharedMaterials;
                if (mats == null || mats.Length == 0)
                {
                    renderer.sharedMaterial = mat;
                }
                else
                {
                    for (int i = 0; i < mats.Length; i++)
                        mats[i] = mat;
                    renderer.sharedMaterials = mats;
                }

                assigned++;
            }

            Debug.Log(
                $"XRPlayground: Applied UR10e materials to {assigned} visual renderers " +
                $"(hid {hidden} collision/library meshes)."
            );
        }

        static bool IsMeshLibraryOnly(string path)
        {
            // Top-level "meshes/…" under the imported base are often unparented payloads.
            // Keep "…/visuals/…" and "…/ur10e/…/visuals/…".
            if (path.Contains("/visuals/"))
                return false;
            // "/meshes/" after robot root, not nested under a link visual
            int idx = path.IndexOf("/meshes/");
            return idx >= 0 && !path.Contains("/visuals/");
        }

        static Material ChooseMaterial(
            string path,
            Material body,
            Material dark,
            Material accent,
            Material pad,
            Material joint
        )
        {
            // Do not match the GameObject root name "UR10e_Robotiq".
            bool isGripperLink =
                path.Contains("/robotiq") ||
                path.Contains("outer_knuckle") ||
                path.Contains("inner_knuckle") ||
                path.Contains("outer_finger") ||
                path.Contains("inner_finger") ||
                path.Contains("finger_pad") ||
                path.Contains("gripper");
            if (isGripperLink)
                return path.Contains("base") ? dark : pad;

            // Secondary mesh pieces under a link are usually dark caps / flanges.
            if (path.EndsWith("/mesh") || path.Contains("_0/mesh") || path.Contains("/mesh_"))
                return joint;

            if (path.Contains("base_link") || path.Contains("/base/base") || path.Contains("base_link_inertia"))
                return dark;

            if (path.Contains("shoulder"))
                return dark;

            if (path.Contains("wrist_3") || path.Contains("wrist3") || path.Contains("tool0") || path.Contains("flange"))
                return accent;

            if (path.Contains("wrist"))
                return dark;

            if (path.Contains("upper_arm") || path.Contains("upperarm") || path.Contains("forearm"))
                return body;

            return body;
        }

        static void EnsureMaterialAssets()
        {
            if (!AssetDatabase.IsValidFolder("Assets/_Project/Features/Robots/UR10e"))
                AssetDatabase.CreateFolder("Assets/_Project/Features/Robots", "UR10e");
            if (!AssetDatabase.IsValidFolder(MaterialsFolder))
                AssetDatabase.CreateFolder("Assets/_Project/Features/Robots/UR10e", "Materials");

            CreateOrUpdateMat("UR10e_Body", BodyColor, metallic: 0.30f, smoothness: 0.55f);
            CreateOrUpdateMat("UR10e_DarkMetal", DarkMetal, metallic: 0.80f, smoothness: 0.45f);
            CreateOrUpdateMat("UR10e_Accent", AccentBlue, metallic: 0.35f, smoothness: 0.50f);
            CreateOrUpdateMat("UR10e_GripperPad", GripperPad, metallic: 0.05f, smoothness: 0.25f);
            CreateOrUpdateMat("UR10e_JointCap", JointCap, metallic: 0.65f, smoothness: 0.40f);
            AssetDatabase.SaveAssets();
        }

        static void CreateOrUpdateMat(string name, Color color, float metallic, float smoothness)
        {
            string path = $"{MaterialsFolder}/{name}.mat";
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Lit");
            if (shader == null)
                shader = Shader.Find("Standard");

            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null)
            {
                mat = new Material(shader) { name = name };
                AssetDatabase.CreateAsset(mat, path);
            }
            else
            {
                mat.shader = shader;
            }

            if (mat.HasProperty("_BaseColor"))
                mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color"))
                mat.SetColor("_Color", color);
            if (mat.HasProperty("_Metallic"))
                mat.SetFloat("_Metallic", metallic);
            if (mat.HasProperty("_Smoothness"))
                mat.SetFloat("_Smoothness", smoothness);
            if (mat.HasProperty("_Glossiness"))
                mat.SetFloat("_Glossiness", smoothness);
            if (mat.HasProperty("_EnvironmentReflections"))
                mat.SetFloat("_EnvironmentReflections", 1f);

            EditorUtility.SetDirty(mat);
        }

        static Material LoadMat(string name) =>
            AssetDatabase.LoadAssetAtPath<Material>($"{MaterialsFolder}/{name}.mat");

        static string GetHierarchyPath(Transform t, string rootName)
        {
            var parts = new System.Collections.Generic.List<string>();
            while (t != null)
            {
                parts.Add(t.name);
                if (t.name == rootName)
                    break;
                t = t.parent;
            }

            parts.Reverse();
            return string.Join("/", parts);
        }
    }
}
#endif
