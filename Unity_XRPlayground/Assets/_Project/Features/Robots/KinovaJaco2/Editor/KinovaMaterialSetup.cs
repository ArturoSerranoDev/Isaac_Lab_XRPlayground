#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace XRPlayground.Robots.Editor
{
    /// <summary>
    /// Creates URP Lit materials for the Isaac USD Kinova and assigns them in-scene.
    /// The Nucleus USD uses OmniPBR; Unity importer leaves MeshRenderer slots empty.
    /// Menu: XRPlayground / Apply Kinova Materials
    /// </summary>
    public static class KinovaMaterialSetup
    {
        const string RobotName = "Kinova_Jaco2_j2n7s300";
        const string MaterialsFolder = "Assets/_Project/Features/Robots/KinovaJaco2/Materials";

        // Approximate Kinova Jaco look (no albedo textures ship with the Isaac instanceable USD).
        static readonly Color BodyColor = new Color(0.82f, 0.84f, 0.86f, 1f);
        static readonly Color DarkMetal = new Color(0.18f, 0.18f, 0.19f, 1f);
        static readonly Color AccentBlue = new Color(0.12f, 0.42f, 0.78f, 1f);
        static readonly Color GripperPad = new Color(0.08f, 0.08f, 0.09f, 1f);
        static readonly Color RingColor = new Color(0.05f, 0.05f, 0.05f, 1f);

        [MenuItem("XRPlayground/Apply Kinova Materials")]
        public static void Apply()
        {
            EnsureMaterialAssets();

            var body = LoadMat("Kinova_Body");
            var dark = LoadMat("Kinova_DarkMetal");
            var accent = LoadMat("Kinova_Accent");
            var pad = LoadMat("Kinova_GripperPad");
            var ring = LoadMat("Kinova_Ring");

            var root = GameObject.Find(RobotName);
            if (root == null)
            {
                Debug.LogError($"XRPlayground: '{RobotName}' not found in the active scene.");
                return;
            }

            int assigned = 0;
            int hiddenCollisions = 0;
            foreach (var renderer in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                string path = GetHierarchyPath(renderer.transform).ToLowerInvariant();
                bool isCollision = path.Contains("/collisions") || path.Contains("/collision");
                if (isCollision)
                {
                    renderer.enabled = false;
                    hiddenCollisions++;
                    continue;
                }

                Material mat = ChooseMaterial(path, body, dark, accent, pad, ring);
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

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log(
                $"XRPlayground: Applied Kinova materials to {assigned} visual renderers " +
                $"(hid {hiddenCollisions} collision meshes)."
            );
        }

        static Material ChooseMaterial(
            string path,
            Material body,
            Material dark,
            Material accent,
            Material pad,
            Material ring
        )
        {
            if (path.Contains("finger"))
                return pad;

            // On multi-mesh links, mesh_1 is usually the black ring / trim.
            if (path.Contains("/mesh_1"))
                return ring;

            if (path.Contains("end_effector") || path.Contains("hand"))
                return dark;

            if (path.Contains("link_base") || path.Contains("link_1") || path.Contains("link_2"))
                return dark;

            if (path.Contains("link_6") || path.Contains("link_7"))
                return accent;

            if (path.Contains("link_"))
                return body;

            return body;
        }

        static void EnsureMaterialAssets()
        {
            if (!AssetDatabase.IsValidFolder("Assets/_Project/Features/Robots/KinovaJaco2"))
                AssetDatabase.CreateFolder("Assets/_Project/Features/Robots", "KinovaJaco2");
            if (!AssetDatabase.IsValidFolder(MaterialsFolder))
                AssetDatabase.CreateFolder("Assets/_Project/Features/Robots/KinovaJaco2", "Materials");

            CreateOrUpdateMat("Kinova_Body", BodyColor, metallic: 0.35f, smoothness: 0.55f);
            CreateOrUpdateMat("Kinova_DarkMetal", DarkMetal, metallic: 0.75f, smoothness: 0.45f);
            CreateOrUpdateMat("Kinova_Accent", AccentBlue, metallic: 0.25f, smoothness: 0.5f);
            CreateOrUpdateMat("Kinova_GripperPad", GripperPad, metallic: 0.05f, smoothness: 0.25f);
            CreateOrUpdateMat("Kinova_Ring", RingColor, metallic: 0.6f, smoothness: 0.4f);
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

            // Reasonable defaults for XR lab lighting
            if (mat.HasProperty("_EnvironmentReflections"))
                mat.SetFloat("_EnvironmentReflections", 1f);

            EditorUtility.SetDirty(mat);
        }

        static Material LoadMat(string name) =>
            AssetDatabase.LoadAssetAtPath<Material>($"{MaterialsFolder}/{name}.mat");

        static string GetHierarchyPath(Transform t)
        {
            var parts = new System.Collections.Generic.List<string>();
            while (t != null)
            {
                parts.Add(t.name);
                if (t.name == RobotName)
                    break;
                t = t.parent;
            }

            parts.Reverse();
            return string.Join("/", parts);
        }
    }
}
#endif
