#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace XRPlayground.Robots.Editor
{
    /// <summary>Replaces Spot's unsupported OmniPBR materials with URP-compatible materials.</summary>
    public static class SpotMaterialSetup
    {
        const string MaterialsFolder = "Assets/_Project/Features/Robots/Spot/Materials";

        public static void ApplyTo(GameObject root)
        {
            if (root == null)
                return;

            EnsureMaterialAssets();
            var body = Load("Spot_Body");
            var leg = Load("Spot_Leg");

            foreach (var renderer in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                var material = GetPath(renderer.transform, root.transform).Contains("/body/") ? body : leg;
                var materials = renderer.sharedMaterials;
                if (materials == null || materials.Length == 0)
                    renderer.sharedMaterial = material;
                else
                {
                    for (int i = 0; i < materials.Length; i++)
                        materials[i] = material;
                    renderer.sharedMaterials = materials;
                }
            }
        }

        static void EnsureMaterialAssets()
        {
            if (!AssetDatabase.IsValidFolder(MaterialsFolder))
                AssetDatabase.CreateFolder("Assets/_Project/Features/Robots/Spot", "Materials");

            CreateOrUpdate("Spot_Body", new Color(0.78f, 0.62f, 0.28f, 1f), 0.25f, 0.42f);
            CreateOrUpdate("Spot_Leg", new Color(0.12f, 0.14f, 0.16f, 1f), 0.75f, 0.38f);
            AssetDatabase.SaveAssets();
        }

        static void CreateOrUpdate(string name, Color color, float metallic, float smoothness)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var path = $"{MaterialsFolder}/{name}.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(shader) { name = name };
                AssetDatabase.CreateAsset(material, path);
            }
            else
                material.shader = shader;

            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
            if (material.HasProperty("_Color")) material.SetColor("_Color", color);
            if (material.HasProperty("_Metallic")) material.SetFloat("_Metallic", metallic);
            if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", smoothness);
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", smoothness);
            EditorUtility.SetDirty(material);
        }

        static Material Load(string name) =>
            AssetDatabase.LoadAssetAtPath<Material>($"{MaterialsFolder}/{name}.mat");

        static string GetPath(Transform transform, Transform root)
        {
            var path = transform.name;
            while (transform.parent != null && transform.parent != root)
            {
                transform = transform.parent;
                path = transform.name + "/" + path;
            }
            return "/" + path + "/";
        }
    }
}
#endif
