#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XRPlayground.VR.Editor
{
    /// <summary>
    /// Builds basic XR lab floor zones: Manipulation, Locomotion, Computer Vision.
    /// Menu: XRPlayground / Setup Lab Zones
    /// </summary>
    public static class XRLabZonesSetup
    {
        // World units: each zone is large enough for several robot stations.
        const float ZoneSize = 12f;
        const float Gap = 2f;
        const float GroundY = 0f;

        [MenuItem("XRPlayground/Setup Lab Zones")]
        public static void Setup()
        {
            Undo.IncrementCurrentGroup();
            Undo.SetCurrentGroupName("XR Lab Zones Setup");

            var env = GameObject.Find("Environment");
            if (env == null)
            {
                env = new GameObject("Environment");
                Undo.RegisterCreatedObjectUndo(env, "Create Environment");
            }

            // Remove old single floor if present
            var oldFloor = env.transform.Find("Floor");
            if (oldFloor != null)
                Undo.DestroyObjectImmediate(oldFloor.gameObject);

            var zonesRoot = env.transform.Find("Lab Zones");
            if (zonesRoot != null)
                Undo.DestroyObjectImmediate(zonesRoot.gameObject);

            zonesRoot = new GameObject("Lab Zones").transform;
            Undo.RegisterCreatedObjectUndo(zonesRoot.gameObject, "Create Lab Zones");
            zonesRoot.SetParent(env.transform, false);

            // Layout along +X: Manipulation | Locomotion | Vision
            float pitch = ZoneSize + Gap;
            CreateZone(zonesRoot, "01_Manipulation", new Vector3(0f, GroundY, 0f), new Color(0.22f, 0.35f, 0.55f), "MANIPULATION");
            CreateZone(zonesRoot, "02_Locomotion", new Vector3(pitch, GroundY, 0f), new Color(0.25f, 0.45f, 0.28f), "LOCOMOTION");
            CreateZone(zonesRoot, "03_ComputerVision", new Vector3(pitch * 2f, GroundY, 0f), new Color(0.50f, 0.32f, 0.18f), "COMPUTER VISION");

            // Shared outer ground under all zones (slightly larger pad)
            var pad = CreatePlane(zonesRoot, "Lab Ground", new Vector3(pitch, GroundY - 0.02f, 0f), pitch * 3f + Gap * 2f, new Color(0.12f, 0.12f, 0.13f));
            pad.transform.SetAsFirstSibling();

            // Anchors for robots / stations
            var manipStations = EnsureChild(zonesRoot.Find("01_Manipulation"), "Robot Stations");
            EnsureChild(zonesRoot.Find("02_Locomotion"), "Robot Stations");
            EnsureChild(zonesRoot.Find("03_ComputerVision"), "Robot Stations");

            PlaceKinovaTrainingRobot(manipStations);

            // Move XR Origin near Manipulation zone entrance
            var origin = GameObject.Find("XR Origin") ?? GameObject.Find("XR Origin (VR)");
            if (origin != null)
            {
                Undo.RecordObject(origin.transform, "Place XR Origin");
                origin.transform.position = new Vector3(-ZoneSize * 0.35f, 0f, -ZoneSize * 0.35f);
            }

            // Move grab cube into manipulation zone if it exists
            var grab = GameObject.Find("Grab Cube");
            if (grab != null)
            {
                Undo.RecordObject(grab.transform, "Place Grab Cube");
                grab.transform.position = new Vector3(0.4f, 1.1f, 0.5f);
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log("XRPlayground: Lab zones created (Manipulation / Locomotion / Computer Vision).");
        }

        const string KinovaUsdPath =
            "Assets/_Project/Features/Robots/KinovaJaco2/USD/j2n7s300_instanceable.usd";
        const string UsdImporterGraphPath =
            "Packages/com.unity.importer.usd/Unity.Importer.USD.Editor/ImportGraph/usdImporter.asset";

        static void PlaceKinovaTrainingRobot(Transform stations)
        {
            if (stations == null)
                return;

            EnsureKinovaUsdImported();

            var existing = stations.Find("Kinova_Jaco2_j2n7s300");
            if (existing != null)
                Undo.DestroyObjectImmediate(existing.gameObject);

            var usdRoot = AssetDatabase.LoadAssetAtPath<GameObject>(KinovaUsdPath);
            GameObject robot;
            if (usdRoot != null)
            {
                robot = (GameObject)PrefabUtility.InstantiatePrefab(usdRoot);
                robot.name = "Kinova_Jaco2_j2n7s300";
            }
            else
            {
                robot = new GameObject("Kinova_Jaco2_j2n7s300");
                var marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                marker.name = "USD_Pending_Marker";
                marker.transform.SetParent(robot.transform, false);
                marker.transform.localScale = new Vector3(0.25f, 0.5f, 0.25f);
                marker.transform.localPosition = new Vector3(0f, 0.5f, 0f);
                Debug.LogWarning(
                    $"XRPlayground: Could not instantiate USD at '{KinovaUsdPath}'. " +
                    "Enable Import (isUsdRoot) on the USD and re-run Setup Lab Zones.");
            }

            Undo.RegisterCreatedObjectUndo(robot, "Place Kinova Jaco2");
            robot.transform.SetParent(stations, false);
            // Station A in Manipulation: leaves room for more robots along +X.
            robot.transform.localPosition = new Vector3(-3f, 0f, -2f);
            robot.transform.localRotation = Quaternion.identity;
            robot.transform.localScale = Vector3.one;
        }

        static void EnsureKinovaUsdImported()
        {
            var importer = AssetImporter.GetAtPath(KinovaUsdPath) as UnityEditor.Importer.USD.UsdModularImporter;
            if (importer == null)
                return;

            var graph = AssetDatabase.LoadAssetAtPath<UnityEngine.Importer.ImporterGraph>(UsdImporterGraphPath);
            var needsReimport = false;
            if (!importer.isUsdRoot)
            {
                importer.isUsdRoot = true;
                needsReimport = true;
            }

            if (graph != null && importer.Graph.asset == null)
            {
                importer.Graph = graph;
                needsReimport = true;
            }

            if (!needsReimport && AssetDatabase.LoadAssetAtPath<GameObject>(KinovaUsdPath) != null)
                return;

            if (needsReimport)
            {
                EditorUtility.SetDirty(importer);
                importer.SaveAndReimport();
            }
        }

        static Transform EnsureChild(Transform parent, string name)
        {
            if (parent == null)
                return null;
            var existing = parent.Find(name);
            if (existing != null)
                return existing;
            var go = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(go, $"Create {name}");
            go.transform.SetParent(parent, false);
            return go.transform;
        }

        static void CreateZone(Transform parent, string name, Vector3 center, Color color, string label)
        {
            var zone = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(zone, $"Create {name}");
            zone.transform.SetParent(parent, false);
            zone.transform.position = center;

            CreatePlane(zone.transform, "Floor", Vector3.zero, ZoneSize, color);

            // Simple low walls on four sides (thin planes standing up)
            float half = ZoneSize * 0.5f;
            float wallH = 0.15f;
            float wallT = 0.08f;
            CreateBox(zone.transform, "Boundary_N", new Vector3(0f, wallH * 0.5f, half), new Vector3(ZoneSize, wallH, wallT), color * 1.2f);
            CreateBox(zone.transform, "Boundary_S", new Vector3(0f, wallH * 0.5f, -half), new Vector3(ZoneSize, wallH, wallT), color * 1.2f);
            CreateBox(zone.transform, "Boundary_E", new Vector3(half, wallH * 0.5f, 0f), new Vector3(wallT, wallH, ZoneSize), color * 1.2f);
            CreateBox(zone.transform, "Boundary_W", new Vector3(-half, wallH * 0.5f, 0f), new Vector3(wallT, wallH, ZoneSize), color * 1.2f);

            // Label marker (empty with name; TextMeshPro optional later)
            var marker = new GameObject($"Label_{label.Replace(' ', '_')}");
            Undo.RegisterCreatedObjectUndo(marker, "Create label marker");
            marker.transform.SetParent(zone.transform, false);
            marker.transform.localPosition = new Vector3(0f, 0.05f, -half + 0.6f);
        }

        static GameObject CreatePlane(Transform parent, string name, Vector3 localPos, float sizeMeters, Color color)
        {
            // Unity default Plane is 10x10 units
            var go = GameObject.CreatePrimitive(PrimitiveType.Plane);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = Vector3.one * (sizeMeters / 10f);
            ApplyColor(go, color);
            Undo.RegisterCreatedObjectUndo(go, $"Create {name}");
            return go;
        }

        static GameObject CreateBox(Transform parent, string name, Vector3 localPos, Vector3 scale, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            ApplyColor(go, color);
            Undo.RegisterCreatedObjectUndo(go, $"Create {name}");
            return go;
        }

        static void ApplyColor(GameObject go, Color color)
        {
            var renderer = go.GetComponent<MeshRenderer>();
            if (renderer == null)
                return;

            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Standard");
            var mat = new Material(shader);
            mat.color = color;
            renderer.sharedMaterial = mat;
        }
    }
}
#endif
