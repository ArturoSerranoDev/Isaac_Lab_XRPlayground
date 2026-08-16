#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;
using XRPlayground.Robots;
using XRPlayground.Robots.Editor;
using XRPlayground.ROS;
using XRPlayground.Policies;

namespace XRPlayground.ROS.Editor
{
    /// <summary>
    /// Places Station B (UR10e + conveyor + sort table + trash) without destroying Kinova Station A.
    /// Link names match Isaac <c>names_ur10e.UR10E_LINK_NAMES</c>.
    /// Menu: XRPlayground / Setup Conveyor Color Station
    /// </summary>
    public static class ConveyorColorSceneSetup
    {
        const string StationName = "Station_B_Conveyor";
        const string RobotName = "UR10e_Robotiq";
        const string BridgeName = "XR Bridge Conveyor";
        const string PanelName = "Conveyor Bridge World UI";
        const string Ur10eUsdPath = "Assets/_Project/Features/Robots/UR10e/USD/ur10e.usd";
        // Meshes live in the base payload (same as Nucleus). Root ur10e.usd is a thin variant layer.
        const string Ur10eVisualUsdPath =
            "Assets/_Project/Features/Robots/UR10e/USD/configuration/ur10e_base.usd";
        const string UsdImporterGraphPath =
            "Packages/com.unity.importer.usd/Unity.Importer.USD.Editor/ImportGraph/usdImporter.asset";

        // Isaac Z-up (x,y,z) → Unity Y-up (x,z,y). Matches conveyor_color_env_cfg.
        static readonly Vector3 BeltIsaacCenter = new Vector3(0.55f, 0f, 0.40f);
        static readonly Vector3 BeltIsaacSize = new Vector3(0.30f, 1.40f, 0.04f);
        // Sort table beside trash at belt end (correct / target color)
        static readonly Vector3 SortTableIsaacCenter = new Vector3(0.20f, 0.78f, 0.405f);
        static readonly Vector3 SortTableIsaacSize = new Vector3(0.36f, 0.36f, 0.03f);
        // Trash at end of conveyor (reject / non-target)
        static readonly Vector3 TrashIsaacCenter = new Vector3(0.55f, 0.78f, 0.405f);
        static readonly Vector3 TrashIsaacSize = new Vector3(0.34f, 0.28f, 0.03f);
        // Robot base offset for arm clearance vs belt + tables
        static readonly Vector3 RobotIsaacPos = new Vector3(-0.22f, -0.10f, 0f);

        [MenuItem("XRPlayground/Setup Conveyor Color Station")]
        public static void Setup()
        {
            var stations = FindRobotStations();
            if (stations == null)
            {
                Debug.LogError("XRPlayground: Run 'Setup Lab Zones' first (creates 01_Manipulation/Robot Stations).");
                return;
            }

            var existing = stations.Find(StationName);
            if (existing != null)
                Undo.DestroyObjectImmediate(existing.gameObject);

            var station = new GameObject(StationName);
            Undo.RegisterCreatedObjectUndo(station, StationName);
            station.transform.SetParent(stations, false);
            station.transform.localPosition = new Vector3(3f, 0f, -2f);
            station.transform.localRotation = Quaternion.identity;

            var robot = PlaceRobot(station.transform);
            if (robot == null)
            {
                Undo.DestroyObjectImmediate(station);
                return;
            }

            Ur10eMaterialSetup.ApplyTo(robot);

            // Belt: Isaac center/size → Unity local (x,z,y)
            var beltUnityPos = XrFrameConverter.IsaacPosToUnity(BeltIsaacCenter);
            var beltUnityScale = new Vector3(BeltIsaacSize.x, BeltIsaacSize.z, BeltIsaacSize.y);
            CreateCube(station.transform, "ConveyorBelt", beltUnityPos, beltUnityScale, new Color(0.25f, 0.25f, 0.28f));

            // Side sort table (correct / target color) — green
            var sortUnityPos = XrFrameConverter.IsaacPosToUnity(SortTableIsaacCenter);
            var sortUnityScale = new Vector3(SortTableIsaacSize.x, SortTableIsaacSize.z, SortTableIsaacSize.y);
            CreateCube(station.transform, "SortTable", sortUnityPos, sortUnityScale, new Color(0.20f, 0.55f, 0.30f));

            // End trash / reject platform — reddish brown
            var trashUnityPos = XrFrameConverter.IsaacPosToUnity(TrashIsaacCenter);
            var trashUnityScale = new Vector3(TrashIsaacSize.x, TrashIsaacSize.z, TrashIsaacSize.y);
            CreateCube(station.transform, "TrashPlatform", trashUnityPos, trashUnityScale, new Color(0.45f, 0.18f, 0.12f));

            var slotsRoot = new GameObject("ObjectSlots");
            Undo.RegisterCreatedObjectUndo(slotsRoot, "ObjectSlots");
            slotsRoot.transform.SetParent(station.transform, false);
            var slots = new Transform[4];
            Color[] cols =
            {
                new Color(0.9f, 0.15f, 0.12f),
                new Color(0.15f, 0.75f, 0.25f),
                new Color(0.15f, 0.35f, 0.9f),
                new Color(0.9f, 0.15f, 0.12f),
            };
            for (int i = 0; i < 4; i++)
            {
                var cube = CreateCube(slotsRoot.transform, $"Cube_{i}", new Vector3(0f, -1f, 0f), Vector3.one * 0.05f, cols[i]);
                cube.SetActive(false);
                slots[i] = cube.transform;
            }

            var bridgeGo = GameObject.Find(BridgeName) ?? new GameObject(BridgeName);
            Undo.RegisterCreatedObjectUndo(bridgeGo, BridgeName);
            var client = bridgeGo.GetComponent<RosTcpClient>() ?? Undo.AddComponent<RosTcpClient>(bridgeGo);
            client.host = "127.0.0.1";
            StationRegistry.ApplyTo(client, "conveyor_color");
            client.autoConnect = false;
            var health = bridgeGo.GetComponent<BridgeHealthMonitor>() ?? Undo.AddComponent<BridgeHealthMonitor>(bridgeGo);
            health.stationId = "conveyor_color";
            health.client = client;
            var hb = bridgeGo.GetComponent<RosHeartbeatPublisher>() ?? Undo.AddComponent<RosHeartbeatPublisher>(bridgeGo);
            hb.client = client;

            WireRobotFollower(robot, station.transform, client);

            var objFollower = station.GetComponent<ConveyorObjectFollower>() ?? Undo.AddComponent<ConveyorObjectFollower>(station);
            objFollower.client = client;
            objFollower.envAnchor = station.transform;
            objFollower.objectSlots = slots;

            var spawnPub = station.GetComponent<ConveyorSpawnPublisher>() ?? Undo.AddComponent<ConveyorSpawnPublisher>(station);
            spawnPub.client = client;
            spawnPub.envAnchor = station.transform;
            spawnPub.publishingEnabled = false;

            var offline = station.GetComponent<ConveyorOfflinePolicyController>()
                ?? Undo.AddComponent<ConveyorOfflinePolicyController>(station);
            var runner = station.GetComponent<OnnxPolicyRunner>() ?? Undo.AddComponent<OnnxPolicyRunner>(station);
            runner.expectedObsDim = ConveyorOfflinePolicyController.ObsDim;
            runner.expectedActionDim = ConveyorOfflinePolicyController.ActionDim;
            var joints = station.GetComponent<OfflineJointDriver>() ?? Undo.AddComponent<OfflineJointDriver>(station);
            offline.policyRunner = runner;
            offline.jointDriver = joints;
            offline.envAnchor = station.transform;
            offline.linkMap = robot.GetComponent<RobotLinkMap>();
            offline.cubeSlots = slots;
            offline.binAnchor = station.transform.Find("SortTable");
            offline.AutoBindLinks();
            health.policyRunner = runner;

            EnsureEventSystem();
            var panel = EnsureWorldUi(station.transform.position);
            panel.client = client;
            panel.spawnPublisher = spawnPub;
            panel.objectFollower = objFollower;
            panel.robotFollower = robot.GetComponent<RobotLinkPoseFollower>();
            panel.offlinePolicy = offline;
            panel.healthMonitor = health;
            panel.mode = ConveyorBridgeUiMode.MirrorIsaac;

            runner.captureActivations = true;
            PolicyNetworkPanelSetup.SetupConveyor();

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log(
                "XRPlayground: Conveyor Color Station B ready (UR10e + Robotiq, bridge :9091, offline ONNX + Network panel). " +
                "Assign Assets/_Project/Features/Policies/Conveyor/policy.onnx to OnnxPolicyRunner.modelAsset, " +
                "then Play → Start Offline Policy.");
        }

        [MenuItem("XRPlayground/Fix Conveyor Robot Mirror")]
        public static void FixConveyorRobotMirror()
        {
            var station = GameObject.Find(StationName);
            var robot = GameObject.Find(RobotName);
            var bridgeGo = GameObject.Find(BridgeName);
            if (station == null || robot == null || bridgeGo == null)
            {
                Debug.LogError(
                    "XRPlayground: Need Station_B_Conveyor + UR10e_Robotiq + XR Bridge Conveyor. " +
                    "Run 'Setup Conveyor Color Station' first.");
                return;
            }

            if (PrefabUtility.IsPartOfPrefabInstance(robot))
            {
                PrefabUtility.UnpackPrefabInstance(
                    robot, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
                Debug.Log("XRPlayground: Unpacked UR10e prefab instance (allows streamed poses).");
            }

            var client = bridgeGo.GetComponent<RosTcpClient>();
            if (client == null)
            {
                Debug.LogError("XRPlayground: RosTcpClient missing on XR Bridge Conveyor.");
                return;
            }

            StationRegistry.ApplyTo(client, "conveyor_color");
            WireRobotFollower(robot, station.transform, client);

            var panel = Object.FindAnyObjectByType<ConveyorBridgePanel>();
            if (panel != null)
            {
                panel.client = client;
                panel.robotFollower = robot.GetComponent<RobotLinkPoseFollower>();
                var offline = station.GetComponent<ConveyorOfflinePolicyController>();
                if (offline != null)
                    panel.offlinePolicy = offline;
            }

            var map = robot.GetComponent<RobotLinkMap>();
            string sample = "?";
            if (map != null && map.TryGet("shoulder_link", out var sh) && sh != null)
                sample = GetPath(sh, robot.transform);

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log(
                "XRPlayground: Conveyor robot mirror rewired (prefer subtree ur10e, envAnchor=Station_B). " +
                $"shoulder_link → {sample}. Play → Connect → check Applied N/N links.");
        }

        static string GetPath(Transform t, Transform root)
        {
            var parts = new List<string>();
            while (t != null)
            {
                parts.Add(t.name);
                if (t == root)
                    break;
                t = t.parent;
            }
            parts.Reverse();
            return string.Join("/", parts);
        }

        static void WireRobotFollower(GameObject robot, Transform station, RosTcpClient client)
        {
            var map = robot.GetComponent<RobotLinkMap>() ?? Undo.AddComponent<RobotLinkMap>(robot);
            map.preferredSubtree = "ur10e";
            map.Rebuild();
            var robotFollower = robot.GetComponent<RobotLinkPoseFollower>() ?? Undo.AddComponent<RobotLinkPoseFollower>(robot);
            robotFollower.client = client;
            robotFollower.linkMap = map;
            robotFollower.envAnchor = station;
            robotFollower.robotStateTopic = RosTopics.ConveyorRobotState;
            robotFollower.followingEnabled = true;
            robotFollower.logFirstApply = true;
            robotFollower.calibrateVisualFrames = false;
            robotFollower.CaptureUnityBindPose();
        }

        static Transform FindRobotStations()
        {
            var manip = GameObject.Find("01_Manipulation");
            if (manip == null)
            {
                var zones = GameObject.Find("Lab Zones");
                if (zones != null)
                {
                    var t = zones.transform.Find("01_Manipulation");
                    if (t != null)
                        manip = t.gameObject;
                }
            }
            if (manip == null)
                return null;
            var stations = manip.transform.Find("Robot Stations");
            if (stations == null)
            {
                var go = new GameObject("Robot Stations");
                Undo.RegisterCreatedObjectUndo(go, "Robot Stations");
                go.transform.SetParent(manip.transform, false);
                stations = go.transform;
            }
            return stations;
        }

        static GameObject PlaceRobot(Transform parent)
        {
            EnsureUr10eUsdImported(Ur10eUsdPath);
            EnsureUr10eUsdImported(Ur10eVisualUsdPath);

            // Prefer the base payload that contains link meshes (verified in Unity import).
            var usd = AssetDatabase.LoadAssetAtPath<GameObject>(Ur10eVisualUsdPath);
            string usedPath = Ur10eVisualUsdPath;
            if (usd == null)
            {
                usd = AssetDatabase.LoadAssetAtPath<GameObject>(Ur10eUsdPath);
                usedPath = Ur10eUsdPath;
            }

            if (usd == null)
            {
                Debug.LogError(
                    "XRPlayground: Missing Isaac UR10e USD meshes. Expected:\n" +
                    $"  {Ur10eVisualUsdPath}\n" +
                    $"  (and optionally {Ur10eUsdPath})\n" +
                    "Copy from Nucleus Assets/Isaac/6.0/Isaac/Robots/UniversalRobots/ur10e/ " +
                    "or repo assets/usd/ur10e/. No cube placeholders.");
                return null;
            }

            var robot = (GameObject)PrefabUtility.InstantiatePrefab(usd);
            robot.name = RobotName;
            Undo.RegisterCreatedObjectUndo(robot, RobotName);
            // Unpack so link transforms are free (USD prefab drivers can freeze streamed poses).
            if (PrefabUtility.IsPartOfPrefabInstance(robot))
            {
                PrefabUtility.UnpackPrefabInstance(
                    robot, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
            }
            robot.transform.SetParent(parent, false);
            robot.transform.localPosition = XrFrameConverter.IsaacPosToUnity(RobotIsaacPos);
            robot.transform.localRotation = Quaternion.identity;
            robot.transform.localScale = Vector3.one;
            int meshes = robot.GetComponentsInChildren<MeshFilter>(true).Length;
            Debug.Log($"XRPlayground: Instantiated UR10e from '{usedPath}' (MeshFilters={meshes}, unpacked).");
            return robot;
        }

        static void EnsureUr10eUsdImported(string usdPath)
        {
            var importer = AssetImporter.GetAtPath(usdPath) as UnityEditor.Importer.USD.UsdModularImporter;
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

            if (!needsReimport && AssetDatabase.LoadAssetAtPath<GameObject>(usdPath) != null)
                return;

            if (needsReimport)
            {
                EditorUtility.SetDirty(importer);
                importer.SaveAndReimport();
            }
        }

        static GameObject CreateCube(Transform parent, string name, Vector3 localPos, Vector3 scale, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Undo.RegisterCreatedObjectUndo(go, name);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            var col = go.GetComponent<Collider>();
            if (col != null)
                Object.DestroyImmediate(col);
            var rend = go.GetComponent<MeshRenderer>();
            if (rend != null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                var mat = new Material(shader) { color = color };
                rend.sharedMaterial = mat;
            }
            return go;
        }

        static void EnsureEventSystem()
        {
            if (Object.FindAnyObjectByType<EventSystem>() != null)
                return;
            var es = new GameObject("EventSystem");
            Undo.RegisterCreatedObjectUndo(es, "EventSystem");
            es.AddComponent<EventSystem>();
            es.AddComponent<StandaloneInputModule>();
        }

        static ConveyorBridgePanel EnsureWorldUi(Vector3 stationPos)
        {
            var existing = GameObject.Find(PanelName);
            if (existing != null)
            {
                var p = existing.GetComponent<ConveyorBridgePanel>();
                if (p != null)
                {
                    EnsureStartPolicyButton(existing.transform, p);
                    return p;
                }
            }

            var root = new GameObject(PanelName);
            Undo.RegisterCreatedObjectUndo(root, PanelName);
            root.transform.position = stationPos + new Vector3(0.4f, 1.4f, -0.9f);
            root.transform.rotation = Quaternion.Euler(0f, 180f, 0f);

            var canvasGo = new GameObject("Canvas");
            canvasGo.transform.SetParent(root.transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 10f;
            canvasGo.AddComponent<GraphicRaycaster>();
            canvasGo.AddComponent<TrackedDeviceGraphicRaycaster>();
            var rt = canvasGo.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(860f, 720f);
            canvasGo.transform.localScale = Vector3.one * 0.0012f;

            var bg = CreateUiObject("Background", canvasGo.transform);
            bg.AddComponent<Image>().color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            StretchFull(bg.GetComponent<RectTransform>());

            var title = CreateText(canvasGo.transform, "Title", "Conveyor Color", 34, FontStyle.Bold);
            SetRect(title.rectTransform, 40, -28, 780, 48);

            var status = CreateText(canvasGo.transform, "Status", "Bridge: disconnected", 20, FontStyle.Normal);
            SetRect(status.rectTransform, 40, -90, 780, 110);
            status.alignment = TextAnchor.UpperLeft;

            var mode = CreateText(canvasGo.transform, "ModeLabel", "Mode: Mirror Isaac", 22, FontStyle.Bold);
            SetRect(mode.rectTransform, 40, -210, 780, 36);

            var target = CreateText(canvasGo.transform, "TargetColor", "Target: —", 22, FontStyle.Bold);
            SetRect(target.rectTransform, 40, -250, 780, 36);

            var connectBtn = CreateButton(canvasGo.transform, "ConnectButton", "Connect bridge", new Color(0.15f, 0.45f, 0.75f));
            SetRect(connectBtn.GetComponent<RectTransform>(), 40, -310, 360, 64);

            var mirrorBtn = CreateButton(canvasGo.transform, "MirrorButton", "Mirror Isaac", new Color(0.25f, 0.45f, 0.28f));
            SetRect(mirrorBtn.GetComponent<RectTransform>(), 40, -390, 360, 64);

            var awaitBtn = CreateButton(canvasGo.transform, "AwaitSpawnButton", "Await Unity spawn", new Color(0.55f, 0.35f, 0.15f));
            SetRect(awaitBtn.GetComponent<RectTransform>(), 420, -390, 380, 64);

            var redBtn = CreateButton(canvasGo.transform, "SpawnRed", "Spawn RED", new Color(0.75f, 0.2f, 0.15f));
            SetRect(redBtn.GetComponent<RectTransform>(), 40, -480, 240, 56);
            var greenBtn = CreateButton(canvasGo.transform, "SpawnGreen", "Spawn GREEN", new Color(0.2f, 0.55f, 0.25f));
            SetRect(greenBtn.GetComponent<RectTransform>(), 300, -480, 240, 56);
            var blueBtn = CreateButton(canvasGo.transform, "SpawnBlue", "Spawn BLUE", new Color(0.2f, 0.35f, 0.75f));
            SetRect(blueBtn.GetComponent<RectTransform>(), 560, -480, 240, 56);

            var policyBtn = CreateButton(canvasGo.transform, "StartPolicy", "Start Offline Policy", new Color(0.45f, 0.20f, 0.55f));
            SetRect(policyBtn.GetComponent<RectTransform>(), 40, -560, 760, 56);

            var panel = root.AddComponent<ConveyorBridgePanel>();
            panel.statusText = status;
            panel.modeText = mode;
            panel.targetColorText = target;
            panel.connectButton = connectBtn;
            panel.mirrorButton = mirrorBtn;
            panel.awaitSpawnButton = awaitBtn;
            panel.spawnRedButton = redBtn;
            panel.spawnGreenButton = greenBtn;
            panel.spawnBlueButton = blueBtn;
            panel.startPolicyButton = policyBtn;
            panel.connectButtonLabel = connectBtn.GetComponentInChildren<Text>();
            panel.startPolicyButtonLabel = policyBtn.GetComponentInChildren<Text>();
            return panel;
        }

        static void EnsureStartPolicyButton(Transform panelRoot, ConveyorBridgePanel panel)
        {
            if (panel.startPolicyButton != null)
                return;
            var canvas = panelRoot.Find("Canvas");
            if (canvas == null)
                return;
            var policyBtn = CreateButton(canvas, "StartPolicy", "Start Offline Policy", new Color(0.45f, 0.20f, 0.55f));
            SetRect(policyBtn.GetComponent<RectTransform>(), 40, -560, 760, 56);
            panel.startPolicyButton = policyBtn;
            panel.startPolicyButtonLabel = policyBtn.GetComponentInChildren<Text>();
            Undo.RegisterCreatedObjectUndo(policyBtn.gameObject, "Start Policy Button");
        }

        static GameObject CreateUiObject(string name, Transform parent)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<RectTransform>();
            return go;
        }

        static Text CreateText(Transform parent, string name, string value, int size, FontStyle style)
        {
            var go = CreateUiObject(name, parent);
            var text = go.AddComponent<Text>();
            text.text = value;
            text.fontSize = size;
            text.fontStyle = style;
            text.color = Color.white;
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (text.font == null)
                text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return text;
        }

        static Button CreateButton(Transform parent, string name, string label, Color color)
        {
            var go = CreateUiObject(name, parent);
            var img = go.AddComponent<Image>();
            img.color = color;
            var btn = go.AddComponent<Button>();
            btn.targetGraphic = img;
            var text = CreateText(go.transform, "Label", label, 20, FontStyle.Bold);
            text.alignment = TextAnchor.MiddleCenter;
            StretchFull(text.rectTransform);
            return btn;
        }

        static void StretchFull(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }

        static void SetRect(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.anchoredPosition = new Vector2(x, y);
            rt.sizeDelta = new Vector2(w, h);
        }
    }
}
#endif
