#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;
using XRPlayground.Policies;
using XRPlayground.Robots;
using XRPlayground.ROS;

namespace XRPlayground.ROS.Editor
{
    /// <summary>
    /// Places Station E (Spot stand/walk/follow) under 02_Locomotion.
    /// Menu: XRPlayground / Setup Spot Follow Station
    /// </summary>
    public static class SpotFollowSceneSetup
    {
        const string StationName = "Station_E_Spot";
        const string RobotName = "Spot_BD";
        const string BridgeName = "XR Bridge Spot";
        const string PanelName = "Spot Bridge World UI";
        const string SpotUsdPath =
            "Assets/_Project/Features/Robots/Spot/USD/spot.usd";
        const string SpotLocoOnnxPath =
            "Assets/_Project/Features/Policies/SpotLoco/policy.onnx";
        const string SpotLocoMetadataPath =
            "Assets/_Project/Features/Policies/SpotLoco/policy.json";
        const string UsdImporterGraphPath =
            "Packages/com.unity.importer.usd/Unity.Importer.USD.Editor/ImportGraph/usdImporter.asset";

        // Isaac Z-up default root (SPOT_CFG init pos)
        static readonly Vector3 BodyIsaacPos = new Vector3(0f, 0f, 0.5f);

        // Approximate standing offsets (Isaac) for procedural bind — names match USD prims.
        // isaacRel = offset from parent link (not from hip root). Visual scale is on *_viz children
        // so joint pivots stay unit-scale (nested mesh scales do not collapse).
        static readonly (string name, Vector3 isaacRel, Vector3 visualScale)[] LegParts =
        {
            ("hip", new Vector3(0f, 0f, 0f), new Vector3(0.08f, 0.08f, 0.08f)),
            ("uleg", new Vector3(0f, 0f, -0.12f), new Vector3(0.06f, 0.22f, 0.06f)),
            ("lleg", new Vector3(0f, 0f, -0.16f), new Vector3(0.05f, 0.22f, 0.05f)),
            ("foot", new Vector3(0f, 0f, -0.12f), new Vector3(0.10f, 0.04f, 0.06f)),
        };

        static readonly (string prefix, Vector3 hipIsaac)[] Legs =
        {
            ("fl", new Vector3(0.28f, 0.12f, 0.42f)),
            ("fr", new Vector3(0.28f, -0.12f, 0.42f)),
            ("hl", new Vector3(-0.28f, 0.12f, 0.42f)),
            ("hr", new Vector3(-0.28f, -0.12f, 0.42f)),
        };

        [MenuItem("XRPlayground/Setup Spot Follow Station")]
        public static void Setup()
        {
            var stations = FindLocomotionStations();
            if (stations == null)
            {
                Debug.LogError("XRPlayground: Run 'Setup Lab Zones' first (creates 02_Locomotion/Robot Stations).");
                return;
            }

            var existing = stations.Find(StationName);
            if (existing != null)
                Undo.DestroyObjectImmediate(existing.gameObject);

            var station = new GameObject(StationName);
            Undo.RegisterCreatedObjectUndo(station, StationName);
            station.transform.SetParent(stations, false);
            station.transform.localPosition = new Vector3(0f, 0f, 2f);
            station.transform.localRotation = Quaternion.identity;

            var robot = BuildSpotHierarchy(station.transform);
            if (robot == null)
                return;

            var bridgeGo = GameObject.Find(BridgeName) ?? new GameObject(BridgeName);
            Undo.RegisterCreatedObjectUndo(bridgeGo, BridgeName);
            var client = bridgeGo.GetComponent<RosTcpClient>() ?? Undo.AddComponent<RosTcpClient>(bridgeGo);
            client.host = "127.0.0.1";
            StationRegistry.ApplyTo(client, "spot_loco");
            client.autoConnect = false;
            var health = bridgeGo.GetComponent<BridgeHealthMonitor>() ?? Undo.AddComponent<BridgeHealthMonitor>(bridgeGo);
            health.stationId = "spot_loco";
            health.client = client;
            var hb = bridgeGo.GetComponent<RosHeartbeatPublisher>() ?? Undo.AddComponent<RosHeartbeatPublisher>(bridgeGo);
            hb.client = client;

            var map = robot.GetComponent<SpotLinkMap>() ?? Undo.AddComponent<SpotLinkMap>(robot);
            map.Rebuild();
            var robotFollower = robot.GetComponent<SpotLinkPoseFollower>()
                ?? Undo.AddComponent<SpotLinkPoseFollower>(robot);
            robotFollower.client = client;
            robotFollower.linkMap = map;
            robotFollower.envAnchor = station.transform;
            robotFollower.robotStateTopic = RosTopics.SpotRobotState;
            robotFollower.followingEnabled = true;

            var playerPub = station.GetComponent<SpotPlayerTargetPublisher>()
                ?? Undo.AddComponent<SpotPlayerTargetPublisher>(station);
            playerPub.client = client;
            playerPub.envAnchor = station.transform;
            playerPub.topic = RosTopics.SpotPlayerPose;
            playerPub.publishingEnabled = true;

            var offline = station.GetComponent<SpotOfflinePolicyController>()
                ?? Undo.AddComponent<SpotOfflinePolicyController>(station);
            var locoRunner = station.GetComponent<OnnxPolicyRunner>() ?? Undo.AddComponent<OnnxPolicyRunner>(station);
            locoRunner.modelAsset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(SpotLocoOnnxPath);
            locoRunner.policyMetadata = AssetDatabase.LoadAssetAtPath<TextAsset>(SpotLocoMetadataPath);
            locoRunner.expectedObsDim = SpotOfflinePolicyController.LocoObsDim;
            locoRunner.expectedActionDim = SpotOfflinePolicyController.LocoActionDim;

            var followGo = station.transform.Find("FollowPolicy")?.gameObject;
            if (followGo == null)
            {
                followGo = new GameObject("FollowPolicy");
                Undo.RegisterCreatedObjectUndo(followGo, "FollowPolicy");
                followGo.transform.SetParent(station.transform, false);
            }
            var followRunner = followGo.GetComponent<OnnxPolicyRunner>()
                ?? Undo.AddComponent<OnnxPolicyRunner>(followGo);
            followRunner.expectedObsDim = SpotOfflinePolicyController.FollowObsDim;
            followRunner.expectedActionDim = SpotOfflinePolicyController.FollowActionDim;

            var joints = station.GetComponent<OfflineJointDriver>() ?? Undo.AddComponent<OfflineJointDriver>(station);
            offline.locoRunner = locoRunner;
            health.policyRunner = locoRunner;
            offline.followRunner = followRunner;
            offline.jointDriver = joints;
            offline.envAnchor = station.transform;
            offline.linkMap = map;
            offline.AutoBindLinks();

            // Apply default joint pose once at setup (capture FK rest).
            joints.BindSpot(map, station.transform, force: true);
            joints.ApplyAnglesRadians(SpotOfflinePolicyController.DefaultJointPos);
            joints.Flush();

            EnsureEventSystem();
            var panel = EnsureWorldUi(station.transform.position);
            panel.client = client;
            panel.robotFollower = robotFollower;
            panel.playerPublisher = playerPub;
            panel.offlinePolicy = offline;

            locoRunner.captureActivations = true;
            PolicyNetworkPanelSetup.SetupSpot();

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log(
                "XRPlayground: Spot Station E ready (17 links, bridge :9094, offline follow+loco). " +
                "Assign Policies/SpotLoco/policy.onnx (+ optional SpotFollow) after train/export, " +
                "then Play → Start Offline Policy.",
                station);
        }

        static GameObject BuildSpotHierarchy(Transform station)
        {
            EnsureSpotUsdImported();

            var usdRoot = AssetDatabase.LoadAssetAtPath<GameObject>(SpotUsdPath);
            if (usdRoot == null)
            {
                Debug.LogError(
                    $"XRPlayground: Could not import Spot USD at '{SpotUsdPath}'. " +
                    "Enable Import (isUsdRoot) and re-run Setup Spot Follow Station.");
                return null;
            }

            var robot = (GameObject)PrefabUtility.InstantiatePrefab(usdRoot);
            Undo.RegisterCreatedObjectUndo(robot, "Place Spot USD");
            robot.name = RobotName;
            robot.transform.SetParent(station, false);
            // The imported USD is in its zero-joint bind pose; place the body at the
            // same Isaac standing root height used by the bridge and offline policy.
            robot.transform.localPosition = XrFrameConverter.IsaacPosToUnity(BodyIsaacPos);
            robot.transform.localRotation = Quaternion.identity;
            robot.transform.localScale = Vector3.one;
            XRPlayground.Robots.Editor.SpotMaterialSetup.ApplyTo(robot);

            return robot;
        }

        static void EnsureSpotUsdImported()
        {
            var importer = AssetImporter.GetAtPath(SpotUsdPath) as UnityEditor.Importer.USD.UsdModularImporter;
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

            if (needsReimport)
            {
                EditorUtility.SetDirty(importer);
                importer.SaveAndReimport();
            }
        }

        static Transform FindLocomotionStations()
        {
            var loco = GameObject.Find("02_Locomotion");
            if (loco == null)
            {
                var zones = GameObject.Find("Lab Zones");
                if (zones != null)
                {
                    var t = zones.transform.Find("02_Locomotion");
                    if (t != null)
                        loco = t.gameObject;
                }
            }
            if (loco == null)
                return null;
            var stations = loco.transform.Find("Robot Stations");
            if (stations == null)
            {
                var go = new GameObject("Robot Stations");
                Undo.RegisterCreatedObjectUndo(go, "Robot Stations");
                go.transform.SetParent(loco.transform, false);
                stations = go.transform;
            }
            return stations;
        }

        /// <summary>
        /// Joint/link pivot at unit scale (FK-safe) with a named mesh child for visibility.
        /// </summary>
        static GameObject CreateLink(Transform parent, string name, Vector3 localPos, Vector3 visualScale, Color color)
        {
            var go = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(go, name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            var viz = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Undo.RegisterCreatedObjectUndo(viz, name + "_viz");
            viz.name = name + "_viz";
            viz.transform.SetParent(go.transform, false);
            viz.transform.localPosition = Vector3.zero;
            viz.transform.localRotation = Quaternion.identity;
            viz.transform.localScale = visualScale;
            var col = viz.GetComponent<Collider>();
            if (col != null)
                Object.DestroyImmediate(col);
            var rend = viz.GetComponent<MeshRenderer>();
            if (rend != null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                rend.sharedMaterial = new Material(shader) { color = color };
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

        static SpotBridgePanel EnsureWorldUi(Vector3 stationPos)
        {
            var existing = GameObject.Find(PanelName);
            if (existing != null)
            {
                var p = existing.GetComponent<SpotBridgePanel>();
                if (p != null)
                    return p;
            }

            var root = new GameObject(PanelName);
            Undo.RegisterCreatedObjectUndo(root, PanelName);
            root.transform.position = stationPos + new Vector3(0.5f, 1.5f, -1.1f);
            root.transform.rotation = Quaternion.Euler(0f, 180f, 0f);

            var canvasGo = new GameObject("Canvas");
            canvasGo.transform.SetParent(root.transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 10f;
            canvasGo.AddComponent<GraphicRaycaster>();
            canvasGo.AddComponent<TrackedDeviceGraphicRaycaster>();
            var rt = canvasGo.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(720f, 420f);
            canvasGo.transform.localScale = Vector3.one * 0.0012f;

            var bg = CreateUiObject("Background", canvasGo.transform);
            bg.AddComponent<Image>().color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            StretchFull(bg.GetComponent<RectTransform>());

            var title = CreateText(canvasGo.transform, "Title", "Spot Follow", 34, FontStyle.Bold);
            SetRect(title.rectTransform, 40, -28, 640, 48);

            var status = CreateText(canvasGo.transform, "Status", "Bridge: disconnected", 20, FontStyle.Normal);
            SetRect(status.rectTransform, 40, -90, 640, 90);
            status.alignment = TextAnchor.UpperLeft;

            var mode = CreateText(canvasGo.transform, "ModeLabel", "Mode: Mirror Isaac", 22, FontStyle.Bold);
            SetRect(mode.rectTransform, 40, -190, 640, 36);

            var connectBtn = CreateButton(canvasGo.transform, "ConnectButton", "Connect bridge", new Color(0.15f, 0.45f, 0.75f));
            SetRect(connectBtn.GetComponent<RectTransform>(), 40, -250, 300, 56);

            var mirrorBtn = CreateButton(canvasGo.transform, "MirrorButton", "Mirror Isaac", new Color(0.25f, 0.45f, 0.28f));
            SetRect(mirrorBtn.GetComponent<RectTransform>(), 360, -250, 300, 56);

            var policyBtn = CreateButton(canvasGo.transform, "StartPolicy", "Start Offline Policy", new Color(0.45f, 0.20f, 0.55f));
            SetRect(policyBtn.GetComponent<RectTransform>(), 40, -330, 620, 56);

            var panel = root.AddComponent<SpotBridgePanel>();
            panel.statusText = status;
            panel.modeText = mode;
            panel.connectButton = connectBtn;
            panel.mirrorButton = mirrorBtn;
            panel.startPolicyButton = policyBtn;
            panel.connectButtonLabel = connectBtn.GetComponentInChildren<Text>();
            panel.startPolicyButtonLabel = policyBtn.GetComponentInChildren<Text>();
            return panel;
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
