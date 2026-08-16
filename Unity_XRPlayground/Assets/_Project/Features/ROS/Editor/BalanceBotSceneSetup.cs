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
    /// Places Station D (2-DOF balance tray + balls). Link names match Isaac names_balance_bot.
    /// Menu: XRPlayground / Setup Balance Bot Station
    /// </summary>
    public static class BalanceBotSceneSetup
    {
        const string StationName = "Station_D_BalanceBot";
        const string RobotName = "BalanceBot_Tray";
        const string BridgeName = "XR Bridge BalanceBot";
        const string PanelName = "BalanceBot Bridge World UI";

        // Isaac Z-up (matches balance_bot_env_cfg)
        static readonly Vector3 PedestalIsaacCenter = new Vector3(0f, 0f, 0.35f);
        static readonly Vector3 PedestalIsaacSize = new Vector3(0.12f, 0.12f, 0.70f);
        static readonly Vector3 TrayIsaacCenter = new Vector3(0f, 0f, 0.75f);
        static readonly Vector3 TrayIsaacSize = new Vector3(0.60f, 0.60f, 0.02f); // 1.5× prior
        const float BallRadius = 0.035f; // half prior; mass kept heavy in Isaac cfg

        [MenuItem("XRPlayground/Setup Balance Bot Station")]
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
            station.transform.localPosition = new Vector3(6f, 0f, -2f);
            station.transform.localRotation = Quaternion.identity;

            var robot = BuildTrayHierarchy(station.transform);
            var balls = BuildBalls(station.transform);

            var bridgeGo = GameObject.Find(BridgeName) ?? new GameObject(BridgeName);
            Undo.RegisterCreatedObjectUndo(bridgeGo, BridgeName);
            var client = bridgeGo.GetComponent<RosTcpClient>() ?? Undo.AddComponent<RosTcpClient>(bridgeGo);
            client.host = "127.0.0.1";
            StationRegistry.ApplyTo(client, "balance_bot");
            client.autoConnect = false;
            var health = bridgeGo.GetComponent<BridgeHealthMonitor>() ?? Undo.AddComponent<BridgeHealthMonitor>(bridgeGo);
            health.stationId = "balance_bot";
            health.client = client;
            var hb = bridgeGo.GetComponent<RosHeartbeatPublisher>() ?? Undo.AddComponent<RosHeartbeatPublisher>(bridgeGo);
            hb.client = client;

            var map = robot.GetComponent<BalanceBotLinkMap>() ?? Undo.AddComponent<BalanceBotLinkMap>(robot);
            map.Rebuild();
            var robotFollower = robot.GetComponent<BalanceBotLinkPoseFollower>()
                ?? Undo.AddComponent<BalanceBotLinkPoseFollower>(robot);
            robotFollower.client = client;
            robotFollower.linkMap = map;
            robotFollower.envAnchor = station.transform;
            robotFollower.robotStateTopic = RosTopics.BalanceBotRobotState;
            robotFollower.followingEnabled = true;

            var ballFollower = station.GetComponent<BalanceBotBallFollower>()
                ?? Undo.AddComponent<BalanceBotBallFollower>(station);
            ballFollower.client = client;
            ballFollower.envAnchor = station.transform;
            ballFollower.ballSlots = balls;

            var offline = station.GetComponent<BalanceBotOfflinePolicyController>()
                ?? Undo.AddComponent<BalanceBotOfflinePolicyController>(station);
            var runner = station.GetComponent<OnnxPolicyRunner>() ?? Undo.AddComponent<OnnxPolicyRunner>(station);
            runner.expectedObsDim = BalanceBotOfflinePolicyController.ObsDim;
            runner.expectedActionDim = BalanceBotOfflinePolicyController.ActionDim;
            var joints = station.GetComponent<OfflineJointDriver>() ?? Undo.AddComponent<OfflineJointDriver>(station);
            offline.policyRunner = runner;
            health.policyRunner = runner;
            offline.jointDriver = joints;
            offline.envAnchor = station.transform;
            offline.linkMap = map;
            offline.ballSlots = balls;
            offline.AutoBindLinks();

            EnsureEventSystem();
            var panel = EnsureWorldUi(station.transform.position);
            panel.client = client;
            panel.ballFollower = ballFollower;
            panel.robotFollower = robotFollower;
            panel.offlinePolicy = offline;

            runner.captureActivations = true;
            PolicyNetworkPanelSetup.SetupBalanceBot();

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log(
                "XRPlayground: Balance Bot Station D ready (2-DOF tray, bridge :9093, offline ONNX + Network panel). " +
                "Assign Assets/_Project/Features/Policies/BalanceBot/policy.onnx after first train/export, " +
                "then Play → Start Offline Policy.");
        }

        static GameObject BuildTrayHierarchy(Transform station)
        {
            var robot = new GameObject(RobotName);
            Undo.RegisterCreatedObjectUndo(robot, RobotName);
            robot.transform.SetParent(station, false);
            robot.transform.localPosition = Vector3.zero;
            robot.transform.localRotation = Quaternion.identity;

            // base_link = pedestal
            var baseUnityPos = XrFrameConverter.IsaacPosToUnity(PedestalIsaacCenter);
            var baseUnityScale = new Vector3(PedestalIsaacSize.x, PedestalIsaacSize.z, PedestalIsaacSize.y);
            var baseLink = CreateCube(robot.transform, "base_link", baseUnityPos, baseUnityScale,
                new Color(0.35f, 0.35f, 0.38f));

            // roll_link at tray gimbal (empty visual pivot)
            var roll = new GameObject("roll_link");
            Undo.RegisterCreatedObjectUndo(roll, "roll_link");
            roll.transform.SetParent(robot.transform, false);
            roll.transform.localPosition = XrFrameConverter.IsaacPosToUnity(TrayIsaacCenter);
            roll.transform.localRotation = Quaternion.identity;

            // tray_link plate (child of roll so FK chain matches Isaac)
            var trayUnityScale = new Vector3(TrayIsaacSize.x, TrayIsaacSize.z, TrayIsaacSize.y);
            var tray = CreateCube(roll.transform, "tray_link", Vector3.zero, trayUnityScale,
                new Color(0.55f, 0.42f, 0.28f));
            _ = baseLink;
            _ = tray;
            return robot;
        }

        static Transform[] BuildBalls(Transform station)
        {
            var slots = new Transform[2];
            Color[] cols =
            {
                new Color(0.92f, 0.55f, 0.12f),
                new Color(0.20f, 0.55f, 0.85f),
            };
            // Rest pose on tray (Isaac tray z=0.75 → Unity Y). Visible by default so the
            // station is not an empty plate until Offline Policy / ROS starts.
            var trayTopIsaac = new Vector3(
                TrayIsaacCenter.x,
                TrayIsaacCenter.y,
                TrayIsaacCenter.z + 0.5f * TrayIsaacSize.z + BallRadius + 0.02f);
            Vector3[] isaacOffsets =
            {
                new Vector3(0f, 0f, 0f),
                new Vector3(0.09f, 0.06f, 0f),
            };
            for (int i = 0; i < 2; i++)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                Undo.RegisterCreatedObjectUndo(go, $"Ball_{i}");
                go.name = $"Ball_{i}";
                go.transform.SetParent(station, false);
                go.transform.localScale = Vector3.one * (BallRadius * 2f);
                go.transform.localPosition = XrFrameConverter.IsaacPosToUnity(trayTopIsaac + isaacOffsets[i]);
                var col = go.GetComponent<Collider>();
                if (col != null)
                    Object.DestroyImmediate(col);
                var rend = go.GetComponent<MeshRenderer>();
                if (rend != null)
                {
                    var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                    rend.sharedMaterial = new Material(shader) { color = cols[i] };
                }
                go.SetActive(true);
                slots[i] = go.transform;
            }
            return slots;
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

        static BalanceBotBridgePanel EnsureWorldUi(Vector3 stationPos)
        {
            var existing = GameObject.Find(PanelName);
            if (existing != null)
            {
                var p = existing.GetComponent<BalanceBotBridgePanel>();
                if (p != null)
                    return p;
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
            rt.sizeDelta = new Vector2(720f, 420f);
            canvasGo.transform.localScale = Vector3.one * 0.0012f;

            var bg = CreateUiObject("Background", canvasGo.transform);
            bg.AddComponent<Image>().color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            StretchFull(bg.GetComponent<RectTransform>());

            var title = CreateText(canvasGo.transform, "Title", "Balance Bot", 34, FontStyle.Bold);
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

            var panel = root.AddComponent<BalanceBotBridgePanel>();
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
