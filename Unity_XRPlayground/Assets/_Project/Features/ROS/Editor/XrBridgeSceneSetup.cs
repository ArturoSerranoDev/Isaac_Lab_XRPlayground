#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;
using XRPlayground.Robots;
using XRPlayground.ROS;

namespace XRPlayground.ROS.Editor
{
    /// <summary>
    /// Builds classic uGUI (Canvas) world-space bridge panel near the Kinova.
    /// Menu: XRPlayground / Setup XR Bridge Scene
    /// </summary>
    public static class XrBridgeSceneSetup
    {
        const string RobotName = "Kinova_Jaco2_j2n7s300";
        const string PanelName = "XR Bridge World UI";

        [MenuItem("XRPlayground/Setup XR Bridge Scene")]
        public static void Setup()
        {
            var bridgeGo = GameObject.Find("XR Bridge") ?? new GameObject("XR Bridge");
            Undo.RegisterCreatedObjectUndo(bridgeGo, "Create XR Bridge");

            var client = bridgeGo.GetComponent<RosTcpClient>() ?? Undo.AddComponent<RosTcpClient>(bridgeGo);
            client.host = "127.0.0.1";
            client.port = 9090;
            client.autoConnect = false; // user connects from world UI

            var hb = bridgeGo.GetComponent<RosHeartbeatPublisher>() ?? Undo.AddComponent<RosHeartbeatPublisher>(bridgeGo);
            hb.client = client;

            var robot = GameObject.Find(RobotName);
            KinovaLinkPoseFollower follower = null;
            if (robot == null)
            {
                Debug.LogError($"XRPlayground: '{RobotName}' not found. Run Setup Lab Zones first.");
            }
            else
            {
                var map = robot.GetComponent<KinovaLinkMap>() ?? Undo.AddComponent<KinovaLinkMap>(robot);
                map.Rebuild();
                follower = robot.GetComponent<KinovaLinkPoseFollower>() ?? Undo.AddComponent<KinovaLinkPoseFollower>(robot);
                follower.client = client;
                follower.linkMap = map;
                follower.rootOffset = robot.transform.position;
                follower.applyEeOnly = false;
            }

            var ball = EnsureBall();
            var pub = ball.GetComponent<BallStatePublisher>() ?? Undo.AddComponent<BallStatePublisher>(ball);
            pub.client = client;
            pub.ballRoot = ball.transform;
            pub.ballBody = ball.GetComponent<Rigidbody>();
            pub.isaacRootOffset = robot != null ? robot.transform.position : Vector3.zero;
            pub.publishWhileHeld = true;
            pub.publishingEnabled = false; // default mirror mode

            EnsureEventSystem();
            var panel = EnsureWorldUi(robot != null ? robot.transform.position : Vector3.zero);
            panel.client = client;
            panel.ballPublisher = pub;
            panel.robotFollower = follower;
            panel.mode = XrBridgeUiMode.MirrorIsaac;

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log("XRPlayground: XR Bridge + classic world-space UI ready near the robot.");
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

        static XrBridgePanel EnsureWorldUi(Vector3 robotPos)
        {
            var existing = GameObject.Find(PanelName);
            if (existing != null)
            {
                var p = existing.GetComponent<XrBridgePanel>();
                if (p != null)
                    return p;
            }

            var root = new GameObject(PanelName);
            Undo.RegisterCreatedObjectUndo(root, PanelName);
            root.transform.position = robotPos + new Vector3(0.35f, 1.35f, -0.85f);
            root.transform.rotation = Quaternion.Euler(0f, 180f, 0f);

            var canvasGo = new GameObject("Canvas");
            canvasGo.transform.SetParent(root.transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 10f;
            canvasGo.AddComponent<GraphicRaycaster>();
            // XR poke / ray UI
            canvasGo.AddComponent<TrackedDeviceGraphicRaycaster>();

            var rt = canvasGo.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(800f, 520f);
            canvasGo.transform.localScale = Vector3.one * 0.0012f;

            // Background panel
            var bg = CreateUiObject("Background", canvasGo.transform);
            var bgImg = bg.AddComponent<Image>();
            bgImg.color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            StretchFull(bg.GetComponent<RectTransform>());

            var title = CreateText(canvasGo.transform, "Title", "XR Bridge", 36, FontStyle.Bold);
            SetRect(title.rectTransform, 40, -30, 720, 50);

            var status = CreateText(canvasGo.transform, "Status", "Bridge: disconnected", 22, FontStyle.Normal);
            SetRect(status.rectTransform, 40, -100, 720, 140);
            status.alignment = TextAnchor.UpperLeft;

            var mode = CreateText(canvasGo.transform, "ModeLabel", "Mode: Mirror Isaac", 24, FontStyle.Bold);
            SetRect(mode.rectTransform, 40, -250, 720, 40);

            var connectBtn = CreateButton(canvasGo.transform, "ConnectButton", "Connect bridge", new Color(0.15f, 0.45f, 0.75f));
            SetRect(connectBtn.GetComponent<RectTransform>(), 40, -320, 340, 70);

            var mirrorBtn = CreateButton(canvasGo.transform, "MirrorButton", "Mirror Isaac", new Color(0.25f, 0.45f, 0.28f));
            SetRect(mirrorBtn.GetComponent<RectTransform>(), 40, -410, 340, 70);

            var awaitBtn = CreateButton(canvasGo.transform, "AwaitThrowButton", "Await player throw", new Color(0.55f, 0.35f, 0.15f));
            SetRect(awaitBtn.GetComponent<RectTransform>(), 400, -410, 360, 70);

            var panel = root.AddComponent<XrBridgePanel>();
            panel.statusText = status;
            panel.modeText = mode;
            panel.connectButton = connectBtn;
            panel.mirrorButton = mirrorBtn;
            panel.awaitThrowButton = awaitBtn;
            panel.connectButtonLabel = connectBtn.GetComponentInChildren<Text>();
            return panel;
        }

        static GameObject EnsureBall()
        {
            var existing = GameObject.Find("Ball");
            if (existing != null)
                return existing;

            var grab = GameObject.Find("Grab Cube");
            if (grab != null)
            {
                Undo.RecordObject(grab, "Rename Grab Cube to Ball");
                grab.name = "Ball";
                return grab;
            }

            var ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Undo.RegisterCreatedObjectUndo(ball, "Create Ball");
            ball.name = "Ball";
            ball.transform.position = new Vector3(0.4f, 1.1f, 0.5f);
            ball.transform.localScale = Vector3.one * 0.0825f; // ~75% of prior visual

            var rb = ball.GetComponent<Rigidbody>() ?? ball.AddComponent<Rigidbody>();
            rb.mass = 0.08f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;

            var grabable = ball.AddComponent<UnityEngine.XR.Interaction.Toolkit.Interactables.XRGrabInteractable>();
            grabable.movementType = UnityEngine.XR.Interaction.Toolkit.Interactables.XRGrabInteractable.MovementType.Instantaneous;
            return ball;
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

            var text = CreateText(go.transform, "Label", label, 22, FontStyle.Bold);
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
