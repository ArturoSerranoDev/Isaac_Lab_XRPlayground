#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Interaction.Toolkit.UI;
using XRPlayground.Policies;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.ROS.Editor
{
    /// <summary>
    /// Station C: Agibot A2D wall-mounted pick-and-place (Isaac Z-up layout) + IL demo UI.
    /// Menu: XRPlayground / Setup Pick Place Table Station
    /// </summary>
    public static class PickPlaceTableSceneSetup
    {
        const string StationName = "Station_C_PickPlace";
        const string RobotName = "Agibot_A2D";
        const string BridgeName = "XR Bridge PickPlace";
        const string PanelName = "PickPlace Bridge World UI";
        const string UsdPath = "Assets/_Project/Features/Robots/AgibotA2D/USD/A2D_physics.usd";
        const string PolicyOnnxPath = "Assets/_Project/Features/Policies/PickPlace/policy.onnx";

        // Left of Kinova BallCatch (-3,-2) and Conveyor (+3,-2).
        static readonly Vector3 StationLocalPos = new Vector3(-6f, 0f, -2f);

        static readonly Vector3 TableIsaacCenter = new Vector3(0.45f, 0f, 0.38f);
        static readonly Vector3 TableIsaacSize = new Vector3(0.75f, 0.60f, 0.04f);
        static readonly Vector3 BucketIsaacCenter = new Vector3(0.45f, -0.18f, 0.38f);
        static readonly Vector3 BucketIsaacSize = new Vector3(0.14f, 0.14f, 0.10f);
        static readonly Vector3 RobotIsaacPos = new Vector3(0f, -0.78f, 0f);

        [MenuItem("XRPlayground/Setup Pick Place Table Station")]
        public static void Setup()
        {
            var stations = FindRobotStations();
            if (stations == null)
            {
                Debug.LogError("XRPlayground: Run 'Setup Lab Zones' first.");
                return;
            }

            var existing = stations.Find(StationName);
            if (existing != null)
                Undo.DestroyObjectImmediate(existing.gameObject);

            var station = new GameObject(StationName);
            Undo.RegisterCreatedObjectUndo(station, StationName);
            station.transform.SetParent(stations, false);
            station.transform.localPosition = StationLocalPos;
            station.transform.localRotation = Quaternion.identity;

            var robot = PlaceRobot(station.transform);
            if (robot == null)
            {
                Undo.DestroyObjectImmediate(station);
                return;
            }

            CreateCube(station.transform, "Table", XrFrameConverter.IsaacPosToUnity(TableIsaacCenter),
                new Vector3(TableIsaacSize.x, TableIsaacSize.z, TableIsaacSize.y), new Color(0.55f, 0.42f, 0.30f),
                keepCollider: true);
            CreateCube(station.transform, "Bucket", XrFrameConverter.IsaacPosToUnity(BucketIsaacCenter),
                new Vector3(BucketIsaacSize.x, BucketIsaacSize.z, BucketIsaacSize.y), new Color(0.35f, 0.35f, 0.40f),
                keepCollider: true);

            var piece = CreateCube(station.transform, "Piece_0",
                XrFrameConverter.IsaacPosToUnity(new Vector3(0.45f, 0f, 0.44f)),
                Vector3.one * 0.04f, new Color(0.9f, 0.15f, 0.12f), keepCollider: true);
            var rb = piece.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;
            var grab = piece.AddComponent<XRGrabInteractable>();
            grab.movementType = XRGrabInteractable.MovementType.Instantaneous;

            var bridgeGo = GameObject.Find(BridgeName) ?? new GameObject(BridgeName);
            Undo.RegisterCreatedObjectUndo(bridgeGo, BridgeName);
            var client = bridgeGo.GetComponent<RosTcpClient>() ?? Undo.AddComponent<RosTcpClient>(bridgeGo);
            client.host = "127.0.0.1";
            client.port = 9092;
            client.autoConnect = false;
            var hb = bridgeGo.GetComponent<RosHeartbeatPublisher>() ?? Undo.AddComponent<RosHeartbeatPublisher>(bridgeGo);
            hb.client = client;

            WireRobotFollower(robot, station.transform, client);

            var offline = station.GetComponent<PickPlaceOfflinePolicyController>()
                ?? Undo.AddComponent<PickPlaceOfflinePolicyController>(station);
            var runner = station.GetComponent<OnnxPolicyRunner>() ?? Undo.AddComponent<OnnxPolicyRunner>(station);
            runner.expectedObsDim = PickPlaceOfflinePolicyController.ObsDim;
            runner.expectedActionDim = PickPlaceOfflinePolicyController.ActionDim;
            TryAssignPolicy(runner);

            var joints = station.GetComponent<OfflineJointDriver>() ?? Undo.AddComponent<OfflineJointDriver>(station);
            var il = station.GetComponent<PickPlaceDemoRecorder>() ?? Undo.AddComponent<PickPlaceDemoRecorder>(station);
            il.imitationLearningEnabled = true;
            il.recordRateHz = 30f;
            il.client = client;
            il.envAnchor = station.transform;
            il.linkMap = robot.GetComponent<AgibotLinkMap>();
            il.jointDriver = joints;
            il.piece = piece.transform;
            il.offlinePolicy = offline;
            il.followGrabbedPiece = true;

            offline.policyRunner = runner;
            offline.jointDriver = joints;
            offline.envAnchor = station.transform;
            offline.linkMap = robot.GetComponent<AgibotLinkMap>();
            offline.piece = piece.transform;
            offline.pieceBody = rb;
            offline.AutoBindLinks();
            il.eeLink = offline.eeLink;

            EnsureEventSystem();
            var panel = EnsureWorldUi(station.transform.position);
            panel.client = client;
            panel.robotFollower = robot.GetComponent<AgibotLinkPoseFollower>();
            panel.offlinePolicy = offline;
            panel.demoRecorder = il;
            panel.mode = PickPlaceBridgeUiMode.MirrorIsaac;

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            bool hasUsd = AssetDatabase.LoadAssetAtPath<GameObject>(UsdPath) != null;
            bool hasOnnx = runner.modelAsset != null;
            Debug.Log(
                "XRPlayground: Pick Place Station C ready (Agibot A2D, bridge :9092, IL recorder). " +
                (hasUsd ? "USD loaded. " : "Using kinematic PLACEHOLDER robot (copy A2D_physics.usd). ") +
                (hasOnnx ? "policy.onnx assigned. " : "Assign Policies/PickPlace/policy.onnx when trained. ") +
                "Play → Record IL Demo (grab Piece_0) or Start Offline Policy.");
        }

        static void TryAssignPolicy(OnnxPolicyRunner runner)
        {
            var asset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(PolicyOnnxPath);
            if (asset != null)
                runner.modelAsset = asset;
        }

        static void WireRobotFollower(GameObject robot, Transform station, RosTcpClient client)
        {
            var map = robot.GetComponent<AgibotLinkMap>() ?? Undo.AddComponent<AgibotLinkMap>(robot);
            map.preferredSubtree = "A2D";
            map.Rebuild();
            var follower = robot.GetComponent<AgibotLinkPoseFollower>() ?? Undo.AddComponent<AgibotLinkPoseFollower>(robot);
            follower.client = client;
            follower.linkMap = map;
            follower.envAnchor = station;
            follower.robotStateTopic = RosTopics.PickPlaceRobotState;
            follower.followingEnabled = true;
            follower.logFirstApply = true;
            follower.calibrateVisualFrames = false;
            follower.CaptureUnityBindPose();
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
            var usd = AssetDatabase.LoadAssetAtPath<GameObject>(UsdPath);
            GameObject robot;
            if (usd != null)
            {
                robot = (GameObject)PrefabUtility.InstantiatePrefab(usd);
                robot.name = RobotName;
                Undo.RegisterCreatedObjectUndo(robot, RobotName);
                if (PrefabUtility.IsPartOfPrefabInstance(robot))
                    PrefabUtility.UnpackPrefabInstance(robot, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
            }
            else
            {
                Debug.LogWarning(
                    "XRPlayground: Missing Agibot A2D USD — creating kinematic placeholder hierarchy.\n" +
                    $"Copy A2D_physics.usd to:\n  {UsdPath}\nSee assets/usd/agibot/A2D/README.md");
                robot = BuildPlaceholderRobot();
                Undo.RegisterCreatedObjectUndo(robot, RobotName);
            }

            robot.transform.SetParent(parent, false);
            robot.transform.localPosition = XrFrameConverter.IsaacPosToUnity(RobotIsaacPos);
            robot.transform.localRotation = Quaternion.identity;
            var map = robot.GetComponent<AgibotLinkMap>() ?? robot.AddComponent<AgibotLinkMap>();
            map.preferredSubtree = "A2D";
            map.Rebuild();
            return robot;
        }

        /// <summary>
        /// Minimal named link tree so AgibotLinkMap + OfflineJointDriver work without Nucleus USD.
        /// </summary>
        static GameObject BuildPlaceholderRobot()
        {
            var root = new GameObject(RobotName);
            var a2d = new GameObject("A2D");
            a2d.transform.SetParent(root.transform, false);

            var baseLink = CreateLink(a2d.transform, "base_link", Vector3.zero, new Vector3(0.25f, 0.12f, 0.18f),
                new Color(0.45f, 0.48f, 0.52f));
            var body = CreateLink(baseLink.transform, "body_link", new Vector3(0f, 0.35f, 0f),
                new Vector3(0.22f, 0.45f, 0.16f), new Color(0.55f, 0.58f, 0.62f));
            CreateLink(body.transform, "head_link", new Vector3(0f, 0.32f, 0.02f),
                new Vector3(0.14f, 0.14f, 0.14f), new Color(0.65f, 0.68f, 0.72f));

            Transform parent = body.transform;
            Vector3[] offsets =
            {
                new Vector3(0.12f, 0.15f, 0f),
                new Vector3(0f, 0f, 0.12f),
                new Vector3(0f, 0f, 0.14f),
                new Vector3(0f, 0f, 0.12f),
                new Vector3(0f, 0f, 0.10f),
                new Vector3(0f, 0f, 0.08f),
                new Vector3(0f, 0f, 0.06f),
            };
            for (int i = 0; i < 7; i++)
            {
                var link = CreateLink(parent, $"right_arm_link{i + 1}", offsets[i],
                    new Vector3(0.06f, 0.06f, 0.10f), new Color(0.25f, 0.55f, 0.75f));
                parent = link.transform;
            }

            var gripBase = CreateLink(parent, "right_gripper_base", new Vector3(0f, 0f, 0.04f),
                new Vector3(0.05f, 0.04f, 0.05f), new Color(0.3f, 0.3f, 0.35f));
            var center = CreateLink(gripBase.transform, "right_gripper_center", new Vector3(0f, 0f, 0.03f),
                new Vector3(0.03f, 0.03f, 0.03f), new Color(0.9f, 0.75f, 0.2f));
            CreateLink(center.transform, "right_Left_Pad_Link", new Vector3(-0.03f, 0f, 0.02f),
                new Vector3(0.015f, 0.04f, 0.04f), new Color(0.2f, 0.2f, 0.22f));
            CreateLink(center.transform, "right_Right_Pad_Link", new Vector3(0.03f, 0f, 0.02f),
                new Vector3(0.015f, 0.04f, 0.04f), new Color(0.2f, 0.2f, 0.22f));

            return root;
        }

        static GameObject CreateLink(Transform parent, string name, Vector3 localPos, Vector3 scale, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = scale;
            Object.DestroyImmediate(go.GetComponent<Collider>());
            var rend = go.GetComponent<MeshRenderer>();
            if (rend != null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                rend.sharedMaterial = new Material(shader) { color = color };
            }
            return go;
        }

        static GameObject CreateCube(Transform parent, string name, Vector3 localPos, Vector3 scale, Color color,
            bool keepCollider = false)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Undo.RegisterCreatedObjectUndo(go, name);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            if (!keepCollider)
                Object.DestroyImmediate(go.GetComponent<Collider>());
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

        static PickPlaceBridgePanel EnsureWorldUi(Vector3 stationPos)
        {
            var existing = GameObject.Find(PanelName);
            if (existing != null)
            {
                var p = existing.GetComponent<PickPlaceBridgePanel>();
                if (p != null)
                    return p;
                Undo.DestroyObjectImmediate(existing);
            }

            var root = new GameObject(PanelName);
            Undo.RegisterCreatedObjectUndo(root, PanelName);
            root.transform.position = stationPos + new Vector3(0.35f, 1.35f, -0.85f);
            root.transform.rotation = Quaternion.Euler(0f, 180f, 0f);

            var canvasGo = new GameObject("Canvas");
            canvasGo.transform.SetParent(root.transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 10f;
            canvasGo.AddComponent<GraphicRaycaster>();
            canvasGo.AddComponent<TrackedDeviceGraphicRaycaster>();
            var rt = canvasGo.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(860f, 640f);
            canvasGo.transform.localScale = Vector3.one * 0.0012f;

            var bg = CreateUiObject("Background", canvasGo.transform);
            bg.AddComponent<Image>().color = new Color(0.08f, 0.09f, 0.11f, 0.92f);
            StretchFull(bg.GetComponent<RectTransform>());

            var title = CreateText(canvasGo.transform, "Title", "Pick-Place (Agibot)", 34, FontStyle.Bold);
            SetRect(title.rectTransform, 40, -28, 780, 48);

            var status = CreateText(canvasGo.transform, "Status", "Bridge: disconnected", 20, FontStyle.Normal);
            SetRect(status.rectTransform, 40, -90, 780, 110);
            status.alignment = TextAnchor.UpperLeft;

            var mode = CreateText(canvasGo.transform, "ModeLabel", "Mode: Mirror Isaac", 22, FontStyle.Bold);
            SetRect(mode.rectTransform, 40, -210, 780, 36);

            var connectBtn = CreateButton(canvasGo.transform, "ConnectButton", "Connect bridge", new Color(0.15f, 0.45f, 0.75f));
            SetRect(connectBtn.GetComponent<RectTransform>(), 40, -270, 360, 64);

            var mirrorBtn = CreateButton(canvasGo.transform, "MirrorButton", "Mirror Isaac", new Color(0.25f, 0.45f, 0.28f));
            SetRect(mirrorBtn.GetComponent<RectTransform>(), 40, -350, 360, 64);

            var recordBtn = CreateButton(canvasGo.transform, "RecordDemoButton", "Record IL Demo", new Color(0.55f, 0.35f, 0.15f));
            SetRect(recordBtn.GetComponent<RectTransform>(), 420, -350, 380, 64);

            var policyBtn = CreateButton(canvasGo.transform, "StartPolicy", "Start Offline Policy", new Color(0.45f, 0.20f, 0.55f));
            SetRect(policyBtn.GetComponent<RectTransform>(), 40, -440, 760, 64);

            var panel = root.AddComponent<PickPlaceBridgePanel>();
            panel.statusText = status;
            panel.modeText = mode;
            panel.connectButton = connectBtn;
            panel.mirrorButton = mirrorBtn;
            panel.recordDemoButton = recordBtn;
            panel.startPolicyButton = policyBtn;
            panel.connectButtonLabel = connectBtn.GetComponentInChildren<Text>();
            panel.startPolicyButtonLabel = policyBtn.GetComponentInChildren<Text>();
            panel.recordDemoButtonLabel = recordBtn.GetComponentInChildren<Text>();
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
