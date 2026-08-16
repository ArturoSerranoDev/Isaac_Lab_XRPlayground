#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;
using XRPlayground.Policies;

namespace XRPlayground.ROS.Editor
{
    /// <summary>
    /// Creates / refreshes world-space Network visualizer panels beside each station's bridge UI.
    /// Menus under XRPlayground / Setup Policy Network Panel …
    /// </summary>
    public static class PolicyNetworkPanelSetup
    {
        const string BallCatchPanelName = "Policy Network World UI";
        const string ConveyorPanelName = "Policy Network Conveyor World UI";
        const string PickPlacePanelName = "Policy Network PickPlace World UI";
        const string BalanceBotPanelName = "Policy Network BalanceBot World UI";
        const string SpotPanelName = "Policy Network Spot World UI";

        const string BallCatchBridgeName = "XR Bridge World UI";
        const string ConveyorBridgeName = "Conveyor Bridge World UI";
        const string PickPlaceBridgeName = "PickPlace Bridge World UI";
        const string BalanceBotBridgeName = "BalanceBot Bridge World UI";
        const string SpotBridgeName = "Spot Bridge World UI";

        const string KinovaName = "Kinova_Jaco2_j2n7s300";
        const string ConveyorStationName = "Station_B_Conveyor";
        const string PickPlaceStationName = "Station_C_PickPlace";
        const string BalanceBotStationName = "Station_D_BalanceBot";
        const string SpotStationName = "Station_E_Spot";

        static readonly Vector3 BesideBridgeLocal = new Vector3(1.15f, 0f, 0f);

        [MenuItem("XRPlayground/Setup Policy Network Panel (BallCatch)")]
        public static void SetupBallCatch()
        {
            var robot = GameObject.Find(KinovaName);
            var runner = robot != null ? robot.GetComponent<OnnxPolicyRunner>() : null;
            var policy = robot != null ? robot.GetComponent<BallCatchOfflinePolicyController>() : null;
            if (runner == null)
                Debug.LogWarning("PolicyNetworkPanelSetup: OnnxPolicyRunner not found on Kinova — run Setup XR Bridge Scene first.");

            var panel = SetupBesideBridge(
                BallCatchPanelName,
                BallCatchBridgeName,
                robot,
                new[] { 30, 256, 128, 64, 8 },
                runner,
                fallbackPos: new Vector3(-3.8f, 1.35f, -2.85f));

            if (runner != null)
                panel.BindBallCatch(runner, policy);

            var bridge = GameObject.Find(BallCatchBridgeName);
            var bridgePanel = bridge != null ? bridge.GetComponent<XrBridgePanel>() : null;
            if (bridgePanel != null && policy != null)
                bridgePanel.offlinePolicy = policy;

            FinishSetup(panel, "BallCatch");
        }

        [MenuItem("XRPlayground/Setup Policy Network Panel (Conveyor)")]
        public static void SetupConveyor()
        {
            var station = GameObject.Find(ConveyorStationName);
            var runner = station != null ? station.GetComponent<OnnxPolicyRunner>() : null;
            var policy = station != null ? station.GetComponent<ConveyorOfflinePolicyController>() : null;
            if (runner == null)
                Debug.LogWarning("PolicyNetworkPanelSetup: OnnxPolicyRunner not found on Station_B_Conveyor — run Setup Conveyor Color Station first.");

            var panel = SetupBesideBridge(
                ConveyorPanelName,
                ConveyorBridgeName,
                station,
                new[] { 66, 256, 128, 64, 7 },
                runner,
                fallbackPos: new Vector3(2.25f, 1.4f, -2.9f));

            if (runner != null)
                panel.BindConveyor(runner, policy);

            var bridge = GameObject.Find(ConveyorBridgeName);
            var bridgePanel = bridge != null ? bridge.GetComponent<ConveyorBridgePanel>() : null;
            if (bridgePanel != null && policy != null)
                bridgePanel.offlinePolicy = policy;

            FinishSetup(panel, "Conveyor");
        }

        [MenuItem("XRPlayground/Setup Policy Network Panel (PickPlace)")]
        public static void SetupPickPlace()
        {
            var station = GameObject.Find(PickPlaceStationName);
            var runner = station != null ? station.GetComponent<OnnxPolicyRunner>() : null;
            var policy = station != null ? station.GetComponent<PickPlaceOfflinePolicyController>() : null;
            if (runner == null)
                Debug.LogWarning("PolicyNetworkPanelSetup: OnnxPolicyRunner not found on Station_C_PickPlace — run Setup Pick Place Station first.");

            var panel = SetupBesideBridge(
                PickPlacePanelName,
                PickPlaceBridgeName,
                station,
                new[] { 30, 256, 128, 64, 8 },
                runner,
                fallbackPos: new Vector3(-6.8f, 1.35f, -2.85f));

            if (runner != null)
                panel.BindPickPlace(runner, policy);

            var bridge = GameObject.Find(PickPlaceBridgeName);
            var bridgePanel = bridge != null ? bridge.GetComponent<PickPlaceBridgePanel>() : null;
            if (bridgePanel != null && policy != null)
                bridgePanel.offlinePolicy = policy;

            FinishSetup(panel, "PickPlace");
        }

        [MenuItem("XRPlayground/Setup Policy Network Panel (BalanceBot)")]
        public static void SetupBalanceBot()
        {
            var station = GameObject.Find(BalanceBotStationName);
            var runner = station != null ? station.GetComponent<OnnxPolicyRunner>() : null;
            var policy = station != null ? station.GetComponent<BalanceBotOfflinePolicyController>() : null;
            if (runner == null)
                Debug.LogWarning("PolicyNetworkPanelSetup: OnnxPolicyRunner not found on Station_D_BalanceBot — run Setup Balance Bot Station first.");

            var panel = SetupBesideBridge(
                BalanceBotPanelName,
                BalanceBotBridgeName,
                station,
                new[] { 20, 256, 128, 64, 2 },
                runner,
                fallbackPos: new Vector3(5.25f, 1.4f, -2.9f));

            if (runner != null)
                panel.BindBalanceBot(runner, policy);

            var bridge = GameObject.Find(BalanceBotBridgeName);
            var bridgePanel = bridge != null ? bridge.GetComponent<BalanceBotBridgePanel>() : null;
            if (bridgePanel != null && policy != null)
                bridgePanel.offlinePolicy = policy;

            FinishSetup(panel, "BalanceBot");
        }

        [MenuItem("XRPlayground/Setup Policy Network Panel (Spot)")]
        public static void SetupSpot()
        {
            var station = GameObject.Find(SpotStationName);
            var runner = station != null ? station.GetComponent<OnnxPolicyRunner>() : null;
            var policy = station != null ? station.GetComponent<SpotOfflinePolicyController>() : null;
            if (runner == null)
                Debug.LogWarning("PolicyNetworkPanelSetup: OnnxPolicyRunner not found on Station_E_Spot — run Setup Spot Follow Station first.");

            var panel = SetupBesideBridge(
                SpotPanelName,
                SpotBridgeName,
                station,
                new[] { 48, 512, 256, 128, 12 },
                runner,
                fallbackPos: new Vector3(0.5f, 1.5f, 1.0f));

            if (runner != null)
                panel.BindSpot(runner, policy);

            var bridge = GameObject.Find(SpotBridgeName);
            var bridgePanel = bridge != null ? bridge.GetComponent<SpotBridgePanel>() : null;
            if (bridgePanel != null && policy != null)
                bridgePanel.offlinePolicy = policy;

            FinishSetup(panel, "Spot");
        }

        [MenuItem("XRPlayground/Setup Policy Network Panels (All Stations)")]
        public static void SetupAll()
        {
            SetupBallCatch();
            SetupConveyor();
            SetupPickPlace();
            SetupBalanceBot();
            SetupSpot();
            Debug.Log("XRPlayground: Policy Network panels ready for BallCatch, Conveyor, PickPlace, BalanceBot, and Spot.");
        }

        static PolicyNetworkPanel SetupBesideBridge(
            string panelName,
            string bridgeName,
            GameObject anchor,
            int[] fallbackSizes,
            OnnxPolicyRunner runner,
            Vector3 fallbackPos)
        {
            var bridge = GameObject.Find(bridgeName);
            Vector3 pos;
            Quaternion rot;
            if (bridge != null)
            {
                pos = bridge.transform.TransformPoint(BesideBridgeLocal);
                rot = bridge.transform.rotation;
            }
            else if (anchor != null)
            {
                pos = anchor.transform.position + new Vector3(1.4f, 1.35f, -0.85f);
                rot = Quaternion.Euler(0f, 180f, 0f);
            }
            else
            {
                pos = fallbackPos;
                rot = Quaternion.Euler(0f, 180f, 0f);
            }

            var panel = EnsurePanel(panelName, pos, rot, fallbackSizes);
            if (runner != null)
                runner.captureActivations = true;
            return panel;
        }

        static void FinishSetup(PolicyNetworkPanel panel, string stationLabel)
        {
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = panel.gameObject;
            bool hasModel = panel.policyRunner != null && panel.policyRunner.modelAsset != null;
            Debug.Log(
                $"XRPlayground: Policy Network panel ready beside {stationLabel} bridge. " +
                (hasModel
                    ? "Play → Start Offline Policy — neurons highlight from real ELU intermediates."
                    : "Idle placeholder until policy.onnx is assigned; then Play → Start Offline Policy."),
                panel);
        }

        static PolicyNetworkPanel EnsurePanel(string panelName, Vector3 pos, Quaternion rot, int[] fallbackSizes)
        {
            var existing = GameObject.Find(panelName);
            if (existing != null)
            {
                var p = existing.GetComponent<PolicyNetworkPanel>();
                if (p != null)
                {
                    existing.transform.SetPositionAndRotation(pos, rot);
                    EnsureUi(existing.transform, p);
                    if (fallbackSizes != null && fallbackSizes.Length >= 2)
                        p.fallbackLayerSizes = fallbackSizes;
                    return p;
                }
            }

            var root = new GameObject(panelName);
            Undo.RegisterCreatedObjectUndo(root, panelName);
            root.transform.SetPositionAndRotation(pos, rot);

            var canvasGo = new GameObject("Canvas");
            canvasGo.transform.SetParent(root.transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 10f;
            canvasGo.AddComponent<GraphicRaycaster>();
            canvasGo.AddComponent<TrackedDeviceGraphicRaycaster>();

            var rt = canvasGo.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(980f, 520f);
            canvasGo.transform.localScale = Vector3.one * 0.0012f;

            var cg = canvasGo.AddComponent<CanvasGroup>();
            cg.alpha = 0.55f;

            var bg = CreateUiObject("Background", canvasGo.transform);
            var bgImg = bg.AddComponent<Image>();
            bgImg.color = new Color(0.07f, 0.08f, 0.11f, 0.94f);
            StretchFull(bg.GetComponent<RectTransform>());

            var accent = CreateUiObject("Accent", canvasGo.transform);
            var accentImg = accent.AddComponent<Image>();
            accentImg.color = new Color(0.35f, 0.55f, 0.85f, 0.85f);
            var accentRt = accent.GetComponent<RectTransform>();
            accentRt.anchorMin = new Vector2(0f, 0f);
            accentRt.anchorMax = new Vector2(0f, 1f);
            accentRt.pivot = new Vector2(0f, 0.5f);
            accentRt.anchoredPosition = Vector2.zero;
            accentRt.sizeDelta = new Vector2(8f, 0f);

            var title = CreateText(canvasGo.transform, "Title", "Network", 34, FontStyle.Bold);
            SetRect(title.rectTransform, 28f, -18f, 520f, 44f);

            var live = CreateUiObject("LiveBadge", canvasGo.transform);
            var liveImg = live.AddComponent<Image>();
            liveImg.color = new Color(0.2f, 0.65f, 0.4f, 0.95f);
            SetRect(live.GetComponent<RectTransform>(), 780f, -24f, 160f, 34f);
            var liveLabel = CreateText(live.transform, "Label", "LIVE", 18, FontStyle.Bold);
            StretchFull(liveLabel.rectTransform);
            liveLabel.alignment = TextAnchor.MiddleCenter;
            live.SetActive(false);

            string archLabel = string.Join(" → ", fallbackSizes);
            var arch = CreateText(canvasGo.transform, "Architecture", archLabel, 20, FontStyle.Normal);
            SetRect(arch.rectTransform, 28f, -62f, 900f, 30f);
            arch.color = new Color(0.7f, 0.78f, 0.9f, 1f);

            var status = CreateText(canvasGo.transform, "Status", "Start Offline Policy to animate activations", 18, FontStyle.Normal);
            SetRect(status.rectTransform, 28f, -460f, 920f, 40f);
            status.color = new Color(0.65f, 0.7f, 0.78f, 1f);

            var content = CreateUiObject("NetworkContent", canvasGo.transform);
            SetRect(content.GetComponent<RectTransform>(), 20f, -100f, 940f, 350f);

            var visualizer = root.AddComponent<PolicyNetworkVisualizer>();
            visualizer.contentRoot = content.GetComponent<RectTransform>();
            visualizer.maxDiscreteNeurons = 48;
            visualizer.drawTopKEdges = true;
            visualizer.topKEdges = 6;

            var panel = root.AddComponent<PolicyNetworkPanel>();
            panel.visualizer = visualizer;
            panel.titleText = title;
            panel.architectureText = arch;
            panel.statusText = status;
            panel.liveBadge = live;
            panel.canvasGroup = cg;
            panel.fallbackLayerSizes = fallbackSizes;
            return panel;
        }

        static void EnsureUi(Transform root, PolicyNetworkPanel panel)
        {
            if (panel.visualizer == null)
                panel.visualizer = root.GetComponent<PolicyNetworkVisualizer>()
                    ?? root.gameObject.AddComponent<PolicyNetworkVisualizer>();
            var canvas = root.Find("Canvas");
            if (canvas == null)
                return;
            if (panel.canvasGroup == null)
                panel.canvasGroup = canvas.GetComponent<CanvasGroup>() ?? canvas.gameObject.AddComponent<CanvasGroup>();
            if (panel.visualizer.contentRoot == null)
            {
                var content = canvas.Find("NetworkContent");
                if (content != null)
                    panel.visualizer.contentRoot = content.GetComponent<RectTransform>();
            }
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
