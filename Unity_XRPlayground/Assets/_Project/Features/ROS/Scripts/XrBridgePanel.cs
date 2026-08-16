using UnityEngine;
using UnityEngine.UI;
using XRPlayground.Policies;

namespace XRPlayground.ROS
{
    public enum XrBridgeUiMode
    {
        MirrorIsaac = 0,
        AwaitPlayerThrow = 1
    }

    /// <summary>
    /// World-space panel controller (classic uGUI). Connects TCP bridge and selects session mode.
    /// </summary>
    public sealed class XrBridgePanel : MonoBehaviour
    {
        public RosTcpClient client;
        public BallStatePublisher ballPublisher;
        public BallPoseFollower ballFollower;
        public KinovaLinkPoseFollower robotFollower;
        public BallCatchOfflinePolicyController offlinePolicy;
        public BridgeHealthMonitor healthMonitor;

        [Header("UI (classic uGUI)")]
        public Text statusText;
        public Text modeText;
        public Button connectButton;
        public Button mirrorButton;
        public Button awaitThrowButton;
        public Button startPolicyButton;
        public Text connectButtonLabel;
        public Text startPolicyButtonLabel;

        public XrBridgeUiMode mode = XrBridgeUiMode.MirrorIsaac;

        string _isaacPhase = "-";
        bool _policyLoaded;
        bool _wired;

        void OnEnable()
        {
            WireButtons();
            if (client != null)
                client.MessageReceived += OnMessage;
            if (healthMonitor == null)
                healthMonitor = FindAnyObjectByType<BridgeHealthMonitor>();
            RefreshUi();
        }

        void OnDisable()
        {
            if (client != null)
                client.MessageReceived -= OnMessage;
        }

        void Update()
        {
            RefreshStatusLine();
        }

        void WireButtons()
        {
            if (_wired)
                return;
            _wired = true;
            if (connectButton != null)
                connectButton.onClick.AddListener(ToggleConnect);
            if (mirrorButton != null)
                mirrorButton.onClick.AddListener(() => SetMode(XrBridgeUiMode.MirrorIsaac));
            if (awaitThrowButton != null)
                awaitThrowButton.onClick.AddListener(() => SetMode(XrBridgeUiMode.AwaitPlayerThrow));
            if (startPolicyButton != null)
                startPolicyButton.onClick.AddListener(ToggleOfflinePolicy);
        }

        public void ToggleOfflinePolicy()
        {
            if (offlinePolicy == null)
            {
                Debug.LogWarning("XrBridgePanel: assign offlinePolicy (BallCatchOfflinePolicyController).", this);
                return;
            }
            if (offlinePolicy.policyRunner != null)
                offlinePolicy.policyRunner.captureActivations = true;
            offlinePolicy.TogglePolicy();
            RefreshUi();
        }

        public void ToggleConnect()
        {
            if (client == null)
                return;
            if (client.IsConnected)
            {
                client.Disconnect();
                client.autoConnect = false;
            }
            else
            {
                client.autoConnect = true;
                client.Connect();
                SendModeCommand();
            }
            RefreshUi();
        }

        public void SetMode(XrBridgeUiMode newMode)
        {
            mode = newMode;
            if (mode == XrBridgeUiMode.AwaitPlayerThrow)
                ResetBallForPlayerThrow();
            ApplyModeLocal();
            SendModeCommand();
            RefreshUi();
        }

        void ApplyModeLocal()
        {
            bool offline = offlinePolicy != null && offlinePolicy.running;
            bool awaitThrow = mode == XrBridgeUiMode.AwaitPlayerThrow;
            bool catching = awaitThrow && _isaacPhase == "catching";
            bool playerOwnsBall = awaitThrow && !catching && !offline;

            if (ballPublisher != null)
            {
                ballPublisher.publishingEnabled = playerOwnsBall;
                ballPublisher.publishWhileHeld = true;
            }

            if (ballFollower != null)
            {
                // Mirror: always follow Isaac. Await throw: follow Isaac only during catch.
                // Offline ONNX: Unity owns the ball.
                ballFollower.followingEnabled = !offline && !playerOwnsBall;
                ballFollower.SetKinematic(!offline && !playerOwnsBall);
            }

            if (robotFollower != null)
                robotFollower.enabled = !offline;
        }

        /// <summary>
        /// Place a grabable ball near the Kinova for the player (env-local Isaac spawn ≈ (0.55, 0, 0.95)).
        /// </summary>
        public void ResetBallForPlayerThrow()
        {
            Transform ball = null;
            if (ballPublisher != null && ballPublisher.ballRoot != null)
                ball = ballPublisher.ballRoot;
            else if (ballFollower != null && ballFollower.ballRoot != null)
                ball = ballFollower.ballRoot;
            if (ball == null)
            {
                var go = GameObject.Find("Ball");
                if (go != null)
                    ball = go.transform;
            }
            if (ball == null)
                return;

            Transform anchor = null;
            if (ballPublisher != null && ballPublisher.envAnchor != null)
                anchor = ballPublisher.envAnchor;
            else if (ballFollower != null && ballFollower.envAnchor != null)
                anchor = ballFollower.envAnchor;
            else
            {
                var robot = GameObject.Find("Kinova_Jaco2_j2n7s300");
                if (robot != null)
                    anchor = robot.transform;
            }

            // Isaac env-local ready pose (Z-up) → Unity, near the arm for XR grab
            Vector3 isaacLocal = new Vector3(0.55f, 0.05f, 0.95f);
            Vector3 unityLocal = XrFrameConverter.IsaacPosToUnity(isaacLocal);
            Vector3 worldPos = anchor != null ? anchor.TransformPoint(unityLocal) : unityLocal;
            ball.position = worldPos;
            ball.rotation = Quaternion.identity;

            var rb = ball.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.isKinematic = false;
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }

            // Diameter matches Isaac radius 0.04125 (Unity default sphere radius 0.5)
            float diameter = 0.0825f;
            ball.localScale = Vector3.one * diameter;
        }

        void SendModeCommand()
        {
            if (client == null || !client.IsConnected)
                return;
            string m = mode == XrBridgeUiMode.AwaitPlayerThrow
                ? RosTopics.ModeAwaitThrow
                : RosTopics.ModeMirror;
            client.PublishJson(RosJson.SerializeSessionCommand(m));
        }

        void OnMessage(string json)
        {
            if (!RosJson.TryParseTopic(json, out var topic))
                return;
            if (topic != RosTopics.SessionStatus)
                return;
            if (!RosJson.TryParseSessionStatus(json, out var status) || status == null)
                return;
            string prevPhase = _isaacPhase;
            _isaacPhase = string.IsNullOrEmpty(status.phase) ? "-" : status.phase;
            _policyLoaded = status.policy_loaded;
            if (!string.IsNullOrEmpty(status.mode))
            {
                if (status.mode == RosTopics.ModeAwaitThrow)
                    mode = XrBridgeUiMode.AwaitPlayerThrow;
                else if (status.mode == RosTopics.ModeMirror)
                    mode = XrBridgeUiMode.MirrorIsaac;
            }

            // When Isaac returns to waiting after a catch, re-spawn the Unity ball
            if (mode == XrBridgeUiMode.AwaitPlayerThrow && prevPhase == "catching" && _isaacPhase == "waiting")
                ResetBallForPlayerThrow();

            ApplyModeLocal();
            RefreshUi();
        }

        void RefreshUi()
        {
            ApplyModeLocal();
            if (modeText != null)
            {
                modeText.text = mode == XrBridgeUiMode.AwaitPlayerThrow
                    ? "Mode: Await player throw"
                    : "Mode: Mirror Isaac";
            }
            if (connectButtonLabel != null)
                connectButtonLabel.text = (client != null && client.IsConnected) ? "Disconnect" : "Connect bridge";
            if (startPolicyButtonLabel != null)
            {
                bool on = offlinePolicy != null && offlinePolicy.running;
                startPolicyButtonLabel.text = on ? "Stop Offline Policy" : "Start Offline Policy";
            }
            RefreshStatusLine();
        }

        void RefreshStatusLine()
        {
            if (statusText == null)
                return;
            bool connected = client != null && client.IsConnected;
            bool offline = offlinePolicy != null && offlinePolicy.running;
            string tip = offline
                ? $"Offline ONNX: {offlinePolicy.StatusLine}"
                : mode == XrBridgeUiMode.AwaitPlayerThrow
                    ? "Grab & throw the ball. It replicates to Isaac; after release Isaac catches."
                    : "Robot + ball mirror Isaac Sim.";
            statusText.text =
                (offline ? "Mode: OFFLINE ONNX (no Isaac)" : connected ? "Bridge: CONNECTED" : "Bridge: disconnected") +
                $"\nIsaac phase: {_isaacPhase}" +
                $"\nPolicy: {(_policyLoaded ? "loaded" : "none")}" +
                $"\n{(healthMonitor != null ? healthMonitor.CompactSummary : "Checks: health monitor missing")}" +
                $"\n{tip}";
        }
    }
}
