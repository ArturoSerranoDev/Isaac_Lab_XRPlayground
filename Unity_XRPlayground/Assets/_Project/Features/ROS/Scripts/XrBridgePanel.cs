using UnityEngine;
using UnityEngine.UI;

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
        public KinovaLinkPoseFollower robotFollower;

        [Header("UI (classic uGUI)")]
        public Text statusText;
        public Text modeText;
        public Button connectButton;
        public Button mirrorButton;
        public Button awaitThrowButton;
        public Text connectButtonLabel;

        public XrBridgeUiMode mode = XrBridgeUiMode.MirrorIsaac;

        string _isaacPhase = "-";
        bool _policyLoaded;
        bool _wired;

        void OnEnable()
        {
            WireButtons();
            if (client != null)
                client.MessageReceived += OnMessage;
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
            ApplyModeLocal();
            SendModeCommand();
            RefreshUi();
        }

        void ApplyModeLocal()
        {
            bool awaitThrow = mode == XrBridgeUiMode.AwaitPlayerThrow;
            if (ballPublisher != null)
            {
                ballPublisher.publishingEnabled = awaitThrow;
                ballPublisher.publishWhileHeld = awaitThrow;
            }
            if (robotFollower != null)
                robotFollower.enabled = true; // always puppet from Isaac when connected
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
            _isaacPhase = string.IsNullOrEmpty(status.phase) ? "-" : status.phase;
            _policyLoaded = status.policy_loaded;
            if (!string.IsNullOrEmpty(status.mode))
            {
                if (status.mode == RosTopics.ModeAwaitThrow)
                    mode = XrBridgeUiMode.AwaitPlayerThrow;
                else if (status.mode == RosTopics.ModeMirror)
                    mode = XrBridgeUiMode.MirrorIsaac;
            }
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
            RefreshStatusLine();
        }

        void RefreshStatusLine()
        {
            if (statusText == null)
                return;
            bool connected = client != null && client.IsConnected;
            string tip = mode == XrBridgeUiMode.AwaitPlayerThrow
                ? "Grab & throw the ball. Robot waits, then tries to catch."
                : "Robot mirrors Isaac Sim. Unity ball is ignored.";
            statusText.text =
                (connected ? "Bridge: CONNECTED" : "Bridge: disconnected") +
                $"\nIsaac phase: {_isaacPhase}" +
                $"\nPolicy: {(_policyLoaded ? "loaded" : "none")}" +
                $"\n{tip}";
        }
    }
}
