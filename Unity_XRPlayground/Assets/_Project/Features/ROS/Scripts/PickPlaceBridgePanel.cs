using UnityEngine;
using UnityEngine.UI;
using XRPlayground.Policies;

namespace XRPlayground.ROS
{
    public enum PickPlaceBridgeUiMode
    {
        MirrorIsaac = 0,
        RecordDemo = 1,
    }

    /// <summary>
    /// World-space UI for Pick-Place Table (Agibot A2D, bridge :9092) + offline ONNX + IL record.
    /// </summary>
    public sealed class PickPlaceBridgePanel : MonoBehaviour
    {
        public RosTcpClient client;
        public AgibotLinkPoseFollower robotFollower;
        public PickPlaceOfflinePolicyController offlinePolicy;
        public PickPlaceDemoRecorder demoRecorder;

        public Text statusText;
        public Text modeText;
        public Button connectButton;
        public Button mirrorButton;
        public Button recordDemoButton;
        public Button startPolicyButton;
        public Text connectButtonLabel;
        public Text startPolicyButtonLabel;
        public Text recordDemoButtonLabel;

        public PickPlaceBridgeUiMode mode = PickPlaceBridgeUiMode.MirrorIsaac;

        string _phase = "-";
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

        void Update() => RefreshStatusLine();

        void WireButtons()
        {
            if (_wired)
                return;
            _wired = true;
            if (connectButton != null)
                connectButton.onClick.AddListener(ToggleConnect);
            if (mirrorButton != null)
                mirrorButton.onClick.AddListener(() => SetMode(PickPlaceBridgeUiMode.MirrorIsaac));
            if (recordDemoButton != null)
                recordDemoButton.onClick.AddListener(() => SetMode(PickPlaceBridgeUiMode.RecordDemo));
            if (startPolicyButton != null)
                startPolicyButton.onClick.AddListener(ToggleOfflinePolicy);
        }

        public void ToggleOfflinePolicy()
        {
            if (offlinePolicy == null)
            {
                Debug.LogWarning("PickPlaceBridgePanel: assign offlinePolicy.", this);
                return;
            }

            if (offlinePolicy.running)
            {
                offlinePolicy.StopPolicy();
            }
            else
            {
                if (mode == PickPlaceBridgeUiMode.RecordDemo)
                    SetMode(PickPlaceBridgeUiMode.MirrorIsaac);
                offlinePolicy.StartPolicy();
            }

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

        public void SetMode(PickPlaceBridgeUiMode newMode)
        {
            mode = newMode;
            ApplyModeLocal();
            SendModeCommand();
            RefreshUi();
        }

        void ApplyModeLocal()
        {
            bool offline = offlinePolicy != null && offlinePolicy.running;
            bool recording = mode == PickPlaceBridgeUiMode.RecordDemo && !offline;

            if (demoRecorder != null)
            {
                demoRecorder.imitationLearningEnabled = true;
                demoRecorder.SetRecording(recording);
            }

            if (robotFollower != null)
                robotFollower.followingEnabled = !offline && !recording;
        }

        void SendModeCommand()
        {
            if (client == null || !client.IsConnected)
                return;
            string m = mode == PickPlaceBridgeUiMode.RecordDemo
                ? RosTopics.ModeRecordDemo
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
            _phase = string.IsNullOrEmpty(status.phase) ? "-" : status.phase;
            _policyLoaded = status.policy_loaded;
            if (status.mode == RosTopics.ModeRecordDemo)
                mode = PickPlaceBridgeUiMode.RecordDemo;
            else if (status.mode == RosTopics.ModeMirror)
                mode = PickPlaceBridgeUiMode.MirrorIsaac;
            ApplyModeLocal();
            RefreshUi();
        }

        void RefreshUi()
        {
            ApplyModeLocal();
            if (modeText != null)
            {
                modeText.text = mode == PickPlaceBridgeUiMode.RecordDemo
                    ? "Mode: Record IL demo"
                    : "Mode: Mirror Isaac";
            }
            if (connectButtonLabel != null)
                connectButtonLabel.text = (client != null && client.IsConnected) ? "Disconnect" : "Connect bridge";
            if (startPolicyButtonLabel != null)
            {
                bool on = offlinePolicy != null && offlinePolicy.running;
                startPolicyButtonLabel.text = on ? "Stop Offline Policy" : "Start Offline Policy";
            }
            if (recordDemoButtonLabel != null)
            {
                bool rec = mode == PickPlaceBridgeUiMode.RecordDemo;
                recordDemoButtonLabel.text = rec ? "Recording… (tap Mirror to stop)" : "Record IL Demo";
            }
            RefreshStatusLine();
        }

        void RefreshStatusLine()
        {
            if (statusText == null)
                return;
            bool connected = client != null && client.IsConnected;
            bool offline = offlinePolicy != null && offlinePolicy.running;
            string tip;
            if (offline)
                tip = $"Offline ONNX: {offlinePolicy.StatusLine}";
            else if (mode == PickPlaceBridgeUiMode.RecordDemo && demoRecorder != null)
                tip = $"IL: {demoRecorder.StatusLine}";
            else
                tip = "Robot mirrors Isaac (:9092). Record demos or run offline ONNX.";

            statusText.text =
                (offline ? "Mode: OFFLINE ONNX (no Isaac)" : connected ? "Bridge: CONNECTED :9092" : "Bridge: disconnected") +
                $"\nPhase: {_phase}  Policy: {(_policyLoaded ? "loaded" : "none")}" +
                $"\n{tip}";
        }
    }
}
