using UnityEngine;
using UnityEngine.UI;

namespace XRPlayground.ROS
{
    public enum ConveyorBridgeUiMode
    {
        MirrorIsaac = 0,
        AwaitSpawn = 1
    }

    /// <summary>
    /// World-space UI for Conveyor Color bridge (port 9091).
    /// </summary>
    public sealed class ConveyorBridgePanel : MonoBehaviour
    {
        public RosTcpClient client;
        public ConveyorSpawnPublisher spawnPublisher;
        public ConveyorObjectFollower objectFollower;
        public RobotLinkPoseFollower robotFollower;

        public Text statusText;
        public Text modeText;
        public Text targetColorText;
        public Button connectButton;
        public Button mirrorButton;
        public Button awaitSpawnButton;
        public Button spawnRedButton;
        public Button spawnGreenButton;
        public Button spawnBlueButton;
        public Text connectButtonLabel;

        public ConveyorBridgeUiMode mode = ConveyorBridgeUiMode.MirrorIsaac;

        string _phase = "-";
        bool _policyLoaded;
        bool _wired;
        int _targetColor;

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
                mirrorButton.onClick.AddListener(() => SetMode(ConveyorBridgeUiMode.MirrorIsaac));
            if (awaitSpawnButton != null)
                awaitSpawnButton.onClick.AddListener(() => SetMode(ConveyorBridgeUiMode.AwaitSpawn));
            if (spawnRedButton != null)
                spawnRedButton.onClick.AddListener(() => spawnPublisher?.SpawnRed());
            if (spawnGreenButton != null)
                spawnGreenButton.onClick.AddListener(() => spawnPublisher?.SpawnGreen());
            if (spawnBlueButton != null)
                spawnBlueButton.onClick.AddListener(() => spawnPublisher?.SpawnBlue());
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

        public void SetMode(ConveyorBridgeUiMode newMode)
        {
            mode = newMode;
            ApplyModeLocal();
            SendModeCommand();
            RefreshUi();
        }

        void ApplyModeLocal()
        {
            bool awaitSpawn = mode == ConveyorBridgeUiMode.AwaitSpawn;
            if (spawnPublisher != null)
                spawnPublisher.publishingEnabled = awaitSpawn;
            if (objectFollower != null)
                objectFollower.followingEnabled = true;
            if (robotFollower != null)
                robotFollower.followingEnabled = true;
        }

        void SendModeCommand()
        {
            if (client == null || !client.IsConnected)
                return;
            string m = mode == ConveyorBridgeUiMode.AwaitSpawn
                ? RosTopics.ModeAwaitSpawn
                : RosTopics.ModeMirror;
            client.PublishJson(RosJson.SerializeSessionCommand(m));
        }

        void OnMessage(string json)
        {
            if (!RosJson.TryParseTopic(json, out var topic))
                return;
            if (topic == RosTopics.SessionStatus)
            {
                if (!RosJson.TryParseSessionStatus(json, out var status) || status == null)
                    return;
                _phase = string.IsNullOrEmpty(status.phase) ? "-" : status.phase;
                _policyLoaded = status.policy_loaded;
                if (status.mode == RosTopics.ModeAwaitSpawn)
                    mode = ConveyorBridgeUiMode.AwaitSpawn;
                else if (status.mode == RosTopics.ModeMirror)
                    mode = ConveyorBridgeUiMode.MirrorIsaac;
                ApplyModeLocal();
                RefreshUi();
            }
            else if (topic == RosTopics.ConveyorRobotState)
            {
                if (RosJson.TryParseRobotState(json, out var state, out _))
                {
                    _targetColor = state.target_color;
                    if (targetColorText != null)
                    {
                        string name = string.IsNullOrEmpty(state.target_color_name)
                            ? ColorName(_targetColor)
                            : state.target_color_name;
                        targetColorText.text = "Target: " + name.ToUpperInvariant();
                    }
                }
            }
        }

        static string ColorName(int c) => c switch { 0 => "red", 1 => "green", 2 => "blue", _ => "?" };

        void RefreshUi()
        {
            ApplyModeLocal();
            if (modeText != null)
            {
                modeText.text = mode == ConveyorBridgeUiMode.AwaitSpawn
                    ? "Mode: Await Unity spawn"
                    : "Mode: Mirror Isaac";
            }
            if (connectButtonLabel != null)
                connectButtonLabel.text = (client != null && client.IsConnected) ? "Disconnect" : "Connect bridge";
            bool showSpawn = mode == ConveyorBridgeUiMode.AwaitSpawn;
            if (spawnRedButton != null)
                spawnRedButton.gameObject.SetActive(showSpawn);
            if (spawnGreenButton != null)
                spawnGreenButton.gameObject.SetActive(showSpawn);
            if (spawnBlueButton != null)
                spawnBlueButton.gameObject.SetActive(showSpawn);
            RefreshStatusLine();
        }

        void RefreshStatusLine()
        {
            if (statusText == null)
                return;
            bool connected = client != null && client.IsConnected;
            string tip = mode == ConveyorBridgeUiMode.AwaitSpawn
                ? "Spawn R/G/B onto the belt. Isaac picks the target color."
                : "Robot + cubes mirror Isaac (auto-spawn).";
            statusText.text =
                (connected ? "Bridge: CONNECTED :9091" : "Bridge: disconnected") +
                $"\nPhase: {_phase}  Policy: {(_policyLoaded ? "loaded" : "none")}" +
                $"\n{tip}";
        }
    }
}
