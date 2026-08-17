using UnityEngine;
using UnityEngine.UI;
using XRPlayground.Deployment;
using XRPlayground.Policies;

namespace XRPlayground.ROS
{
    /// <summary>
    /// World-space UI for Spot bridge (port 9094) + offline ONNX follow.
    /// </summary>
    public sealed class SpotBridgePanel : MonoBehaviour
    {
        public RosTcpClient client;
        public SpotLinkPoseFollower robotFollower;
        public SpotPlayerTargetPublisher playerPublisher;
        public SpotOfflinePolicyController offlinePolicy;
        public DeploymentModeController deploymentModes;
        public string stationId = "spot_follow";

        public Text statusText;
        public Text modeText;
        public Button connectButton;
        public Button mirrorButton;
        public Button startPolicyButton;
        public Text connectButtonLabel;
        public Text startPolicyButtonLabel;

        string _phase = "-";
        bool _policyLoaded;
        bool _wired;

        void OnEnable()
        {
            if (deploymentModes == null)
                deploymentModes = FindAnyObjectByType<DeploymentModeController>();
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
                mirrorButton.onClick.AddListener(SetMirror);
            if (startPolicyButton != null)
                startPolicyButton.onClick.AddListener(ToggleOfflinePolicy);
        }

        void ToggleConnect()
        {
            if (client == null)
                return;
            if (client.IsConnected)
                client.Disconnect();
            else
                client.Connect();
            RefreshUi();
        }

        void SetMirror()
        {
            if (deploymentModes != null)
            {
                deploymentModes.SelectStation(stationId);
                deploymentModes.SetMirror();
                RefreshUi();
                return;
            }
            if (robotFollower != null)
                robotFollower.followingEnabled = true;
            if (playerPublisher != null)
                playerPublisher.publishingEnabled = true;
            if (offlinePolicy != null && offlinePolicy.running)
                offlinePolicy.StopPolicy();
            RefreshUi();
        }

        void ToggleOfflinePolicy()
        {
            if (deploymentModes != null)
            {
                deploymentModes.SelectStation(stationId);
                StationRuntime runtime = FindDeploymentRuntime();
                if (runtime != null && runtime.Mode == StationMode.OfflinePolicy)
                    deploymentModes.SetDisabled();
                else
                    deploymentModes.SetOfflinePolicy();
                RefreshUi();
                return;
            }
            if (offlinePolicy == null)
                return;
            if (offlinePolicy.running)
                offlinePolicy.StopPolicy();
            else
            {
                if (robotFollower != null)
                    robotFollower.followingEnabled = false;
                offlinePolicy.StartPolicy();
            }
            RefreshUi();
        }

        void OnMessage(string json)
        {
            if (!RosJson.TryParseTopic(json, out var topic))
                return;
            if (topic != RosTopics.SessionStatus)
                return;
            if (!RosJson.TryParseSessionStatus(json, out var status) || status == null)
                return;
            _phase = status.phase ?? "-";
            _policyLoaded = status.policy_loaded;
            RefreshUi();
        }

        void RefreshUi()
        {
            StationRuntime runtime = FindDeploymentRuntime();
            bool offline = deploymentModes != null
                ? runtime != null && runtime.Mode == StationMode.OfflinePolicy
                : offlinePolicy != null && offlinePolicy.running;
            if (modeText != null)
                modeText.text = offline ? "Offline Follow" : "Mirror Isaac";
            if (connectButtonLabel != null)
                connectButtonLabel.text = client != null && client.IsConnected ? "Disconnect" : "Connect";
            if (startPolicyButtonLabel != null)
                startPolicyButtonLabel.text =
                    offline ? "Stop Offline Policy" : "Start Offline Policy";
            RefreshStatusLine();
        }

        void RefreshStatusLine()
        {
            if (statusText == null)
                return;
            string conn = client != null && client.IsConnected ? "connected" : "idle";
            StationRuntime runtime = FindDeploymentRuntime();
            bool genericOffline = runtime != null && runtime.Mode == StationMode.OfflinePolicy;
            string pol = genericOffline
                ? "local ONNX hierarchy"
                : offlinePolicy != null && offlinePolicy.running
                    ? offlinePolicy.StatusLine
                    : (_policyLoaded ? "isaac policy loaded" : "zero/random actions");
            statusText.text = $"Spot :9094 · {conn} · {_phase} · {pol}";
        }

        StationRuntime FindDeploymentRuntime()
        {
            if (deploymentModes == null)
                return null;
            DeploymentCatalogBinder binder = deploymentModes.binder != null
                ? deploymentModes.binder
                : FindAnyObjectByType<DeploymentCatalogBinder>();
            if (binder == null)
                return null;
            foreach (StationRuntime runtime in binder.GetComponentsInChildren<StationRuntime>(true))
                if (runtime.stationId == stationId)
                    return runtime;
            return null;
        }
    }
}
