using UnityEngine;
using UnityEngine.UI;

namespace XRPlayground.Policies
{
    /// <summary>
    /// World-space "Network" panel beside bridge / offline-policy controls.
    /// Listens to <see cref="OnnxPolicyRunner"/> while Offline Policy is running.
    /// Reusable across BallCatch / Conveyor / PickPlace stations.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PolicyNetworkPanel : MonoBehaviour
    {
        [Header("Sources")]
        public OnnxPolicyRunner policyRunner;
        [Tooltip("Optional BallCatch controller — panel goes live when running.")]
        public BallCatchOfflinePolicyController ballCatchPolicy;
        [Tooltip("Optional Conveyor controller.")]
        public ConveyorOfflinePolicyController conveyorPolicy;
        [Tooltip("Optional PickPlace controller.")]
        public PickPlaceOfflinePolicyController pickPlacePolicy;
        [Tooltip("Optional BalanceBot controller.")]
        public BalanceBotOfflinePolicyController balanceBotPolicy;
        [Tooltip("Optional Spot follow/loco controller.")]
        public SpotOfflinePolicyController spotPolicy;

        [Header("UI")]
        public PolicyNetworkVisualizer visualizer;
        public Text titleText;
        public Text architectureText;
        public Text statusText;
        public GameObject liveBadge;
        public CanvasGroup canvasGroup;

        [Header("Behaviour")]
        [Tooltip("Dim the panel when offline policy is not running (still visible for architecture).")]
        public float idleAlpha = 0.55f;
        public float liveAlpha = 1f;
        [Tooltip("Fallback architecture when model not yet loaded.")]
        public int[] fallbackLayerSizes = { 30, 256, 128, 64, 8 };

        bool _subscribed;
        bool _wasRunning;

        void OnEnable()
        {
            Subscribe(true);
            RefreshStaticLabels();
            if (visualizer != null)
                visualizer.EnsureArchitecture(ResolveFallbackSizes(), ArchitectureFallbackLabel());
            ApplyRunningState(IsAnyPolicyRunning(), force: true);
        }

        void OnDisable() => Subscribe(false);

        void Update()
        {
            bool running = IsAnyPolicyRunning();
            if (running != _wasRunning)
                ApplyRunningState(running, force: false);

            // If runner loads mid-session, rebuild labels.
            if (policyRunner != null && !string.IsNullOrEmpty(policyRunner.ArchitectureLabel))
            {
                if (architectureText != null && architectureText.text != policyRunner.ArchitectureLabel)
                    architectureText.text = policyRunner.ArchitectureLabel;
            }
        }

        public void BindBallCatch(OnnxPolicyRunner runner, BallCatchOfflinePolicyController policy)
        {
            Bind(runner, policy, null, null, null, null);
            fallbackLayerSizes = new[] { 30, 256, 128, 64, 8 };
            RefreshAfterBind();
        }

        public void BindConveyor(OnnxPolicyRunner runner, ConveyorOfflinePolicyController policy)
        {
            Bind(runner, null, policy, null, null, null);
            fallbackLayerSizes = new[] { 66, 256, 128, 64, 7 };
            RefreshAfterBind();
        }

        public void BindPickPlace(OnnxPolicyRunner runner, PickPlaceOfflinePolicyController policy)
        {
            Bind(runner, null, null, policy, null, null);
            fallbackLayerSizes = new[] { 30, 256, 128, 64, 8 };
            RefreshAfterBind();
        }

        public void BindBalanceBot(OnnxPolicyRunner runner, BalanceBotOfflinePolicyController policy)
        {
            Bind(runner, null, null, null, policy, null);
            fallbackLayerSizes = new[] { 20, 256, 128, 64, 2 };
            RefreshAfterBind();
        }

        public void BindSpot(OnnxPolicyRunner runner, SpotOfflinePolicyController policy)
        {
            Bind(runner, null, null, null, null, policy);
            fallbackLayerSizes = new[] { 48, 512, 256, 128, 12 };
            RefreshAfterBind();
        }

        void Bind(
            OnnxPolicyRunner runner,
            BallCatchOfflinePolicyController ballCatch,
            ConveyorOfflinePolicyController conveyor,
            PickPlaceOfflinePolicyController pickPlace,
            BalanceBotOfflinePolicyController balanceBot,
            SpotOfflinePolicyController spot)
        {
            Subscribe(false);
            policyRunner = runner;
            ballCatchPolicy = ballCatch;
            conveyorPolicy = conveyor;
            pickPlacePolicy = pickPlace;
            balanceBotPolicy = balanceBot;
            spotPolicy = spot;
            Subscribe(true);
        }

        void RefreshAfterBind()
        {
            RefreshStaticLabels();
            if (visualizer != null)
                visualizer.EnsureArchitecture(ResolveFallbackSizes(), ArchitectureFallbackLabel());
            ApplyRunningState(IsAnyPolicyRunning(), force: true);
        }

        void Subscribe(bool on)
        {
            if (policyRunner == null)
            {
                _subscribed = false;
                return;
            }
            if (on && !_subscribed)
            {
                policyRunner.InferenceCompleted += OnInference;
                policyRunner.captureActivations = true;
                _subscribed = true;
            }
            else if (!on && _subscribed)
            {
                policyRunner.InferenceCompleted -= OnInference;
                _subscribed = false;
            }
        }

        void OnInference(PolicyInferenceSnapshot snap)
        {
            if (visualizer != null)
                visualizer.ApplySnapshot(snap);
            if (statusText != null)
            {
                statusText.text = snap.HasRealIntermediates
                    ? "Highlight = |activation| · real ELU tensors"
                    : "Highlight = |obs|/|act| · hidden approx unavailable";
            }
            if (architectureText != null && !string.IsNullOrEmpty(snap.ArchitectureLabel))
                architectureText.text = snap.ArchitectureLabel;
        }

        void ApplyRunningState(bool running, bool force)
        {
            _wasRunning = running;
            if (canvasGroup != null)
                canvasGroup.alpha = running ? liveAlpha : idleAlpha;
            if (liveBadge != null)
                liveBadge.SetActive(running);
            if (!running && statusText != null)
                statusText.text = IdleStatusMessage();
            if (force || running)
                RefreshStaticLabels();
        }

        string IdleStatusMessage()
        {
            if (policyRunner == null)
                return "Wire OnnxPolicyRunner to this panel";
            if (policyRunner.modelAsset == null)
                return "Assign policy.onnx — panel lights up when Offline Policy runs";
            return "Start Offline Policy to animate activations";
        }

        void RefreshStaticLabels()
        {
            if (titleText != null)
                titleText.text = "Network";
            if (architectureText != null)
            {
                architectureText.text = policyRunner != null && !string.IsNullOrEmpty(policyRunner.ArchitectureLabel)
                    ? policyRunner.ArchitectureLabel
                    : ArchitectureFallbackLabel();
            }
        }

        bool IsAnyPolicyRunning()
        {
            if (ballCatchPolicy != null && ballCatchPolicy.running) return true;
            if (conveyorPolicy != null && conveyorPolicy.running) return true;
            if (pickPlacePolicy != null && pickPlacePolicy.running) return true;
            if (balanceBotPolicy != null && balanceBotPolicy.running) return true;
            if (spotPolicy != null && spotPolicy.running) return true;
            return false;
        }

        int[] ResolveFallbackSizes()
        {
            if (policyRunner != null)
            {
                int obs = policyRunner.expectedObsDim;
                int act = policyRunner.expectedActionDim;
                return new[] { obs, 256, 128, 64, act };
            }
            return fallbackLayerSizes != null && fallbackLayerSizes.Length >= 2
                ? fallbackLayerSizes
                : new[] { 30, 256, 128, 64, 8 };
        }

        string ArchitectureFallbackLabel()
        {
            var sizes = ResolveFallbackSizes();
            return string.Join(" → ", sizes);
        }
    }
}
