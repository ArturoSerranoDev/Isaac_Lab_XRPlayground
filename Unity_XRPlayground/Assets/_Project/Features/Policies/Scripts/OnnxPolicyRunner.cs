using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Loads an RSL-RL ONNX (<see cref="ModelAsset"/>) and runs deterministic mean actions on CPU.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class OnnxPolicyRunner : MonoBehaviour
    {
        [Tooltip("Assign the imported policy.onnx (Inference Engine ModelAsset).")]
        public ModelAsset modelAsset;

        [Tooltip("Must match training observation size (Conveyor=66, BallCatch=28).")]
        public int expectedObsDim = 66;

        [Tooltip("Must match training action size (Conveyor=7, BallCatch=8).")]
        public int expectedActionDim = 7;

        public BackendType backend = BackendType.CPU;

        Worker _worker;
        Model _model;
        float[] _actionScratch;
        bool _ready;

        public bool IsReady => _ready;

        void OnEnable()
        {
            if (modelAsset != null)
                TryLoad();
        }

        void OnDisable() => DisposeWorker();

        void OnDestroy() => DisposeWorker();

        [ContextMenu("Reload Model")]
        public bool TryLoad()
        {
            DisposeWorker();
            _ready = false;
            if (modelAsset == null)
            {
                Debug.LogWarning("OnnxPolicyRunner: assign a ModelAsset (policy.onnx).", this);
                return false;
            }

            try
            {
                _model = ModelLoader.Load(modelAsset);
                _worker = new Worker(_model, backend);
                _actionScratch = new float[Mathf.Max(1, expectedActionDim)];
                _ready = true;
                Debug.Log(
                    $"OnnxPolicyRunner: loaded '{modelAsset.name}' (obs={expectedObsDim}, act={expectedActionDim}, {backend}).",
                    this);
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"OnnxPolicyRunner: failed to load model — {ex.Message}", this);
                DisposeWorker();
                return false;
            }
        }

        /// <summary>Run policy; writes into <paramref name="actionsOut"/> (length >= expectedActionDim).</summary>
        public bool TryInfer(float[] obs, float[] actionsOut)
        {
            if (!_ready && !TryLoad())
                return false;
            if (obs == null || obs.Length < expectedObsDim)
            {
                Debug.LogError($"OnnxPolicyRunner: obs length {obs?.Length ?? 0} < {expectedObsDim}", this);
                return false;
            }
            if (actionsOut == null || actionsOut.Length < expectedActionDim)
            {
                Debug.LogError($"OnnxPolicyRunner: actionsOut too small", this);
                return false;
            }

            using var input = new Tensor<float>(new TensorShape(1, expectedObsDim), obs);
            _worker.Schedule(input);
            var peek = _worker.PeekOutput() as Tensor<float>;
            if (peek == null)
            {
                Debug.LogError("OnnxPolicyRunner: null output tensor", this);
                return false;
            }

            using var cpu = peek.ReadbackAndClone();
            var data = cpu.DownloadToArray();
            int n = Mathf.Min(expectedActionDim, data.Length);
            for (int i = 0; i < n; i++)
                actionsOut[i] = Mathf.Clamp(data[i], -1f, 1f);
            for (int i = n; i < expectedActionDim; i++)
                actionsOut[i] = 0f;
            return true;
        }

        void DisposeWorker()
        {
            _worker?.Dispose();
            _worker = null;
            _model = null;
            _ready = false;
        }
    }
}
