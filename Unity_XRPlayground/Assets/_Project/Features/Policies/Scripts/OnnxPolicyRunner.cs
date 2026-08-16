using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Loads an RSL-RL ONNX (<see cref="ModelAsset"/>) and runs deterministic mean actions on CPU.
    /// Optionally peeks ELU / Dense intermediates for <see cref="PolicyNetworkVisualizer"/>.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class OnnxPolicyRunner : MonoBehaviour
    {
        [Tooltip("Assign the imported policy.onnx (Inference Engine ModelAsset).")]
        public ModelAsset modelAsset;

        [Tooltip("The policy.json exported alongside modelAsset. Editor auto-discovers the adjacent sidecar.")]
        public TextAsset policyMetadata;

        [Tooltip("Optional station task contract. A different policy task is rejected before inference.")]
        public string expectedTaskId;

        [Tooltip("Must match training observation size (Conveyor=66, BallCatch=30, PickPlace=30).")]
        public int expectedObsDim = 66;

        [Tooltip("Must match training action size (Conveyor=7, BallCatch=8).")]
        public int expectedActionDim = 7;

        public BackendType backend = BackendType.CPU;

        [Header("Network viz")]
        [Tooltip("Add Inference Engine outputs on ELU / Dense layers and copy activations each infer.")]
        public bool captureActivations = true;

        Worker _worker;
        Model _model;
        float[] _actionScratch;
        bool _ready;

        readonly List<(string name, int tensorIndex, int sizeHint)> _probeOutputs = new();
        readonly PolicyInferenceSnapshot _snapshot = new();
        float[][] _layerBuffers = Array.Empty<float[]>();
        int[] _layerSizes = Array.Empty<int>();
        string _architectureLabel = "";
        bool _hasRealIntermediates;

        PolicyMetadata _metadata;

        public bool IsReady => _ready;
        public PolicyInferenceSnapshot LatestSnapshot => _snapshot;
        public string ArchitectureLabel => _architectureLabel;
        public bool HasRealIntermediates => _hasRealIntermediates;
        public PolicyMetadata Metadata => _metadata;
        public event Action<PolicyInferenceSnapshot> InferenceCompleted;

        void OnEnable()
        {
            if (modelAsset != null)
                TryLoad();
        }

        void OnDisable() => DisposeWorker();

        void OnDestroy() => DisposeWorker();

        void OnValidate()
        {
#if UNITY_EDITOR
            AutoAssignAdjacentMetadata();
#endif
        }

        [ContextMenu("Reload Model")]
        public bool TryLoad()
        {
            DisposeWorker();
            _ready = false;
            _hasRealIntermediates = false;
            _probeOutputs.Clear();
            if (modelAsset == null)
            {
                Debug.LogWarning("OnnxPolicyRunner: assign a ModelAsset (policy.onnx).", this);
                return false;
            }

            if (!TryApplyPolicyMetadata(out var metadataError))
            {
                Debug.LogError($"OnnxPolicyRunner: {metadataError}", this);
                return false;
            }

            try
            {
                _model = ModelLoader.Load(modelAsset);
                if (captureActivations)
                    ConfigureActivationProbes(_model);
                _worker = new Worker(_model, backend);
                _actionScratch = new float[Mathf.Max(1, expectedActionDim)];
                EnsureLayerBuffers();
                _ready = true;
                Debug.Log(
                    $"OnnxPolicyRunner: loaded '{modelAsset.name}' (obs={expectedObsDim}, act={expectedActionDim}, " +
                    $"{backend}, arch={_architectureLabel}, intermediates={_hasRealIntermediates}).",
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

        /// <summary>Reads the sidecar and applies its dimensions before the model worker is created.</summary>
        public bool TryApplyPolicyMetadata(out string error)
        {
#if UNITY_EDITOR
            AutoAssignAdjacentMetadata();
#endif
            if (!PolicyMetadata.TryParse(policyMetadata, out _metadata, out error))
                return false;
            if (!string.IsNullOrEmpty(expectedTaskId) && _metadata.task_id != expectedTaskId)
            {
                error = $"policy task '{_metadata.task_id}' does not match station task '{expectedTaskId}'";
                return false;
            }

            expectedObsDim = _metadata.obs_dim;
            expectedActionDim = _metadata.action_dim;
            foreach (var behaviour in GetComponents<MonoBehaviour>())
            {
                if (behaviour is IPolicyMetadataConsumer consumer)
                    consumer.ApplyPolicyMetadata(_metadata);
            }
            error = null;
            return true;
        }

        /// <summary>Checks the loaded sidecar against a controller or station contract.</summary>
        public bool MatchesContract(string taskId, int obsDim, int actionDim, out string error)
        {
            if (!TryApplyPolicyMetadata(out error))
                return false;
            if (_metadata.task_id != taskId || _metadata.obs_dim != obsDim || _metadata.action_dim != actionDim)
            {
                error = $"expected {taskId} (obs={obsDim}, action={actionDim}); sidecar declares " +
                    $"{_metadata.task_id} (obs={_metadata.obs_dim}, action={_metadata.action_dim})";
                return false;
            }
            error = null;
            return true;
        }

#if UNITY_EDITOR
        void AutoAssignAdjacentMetadata()
        {
            if (modelAsset == null)
                return;
            string modelPath = UnityEditor.AssetDatabase.GetAssetPath(modelAsset);
            if (string.IsNullOrEmpty(modelPath))
                return;
            string directory = System.IO.Path.GetDirectoryName(modelPath)?.Replace('\\', '/');
            if (string.IsNullOrEmpty(directory))
                return;
            var adjacent = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>($"{directory}/policy.json");
            if (adjacent != null && adjacent != policyMetadata)
            {
                policyMetadata = adjacent;
                UnityEditor.EditorUtility.SetDirty(this);
            }
        }
#endif

        /// <summary>Run policy; input/output dimensions must exactly match the exported sidecar contract.</summary>
        public bool TryInfer(float[] obs, float[] actionsOut)
        {
            if (!_ready && !TryLoad())
                return false;
            if (obs == null || obs.Length != expectedObsDim)
            {
                Debug.LogError($"OnnxPolicyRunner: obs length {obs?.Length ?? 0} != {expectedObsDim}", this);
                return false;
            }
            if (actionsOut == null || actionsOut.Length != expectedActionDim)
            {
                Debug.LogError($"OnnxPolicyRunner: action buffer length {actionsOut?.Length ?? 0} != {expectedActionDim}", this);
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
            if (data.Length != expectedActionDim)
            {
                Debug.LogError($"OnnxPolicyRunner: model output length {data.Length} != sidecar action_dim {expectedActionDim}", this);
                return false;
            }
            for (int i = 0; i < expectedActionDim; i++)
                actionsOut[i] = Mathf.Clamp(data[i], -1f, 1f);

            if (captureActivations)
                CaptureSnapshot(obs, actionsOut);

            return true;
        }

        void ConfigureActivationProbes(Model model)
        {
            _probeOutputs.Clear();
            var denseOutSizes = new List<int>();

            foreach (var layer in model.layers)
            {
                string typeName = layer.GetType().Name;
                if (typeName == "Dense" || typeName == "DenseBatched")
                {
                    int size = ResolveDenseOutputSize(model, layer);
                    if (size > 0)
                        denseOutSizes.Add(size);
                }
            }

            // Prefer post-ELU tensors (true hidden activations for RSL-RL MLP).
            int eluCount = 0;
            foreach (var layer in model.layers)
            {
                if (layer.GetType().Name != "Elu")
                    continue;
                int sizeHint = eluCount < denseOutSizes.Count ? denseOutSizes[eluCount] : 0;
                string name = $"viz_elu_{eluCount}";
                model.AddOutput(name, layer.outputs[0]);
                _probeOutputs.Add((name, layer.outputs[0], sizeHint));
                eluCount++;
            }

            // Fallback: Dense outputs if the export fused activations (no separate Elu).
            if (_probeOutputs.Count == 0)
            {
                int denseIdx = 0;
                foreach (var layer in model.layers)
                {
                    string typeName = layer.GetType().Name;
                    if (typeName != "Dense" && typeName != "DenseBatched")
                        continue;
                    // Skip final action Dense — already the model output.
                    if (denseIdx >= denseOutSizes.Count - 1)
                        break;
                    int sizeHint = denseOutSizes[denseIdx];
                    string name = $"viz_dense_{denseIdx}";
                    model.AddOutput(name, layer.outputs[0]);
                    _probeOutputs.Add((name, layer.outputs[0], sizeHint));
                    denseIdx++;
                }
            }

            _hasRealIntermediates = _probeOutputs.Count > 0;

            var sizes = new List<int> { expectedObsDim };
            foreach (var p in _probeOutputs)
            {
                if (p.sizeHint > 0)
                    sizes.Add(p.sizeHint);
            }
            sizes.Add(expectedActionDim);
            _layerSizes = sizes.ToArray();
            _architectureLabel = string.Join(" → ", _layerSizes);
        }

        static int ResolveDenseOutputSize(Model model, Layer layer)
        {
            if (layer.inputs == null || layer.inputs.Length < 3)
                return 0;
            int biasIndex = layer.inputs[2];
            foreach (var c in model.constants)
            {
                if (c.index != biasIndex)
                    continue;
                // Bias is 1-D (outFeatures) or last dim of shape.
                if (c.shape.rank >= 1)
                    return c.shape[c.shape.rank - 1];
            }
            int weightIndex = layer.inputs[1];
            foreach (var c in model.constants)
            {
                if (c.index != weightIndex)
                    continue;
                if (c.shape.rank >= 2)
                    return c.shape[c.shape.rank - 1];
            }
            return 0;
        }

        void EnsureLayerBuffers()
        {
            if (_layerSizes == null || _layerSizes.Length == 0)
            {
                _layerSizes = new[] { expectedObsDim, 256, 128, 64, expectedActionDim };
                _architectureLabel = string.Join(" → ", _layerSizes) + " (default)";
            }

            _layerBuffers = new float[_layerSizes.Length][];
            for (int i = 0; i < _layerSizes.Length; i++)
                _layerBuffers[i] = new float[Mathf.Max(1, _layerSizes[i])];

            _snapshot.LayerSizes = _layerSizes;
            _snapshot.LayerActivations = _layerBuffers;
            _snapshot.ArchitectureLabel = _architectureLabel;
            _snapshot.HasRealIntermediates = _hasRealIntermediates;
        }

        void CaptureSnapshot(float[] obs, float[] actions)
        {
            if (_layerBuffers == null || _layerBuffers.Length == 0)
                EnsureLayerBuffers();

            // Input column: |obs|
            var inputBuf = _layerBuffers[0];
            int inN = Mathf.Min(inputBuf.Length, expectedObsDim, obs.Length);
            for (int i = 0; i < inN; i++)
                inputBuf[i] = Mathf.Abs(obs[i]);
            for (int i = inN; i < inputBuf.Length; i++)
                inputBuf[i] = 0f;

            // Hidden probes
            int hiddenLayer = 1;
            for (int p = 0; p < _probeOutputs.Count && hiddenLayer < _layerBuffers.Length - 1; p++, hiddenLayer++)
            {
                CopyPeekToBuffer(_probeOutputs[p].name, _layerBuffers[hiddenLayer]);
            }

            // If probes missing / fewer than expected, leave remaining hidden as zero (still show structure).
            for (; hiddenLayer < _layerBuffers.Length - 1; hiddenLayer++)
            {
                var buf = _layerBuffers[hiddenLayer];
                Array.Clear(buf, 0, buf.Length);
            }

            // Output column: |actions|
            var outBuf = _layerBuffers[_layerBuffers.Length - 1];
            int outN = Mathf.Min(outBuf.Length, expectedActionDim, actions.Length);
            for (int i = 0; i < outN; i++)
                outBuf[i] = Mathf.Abs(actions[i]);
            for (int i = outN; i < outBuf.Length; i++)
                outBuf[i] = 0f;

            _snapshot.LayerSizes = _layerSizes;
            _snapshot.LayerActivations = _layerBuffers;
            _snapshot.HasRealIntermediates = _hasRealIntermediates;
            _snapshot.ArchitectureLabel = _architectureLabel;
            _snapshot.SourceLabel = _hasRealIntermediates
                ? "real ELU intermediates"
                : "obs/actions only (no hidden probes)";

            InferenceCompleted?.Invoke(_snapshot);
        }

        void CopyPeekToBuffer(string outputName, float[] dest)
        {
            try
            {
                var peek = _worker.PeekOutput(outputName) as Tensor<float>;
                if (peek == null)
                {
                    Array.Clear(dest, 0, dest.Length);
                    return;
                }
                using var cpu = peek.ReadbackAndClone();
                var data = cpu.DownloadToArray();
                int n = Mathf.Min(dest.Length, data.Length);
                for (int i = 0; i < n; i++)
                    dest[i] = Mathf.Abs(data[i]);
                for (int i = n; i < dest.Length; i++)
                    dest[i] = 0f;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"OnnxPolicyRunner: failed peek '{outputName}': {ex.Message}", this);
                Array.Clear(dest, 0, dest.Length);
            }
        }

        void DisposeWorker()
        {
            _worker?.Dispose();
            _worker = null;
            _model = null;
            _ready = false;
            _probeOutputs.Clear();
        }
    }
}
