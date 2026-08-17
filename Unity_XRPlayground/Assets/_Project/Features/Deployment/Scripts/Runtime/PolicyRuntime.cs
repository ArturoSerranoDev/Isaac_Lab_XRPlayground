using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class PolicyRuntime : MonoBehaviour
    {
        public ModelAsset modelAsset;
        public TextAsset policyContractAsset;
        public string expectedPolicyId;
        public string expectedStationId;
        public BackendType backend = BackendType.CPU;
        [Tooltip("Diagnostics only. Production uses bundles promoted with evaluation.ready=true.")]
        public bool allowUnpromotedCandidate;

        Worker _worker;
        PolicyContract _contract;
        bool _ready;
        float[] _lastRawActions;
        TensorShape[] _inputShapes;
        float[][] _recurrentValues;

        public bool IsReady => _ready;
        public PolicyContract Contract => _contract;
        public float[] LastRawActions =>
            _lastRawActions != null ? (float[])_lastRawActions.Clone() : null;
        public event Action<float[], float[]> InferenceCompleted;

        void OnEnable()
        {
            if (modelAsset != null && policyContractAsset != null)
                TryLoad(out _);
        }

        void OnDisable() => DisposeWorker();
        void OnDestroy() => DisposeWorker();

        [ContextMenu("Reload Policy Bundle")]
        public void ReloadFromContextMenu()
        {
            if (!TryLoad(out string error))
                Debug.LogError($"PolicyRuntime: {error}", this);
        }

        public bool TryLoad(out string error)
        {
            DisposeWorker();
            if (modelAsset == null)
            {
                error = "policy.onnx ModelAsset is not assigned";
                return false;
            }
            if (!PolicyContract.TryParse(policyContractAsset, out _contract, out error))
                return false;
            if (!allowUnpromotedCandidate && (_contract.evaluation == null || !_contract.evaluation.ready))
            {
                error = "policy bundle is a candidate and has not passed promotion gates";
                return false;
            }
            if (!string.IsNullOrEmpty(expectedPolicyId) && _contract.policy_id != expectedPolicyId)
            {
                error = $"policy '{_contract.policy_id}' does not match '{expectedPolicyId}'";
                return false;
            }
            if (!string.IsNullOrEmpty(expectedStationId) && _contract.station_id != expectedStationId)
            {
                error = $"station '{_contract.station_id}' does not match '{expectedStationId}'";
                return false;
            }
            try
            {
                Model model = ModelLoader.Load(modelAsset);
                if (model.inputs.Count != _contract.onnx.inputs.Length ||
                    model.outputs.Count != _contract.onnx.outputs.Length)
                {
                    error = $"model tensor counts {model.inputs.Count}/{model.outputs.Count} " +
                        $"do not match contract {_contract.onnx.inputs.Length}/" +
                        $"{_contract.onnx.outputs.Length}";
                    return false;
                }
                _inputShapes = new TensorShape[model.inputs.Count];
                for (int i = 0; i < model.inputs.Count; i++)
                {
                    PolicyTensorInfo expected = _contract.onnx.inputs[i];
                    if (model.inputs[i].name != expected.name || !model.inputs[i].shape.IsStatic() ||
                        !ShapesEqual(model.inputs[i].shape.ToIntArray(), expected.shape))
                    {
                        error = $"model input {i} does not match policy.contract.json";
                        return false;
                    }
                    _inputShapes[i] = model.inputs[i].shape.ToTensorShape();
                }
                for (int i = 0; i < model.outputs.Count; i++)
                {
                    if (model.outputs[i].name != _contract.onnx.outputs[i].name)
                    {
                        error = $"model output {i} does not match policy.contract.json";
                        return false;
                    }
                }
                _recurrentValues = new float[Mathf.Max(0, model.inputs.Count - 1)][];
                for (int i = 0; i < _recurrentValues.Length; i++)
                    _recurrentValues[i] = new float[FlatSize(_contract.onnx.inputs[i + 1].shape)];
                _worker = new Worker(model, backend);
                _ready = true;
                error = null;
                return true;
            }
            catch (Exception exception)
            {
                error = $"could not load ONNX: {exception.Message}";
                DisposeWorker();
                return false;
            }
        }

        public bool TryInfer(float[] observations, float[] processedActions, out string error)
        {
            if (!_ready && !TryLoad(out error))
                return false;
            if (observations == null || observations.Length != _contract.ObservationDimension)
            {
                error = $"observation length {observations?.Length ?? 0} != {_contract.ObservationDimension}";
                return false;
            }
            if (processedActions == null || processedActions.Length != _contract.ActionDimension)
            {
                error = $"action length {processedActions?.Length ?? 0} != {_contract.ActionDimension}";
                return false;
            }
            try
            {
                var inputs = new Tensor[_inputShapes.Length];
                try
                {
                    inputs[0] = new Tensor<float>(_inputShapes[0], observations);
                    for (int i = 1; i < inputs.Length; i++)
                        inputs[i] = new Tensor<float>(_inputShapes[i], _recurrentValues[i - 1]);
                    _worker.Schedule(inputs);
                    if (!TryReadOutput(
                            _contract.onnx.output_name,
                            processedActions.Length,
                            out float[] raw,
                            out error))
                        return false;
                    var nextState = new float[_recurrentValues.Length][];
                    for (int i = 0; i < nextState.Length; i++)
                    {
                        PolicyTensorInfo state = _contract.onnx.outputs[i + 1];
                        if (!TryReadOutput(
                                state.name, FlatSize(state.shape), out nextState[i], out error))
                            return false;
                    }
                    _recurrentValues = nextState;
                    Array.Copy(raw, processedActions, raw.Length);
                    _contract.ClipActionsInPlace(processedActions);
                    _lastRawActions = (float[])raw.Clone();
                }
                finally
                {
                    foreach (Tensor input in inputs)
                        input?.Dispose();
                }
                InferenceCompleted?.Invoke(
                    (float[])_lastRawActions.Clone(),
                    (float[])processedActions.Clone());
                error = null;
                return true;
            }
            catch (Exception exception)
            {
                error = $"inference failed: {exception.Message}";
                return false;
            }
        }

        public void ResetState()
        {
            if (_recurrentValues == null)
                return;
            foreach (float[] state in _recurrentValues)
                if (state != null)
                    Array.Clear(state, 0, state.Length);
        }

        bool TryReadOutput(string name, int expectedLength, out float[] values, out string error)
        {
            values = null;
            Tensor<float> output = _worker.PeekOutput(name) as Tensor<float>;
            if (output == null)
            {
                error = $"model output '{name}' is null or not float";
                return false;
            }
            using var cpu = output.ReadbackAndClone();
            values = cpu.DownloadToArray();
            if (values.Length != expectedLength)
            {
                error = $"model output '{name}' length {values.Length} != {expectedLength}";
                values = null;
                return false;
            }
            error = null;
            return true;
        }

        void DisposeWorker()
        {
            _worker?.Dispose();
            _worker = null;
            _ready = false;
            _lastRawActions = null;
            _inputShapes = null;
            _recurrentValues = null;
        }

        static bool ShapesEqual(int[] actual, int[] expected)
        {
            if (actual == null || expected == null || actual.Length != expected.Length)
                return false;
            for (int i = 0; i < actual.Length; i++)
                if (actual[i] != expected[i])
                    return false;
            return true;
        }

        static int FlatSize(int[] shape)
        {
            int result = 1;
            foreach (int dimension in shape)
                result *= dimension;
            return result;
        }
    }
}
