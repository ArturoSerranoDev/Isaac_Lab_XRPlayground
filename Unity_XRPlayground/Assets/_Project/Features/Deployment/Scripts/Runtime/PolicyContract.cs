using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class PolicyTensorInfo
    {
        public string name;
        public int[] shape;
    }

    [Serializable]
    public sealed class PolicyOnnxContract
    {
        public string sha256;
        public int opset;
        public string input_name;
        public int[] input_shape;
        public string output_name;
        public int[] output_shape;
        public PolicyTensorInfo[] inputs;
        public PolicyTensorInfo[] outputs;
    }

    [Serializable]
    public sealed class PolicyTerm
    {
        public string name;
        public int[] shape;
        public string dtype;
        public string frame;
        public string units;
        public float scale = 1f;
        public float offset;
        public float[] scale_values;
        public float[] offset_values;
        public float[] velocity_scale;
        public int history = 1;
        public string[] names;
        public string target_type;
        public string integration;
        public float[] clip;
        public float[] target_range;

        public int FlatSize
        {
            get
            {
                if (shape == null || shape.Length == 0)
                    return 0;
                int value = 1;
                foreach (int dimension in shape)
                    value *= dimension;
                return value;
            }
        }

        public float ScaleAt(int index)
        {
            return scale_values != null && index >= 0 && index < scale_values.Length
                ? scale_values[index]
                : scale;
        }

        public float OffsetAt(int index)
        {
            return offset_values != null && index >= 0 && index < offset_values.Length
                ? offset_values[index]
                : offset;
        }

        public float VelocityScaleAt(int jointIndex)
        {
            return velocity_scale != null && jointIndex >= 0 && jointIndex < velocity_scale.Length
                ? velocity_scale[jointIndex]
                : 1f;
        }
    }

    [Serializable]
    public sealed class PolicyTiming
    {
        public int source_physics_hz;
        public int deployment_physics_hz;
        public int policy_hz;
        public float source_sim_dt;
        public float deployment_sim_dt;
        public float policy_dt;
        public int deployment_decimation;
    }

    [Serializable]
    public sealed class PolicyMetricThreshold
    {
        public string metric;
        public string comparison;
        public float value;
    }

    [Serializable]
    public sealed class PolicyEvaluation
    {
        public string primary_metric;
        public float minimum_isaac_score;
        public float minimum_unity_relative_score;
        public float torch_onnx_max_abs_error;
        public float torch_onnx_limit;
        public float python_unity_limit;
        public float observation_limit;
        public int seeded_scenarios;
        public PolicyMetricThreshold[] task_thresholds;
        public bool ready;
        public string release_id;
    }

    [Serializable]
    public sealed class PolicyDependency
    {
        public string policy_id;
        public string model_sha256;
        public string torchscript_sha256;
    }

    [Serializable]
    public sealed class PolicyContract
    {
        public int schema_version;
        public string policy_id;
        public string source_task_id;
        public string station_id;
        public string robot_id;
        public string adapter_id;
        public string checkpoint;
        public string checkpoint_sha256;
        public string config_sha256;
        public string model_sha256;
        public PolicyOnnxContract onnx;
        public bool normalization_embedded;
        public PolicyTensorInfo[] recurrent_state;
        public string torchscript_sha256;
        public PolicyDependency[] dependencies;
        public PolicyTerm[] observations;
        public PolicyTerm[] actions;
        public PolicyTiming timing;
        public string physics_profile_id;
        public string assist_profile_id;
        public PolicyEvaluation evaluation;

        public int ObservationDimension => SumDimensions(observations);
        public int ActionDimension => SumDimensions(actions);

        public static bool TryParse(TextAsset asset, out PolicyContract contract, out string error)
        {
            contract = null;
            error = null;
            if (asset == null || string.IsNullOrWhiteSpace(asset.text))
            {
                error = "policy.contract.json is not assigned";
                return false;
            }
            try
            {
                contract = JsonUtility.FromJson<PolicyContract>(asset.text);
            }
            catch (Exception exception)
            {
                error = $"policy contract JSON is invalid: {exception.Message}";
                return false;
            }
            return contract != null && contract.Validate(out error);
        }

        public bool Validate(out string error)
        {
            if (schema_version != 2)
                return Fail($"unsupported contract schema {schema_version}", out error);
            if (string.IsNullOrWhiteSpace(policy_id) || string.IsNullOrWhiteSpace(station_id) ||
                string.IsNullOrWhiteSpace(robot_id) || string.IsNullOrWhiteSpace(adapter_id))
                return Fail("contract IDs must be non-empty", out error);
            if (!normalization_embedded)
                return Fail(
                    "Unity deployment requires observation normalization embedded in ONNX",
                    out error);
            if (!IsSha256(checkpoint_sha256) || !IsSha256(config_sha256) || !IsSha256(model_sha256))
                return Fail("contract hashes must be lowercase SHA-256 values", out error);
            if (!string.IsNullOrEmpty(torchscript_sha256) && !IsSha256(torchscript_sha256))
                return Fail("TorchScript hash must be a lowercase SHA-256 value", out error);
            if (dependencies != null)
            {
                var dependencyIds = new System.Collections.Generic.HashSet<string>();
                foreach (PolicyDependency dependency in dependencies)
                    if (dependency == null || string.IsNullOrWhiteSpace(dependency.policy_id) ||
                        !dependencyIds.Add(dependency.policy_id) ||
                        !IsSha256(dependency.model_sha256) || !IsSha256(dependency.torchscript_sha256))
                        return Fail("policy dependencies must have unique IDs and valid hashes", out error);
            }
            if (onnx == null || onnx.opset != 15)
                return Fail("Unity Inference Engine 2.3 requires ONNX opset 15", out error);
            if (onnx.sha256 != model_sha256 || string.IsNullOrWhiteSpace(onnx.input_name) ||
                string.IsNullOrWhiteSpace(onnx.output_name))
                return Fail("ONNX hash and tensor names must match the bundle contract", out error);
            if (onnx.input_shape == null || onnx.input_shape.Length != 2 || onnx.input_shape[0] != 1 ||
                onnx.input_shape[1] != ObservationDimension)
                return Fail("ONNX input shape does not match ordered observations", out error);
            if (onnx.output_shape == null || onnx.output_shape.Length != 2 || onnx.output_shape[0] != 1 ||
                onnx.output_shape[1] != ActionDimension)
                return Fail("ONNX output shape does not match ordered actions", out error);
            if (!ValidateTensorSignature(out error))
                return false;
            if (timing == null || timing.deployment_physics_hz <= 0 || timing.policy_hz <= 0 ||
                timing.deployment_physics_hz % timing.policy_hz != 0)
                return Fail("policy cadence must evenly divide deployment physics", out error);
            int expectedDecimation = timing.deployment_physics_hz / timing.policy_hz;
            if (timing.source_physics_hz <= 0 || timing.deployment_decimation != expectedDecimation ||
                !Approximately(timing.source_sim_dt, 1f / timing.source_physics_hz) ||
                !Approximately(timing.deployment_sim_dt, 1f / timing.deployment_physics_hz) ||
                !Approximately(timing.policy_dt, 1f / timing.policy_hz))
                return Fail("policy timing values are internally inconsistent", out error);
            if (evaluation == null || evaluation.seeded_scenarios != 100 ||
                evaluation.task_thresholds == null || evaluation.task_thresholds.Length == 0)
                return Fail("evaluation must define the 100-seed task thresholds", out error);
            var thresholdMetrics = new System.Collections.Generic.HashSet<string>();
            foreach (PolicyMetricThreshold threshold in evaluation.task_thresholds)
                if (threshold == null || string.IsNullOrWhiteSpace(threshold.metric) ||
                    !thresholdMetrics.Add(threshold.metric) ||
                    (threshold.comparison != "min" && threshold.comparison != "max" &&
                     threshold.comparison != "equal") ||
                    float.IsNaN(threshold.value) || float.IsInfinity(threshold.value))
                    return Fail("evaluation task thresholds are invalid", out error);
            if (!ValidateTerms(observations, false, out error) || !ValidateTerms(actions, true, out error))
                return false;
            error = null;
            return true;
        }

        bool ValidateTensorSignature(out string error)
        {
            if (!ValidateTensorList(onnx.inputs, "input", out error) ||
                !ValidateTensorList(onnx.outputs, "output", out error))
                return false;
            if (!TensorEquals(onnx.inputs[0], onnx.input_name, onnx.input_shape) ||
                !TensorEquals(onnx.outputs[0], onnx.output_name, onnx.output_shape))
                return Fail("ONNX primary tensor descriptors are inconsistent", out error);
            if (onnx.inputs.Length != onnx.outputs.Length)
                return Fail(
                    "recurrent ONNX requires one state output for every state input",
                    out error);
            int stateCount = onnx.inputs.Length - 1;
            if ((recurrent_state?.Length ?? 0) != stateCount * 2)
                return Fail(
                    "recurrent_state must describe ordered extra ONNX inputs and outputs",
                    out error);
            for (int i = 0; i < stateCount; i++)
            {
                PolicyTensorInfo input = onnx.inputs[i + 1];
                PolicyTensorInfo output = onnx.outputs[i + 1];
                if (!ShapesEqual(input.shape, output.shape))
                    return Fail($"recurrent state pair {i + 1} has different shapes", out error);
                if (!TensorEquals(recurrent_state[i], input.name, input.shape) ||
                    !TensorEquals(recurrent_state[stateCount + i], output.name, output.shape))
                    return Fail("recurrent_state ordering differs from ONNX tensors", out error);
            }
            error = null;
            return true;
        }

        static bool ValidateTensorList(
            PolicyTensorInfo[] tensors,
            string kind,
            out string error)
        {
            if (tensors == null || tensors.Length == 0)
                return Fail($"ONNX {kind} tensors must be present", out error);
            var names = new System.Collections.Generic.HashSet<string>();
            foreach (PolicyTensorInfo tensor in tensors)
            {
                if (tensor == null || string.IsNullOrWhiteSpace(tensor.name) ||
                    !names.Add(tensor.name) || tensor.shape == null || tensor.shape.Length == 0)
                    return Fail($"ONNX {kind} tensor is invalid or duplicated", out error);
                foreach (int dimension in tensor.shape)
                    if (dimension <= 0)
                        return Fail($"ONNX {kind} tensor shape must be static and positive", out error);
            }
            error = null;
            return true;
        }

        static bool TensorEquals(PolicyTensorInfo tensor, string name, int[] shape) =>
            tensor != null && tensor.name == name && ShapesEqual(tensor.shape, shape);

        static bool ShapesEqual(int[] left, int[] right)
        {
            if (left == null || right == null || left.Length != right.Length)
                return false;
            for (int i = 0; i < left.Length; i++)
                if (left[i] != right[i])
                    return false;
            return true;
        }

        public void ClipActionsInPlace(float[] values)
        {
            if (values == null || actions == null)
                return;
            int cursor = 0;
            foreach (PolicyTerm term in actions)
            {
                int count = term?.FlatSize ?? 0;
                if (term?.clip != null && term.clip.Length == 2)
                {
                    for (int i = 0; i < count && cursor + i < values.Length; i++)
                        values[cursor + i] = Mathf.Clamp(values[cursor + i], term.clip[0], term.clip[1]);
                }
                cursor += count;
            }
        }

        static int SumDimensions(PolicyTerm[] terms)
        {
            int total = 0;
            if (terms != null)
                foreach (PolicyTerm term in terms)
                    total += term?.FlatSize ?? 0;
            return total;
        }

        static bool ValidateTerms(PolicyTerm[] terms, bool actions, out string error)
        {
            if (terms == null || terms.Length == 0)
                return Fail(actions ? "contract has no actions" : "contract has no observations", out error);
            foreach (PolicyTerm term in terms)
            {
                if (term == null || string.IsNullOrWhiteSpace(term.name) || term.FlatSize <= 0)
                    return Fail("contract contains an invalid policy term", out error);
                if (term.scale_values != null && term.scale_values.Length != term.FlatSize)
                    return Fail($"term '{term.name}' scale_values length does not match its shape", out error);
                if (term.offset_values != null && term.offset_values.Length != term.FlatSize)
                    return Fail($"term '{term.name}' offset_values length does not match its shape", out error);
                if (term.velocity_scale != null &&
                    (term.names == null || term.velocity_scale.Length != term.names.Length))
                    return Fail($"term '{term.name}' velocity_scale length does not match its joints", out error);
                if (actions && string.IsNullOrWhiteSpace(term.target_type))
                    return Fail($"action term '{term.name}' has no target_type", out error);
            }
            error = null;
            return true;
        }

        static bool IsSha256(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length != 64)
                return false;
            foreach (char item in value)
                if (!((item >= '0' && item <= '9') || (item >= 'a' && item <= 'f')))
                    return false;
            return true;
        }

        static bool Approximately(float left, float right) => Mathf.Abs(left - right) <= 1e-6f;

        static bool Fail(string value, out string error)
        {
            error = value;
            return false;
        }
    }
}
