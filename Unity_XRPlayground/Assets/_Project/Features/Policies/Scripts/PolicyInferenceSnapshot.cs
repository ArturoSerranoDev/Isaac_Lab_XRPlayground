using System;

namespace XRPlayground.Policies
{
    /// <summary>
    /// One policy forward pass for network visualization.
    /// Hidden layers are real ELU post-activations when the ONNX graph exposes them.
    /// </summary>
    public sealed class PolicyInferenceSnapshot
    {
        /// <summary>Layer sizes including input and output, e.g. [30, 256, 128, 64, 8].</summary>
        public int[] LayerSizes = Array.Empty<int>();

        /// <summary>
        /// Activations per layer. Index 0 = observation (raw |x| or normalized),
        /// last = actions, middle = hidden ELU (or approx). Values are visualization intensity (non-negative).
        /// </summary>
        public float[][] LayerActivations = Array.Empty<float[]>();

        /// <summary>True when hidden layers come from Inference Engine intermediate tensors.</summary>
        public bool HasRealIntermediates;

        public string ArchitectureLabel = "";
        public string SourceLabel = "idle";
    }
}
