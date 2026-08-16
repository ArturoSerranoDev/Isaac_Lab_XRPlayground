using System;
using UnityEngine;

namespace XRPlayground.Policies
{
    /// <summary>Fields intentionally mirror the exporter-written policy.json sidecar.</summary>
    [Serializable]
    public sealed class PolicyMetadata
    {
        public string task_id;
        public int obs_dim;
        public int action_dim;
        public float action_scale;
        public float dt;
        public string frame;
        public string checkpoint;
        public string runtime;

        public bool HasUsableDimensions => obs_dim > 0 && action_dim > 0;

        public static bool TryParse(TextAsset asset, out PolicyMetadata metadata, out string error)
        {
            metadata = null;
            error = null;
            if (asset == null || string.IsNullOrWhiteSpace(asset.text))
            {
                error = "policy.json sidecar is not assigned";
                return false;
            }
            try
            {
                metadata = JsonUtility.FromJson<PolicyMetadata>(asset.text);
            }
            catch (Exception exception)
            {
                error = $"policy.json is invalid JSON: {exception.Message}";
                return false;
            }
            if (metadata == null || string.IsNullOrWhiteSpace(metadata.task_id))
            {
                error = "policy.json has no task_id";
                return false;
            }
            if (!metadata.HasUsableDimensions)
            {
                error = $"policy.json has invalid dimensions obs={metadata?.obs_dim}, action={metadata?.action_dim}";
                return false;
            }
            return true;
        }
    }

    /// <summary>Implemented by offline controllers that consume action_scale and dt from policy.json.</summary>
    public interface IPolicyMetadataConsumer
    {
        void ApplyPolicyMetadata(PolicyMetadata metadata);
    }
}
