using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using XRPlayground.Policies;

namespace XRPlayground.ROS
{
    /// <summary>
    /// Observes a bridge at runtime and reports the entire station contract, not just TCP state.
    /// Add it to the same GameObject as RosTcpClient and set stationId from StationRegistry.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class BridgeHealthMonitor : MonoBehaviour
    {
        public string stationId;
        public RosTcpClient client;
        public OnnxPolicyRunner policyRunner;
        [Min(0.1f)] public float freshnessSeconds = 3f;

        readonly Dictionary<string, float> _lastTopicAt = new(StringComparer.Ordinal);
        StationDefinition _station;

        public bool IsPortHealthy => client != null && client.IsConnected;
        public bool IsHeartbeatHealthy => IsFresh(RosTopics.Heartbeat);
        public bool IsTopicFlowHealthy
        {
            get
            {
                if (_station?.required_topics == null)
                    return false;
                foreach (var topic in _station.required_topics)
                {
                    if (!IsFresh(topic))
                        return false;
                }
                return true;
            }
        }

        public bool IsPolicyHealthy
        {
            get
            {
                if (_station == null || policyRunner == null || policyRunner.modelAsset == null)
                    return false;
                return policyRunner.MatchesContract(
                    _station.policy_task_id, _station.obs_dim, _station.action_dim, out _);
            }
        }

        public bool IsHealthy => IsPortHealthy && IsHeartbeatHealthy && IsTopicFlowHealthy && IsPolicyHealthy;

        public string CompactSummary =>
            $"Checks: TCP {(IsPortHealthy ? "OK" : "DOWN")} | " +
            $"HB {(IsHeartbeatHealthy ? "OK" : "STALE")} | " +
            $"topics {(IsTopicFlowHealthy ? "OK" : "STALE")} | " +
            $"ONNX {(IsPolicyHealthy ? "OK" : "MISMATCH")}";

        void Awake()
        {
            if (client == null)
                client = GetComponent<RosTcpClient>();
            ResolveStation();
        }

        void OnEnable()
        {
            if (client != null)
                client.MessageReceived += OnMessage;
        }

        void OnDisable()
        {
            if (client != null)
                client.MessageReceived -= OnMessage;
        }

        [ContextMenu("Log Bridge Health")]
        public void LogHealth() => Debug.Log(BuildReport(), this);

        public string BuildReport()
        {
            ResolveStation();
            var report = new StringBuilder();
            string name = _station == null ? stationId : _station.id;
            report.Append($"{name}: {(IsHealthy ? "HEALTHY" : "NEEDS ATTENTION")}");
            report.Append($"\nTCP: {(IsPortHealthy ? "OK" : "DOWN")} {client?.host}:{client?.port}");
            report.Append($"\nHeartbeat: {(IsHeartbeatHealthy ? "OK" : "STALE")} {AgeText(RosTopics.Heartbeat)}");
            if (_station != null)
            {
                report.Append("\nTopics:");
                foreach (var topic in _station.required_topics)
                    report.Append($"\n  {(IsFresh(topic) ? "OK" : "STALE")} {topic} {AgeText(topic)}");
                string policy = policyRunner?.Metadata == null
                    ? "sidecar not loaded"
                    : $"{policyRunner.Metadata.task_id}, obs={policyRunner.Metadata.obs_dim}, action={policyRunner.Metadata.action_dim}";
                report.Append($"\nPolicy: {(IsPolicyHealthy ? "OK" : "MISMATCH")} {policy}");
            }
            else
            {
                report.Append("\nStation manifest: NOT FOUND");
            }
            return report.ToString();
        }

        void OnMessage(string json)
        {
            if (RosJson.TryParseTopic(json, out var topic) && !string.IsNullOrEmpty(topic))
                _lastTopicAt[topic] = Time.realtimeSinceStartup;
        }

        bool IsFresh(string topic)
        {
            return _lastTopicAt.TryGetValue(topic, out var last) &&
                Time.realtimeSinceStartup - last <= freshnessSeconds;
        }

        string AgeText(string topic)
        {
            if (!_lastTopicAt.TryGetValue(topic, out var last))
                return "(not received)";
            return $"({Time.realtimeSinceStartup - last:0.0}s ago)";
        }

        void ResolveStation()
        {
            if (_station == null && !string.IsNullOrEmpty(stationId))
                StationRegistry.TryGet(stationId, out _station);
        }
    }
}
