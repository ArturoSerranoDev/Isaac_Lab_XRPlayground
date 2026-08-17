using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public class BridgeEnvelopeHeader
    {
        public int schema_version;
        public string station_id;
        public long sequence;
        public double sim_time_s;
        public string frame_id;
        public string message_type;
    }

    [Serializable]
    public sealed class NamedJointState
    {
        public string name;
        public float position;
        public float velocity;
    }

    [Serializable]
    public sealed class NamedPoseState
    {
        public string name;
        public float[] position;
        public float[] orientation_xyzw;
    }

    [Serializable]
    public sealed class RobotStatePayload
    {
        public NamedJointState[] joints;
        public NamedPoseState[] links;
        public NamedPoseState root;
    }

    [Serializable]
    public sealed class SceneObjectState
    {
        public string id;
        public bool active;
        public float[] position;
        public float[] orientation_xyzw;
        public float[] linear_velocity;
        public float[] angular_velocity;
        public bool grasped;
        public int color;
    }

    [Serializable]
    public sealed class ObjectsStatePayload
    {
        public SceneObjectState[] objects;
    }

    [Serializable]
    sealed class RobotEnvelope : BridgeEnvelopeHeader
    {
        public RobotStatePayload payload;
    }

    [Serializable]
    sealed class ObjectsEnvelope : BridgeEnvelopeHeader
    {
        public ObjectsStatePayload payload;
    }

    [Serializable]
    sealed class SessionCommandPayload
    {
        public string mode;
        public string command;
        public string parameters_json;
    }

    [Serializable]
    sealed class SessionCommandEnvelope : BridgeEnvelopeHeader
    {
        public SessionCommandPayload payload;
    }

    [DisallowMultipleComponent]
    public sealed class BridgeV2Client : MonoBehaviour
    {
        public LengthPrefixedJsonClient transport;
        public string stationId;
        [Min(0.01f)] public float staleAfterSeconds = 0.25f;
        [Min(0.1f)] public float disconnectAfterSeconds = 2f;

        long _lastSequence = -1;
        long _outgoingSequence;
        float _lastReceiveRealtime = float.NegativeInfinity;
        bool _reconnectRequested;

        public bool IsStale => Time.realtimeSinceStartup - _lastReceiveRealtime > staleAfterSeconds;
        public bool IsLogicallyDisconnected => Time.realtimeSinceStartup - _lastReceiveRealtime > disconnectAfterSeconds;
        public double LastSimulationTime { get; private set; }
        public event Action<RobotStatePayload, double> RobotStateReceived;
        public event Action<ObjectsStatePayload, double> ObjectsStateReceived;
        public event Action<bool> StaleChanged;

        bool _wasStale = true;

        void OnEnable()
        {
            if (transport != null)
                transport.MessageReceived += OnJson;
        }

        void OnDisable()
        {
            if (transport != null)
                transport.MessageReceived -= OnJson;
        }

        void Update()
        {
            bool stale = IsStale;
            if (stale != _wasStale)
            {
                _wasStale = stale;
                StaleChanged?.Invoke(stale);
            }
            if (transport == null)
                return;
            if (IsLogicallyDisconnected && transport.IsConnected && !_reconnectRequested)
            {
                _reconnectRequested = true;
                transport.Disconnect();
            }
            else if (_reconnectRequested && !transport.IsConnected)
            {
                _reconnectRequested = false;
                transport.Connect();
            }
        }

        void OnJson(string json)
        {
            BridgeEnvelopeHeader header;
            try
            {
                header = JsonUtility.FromJson<BridgeEnvelopeHeader>(json);
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Bridge v2 JSON rejected: {exception.Message}", this);
                return;
            }
            if (header == null || header.schema_version != 2 || header.station_id != stationId ||
                string.IsNullOrEmpty(header.message_type))
                return;
            if (header.sequence <= _lastSequence)
                return;
            _lastSequence = header.sequence;
            LastSimulationTime = header.sim_time_s;
            _lastReceiveRealtime = Time.realtimeSinceStartup;

            if (header.message_type == "robot_state")
            {
                RobotEnvelope envelope = JsonUtility.FromJson<RobotEnvelope>(json);
                if (envelope?.payload != null)
                    RobotStateReceived?.Invoke(envelope.payload, header.sim_time_s);
            }
            else if (header.message_type == "objects_state")
            {
                ObjectsEnvelope envelope = JsonUtility.FromJson<ObjectsEnvelope>(json);
                if (envelope?.payload != null)
                    ObjectsStateReceived?.Invoke(envelope.payload, header.sim_time_s);
            }
        }

        public void PublishSessionCommand(
            string mode,
            string command = null,
            string parametersJson = null)
        {
            if (transport == null)
                return;
            var envelope = new SessionCommandEnvelope
            {
                schema_version = 2,
                station_id = stationId,
                sequence = _outgoingSequence++,
                sim_time_s = LastSimulationTime,
                frame_id = "unity_env",
                message_type = "session_command",
                payload = new SessionCommandPayload
                {
                    mode = mode,
                    command = command,
                    parameters_json = parametersJson,
                },
            };
            transport.PublishJson(JsonUtility.ToJson(envelope));
        }
    }
}
