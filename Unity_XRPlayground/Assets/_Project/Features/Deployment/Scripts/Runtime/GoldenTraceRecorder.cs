using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class ContactTraceTelemetry
    {
        public string sensor_id;
        public string[] body_ids = Array.Empty<string>();
    }

    [Serializable]
    public sealed class AssistTraceTelemetry
    {
        public string profile_id;
        public bool active;
        public float force_n;
        public float torque_nm;
        public bool broke;
    }

    [Serializable]
    public sealed class LoopClosureTraceTelemetry
    {
        public string constraint_id;
        public float anchor_error_m;
        public float axis_error_deg;
        public string position_unit;
        public string velocity_unit;
        public float position;
        public float velocity;
        public float proxy_mass_kg;
        public bool healthy;
    }

    [Serializable]
    public sealed class GoldenTraceSample
    {
        public int schema_version = 1;
        public string station_id;
        public string policy_id;
        public int reset_seed;
        public float sim_time_s;
        public StationTelemetrySnapshot reset_state;
        public float[] observations;
        public float[] raw_actions;
        public float[] processed_actions;
        public JointStateTelemetry joint_state;
        public BodyStateTelemetry root_state;
        public ObjectStateTelemetry[] object_state;
        public ContactTraceTelemetry[] contacts;
        public AssistTraceTelemetry[] assist;
        public LoopClosureTraceTelemetry[] loop_closures =
            Array.Empty<LoopClosureTraceTelemetry>();
        public float task_score;

        public bool Validate(int observationDimension, int actionDimension, out string error)
        {
            if (schema_version != 1 || string.IsNullOrWhiteSpace(station_id) ||
                string.IsNullOrWhiteSpace(policy_id))
                return Fail("trace IDs or schema are invalid", out error);
            if (reset_state == null || joint_state == null || root_state == null ||
                object_state == null || contacts == null || assist == null)
                return Fail("trace state fields must be present", out error);
            if (observations == null || observations.Length != observationDimension ||
                raw_actions == null || raw_actions.Length != actionDimension ||
                processed_actions == null || processed_actions.Length != actionDimension)
                return Fail("trace observation/action dimensions do not match the contract", out error);
            if (!Finite(observations) || !Finite(raw_actions) || !Finite(processed_actions) ||
                !float.IsFinite(sim_time_s) || !float.IsFinite(task_score) ||
                !Finite(joint_state.position) || !Finite(joint_state.velocity) ||
                !Finite(root_state.position) || !Finite(root_state.orientation_xyzw) ||
                !Finite(root_state.linear_velocity) || !Finite(root_state.angular_velocity))
                return Fail("trace contains a non-finite value", out error);
            if (joint_state.names == null || joint_state.position_units == null ||
                joint_state.velocity_units == null || joint_state.position == null ||
                joint_state.velocity == null ||
                joint_state.names.Length != joint_state.position_units.Length ||
                joint_state.names.Length != joint_state.velocity_units.Length ||
                joint_state.names.Length != joint_state.position.Length ||
                joint_state.names.Length != joint_state.velocity.Length)
                return Fail("trace joint state arrays differ in length", out error);
            foreach (ObjectStateTelemetry item in object_state)
                if (item == null || string.IsNullOrWhiteSpace(item.id) || !FiniteBody(item))
                    return Fail("trace contains an invalid object state", out error);
            foreach (AssistTraceTelemetry item in assist)
                if (item == null || !float.IsFinite(item.force_n) || !float.IsFinite(item.torque_nm))
                    return Fail("trace contains invalid assistance telemetry", out error);
            if (loop_closures == null)
                return Fail("trace loop-closure telemetry must be present", out error);
            foreach (LoopClosureTraceTelemetry item in loop_closures)
                if (item == null || string.IsNullOrWhiteSpace(item.constraint_id) ||
                    !float.IsFinite(item.anchor_error_m) ||
                    !float.IsFinite(item.axis_error_deg) ||
                    string.IsNullOrWhiteSpace(item.position_unit) ||
                    string.IsNullOrWhiteSpace(item.velocity_unit) ||
                    !float.IsFinite(item.position) ||
                    !float.IsFinite(item.velocity) ||
                    !float.IsFinite(item.proxy_mass_kg) || item.proxy_mass_kg <= 0f)
                    return Fail("trace contains invalid loop-closure telemetry", out error);
            error = null;
            return true;
        }

        static bool FiniteBody(BodyStateTelemetry value) =>
            value.position?.Length == 3 && value.orientation_xyzw?.Length == 4 &&
            value.linear_velocity?.Length == 3 && value.angular_velocity?.Length == 3 &&
            Finite(value.position) && Finite(value.orientation_xyzw) &&
            Finite(value.linear_velocity) && Finite(value.angular_velocity);

        static bool Finite(float[] values)
        {
            if (values == null)
                return false;
            foreach (float value in values)
                if (!float.IsFinite(value))
                    return false;
            return true;
        }

        static bool Fail(string message, out string error)
        {
            error = message;
            return false;
        }
    }

    [DisallowMultipleComponent]
    public sealed class GoldenTraceRecorder : MonoBehaviour
    {
        public StationRuntime station;
        [Tooltip("Empty writes to Application.persistentDataPath/DeploymentTraces.")]
        public string outputDirectory;
        [Min(1)] public int flushEverySamples = 60;
        public bool recordOnEnable;

        readonly List<AssistTraceTelemetry> _assistEvents = new();
        readonly List<ContactGripAssist> _gripAssists = new();
        readonly List<SpotUprightAssist> _spotAssists = new();
        StreamWriter _writer;
        string _temporaryPath;
        string _finalPath;
        StationTelemetrySnapshot _resetState;
        int _resetSeed;

        public bool IsRecording => _writer != null;
        public int SampleCount { get; private set; }
        public string LastCompletedPath { get; private set; }

        void OnEnable()
        {
            if (recordOnEnable && !StartRecording(out string error))
                Debug.LogError($"GoldenTraceRecorder: {error}", this);
        }

        void OnDisable() => StopRecording();

        void Update()
        {
            if (IsRecording && (station == null || station.Mode != StationMode.OfflinePolicy ||
                                station.Adapter == null || !station.Adapter.IsHealthy))
            {
                Debug.LogError("Golden trace aborted because the OfflinePolicy station became unhealthy", this);
                AbortRecording();
            }
        }

        [ContextMenu("Start Golden Trace Recording")]
        public void StartFromContextMenu()
        {
            if (!StartRecording(out string error))
                Debug.LogError($"GoldenTraceRecorder: {error}", this);
        }

        [ContextMenu("Stop And Finalize Golden Trace")]
        public void StopRecording()
        {
            if (_writer == null)
                return;
            Unsubscribe();
            _writer.Flush();
            if (_writer.BaseStream is FileStream fileStream)
                fileStream.Flush(true);
            _writer.Dispose();
            _writer = null;
            if (SampleCount <= 0)
            {
                File.Delete(_temporaryPath);
                return;
            }
            if (File.Exists(_finalPath))
            {
                string backup = _finalPath + ".previous." + DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ");
                File.Replace(_temporaryPath, _finalPath, backup);
            }
            else
            {
                File.Move(_temporaryPath, _finalPath);
            }
            LastCompletedPath = _finalPath;
            Debug.Log($"Golden trace finalized: {_finalPath} ({SampleCount} samples)", this);
        }

        public void AbortRecording()
        {
            if (_writer == null)
                return;
            Unsubscribe();
            _writer.Dispose();
            _writer = null;
            if (!string.IsNullOrEmpty(_temporaryPath) && File.Exists(_temporaryPath))
                File.Delete(_temporaryPath);
            SampleCount = 0;
        }

        public bool StartRecording(out string error)
        {
            if (IsRecording)
            {
                error = "a recording is already active";
                return false;
            }
            if (station == null)
                station = GetComponentInParent<StationRuntime>();
            if (station == null || station.Mode != StationMode.OfflinePolicy ||
                station.Adapter == null || !station.Adapter.IsHealthy ||
                station.policy?.Contract == null)
            {
                error = "a healthy OfflinePolicy StationRuntime, adapter and contract are required";
                return false;
            }
            try
            {
                _resetState = station.Adapter.CaptureTelemetry();
                _resetSeed = station.resetSeed;
                string root = string.IsNullOrWhiteSpace(outputDirectory)
                    ? Path.Combine(Application.persistentDataPath, "DeploymentTraces")
                    : Path.GetFullPath(outputDirectory);
                string session = $"{station.stationId}_{station.policy.Contract.policy_id}_" +
                                 DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ");
                string directory = Path.Combine(root, session);
                Directory.CreateDirectory(directory);
                _finalPath = Path.Combine(directory, "golden_trace.jsonl");
                _temporaryPath = _finalPath + ".partial";
                _writer = new StreamWriter(new FileStream(
                    _temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read));
                SampleCount = 0;
                LastCompletedPath = null;
                Subscribe();
                error = null;
                return true;
            }
            catch (Exception exception)
            {
                _writer?.Dispose();
                _writer = null;
                error = exception.Message;
                return false;
            }
        }

        void Subscribe()
        {
            station.StationReset += OnStationReset;
            station.PolicyStepCompleted += OnPolicyStep;
            _gripAssists.Clear();
            _spotAssists.Clear();
            _gripAssists.AddRange(station.GetComponentsInChildren<ContactGripAssist>(true));
            _spotAssists.AddRange(station.GetComponentsInChildren<SpotUprightAssist>(true));
            foreach (ContactGripAssist item in _gripAssists)
                item.Telemetry += OnAssist;
            foreach (SpotUprightAssist item in _spotAssists)
                item.Telemetry += OnAssist;
        }

        void Unsubscribe()
        {
            if (station != null)
            {
                station.StationReset -= OnStationReset;
                station.PolicyStepCompleted -= OnPolicyStep;
            }
            foreach (ContactGripAssist item in _gripAssists)
                if (item != null)
                    item.Telemetry -= OnAssist;
            foreach (SpotUprightAssist item in _spotAssists)
                if (item != null)
                    item.Telemetry -= OnAssist;
            _gripAssists.Clear();
            _spotAssists.Clear();
        }

        void OnStationReset(int seed)
        {
            _resetSeed = seed;
            _resetState = station.Adapter.CaptureTelemetry();
            _assistEvents.Clear();
        }

        void OnAssist(AssistTelemetry value)
        {
            _assistEvents.Add(new AssistTraceTelemetry
            {
                profile_id = value.ProfileId,
                active = value.Active,
                force_n = value.Force,
                torque_nm = value.Torque,
                broke = value.Broke,
            });
        }

        void OnPolicyStep(
            float[] observations,
            float[] rawActions,
            float[] processedActions,
            float taskScore)
        {
            if (_writer == null)
                return;
            try
            {
                StationTelemetrySnapshot state = station.Adapter.CaptureTelemetry();
                var sample = new GoldenTraceSample
                {
                    station_id = station.stationId,
                    policy_id = station.policy.Contract.policy_id,
                    reset_seed = _resetSeed,
                    sim_time_s = Time.fixedTime,
                    reset_state = _resetState,
                    observations = observations,
                    raw_actions = rawActions,
                    processed_actions = processedActions,
                    joint_state = state.joint_state,
                    root_state = state.root_state,
                    object_state = state.object_state,
                    contacts = CaptureContacts(),
                    assist = _assistEvents.ToArray(),
                    loop_closures = CaptureLoopClosures(),
                    task_score = taskScore,
                };
                if (!sample.Validate(
                        station.policy.Contract.ObservationDimension,
                        station.policy.Contract.ActionDimension,
                        out string error))
                    throw new InvalidDataException(error);
                _writer.WriteLine(JsonUtility.ToJson(sample, false));
                _assistEvents.Clear();
                SampleCount++;
                if (SampleCount % Mathf.Max(1, flushEverySamples) == 0)
                    _writer.Flush();
            }
            catch (Exception exception)
            {
                Debug.LogError($"Golden trace recording stopped: {exception.Message}", this);
                StopRecording();
            }
        }

        ContactTraceTelemetry[] CaptureContacts()
        {
            ContactSensor[] sensors = station.GetComponentsInChildren<ContactSensor>(true);
            var result = new ContactTraceTelemetry[sensors.Length];
            for (int i = 0; i < sensors.Length; i++)
            {
                var names = new List<string>();
                foreach (Rigidbody body in sensors[i].Contacts)
                    if (body != null)
                        names.Add(body.name);
                foreach (Collider collider in sensors[i].Colliders)
                    if (collider != null && collider.attachedRigidbody == null)
                        names.Add(collider.name);
                names.Sort(StringComparer.Ordinal);
                result[i] = new ContactTraceTelemetry
                {
                    sensor_id = sensors[i].name,
                    body_ids = names.ToArray(),
                };
            }
            return result;
        }

        LoopClosureTraceTelemetry[] CaptureLoopClosures()
        {
            ArticulationLoopClosureBinding binding = station.offlineRig != null
                ? station.offlineRig.GetComponent<ArticulationLoopClosureBinding>()
                : null;
            if (binding == null)
                return Array.Empty<LoopClosureTraceTelemetry>();
            ArticulationRobotDriver robot = station.offlineRig != null
                ? station.offlineRig.GetComponent<ArticulationRobotDriver>()
                : null;
            var result = new List<LoopClosureTraceTelemetry>();
            foreach (KeyValuePair<string, ArticulationLoopClosureProxy> item in binding.Proxies)
            {
                ArticulationLoopClosureProxy proxy = item.Value;
                if (proxy == null)
                    continue;
                RobotJointDefinition joint = robot?.definition?.FindJoint(item.Key);
                result.Add(new LoopClosureTraceTelemetry
                {
                    constraint_id = item.Key,
                    anchor_error_m = proxy.AnchorErrorMeters,
                    axis_error_deg = proxy.AxisErrorDegrees,
                    position_unit = joint?.positionUnit ?? "radian",
                    velocity_unit = joint?.VelocityUnit ?? "radian_per_second",
                    position = proxy.JointPositionRadians,
                    velocity = proxy.JointVelocityRadiansPerSecond,
                    proxy_mass_kg = proxy.proxyBody != null ? proxy.proxyBody.mass : proxy.proxyMass,
                    healthy = proxy.TryValidate(out _),
                });
            }
            result.Sort((first, second) =>
                string.CompareOrdinal(first.constraint_id, second.constraint_id));
            return result.ToArray();
        }
    }
}
