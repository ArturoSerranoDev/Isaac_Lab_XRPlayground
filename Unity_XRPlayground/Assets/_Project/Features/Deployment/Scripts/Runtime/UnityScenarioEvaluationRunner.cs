using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class UnityScenarioEvaluationRunner : MonoBehaviour
    {
        const int RequiredScenarioCount = 100;

        public StationRuntime station;
        [Min(0)] public int firstSeed;
        [Min(0.1f)] public float scenarioDurationSeconds = 10f;
        public string outputRoot;

        StreamWriter _writer;
        string _partialPath;
        string _finalPath;
        int _scenarioIndex;
        float _elapsed;
        bool _running;
        bool _hasNan;
        bool _invalidActions;
        bool _missingJoints;
        bool _jointLimitViolations;

        public bool IsRunning => _running;
        public string FinalPath => _finalPath;

        void Awake()
        {
            if (station == null)
                station = GetComponent<StationRuntime>();
        }

        void OnEnable()
        {
            if (station != null)
                station.PolicyStepCompleted += OnPolicyStep;
        }

        void OnDisable()
        {
            if (station != null)
                station.PolicyStepCompleted -= OnPolicyStep;
            AbortEvaluation();
        }

        void Update()
        {
            if (_running && (station == null || station.Mode != StationMode.OfflinePolicy ||
                             station.Adapter == null || !station.Adapter.IsHealthy))
            {
                Debug.LogError(
                    "Unity evaluation aborted because the OfflinePolicy station became unhealthy", this);
                AbortEvaluation();
            }
        }

        [ContextMenu("Start 100-Seed Unity Evaluation")]
        public void StartFromContext()
        {
            if (!StartEvaluation(out string error))
                Debug.LogError($"UnityScenarioEvaluationRunner: {error}", this);
        }

        [ContextMenu("Abort Unity Evaluation")]
        public void AbortFromContext() => AbortEvaluation();

        public bool StartEvaluation(out string error)
        {
            if (_running)
            {
                error = "evaluation is already running";
                return false;
            }
            if (station == null || station.Mode != StationMode.OfflinePolicy ||
                station.Adapter == null || station.policy?.Contract == null)
            {
                error = "a healthy OfflinePolicy station and loaded contract are required";
                return false;
            }
            string root = string.IsNullOrWhiteSpace(outputRoot)
                ? Path.Combine(Application.persistentDataPath, "DeploymentEvaluations")
                : outputRoot;
            string session = $"{station.policy.Contract.policy_id}_{DateTime.UtcNow:yyyyMMdd_HHmmss_fff}";
            string directory = Path.Combine(root, session);
            Directory.CreateDirectory(directory);
            _finalPath = Path.Combine(directory, "unity.scenarios.jsonl");
            _partialPath = _finalPath + ".partial";
            if (File.Exists(_finalPath) || File.Exists(_partialPath))
            {
                error = $"evaluation destination already exists: {directory}";
                return false;
            }
            _writer = new StreamWriter(_partialPath, false, new UTF8Encoding(false));
            _scenarioIndex = 0;
            _running = true;
            if (!BeginScenario(out error))
            {
                AbortEvaluation();
                return false;
            }
            error = null;
            return true;
        }

        bool BeginScenario(out string error)
        {
            int seed = firstSeed + _scenarioIndex;
            if (!station.ResetActiveStation(seed, out error))
                return false;
            station.Adapter.BeginEvaluationScenario(seed);
            _elapsed = 0f;
            _hasNan = false;
            _invalidActions = false;
            _missingJoints = false;
            _jointLimitViolations = false;
            return true;
        }

        void OnPolicyStep(float[] observations, float[] rawActions, float[] processedActions, float taskScore)
        {
            if (!_running)
                return;
            PolicyContract contract = station.policy.Contract;
            ValidateTelemetry(station.Adapter, out bool telemetryNonFinite, out bool telemetryMissing);
            _hasNan |= !Finite(observations) || telemetryNonFinite;
            _invalidActions |= rawActions == null || processedActions == null ||
                               rawActions.Length != contract.ActionDimension ||
                               processedActions.Length != contract.ActionDimension ||
                               !Finite(rawActions) || !Finite(processedActions);
            _missingJoints |= !station.Adapter.IsHealthy || telemetryMissing;
            _jointLimitViolations |= station.Adapter.HasJointLimitViolation;
            station.Adapter.SampleEvaluationScenario(contract.timing.policy_dt);
            _elapsed += contract.timing.policy_dt;
            if (_elapsed + 1e-6f < scenarioDurationSeconds)
                return;
            if (!CompleteScenario(out string error))
            {
                Debug.LogError($"UnityScenarioEvaluationRunner: {error}", this);
                AbortEvaluation();
            }
        }

        bool CompleteScenario(out string error)
        {
            StationEvaluationSummary summary = station.Adapter.CompleteEvaluationScenario();
            var record = new StationScenarioRecord
            {
                policy_id = station.policy.Contract.policy_id,
                seed = firstSeed + _scenarioIndex,
                normalized_task_score = summary.normalized_task_score,
                task_metrics = summary.task_metrics,
                has_nan = _hasNan,
                invalid_actions = _invalidActions,
                missing_joints = _missingJoints,
                joint_limit_violations = _jointLimitViolations,
            };
            if (!record.Validate(out error))
                return false;
            _writer.WriteLine(JsonUtility.ToJson(record));
            _writer.Flush();
            _scenarioIndex++;
            if (_scenarioIndex < RequiredScenarioCount)
                return BeginScenario(out error);

            _writer.Dispose();
            _writer = null;
            File.Move(_partialPath, _finalPath);
            _running = false;
            Debug.Log($"Unity evaluation completed: {_finalPath}", this);
            error = null;
            return true;
        }

        void AbortEvaluation()
        {
            _running = false;
            _writer?.Dispose();
            _writer = null;
            if (!string.IsNullOrEmpty(_partialPath) && File.Exists(_partialPath))
                File.Delete(_partialPath);
        }

        static void ValidateTelemetry(
            IStationAdapter adapter, out bool nonFinite, out bool missingJoints)
        {
            nonFinite = false;
            missingJoints = false;
            try
            {
                StationTelemetrySnapshot telemetry = adapter.CaptureTelemetry();
                if (telemetry?.joint_state == null || telemetry.root_state == null ||
                    telemetry.object_state == null ||
                    telemetry.joint_state.names == null ||
                    telemetry.joint_state.position_units == null ||
                    telemetry.joint_state.velocity_units == null ||
                    telemetry.joint_state.position == null ||
                    telemetry.joint_state.velocity == null ||
                    telemetry.joint_state.names.Length != telemetry.joint_state.position_units.Length ||
                    telemetry.joint_state.names.Length != telemetry.joint_state.velocity_units.Length ||
                    telemetry.joint_state.names.Length != telemetry.joint_state.position.Length ||
                    telemetry.joint_state.names.Length != telemetry.joint_state.velocity.Length)
                {
                    missingJoints = true;
                    return;
                }
                nonFinite |= !Finite(telemetry.joint_state.position) ||
                             !Finite(telemetry.joint_state.velocity) ||
                             !BodyFinite(telemetry.root_state);
                foreach (ObjectStateTelemetry state in telemetry.object_state)
                    if (state == null || !BodyFinite(state))
                        nonFinite = true;
            }
            catch
            {
                missingJoints = true;
            }
        }

        static bool BodyFinite(BodyStateTelemetry state) =>
            state.position?.Length == 3 && state.orientation_xyzw?.Length == 4 &&
            state.linear_velocity?.Length == 3 && state.angular_velocity?.Length == 3 &&
            Finite(state.position) && Finite(state.orientation_xyzw) &&
            Finite(state.linear_velocity) && Finite(state.angular_velocity);

        static bool Finite(float[] values)
        {
            if (values == null)
                return false;
            foreach (float value in values)
                if (float.IsNaN(value) || float.IsInfinity(value))
                    return false;
            return true;
        }
    }
}
