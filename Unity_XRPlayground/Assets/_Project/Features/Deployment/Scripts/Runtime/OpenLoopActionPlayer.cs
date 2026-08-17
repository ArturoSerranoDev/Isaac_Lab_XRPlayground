using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class OpenLoopActionFrame
    {
        public float[] processed_action;
    }

    [Serializable]
    public sealed class OpenLoopActionSequence
    {
        public int schema_version = 1;
        public string station_id;
        public string policy_id;
        public int seed;
        public OpenLoopActionFrame[] frames;
    }

    [DisallowMultipleComponent]
    public sealed class OpenLoopActionPlayer : MonoBehaviour
    {
        public StationRuntime station;
        public GoldenTraceRecorder recorder;
        public TextAsset actionSequence;

        OpenLoopActionSequence _sequence;
        int _frame;
        int _physicsSteps;
        bool _running;

        public bool IsRunning => _running;

        void Awake()
        {
            if (station == null)
                station = GetComponent<StationRuntime>();
            if (recorder == null)
                recorder = GetComponent<GoldenTraceRecorder>();
        }

        void OnDisable() => Abort();

        [ContextMenu("Play Open-Loop Calibration Sequence")]
        public void StartFromContext()
        {
            if (!StartSequence(out string error))
                Debug.LogError($"OpenLoopActionPlayer: {error}", this);
        }

        [ContextMenu("Abort Open-Loop Calibration Sequence")]
        public void AbortFromContext() => Abort();

        public bool StartSequence(out string error)
        {
            if (_running)
            {
                error = "an open-loop sequence is already running";
                return false;
            }
            if (actionSequence == null || station == null || recorder == null ||
                station.policy?.Contract == null)
            {
                error = "station, recorder, loaded contract and action sequence are required";
                return false;
            }
            try
            {
                _sequence = JsonUtility.FromJson<OpenLoopActionSequence>(actionSequence.text);
            }
            catch (Exception exception)
            {
                error = $"action sequence JSON is invalid: {exception.Message}";
                return false;
            }
            PolicyContract contract = station.policy.Contract;
            if (_sequence == null || _sequence.schema_version != 1 ||
                _sequence.station_id != station.stationId ||
                _sequence.policy_id != contract.policy_id ||
                _sequence.frames == null || _sequence.frames.Length == 0)
            {
                error = "action sequence schema, IDs or frames are invalid";
                return false;
            }
            foreach (OpenLoopActionFrame frame in _sequence.frames)
                if (frame?.processed_action == null ||
                    frame.processed_action.Length != contract.ActionDimension ||
                    !Finite(frame.processed_action))
                {
                    error = "an action frame does not match the policy contract";
                    return false;
                }
            if (!station.BeginOpenLoopControl(_sequence.seed, out error))
                return false;
            if (!recorder.StartRecording(out error))
            {
                station.EndOpenLoopControl();
                return false;
            }
            _frame = 0;
            _physicsSteps = 0;
            _running = true;
            error = null;
            return true;
        }

        void FixedUpdate()
        {
            if (!_running)
                return;
            int cadence = Mathf.Max(1, station.policy.Contract.timing.deployment_decimation);
            if ((_physicsSteps++ % cadence) != 0)
                return;
            if (!station.ApplyOpenLoopAction(
                    _sequence.frames[_frame].processed_action, out string error))
            {
                Debug.LogError($"OpenLoopActionPlayer: {error}", this);
                Abort();
                return;
            }
            _frame++;
            if (_frame < _sequence.frames.Length)
                return;
            station.EndOpenLoopControl();
            recorder.StopRecording();
            _running = false;
            Debug.Log($"Open-loop trace completed: {recorder.LastCompletedPath}", this);
        }

        void Abort()
        {
            if (!_running)
                return;
            _running = false;
            station?.EndOpenLoopControl();
            recorder?.AbortRecording();
        }

        static bool Finite(float[] values)
        {
            foreach (float value in values)
                if (float.IsNaN(value) || float.IsInfinity(value))
                    return false;
            return true;
        }
    }
}
