using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    public abstract class StationAdapterBase : MonoBehaviour, IStationAdapter
    {
        protected PolicyContract Contract { get; private set; }
        protected bool Healthy { get; set; } = true;
        protected float EvaluationElapsed { get; private set; }
        protected StationTaskMetrics EvaluationMetrics { get; private set; } = new();

        public abstract string AdapterId { get; }
        public abstract int ObservationDimension { get; }
        public abstract int ActionDimension { get; }
        public virtual float TaskScore => 0f;
        public bool IsHealthy => Healthy;
        public virtual bool HasJointLimitViolation => false;

        public virtual void BindContract(PolicyContract contract)
        {
            if (contract == null || contract.adapter_id != AdapterId)
                throw new ArgumentException("Policy contract does not match station adapter", nameof(contract));
            Contract = contract;
        }

        public abstract void ResetStation(int seed);
        public abstract void BuildObservation(float[] destination);
        public abstract void ApplyAction(float[] action, float policyDeltaTime);
        public virtual StationTelemetrySnapshot CaptureTelemetry() =>
            throw new InvalidOperationException($"{AdapterId} has no telemetry implementation");

        public virtual void BeginEvaluationScenario(int seed)
        {
            EvaluationElapsed = 0f;
            EvaluationMetrics = new StationTaskMetrics();
        }

        public virtual void SampleEvaluationScenario(float policyDeltaTime)
        {
            EvaluationElapsed += Mathf.Max(0f, policyDeltaTime);
        }

        public virtual StationEvaluationSummary CompleteEvaluationScenario() =>
            new()
            {
                normalized_task_score = Mathf.Clamp01(TaskScore),
                task_metrics = EvaluationMetrics,
            };

        public virtual bool TryHandleCommand(string command, string payloadJson, out string error)
        {
            error = $"adapter '{AdapterId}' does not support command '{command}'";
            return false;
        }

        protected static void Write(float[] destination, ref int cursor, float value)
        {
            destination[cursor++] = float.IsFinite(value) ? value : 0f;
        }

        protected static void Write(float[] destination, ref int cursor, Vector3 value, float scale = 1f)
        {
            Write(destination, ref cursor, value.x * scale);
            Write(destination, ref cursor, value.y * scale);
            Write(destination, ref cursor, value.z * scale);
        }

        protected static void WriteTermValue(
            float[] destination, ref int cursor, PolicyTerm term, int component, float value)
        {
            Write(destination, ref cursor, value * term.ScaleAt(component) + term.OffsetAt(component));
        }

        protected static void WriteTermVector(
            float[] destination, ref int cursor, PolicyTerm term, Vector3 value)
        {
            WriteTermValue(destination, ref cursor, term, 0, value.x);
            WriteTermValue(destination, ref cursor, term, 1, value.y);
            WriteTermValue(destination, ref cursor, term, 2, value.z);
        }

        protected static Vector3 IsaacLocalPosition(Transform anchor, Vector3 worldPosition)
        {
            Vector3 unityLocal = anchor != null ? anchor.InverseTransformPoint(worldPosition) : worldPosition;
            return DeploymentFrameConverter.UnityToIsaac(unityLocal);
        }

        protected static Vector3 IsaacDirection(Transform anchor, Vector3 worldDirection)
        {
            Vector3 unityLocal = anchor != null ? anchor.InverseTransformDirection(worldDirection) : worldDirection;
            return DeploymentFrameConverter.UnityToIsaac(unityLocal);
        }

        protected static Vector3 MeanPosition(Transform[] values)
        {
            if (values == null || values.Length == 0)
                return Vector3.zero;
            Vector3 sum = Vector3.zero;
            int count = 0;
            foreach (Transform value in values)
            {
                if (value == null)
                    continue;
                sum += value.position;
                count++;
            }
            return count > 0 ? sum / count : Vector3.zero;
        }

        protected static StationTelemetrySnapshot Capture(
            ArticulationRobotDriver robot,
            ArticulationBody root,
            Transform anchor,
            params ObjectTelemetryBinding[] objects)
        {
            return DeploymentTelemetryCapture.Create(robot, root, anchor, objects);
        }

        protected static ObjectTelemetryBinding[] Bindings(PhysicalObjectSlot[] slots)
        {
            if (slots == null)
                return Array.Empty<ObjectTelemetryBinding>();
            var result = new ObjectTelemetryBinding[slots.Length];
            for (int i = 0; i < slots.Length; i++)
            {
                PhysicalObjectSlot slot = slots[i];
                result[i] = slot == null
                    ? new ObjectTelemetryBinding($"slot_{i}", null, false)
                    : new ObjectTelemetryBinding(
                        string.IsNullOrWhiteSpace(slot.id) ? $"slot_{i}" : slot.id,
                        slot.body,
                        slot.active,
                        slot.grasped,
                        slot.color);
            }
            return result;
        }

        protected void VerifyCursor(int cursor, int expected)
        {
            if (cursor != expected)
            {
                Healthy = false;
                throw new InvalidOperationException($"{AdapterId} wrote {cursor} values; expected {expected}");
            }
        }
    }
}
