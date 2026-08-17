using System;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class JointStateTelemetry
    {
        public string[] names = Array.Empty<string>();
        public string[] position_units = Array.Empty<string>();
        public string[] velocity_units = Array.Empty<string>();
        public float[] position = Array.Empty<float>();
        public float[] velocity = Array.Empty<float>();
    }

    [Serializable]
    public class BodyStateTelemetry
    {
        public float[] position = Array.Empty<float>();
        public float[] orientation_xyzw = Array.Empty<float>();
        public float[] linear_velocity = Array.Empty<float>();
        public float[] angular_velocity = Array.Empty<float>();
    }

    [Serializable]
    public sealed class ObjectStateTelemetry : BodyStateTelemetry
    {
        public string id;
        public bool active;
        public bool grasped;
        public int color = -1;
    }

    [Serializable]
    public sealed class StationTelemetrySnapshot
    {
        public JointStateTelemetry joint_state = new();
        public BodyStateTelemetry root_state = new();
        public ObjectStateTelemetry[] object_state = Array.Empty<ObjectStateTelemetry>();
    }

    public readonly struct ObjectTelemetryBinding
    {
        public readonly string Id;
        public readonly Rigidbody Body;
        public readonly bool Active;
        public readonly bool Grasped;
        public readonly int Color;

        public ObjectTelemetryBinding(
            string id,
            Rigidbody body,
            bool active = true,
            bool grasped = false,
            int color = -1)
        {
            Id = id;
            Body = body;
            Active = active;
            Grasped = grasped;
            Color = color;
        }
    }

    public static class DeploymentTelemetryCapture
    {
        public static StationTelemetrySnapshot Create(
            ArticulationRobotDriver robot,
            ArticulationBody root,
            Transform anchor,
            ObjectTelemetryBinding[] objects)
        {
            if (robot == null || root == null)
                throw new InvalidOperationException("Telemetry requires a robot driver and articulation root");
            return new StationTelemetrySnapshot
            {
                joint_state = robot.CaptureJointState(),
                root_state = Body(root.transform, root.linearVelocity, root.angularVelocity, anchor),
                object_state = Objects(objects, anchor),
            };
        }

        public static BodyStateTelemetry Body(
            Transform body,
            Vector3 linearVelocityUnity,
            Vector3 angularVelocityUnity,
            Transform anchor)
        {
            Vector3 localPosition = anchor != null
                ? anchor.InverseTransformPoint(body.position)
                : body.position;
            Quaternion localRotation = anchor != null
                ? Quaternion.Inverse(anchor.rotation) * body.rotation
                : body.rotation;
            Vector3 localLinear = anchor != null
                ? anchor.InverseTransformDirection(linearVelocityUnity)
                : linearVelocityUnity;
            Vector3 localAngular = anchor != null
                ? anchor.InverseTransformDirection(angularVelocityUnity)
                : angularVelocityUnity;
            return new BodyStateTelemetry
            {
                position = Values(DeploymentFrameConverter.UnityToIsaac(localPosition)),
                orientation_xyzw = Values(DeploymentFrameConverter.UnityToIsaac(localRotation)),
                linear_velocity = Values(DeploymentFrameConverter.UnityToIsaac(localLinear)),
                angular_velocity = Values(DeploymentFrameConverter.UnityAngularToIsaac(localAngular)),
            };
        }

        public static ObjectStateTelemetry[] Objects(
            ObjectTelemetryBinding[] bindings,
            Transform anchor)
        {
            if (bindings == null)
                return Array.Empty<ObjectStateTelemetry>();
            var result = new ObjectStateTelemetry[bindings.Length];
            for (int i = 0; i < bindings.Length; i++)
            {
                ObjectTelemetryBinding binding = bindings[i];
                bool active = binding.Active && binding.Body != null &&
                              binding.Body.gameObject.activeInHierarchy;
                BodyStateTelemetry state = active
                    ? Body(
                        binding.Body.transform,
                        binding.Body.linearVelocity,
                        binding.Body.angularVelocity,
                        anchor)
                    : new BodyStateTelemetry
                    {
                        position = new float[3],
                        orientation_xyzw = new[] { 0f, 0f, 0f, 1f },
                        linear_velocity = new float[3],
                        angular_velocity = new float[3],
                    };
                result[i] = new ObjectStateTelemetry
                {
                    id = binding.Id,
                    active = active,
                    grasped = active && binding.Grasped,
                    color = binding.Color,
                    position = state.position,
                    orientation_xyzw = state.orientation_xyzw,
                    linear_velocity = state.linear_velocity,
                    angular_velocity = state.angular_velocity,
                };
            }
            return result;
        }

        public static float[] Values(Vector3 value) => new[] { value.x, value.y, value.z };
        public static float[] Values(Quaternion value) => new[] { value.x, value.y, value.z, value.w };
    }
}
