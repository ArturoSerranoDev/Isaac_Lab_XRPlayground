using UnityEngine;

namespace XRPlayground.Deployment
{
    public static class DeploymentFrameConverter
    {
        public static Vector3 IsaacToUnity(Vector3 value) => new(value.x, value.z, value.y);
        public static Vector3 UnityToIsaac(Vector3 value) => new(value.x, value.z, value.y);
        // The Y/Z swap changes handedness. Angular velocity is an axial
        // vector, so it gains the determinant sign that polar vectors such as
        // position and linear velocity do not.
        public static Vector3 IsaacAngularToUnity(Vector3 value) =>
            new(-value.x, -value.z, -value.y);
        public static Vector3 UnityAngularToIsaac(Vector3 value) =>
            new(-value.x, -value.z, -value.y);
        public static Quaternion IsaacToUnity(Quaternion value) =>
            new(value.x, value.z, value.y, -value.w);
        public static Quaternion UnityToIsaac(Quaternion value) =>
            new(value.x, value.z, value.y, -value.w);

        public static Vector3 Vector3From(float[] values) =>
            values != null && values.Length >= 3
                ? new Vector3(values[0], values[1], values[2])
                : Vector3.zero;

        public static Quaternion QuaternionFrom(float[] values) =>
            values != null && values.Length >= 4
                ? new Quaternion(values[0], values[1], values[2], values[3])
                : Quaternion.identity;
    }
}
