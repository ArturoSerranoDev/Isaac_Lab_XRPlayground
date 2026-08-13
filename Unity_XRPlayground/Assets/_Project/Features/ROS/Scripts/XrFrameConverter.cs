using System;
using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>Isaac Z-up ↔ Unity Y-up. Wire payloads use Isaac coordinates.</summary>
    public static class XrFrameConverter
    {
        public static Vector3 IsaacPosToUnity(Vector3 p) => new Vector3(p.x, p.z, p.y);

        public static Vector3 UnityPosToIsaac(Vector3 p) => new Vector3(p.x, p.z, p.y);

        public static Quaternion IsaacQuatToUnity(Quaternion qIsaacXyzw)
        {
            // Match Python bridge: (x, y, z, w)_isaac → (x, z, y, -w)_unity
            return new Quaternion(qIsaacXyzw.x, qIsaacXyzw.z, qIsaacXyzw.y, -qIsaacXyzw.w);
        }

        public static Quaternion UnityQuatToIsaac(Quaternion qUnity)
        {
            return new Quaternion(qUnity.x, qUnity.z, qUnity.y, -qUnity.w);
        }

        public static float[] ToArray(Vector3 v) => new[] { v.x, v.y, v.z };

        public static float[] ToXyzw(Quaternion q) => new[] { q.x, q.y, q.z, q.w };

        public static Vector3 FromArray3(float[] a)
        {
            if (a == null || a.Length < 3)
                return Vector3.zero;
            return new Vector3(a[0], a[1], a[2]);
        }

        public static Quaternion FromXyzw(float[] a)
        {
            if (a == null || a.Length < 4)
                return Quaternion.identity;
            return new Quaternion(a[0], a[1], a[2], a[3]);
        }
    }
}
