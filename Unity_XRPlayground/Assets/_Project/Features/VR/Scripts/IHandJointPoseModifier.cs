using UnityEngine;
using UnityEngine.XR.Hands;

namespace XRPlayground.VR
{
    /// <summary>
    /// Optional post-process for tracked joint locals (used by finger-wrap grab).
    /// </summary>
    public interface IHandJointPoseModifier
    {
        /// <summary>
        /// Blend or replace a tracking-local joint pose. Return true if <paramref name="modifiedLocal"/> is used.
        /// </summary>
        bool TryModifyJointLocalPose(XRHandJointID id, in Pose trackedLocal, out Pose modifiedLocal);
    }
}
