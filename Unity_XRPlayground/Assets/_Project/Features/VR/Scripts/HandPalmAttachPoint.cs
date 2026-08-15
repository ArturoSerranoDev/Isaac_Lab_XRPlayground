using UnityEngine;
using UnityEngine.XR.Hands;

namespace XRPlayground.VR
{
    /// <summary>
    /// Keeps a child <c>PalmAttach</c> aligned to the tracked palm (hold point for direct grab).
    /// Place on Left/Right Hand next to <see cref="XRHandTrackingEvents"/>.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(XRHandTrackingEvents))]
    public sealed class HandPalmAttachPoint : MonoBehaviour
    {
        [SerializeField] XRHandTrackingEvents m_HandTrackingEvents;
        [SerializeField] Transform m_PalmAttach;
        [Tooltip("Offset in palm-local space (meters). +Y is out of the palm (XR Hands).")]
        [SerializeField] Vector3 m_LocalHoldOffset = new Vector3(0f, 0.02f, 0f);

        public Transform PalmAttach => m_PalmAttach;

        void Awake()
        {
            if (m_HandTrackingEvents == null)
                m_HandTrackingEvents = GetComponent<XRHandTrackingEvents>();
            EnsureAttach();
        }

        void OnEnable()
        {
            if (m_HandTrackingEvents == null)
                return;
            m_HandTrackingEvents.jointsUpdated.AddListener(OnJointsUpdated);
        }

        void OnDisable()
        {
            if (m_HandTrackingEvents == null)
                return;
            m_HandTrackingEvents.jointsUpdated.RemoveListener(OnJointsUpdated);
        }

        void EnsureAttach()
        {
            if (m_PalmAttach != null)
                return;

            var existing = transform.Find("PalmAttach");
            if (existing != null)
            {
                m_PalmAttach = existing;
                return;
            }

            var go = new GameObject("PalmAttach");
            m_PalmAttach = go.transform;
            m_PalmAttach.SetParent(transform, false);
            m_PalmAttach.localPosition = m_LocalHoldOffset;
            m_PalmAttach.localRotation = Quaternion.identity;
        }

        void OnJointsUpdated(XRHandJointsUpdatedEventArgs args)
        {
            EnsureAttach();
            var joint = args.hand.GetJoint(XRHandJointID.Palm);
            if (!joint.TryGetPose(out var palmPose))
            {
                // Fall back to wrist if palm is unavailable.
                joint = args.hand.GetJoint(XRHandJointID.Wrist);
                if (!joint.TryGetPose(out palmPose))
                    return;
            }

            var root = m_HandTrackingEvents.rootPose;
            var localPos = Quaternion.Inverse(root.rotation) * (palmPose.position - root.position);
            var localRot = Quaternion.Inverse(root.rotation) * palmPose.rotation;
            m_PalmAttach.localPosition = localPos + localRot * m_LocalHoldOffset;
            m_PalmAttach.localRotation = localRot;
        }
    }
}
