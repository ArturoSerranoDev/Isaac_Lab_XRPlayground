using UnityEngine;
using UnityEngine.XR.Hands;

namespace XRPlayground.VR
{
    /// <summary>
    /// Drives this transform from <see cref="XRHandTrackingEvents"/> root pose.
    /// Place on each hand root that also owns interactors.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(XRHandTrackingEvents))]
    public class XRHandRootPoseDriver : MonoBehaviour
    {
        [SerializeField]
        XRHandTrackingEvents m_HandTrackingEvents;

        [SerializeField]
        bool m_HideWhenNotTracked = true;

        void Awake()
        {
            if (m_HandTrackingEvents == null)
                m_HandTrackingEvents = GetComponent<XRHandTrackingEvents>();
        }

        void OnEnable()
        {
            if (m_HandTrackingEvents == null)
                return;

            m_HandTrackingEvents.poseUpdated.AddListener(OnPoseUpdated);
            m_HandTrackingEvents.trackingLost.AddListener(OnTrackingLost);
            m_HandTrackingEvents.trackingAcquired.AddListener(OnTrackingAcquired);

            if (m_HideWhenNotTracked)
                SetVisualActive(m_HandTrackingEvents.handIsTracked);
        }

        void OnDisable()
        {
            if (m_HandTrackingEvents == null)
                return;

            m_HandTrackingEvents.poseUpdated.RemoveListener(OnPoseUpdated);
            m_HandTrackingEvents.trackingLost.RemoveListener(OnTrackingLost);
            m_HandTrackingEvents.trackingAcquired.RemoveListener(OnTrackingAcquired);
        }

        void OnPoseUpdated(Pose rootPose)
        {
            transform.SetPositionAndRotation(rootPose.position, rootPose.rotation);
        }

        void OnTrackingAcquired()
        {
            if (m_HideWhenNotTracked)
                SetVisualActive(true);
        }

        void OnTrackingLost()
        {
            if (m_HideWhenNotTracked)
                SetVisualActive(false);
        }

        void SetVisualActive(bool active)
        {
            for (var i = 0; i < transform.childCount; ++i)
                transform.GetChild(i).gameObject.SetActive(active);
        }
    }
}
