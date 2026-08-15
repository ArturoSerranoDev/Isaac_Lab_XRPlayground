using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;

namespace XRPlayground.VR
{
    /// <summary>
    /// AutoHand-style palm attract: while selecting, the interactor attach point slides from the
    /// object toward the palm so Instantaneous/Velocity grab pulls the object in (no ray/far grab).
    /// Place on the Near-Far / Direct grab interactor child.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(XRBaseInteractor))]
    public sealed class PalmAttractGrab : MonoBehaviour
    {
        [SerializeField] XRBaseInteractor m_Interactor;
        [SerializeField] HandPalmAttachPoint m_PalmAttach;
        [SerializeField] HandFingerWrapDriver m_FingerWrap;

        [Tooltip("How fast the attach point (and thus the object) slides toward the palm.")]
        [SerializeField] float m_AttractSpeed = 10f;

        [Tooltip("Distance at which attract settles and fingers finish closing.")]
        [SerializeField] float m_SettleDistance = 0.025f;

        [Tooltip("Extra push of the hold point away from the palm along palm forward.")]
        [SerializeField] float m_HoldSurfacePadding = 0.01f;

        Transform m_DynamicAttach;
        Transform m_Grabbed;
        Collider m_GrabbedCollider;
        bool m_Attracting;
        float m_AttractT;

        public bool IsAttracting => m_Attracting;
        public float AttractT => m_AttractT;
        public Transform GrabbedTransform => m_Grabbed;

        void Awake()
        {
            if (m_Interactor == null)
                m_Interactor = GetComponent<XRBaseInteractor>();
            if (m_PalmAttach == null)
                m_PalmAttach = GetComponentInParent<HandPalmAttachPoint>();
            if (m_FingerWrap == null)
                m_FingerWrap = GetComponentInParent<HandFingerWrapDriver>();

            var attachGo = new GameObject("AttractAttach");
            m_DynamicAttach = attachGo.transform;
            m_DynamicAttach.SetParent(transform, false);
        }

        void OnEnable()
        {
            if (m_Interactor == null)
                return;
            m_Interactor.selectEntered.AddListener(OnSelectEntered);
            m_Interactor.selectExited.AddListener(OnSelectExited);
        }

        void OnDisable()
        {
            if (m_Interactor == null)
                return;
            m_Interactor.selectEntered.RemoveListener(OnSelectEntered);
            m_Interactor.selectExited.RemoveListener(OnSelectExited);
            EndGrab();
        }

        void OnSelectEntered(SelectEnterEventArgs args)
        {
            m_Grabbed = args.interactableObject.transform;
            m_GrabbedCollider = m_Grabbed.GetComponentInChildren<Collider>();
            m_Attracting = true;
            m_AttractT = 0f;

            // Start attach at the object so XRIT hold begins where the object is, then slide to palm.
            var start = EstimateHoldWorldPose(m_Grabbed, m_GrabbedCollider, m_PalmAttach != null ? m_PalmAttach.PalmAttach : transform);
            m_DynamicAttach.SetPositionAndRotation(start.position, start.rotation);
            m_Interactor.attachTransform = m_DynamicAttach;

            if (m_FingerWrap != null)
                m_FingerWrap.BeginWrap(m_Grabbed, m_GrabbedCollider);
        }

        void OnSelectExited(SelectExitEventArgs args)
        {
            EndGrab();
        }

        void EndGrab()
        {
            m_Attracting = false;
            m_AttractT = 0f;
            m_Grabbed = null;
            m_GrabbedCollider = null;
            if (m_FingerWrap != null)
                m_FingerWrap.EndWrap();
        }

        void LateUpdate()
        {
            if (!m_Attracting || m_Grabbed == null || m_PalmAttach == null || m_PalmAttach.PalmAttach == null)
                return;

            var palm = m_PalmAttach.PalmAttach;
            var target = EstimateHoldWorldPose(m_Grabbed, m_GrabbedCollider, palm);
            float step = 1f - Mathf.Exp(-m_AttractSpeed * Time.deltaTime);
            m_DynamicAttach.position = Vector3.Lerp(m_DynamicAttach.position, target.position, step);
            m_DynamicAttach.rotation = Quaternion.Slerp(m_DynamicAttach.rotation, target.rotation, step);

            float dist = Vector3.Distance(m_DynamicAttach.position, target.position);
            m_AttractT = Mathf.Clamp01(1f - dist / Mathf.Max(0.05f, m_SettleDistance * 4f));
            if (dist <= m_SettleDistance)
                m_AttractT = 1f;

            if (m_FingerWrap != null)
                m_FingerWrap.SetWrapAmount(m_AttractT);
        }

        Pose EstimateHoldWorldPose(Transform grabbed, Collider col, Transform palm)
        {
            Vector3 center = grabbed.position;
            if (col != null)
                center = col.bounds.center;

            float radius = 0.04f;
            if (col != null)
                radius = Mathf.Max(0.015f, col.bounds.extents.magnitude * 0.35f);

            // XR Hands: palm.up points out of the palm; sit the object on that face.
            Vector3 holdPos = palm.position + palm.up * (radius + m_HoldSurfacePadding);
            Quaternion holdRot = palm.rotation;
            return new Pose(holdPos, holdRot);
        }
    }
}
