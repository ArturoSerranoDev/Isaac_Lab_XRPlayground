using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace XRPlayground.VR
{
    /// <summary>
    /// Closes finger joint visuals around a grabbed collider (AutoHand-style envelop).
    /// Works with <see cref="XRHandJointSphereVisual"/> via <see cref="IHandJointPoseModifier"/>.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(XRHandTrackingEvents))]
    public sealed class HandFingerWrapDriver : MonoBehaviour, IHandJointPoseModifier
    {
        static readonly XRHandJointID[][] s_FingerChains =
        {
            new[]
            {
                XRHandJointID.ThumbMetacarpal, XRHandJointID.ThumbProximal,
                XRHandJointID.ThumbDistal, XRHandJointID.ThumbTip,
            },
            new[]
            {
                XRHandJointID.IndexMetacarpal, XRHandJointID.IndexProximal,
                XRHandJointID.IndexIntermediate, XRHandJointID.IndexDistal, XRHandJointID.IndexTip,
            },
            new[]
            {
                XRHandJointID.MiddleMetacarpal, XRHandJointID.MiddleProximal,
                XRHandJointID.MiddleIntermediate, XRHandJointID.MiddleDistal, XRHandJointID.MiddleTip,
            },
            new[]
            {
                XRHandJointID.RingMetacarpal, XRHandJointID.RingProximal,
                XRHandJointID.RingIntermediate, XRHandJointID.RingDistal, XRHandJointID.RingTip,
            },
            new[]
            {
                XRHandJointID.LittleMetacarpal, XRHandJointID.LittleProximal,
                XRHandJointID.LittleIntermediate, XRHandJointID.LittleDistal, XRHandJointID.LittleTip,
            },
        };

        [SerializeField] XRHandTrackingEvents m_HandTrackingEvents;
        [SerializeField] float m_ContactPadding = 0.008f;
        [SerializeField] float m_MaxCurlBlend = 0.92f;
        [Tooltip("How quickly wrap amount rises when attract is already settled.")]
        [SerializeField] float m_WrapRampSpeed = 4f;

        Transform m_Target;
        Collider m_TargetCollider;
        bool m_Wrapping;
        float m_WrapAmount;
        float m_ExternalWrapAmount = -1f;

        readonly Dictionary<XRHandJointID, Pose> m_CachedTracked = new Dictionary<XRHandJointID, Pose>();

        public bool IsWrapping => m_Wrapping;
        public float WrapAmount => m_WrapAmount;

        void Awake()
        {
            if (m_HandTrackingEvents == null)
                m_HandTrackingEvents = GetComponent<XRHandTrackingEvents>();
        }

        public void BeginWrap(Transform target, Collider targetCollider)
        {
            m_Target = target;
            m_TargetCollider = targetCollider;
            m_Wrapping = target != null;
            m_WrapAmount = 0f;
            m_ExternalWrapAmount = 0f;
        }

        public void EndWrap()
        {
            m_Wrapping = false;
            m_Target = null;
            m_TargetCollider = null;
            m_WrapAmount = 0f;
            m_ExternalWrapAmount = -1f;
        }

        /// <summary>Driven by <see cref="PalmAttractGrab"/> (0–1).</summary>
        public void SetWrapAmount(float amount)
        {
            m_ExternalWrapAmount = Mathf.Clamp01(amount);
        }

        void Update()
        {
            if (!m_Wrapping)
            {
                m_WrapAmount = 0f;
                return;
            }

            float target = m_ExternalWrapAmount >= 0f ? m_ExternalWrapAmount : 1f;
            // Keep closing a bit after attract settles so fingers finish enveloping.
            target = Mathf.Clamp01(target * 0.85f + 0.15f);
            m_WrapAmount = Mathf.MoveTowards(m_WrapAmount, target, m_WrapRampSpeed * Time.deltaTime);
        }

        public bool TryModifyJointLocalPose(XRHandJointID id, in Pose trackedLocal, out Pose modifiedLocal)
        {
            modifiedLocal = trackedLocal;
            if (!m_Wrapping || m_WrapAmount <= 0.001f || m_Target == null || m_HandTrackingEvents == null)
                return false;

            m_CachedTracked[id] = trackedLocal;

            if (!IsFingerJoint(id))
                return false;

            var root = m_HandTrackingEvents.rootPose;
            Vector3 trackedWorld = root.position + root.rotation * trackedLocal.position;

            Vector3 surfaceWorld = SampleSurface(trackedWorld);
            Vector3 surfaceLocalPos = Quaternion.Inverse(root.rotation) * (surfaceWorld - root.position);

            // Curl tips strongly; proximal joints less so they keep a natural chain shape.
            float jointWeight = TipWeight(id);
            float blend = m_WrapAmount * jointWeight * m_MaxCurlBlend;
            if (blend <= 0.001f)
                return false;

            Vector3 curledPos = Vector3.Lerp(trackedLocal.position, surfaceLocalPos, blend);

            // Face the tip toward the object center for a grasp-looking orientation.
            Vector3 toCenterWorld = (m_TargetCollider != null ? m_TargetCollider.bounds.center : m_Target.position) - trackedWorld;
            Quaternion faceRot = trackedLocal.rotation;
            if (toCenterWorld.sqrMagnitude > 1e-8f)
            {
                Vector3 localFwd = Quaternion.Inverse(root.rotation) * toCenterWorld.normalized;
                faceRot = Quaternion.Slerp(trackedLocal.rotation, Quaternion.LookRotation(localFwd, Vector3.up), blend);
            }

            // Soft chain: pull intermediate joints partway toward a line from metacarpal to curled tip.
            if (TryGetChainCurl(id, curledPos, out var chainPos))
                curledPos = Vector3.Lerp(curledPos, chainPos, 0.35f * m_WrapAmount);

            modifiedLocal = new Pose(curledPos, faceRot);
            return true;
        }

        Vector3 SampleSurface(Vector3 fromWorld)
        {
            if (m_TargetCollider != null)
            {
                Vector3 closest = m_TargetCollider.ClosestPoint(fromWorld);
                Vector3 n = (fromWorld - closest).normalized;
                if (n.sqrMagnitude < 1e-6f)
                    n = (fromWorld - m_TargetCollider.bounds.center).normalized;
                return closest + n * m_ContactPadding;
            }

            return m_Target.position;
        }

        bool TryGetChainCurl(XRHandJointID id, Vector3 tipLocalGoal, out Vector3 chainPos)
        {
            chainPos = tipLocalGoal;
            foreach (var chain in s_FingerChains)
            {
                int idx = System.Array.IndexOf(chain, id);
                if (idx < 0)
                    continue;
                if (!m_CachedTracked.TryGetValue(chain[0], out var basePose))
                    return false;

                float t = chain.Length <= 1 ? 1f : (float)idx / (chain.Length - 1);
                chainPos = Vector3.Lerp(basePose.position, tipLocalGoal, t);
                return true;
            }

            return false;
        }

        static bool IsFingerJoint(XRHandJointID id)
        {
            foreach (var chain in s_FingerChains)
            {
                for (int i = 0; i < chain.Length; i++)
                {
                    if (chain[i] == id)
                        return true;
                }
            }

            return false;
        }

        static float TipWeight(XRHandJointID id)
        {
            string n = id.ToString();
            if (n.EndsWith("Tip"))
                return 1f;
            if (n.EndsWith("Distal"))
                return 0.85f;
            if (n.EndsWith("Intermediate"))
                return 0.55f;
            if (n.EndsWith("Proximal"))
                return 0.3f;
            if (n.EndsWith("Metacarpal"))
                return 0.12f;
            return 0.4f;
        }
    }
}
