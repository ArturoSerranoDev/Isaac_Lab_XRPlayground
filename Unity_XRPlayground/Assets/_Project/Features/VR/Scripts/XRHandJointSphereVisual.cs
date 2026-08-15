using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace XRPlayground.VR
{
    /// <summary>
    /// Simple hand mesh: small spheres on tracked joints + grey LineRenderers along finger bones.
    /// Joint poses are tracking-local (same space as <see cref="XRHandRootPoseDriver"/>).
    /// Parent this under the hand root (Camera Offset → Left/Right Hand).
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(XRHandTrackingEvents))]
    public sealed class XRHandJointSphereVisual : MonoBehaviour
    {
        static readonly XRHandJointID[] s_AllJoints =
        {
            XRHandJointID.Wrist,
            XRHandJointID.Palm,
            XRHandJointID.ThumbMetacarpal,
            XRHandJointID.ThumbProximal,
            XRHandJointID.ThumbDistal,
            XRHandJointID.ThumbTip,
            XRHandJointID.IndexMetacarpal,
            XRHandJointID.IndexProximal,
            XRHandJointID.IndexIntermediate,
            XRHandJointID.IndexDistal,
            XRHandJointID.IndexTip,
            XRHandJointID.MiddleMetacarpal,
            XRHandJointID.MiddleProximal,
            XRHandJointID.MiddleIntermediate,
            XRHandJointID.MiddleDistal,
            XRHandJointID.MiddleTip,
            XRHandJointID.RingMetacarpal,
            XRHandJointID.RingProximal,
            XRHandJointID.RingIntermediate,
            XRHandJointID.RingDistal,
            XRHandJointID.RingTip,
            XRHandJointID.LittleMetacarpal,
            XRHandJointID.LittleProximal,
            XRHandJointID.LittleIntermediate,
            XRHandJointID.LittleDistal,
            XRHandJointID.LittleTip,
        };

        // Bone chains drawn as polylines (parent → child joints).
        static readonly XRHandJointID[][] s_BoneChains =
        {
            new[] { XRHandJointID.Wrist, XRHandJointID.Palm },
            new[]
            {
                XRHandJointID.Palm, XRHandJointID.ThumbMetacarpal, XRHandJointID.ThumbProximal,
                XRHandJointID.ThumbDistal, XRHandJointID.ThumbTip,
            },
            new[]
            {
                XRHandJointID.Palm, XRHandJointID.IndexMetacarpal, XRHandJointID.IndexProximal,
                XRHandJointID.IndexIntermediate, XRHandJointID.IndexDistal, XRHandJointID.IndexTip,
            },
            new[]
            {
                XRHandJointID.Palm, XRHandJointID.MiddleMetacarpal, XRHandJointID.MiddleProximal,
                XRHandJointID.MiddleIntermediate, XRHandJointID.MiddleDistal, XRHandJointID.MiddleTip,
            },
            new[]
            {
                XRHandJointID.Palm, XRHandJointID.RingMetacarpal, XRHandJointID.RingProximal,
                XRHandJointID.RingIntermediate, XRHandJointID.RingDistal, XRHandJointID.RingTip,
            },
            new[]
            {
                XRHandJointID.Palm, XRHandJointID.LittleMetacarpal, XRHandJointID.LittleProximal,
                XRHandJointID.LittleIntermediate, XRHandJointID.LittleDistal, XRHandJointID.LittleTip,
            },
        };

        [SerializeField] XRHandTrackingEvents m_HandTrackingEvents;
        [SerializeField] float m_JointRadius = 0.008f;
        [SerializeField] float m_TipRadius = 0.01f;
        [SerializeField] float m_LineWidth = 0.004f;
        [SerializeField] Color m_JointColor = new Color(0.55f, 0.55f, 0.58f, 1f);
        [SerializeField] Color m_LineColor = new Color(0.18f, 0.18f, 0.20f, 1f);

        readonly Dictionary<XRHandJointID, Transform> m_JointSpheres = new Dictionary<XRHandJointID, Transform>();
        readonly List<LineRenderer> m_BoneLines = new List<LineRenderer>();
        readonly List<IHandJointPoseModifier> m_PoseModifiers = new List<IHandJointPoseModifier>();
        Transform m_VisualRoot;
        Material m_JointMat;
        Material m_LineMat;

        void Awake()
        {
            if (m_HandTrackingEvents == null)
                m_HandTrackingEvents = GetComponent<XRHandTrackingEvents>();
            RefreshPoseModifiers();
            EnsureVisuals();
        }

        /// <summary>Re-collect <see cref="IHandJointPoseModifier"/> components on this hand.</summary>
        public void RefreshPoseModifiers()
        {
            m_PoseModifiers.Clear();
            GetComponents(m_PoseModifiers);
        }

        void OnEnable()
        {
            if (m_HandTrackingEvents == null)
                return;
            m_HandTrackingEvents.jointsUpdated.AddListener(OnJointsUpdated);
            m_HandTrackingEvents.trackingLost.AddListener(OnTrackingLost);
            m_HandTrackingEvents.trackingAcquired.AddListener(OnTrackingAcquired);
            if (m_VisualRoot != null)
                m_VisualRoot.gameObject.SetActive(m_HandTrackingEvents.handIsTracked);
        }

        void OnDisable()
        {
            if (m_HandTrackingEvents == null)
                return;
            m_HandTrackingEvents.jointsUpdated.RemoveListener(OnJointsUpdated);
            m_HandTrackingEvents.trackingLost.RemoveListener(OnTrackingLost);
            m_HandTrackingEvents.trackingAcquired.RemoveListener(OnTrackingAcquired);
        }

        void OnDestroy()
        {
            if (m_JointMat != null)
                Destroy(m_JointMat);
            if (m_LineMat != null)
                Destroy(m_LineMat);
        }

        void OnTrackingAcquired()
        {
            if (m_VisualRoot != null)
                m_VisualRoot.gameObject.SetActive(true);
        }

        void OnTrackingLost()
        {
            if (m_VisualRoot != null)
                m_VisualRoot.gameObject.SetActive(false);
        }

        void OnJointsUpdated(XRHandJointsUpdatedEventArgs args)
        {
            var hand = args.hand;
            var root = m_HandTrackingEvents.rootPose;

            foreach (var id in s_AllJoints)
            {
                if (!m_JointSpheres.TryGetValue(id, out var t) || t == null)
                    continue;
                var joint = hand.GetJoint(id);
                if (joint.TryGetPose(out var pose))
                {
                    // Joint + root poses share tracking space; convert to hand-root local.
                    var localPos = Quaternion.Inverse(root.rotation) * (pose.position - root.position);
                    var localRot = Quaternion.Inverse(root.rotation) * pose.rotation;
                    var localPose = new Pose(localPos, localRot);
                    for (int m = 0; m < m_PoseModifiers.Count; m++)
                    {
                        if (m_PoseModifiers[m] != null &&
                            m_PoseModifiers[m].TryModifyJointLocalPose(id, in localPose, out var modified))
                            localPose = modified;
                    }

                    t.localPosition = localPose.position;
                    t.localRotation = localPose.rotation;
                    t.gameObject.SetActive(true);
                }
                else
                {
                    t.gameObject.SetActive(false);
                }
            }

            for (int c = 0; c < s_BoneChains.Length; c++)
            {
                var chain = s_BoneChains[c];
                var line = m_BoneLines[c];
                int count = 0;
                for (int i = 0; i < chain.Length; i++)
                {
                    if (!m_JointSpheres.TryGetValue(chain[i], out var jt) || jt == null || !jt.gameObject.activeSelf)
                        continue;
                    count++;
                }

                if (count < 2)
                {
                    line.enabled = false;
                    continue;
                }

                line.enabled = true;
                line.positionCount = count;
                int idx = 0;
                for (int i = 0; i < chain.Length; i++)
                {
                    if (!m_JointSpheres.TryGetValue(chain[i], out var jt) || jt == null || !jt.gameObject.activeSelf)
                        continue;
                    line.SetPosition(idx++, jt.position);
                }
            }
        }

        void EnsureVisuals()
        {
            if (m_VisualRoot != null)
                return;

            m_VisualRoot = new GameObject("Joint Visuals").transform;
            m_VisualRoot.SetParent(transform, false);

            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            m_JointMat = new Material(shader) { name = "HandJointMat", color = m_JointColor };
            if (m_JointMat.HasProperty("_BaseColor"))
                m_JointMat.SetColor("_BaseColor", m_JointColor);
            if (m_JointMat.HasProperty("_Metallic"))
                m_JointMat.SetFloat("_Metallic", 0.1f);
            if (m_JointMat.HasProperty("_Smoothness"))
                m_JointMat.SetFloat("_Smoothness", 0.35f);

            m_LineMat = new Material(Shader.Find("Sprites/Default") ?? shader) { name = "HandBoneMat", color = m_LineColor };

            foreach (var id in s_AllJoints)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = id.ToString();
                go.transform.SetParent(m_VisualRoot, false);
                float r = id.ToString().EndsWith("Tip") ? m_TipRadius : m_JointRadius;
                go.transform.localScale = Vector3.one * (r * 2f);
                var col = go.GetComponent<Collider>();
                if (col != null)
                    Destroy(col);
                var rend = go.GetComponent<MeshRenderer>();
                if (rend != null)
                    rend.sharedMaterial = m_JointMat;
                go.SetActive(false);
                m_JointSpheres[id] = go.transform;
            }

            for (int c = 0; c < s_BoneChains.Length; c++)
            {
                var lineGo = new GameObject($"BoneLine_{c}");
                lineGo.transform.SetParent(m_VisualRoot, false);
                var line = lineGo.AddComponent<LineRenderer>();
                line.sharedMaterial = m_LineMat;
                line.startColor = m_LineColor;
                line.endColor = m_LineColor;
                line.startWidth = m_LineWidth;
                line.endWidth = m_LineWidth * 0.75f;
                line.useWorldSpace = true;
                line.positionCount = 0;
                line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                line.receiveShadows = false;
                line.enabled = false;
                m_BoneLines.Add(line);
            }
        }
    }
}
