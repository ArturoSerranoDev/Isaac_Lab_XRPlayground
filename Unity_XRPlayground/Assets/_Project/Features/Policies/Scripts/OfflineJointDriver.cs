using System.Collections.Generic;
using UnityEngine;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Offline visual FK that matches ROS mirror: compute Isaac Z-up body poses, then
    /// <see cref="XrFrameConverter.ApplyIsaacBody"/> (pos xzy, quat x,z,y,-w + envAnchor).
    /// Does not twist USD localRotation (that breaks the imported hierarchy).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class OfflineJointDriver : MonoBehaviour
    {
        [System.Serializable]
        public struct JointBinding
        {
            public string name;
            public Transform link;
            public Vector3 localAxis;
            [HideInInspector] public Quaternion restLocalRotation;
        }

        [Tooltip("Legacy inspector list (filled by Bind*). FK uses the kinematic chain, not localRotation.")]
        public JointBinding[] joints;

        public Transform envAnchor;

        struct Node
        {
            public string name;
            public int parent;
            public int joint;
            public float mimic;
            public Vector3 axisIsaac;
            public Vector3 restPosIsaac;
            public Quaternion restRotIsaac;
            public Transform link;
        }

        Node[] _nodes;
        float[] _restJoints;
        float[] _pending;
        bool _dirty;

        void LateUpdate()
        {
            if (!_dirty || _nodes == null)
                return;
            _dirty = false;
            ApplyPending();
        }

        public void CaptureRestPose()
        {
            CaptureIsaacRestFromUnity();
        }

        public void ApplyAnglesRadians(float[] angles)
        {
            if (angles == null)
                return;
            if (_pending == null || _pending.Length != angles.Length)
                _pending = new float[angles.Length];
            System.Array.Copy(angles, _pending, angles.Length);
            _dirty = true;
        }

        public bool TryBindByNames(Transform searchRoot, string[] linkNames, Vector3 defaultAxis)
        {
            if (linkNames == null)
                return false;
            joints = new JointBinding[linkNames.Length];
            for (int i = 0; i < linkNames.Length; i++)
            {
                joints[i] = new JointBinding
                {
                    name = linkNames[i],
                    link = FindDeep(searchRoot, linkNames[i]),
                    localAxis = defaultAxis,
                };
            }
            return true;
        }

        public void BindUr10e(RobotLinkMap map, Transform anchor)
        {
            envAnchor = anchor != null ? anchor : envAnchor;
            if (map != null)
                map.Rebuild();

            // Isaac UR10e_ROBOTIQ_2F_85 default_joint_pos (must match USD bind).
            _restJoints = new[]
            {
                Mathf.PI, -Mathf.PI * 0.5f, Mathf.PI * 0.5f, -Mathf.PI * 0.5f, -Mathf.PI * 0.5f, 0f, 0f,
            };

            var z = Vector3.forward;
            var y = Vector3.up;
            var list = new List<Node>(16);
            Add(list, map, "base_link", -1, -1, 0f, z);
            Add(list, map, "shoulder_link", IndexOf(list, "base_link"), 0, 1f, z);
            Add(list, map, "upper_arm_link", IndexOf(list, "shoulder_link"), 1, 1f, y);
            Add(list, map, "forearm_link", IndexOf(list, "upper_arm_link"), 2, 1f, y);
            Add(list, map, "wrist_1_link", IndexOf(list, "forearm_link"), 3, 1f, y);
            Add(list, map, "wrist_2_link", IndexOf(list, "wrist_1_link"), 4, 1f, z);
            Add(list, map, "wrist_3_link", IndexOf(list, "wrist_2_link"), 5, 1f, y);
            Add(list, map, "base_link_0", IndexOf(list, "wrist_3_link"), -1, 0f, z);
            Add(list, map, "left_outer_knuckle", IndexOf(list, "base_link_0"), 6, 1f, z);
            Add(list, map, "left_outer_finger", IndexOf(list, "left_outer_knuckle"), -1, 0f, z);
            Add(list, map, "left_inner_knuckle", IndexOf(list, "base_link_0"), 6, 1f, z);
            Add(list, map, "left_inner_finger", IndexOf(list, "left_inner_knuckle"), -1, 0f, z);
            Add(list, map, "right_outer_knuckle", IndexOf(list, "base_link_0"), 6, -1f, z);
            Add(list, map, "right_outer_finger", IndexOf(list, "right_outer_knuckle"), -1, 0f, z);
            Add(list, map, "right_inner_knuckle", IndexOf(list, "base_link_0"), 6, -1f, z);
            Add(list, map, "right_inner_finger", IndexOf(list, "right_inner_knuckle"), -1, 0f, z);
            _nodes = list.ToArray();
            FillLegacyJoints();
            CaptureIsaacRestFromUnity();
        }

        public void BindKinova(KinovaLinkMap map, Transform anchor)
        {
            envAnchor = anchor != null ? anchor : envAnchor;
            if (map != null)
                map.Rebuild();

            _restJoints = new[] { 0f, 2.35f, 0.25f, 1.65f, 1.40f, 0.35f, 0f, 0.2f };

            var z = Vector3.forward;
            var list = new List<Node>(16);
            Add(list, map, "j2n7s300_link_base", -1, -1, 0f, z);
            Add(list, map, "j2n7s300_link_1", IndexOf(list, "j2n7s300_link_base"), 0, 1f, z);
            Add(list, map, "j2n7s300_link_2", IndexOf(list, "j2n7s300_link_1"), 1, 1f, z);
            Add(list, map, "j2n7s300_link_3", IndexOf(list, "j2n7s300_link_2"), 2, 1f, z);
            Add(list, map, "j2n7s300_link_4", IndexOf(list, "j2n7s300_link_3"), 3, 1f, z);
            Add(list, map, "j2n7s300_link_5", IndexOf(list, "j2n7s300_link_4"), 4, 1f, z);
            Add(list, map, "j2n7s300_link_6", IndexOf(list, "j2n7s300_link_5"), 5, 1f, z);
            Add(list, map, "j2n7s300_link_7", IndexOf(list, "j2n7s300_link_6"), 6, 1f, z);
            Add(list, map, "j2n7s300_end_effector", IndexOf(list, "j2n7s300_link_7"), -1, 0f, z);
            Add(list, map, "j2n7s300_link_finger_1", IndexOf(list, "j2n7s300_end_effector"), 7, 1f, z);
            Add(list, map, "j2n7s300_link_finger_2", IndexOf(list, "j2n7s300_end_effector"), 7, 1f, z);
            Add(list, map, "j2n7s300_link_finger_3", IndexOf(list, "j2n7s300_end_effector"), 7, 1f, z);
            Add(list, map, "j2n7s300_link_finger_tip_1", IndexOf(list, "j2n7s300_link_finger_1"), 7, 1f, z);
            Add(list, map, "j2n7s300_link_finger_tip_2", IndexOf(list, "j2n7s300_link_finger_2"), 7, 1f, z);
            Add(list, map, "j2n7s300_link_finger_tip_3", IndexOf(list, "j2n7s300_link_finger_3"), 7, 1f, z);
            _nodes = list.ToArray();
            FillLegacyJoints();
            CaptureIsaacRestFromUnity();
        }

        public void BindAgibotA2D(AgibotLinkMap map, Transform anchor)
        {
            envAnchor = anchor != null ? anchor : envAnchor;
            if (map != null)
                map.Rebuild();

            // Isaac AGIBOT_A2D_CFG default right arm + open gripper
            _restJoints = new[]
            {
                1.0817f, -0.5907f, -0.3442f, 1.2819f, -0.6928f, -0.7f, 0f, 0.994f,
            };

            var z = Vector3.forward;
            var list = new List<Node>(20);
            Add(list, map, "base_link", -1, -1, 0f, z);
            Add(list, map, "body_link", IndexOf(list, "base_link"), -1, 0f, z);
            Add(list, map, "head_link", IndexOf(list, "body_link"), -1, 0f, z);
            Add(list, map, "right_arm_link1", IndexOf(list, "body_link"), 0, 1f, z);
            Add(list, map, "right_arm_link2", IndexOf(list, "right_arm_link1"), 1, 1f, z);
            Add(list, map, "right_arm_link3", IndexOf(list, "right_arm_link2"), 2, 1f, z);
            Add(list, map, "right_arm_link4", IndexOf(list, "right_arm_link3"), 3, 1f, z);
            Add(list, map, "right_arm_link5", IndexOf(list, "right_arm_link4"), 4, 1f, z);
            Add(list, map, "right_arm_link6", IndexOf(list, "right_arm_link5"), 5, 1f, z);
            Add(list, map, "right_arm_link7", IndexOf(list, "right_arm_link6"), 6, 1f, z);
            Add(list, map, "right_gripper_base", IndexOf(list, "right_arm_link7"), -1, 0f, z);
            Add(list, map, "right_gripper_center", IndexOf(list, "right_gripper_base"), 7, 1f, z);
            Add(list, map, "right_Left_Pad_Link", IndexOf(list, "right_gripper_center"), 7, 1f, z);
            Add(list, map, "right_Right_Pad_Link", IndexOf(list, "right_gripper_center"), 7, -1f, z);
            _nodes = list.ToArray();
            FillLegacyJoints();
            CaptureIsaacRestFromUnity();
        }

        void Add(List<Node> list, AgibotLinkMap map, string name, int parent, int joint, float mimic, Vector3 axis)
        {
            Transform t = null;
            map?.TryGet(name, out t);
            list.Add(new Node
            {
                name = name,
                parent = parent,
                joint = joint,
                mimic = mimic,
                axisIsaac = axis,
                link = t,
            });
        }

        void Add(List<Node> list, RobotLinkMap map, string name, int parent, int joint, float mimic, Vector3 axis)
        {
            Transform t = null;
            map?.TryGet(name, out t);
            list.Add(new Node
            {
                name = name,
                parent = parent,
                joint = joint,
                mimic = mimic,
                axisIsaac = axis,
                link = t,
            });
        }

        void Add(List<Node> list, KinovaLinkMap map, string name, int parent, int joint, float mimic, Vector3 axis)
        {
            Transform t = null;
            map?.TryGet(name, out t);
            list.Add(new Node
            {
                name = name,
                parent = parent,
                joint = joint,
                mimic = mimic,
                axisIsaac = axis,
                link = t,
            });
        }

        static int IndexOf(List<Node> list, string name)
        {
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i].name == name)
                    return i;
            }
            return -1;
        }

        void FillLegacyJoints()
        {
            if (_nodes == null)
                return;
            joints = new JointBinding[_nodes.Length];
            for (int i = 0; i < _nodes.Length; i++)
            {
                joints[i] = new JointBinding
                {
                    name = _nodes[i].name,
                    link = _nodes[i].link,
                    localAxis = _nodes[i].axisIsaac,
                };
            }
        }

        void CaptureIsaacRestFromUnity()
        {
            if (_nodes == null || envAnchor == null)
                return;
            for (int i = 0; i < _nodes.Length; i++)
            {
                var n = _nodes[i];
                if (n.link == null)
                    continue;
                XrFrameConverter.UnityWorldToIsaacLocal(
                    envAnchor, n.link.position, n.link.rotation, out n.restPosIsaac, out n.restRotIsaac);
                _nodes[i] = n;
            }
        }

        void ApplyPending()
        {
            if (_nodes == null || envAnchor == null || _pending == null)
                return;

            var pos = new Vector3[_nodes.Length];
            var rot = new Quaternion[_nodes.Length];

            for (int i = 0; i < _nodes.Length; i++)
            {
                var n = _nodes[i];
                if (n.parent < 0)
                {
                    pos[i] = n.restPosIsaac;
                    rot[i] = n.restRotIsaac;
                }
                else
                {
                    var p = _nodes[n.parent];
                    Quaternion parentRestInv = InverseIsaac(p.restRotIsaac);
                    Quaternion relRot = MulIsaac(parentRestInv, n.restRotIsaac);
                    Vector3 relPos = RotateIsaac(parentRestInv, n.restPosIsaac - p.restPosIsaac);

                    float dq = 0f;
                    if (n.joint >= 0 && n.joint < _pending.Length)
                    {
                        float rest = n.joint < _restJoints.Length ? _restJoints[n.joint] : 0f;
                        dq = (_pending[n.joint] - rest) * n.mimic;
                    }

                    Quaternion jointRot = IsaacAngleAxis(dq, n.axisIsaac);
                    Quaternion childRot = MulIsaac(MulIsaac(rot[n.parent], relRot), jointRot);
                    Vector3 childPos = pos[n.parent] + RotateIsaac(rot[n.parent], relPos);
                    pos[i] = childPos;
                    rot[i] = childRot;
                }

                if (n.link != null)
                    XrFrameConverter.ApplyIsaacBody(n.link, envAnchor, pos[i], rot[i]);
            }
        }

        static Quaternion IsaacAngleAxis(float radians, Vector3 axis)
        {
            if (axis.sqrMagnitude < 1e-8f)
                return new Quaternion(0f, 0f, 0f, 1f);
            axis.Normalize();
            float h = radians * 0.5f;
            float s = Mathf.Sin(h);
            return new Quaternion(axis.x * s, axis.y * s, axis.z * s, Mathf.Cos(h));
        }

        static Quaternion InverseIsaac(Quaternion q)
        {
            return new Quaternion(-q.x, -q.y, -q.z, q.w);
        }

        static Quaternion MulIsaac(Quaternion a, Quaternion b)
        {
            return a * b;
        }

        static Vector3 RotateIsaac(Quaternion q, Vector3 v)
        {
            var qv = new Quaternion(v.x, v.y, v.z, 0f);
            var r = MulIsaac(MulIsaac(q, qv), InverseIsaac(q));
            return new Vector3(r.x, r.y, r.z);
        }

        static Transform FindDeep(Transform root, string name)
        {
            if (root == null)
                return null;
            if (root.name == name)
                return root;
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
            {
                if (t.name == name)
                    return t;
            }
            return null;
        }
    }
}
