using UnityEngine;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Unity-only Agibot pick-place: 30-D obs → ONNX → 8-D actions (right arm + gripper).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PickPlaceOfflinePolicyController : MonoBehaviour
    {
        public const int ObsDim = 30;
        public const int ActionDim = 8;

        [Header("Policy")]
        public OnnxPolicyRunner policyRunner;
        public bool running;

        [Header("Scene refs")]
        public Transform envAnchor;
        public AgibotLinkMap linkMap;
        public OfflineJointDriver jointDriver;
        public Transform piece;
        public Rigidbody pieceBody;
        public Transform eeLink;

        [Header("Training-matched params (Isaac Z-up)")]
        public float actionScale = 5f;
        public float dofVelocityScale = 0.1f;
        public float controlDt = 1f / 60f;
        public float gripperOpen = 0.994f;
        public float gripperClose = 0.20f;
        public Vector3 bucketPosIsaac = new Vector3(0.22f, 0f, 0.54f);
        public Vector3 spawnPosIsaacMin = new Vector3(0.40f, -0.22f, 0.44f);
        public Vector3 spawnPosIsaacMax = new Vector3(0.70f, 0.22f, 0.44f);
        public float liftHeight = 0.52f;
        public float graspDist = 0.12f;

        static readonly string[] ArmJointNames =
        {
            "right_arm_joint1",
            "right_arm_joint2",
            "right_arm_joint3",
            "right_arm_joint4",
            "right_arm_joint5",
            "right_arm_joint6",
            "right_arm_joint7",
        };

        readonly float[] _armPos = new float[7];
        readonly float[] _armVel = new float[7];
        readonly float[] _prevArm = new float[7];
        float _gripPos;
        float _gripVel;
        float _prevGrip;
        readonly float[] _obs = new float[ObsDim];
        readonly float[] _actions = new float[ActionDim];
        Vector3 _pieceIsaac;
        Vector3 _pieceVelIsaac;
        bool _grasped;
        float _accum;
        bool _followersPaused;

        public string StatusLine { get; private set; } = "idle";

        void Awake()
        {
            if (envAnchor == null)
                envAnchor = transform;
            if (policyRunner == null)
                policyRunner = GetComponent<OnnxPolicyRunner>();
            if (jointDriver == null)
                jointDriver = GetComponent<OfflineJointDriver>();
            if (piece != null && pieceBody == null)
                pieceBody = piece.GetComponent<Rigidbody>();
            AutoBindLinks();
        }

        [ContextMenu("Auto-Bind Links")]
        public void AutoBindLinks()
        {
            if (linkMap != null)
            {
                linkMap.Rebuild();
                if (eeLink == null)
                    linkMap.TryGet("right_gripper_center", out eeLink);
            }

            if (jointDriver != null)
            {
                jointDriver.envAnchor = envAnchor != null ? envAnchor : transform;
                jointDriver.BindAgibotA2D(linkMap, jointDriver.envAnchor);
            }
        }

        public void StartPolicy()
        {
            if (policyRunner == null || policyRunner.modelAsset == null)
            {
                Debug.LogError("PickPlaceOfflinePolicyController: assign policy.onnx.", this);
                StatusLine = "missing model";
                return;
            }

            policyRunner.expectedObsDim = ObsDim;
            policyRunner.expectedActionDim = ActionDim;
            if (!policyRunner.TryLoad())
            {
                StatusLine = "model load failed";
                return;
            }

            PauseBridgeFollowers(true);
            AutoBindLinks();
            ResetEpisode();
            running = true;
            StatusLine = "running";
        }

        public void StopPolicy()
        {
            running = false;
            PauseBridgeFollowers(false);
            StatusLine = "stopped";
        }

        public void TogglePolicy()
        {
            if (running)
                StopPolicy();
            else
                StartPolicy();
        }

        void Update()
        {
            if (!running)
                return;
            _accum += Time.deltaTime;
            while (_accum >= controlDt)
            {
                _accum -= controlDt;
                Step(controlDt);
            }
        }

        void Step(float dt)
        {
            UpdateGraspMagnet();
            BuildObs();
            if (!policyRunner.TryInfer(_obs, _actions))
            {
                StatusLine = "infer failed";
                StopPolicy();
                return;
            }
            ApplyActions(dt);
            SyncVisuals();
            StatusLine = _grasped ? "grasped" : "approach";
        }

        void ResetEpisode()
        {
            _grasped = false;
            RandomSpawnPiece();
            for (int i = 0; i < 7; i++)
                _prevArm[i] = _armPos[i];
            _prevGrip = _gripPos;
        }

        void RandomSpawnPiece()
        {
            _pieceIsaac = new Vector3(
                Random.Range(spawnPosIsaacMin.x, spawnPosIsaacMax.x),
                Random.Range(spawnPosIsaacMin.y, spawnPosIsaacMax.y),
                spawnPosIsaacMin.z);
            _pieceVelIsaac = Vector3.zero;
            if (piece != null)
            {
                var unity = XrFrameConverter.IsaacPosToUnity(_pieceIsaac);
                piece.localPosition = unity;
                if (pieceBody != null)
                {
                    pieceBody.linearVelocity = Vector3.zero;
                    pieceBody.angularVelocity = Vector3.zero;
                }
            }
        }

        void BuildObs()
        {
            ReadArmState();
            var eeIsaac = EePosIsaac();
            var eeToPiece = _pieceIsaac - eeIsaac;
            var pieceToBucket = bucketPosIsaac - _pieceIsaac;
            float lifted = _pieceIsaac.z > liftHeight ? 1f : 0f;

            int o = 0;
            for (int i = 0; i < 7; i++)
                _obs[o++] = _armPos[i];
            for (int i = 0; i < 7; i++)
                _obs[o++] = _armVel[i];
            _obs[o++] = _gripPos;
            _obs[o++] = _gripVel;
            _obs[o++] = _pieceIsaac.x;
            _obs[o++] = _pieceIsaac.y;
            _obs[o++] = _pieceIsaac.z;
            _obs[o++] = _pieceVelIsaac.x;
            _obs[o++] = _pieceVelIsaac.y;
            _obs[o++] = _pieceVelIsaac.z;
            _obs[o++] = eeToPiece.x;
            _obs[o++] = eeToPiece.y;
            _obs[o++] = eeToPiece.z;
            _obs[o++] = pieceToBucket.x;
            _obs[o++] = pieceToBucket.y;
            _obs[o++] = pieceToBucket.z;
            _obs[o++] = _grasped ? 1f : 0f;
            _obs[o++] = lifted;
        }

        void ReadArmState()
        {
            for (int i = 0; i < 7; i++)
            {
                _armVel[i] = (_armPos[i] - _prevArm[i]) / controlDt * dofVelocityScale;
                _prevArm[i] = _armPos[i];
            }
            _gripVel = (_gripPos - _prevGrip) / controlDt * dofVelocityScale;
            _prevGrip = _gripPos;
        }

        Vector3 EePosIsaac()
        {
            if (eeLink == null || envAnchor == null)
                return Vector3.zero;
            return WorldToIsaacLocal(eeLink.position);
        }

        Vector3 WorldToIsaacLocal(Vector3 world)
        {
            Vector3 unityLocal = envAnchor.InverseTransformPoint(world);
            return XrFrameConverter.UnityPosToIsaac(unityLocal);
        }

        void ApplyActions(float dt)
        {
            for (int i = 0; i < 7; i++)
                _armPos[i] += _actions[i] * actionScale * dt;
            _gripPos += _actions[7] * actionScale * dt;
            _gripPos = Mathf.Clamp(_gripPos, gripperClose, gripperOpen);

            var angles = new float[8];
            for (int i = 0; i < 7; i++)
                angles[i] = _armPos[i];
            angles[7] = _gripPos;
            jointDriver?.ApplyAnglesRadians(angles);
        }

        void UpdateGraspMagnet()
        {
            var ee = EePosIsaac();
            float dist = Vector3.Distance(ee, _pieceIsaac);
            bool closing = _gripPos < 0.55f * (gripperOpen + gripperClose);
            _grasped = closing && dist < graspDist;
            if (_grasped)
            {
                _pieceIsaac = ee;
                _pieceVelIsaac = Vector3.zero;
            }
        }

        void SyncVisuals()
        {
            if (piece == null)
                return;
            piece.localPosition = XrFrameConverter.IsaacPosToUnity(_pieceIsaac);
        }

        void PauseBridgeFollowers(bool pause)
        {
            if (_followersPaused == pause)
                return;
            _followersPaused = pause;
            var followers = FindObjectsByType<ROS.AgibotLinkPoseFollower>(FindObjectsSortMode.None);
            foreach (var f in followers)
            {
                if (f != null && f.gameObject.transform.IsChildOf(transform.root))
                    f.followingEnabled = !pause;
            }
            var urFollowers = FindObjectsByType<ROS.RobotLinkPoseFollower>(FindObjectsSortMode.None);
            foreach (var f in urFollowers)
            {
                if (f != null && f.gameObject.transform.IsChildOf(transform.root))
                    f.followingEnabled = !pause;
            }
        }
    }
}
