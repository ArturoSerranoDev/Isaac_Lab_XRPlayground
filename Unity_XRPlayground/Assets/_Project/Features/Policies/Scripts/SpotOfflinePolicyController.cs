using UnityEngine;
using UnityEngine.XR;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Offline Spot follow: optional SpotFollow ONNX (9→3) or P-track HMD → (vx,vy,ωz),
    /// then SpotLoco ONNX (48→12) joint targets. Frames via <see cref="XrFrameConverter"/> only.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SpotOfflinePolicyController : MonoBehaviour, IPolicyMetadataConsumer
    {
        public const int LocoObsDim = 48;
        public const int LocoActionDim = 12;
        public const int FollowObsDim = 9;
        public const int FollowActionDim = 3;
        const string LocoTaskId = "Template-Xrplayground-Spot-Loco-Walk-v0";
        const string FollowTaskId = "Template-Xrplayground-Spot-Follow-v0";

        [Header("Policies")]
        public OnnxPolicyRunner locoRunner;
        public OnnxPolicyRunner followRunner;
        public bool running;

        [Header("Scene refs")]
        public Transform envAnchor;
        public SpotLinkMap linkMap;
        public OfflineJointDriver jointDriver;
        [Tooltip("Optional; otherwise XR center-eye / Camera.main.")]
        public Transform playerTarget;

        [Header("Training-matched params")]
        public float locoActionScale = 0.2f;
        public float locoControlDt = 0.02f;
        public float followControlDt = 0.2f;
        public float followGainLin = 1.2f;
        public float followGainYaw = 1.5f;
        public float maxLinVel = 1.5f;
        public float maxAngVel = 1.2f;
        public float followStandoff = 1.2f;

        /// <summary>Isaac SPOT_CFG default joint pose (12).</summary>
        public static readonly float[] DefaultJointPos =
        {
            0.1f, -0.1f, 0.1f, -0.1f,
            0.9f, 0.9f, 1.1f, 1.1f,
            -1.5f, -1.5f, -1.5f, -1.5f,
        };

        readonly float[] _locoObs = new float[LocoObsDim];
        readonly float[] _locoAct = new float[LocoActionDim];
        readonly float[] _followObs = new float[FollowObsDim];
        readonly float[] _followAct = new float[FollowActionDim];
        readonly float[] _joints = new float[LocoActionDim];
        readonly float[] _jointVel = new float[LocoActionDim];
        readonly float[] _lastLocoAct = new float[LocoActionDim];
        readonly float[] _velCmd = new float[3];

        Vector3 _baseLinVel;
        Vector3 _baseAngVel;
        Vector3 _bodyIsaacPos = new Vector3(0f, 0f, 0.5f);
        Quaternion _bodyIsaacRot = Quaternion.identity;
        float _locoAccum;
        float _followAccum;
        bool _followersPaused;

        public string StatusLine { get; private set; } = "idle";

        void Awake()
        {
            if (envAnchor == null)
                envAnchor = transform;
            if (locoRunner == null)
                locoRunner = GetComponent<OnnxPolicyRunner>();
            if (jointDriver == null)
                jointDriver = GetComponent<OfflineJointDriver>();
            AutoBindLinks();
        }

        [ContextMenu("Auto-Bind Links")]
        public void AutoBindLinks()
        {
            if (linkMap != null)
                linkMap.Rebuild();
            if (jointDriver != null)
                jointDriver.BindSpot(linkMap, envAnchor, force: false);
        }

        public void StartPolicy()
        {
            if (locoRunner == null)
            {
                StatusLine = "missing loco OnnxPolicyRunner";
                return;
            }

            locoRunner.expectedTaskId = LocoTaskId;
            string locoContractError = "ONNX load failed";
            if (!locoRunner.TryLoad() || !locoRunner.MatchesContract(LocoTaskId, LocoObsDim, LocoActionDim, out locoContractError))
            {
                StatusLine = "loco policy contract mismatch";
                Debug.LogError($"SpotOffline: {locoContractError}", this);
                running = false;
                return;
            }

            if (followRunner != null)
            {
                followRunner.expectedTaskId = FollowTaskId;
                string followContractError = "ONNX load failed";
                if (!followRunner.TryLoad() || !followRunner.MatchesContract(FollowTaskId, FollowObsDim, FollowActionDim, out followContractError))
                    Debug.LogWarning($"SpotOffline: SpotFollow unavailable ({followContractError}) — using HMD P-track for vel cmds.", this);
            }

            if (jointDriver != null && !jointDriver.IsBound)
                jointDriver.BindSpot(linkMap, envAnchor, force: true);

            PauseFollowers(true);
            ResetEpisode();
            running = true;
            StatusLine = followRunner != null && followRunner.modelAsset != null
                ? "running (follow+loco)"
                : "running (P-track+loco)";
        }

        public void StopPolicy()
        {
            running = false;
            PauseFollowers(false);
            StatusLine = "stopped";
        }

        public void ApplyPolicyMetadata(PolicyMetadata metadata)
        {
            if (metadata.task_id == LocoTaskId)
            {
                if (metadata.action_scale > 0f)
                    locoActionScale = metadata.action_scale;
                if (metadata.dt > 0f)
                    locoControlDt = metadata.dt;
            }
            else if (metadata.task_id == FollowTaskId && metadata.dt > 0f)
            {
                followControlDt = metadata.dt;
            }
        }

        void PauseFollowers(bool pause)
        {
            _followersPaused = pause;
            var pose = GetComponent<SpotLinkPoseFollower>();
            if (pose != null)
                pose.followingEnabled = !pause;
        }

        void ResetEpisode()
        {
            System.Array.Copy(DefaultJointPos, _joints, LocoActionDim);
            System.Array.Clear(_jointVel, 0, LocoActionDim);
            System.Array.Clear(_lastLocoAct, 0, LocoActionDim);
            System.Array.Clear(_velCmd, 0, 3);
            _baseLinVel = Vector3.zero;
            _baseAngVel = Vector3.zero;
            _bodyIsaacPos = new Vector3(0f, 0f, 0.5f);
            _bodyIsaacRot = Quaternion.identity;
            _locoAccum = 0f;
            _followAccum = 0f;
            jointDriver?.SetRootPoseIsaac(_bodyIsaacPos, _bodyIsaacRot);
            jointDriver?.ApplyAnglesRadians(_joints);
            jointDriver?.Flush();
        }

        void Update()
        {
            if (!running)
                return;

            _locoAccum += Time.deltaTime;
            _followAccum += Time.deltaTime;

            while (_followAccum >= followControlDt)
            {
                _followAccum -= followControlDt;
                UpdateVelocityCommand();
            }

            while (_locoAccum >= locoControlDt)
            {
                _locoAccum -= locoControlDt;
                StepLoco();
            }
        }

        void UpdateVelocityCommand()
        {
            if (followRunner != null && followRunner.modelAsset != null && BuildFollowObs())
            {
                if (followRunner.TryInfer(_followObs, _followAct))
                {
                    _velCmd[0] = Mathf.Clamp(_followAct[0], -maxLinVel, maxLinVel);
                    _velCmd[1] = Mathf.Clamp(_followAct[1], -maxLinVel, maxLinVel);
                    _velCmd[2] = Mathf.Clamp(_followAct[2], -maxAngVel, maxAngVel);
                    return;
                }
            }

            // Heuristic P-track: HMD pose → body-relative (dx,dy,dyaw) → vel cmd
            if (!TryGetPlayerIsaac(out Vector3 playerPos, out Quaternion playerRot))
            {
                _velCmd[0] = _velCmd[1] = _velCmd[2] = 0f;
                return;
            }

            Vector3 toPlayer = playerPos - _bodyIsaacPos;
            toPlayer.z = 0f;
            float yaw = BodyYaw();
            float c = Mathf.Cos(-yaw);
            float s = Mathf.Sin(-yaw);
            float dx = c * toPlayer.x - s * toPlayer.y;
            float dy = s * toPlayer.x + c * toPlayer.y;
            float dist = Mathf.Sqrt(dx * dx + dy * dy);
            float ahead = Mathf.Max(dist - followStandoff, 0f);
            float targetYaw = Mathf.Atan2(toPlayer.y, toPlayer.x);
            float dyaw = Mathf.DeltaAngle(yaw * Mathf.Rad2Deg, targetYaw * Mathf.Rad2Deg) * Mathf.Deg2Rad;

            _velCmd[0] = Mathf.Clamp(followGainLin * ahead * (dx / Mathf.Max(dist, 1e-3f)), -maxLinVel, maxLinVel);
            _velCmd[1] = Mathf.Clamp(followGainLin * 0.5f * dy, -maxLinVel, maxLinVel);
            _velCmd[2] = Mathf.Clamp(followGainYaw * dyaw, -maxAngVel, maxAngVel);
            _ = playerRot;
        }

        void StepLoco()
        {
            BuildLocoObs();
            if (!locoRunner.TryInfer(_locoObs, _locoAct))
            {
                StatusLine = "loco infer failed";
                return;
            }

            for (int i = 0; i < LocoActionDim; i++)
            {
                float target = DefaultJointPos[i] + Mathf.Clamp(_locoAct[i], -1f, 1f) * locoActionScale;
                _jointVel[i] = (target - _joints[i]) / locoControlDt;
                _joints[i] = target;
                _lastLocoAct[i] = _locoAct[i];
            }

            // Integrate toy base motion from vel cmd (visual follow feedback)
            float yaw = BodyYaw();
            float c = Mathf.Cos(yaw);
            float s = Mathf.Sin(yaw);
            Vector3 worldVel = new Vector3(
                c * _velCmd[0] - s * _velCmd[1],
                s * _velCmd[0] + c * _velCmd[1],
                0f);
            _bodyIsaacPos += worldVel * locoControlDt;
            _bodyIsaacPos.z = 0.5f;
            yaw += _velCmd[2] * locoControlDt;
            float half = yaw * 0.5f;
            _bodyIsaacRot = new Quaternion(0f, 0f, Mathf.Sin(half), Mathf.Cos(half));
            _baseLinVel = new Vector3(_velCmd[0], _velCmd[1], 0f);
            _baseAngVel = new Vector3(0f, 0f, _velCmd[2]);

            jointDriver?.SetRootPoseIsaac(_bodyIsaacPos, _bodyIsaacRot);
            jointDriver?.ApplyAnglesRadians(_joints);
            jointDriver?.Flush();
            StatusLine = $"vx={_velCmd[0]:F2} vy={_velCmd[1]:F2} wz={_velCmd[2]:F2}";
        }

        bool BuildFollowObs()
        {
            if (!TryGetPlayerIsaac(out Vector3 playerPos, out _))
                return false;

            Vector3 toPlayer = playerPos - _bodyIsaacPos;
            float yaw = BodyYaw();
            float c = Mathf.Cos(-yaw);
            float s = Mathf.Sin(-yaw);
            float dx = c * toPlayer.x - s * toPlayer.y;
            float dy = s * toPlayer.x + c * toPlayer.y;
            float targetYaw = Mathf.Atan2(toPlayer.y, toPlayer.x);
            float dyaw = Mathf.DeltaAngle(yaw * Mathf.Rad2Deg, targetYaw * Mathf.Rad2Deg) * Mathf.Deg2Rad;

            _followObs[0] = _baseLinVel.x;
            _followObs[1] = _baseLinVel.y;
            _followObs[2] = _baseLinVel.z;
            // projected gravity in body frame (level → (0,0,-1))
            _followObs[3] = 0f;
            _followObs[4] = 0f;
            _followObs[5] = -1f;
            _followObs[6] = dx;
            _followObs[7] = dy;
            _followObs[8] = dyaw;
            return true;
        }

        void BuildLocoObs()
        {
            int o = 0;
            _locoObs[o++] = _baseLinVel.x;
            _locoObs[o++] = _baseLinVel.y;
            _locoObs[o++] = _baseLinVel.z;
            _locoObs[o++] = _baseAngVel.x;
            _locoObs[o++] = _baseAngVel.y;
            _locoObs[o++] = _baseAngVel.z;
            _locoObs[o++] = 0f;
            _locoObs[o++] = 0f;
            _locoObs[o++] = -1f;
            _locoObs[o++] = _velCmd[0];
            _locoObs[o++] = _velCmd[1];
            _locoObs[o++] = _velCmd[2];
            for (int i = 0; i < LocoActionDim; i++)
                _locoObs[o++] = _joints[i] - DefaultJointPos[i];
            for (int i = 0; i < LocoActionDim; i++)
                _locoObs[o++] = _jointVel[i];
            for (int i = 0; i < LocoActionDim; i++)
                _locoObs[o++] = _lastLocoAct[i];
        }

        float BodyYaw()
        {
            // Isaac Z-up yaw from body quat (xyzw)
            var q = _bodyIsaacRot;
            return Mathf.Atan2(2f * (q.w * q.z + q.x * q.y), 1f - 2f * (q.y * q.y + q.z * q.z));
        }

        bool TryGetPlayerIsaac(out Vector3 pos, out Quaternion rot)
        {
            if (playerTarget != null)
            {
                XrFrameConverter.UnityWorldToIsaacLocal(
                    envAnchor, playerTarget.position, playerTarget.rotation, out pos, out rot);
                return true;
            }

            var devices = new System.Collections.Generic.List<InputDevice>();
            InputDevices.GetDevicesAtXRNode(XRNode.CenterEye, devices);
            if (devices.Count > 0)
            {
                var d = devices[0];
                if (d.TryGetFeatureValue(CommonUsages.centerEyePosition, out Vector3 p) &&
                    d.TryGetFeatureValue(CommonUsages.centerEyeRotation, out Quaternion r))
                {
                    XrFrameConverter.UnityWorldToIsaacLocal(envAnchor, p, r, out pos, out rot);
                    return true;
                }
            }

            if (Camera.main != null)
            {
                XrFrameConverter.UnityWorldToIsaacLocal(
                    envAnchor, Camera.main.transform.position, Camera.main.transform.rotation, out pos, out rot);
                return true;
            }

            pos = Vector3.zero;
            rot = Quaternion.identity;
            return false;
        }
    }
}
