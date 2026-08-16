using UnityEngine;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Unity-only Ball Catch loop: 30-D obs → ONNX → 8-D actions. Requires a ball Transform/Rigidbody.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class BallCatchOfflinePolicyController : MonoBehaviour, IPolicyMetadataConsumer
    {
        public const int ObsDim = 30;
        public const int ActionDim = 8;
        const string TaskId = "Template-Xrplayground-Ball-Catch-Direct-v0";

        [Header("Policy")]
        public OnnxPolicyRunner policyRunner;
        public bool running;

        [Header("Scene refs")]
        public Transform envAnchor;
        public KinovaLinkMap linkMap;
        public OfflineJointDriver jointDriver;
        public Transform ball;
        public Rigidbody ballBody;
        public Transform eeLink;
        public Transform tip1;
        public Transform tip2;
        public Transform tip3;

        [Header("Training-matched params")]
        public float actionScale = 5.0f;
        [Tooltip("Matches Isaac robot_dof_speed_scales on gripper joints.")]
        public float gripperSpeedScale = 1.2f;
        [Tooltip("Ball gravity scale (matches Isaac buoyancy so the ball floats into the cup).")]
        public float ballGravityScale = 0.55f;
        public float dofVelocityScale = 0.1f;
        public float controlDt = 1f / 60f;
        public float gripperOpen = 0.04f;
        public float gripperClose = 1.10f;
        [Tooltip("Episode length — matches Isaac episode_length_s.")]
        public float episodeLengthSeconds = 6f;
        [Tooltip("Isaac Z local height below which the ball counts as on the floor.")]
        public float fallHeightThreshold = 0.06f;
        [System.Obsolete("Speed-based toss; kept for serialized scenes.")]
        public float throwFlightSecondsMax = 1.15f;
        [System.Obsolete("Ballistic launch uses aim + flight time; kept for serialized scenes.")]
        public Vector3 throwVelIsaacMin = new Vector3(-0.6f, -0.15f, 0.8f);
        [System.Obsolete("Ballistic launch uses aim + flight time; kept for serialized scenes.")]
        public Vector3 throwVelIsaacMax = new Vector3(-0.3f, 0.15f, 1.6f);

        static readonly string[] ArmLinkNames =
        {
            "j2n7s300_link_1",
            "j2n7s300_link_2",
            "j2n7s300_link_3",
            "j2n7s300_link_4",
            "j2n7s300_link_5",
            "j2n7s300_link_6",
            "j2n7s300_link_7",
        };

        static readonly string[] FingerLinkNames =
        {
            "j2n7s300_link_finger_1",
            "j2n7s300_link_finger_2",
            "j2n7s300_link_finger_3",
        };

        readonly float[] _armPos = new float[7];
        readonly float[] _armVel = new float[7];
        readonly float[] _prevArm = new float[7];
        float _gripPos;
        float _gripVel;
        float _prevGrip;
        readonly float[] _obs = new float[ObsDim];
        readonly float[] _actions = new float[ActionDim];
        float _accum;
        float _episodeTime;
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
            if (ball != null && ballBody == null)
                ballBody = ball.GetComponent<Rigidbody>();
            AutoBindLinks();
        }

        [ContextMenu("Auto-Bind Links")]
        public void AutoBindLinks()
        {
            BindLinks(forceJointBind: false);
        }

        [ContextMenu("Rebuild Kinova FK Bind (arm must be at Isaac default pose)")]
        public void RebuildKinovaBind()
        {
            BindLinks(forceJointBind: true);
            // Snap commanded + visual rest to Isaac defaults after a forced recapture.
            ResetArmStateToIsaacDefault();
            SyncArmVisuals();
            if (jointDriver != null)
            {
                var rest = new float[8];
                for (int i = 0; i < 7; i++)
                    rest[i] = _armPos[i];
                rest[7] = _gripPos;
                jointDriver.RecaptureRest(rest);
                SyncArmVisuals();
            }
            Debug.Log("BallCatchOfflinePolicyController: rebuilt Kinova FK rest at Isaac default pose.", this);
        }

        void BindLinks(bool forceJointBind)
        {
            if (linkMap != null)
            {
                linkMap.Rebuild();
                if (eeLink == null)
                    linkMap.TryGet("j2n7s300_end_effector", out eeLink);
                if (tip1 == null)
                    linkMap.TryGet("j2n7s300_link_finger_tip_1", out tip1);
                if (tip2 == null)
                    linkMap.TryGet("j2n7s300_link_finger_tip_2", out tip2);
                if (tip3 == null)
                    linkMap.TryGet("j2n7s300_link_finger_tip_3", out tip3);
            }

            if (jointDriver != null)
            {
                jointDriver.envAnchor = envAnchor != null ? envAnchor : transform;
                // Do not re-capture rest every StartPolicy — that freezes FK at a mid-motion pose.
                jointDriver.BindKinova(linkMap, jointDriver.envAnchor, force: forceJointBind);
            }
        }

        void ResetArmStateToIsaacDefault()
        {
            _armPos[0] = 0f;
            _armPos[1] = 2.35f;
            _armPos[2] = 0.25f;
            _armPos[3] = 1.65f;
            _armPos[4] = 1.40f;
            _armPos[5] = 0.35f;
            _armPos[6] = 0f;
            for (int i = 0; i < 7; i++)
            {
                _armVel[i] = 0f;
                _prevArm[i] = _armPos[i];
            }
            _gripPos = gripperOpen;
            _gripVel = 0f;
            _prevGrip = _gripPos;
        }

        public void StartPolicy()
        {
            if (policyRunner == null || policyRunner.modelAsset == null)
            {
                Debug.LogError("BallCatchOfflinePolicyController: assign OnnxPolicyRunner.modelAsset (policy.onnx).", this);
                StatusLine = "missing model";
                return;
            }
            if (ball == null)
            {
                Debug.LogError("BallCatchOfflinePolicyController: assign the ball Transform.", this);
                StatusLine = "missing ball";
                return;
            }

            policyRunner.expectedTaskId = TaskId;
            string contractError = "ONNX load failed";
            if (!policyRunner.TryLoad() || !policyRunner.MatchesContract(TaskId, ObsDim, ActionDim, out contractError))
            {
                StatusLine = "policy contract mismatch";
                Debug.LogError($"BallCatchOfflinePolicyController: {contractError}", this);
                return;
            }

            PauseBridgeFollowers(true);
            BindLinks(forceJointBind: false);
            if (jointDriver != null && !jointDriver.IsBound)
                BindLinks(forceJointBind: true);
            ResetEpisodeAndThrow();
            running = true;
            StatusLine = "running";
            Debug.Log("BallCatchOfflinePolicyController: started (Unity-only ONNX, 30-D obs).", this);
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

        public void ApplyPolicyMetadata(PolicyMetadata metadata)
        {
            if (metadata.action_scale > 0f)
                actionScale = metadata.action_scale;
            if (metadata.dt > 0f)
                controlDt = metadata.dt;
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

        void FixedUpdate()
        {
            // Reduced gravity so the ball floats into the cup (matches Isaac buoyancy).
            if (running && ballBody != null && !ballBody.isKinematic)
            {
                float lift = (1f - ballGravityScale) * -Physics.gravity.y * ballBody.mass;
                ballBody.AddForce(Vector3.up * lift, ForceMode.Force);
            }
        }

        void Step(float dt)
        {
            _episodeTime += dt;
            // Same reset rule as Isaac: floor drop OR episode timeout — never on near-grip.
            Vector3 ballIsaac = BallIsaacLocal();
            bool onFloor = ballIsaac.z < fallHeightThreshold;
            bool timedOut = _episodeTime >= episodeLengthSeconds;
            if (onFloor || timedOut)
            {
                StatusLine = onFloor ? "floor — reset" : "timeout — reset";
                ResetEpisodeAndThrow();
                return;
            }

            BuildObs();
            if (!policyRunner.TryInfer(_obs, _actions))
            {
                StatusLine = "infer failed";
                StopPolicy();
                return;
            }
            ApplyActions(dt);
            SyncArmVisuals();
            StatusLine = $"running t={_episodeTime:F1}s";
        }

        void ResetEpisodeAndThrow()
        {
            _episodeTime = 0f;
            ResetArmStateToIsaacDefault();
            // Pose the visual arm first so tip/EE aim uses the Isaac default cup, not a stale mesh.
            SyncArmVisuals();
            jointDriver?.Flush();

            Vector3 tipI = Vector3.zero;
            int tipCount = 0;
            if (tip1 != null) { tipI += XrFrameConverter.UnityPosToIsaac(envAnchor.InverseTransformPoint(tip1.position)); tipCount++; }
            if (tip2 != null) { tipI += XrFrameConverter.UnityPosToIsaac(envAnchor.InverseTransformPoint(tip2.position)); tipCount++; }
            if (tip3 != null) { tipI += XrFrameConverter.UnityPosToIsaac(envAnchor.InverseTransformPoint(tip3.position)); tipCount++; }
            if (tipCount > 0) tipI /= tipCount;
            Vector3 eeI = eeLink != null
                ? XrFrameConverter.UnityPosToIsaac(envAnchor.InverseTransformPoint(eeLink.position))
                : tipI;
            Vector3 cupAim = 0.55f * tipI + 0.45f * eeI;

            // Match Isaac reachable curriculum (easy→mid): always aimed at cup.
            Vector3 releaseI = cupAim + new Vector3(
                Random.Range(0.12f, 0.30f),
                Random.Range(-0.12f, 0.12f),
                -Random.Range(0.02f, 0.08f));
            Vector3 targetI = cupAim + new Vector3(
                Random.Range(-0.03f, 0.03f),
                Random.Range(-0.03f, 0.03f),
                Random.Range(-0.025f, 0.025f));
            Vector3 midI = 0.5f * (releaseI + targetI);
            midI.z += Random.Range(0.05f, 0.12f);

            float flightT = Random.Range(0.55f, 0.80f);
            Vector3 dI = targetI - releaseI;
            float gEff = 9.81f * ballGravityScale;
            Vector3 gI = new Vector3(0f, 0f, -gEff);
            Vector3 velI = dI / flightT - 0.5f * gI * flightT;
            float linearZ = releaseI.z + dI.z * 0.5f;
            velI.z += 2f * (midI.z - linearZ) / flightT;

            Vector3 posU = envAnchor.TransformPoint(XrFrameConverter.IsaacPosToUnity(releaseI));
            Vector3 velU = envAnchor.TransformDirection(XrFrameConverter.IsaacPosToUnity(velI));

            ball.position = posU;
            if (ballBody != null)
            {
                ballBody.isKinematic = false;
                ballBody.linearVelocity = velU;
                ballBody.angularVelocity = Vector3.zero;
            }
        }

        void ApplyActions(float dt)
        {
            for (int i = 0; i < 7; i++)
            {
                _prevArm[i] = _armPos[i];
                _armPos[i] += _actions[i] * actionScale * dt;
                _armPos[i] = Mathf.Clamp(_armPos[i], -Mathf.PI * 2f, Mathf.PI * 2f);
                _armVel[i] = (_armPos[i] - _prevArm[i]) / Mathf.Max(dt, 1e-5f);
            }
            _prevGrip = _gripPos;
            _gripPos += _actions[7] * actionScale * gripperSpeedScale * dt;
            _gripPos = Mathf.Clamp(_gripPos, gripperOpen, gripperClose);
            _gripVel = (_gripPos - _prevGrip) / Mathf.Max(dt, 1e-5f);
        }

        void BuildObs()
        {
            int o = 0;
            for (int i = 0; i < 7; i++)
                _obs[o++] = _armPos[i];
            for (int i = 0; i < 7; i++)
                _obs[o++] = _armVel[i] * dofVelocityScale;
            _obs[o++] = _gripPos;
            _obs[o++] = _gripVel * dofVelocityScale;

            Vector3 ballIsaac = BallIsaacLocal();
            Vector3 ballVelIsaac = BallVelIsaac();
            Vector3 ee = WorldToIsaacLocal(eeLink != null ? eeLink.position : envAnchor.position);
            Vector3 tip = TipCenterIsaac();

            _obs[o++] = ballIsaac.x;
            _obs[o++] = ballIsaac.y;
            _obs[o++] = ballIsaac.z;
            _obs[o++] = ballVelIsaac.x;
            _obs[o++] = ballVelIsaac.y;
            _obs[o++] = ballVelIsaac.z;

            Vector3 toBall = ballIsaac - ee;
            _obs[o++] = toBall.x;
            _obs[o++] = toBall.y;
            _obs[o++] = toBall.z;

            Vector3 toBallFingers = ballIsaac - tip;
            _obs[o++] = toBallFingers.x;
            _obs[o++] = toBallFingers.y;
            _obs[o++] = toBallFingers.z;

            // Grasp geometry (matches Isaac): align + radial off-axis error.
            Vector3 graspAxis = tip - ee;
            float axisLen = Mathf.Max(graspAxis.magnitude, 1e-6f);
            graspAxis /= axisLen;
            float distEe = Mathf.Max(toBall.magnitude, 1e-6f);
            float graspAlign = Vector3.Dot(toBall / distEe, graspAxis);
            float ballAlong = Vector3.Dot(toBall, graspAxis);
            float ballRadial = (toBall - graspAxis * ballAlong).magnitude;
            _obs[o++] = graspAlign;
            _obs[o++] = ballRadial;
            // 7+7+1+1+3+3+3+3+1+1 = 30
        }

        Vector3 BallIsaacLocal()
        {
            if (ball == null || envAnchor == null)
                return Vector3.zero;
            return WorldToIsaacLocal(ball.position);
        }

        Vector3 BallVelIsaac()
        {
            if (ballBody == null || envAnchor == null)
                return Vector3.zero;
            Vector3 localUnity = envAnchor.InverseTransformDirection(ballBody.linearVelocity);
            return XrFrameConverter.UnityPosToIsaac(localUnity);
        }

        Vector3 TipCenterIsaac()
        {
            int n = 0;
            Vector3 sum = Vector3.zero;
            if (tip1 != null) { sum += tip1.position; n++; }
            if (tip2 != null) { sum += tip2.position; n++; }
            if (tip3 != null) { sum += tip3.position; n++; }
            if (n == 0)
                return WorldToIsaacLocal(eeLink != null ? eeLink.position : envAnchor.position);
            return WorldToIsaacLocal(sum / n);
        }

        Vector3 WorldToIsaacLocal(Vector3 world)
        {
            Vector3 unityLocal = envAnchor.InverseTransformPoint(world);
            return XrFrameConverter.UnityPosToIsaac(unityLocal);
        }

        void SyncArmVisuals()
        {
            if (jointDriver == null)
                return;
            // 8 absolute joint targets (7 arm + shared gripper) — matches BindKinova joint indices.
            var angles = new float[8];
            for (int i = 0; i < 7; i++)
                angles[i] = _armPos[i];
            angles[7] = _gripPos;
            jointDriver.ApplyAnglesRadians(angles);
        }

        void PauseBridgeFollowers(bool pause)
        {
            if (_followersPaused == pause)
                return;
            _followersPaused = pause;
            var robotFollower = GetComponentInChildren<KinovaLinkPoseFollower>(true);
            if (robotFollower != null)
                robotFollower.enabled = !pause;
            var ballFollower = GetComponentInChildren<BallPoseFollower>(true);
            if (ballFollower != null)
                ballFollower.followingEnabled = !pause;
        }
    }
}
