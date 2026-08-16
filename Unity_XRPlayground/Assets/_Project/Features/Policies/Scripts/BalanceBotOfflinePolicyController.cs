using UnityEngine;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Unity-only Balance Bot loop: rebuilds 20-D obs, runs ONNX, applies 2-D tray tilt.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class BalanceBotOfflinePolicyController : MonoBehaviour, IPolicyMetadataConsumer
    {
        public const int ObsDim = 20;
        public const int ActionDim = 2;
        const string TaskId = "Template-Xrplayground-Balance-Bot-Direct-v0";

        [Header("Policy")]
        public OnnxPolicyRunner policyRunner;
        public bool running;

        [Header("Scene refs")]
        public Transform envAnchor;
        public BalanceBotLinkMap linkMap;
        public OfflineJointDriver jointDriver;
        public Transform[] ballSlots = new Transform[2];

        [Header("Training-matched params (Isaac Z-up env frame)")]
        public float actionScale = 1.5f;
        public float dofVelocityScale = 0.25f;
        public float controlDt = 1f / 60f;
        public float maxTiltRad = 0.40f;
        public Vector3 trayCenterIsaac = new Vector3(0f, 0f, 0.75f);
        public float ballRadius = 0.035f;
        public float spawnHeightAboveTray = 0.08f;
        public float spawnXyHalf = 0.12f;
        public float trayHalfXy = 0.27f;
        public float fallZ = 0.35f;
        public int curriculumStage = 1;
        public int nBalls = 2;

        readonly float[] _obs = new float[ObsDim];
        readonly float[] _actions = new float[ActionDim];
        readonly float[] _joints = new float[2];
        readonly float[] _jointVel = new float[2];
        readonly bool[] _ballActive = new bool[2];
        readonly Vector3[] _ballIsaac = new Vector3[2];
        readonly Vector3[] _ballVelIsaac = new Vector3[2];
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
            AutoBindLinks();
        }

        [ContextMenu("Auto-Bind Links")]
        public void AutoBindLinks()
        {
            if (linkMap != null)
                linkMap.Rebuild();
            if (jointDriver != null)
                jointDriver.BindBalanceBot(linkMap, envAnchor, force: false);
        }

        public void StartPolicy()
        {
            if (policyRunner == null)
            {
                StatusLine = "missing OnnxPolicyRunner";
                return;
            }

            policyRunner.expectedTaskId = TaskId;
            string contractError = "ONNX load failed";
            if (!policyRunner.TryLoad() || !policyRunner.MatchesContract(TaskId, ObsDim, ActionDim, out contractError))
            {
                StatusLine = "policy contract mismatch";
                Debug.LogError($"BalanceBotOfflinePolicyController: {contractError}", this);
                running = false;
                return;
            }

            if (jointDriver != null && !jointDriver.IsBound)
                jointDriver.BindBalanceBot(linkMap, envAnchor, force: true);

            PauseFollowers(true);
            ResetEpisode();
            running = true;
            StatusLine = "running";
        }

        public void StopPolicy()
        {
            running = false;
            PauseFollowers(false);
            StatusLine = "stopped";
        }

        public void ApplyPolicyMetadata(PolicyMetadata metadata)
        {
            if (metadata.action_scale > 0f)
                actionScale = metadata.action_scale;
            if (metadata.dt > 0f)
                controlDt = metadata.dt;
        }

        void PauseFollowers(bool pause)
        {
            _followersPaused = pause;
            var pose = GetComponent<BalanceBotLinkPoseFollower>();
            if (pose != null)
                pose.followingEnabled = !pause;
            var balls = GetComponent<BalanceBotBallFollower>();
            if (balls != null)
                balls.followingEnabled = !pause;
        }

        void ResetEpisode()
        {
            _joints[0] = 0f;
            _joints[1] = 0f;
            _jointVel[0] = 0f;
            _jointVel[1] = 0f;
            nBalls = curriculumStage >= 1 ? 2 : 1;
            SpawnBalls(nBalls);
            jointDriver?.ApplyAnglesRadians(_joints);
            jointDriver?.Flush();
        }

        void SpawnBalls(int count)
        {
            float z = trayCenterIsaac.z + 0.01f + ballRadius + spawnHeightAboveTray;
            Vector2[] offsets = { Vector2.zero, new Vector2(0.09f, 0.06f) };
            for (int i = 0; i < 2; i++)
            {
                _ballActive[i] = i < count;
                if (ballSlots == null || i >= ballSlots.Length || ballSlots[i] == null)
                    continue;

                if (!_ballActive[i])
                {
                    ballSlots[i].gameObject.SetActive(false);
                    continue;
                }

                ballSlots[i].gameObject.SetActive(true);
                var isaac = new Vector3(
                    trayCenterIsaac.x + offsets[i].x,
                    trayCenterIsaac.y + offsets[i].y,
                    z);
                _ballIsaac[i] = isaac;
                _ballVelIsaac[i] = Vector3.zero;
                XrFrameConverter.ApplyIsaacBody(ballSlots[i], envAnchor, isaac, new Quaternion(0f, 0f, 0f, 1f));
            }
        }

        void Update()
        {
            if (!running)
                return;

            _accum += Time.deltaTime;
            while (_accum >= controlDt)
            {
                _accum -= controlDt;
                StepOnce();
            }
        }

        void StepOnce()
        {
            // Integrate simple tray tilt (matches Isaac rate command)
            for (int a = 0; a < ActionDim; a++)
            {
                _jointVel[a] = Mathf.Clamp(_actions[a], -1f, 1f) * actionScale;
                _joints[a] = Mathf.Clamp(_joints[a] + _jointVel[a] * controlDt, -maxTiltRad, maxTiltRad);
            }

            jointDriver?.ApplyAnglesRadians(_joints);
            jointDriver?.Flush();

            // Toy Unity physics for balls: project gravity onto tilted tray plane (visual approx)
            var tilt = Quaternion.Euler(_joints[0] * Mathf.Rad2Deg, 0f, _joints[1] * Mathf.Rad2Deg);
            // Isaac: roll X then pitch Y — approximate with euler for visual roll-off
            Vector3 gravityIsaac = new Vector3(0f, 0f, -9.81f);
            Vector3 trayNormal = tilt * Vector3.forward; // wrong for Isaac Z-up...
            // Isaac tray normal is rotated Z-up:
            trayNormal = RotateIsaac(QuatFromEuler(_joints[0], _joints[1], 0f), new Vector3(0f, 0f, 1f));
            Vector3 gTangent = gravityIsaac - Vector3.Dot(gravityIsaac, trayNormal) * trayNormal;

            bool anyDropped = false;
            for (int i = 0; i < 2; i++)
            {
                if (!_ballActive[i])
                    continue;
                _ballVelIsaac[i] += gTangent * controlDt;
                _ballVelIsaac[i] *= 0.992f;
                _ballIsaac[i] += _ballVelIsaac[i] * controlDt;
                // Keep roughly on tray height along normal (soft constraint)
                float height = trayCenterIsaac.z + 0.01f + ballRadius;
                _ballIsaac[i].z = Mathf.Max(_ballIsaac[i].z, height - 0.02f);

                float dx = Mathf.Abs(_ballIsaac[i].x - trayCenterIsaac.x);
                float dy = Mathf.Abs(_ballIsaac[i].y - trayCenterIsaac.y);
                if (dx > trayHalfXy || dy > trayHalfXy || _ballIsaac[i].z < fallZ)
                {
                    anyDropped = true;
                    _ballActive[i] = false;
                    if (ballSlots != null && i < ballSlots.Length && ballSlots[i] != null)
                        ballSlots[i].gameObject.SetActive(false);
                }
                else if (ballSlots != null && i < ballSlots.Length && ballSlots[i] != null)
                {
                    XrFrameConverter.ApplyIsaacBody(
                        ballSlots[i], envAnchor, _ballIsaac[i], new Quaternion(0f, 0f, 0f, 1f));
                }
            }

            if (anyDropped)
            {
                StatusLine = "dropped — reset";
                ResetEpisode();
            }

            BuildObs();
            if (!policyRunner.TryInfer(_obs, _actions))
            {
                StatusLine = "infer failed";
                return;
            }
            StatusLine = $"roll={_joints[0]:F2} pitch={_joints[1]:F2} n={nBalls}";
        }

        void BuildObs()
        {
            int o = 0;
            _obs[o++] = _joints[0];
            _obs[o++] = _joints[1];
            _obs[o++] = _jointVel[0] * dofVelocityScale;
            _obs[o++] = _jointVel[1] * dofVelocityScale;
            for (int i = 0; i < 2; i++)
            {
                float active = _ballActive[i] ? 1f : 0f;
                Vector3 rel = _ballActive[i] ? (_ballIsaac[i] - trayCenterIsaac) : Vector3.zero;
                Vector3 vel = _ballActive[i] ? _ballVelIsaac[i] : Vector3.zero;
                _obs[o++] = rel.x;
                _obs[o++] = rel.y;
                _obs[o++] = rel.z;
                _obs[o++] = vel.x;
                _obs[o++] = vel.y;
                _obs[o++] = vel.z;
                _obs[o++] = active;
            }
            _obs[o++] = nBalls;
            _obs[o++] = curriculumStage;
        }

        static Quaternion QuatFromEuler(float roll, float pitch, float yaw)
        {
            float cy = Mathf.Cos(yaw * 0.5f);
            float sy = Mathf.Sin(yaw * 0.5f);
            float cp = Mathf.Cos(pitch * 0.5f);
            float sp = Mathf.Sin(pitch * 0.5f);
            float cr = Mathf.Cos(roll * 0.5f);
            float sr = Mathf.Sin(roll * 0.5f);
            // wxyz Isaac order constructed then used as Quaternion(x,y,z,w)
            float w = cr * cp * cy + sr * sp * sy;
            float x = sr * cp * cy - cr * sp * sy;
            float y = cr * sp * cy + sr * cp * sy;
            float z = cr * cp * sy - sr * sp * cy;
            return new Quaternion(x, y, z, w);
        }

        static Vector3 RotateIsaac(Quaternion q, Vector3 v)
        {
            // q is xyzw Unity quaternion representing Isaac rotation
            return q * v;
        }
    }
}
