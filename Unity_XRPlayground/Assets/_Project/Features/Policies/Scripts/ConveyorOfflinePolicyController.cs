using UnityEngine;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.Policies
{
    /// <summary>
    /// Unity-only Conveyor Color loop: rebuilds 66-D obs, runs ONNX, applies 7-D actions.
    /// Assign cube slots + ModelAsset; press Start from the UI button.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class ConveyorOfflinePolicyController : MonoBehaviour, IPolicyMetadataConsumer
    {
        public const int ObsDim = 66;
        public const int ActionDim = 7;
        const string TaskId = "Template-Xrplayground-Conveyor-Color-Direct-v0";

        [Header("Policy")]
        public OnnxPolicyRunner policyRunner;
        public bool running;

        [Header("Scene refs")]
        public Transform envAnchor;
        public RobotLinkMap linkMap;
        public OfflineJointDriver jointDriver;
        [Tooltip("Four reusable cube transforms (same as ConveyorObjectFollower slots).")]
        public Transform[] cubeSlots = new Transform[4];
        public Transform eeLink;
        public Transform binAnchor;

        [Header("Training-matched params (Isaac Z-up env frame)")]
        public float actionScale = 4f;
        public float dofVelocityScale = 0.1f;
        public float controlDt = 1f / 60f;
        public float beltSpeed = 0.22f;
        public float spawnIntervalS = 2.5f;
        public int targetColor;
        public Vector3 beltPosIsaac = new Vector3(0.55f, 0f, 0.40f);
        public float beltHalfX = 0.11f;
        public float beltYMin = -0.62f;
        public float beltYMax = 0.62f;
        public float spawnYMin = -0.60f;
        public float spawnYMax = -0.40f;
        public float cubeHalfSize = 0.025f;
        public Vector3 binPosIsaac = new Vector3(0.20f, 0.78f, 0.405f);
        public float gripperOpen;
        public float gripperClose = 0.785f;

        static readonly string[] ArmLinkNames =
        {
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        };

        static readonly string[] GripperLinkNames = { "left_inner_finger", "right_inner_finger" };

        readonly float[] _armPos = new float[6];
        readonly float[] _armVel = new float[6];
        readonly float[] _prevArm = new float[6];
        float _gripPos;
        float _gripVel;
        float _prevGrip;
        readonly bool[] _active = new bool[4];
        readonly int[] _color = new int[4];
        readonly Vector3[] _cubeIsaac = new Vector3[4];
        readonly Vector3[] _cubeVelIsaac = new Vector3[4];
        readonly float[] _obs = new float[ObsDim];
        readonly float[] _actions = new float[ActionDim];
        float _spawnTimer;
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
            {
                linkMap.Rebuild();
                if (eeLink == null)
                    linkMap.TryGet("wrist_3_link", out eeLink);
            }

            if (jointDriver != null)
            {
                jointDriver.envAnchor = envAnchor != null ? envAnchor : transform;
                jointDriver.BindUr10e(linkMap, jointDriver.envAnchor);
            }
        }

        /// <summary>UI entry: start offline policy using assigned cubes / target color.</summary>
        public void StartPolicy()
        {
            if (policyRunner == null || policyRunner.modelAsset == null)
            {
                Debug.LogError("ConveyorOfflinePolicyController: assign OnnxPolicyRunner.modelAsset (policy.onnx).", this);
                StatusLine = "missing model";
                return;
            }

            policyRunner.expectedTaskId = TaskId;
            string contractError = "ONNX load failed";
            if (!policyRunner.TryLoad() || !policyRunner.MatchesContract(TaskId, ObsDim, ActionDim, out contractError))
            {
                StatusLine = "policy contract mismatch";
                Debug.LogError($"ConveyorOfflinePolicyController: {contractError}", this);
                return;
            }

            PauseBridgeFollowers(true);
            AutoBindLinks();
            ResetEpisode();
            running = true;
            StatusLine = "running";
            Debug.Log("ConveyorOfflinePolicyController: started (Unity-only ONNX).", this);
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

        void Step(float dt)
        {
            MaybeSpawn(dt);
            ApplyBelt(dt);
            BuildObs();
            if (!policyRunner.TryInfer(_obs, _actions))
            {
                StatusLine = "infer failed";
                StopPolicy();
                return;
            }
            ApplyActions(dt);
            SyncCubeVisuals();
            SyncArmVisuals();
            StatusLine = $"run t={_spawnTimer:0.0}s target={ColorName(targetColor)}";
        }

        void ResetEpisode()
        {
            _armPos[0] = Mathf.PI;
            _armPos[1] = -Mathf.PI * 0.5f;
            _armPos[2] = Mathf.PI * 0.5f;
            _armPos[3] = -Mathf.PI * 0.5f;
            _armPos[4] = -Mathf.PI * 0.5f;
            _armPos[5] = 0f;
            for (int i = 0; i < 6; i++)
            {
                _armVel[i] = 0f;
                _prevArm[i] = _armPos[i];
            }
            _gripPos = gripperOpen;
            _gripVel = 0f;
            _prevGrip = _gripPos;
            targetColor = Random.Range(0, 3);
            _spawnTimer = 0f;
            for (int i = 0; i < 4; i++)
            {
                _active[i] = false;
                _color[i] = 0;
                _cubeIsaac[i] = new Vector3(0f, 0f, -0.5f);
                _cubeVelIsaac[i] = Vector3.zero;
                if (cubeSlots != null && i < cubeSlots.Length && cubeSlots[i] != null)
                    cubeSlots[i].gameObject.SetActive(false);
            }
            SyncArmVisuals();
        }

        void MaybeSpawn(float dt)
        {
            _spawnTimer += dt;
            if (_spawnTimer < spawnIntervalS)
                return;
            int free = -1;
            for (int i = 0; i < 4; i++)
            {
                if (!_active[i])
                {
                    free = i;
                    break;
                }
            }
            if (free < 0)
                return;
            _spawnTimer = 0f;
            ActivateSlot(free, Random.Range(0, 3));
        }

        public void SpawnColor(int color)
        {
            for (int i = 0; i < 4; i++)
            {
                if (_active[i])
                    continue;
                ActivateSlot(i, Mathf.Clamp(color, 0, 2));
                return;
            }
        }

        void ActivateSlot(int slot, int color)
        {
            float x = beltPosIsaac.x + Random.Range(-beltHalfX, beltHalfX);
            float y = Random.Range(spawnYMin, spawnYMax);
            float z = beltPosIsaac.z + 0.02f + cubeHalfSize;
            _active[slot] = true;
            _color[slot] = color;
            _cubeIsaac[slot] = new Vector3(x, y, z);
            _cubeVelIsaac[slot] = new Vector3(0f, beltSpeed, 0f);
            if (cubeSlots != null && slot < cubeSlots.Length && cubeSlots[slot] != null)
            {
                cubeSlots[slot].gameObject.SetActive(true);
                ApplyCubeColor(cubeSlots[slot], color);
            }
        }

        void ApplyBelt(float dt)
        {
            for (int i = 0; i < 4; i++)
            {
                if (!_active[i])
                    continue;
                var p = _cubeIsaac[i];
                p.y += beltSpeed * dt;
                p.x = Mathf.Clamp(p.x, beltPosIsaac.x - beltHalfX, beltPosIsaac.x + beltHalfX);
                if (p.y > beltYMax + 0.05f)
                {
                    _active[i] = false;
                    p = new Vector3(0f, 0f, -0.5f);
                    _cubeVelIsaac[i] = Vector3.zero;
                    if (cubeSlots != null && i < cubeSlots.Length && cubeSlots[i] != null)
                        cubeSlots[i].gameObject.SetActive(false);
                }
                else
                    _cubeVelIsaac[i] = new Vector3(0f, beltSpeed, 0f);
                _cubeIsaac[i] = p;
            }
        }

        void ApplyActions(float dt)
        {
            for (int i = 0; i < 6; i++)
            {
                _prevArm[i] = _armPos[i];
                _armPos[i] += _actions[i] * actionScale * dt;
                _armPos[i] = Mathf.Clamp(_armPos[i], -Mathf.PI * 2f, Mathf.PI * 2f);
                _armVel[i] = (_armPos[i] - _prevArm[i]) / Mathf.Max(dt, 1e-5f);
            }
            _prevGrip = _gripPos;
            float g = 0.5f * (_actions[6] + 1f);
            _gripPos = gripperOpen + g * (gripperClose - gripperOpen);
            _gripVel = (_gripPos - _prevGrip) / Mathf.Max(dt, 1e-5f);
        }

        void BuildObs()
        {
            int o = 0;
            for (int i = 0; i < 6; i++)
                _obs[o++] = _armPos[i];
            for (int i = 0; i < 6; i++)
                _obs[o++] = _armVel[i] * dofVelocityScale;
            _obs[o++] = _gripPos;
            _obs[o++] = _gripVel * dofVelocityScale;

            // target one-hot (3)
            for (int c = 0; c < 3; c++)
                _obs[o++] = c == targetColor ? 1f : 0f;

            Vector3 ee = EeIsaacLocal();
            for (int slot = 0; slot < 4; slot++)
            {
                float active = _active[slot] ? 1f : 0f;
                Vector3 pos = _active[slot] ? _cubeIsaac[slot] : Vector3.zero;
                Vector3 vel = _active[slot] ? _cubeVelIsaac[slot] : Vector3.zero;
                _obs[o++] = pos.x * active;
                _obs[o++] = pos.y * active;
                _obs[o++] = pos.z * active;
                _obs[o++] = vel.x * active;
                _obs[o++] = vel.y * active;
                _obs[o++] = vel.z * active;
                for (int c = 0; c < 3; c++)
                    _obs[o++] = (c == _color[slot] ? 1f : 0f) * active;
                _obs[o++] = active;
            }

            _obs[o++] = ee.x;
            _obs[o++] = ee.y;
            _obs[o++] = ee.z;

            Vector3 nearestDelta = Vector3.zero;
            float best = 1e6f;
            for (int slot = 0; slot < 4; slot++)
            {
                if (!_active[slot] || _color[slot] != targetColor)
                    continue;
                Vector3 d = _cubeIsaac[slot] - ee;
                float dist = d.magnitude;
                if (dist < best)
                {
                    best = dist;
                    nearestDelta = d;
                }
            }
            _obs[o++] = nearestDelta.x;
            _obs[o++] = nearestDelta.y;
            _obs[o++] = nearestDelta.z;

            Vector3 binDelta = binPosIsaac - ee;
            _obs[o++] = binDelta.x;
            _obs[o++] = binDelta.y;
            _obs[o++] = binDelta.z;

            // 6+6+1+1+3 + 4*(3+3+3+1) + 3+3+3 = 14+3+40+9 = 66
        }

        Vector3 EeIsaacLocal()
        {
            if (eeLink == null || envAnchor == null)
                return Vector3.zero;
            Vector3 unityLocal = envAnchor.InverseTransformPoint(eeLink.position);
            return XrFrameConverter.UnityPosToIsaac(unityLocal);
        }

        void SyncCubeVisuals()
        {
            if (cubeSlots == null || envAnchor == null)
                return;
            for (int i = 0; i < 4 && i < cubeSlots.Length; i++)
            {
                var t = cubeSlots[i];
                if (t == null || !_active[i])
                    continue;
                t.position = envAnchor.TransformPoint(XrFrameConverter.IsaacPosToUnity(_cubeIsaac[i]));
            }
        }

        void SyncArmVisuals()
        {
            if (jointDriver == null)
                return;
            var angles = new float[8];
            for (int i = 0; i < 6; i++)
                angles[i] = _armPos[i];
            angles[6] = _gripPos;
            angles[7] = _gripPos;
            jointDriver.ApplyAnglesRadians(angles);
        }

        void PauseBridgeFollowers(bool pause)
        {
            if (_followersPaused == pause)
                return;
            _followersPaused = pause;
            var robotFollower = GetComponentInChildren<RobotLinkPoseFollower>(true);
            if (robotFollower != null)
                robotFollower.followingEnabled = !pause;
            var objFollower = GetComponentInChildren<ConveyorObjectFollower>(true)
                ?? envAnchor?.GetComponent<ConveyorObjectFollower>();
            if (objFollower != null)
                objFollower.followingEnabled = !pause;
        }

        static void ApplyCubeColor(Transform cube, int color)
        {
            var rend = cube.GetComponentInChildren<MeshRenderer>();
            if (rend == null)
                return;
            Color c = color switch
            {
                1 => new Color(0.15f, 0.75f, 0.25f),
                2 => new Color(0.15f, 0.35f, 0.90f),
                _ => new Color(0.90f, 0.15f, 0.12f),
            };
            if (rend.material != null)
                rend.material.color = c;
        }

        static string ColorName(int c) => c switch { 1 => "G", 2 => "B", _ => "R" };
    }
}
