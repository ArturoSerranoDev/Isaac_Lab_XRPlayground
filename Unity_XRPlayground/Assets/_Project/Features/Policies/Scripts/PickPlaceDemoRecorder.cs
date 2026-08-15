using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using XRPlayground.ROS;
using XRPlayground.Robots;

namespace XRPlayground.Policies
{
    /// <summary>
    /// VR / editor imitation-learning demos for Agibot pick-place.
    /// Records (obs, action) at <see cref="recordRateHz"/>, publishes
    /// <c>/xr/pick_place/demo_record</c>, and appends JSONL under persistentDataPath.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PickPlaceDemoRecorder : MonoBehaviour
    {
        [Tooltip("Matches pick_place_table_env_cfg.imitation_learning_enabled")]
        public bool imitationLearningEnabled;

        [Tooltip("Target publish / sample rate (Hz) when recording.")]
        public float recordRateHz = 30f;

        [Header("Refs")]
        public RosTcpClient client;
        public Transform envAnchor;
        public AgibotLinkMap linkMap;
        public OfflineJointDriver jointDriver;
        public Transform piece;
        public Transform eeLink;
        public PickPlaceOfflinePolicyController offlinePolicy;

        [Header("Teleop while recording")]
        [Tooltip("When the piece is XR-grabbed, nudge arm joints toward EE→piece error.")]
        public bool followGrabbedPiece = true;
        public float teleopGain = 2.5f;
        public float actionScale = 5f;
        public float gripperOpen = 0.994f;
        public float gripperClose = 0.20f;
        public float controlDt = 1f / 60f;
        public Vector3 bucketPosIsaac = new Vector3(0.45f, -0.18f, 0.44f);
        public float liftHeight = 0.46f;
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
        readonly float[] _prevArm = new float[7];
        readonly float[] _armVel = new float[7];
        readonly float[] _obs = new float[PickPlaceOfflinePolicyController.ObsDim];
        readonly float[] _actions = new float[PickPlaceOfflinePolicyController.ActionDim];
        float _gripPos = 0.994f;
        float _prevGrip;
        float _gripVel;
        float _accum;
        float _sampleAccum;
        bool _recording;
        bool _pieceGrabbed;
        int _frameCount;
        string _jsonlPath;
        StreamWriter _writer;
        XRGrabInteractable _grab;
        Vector3 _pieceIsaac;
        Vector3 _pieceVelIsaac;
        Vector3 _prevPieceIsaac;
        bool _hasPrevPiece;

        public string StatusLine { get; private set; } = "IL disabled";
        public int FrameCount => _frameCount;
        public bool IsRecording => _recording;

        void Awake()
        {
            if (envAnchor == null)
                envAnchor = transform;
            if (offlinePolicy == null)
                offlinePolicy = GetComponent<PickPlaceOfflinePolicyController>();
            if (jointDriver == null)
                jointDriver = GetComponent<OfflineJointDriver>();
            BindPieceGrab();
            InitArmFromRest();
        }

        void OnDisable()
        {
            UnbindPieceGrab();
            CloseWriter();
        }

        public void SetRecording(bool on)
        {
            if (!imitationLearningEnabled)
            {
                _recording = false;
                StatusLine = "IL disabled (set imitationLearningEnabled)";
                CloseWriter();
                return;
            }

            if (on == _recording)
            {
                StatusLine = _recording ? $"recording frames={_frameCount}" : "idle";
                return;
            }

            _recording = on;
            if (_recording)
            {
                _frameCount = 0;
                _sampleAccum = 0f;
                _hasPrevPiece = false;
                InitArmFromRest();
                OpenWriter();
                StatusLine = "recording";
                Debug.Log($"PickPlaceDemoRecorder: started → {_jsonlPath}", this);
            }
            else
            {
                CloseWriter();
                StatusLine = $"idle (saved {_frameCount} frames)";
                Debug.Log($"PickPlaceDemoRecorder: stopped ({_frameCount} frames).", this);
            }
        }

        void Update()
        {
            if (!_recording || !imitationLearningEnabled)
                return;
            if (offlinePolicy != null && offlinePolicy.running)
                return;

            float dt = Time.deltaTime;
            _accum += dt;
            while (_accum >= controlDt)
            {
                _accum -= controlDt;
                StepTeleop(controlDt);
            }

            float period = 1f / Mathf.Max(1f, recordRateHz);
            _sampleAccum += dt;
            while (_sampleAccum >= period)
            {
                _sampleAccum -= period;
                SampleAndPublish();
            }
        }

        void StepTeleop(float dt)
        {
            SyncPieceIsaac();
            if (followGrabbedPiece && _pieceGrabbed && eeLink != null && envAnchor != null)
            {
                var ee = WorldToIsaacLocal(eeLink.position);
                var err = _pieceIsaac - ee;
                // Crude right-arm teleop: map EE error into first 3 joints + wrist toward grasp.
                _actions[0] = Mathf.Clamp(err.y * teleopGain, -1f, 1f);
                _actions[1] = Mathf.Clamp(-err.z * teleopGain, -1f, 1f);
                _actions[2] = Mathf.Clamp(err.x * teleopGain, -1f, 1f);
                _actions[3] = Mathf.Clamp(err.z * teleopGain * 0.5f, -1f, 1f);
                _actions[4] = 0f;
                _actions[5] = 0f;
                _actions[6] = 0f;
                float dist = err.magnitude;
                _actions[7] = dist < graspDist ? -1f : 1f;
            }
            else
            {
                for (int i = 0; i < _actions.Length; i++)
                    _actions[i] = 0f;
                _actions[7] = _pieceGrabbed ? -1f : 1f;
            }

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

        void SampleAndPublish()
        {
            BuildObs();
            var packet = new DemoRecordData
            {
                obs = (float[])_obs.Clone(),
                action = (float[])_actions.Clone(),
                joint_names = ArmJointNames,
                joint_positions = (float[])_armPos.Clone(),
                gripper = _gripPos,
                grasped = _pieceGrabbed && Vector3.Distance(EePosIsaac(), _pieceIsaac) < graspDist,
                frame = _frameCount,
            };

            if (client != null && client.IsConnected)
                client.PublishJson(RosJson.SerializeDemoRecord(packet));

            if (_writer != null)
            {
                _writer.WriteLine(JsonUtility.ToJson(packet));
                _writer.Flush();
            }

            _frameCount++;
            StatusLine = $"recording frames={_frameCount}";
        }

        void BuildObs()
        {
            for (int i = 0; i < 7; i++)
            {
                _armVel[i] = (_armPos[i] - _prevArm[i]) / controlDt * 0.1f;
                _prevArm[i] = _armPos[i];
            }
            _gripVel = (_gripPos - _prevGrip) / controlDt * 0.1f;
            _prevGrip = _gripPos;

            var eeIsaac = EePosIsaac();
            var eeToPiece = _pieceIsaac - eeIsaac;
            var pieceToBucket = bucketPosIsaac - _pieceIsaac;
            float lifted = _pieceIsaac.z > liftHeight ? 1f : 0f;
            bool grasped = _pieceGrabbed && eeToPiece.magnitude < graspDist;

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
            _obs[o++] = grasped ? 1f : 0f;
            _obs[o++] = lifted;
        }

        void SyncPieceIsaac()
        {
            if (piece == null || envAnchor == null)
                return;
            _pieceIsaac = WorldToIsaacLocal(piece.position);
            if (_hasPrevPiece)
                _pieceVelIsaac = (_pieceIsaac - _prevPieceIsaac) / Mathf.Max(Time.deltaTime, 1e-4f);
            else
                _pieceVelIsaac = Vector3.zero;
            _prevPieceIsaac = _pieceIsaac;
            _hasPrevPiece = true;
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

        void InitArmFromRest()
        {
            // Match OfflineJointDriver.BindAgibotA2D rest
            float[] rest = { 1.0817f, -0.5907f, -0.3442f, 1.2819f, -0.6928f, -0.7f, 0f };
            for (int i = 0; i < 7; i++)
            {
                _armPos[i] = rest[i];
                _prevArm[i] = rest[i];
            }
            _gripPos = gripperOpen;
            _prevGrip = _gripPos;
            if (linkMap != null && eeLink == null)
                linkMap.TryGet("right_gripper_center", out eeLink);
            jointDriver?.BindAgibotA2D(linkMap, envAnchor != null ? envAnchor : transform);
            var angles = new float[8];
            for (int i = 0; i < 7; i++)
                angles[i] = _armPos[i];
            angles[7] = _gripPos;
            jointDriver?.ApplyAnglesRadians(angles);
        }

        void BindPieceGrab()
        {
            if (piece == null)
                return;
            _grab = piece.GetComponent<XRGrabInteractable>();
            if (_grab == null)
                return;
            _grab.selectEntered.AddListener(_ => _pieceGrabbed = true);
            _grab.selectExited.AddListener(_ => _pieceGrabbed = false);
        }

        void UnbindPieceGrab()
        {
            if (_grab == null)
                return;
            _grab.selectEntered.RemoveAllListeners();
            _grab.selectExited.RemoveAllListeners();
        }

        void OpenWriter()
        {
            CloseWriter();
            string dir = Path.Combine(Application.persistentDataPath, "PickPlaceDemos");
            Directory.CreateDirectory(dir);
            string stamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            _jsonlPath = Path.Combine(dir, $"demo_{stamp}.jsonl");
            _writer = new StreamWriter(_jsonlPath, false, Encoding.UTF8);
        }

        void CloseWriter()
        {
            if (_writer == null)
                return;
            _writer.Flush();
            _writer.Dispose();
            _writer = null;
        }
    }
}
