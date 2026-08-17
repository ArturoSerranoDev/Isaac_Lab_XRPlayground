using UnityEngine;
using System;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class StationRuntime : MonoBehaviour
    {
        public string stationId;
        public StationMode mode = StationMode.Disabled;
        public PolicyRuntime policy;
        public MonoBehaviour adapterBehaviour;
        public MirrorRig mirrorRig;
        public GameObject offlineRig;
        public int resetSeed = 42;

        static StationRuntime s_ActiveOffline;
        static bool s_PhysicsOverrideActive;
        static float s_PreviousFixedDeltaTime;
        static int s_PreviousSolverIterations;
        static int s_PreviousSolverVelocityIterations;
        static float s_PreviousContactOffset;
        static float s_PreviousMaxDepenetrationVelocity;
        static int s_PreviousTargetFrameRate;
        IStationAdapter _adapter;
        float[] _observations;
        float[] _actions;
        int _physicsSteps;
        StationMode _appliedMode = (StationMode)(-1);
        bool _externalControl;

        public StationMode Mode => mode;
        public IStationAdapter Adapter => _adapter;
        public event Action<int> StationReset;
        public event Action<float[], float[], float[], float> PolicyStepCompleted;

        void Awake()
        {
            _adapter = adapterBehaviour as IStationAdapter;
            if (adapterBehaviour != null && _adapter == null)
                Debug.LogError("StationRuntime adapterBehaviour must implement IStationAdapter", this);
        }

        void OnEnable()
        {
            if (!ApplyMode(mode))
            {
                mode = StationMode.Disabled;
                ApplyMode(mode);
            }
        }

        void OnDisable()
        {
            _externalControl = false;
            if (s_ActiveOffline == this)
                ReleasePhysicsProfile();
            _appliedMode = (StationMode)(-1);
        }

        void OnValidate()
        {
            if (!string.IsNullOrEmpty(stationId) && policy != null)
                policy.expectedStationId = stationId;
        }

        public bool SetMode(StationMode requested, out string error)
        {
            if (requested == StationMode.OfflinePolicy && s_ActiveOffline != null && s_ActiveOffline != this)
            {
                error = $"offline station '{s_ActiveOffline.stationId}' is already active";
                return false;
            }
            StationMode previous = mode;
            if (ApplyMode(requested, out error))
            {
                mode = requested;
                return true;
            }
            mode = previous;
            return false;
        }

        public bool ResetActiveStation(int seed, out string error)
        {
            resetSeed = seed;
            if (_appliedMode == StationMode.OfflinePolicy && _adapter != null)
            {
                try
                {
                    policy?.ResetState();
                    _adapter.ResetStation(seed);
                }
                catch (Exception exception)
                {
                    error = $"adapter reset failed: {exception.Message}";
                    return false;
                }
                error = _adapter.IsHealthy ? null : "adapter reset health check failed";
                if (_adapter.IsHealthy)
                    StationReset?.Invoke(seed);
                return _adapter.IsHealthy;
            }
            if (_appliedMode == StationMode.Mirror && mirrorRig?.bridge != null)
            {
                mirrorRig.bridge.PublishSessionCommand("mirror", "reset");
                error = null;
                return true;
            }
            error = "station is disabled";
            return false;
        }

        public bool SendCommand(string command, string payloadJson, out string error)
        {
            if (string.IsNullOrWhiteSpace(command))
            {
                error = "command must be non-empty";
                return false;
            }
            if (_appliedMode == StationMode.OfflinePolicy && _adapter != null)
                return _adapter.TryHandleCommand(command, payloadJson, out error);
            if (_appliedMode == StationMode.Mirror && mirrorRig?.bridge != null)
            {
                mirrorRig.bridge.PublishSessionCommand("mirror", command, payloadJson);
                error = null;
                return true;
            }
            error = "station is disabled";
            return false;
        }

        public bool BeginOpenLoopControl(int seed, out string error)
        {
            if (_appliedMode != StationMode.OfflinePolicy || _adapter == null ||
                policy?.Contract == null)
            {
                error = "open-loop control requires a healthy OfflinePolicy station";
                return false;
            }
            _externalControl = true;
            if (!ResetActiveStation(seed, out error))
            {
                _externalControl = false;
                return false;
            }
            _physicsSteps = 0;
            return true;
        }

        public bool ApplyOpenLoopAction(float[] processedAction, out string error)
        {
            if (!_externalControl || _appliedMode != StationMode.OfflinePolicy ||
                policy?.Contract == null || _adapter == null)
            {
                error = "open-loop control is not active";
                return false;
            }
            if (processedAction == null || processedAction.Length != policy.Contract.ActionDimension)
            {
                error = "open-loop action dimensions do not match the contract";
                return false;
            }
            foreach (float value in processedAction)
                if (float.IsNaN(value) || float.IsInfinity(value))
                {
                    error = "open-loop action contains a non-finite value";
                    return false;
                }
            try
            {
                _adapter.BuildObservation(_observations);
                _adapter.ApplyAction(processedAction, policy.Contract.timing.policy_dt);
                PolicyStepCompleted?.Invoke(
                    (float[])_observations.Clone(),
                    (float[])processedAction.Clone(),
                    (float[])processedAction.Clone(),
                    _adapter.TaskScore);
                error = null;
                return true;
            }
            catch (Exception exception)
            {
                error = $"open-loop action failed: {exception.Message}";
                return false;
            }
        }

        public void EndOpenLoopControl() => _externalControl = false;

        bool ApplyMode(StationMode requested)
        {
            bool result = ApplyMode(requested, out string error);
            if (!result)
                Debug.LogError($"StationRuntime '{stationId}': {error}", this);
            return result;
        }

        bool ApplyMode(StationMode requested, out string error)
        {
            if (_appliedMode == requested)
            {
                error = null;
                return true;
            }
            if (requested == StationMode.OfflinePolicy)
            {
                error = null;
                if (s_ActiveOffline != null && s_ActiveOffline != this)
                {
                    error = $"offline station '{s_ActiveOffline.stationId}' is already active";
                    return false;
                }
                if (_adapter == null || policy == null || !policy.TryLoad(out error))
                {
                    error ??= "offline mode requires a station adapter and policy runtime";
                    return false;
                }
                PolicyContract contract = policy.Contract;
                if (_adapter.AdapterId != contract.adapter_id)
                {
                    error = $"adapter '{_adapter.AdapterId}' != contract '{contract.adapter_id}'";
                    return false;
                }
                if (_adapter.ObservationDimension != contract.ObservationDimension ||
                    _adapter.ActionDimension != contract.ActionDimension)
                {
                    error = "adapter dimensions do not match the policy contract";
                    return false;
                }
                if (!StationCatalog.TryLoad(out StationCatalog catalog, out error))
                    return false;
                StationCatalogEntry station = catalog.Find(stationId);
                StationCatalogPhysicsProfile physicsProfile =
                    station != null ? catalog.FindPhysicsProfile(station.physics_profile_id) : null;
                if (station == null || physicsProfile == null ||
                    station.robot_id != contract.robot_id ||
                    physicsProfile.profile_id != contract.physics_profile_id ||
                    physicsProfile.physics_hz != contract.timing.deployment_physics_hz)
                {
                    error = "catalog robot or physics profile does not match the policy contract";
                    return false;
                }
                OfflineRigIdentity rigIdentity =
                    offlineRig != null ? offlineRig.GetComponent<OfflineRigIdentity>() : null;
                if (rigIdentity == null ||
                    !rigIdentity.ValidateForStation(catalog, station, true, out error))
                {
                    error ??= "offline mode requires a validated normalized or procedural rig";
                    return false;
                }
                if (contract.evaluation.ready && !physicsProfile.calibrated)
                {
                    error = "production policy cannot run against an uncalibrated physics profile";
                    return false;
                }
                // Articulation drivers bind in Awake/OnEnable. Station shells
                // keep offline rigs inactive in Mirror/Disabled modes, so the
                // rig must be active before adapter BindContract/ResetStation
                // performs its health check. Roll it back on every failure.
                bool activatedOfflineRig = !offlineRig.activeSelf;
                if (activatedOfflineRig)
                    offlineRig.SetActive(true);
                ApplyPhysicsProfile(contract, physicsProfile, catalog.runtime.target_render_hz);
                try
                {
                    policy.ResetState();
                    _adapter.BindContract(contract);
                    _observations = new float[contract.ObservationDimension];
                    _actions = new float[contract.ActionDimension];
                    _physicsSteps = 0;
                    _adapter.ResetStation(resetSeed);
                }
                catch (System.Exception exception)
                {
                    error = $"station adapter initialization failed: {exception.Message}";
                    if (activatedOfflineRig)
                        offlineRig.SetActive(false);
                    ReleasePhysicsProfile();
                    return false;
                }
                StationReset?.Invoke(resetSeed);
                if (!_adapter.IsHealthy)
                {
                    error = "station adapter failed its reset health check";
                    if (activatedOfflineRig)
                        offlineRig.SetActive(false);
                    if (s_ActiveOffline == this)
                        ReleasePhysicsProfile();
                    return false;
                }
            }
            else if (s_ActiveOffline == this)
            {
                _externalControl = false;
                ReleasePhysicsProfile();
            }
            if (mirrorRig != null)
                mirrorRig.gameObject.SetActive(requested == StationMode.Mirror);
            if (offlineRig != null)
                offlineRig.SetActive(requested == StationMode.OfflinePolicy);
            _appliedMode = requested;
            error = null;
            return true;
        }

        void FixedUpdate()
        {
            if (mode != _appliedMode)
            {
                if (!ApplyMode(mode))
                {
                    mode = StationMode.Disabled;
                    ApplyMode(mode);
                }
            }
            if (_appliedMode != StationMode.OfflinePolicy || policy?.Contract == null || _adapter == null)
                return;
            if (_externalControl)
                return;
            int cadence = Mathf.Max(1, policy.Contract.timing.deployment_decimation);
            if ((_physicsSteps++ % cadence) != 0)
                return;
            try
            {
                _adapter.BuildObservation(_observations);
                if (!policy.TryInfer(_observations, _actions, out string error))
                    throw new InvalidOperationException(error);
                _adapter.ApplyAction(_actions, policy.Contract.timing.policy_dt);
                PolicyStepCompleted?.Invoke(
                    (float[])_observations.Clone(),
                    policy.LastRawActions,
                    (float[])_actions.Clone(),
                    _adapter.TaskScore);
            }
            catch (Exception exception)
            {
                Debug.LogError($"StationRuntime '{stationId}' policy step failed: {exception.Message}", this);
                mode = StationMode.Disabled;
                ApplyMode(mode);
            }
        }

        void ApplyPhysicsProfile(
            PolicyContract contract,
            StationCatalogPhysicsProfile physicsProfile,
            int targetRenderHz)
        {
            if (!s_PhysicsOverrideActive)
            {
                s_PreviousFixedDeltaTime = Time.fixedDeltaTime;
                s_PreviousSolverIterations = Physics.defaultSolverIterations;
                s_PreviousSolverVelocityIterations = Physics.defaultSolverVelocityIterations;
                s_PreviousContactOffset = Physics.defaultContactOffset;
                s_PreviousMaxDepenetrationVelocity = Physics.defaultMaxDepenetrationVelocity;
                s_PreviousTargetFrameRate = Application.targetFrameRate;
                s_PhysicsOverrideActive = true;
            }
            s_ActiveOffline = this;
            Time.fixedDeltaTime = contract.timing.deployment_sim_dt;
            Physics.defaultSolverIterations = physicsProfile.solver_iterations;
            Physics.defaultSolverVelocityIterations = physicsProfile.solver_velocity_iterations;
            Physics.defaultContactOffset = physicsProfile.default_contact_offset_m;
            Physics.defaultMaxDepenetrationVelocity =
                physicsProfile.maximum_depenetration_velocity_mps;
            Application.targetFrameRate = targetRenderHz;
        }

        static void ReleasePhysicsProfile()
        {
            s_ActiveOffline = null;
            if (!s_PhysicsOverrideActive)
                return;
            Time.fixedDeltaTime = s_PreviousFixedDeltaTime;
            Physics.defaultSolverIterations = s_PreviousSolverIterations;
            Physics.defaultSolverVelocityIterations = s_PreviousSolverVelocityIterations;
            Physics.defaultContactOffset = s_PreviousContactOffset;
            Physics.defaultMaxDepenetrationVelocity = s_PreviousMaxDepenetrationVelocity;
            Application.targetFrameRate = s_PreviousTargetFrameRate;
            s_PhysicsOverrideActive = false;
        }
    }
}
