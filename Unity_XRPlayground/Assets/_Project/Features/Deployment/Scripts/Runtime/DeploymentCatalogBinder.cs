using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class DeploymentCatalogBinder : MonoBehaviour
    {
        public bool configureOnAwake = true;

        void Awake()
        {
            if (configureOnAwake && !Configure(out string error))
                Debug.LogError($"Deployment catalog binding failed: {error}", this);
        }

        [ContextMenu("Configure From Generated Catalog")]
        public void ConfigureFromContextMenu()
        {
            if (!Configure(out string error))
                Debug.LogError($"Deployment catalog binding failed: {error}", this);
        }

        public bool Configure(out string error)
        {
            if (!StationCatalog.TryLoad(out StationCatalog catalog, out error))
                return false;
            foreach (StationRuntime runtime in GetComponentsInChildren<StationRuntime>(true))
            {
                StationCatalogEntry station = catalog.Find(runtime.stationId);
                if (station == null)
                {
                    error = $"scene station '{runtime.stationId}' is absent from the catalog";
                    return false;
                }
                if (runtime.policy != null)
                    runtime.policy.expectedStationId = station.station_id;
                StationCatalogAssistProfile assistSource =
                    catalog.FindAssistProfile(station.assist_profile_id);
                if (assistSource == null || !System.Enum.TryParse(assistSource.kind, out AssistKind kind))
                {
                    error = $"station '{station.station_id}' has an invalid assist profile";
                    return false;
                }
                var assist = ScriptableObject.CreateInstance<AssistProfile>();
                assist.hideFlags = HideFlags.HideAndDontSave;
                assist.profileId = assistSource.profile_id;
                assist.kind = kind;
                assist.requiredDistinctContacts = assistSource.required_distinct_contacts;
                assist.breakForce = assistSource.break_force_n;
                assist.breakTorque = assistSource.break_torque_nm;
                assist.uprightAngularDamping = assistSource.upright_angular_damping;
                assist.maximumUprightTiltDegrees = assistSource.maximum_upright_tilt_degrees;
                assist.enabledInProduction = assistSource.enabled_in_production;
                foreach (ContactGripAssist grip in runtime.GetComponentsInChildren<ContactGripAssist>(true))
                    grip.profile = assist;
                foreach (SpotUprightAssist upright in runtime.GetComponentsInChildren<SpotUprightAssist>(true))
                    upright.profile = assist;
                BridgeV2Client bridge = runtime.GetComponentInChildren<BridgeV2Client>(true);
                if (bridge == null)
                    continue;
                bridge.stationId = station.station_id;
                bridge.staleAfterSeconds = catalog.bridge.stale_after_seconds;
                bridge.disconnectAfterSeconds = catalog.bridge.disconnect_after_seconds;
                if (bridge.transport != null)
                {
                    bridge.transport.host = catalog.bridge.host;
                    bridge.transport.port = station.port;
                    bridge.transport.maxMessageBytes = catalog.bridge.max_message_bytes;
                }
            }
            error = null;
            return true;
        }

        public bool SetMode(string stationId, StationMode mode, out string error)
        {
            foreach (StationRuntime runtime in GetComponentsInChildren<StationRuntime>(true))
                if (runtime.stationId == stationId)
                    return runtime.SetMode(mode, out error);
            error = $"scene has no station runtime '{stationId}'";
            return false;
        }
    }
}
