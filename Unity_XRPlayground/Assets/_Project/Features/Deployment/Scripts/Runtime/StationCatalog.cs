using System;
using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.Deployment
{
    [Serializable]
    public sealed class StationCatalogBridge
    {
        public string host;
        public float stale_after_seconds;
        public float disconnect_after_seconds;
        public int max_message_bytes;
    }

    [Serializable]
    public sealed class StationCatalogRuntime
    {
        public string platform;
        public int target_render_hz;
        public int minimum_render_hz;
        public int maximum_active_offline_stations;
        public string default_inference_backend;
    }

    [Serializable]
    public sealed class StationCatalogPhysicsProfile
    {
        public string profile_id;
        public int physics_hz;
        public int solver_iterations;
        public int solver_velocity_iterations;
        public float default_contact_offset_m;
        public float maximum_depenetration_velocity_mps;
        public bool calibrated;
    }

    [Serializable]
    public sealed class StationCatalogAssistProfile
    {
        public string profile_id;
        public string kind;
        public int required_distinct_contacts;
        public float break_force_n;
        public float break_torque_nm;
        public float upright_angular_damping;
        public float maximum_upright_tilt_degrees;
        public bool enabled_in_production;
    }

    [Serializable]
    public sealed class StationCatalogRobotAsset
    {
        public string robot_id;
        public string provenance;
        public string redistribution_license;
        public string[] license_paths;
        public bool redistribution_verified;
    }

    [Serializable]
    public sealed class StationCatalogPolicy
    {
        public string policy_id;
        public string source_task_id;
        public string unity_folder;
        public int observation_dim;
        public int action_dim;
        public int physics_hz;
        public int policy_hz;
    }

    [Serializable]
    public sealed class StationCatalogOfflineBindings
    {
        public string root_link;
        public string end_effector_link;
        public string[] contact_links;
        public float[] robot_position_isaac;
    }

    [Serializable]
    public sealed class StationCatalogEntry
    {
        public string station_id;
        public string display_name;
        public string robot_id;
        public string adapter_id;
        public int port;
        public string physics_profile_id;
        public string assist_profile_id;
        public string[] policy_ids;
        public string[] bridge_payloads;
        public StationCatalogOfflineBindings offline_bindings;
        public StationCatalogPolicy[] policies;
    }

    [Serializable]
    public sealed class StationCatalog
    {
        public int schema_version;
        public StationCatalogBridge bridge;
        public StationCatalogRuntime runtime;
        public StationCatalogPhysicsProfile[] physics_profiles;
        public StationCatalogAssistProfile[] assist_profiles;
        public StationCatalogRobotAsset[] robot_assets;
        public StationCatalogEntry[] stations;

        public static bool TryLoad(out StationCatalog catalog, out string error)
        {
            TextAsset asset = Resources.Load<TextAsset>("XRPlayground/stations");
            if (asset == null)
            {
                catalog = null;
                error = "Generated Resources/XRPlayground/stations.json is missing";
                return false;
            }
            catalog = JsonUtility.FromJson<StationCatalog>(asset.text);
            if (catalog == null)
            {
                error = "generated station catalog JSON is invalid";
                return false;
            }
            return catalog.Validate(out error);
        }

        public StationCatalogEntry Find(string stationId)
        {
            if (stations != null)
                foreach (StationCatalogEntry station in stations)
                    if (station != null && station.station_id == stationId)
                        return station;
            return null;
        }

        public StationCatalogPhysicsProfile FindPhysicsProfile(string profileId)
        {
            if (physics_profiles != null)
                foreach (StationCatalogPhysicsProfile profile in physics_profiles)
                    if (profile != null && profile.profile_id == profileId)
                        return profile;
            return null;
        }

        public StationCatalogAssistProfile FindAssistProfile(string profileId)
        {
            if (assist_profiles != null)
                foreach (StationCatalogAssistProfile profile in assist_profiles)
                    if (profile != null && profile.profile_id == profileId)
                        return profile;
            return null;
        }

        public StationCatalogRobotAsset FindRobotAsset(string robotId)
        {
            if (robot_assets != null)
                foreach (StationCatalogRobotAsset asset in robot_assets)
                    if (asset != null && asset.robot_id == robotId)
                        return asset;
            return null;
        }

        public bool Validate(out string error)
        {
            if (schema_version != 3 || bridge == null || runtime == null || stations == null ||
                physics_profiles == null || assist_profiles == null || robot_assets == null)
                return Fail("station catalog schema is incomplete", out error);
            if (bridge.stale_after_seconds <= 0f ||
                bridge.disconnect_after_seconds <= bridge.stale_after_seconds)
                return Fail("bridge stale/disconnect thresholds are invalid", out error);
            if (runtime.maximum_active_offline_stations != 1 || runtime.target_render_hz < 72)
                return Fail("runtime must enforce one offline station and PCVR cadence", out error);
            var stationIds = new HashSet<string>();
            var physicsIds = new HashSet<string>();
            foreach (StationCatalogPhysicsProfile profile in physics_profiles)
                if (profile == null || string.IsNullOrWhiteSpace(profile.profile_id) ||
                    !physicsIds.Add(profile.profile_id) || profile.physics_hz <= 0 ||
                    profile.solver_iterations <= 0 || profile.solver_velocity_iterations <= 0)
                    return Fail("station catalog contains an invalid physics profile", out error);
            var assistIds = new HashSet<string>();
            foreach (StationCatalogAssistProfile profile in assist_profiles)
                if (profile == null || string.IsNullOrWhiteSpace(profile.profile_id) ||
                    !assistIds.Add(profile.profile_id) || profile.required_distinct_contacts < 0)
                    return Fail("station catalog contains an invalid assist profile", out error);
            var robotIds = new HashSet<string>();
            foreach (StationCatalogRobotAsset asset in robot_assets)
                if (asset == null || string.IsNullOrWhiteSpace(asset.robot_id) ||
                    !robotIds.Add(asset.robot_id) || string.IsNullOrWhiteSpace(asset.provenance) ||
                    string.IsNullOrWhiteSpace(asset.redistribution_license) ||
                    asset.license_paths == null ||
                    (asset.redistribution_verified &&
                     !string.Equals(asset.redistribution_license, "project-authored",
                         StringComparison.OrdinalIgnoreCase) && asset.license_paths.Length == 0) ||
                    HasInvalidOrDuplicateValues(asset.license_paths))
                    return Fail("station catalog contains an invalid robot asset", out error);
            foreach (StationCatalogEntry station in stations)
            {
                if (station == null || string.IsNullOrWhiteSpace(station.station_id) ||
                    !stationIds.Add(station.station_id) || station.port <= 0 || station.policies == null ||
                    station.offline_bindings == null ||
                    string.IsNullOrWhiteSpace(station.offline_bindings.root_link) ||
                    station.offline_bindings.contact_links == null ||
                    station.offline_bindings.robot_position_isaac == null ||
                    station.offline_bindings.robot_position_isaac.Length != 3)
                    return Fail("station catalog contains an invalid or duplicate station", out error);
                var contactLinks = new HashSet<string>();
                foreach (string link in station.offline_bindings.contact_links)
                    if (string.IsNullOrWhiteSpace(link) || !contactLinks.Add(link))
                        return Fail(
                            $"station '{station.station_id}' has invalid offline contact bindings",
                            out error);
                if (!physicsIds.Contains(station.physics_profile_id) ||
                    !assistIds.Contains(station.assist_profile_id) ||
                    !robotIds.Contains(station.robot_id))
                    return Fail($"station '{station.station_id}' references an unknown profile", out error);
                foreach (StationCatalogPolicy policy in station.policies)
                    if (policy == null || policy.observation_dim <= 0 || policy.action_dim <= 0 ||
                        policy.policy_hz <= 0 || policy.physics_hz % policy.policy_hz != 0)
                        return Fail($"station '{station.station_id}' contains an invalid policy", out error);
            }
            error = null;
            return true;
        }

        static bool HasInvalidOrDuplicateValues(IEnumerable<string> values)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (string value in values)
                if (string.IsNullOrWhiteSpace(value) || !seen.Add(value))
                    return true;
            return false;
        }

        static bool Fail(string message, out string error)
        {
            error = message;
            return false;
        }
    }
}
