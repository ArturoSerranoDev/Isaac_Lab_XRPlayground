using System;
using System.Collections.Generic;
using UnityEngine;

namespace XRPlayground.ROS
{
    /// <summary>Runtime view of the single shared station manifest.</summary>
    [Serializable]
    public sealed class StationDefinition
    {
        public string id;
        public string[] task_ids;
        public string policy_task_id;
        public string policy_folder;
        public string bridge_script;
        public int port;
        public int obs_dim;
        public int action_dim;
        public string[] required_topics;
    }

    [Serializable]
    sealed class StationManifest
    {
        public StationDefinition[] stations;
    }

    /// <summary>
    /// Resolves station/task/port/topic metadata from Assets/Resources/XRPlayground/stations.json.
    /// Python tooling reads that exact same file; do not recreate values in station setup scripts.
    /// </summary>
    public static class StationRegistry
    {
        const string ResourcePath = "XRPlayground/stations";
        static readonly Dictionary<string, StationDefinition> ById = new(StringComparer.Ordinal);
        static bool _loaded;

        static void EnsureLoaded()
        {
            if (_loaded)
                return;
            _loaded = true;
            var asset = Resources.Load<TextAsset>(ResourcePath);
            if (asset == null)
            {
                Debug.LogError("XRPlayground: station manifest is missing at Resources/XRPlayground/stations.json.");
                return;
            }
            var manifest = JsonUtility.FromJson<StationManifest>(asset.text);
            if (manifest?.stations == null)
                return;
            foreach (var station in manifest.stations)
            {
                if (station != null && !string.IsNullOrEmpty(station.id))
                    ById[station.id] = station;
            }
        }

        public static bool TryGet(string stationId, out StationDefinition station)
        {
            EnsureLoaded();
            return ById.TryGetValue(stationId, out station);
        }

        public static bool TryGetForTask(string taskId, out StationDefinition station)
        {
            EnsureLoaded();
            foreach (var candidate in ById.Values)
            {
                if (candidate.task_ids == null)
                    continue;
                foreach (var id in candidate.task_ids)
                {
                    if (id == taskId)
                    {
                        station = candidate;
                        return true;
                    }
                }
            }
            station = null;
            return false;
        }

        public static bool ApplyTo(RosTcpClient client, string stationId)
        {
            if (client == null || !TryGet(stationId, out var station))
                return false;
            client.port = station.port;
            return true;
        }
    }
}
