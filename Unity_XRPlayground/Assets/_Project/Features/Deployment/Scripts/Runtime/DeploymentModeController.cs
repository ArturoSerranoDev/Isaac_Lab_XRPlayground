using UnityEngine;

namespace XRPlayground.Deployment
{
    [DisallowMultipleComponent]
    public sealed class DeploymentModeController : MonoBehaviour
    {
        public DeploymentCatalogBinder binder;
        public string selectedStationId = "ball_catch";
        public int resetSeed = 42;

        public void SelectStation(string stationId) => selectedStationId = stationId;
        [ContextMenu("Set Selected Station Disabled")]
        public void SetDisabled() => SetMode(StationMode.Disabled);
        [ContextMenu("Set Selected Station Mirror")]
        public void SetMirror() => SetMode(StationMode.Mirror);
        [ContextMenu("Set Selected Station Offline Policy")]
        public void SetOfflinePolicy() => SetMode(StationMode.OfflinePolicy);

        [ContextMenu("Reset Selected Station")]
        public void ResetStation()
        {
            StationRuntime runtime = FindSelected();
            string error = null;
            if (runtime == null || !runtime.ResetActiveStation(resetSeed, out error))
                Debug.LogError(error ?? $"Station '{selectedStationId}' is missing", this);
        }

        [ContextMenu("Launch Ball")]
        public void LaunchBall()
        {
            SendCommand("launch_ball", "{\"position\":[0.55,0.0,0.95],\"velocity\":[-1.5,0.0,1.2]}");
        }

        public void SendCommand(string command, string payloadJson = "{}")
        {
            StationRuntime runtime = FindSelected();
            string error = null;
            if (runtime == null || !runtime.SendCommand(command, payloadJson, out error))
                Debug.LogError(error ?? $"Station '{selectedStationId}' is missing", this);
        }

        void SetMode(StationMode mode)
        {
            if (binder == null)
                binder = FindAnyObjectByType<DeploymentCatalogBinder>();
            string error = null;
            if (binder == null || !binder.SetMode(selectedStationId, mode, out error))
                Debug.LogError(error ?? "DeploymentCatalogBinder is missing", this);
        }

        StationRuntime FindSelected()
        {
            if (binder == null)
                binder = FindAnyObjectByType<DeploymentCatalogBinder>();
            if (binder == null)
                return null;
            foreach (StationRuntime runtime in binder.GetComponentsInChildren<StationRuntime>(true))
                if (runtime.stationId == selectedStationId)
                    return runtime;
            return null;
        }
    }
}
