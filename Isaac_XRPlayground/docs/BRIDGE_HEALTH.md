# Bridge health and station contract

`Unity_XRPlayground/Assets/Resources/XRPlayground/stations.json` is the canonical record for every XR station's task IDs, TCP port, required topics, policy package, and observation/action dimensions. Unity's station setup scripts and the Python launcher both consume it.

## Unity

Run the existing station setup menu. It adds a `BridgeHealthMonitor` to the bridge object and the Ball Catch / Conveyor world panels show four live checks:

- TCP port connection
- Isaac heartbeat freshness
- required-topic flow
- assigned ONNX policy sidecar contract

Use **Log Bridge Health** on the component for the per-topic report and packet ages.

`OnnxPolicyRunner` automatically discovers the adjacent `policy.json` when `policy.onnx` is assigned in the Editor. The sidecar is serialized into the scene for builds, supplies dimensions/action scale/dt, and rejects a model whose `task_id`, observation dimension, or action dimension does not match the offline controller.

## Command-line smoke test

With a bridge running, run:

```powershell
python Isaac_XRPlayground/scripts/bridge/health_check.py --station ball_catch --timeout 5
python Isaac_XRPlayground/scripts/bridge/health_check.py --station conveyor_color --timeout 5
```

The test verifies port reachability, receives a heartbeat and every required topic, then validates the station's `policy.json` contract and ONNX file. Use `--offline` to validate only the policy package and manifest before starting Isaac.

```powershell
python Isaac_XRPlayground/scripts/bridge/health_check.py --station ball_catch --offline
```

An expected failure such as `ONNX assignment unavailable` means the station has no exported `policy.onnx` yet; export one before treating the station as ready for offline play.
