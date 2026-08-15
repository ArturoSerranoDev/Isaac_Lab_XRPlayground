# Agibot A2D table pick-and-place

Wall-mounted **Agibot A2D** (right arm) picks random cubes from a table and places them in a front-center bucket. Matches the Unity station layout and ONNX offline policy.

## Task ID

`Template-Xrplayground-Pick-Place-Table-Direct-v0`

## Train (RSL-RL)

From `Isaac_XRPlayground/`:

```bat
python scripts/rsl_rl/train.py ^
  --task=Template-Xrplayground-Pick-Place-Table-Direct-v0 ^
  --num_envs=128 --viz none --max_iterations=2500 --export_onnx
```

Logs: `logs/rsl_rl/xrplayground_pick_place_table_direct/`  
Unity export: `Unity_XRPlayground/Assets/_Project/Features/Policies/PickPlace/`

Health check:

```bat
python scripts/rsl_rl/train_health.py --log-root logs/rsl_rl/xrplayground_pick_place_table_direct --task pick_place
```

Target gates: `Metrics/place_rate` ≥ 0.15 sustained, grasp_rate rising, mean reward stable.

## Obs / actions (must match Unity)

| | Dim | Notes |
|---|-----|--------|
| Obs | 30 | 7 arm pos + 7 vel + grip pos/vel + piece pos/vel + ee→piece + piece→bucket + grasped + lifted |
| Actions | 8 | 7 right-arm deltas + 1 gripper |
| action_scale | 5.0 | |

## Scene (Isaac Z-up)

| Object | Center (x,y,z) | Size |
|--------|----------------|------|
| Robot | (0, -0.78, 0) | wall-mounted, torso frozen |
| Table | (0.45, 0, 0.38) | (0.75, 0.60, 0.04) |
| Bucket | (0.45, -0.18, 0.38) | (0.14, 0.14, 0.10) |
| Spawn XY | x∈[0.30,0.60], y∈[-0.22,0.22] | on table |

## Unity

1. Copy USD from Nucleus: `Robots/Agibot/A2D/A2D_physics.usd` → `Assets/_Project/Features/Robots/AgibotA2D/USD/`
2. Menu: **XRPlayground → Setup Pick Place Table Station**
3. Assign `Policies/PickPlace/policy.onnx` to `OnnxPolicyRunner`
4. Play → **Start Offline Policy**

Bridge (mirror Isaac): port **9092**, script `scripts/bridge/run_xr_bridge_pick_place.py`

## Imitation learning (later)

`pick_place_table_env_cfg.py` exposes:

- `imitation_learning_enabled = False` — flip when IL pipeline is ready
- Unity stub: `PickPlaceDemoRecorder.cs` (VR grab → record `(obs, action)` demos)

Planned flow:

1. Unity VR: user grasps piece with XR Interaction Toolkit
2. Recorder publishes `/xr/pick_place/demo_record` at `demo_record_rate_hz`
3. Isaac Lab Mimic / custom datagen ingests demos for behavior cloning or RL fine-tune

Not implemented in this milestone — RL-only training first.
