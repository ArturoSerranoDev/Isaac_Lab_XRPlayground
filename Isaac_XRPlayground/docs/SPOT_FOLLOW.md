# Spot Follow-Me — Task station checklist

Goal: train Spot stand→walk (loco) then pose-follow over pretrained loco, reuse in Unity XR (ROS mirror + offline HMD follow).

## Identity

| Field | Value |
|-------|--------|
| Task name | Spot Follow-Me (stand → walk → follow) |
| Gym IDs | `Template-Xrplayground-Spot-Loco-Stand-v0`, `…-Spot-Loco-Walk-v0`, `…-Spot-Follow-v0` |
| Isaac package path | `source/XRPlayground/XRPlayground/tasks/manager_based/spot_loco/` + `spot_follow/` |
| Unity station name | `Station_E_Spot` |
| Unity setup menu | `XRPlayground → Setup Spot Follow Station` |
| ROS TCP port | `9094` |
| Unity policy folders | `Assets/_Project/Features/Policies/SpotLoco/`, `SpotFollow/` |
| RSL-RL experiment names | `xrplayground_spot_loco`, `xrplayground_spot_follow` |

## 1. Robot asset

| Field | Value |
|-------|--------|
| Isaac USD / asset cfg | `isaaclab_assets.robots.spot.SPOT_CFG` (Nucleus `Robots/BostonDynamics/spot/spot.usd`); local cache `assets/usd/spot/spot.usd` + `Isaac_XRPlayground/assets/usd/spot/spot.usd` |
| Unity USD path | `Assets/_Project/Features/Robots/Spot/USD/spot.usd` (file present; **no usable Unity GameObject import** — station uses procedural unit-scale pivots + `*_viz` meshes from setup menu) |
| Source / Nucleus URL | `ISAAC_NUCLEUS_DIR/Robots/BostonDynamics/spot/spot.usd` |
| Default joint vector (Isaac, rad) | `[0.1, 0.9, -1.5, -0.1, 0.9, -1.5, 0.1, 1.1, -1.5, -0.1, 1.1, -1.5]` (`fl_hx…hr_kn`) |
| Gripper open / close | N/A |

- [x] USD present in **both** Isaac cache and Unity (Nucleus still used at train time if local schema refs missing)
- [x] Unity setup menu leaves robot at **default joint pose** after bind
- [x] Offline FK rest captured only at that default (`BindSpot(..., force: false)` on Start Policy)

## 2. Name table (links / joints)

| Role | Isaac / USD prim name | Unity map field |
|------|----------------------|-----------------|
| Base | `body` | `SpotLinkMap` |
| FL chain | `fl_hip`, `fl_uleg`, `fl_lleg`, `fl_foot` | same |
| FR / HL / HR | `fr_*` / `hl_*` / `hr_*` | same |
| Joints (12) | `fl_hx`, `fl_hy`, `fl_kn`, … `hr_kn` | `OfflineJointDriver.BindSpot` |

Shared code:

- Isaac: `bridge/names_spot.py`
- Unity: `SpotLinkMap.cs` + `OfflineJointDriver.BindSpot`

- [x] Names taken from **real USD** prims
- [x] Bind assert: **17 / 17** expected links
- [x] Offline joint count matches loco action layout (`12`)
- [x] ROS pose follower uses the **same** names

## 3. Pose table (Isaac Z-up → Unity via XrFrameConverter)

| Object | Isaac (x, y, z) | Unity local (after converter) | Notes |
|--------|-----------------|-------------------------------|-------|
| Spot body root | `(0, 0, 0.5)` | `IsaacPosToUnity` | Station under `02_Locomotion` at local `(0,0,2)` |
| HMD / player target | center-eye world → env-local Isaac | `SpotPlayerTargetPublisher` / offline controller | **Only** `XrFrameConverter` |
| Pose command (follow) | body-relative `(dx, dy, dyaw)` | built from HMD − body | Matches nav `pose_command` |

- [x] Layout under locomotion zone
- [x] Setup menu + offline use the same converter
- [x] Only `XrFrameConverter` used for frame changes

## 4. Obs / action contract

### SpotLoco (low-level)

| Field | Isaac | Unity | `SpotLoco/policy.json` |
|-------|-------|-------|------------------------|
| `obs_dim` | **48** | **48** | **48** |
| `action_dim` | 12 | 12 | 12 |
| `action_scale` | 0.2 | 0.2 | 0.2 |
| Control `dt` | 0.02 (50 Hz) | 0.02 | 0.02 |
| Obs layout | lin_vel(3), ang_vel(3), grav(3), vel_cmd(3), q_rel(12), qd_rel(12), last_a(12) | same | same |

### SpotFollow (high-level)

| Field | Isaac | Unity | `SpotFollow/policy.json` |
|-------|-------|-------|--------------------------|
| `obs_dim` | **9** | **9** | **9** |
| `action_dim` | 3 | 3 | 3 |
| `action_scale` | 1.0 | 1.0 | 1.0 |
| Control `dt` | 0.2 | 0.2 | 0.2 |
| Obs layout | lin_vel(3), grav(3), pose_cmd(3) | same | same |
| Actions | `(vx, vy, ωz)` → loco `velocity_commands` | same | same |

- [x] Controllers match training dims
- [ ] Export + Unity ONNX copy after GPU train (placeholders until then)
- [x] Obs/layout changes update Unity in the same package

## 5. ROS mirror

| Field | Value |
|-------|--------|
| Bridge script | `scripts/bridge/run_xr_bridge_spot.py` |
| Topics | `/xr/spot/robot_state`, `/xr/spot/player_pose`, `/xr/heartbeat`, `/xr/session_*` |
| Unity client / followers | `SpotLinkPoseFollower`, `SpotPlayerTargetPublisher`, `SpotBridgePanel` |

- [ ] Connect → links track Isaac within visual tolerance (smoke when GPU free)
- [ ] HMD pose appears on Isaac bridge (`player_pose_ok` in session status)

## 6. Offline ONNX

| Field | Value |
|-------|--------|
| Controller | `SpotOfflinePolicyController.cs` |
| Joint driver bind method | `BindSpot` |
| Model asset paths | `Policies/SpotLoco/policy.onnx`, optional `Policies/SpotFollow/policy.onnx` |

- [ ] Start from **default pose** → legs move like Isaac play (needs trained ONNX)
- [x] FK rest **not** re-captured every start
- [x] Sidecar `policy.json` present (placeholders until train)

## 7. Verification order (do in order)

1. [x] Setup menu / docs create station with named hierarchy + maps + poses
2. [ ] ROS mirror smoke (~10 s) — GPU currently on BallCatch
3. [ ] Offline ONNX smoke from default pose (after export)
4. [ ] Train Phase A Stand → Phase B Walk → export SpotLoco
5. [ ] Train Phase C Follow with `XRPLAYGROUND_SPOT_LOCO_POLICY` → export SpotFollow
6. [x] This doc filled

## Train / export commands (GPU free)

Isaac Python: `C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe`  
Cwd: `Isaac_XRPlayground`

```powershell
# Phase A — Stand
& "...\python.exe" scripts/rsl_rl/train.py --task=Template-Xrplayground-Spot-Loco-Stand-v0 --num_envs=128 --viz none --max_iterations=500 --export_onnx

# Phase B — Walk (resume from Stand run, or fresh)
& "...\python.exe" scripts/rsl_rl/train.py --task=Template-Xrplayground-Spot-Loco-Walk-v0 --num_envs=128 --viz none --max_iterations=2000 --export_onnx --resume --load_run=<stand_or_walk_run>

# Export loco if needed
& "...\python.exe" scripts/rsl_rl/export_onnx.py --task=Template-Xrplayground-Spot-Loco-Walk-v0 --ckpt=logs/rsl_rl/xrplayground_spot_loco/<run>/model_XXXX.pt --num_envs=1 --viz none

# Phase C — Follow (point at loco JIT/pt used by PreTrainedPolicyAction)
$env:XRPLAYGROUND_SPOT_LOCO_POLICY = "logs/rsl_rl/xrplayground_spot_loco/<run>/exported/policy.pt"
& "...\python.exe" scripts/rsl_rl/train.py --task=Template-Xrplayground-Spot-Follow-v0 --num_envs=64 --viz none --max_iterations=1500 --export_onnx
```

Bridge smoke:

```powershell
& "...\python.exe" scripts/bridge/run_xr_bridge_spot.py --task=Template-Xrplayground-Spot-Loco-Walk-Play-v0 --port=9094 --num_envs=1 --viz kit
```

## Lessons / do-not-repeat

- Local `spot.usd` may reference missing `configuration/spot_robot_schema.usd` — prefer Nucleus `SPOT_CFG` for train; Unity station is **procedural** (do not expect imported USD mesh)
- Nested scaled primitives under a non-unit `body` crush child lossyScale to ~0 — joint pivots must stay `localScale=(1,1,1)` with mesh on `*_viz` children
- Do not dual-train with BallCatch / other GPU jobs
- Never invent a second frame transform; HMD → Isaac only via `XrFrameConverter`
- Follow train requires a valid loco `policy.pt` path

## References

- Standards rule: `.cursor/rules/isaac-unity-training-station.mdc`
- ONNX: `docs/ONNX_UNITY.md`
- Bridge: `docs/UNITY_ISAAC_BRIDGE.md`
- Isaac refs: `Isaac-Velocity-Flat-Spot-v0`, Anymal-C `navigation_env_cfg.py`
