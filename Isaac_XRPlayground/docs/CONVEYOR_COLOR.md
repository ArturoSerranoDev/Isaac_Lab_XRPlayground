# Conveyor Color Detection

UR10e + Robotiq 2F-85 sorts **target-colored** cubes (red / green / blue) from a conveyor:
- **Correct (target)** → side **SortTable** beside the robot
- **Reject (other colors)** → **TrashPlatform** at the end of the belt

## Training (Isaac)

Gym id: `Template-Xrplayground-Conveyor-Color-Direct-v0`

```bat
cd Isaac_XRPlayground
conda activate env_isaaclab
python -m scripts.launcher   REM [1] Train → Conveyor Color Detection
```

Or profile preset **Conveyor color** in the launcher `[P]` menu.

**MVP observations are state-based** (color ID one-hot + poses), not RGB camera. Vision can be added later.

### Scene

- UR10e + Robotiq 2F-85 (same asset Unity mirrors)
- Kinematic belt along +Y; cubes spawn randomly **on** the belt and are driven along +Y
- **SortTable** (green) beside **TrashPlatform** at the belt end — success drop for target color
- **TrashPlatform** (red-brown) at belt end — reject zone for non-target
- Robot base offset (−X/−Y) for arm clearance vs belt and end tables
- Pool of 4 reusable cubes; spawn every `spawn_interval_s`
- Target color sampled per episode
- Success: drop target cube on the sort table
- Fail: target falls, target reaches trash, or wrong color on sort table

## Bridge (Unity ↔ Isaac)

| | Ball Catch | Conveyor Color |
|--|------------|----------------|
| Port | **9090** | **9091** |
| Script | `scripts/bridge/run_xr_bridge.py` | `scripts/bridge/run_xr_bridge_conveyor.py` |
| Modes | `mirror` / `await_throw` | `mirror` / `await_spawn` |

### Topics (`/xr/conveyor/...`)

| Topic | Direction | Role |
|-------|-----------|------|
| `/xr/conveyor/robot_state` | Isaac → Unity | Links + EE + `target_color` |
| `/xr/conveyor/objects_state` | Isaac → Unity | Cube poses / colors / active |
| `/xr/conveyor/spawn` | Unity → Isaac | `{ color, position? }` activate slot |
| `/xr/session_command` | Unity → Isaac | `mirror` \| `await_spawn` |

### Isaac

Launcher **[B] XR Bridge** → pick **Conveyor Color Detection** → Mirror or Await spawn.

```bat
python scripts\bridge\run_xr_bridge_conveyor.py --num_envs=1 --mode=mirror --port=9091 --viz kit
```

Use `--checkpoint=...` for a trained policy. For XR with a headset, prefer `--no-real-time` (default in launcher now): Kit viewport + Unity VR on the same GPU already exceeds wall-clock step time, so `--real-time` only adds sleep and makes Isaac feel slower.

### Unity

1. Lab zones already set up (Kinova Station A stays).
2. Ensure the real Nucleus USD is under
   `Unity_XRPlayground/Assets/_Project/Features/Robots/UR10e/USD/`
   (`ur10e.usd` + `configuration/ur10e_base.usd`, same as Isaac).
   Repo mirror: `assets/usd/ur10e/`.
3. Menu **XRPlayground → Setup Conveyor Color Station**
   - Instantiates the USD (no cube placeholders)
   - `RobotLinkMap` + pose follower (env anchor = robot root)
   - World UI + bridge on **port 9091**
4. Optional: **XRPlayground → Audit UR10e Hierarchy**
5. Play → Connect → Mirror (or Await spawn + Spawn R/G/B)

## Files

| Area | Path |
|------|------|
| Env | `source/.../tasks/direct/conveyor_color/` |
| Bridge | `source/.../bridge/conveyor_color_bridge.py`, `names_ur10e.py` |
| Runner | `scripts/bridge/run_xr_bridge_conveyor.py` |
| Unity setup | `Assets/.../Editor/ConveyorColorSceneSetup.cs` |
| Link audit | `Assets/.../UR10e/Editor/Ur10eHierarchyAudit.cs` |
