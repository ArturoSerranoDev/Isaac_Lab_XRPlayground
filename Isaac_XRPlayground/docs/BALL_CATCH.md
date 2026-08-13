# Ball Catch — Sim Training → XR Play

Goal: train a manipulator to catch thrown balls in Isaac Lab, then reuse the policy when a real player throws in XR.

## Phase map

| Phase | Where | What |
|-------|--------|------|
| **1 — Sim catch (now)** | `ball_catch/` task | Scripted random throws → PPO learns intercept + close gripper |
| **2 — Better physics** | Same task | Tune throw ranges, drag, gripper friction |
| **3 — Human throw proxy** | Isaac + input device | Map controller/hand pose → `_launch_ball()` pose & velocity |
| **4 — XR playground** | `Unity_XRPlayground/` | Player throw in VR; stream ball spawn state to Isaac (or Unity physics + policy ONNX) |
| **5 — Live policy** | Play mode / custom script | Load checkpoint; each XR throw calls env reset with external throw params |

## Current task (Phase 1)

- **Gym ID:** `Template-Xrplayground-Ball-Catch-Direct-v0`
- **Robot:** Kinova Jaco2 7-DoF + 3-finger gripper (instanceable USD; open 0.2, close 1.2)
- **Ball:** 8 cm sphere, ~57 g, spawned with a randomized toss toward the arm
- **Actions:** 8-D joint position deltas (7 arm + 1 shared gripper command)
- **Observations:** arm/gripper state, ball pose/velocity, gripper→ball vector
- **Success:** ball near closing gripper, low speed, not on ground
- **Checkpoints:** UR10e / Franka runs will not load — train a new policy after this robot swap

### Launcher

1. **Profiles → `catch`** (128 headless envs, 8 visual, 2000 iters)
2. **Train → Ball Catch (Kinova Jaco2)**
3. Start with **8 visual envs**, Fabric + `cuda:0`
4. **Play** with `--real-time` once checkpoints exist

### Smoke test (CLI)

```bat
cd Isaac_XRPlayground
conda activate env_isaaclab
python scripts\random_agent.py --task=Template-Xrplayground-Ball-Catch-Direct-v0 --num_envs=4 --viz kit --real-time
```

## XR integration (later)

The throw is isolated in `BallCatchEnv._launch_ball()`. For XR:

1. Replace random `throw_pos_*` / `throw_vel_*` sampling with values from the XR client (hand release pose + velocity).
2. Keep observations/actions identical so the trained policy still applies.
3. Unity side: on "ball release", send `(position, linear_velocity)` to Isaac via socket/ROS/shared memory.
4. Optional: train with domain randomization on throw parameters so the policy generalizes to human throws.

## Files to edit first

| File | Purpose |
|------|---------|
| `ball_catch_env_cfg.py` | throw ranges, robot choice, reward scales, `num_envs` |
| `ball_catch_env.py` | `_launch_ball()`, success criteria, observations |
| `agents/skrl_ppo_cfg.yaml` | network size, rollouts, learning rate |

## Shared assets

Place custom ball USD/meshes under monorepo `assets/ball_catch/` when you outgrow the procedural sphere.
