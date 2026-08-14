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
- **Ball:** ~8.25 cm diameter sphere (radius 0.04125), ~80 g, spawned with a randomized toss toward the arm
- **Actions:** 8-D joint position deltas (7 arm + 1 shared gripper command)
- **Observations:** arm/gripper state, ball pose/velocity, gripper→ball vector
- **Success:** ball in the finger aperture along the EE→tip_center grasp axis (along/radial bounds + tip proximity + finger close + alignment); cupping on the dorsal gripper fails the episode
- **Observations:** arm/gripper state, ball pose/velocity, vectors EE→ball and tip_center→ball (28-D)
- **Checkpoints:** train from scratch after grasp/obs/reward changes — old policies will not match (cupping was previously rewardable)

### Launcher

1. **Profiles → `catch`** (128 headless envs, 8 visual, 2000 iters)
2. **Train → Ball Catch (Kinova Jaco2)** with **Export ONNX = Y** (RSL-RL)
3. Start with **8 visual envs**, Fabric + `cuda:0`
4. **[E] Export ONNX** anytime from a checkpoint; Unity: assign `Policies/BallCatch/policy.onnx` → **Start Offline Policy**

See [`ONNX_UNITY.md`](ONNX_UNITY.md).

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
3. Unity side: on "ball release", send `(position, linear_velocity)` to Isaac via the TCP ROS-like bridge (see [`UNITY_ISAAC_BRIDGE.md`](UNITY_ISAAC_BRIDGE.md)).
4. Optional: train with domain randomization on throw parameters so the policy generalizes to human throws.

**Bridge (Phase 4 MVP):** Launcher **[B] XR Bridge → Unity** (pick **Mirror Isaac** or **Await throw**), or `python scripts/bridge/run_xr_bridge.py --task=Template-Xrplayground-Ball-Catch-Direct-v0 --num_envs=1 --mode=mirror`. Then Unity **XRPlayground → Setup XR Bridge Scene**.

## Files to edit first

| File | Purpose |
|------|---------|
| `ball_catch_env_cfg.py` | throw ranges, robot choice, reward scales, `num_envs` |
| `ball_catch_env.py` | `_launch_ball()`, success criteria, observations |
| `agents/skrl_ppo_cfg.yaml` | network size, rollouts, learning rate |

## Shared assets

Place custom ball USD/meshes under monorepo `assets/ball_catch/` when you outgrow the procedural sphere.
