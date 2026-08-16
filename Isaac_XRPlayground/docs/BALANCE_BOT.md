# Balance Bot — Task station checklist

Goal: train a 2-DOF tilting tray to keep hand-sized ball(s) from falling, then reuse the policy in Unity XR (ROS mirror + offline ONNX).

## Identity

| Field | Value |
|-------|--------|
| Task name | Balance Bot (ball-on-plate) |
| Gym ID | `Template-Xrplayground-Balance-Bot-Direct-v0` |
| Isaac package path | `source/XRPlayground/XRPlayground/tasks/direct/balance_bot/` |
| Unity station name | `Station_D_BalanceBot` |
| Unity setup menu | `XRPlayground → Setup Balance Bot Station` |
| ROS TCP port | `9093` |
| Unity policy folder | `Assets/_Project/Features/Policies/BalanceBot/` |
| RSL-RL experiment name | `xrplayground_balance_bot_direct` |

## 1. Robot asset

| Field | Value |
|-------|--------|
| Isaac USD / asset cfg | Procedural kinematic cuboids (`base_link`, `tray_link`) in `balance_bot_env_cfg.py`; visual ref `assets/usd/balance_bot/balance_tray.usda` |
| Unity USD path | `Assets/_Project/Features/Robots/BalanceBot/USD/balance_tray.usda` (+ scene primitives from setup menu) |
| Source / Nucleus URL (if downloaded) | N/A — MVP tray, not a CMU ballbot |
| Default joint vector (Isaac, rad) | `[0.0, 0.0]` (roll, pitch) |
| Gripper open / close | N/A |

- [x] USD / named hierarchy present in **both** Isaac and Unity
- [x] Unity scene leaves tray at **default joint pose** after setup menu
- [x] Offline FK rest captured only at that default (never on every Start Policy)

## 2. Name table (links / joints)

| Role | Isaac / USD prim name | Unity map field |
|------|----------------------|-----------------|
| Base / pedestal | `base_link` | `BalanceBotLinkMap` |
| Roll gimbal | `roll_link` | same |
| Tray / plate | `tray_link` | same |
| Joints | `roll_joint`, `pitch_joint` | `OfflineJointDriver.BindBalanceBot` |

Shared code:

- Isaac: `bridge/names_balance_bot.py`
- Unity: `BalanceBotLinkMap.cs` + `OfflineJointDriver.BindBalanceBot`

- [x] Names taken from **real** prim / hierarchy names (not placeholders)
- [x] Bind assert: `3 / 3` expected links bound
- [x] Offline joint count matches policy action layout (`2`)
- [x] ROS pose follower uses the **same** names

## 3. Pose table (Isaac Z-up → Unity via XrFrameConverter)

| Object | Isaac (x, y, z) | Unity local (after converter) | Notes |
|--------|-----------------|-------------------------------|-------|
| Pedestal | `(0, 0, 0.35)` size `(0.12, 0.12, 0.70)` | `IsaacPosToUnity` + size `xzy` | Station at lab `(6, 0, -2)` |
| Tray center | `(0, 0, 0.75)` size `(0.60, 0.60, 0.02)` | same | Level at default pose |
| Ball spawn | tray center + ~0.08 m above surface | slots `Ball_0` / `Ball_1` | Radius `0.035` m, mass `0.057` kg (dense) |

- [x] Layout makes sense for the skill
- [x] Unity setup menu writes the **same** numbers as Isaac cfg
- [x] Only `XrFrameConverter` used for frame changes

## 4. Obs / action contract

| Field | Isaac | Unity | `policy.json` |
|-------|-------|-------|---------------|
| `obs_dim` | **20** | **20** | **20** |
| `action_dim` | 2 | 2 | 2 |
| `action_scale` | 1.5 | 1.5 | 1.5 |
| Control `dt` | 1/60 | 1/60 | ~0.01667 |
| Obs layout | roll, pitch, droll·s, dpitch·s, ball0_rel(3), ball0_vel(3), ball0_act, ball1_rel(3), ball1_vel(3), ball1_act, n_balls, curriculum_stage | same | same |

Curriculum (fixed 2026-08-15):

- `common_step_counter` advances **once per env.step** (not summed across envs). With `num_steps_per_env=16`, 1000 iters ≈ **16k** steps — old `curriculum_steps=80000` never unlocked stage 1.
- Now: `curriculum_steps=10000`, hold-EMA unlock (`curriculum_unlock_hold_steps=350`, patience 40), and `curriculum_start_stage` (set **1** after proven 1-ball / for 2-ball resume; **0** for cold start).
- Stage 0 = **1 ball**, stage 1 = **2 balls**.

Metrics: `hold_rate`, `drop_rate`, `n_balls`, `curriculum_stage`, `mean_hold_steps`, `hold_ema`. Prefer **episode length / drop_rate** for 2-ball — dual-ball near-center gate keeps `mean_hold_steps` at 0 even when episodes complete.

- [x] Controller `ObsDim` / `ActionDim` match training (**20** / 2)
- [x] Export + Unity copy use TorchScript / Unity-safe ONNX (2-ball run `2026-08-15_17-16-17`)
- [x] Changing obs/layout/curriculum updates Unity **in the same** change

## 5. ROS mirror

| Field | Value |
|-------|--------|
| Bridge script | `scripts/bridge/run_xr_bridge_balance_bot.py` |
| Topics (robot / balls) | `/xr/balance_bot/robot_state`, `/xr/balance_bot/balls_state` |
| Unity client / followers | `BalanceBotLinkPoseFollower`, `BalanceBotBallFollower` |

- [ ] Connect → links track Isaac within visual tolerance (smoke when GPU free)
- [ ] Ball topics match frames above

## 6. Offline ONNX

| Field | Value |
|-------|--------|
| Controller | `BalanceBotOfflinePolicyController.cs` |
| Joint driver bind method | `BindBalanceBot` |
| Model asset path | `Policies/BalanceBot/policy.onnx` |

- [ ] Start from **default pose** — tray motion resembles Isaac `play.py` (2-ball ONNX ready; Unity smoke pending)
- [x] FK rest **not** re-captured every start (`force: false` unless unbound)
- [x] `policy.json` present (checkpoint `model_1998.pt`, 2-ball; set `curriculumStage=1`)

## 7. Verification order (do in order)

1. [x] Setup menu / docs create station with tray hierarchy + maps + poses
2. [ ] ROS mirror smoke (~10 s)
3. [ ] Offline ONNX smoke from default pose (2 balls)
4. [x] 1-ball train + ONNX (`2026-08-15_17-00-25`)
5. [x] Curriculum fix + 2-ball resume train + Unity overwrite (`2026-08-15_17-16-17`)
6. [x] This doc filled


## First train results (2026-08-15) — 1 ball

| Field | Value |
|-------|--------|
| Run dir | `logs/rsl_rl/xrplayground_balance_bot_direct/2026-08-15_17-00-25` |
| Config | `num_envs=256`, `max_iterations=1000`, `--export_onnx` |
| Final ckpt | `model_999.pt` |
| `Metrics/mean_hold_steps` | **~470** (episode len ~479) |
| `Train/mean_reward` | **~16 → ~1836** |
| `Metrics/n_balls` / `curriculum_stage` | stayed **1 / 0** (bug: `curriculum_steps=80000` vs ~16k common steps) |

## 2-ball resume (2026-08-15)

| Field | Value |
|-------|--------|
| Run dir | `logs/rsl_rl/xrplayground_balance_bot_direct/2026-08-15_17-16-17` |
| Config | resume `model_999.pt`, `num_envs=128`, iters **999→1998**, `curriculum_start_stage=1` |
| Final ckpt | `model_1998.pt` (best-by-reward window ≈ `model_1650.pt`) |
| Unity assets | `Policies/BalanceBot/policy.onnx` + `policy.json` (**obs_dim=20**, overwritten) |
| `Metrics/n_balls` / `curriculum_stage` | **2.0 / 1.0** entire run |
| `Train/mean_reward` (last50) | **~1281** (start ~25 → ~1281) |
| `Train/mean_episode_length` (last50) | **479** (full episode) |
| `Metrics/hold_rate` / `drop_rate` (last50) | **1.0 / 0.0** |
| `Metrics/mean_hold_steps` | **0** (metric gate too strict for 2 balls; ignore) |

## Train commands

1-ball / cold start (`curriculum_start_stage=0` in cfg first):

```bat
cd Isaac_XRPlayground
C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe scripts\rsl_rl\train.py --task=Template-Xrplayground-Balance-Bot-Direct-v0 --num_envs=256 --viz none --max_iterations=1000 --export_onnx
```

2-ball resume (current cfg forces stage 1):

```bat
python scripts\rsl_rl\train.py --task=Template-Xrplayground-Balance-Bot-Direct-v0 --num_envs=128 --viz none --max_iterations=1000 --export_onnx --resume --load_run=2026-08-15_17-00-25 --checkpoint=model_999.pt
```

Smoke (few iters / env create):

```bat
python scripts\random_agent.py --task=Template-Xrplayground-Balance-Bot-Direct-v0 --num_envs=4 --viz kit --real-time
```

Bridge:

```bat
python scripts\bridge\run_xr_bridge_balance_bot.py --num_envs=1 --mode=mirror
```

Launcher: Profiles → `balance`; task key `balance_bot`.

## Lessons / do-not-repeat

- Prefer ball-on-plate tray MVP over blocking on a full ballbot USD
- Quaternions in Isaac Lab asset init / buffers are **(x, y, z, w)**
- Do not re-capture offline FK rest on every Start Policy
- Curriculum thresholds must use **common_step_counter** scale (~`max_iters * num_steps_per_env`), not total transitions across envs
- Obs is **20-D** (was mislabeled 18 in early docs); Unity `ObsDim` must match ONNX input

## References

- Standards rule: `.cursor/rules/isaac-unity-training-station.mdc`
- ONNX: `docs/ONNX_UNITY.md`
- Bridge: `docs/UNITY_ISAAC_BRIDGE.md`
- Network viz: `Unity_XRPlayground/Assets/_Project/Features/Policies/NETWORK_VIZ.md`
- Examples: `BALL_CATCH.md`, `CONVEYOR_COLOR.md`, `PICK_PLACE_TABLE.md`
