# Ball Catch — Sim Training → XR Play

Goal: train a manipulator to catch thrown balls in Isaac Lab, then reuse the policy when a real player throws in XR.

## Phase map

| Phase | Where | What |
|-------|--------|------|
| **1 — Sim catch (now)** | `ball_catch/` task | Two-phase PPO: Wrap → Throw |
| **2 — Better physics** | Same task | Tune throw ranges, drag, gripper friction |
| **3 — Human throw proxy** | Isaac + input device | Map controller/hand pose → `_launch_ball()` pose & velocity |
| **4 — XR playground** | `Unity_XRPlayground/` | Player throw in VR; stream ball spawn state to Isaac (or Unity physics + policy ONNX) |
| **5 — Live policy** | Play mode / custom script | Load checkpoint; each XR throw calls env reset with external throw params |

## Success definition (no cheats)

**Primary metric: `Metrics/soft_grasp`** (Wrap phase)

An episode counts as soft_grasp success only when **all** of:

1. Ball is **enclosed** in the finger aperture (mid-way along EE→tip axis, low radial error)
2. **All three fingertips** are near the ball (`tip_max < success_tip_max`), not tip-center mean alone
3. Gripper facing the ball (`grasp_align`), closing, settled speed, wrap symmetry
4. Soft_grasp held for `grasp_hold_steps` (≥ 8 control steps)
5. Ball **never dropped** this episode, and still soft_grasp on the last step

**Throw phase gate: `Metrics/throw_soft_grasp`** — same criteria, but only on throw-spawned episodes (in-hand excluded). Copy ONNX to Unity only when this sustains ≥ ~0.8–0.95.

Not success: tip-platform balance, poke with one finger, dorsal/forearm cup, brief brush-by, magnet-assisted latch.

`Metrics/catch_rate` is an **alias of soft_grasp** (same number).

## Two-phase training (current)

| Phase | Gym ID | In-hand | Goal | PPO |
|-------|--------|---------|------|-----|
| **Wrap** | `Template-Xrplayground-Ball-Catch-Wrap-v0` | `in_hand_spawn_p ≈ 1` | soft_grasp last100 ≥ ~0.9 | low entropy |
| **Throw-A** | `…-Throw-A-v0` | fades; **50% drift** | `throw_rolling` ≥ ~0.75 | low entropy; **arm frozen** |
| **Throw-B** | `…-Throw-B-v0` | fades → 0 | `throw_rolling` ≥ ~0.8–0.95 | higher entropy; **free arm** |

Legacy: `…-Throw-v0` aliases Throw-B. Mixed: `…-Direct-v0`.

### Pipeline (recommended)

```bat
cd Isaac_XRPlayground
C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe scripts\rsl_rl\train_ball_catch_throw_pipeline.py
```

Resumes wrap `2026-08-16_06-05-50/model_5449.pt` → Throw-A → Throw-B; copies Unity only if B gate passes.

### No-cheat curriculum

| Mechanism | What it does | Fade / gate |
|-----------|--------------|-------------|
| Throw difficulty | Easy near-hand lobs → wider reachable throws | `curriculum_cap` rises when **throw_soft_grasp EMA** ≥ unlock (~0.40) |
| In-hand spawn | Wrap forces ~1.0; Throw **fades** (no cliff to 0) | Mixed fades by `spawn_in_hand_until` |
| Finger pre-close | **In-hand resets only** (throw lobs stay open) | Same as in-hand episodes |
| Rewards | Dense approach small; **vel-match near contact**; intercept enclose bonus; hold + end_hold; drop-after-latch penalty | No catch assist |
| Ball buoyancy | `ball_gravity_scale=0.55` | **Fixed** — Unity-matched (not a catch magnet) |
| Grip friction | Ball static/dynamic friction raised | Still no magnet assist |
| Catch assist | Spring / pose blend / finger drive | **Disabled** |

Forbidden: permanent catch assist, counting near-ball as catch, soft_grasp_catch ≠ soft_grasp.

Terminations: **drop or timeout only** — never reset on latch.

### Throw transfer (why the first Throw run failed)

Wrap freezes the arm and practices enclose only. Resuming with **forced `in_hand_p=0`** was a cliff: arm never learned intercept → latch-then-drop (`throw_soft_grasp` ≈ 0.02).

Throw now (soft handoff):
1. Starts mostly in-hand + **in-aperture drift** (small residual speed; arm frozen in drift band)
2. Relaxed enclose gates on in-hand/drift (same as Wrap) so wrap skill transfers
3. Curriculum advances inside the drift band on `throw_catch_ema`; **leaves drift only when EMA ≥ ~0.55**
4. Then parabolic throws with free arm; in-hand fades to 0
5. Milder post-latch gripper close; open gripper on true lobs
6. Higher PPO entropy / lr so the arm can re-learn after wrap freeze

## Current task contract

- **Robot:** Kinova Jaco2 7-DoF + 3-finger gripper (open 0.04, wrap-close 1.10)
- **Ball:** ~8.25 cm diameter; parabolic tosses aimed at cup volume
- **Actions:** 8-D joint position deltas (7 arm + 1 gripper); `action_scale=5.0`
- **Observations:** **30-D** (unchanged) — arm/gripper state, ball pose/vel, EE→ball, tip_center→ball, `grasp_align`, `ball_radial`
- **Do not** resume pre-redesign failed runs. Start Wrap fresh; Throw resumes Wrap only.

### Train — Phase Wrap

```bat
cd Isaac_XRPlayground
C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe scripts\rsl_rl\train.py --task=Template-Xrplayground-Ball-Catch-Wrap-v0 --num_envs=64 --viz none --max_iterations=1200 --export_onnx --no_copy_to_unity
```

Gate Wrap → Throw: `Metrics/soft_grasp` last100 ≥ ~0.85–0.9.

### Train — Phase Throw (resume Wrap)

```bat
cd Isaac_XRPlayground
C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe scripts\rsl_rl\train.py --task=Template-Xrplayground-Ball-Catch-Throw-v0 --num_envs=64 --viz none --max_iterations=3000 --resume --load_run=2026-08-16_06-05-50 --checkpoint=model_5449.pt --export_onnx --no_copy_to_unity
```

Gate Throw → Unity: `Metrics/throw_soft_grasp` / `throw_catch_ema` ≥ ~0.8–0.95, then export **without** `--no_copy_to_unity` (or copy `exported/` into `Policies/BallCatch/`).

Watch during Throw: `in_hand_spawn_p` should fall as `throw_catch_ema` rises; if `soft_grasp` is high but `throw_soft_grasp` stays ~0, curriculum is stuck (expected until near-lobs start sticking).

### Smoke test

```bat
cd Isaac_XRPlayground
python scripts\random_agent.py --task=Template-Xrplayground-Ball-Catch-Wrap-v0 --num_envs=4 --viz kit --real-time
```

### Launcher

1. Profiles → `catch`
2. Train → **Ball Catch — Phase Wrap** then **Phase Throw** (RSL-RL, Export ONNX)
3. Unity: assign `Policies/BallCatch/policy.onnx` → **Start Offline Policy** only after Throw gate

See [`ONNX_UNITY.md`](ONNX_UNITY.md).

## XR integration (later)

The throw is isolated in `BallCatchEnv._launch_ball()`. For XR:

1. Replace random throw sampling with values from the XR client (hand release pose + velocity).
2. Keep observations/actions identical so the trained policy still applies.
3. Unity side: on "ball release", send `(position, linear_velocity)` to Isaac via the TCP ROS-like bridge (see [`UNITY_ISAAC_BRIDGE.md`](UNITY_ISAAC_BRIDGE.md)).
4. Optional: train with domain randomization on throw parameters so the policy generalizes to human throws.

**Bridge (Phase 4 MVP):** Launcher **[B] XR Bridge → Unity**, or `python scripts/bridge/run_xr_bridge.py --task=Template-Xrplayground-Ball-Catch-Direct-v0 --num_envs=1 --mode=mirror`.

**Network viz (Unity):** **XRPlayground → Setup Policy Network Panels (All Stations)**. See `NETWORK_VIZ.md`.

## Files

| File | Purpose |
|------|---------|
| `ball_catch_env_cfg.py` | Wrap/Throw cfgs, rewards, friction, curriculum |
| `ball_catch_env.py` | launch, soft_grasp, vel-match, intercept bonus, metrics |
| `agents/rsl_rl_ppo_cfg.py` | Wrap (high entropy) / Throw (low entropy) PPO |

## Shared assets

Place custom ball USD/meshes under monorepo `assets/ball_catch/` when you outgrow the procedural sphere.

## Training log (2026-08-16 two-phase run)

| Phase | Run | Gate | Result |
|-------|-----|------|--------|
| Wrap | `logs/rsl_rl/xrplayground_ball_catch_2phase/2026-08-16_06-05-50` (`model_5449.pt`) | soft_grasp last100 ≥ ~0.85–0.9 | **PASS** — catch_ema ≈ 0.94, last100 soft_grasp ≈ 0.89–0.94 |
| Throw | `.../2026-08-16_06-49-22` (resume wrap `model_5449.pt`, 2000 iters) | throw_soft_grasp ≥ ~0.8–0.95 | **FAIL** — catch_ema ≈ 0.02; latch-then-drop; cliff `in_hand_p=0` |
| Unity | `Policies/BallCatch/` | copy only on throw gate | **Not updated** |

### Fix applied (2026-08-16 afternoon / evening)

Soft Throw handoff → **Throw-A / Throw-B** pipeline:

| Piece | Status |
|-------|--------|
| Gym IDs `…-Throw-A-v0` / `…-Throw-B-v0` | Done |
| Rolling `throw_rolling` gate + drift ramp | Done |
| Throw-A success truncation after sustained latch | Done |
| Pipeline script `train_ball_catch_throw_pipeline.py` | Done |
| Throw-A full train | **Done** `2026-08-16_19-32-35` — last100 `throw_rolling` ≈ 0.46 (peak ≈ 0.57); below 0.65 gate but usable |
| Throw-B + Unity | **FAIL** `2026-08-16_21-29-16` from A `model_8250.pt` (~3500 iters). Peak `throw_rolling` ≈ 0.64 @ ~8436; last100 ≈ 0.35; final ≈ 0.07. Unity **not** copied (gate ≥ ~0.8) |

Spot Follow was stopped for GPU: resume `xrplayground_spot_follow/2026-08-16_00-26-37` `model_1200.pt` (see `logs/spot_follow_resume_2026-08-16.txt`).
