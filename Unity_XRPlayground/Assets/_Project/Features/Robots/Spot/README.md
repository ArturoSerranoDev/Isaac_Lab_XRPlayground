# Spot (Boston Dynamics)

Shared USD + name table for Isaac ↔ Unity XR station **Station_E_Spot**.

## Shared link / joint names

| Role | Name |
|------|------|
| Base | `body` |
| Front-left chain | `fl_hip`, `fl_uleg`, `fl_lleg`, `fl_foot` |
| Front-right | `fr_*` |
| Hind-left | `hl_*` |
| Hind-right | `hr_*` |
| Joints (12) | `fl_hx`, `fl_hy`, `fl_kn`, … `hr_kn` |

Default standing pose (rad, matches `isaaclab_assets.robots.spot.SPOT_CFG`):

```
[0.1, 0.9, -1.5, -0.1, 0.9, -1.5, 0.1, 1.1, -1.5, -0.1, 1.1, -1.5]
```

## Assets

| Side | Path |
|------|------|
| Monorepo cache | `assets/usd/spot/spot.usd` |
| Isaac project | `Isaac_XRPlayground/assets/usd/spot/spot.usd` |
| Unity | `Assets/_Project/Features/Robots/Spot/USD/spot.usd` |
| Nucleus | `Robots/BostonDynamics/spot/spot.usd` |

Isaac training still uses Nucleus via `SPOT_CFG` when the local USD is missing schema refs (`configuration/spot_robot_schema.usd`).

## Unity

**XRPlayground → Setup Spot Follow Station** builds a **procedural** hierarchy under `02_Locomotion/Robot Stations/Station_E_Spot` with the real USD prim names.

- Nucleus `spot.usd` is present under `USD/` but Unity import has no usable GameObject mesh (schema/refs incomplete) — do **not** expect a imported USD mesh in the Scene.
- Setup uses unit-scale joint pivots (`body`, `fl_hip`, …) plus `*_viz` cube children so nested scales stay visible.
- Bind assert: **17/17** links on `SpotLinkMap`.
- Bridge port **9094**; offline: `SpotOfflinePolicyController` + `OfflineJointDriver.BindSpot`.

Re-run the menu anytime to recreate the station cleanly, then save SampleScene.
