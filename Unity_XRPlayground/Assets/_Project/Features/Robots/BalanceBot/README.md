# Balance Bot tray asset

Procedural 2-DOF ball-on-plate tray (not a full CMU ballbot).

## Shared link / joint names

| Role | Name |
|------|------|
| Pedestal | `base_link` |
| Roll gimbal | `roll_link` |
| Plate | `tray_link` |
| Joints | `roll_joint` (about +X), `pitch_joint` (about +Y) |

Default pose: `[roll, pitch] = [0, 0]` (level tray).

## Isaac

Spawned procedurally in `balance_bot_env_cfg.py` as kinematic cuboids (`base_link`, `tray_link`) plus virtual `roll_link` poses from the bridge. Balls: `Ball_0`, `Ball_1` (radius 0.035 m, mass 0.057 kg).

## Unity

Built by **XRPlayground → Setup Balance Bot Station** as a Transform hierarchy under `BalanceBot_Tray` (primitives). Optional future: author a USD here and import like UR10e.

## Layout (Isaac Z-up)

| Object | Position | Size |
|--------|----------|------|
| Pedestal | `(0, 0, 0.35)` | `(0.12, 0.12, 0.70)` |
| Tray center | `(0, 0, 0.75)` | `(0.60, 0.60, 0.02)` |
