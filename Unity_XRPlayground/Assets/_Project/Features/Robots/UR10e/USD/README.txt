# UR10e USD (same Nucleus asset as Isaac training)

Source:
`Assets/Isaac/6.0/Isaac/Robots/UniversalRobots/ur10e/`

## In this folder now

- `ur10e.usd` — thin root / variants (little/no mesh alone)
- `configuration/ur10e_base.usd` — **arm meshes + link names** (what Station B instantiates)

Unity setup uses `configuration/ur10e_base.usd` as the visual robot (35 MeshFilters).

Still optional for full Robotiq 2F-85 visuals:
- `configuration/ur10e_Gripper_2F_85.usd` + Robotiq/2F-85 tree from Nucleus

Repo mirror: `assets/usd/ur10e/`
