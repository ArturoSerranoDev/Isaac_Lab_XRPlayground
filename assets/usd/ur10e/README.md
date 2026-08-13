# UR10e (Universal Robots) USD

Isaac Lab training robot (`UR10e_ROBOTIQ_2F_85_CFG`).

Source (Isaac 6.0 Nucleus / S3 mirror):
`Assets/Isaac/6.0/Isaac/Robots/UniversalRobots/ur10e/`

## Layout

- `ur10e.usd` — articulation root (same path Isaac Lab uses)
- `configuration/ur10e_base.usd` — mesh payload
- `configuration/ur10e_physics.usd`, `ur10e_sensor.usd`, `ur10e_robot.usd`
- `configuration/ur10e_Gripper_2F_85.usd` — Robotiq 2F-85 variant

Unity copy:
`Unity_XRPlayground/Assets/_Project/Features/Robots/UR10e/USD/`

Gripper meshes may also need sibling Robotiq assets from:
`Assets/Isaac/6.0/Isaac/Robots/Robotiq/2F-85/`
