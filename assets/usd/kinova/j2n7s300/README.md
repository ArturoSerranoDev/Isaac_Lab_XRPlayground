# Kinova Jaco2 (j2n7s300) USD

Isaac Lab training robot (`KINOVA_JACO2_N7S300_CFG`).

Source (Isaac 6.0 Nucleus mirror):
`Assets/Isaac/6.0/Isaac/Robots/Kinova/Jaco2/J2N7S300/`

## Layout

- `j2n7s300_instanceable.usd` — articulation root (same path Isaac Lab uses)
- `Props/instanceable_meshes.usd` — mesh payloads referenced by the root
- `configuration/j2n7s300_instanceable_robot_schema.usd` — robot schema

Unity copy lives at:
`Unity_XRPlayground/Assets/_Project/Features/Robots/KinovaJaco2/USD/`

In Unity, mark the root USD as **Import** (`isUsdRoot`) and assign the default USD Importer Graph
(`Packages/com.unity.importer.usd/.../usdImporter.asset`). Menu **XRPlayground → Setup Lab Zones**
does this automatically when placing the robot.
