# Agibot A2D USD for Unity

Source (Isaac Lab Nucleus / S3):

```
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Agibot/A2D/A2D_physics.usd
```

Also reachable as:

```
IsaacLab/Nucleus/Robots/Agibot/A2D/A2D_physics.usd
```

Copy into this repo:

```
Unity_XRPlayground/Assets/_Project/Features/Robots/AgibotA2D/USD/A2D_physics.usd
assets/usd/agibot/A2D/A2D_physics.usd
```

After import in Unity, run **XRPlayground → Setup Pick Place Table Station** (or **Rebuild Link Map** on `AgibotLinkMap`).

## Body names (must match Isaac)

USD prim / Isaac `body_names` (not URDF-style aliases):

| Role | Prim name |
|------|-----------|
| Base | `base_link` |
| Lift / pitch | `link_up_down_body`, `link_pitch_body`, `link_arm` |
| Head | `link_yaw_head`, `link_pitch_head` |
| Right arm | `base_link_r`, `Link1_r` … `Link7_r` |
| Gripper | `right_base_link`, `right_gripper_center`, `right_Left_Pad_Link`, `right_Right_Pad_Link` |

Joints: `right_arm_joint1`…`7`, `right_hand_joint1` (all arm revolutes axis **Z** in Isaac).

Wall-mounted pose in Unity matches Isaac: robot root at Isaac `(0, -0.78, 0)` → Unity via `XrFrameConverter.IsaacPosToUnity`.
