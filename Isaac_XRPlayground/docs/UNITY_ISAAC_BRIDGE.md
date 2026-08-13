# Unity ↔ Isaac Kinova Bridge

TCP “ROS-like” bridge for XR playground ↔ Isaac Lab ball-catch.

## Authority

- **Isaac** owns physics and (optionally) the trained policy.
- **Unity** owns XR input / rendering and puppets the Kinova USD from streamed link poses.

## Hierarchy mapping

Same Nucleus USD (`j2n7s300_instanceable.usd`). **Link names match.**

| Role | Isaac | Unity |
|------|--------|--------|
| Root | Articulation `/World/envs/env_0/Robot` | `Kinova_Jaco2_j2n7s300` |
| Base | `j2n7s300_link_base` | same (flat sibling) |
| Arm links | `j2n7s300_link_1` … `_7` | same |
| EE body | `j2n7s300_end_effector` | same |
| Fingers | `j2n7s300_link_finger_*` / `_tip_*` | same |
| Joints | `j2n7s300_joint_1` … `_7` + 6 finger joints | **not present** as GameObjects |

Unity imports links as **flat siblings**. Drive visuals with **per-link poses**, not joint angles alone.

Unity menu **XRPlayground → Audit Kinova Hierarchy** compares scene names to this list.

## Topics (JSON envelope)

```json
{ "topic": "/xr/...", "stamp_s": 0.0, "frame_id": "isaac_env", "data": { } }
```

Framing on the wire: **4-byte little-endian length** + UTF-8 JSON.

| Topic | Direction | Data |
|-------|-----------|------|
| `/xr/ball_state` | Unity → Isaac | `position[3]`, `orientation_xyzw[4]`, `linear_velocity[3]`, `angular_velocity[3]`, `grasped` |
| `/xr/robot_state` | Isaac → Unity | `joint_names`, `joint_positions`, `ee`, `links[{name,position,orientation_xyzw}]` |
| `/xr/heartbeat` | both | `role`, time fields |

Positions/orientations in `/xr/*` payloads use **Isaac Z-up** frame (env-local for ball/robot relative to env origin). Quaternions on the wire are **(x, y, z, w)** — the same convention as current Isaac Lab `body_link_quat_w` / `write_root_pose_to_sim`.

Unity converts with `XrFrameConverter` (Z-up → Y-up) and places links in the `KinovaLinkPoseFollower.envAnchor` frame (usually the Kinova root at the Manipulation station). Optional `calibrateVisualFrames` can bake a per-link USD visual correction when Unity bind and Isaac pose share the same joint configuration (leave off by default — Isaac init pose ≠ USD import rest pose).

### Conveyor Color (second station)

See [CONVEYOR_COLOR.md](CONVEYOR_COLOR.md). Uses **port 9091** and topics under `/xr/conveyor/*`. Unity menu **Setup Conveyor Color Station** adds Station B without touching Kinova / Ball Catch.

## Session modes

| Mode | Unity | Isaac |
|------|--------|--------|
| **mirror** | Puppets robot + ball from Isaac | Owns throws + actions (policy / zero / random) |
| **await_throw** | Spawns grabable ball near Kinova; publishes pose while held; on release sends `throw_event` | Replicates Unity ball while waiting, then catches with policy |

Unity world UI (classic **Canvas / uGUI**, not UI Toolkit): **XRPlayground → Setup XR Bridge Scene**
creates `XR Bridge World UI` near the Kinova with Connect / Mirror / Await throw.

## Run

### Isaac via launcher (recommended)

From `Isaac_XRPlayground`, start the interactive launcher and choose **[B] XR Bridge → Unity**.

The wizard asks for:

1. Training task (defaults to Ball Catch)
2. **Session mode** — **Mirror Isaac** or **Await player throw**
3. Checkpoint (optional for mirror/debug; recommended for catch)
4. Host / port / real-time

Then connect from Unity with the matching mode button.

### Isaac (manual CLI)

```bat
cd Isaac_XRPlayground
conda activate env_isaaclab
python scripts\bridge\run_xr_bridge.py --task=Template-Xrplayground-Ball-Catch-Direct-v0 --num_envs=1 --mode=mirror --checkpoint=PATH\TO\agent.pt --viz kit
```

`--mode=await_throw` starts in wait-for-player mode. Unity UI can switch modes at runtime.
Omit `--checkpoint` only for mirror/debug; catch needs a trained policy.

**Performance:** Kit viewport + PhysX on an RTX 3060 Ti is heavy with 1 interactive env. The bridge defaults to `--real-time` (caps to `step_dt ≈ 1/60 s`). If the viewport already takes longer than that per step, the sim feels sluggish. Use `--no-real-time` to run as fast as rendering allows, or lower Kit render quality / close unused viewports.

### Unity

1. Open `SampleScene`.
2. Ensure `Kinova_Jaco2_j2n7s300` has `KinovaLinkMap` + `KinovaLinkPoseFollower`.
3. Scene object with `RosTcpClient` (host `127.0.0.1`, port `9090`).
4. Ball / Grab Cube has `BallStatePublisher`.
5. Enter Play Mode after Isaac bridge is listening.

Menu **XRPlayground → Setup XR Bridge Scene** wires defaults.

## Smoke checklist

1. Isaac alone: heartbeats + `/xr/robot_state` in console (`--log_robot`).
2. Unity connects: Kinova links **assemble and track** Isaac (same pose/rotation, not floating pieces).
3. Grab/release ball in Unity (await_throw): Isaac ball pose updates.

### Unity menus

- **XRPlayground → Audit Kinova Hierarchy** — expect 15 links, 0 missing.
- **XRPlayground → Setup XR Bridge Scene** — creates `XR Bridge`, attaches map/follower/publisher.
