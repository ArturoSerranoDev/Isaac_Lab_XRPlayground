# Task station checklist — TEMPLATE

Copy this file to `Isaac_XRPlayground/docs/<TASK_NAME>.md` and fill every section.
A training is **not done** until the verification checkboxes pass.

## Identity

| Field | Value |
|-------|--------|
| Task name | |
| Gym ID | `Template-Xrplayground-…-Direct-v0` |
| Isaac package path | `source/XRPlayground/.../tasks/direct/<task>/` |
| Unity station name | `Station_…` |
| Unity setup menu | `XRPlayground → …` |
| ROS TCP port | |
| Unity policy folder | `Assets/_Project/Features/Policies/<Name>/` |
| RSL-RL experiment name | |

## 1. Robot asset

| Field | Value |
|-------|--------|
| Isaac USD / asset cfg | |
| Unity USD path | `Assets/_Project/Features/Robots/…/USD/….usd` |
| Source / Nucleus URL (if downloaded) | |
| Default joint vector (Isaac, rad) | `[…]` |
| Gripper open / close | |

- [ ] USD present in **both** Isaac and Unity
- [ ] Unity scene leaves robot at **default joint pose** after setup menu
- [ ] Offline FK rest captured only at that default (never on every Start Policy)

## 2. Name table (links / joints)

| Role | Isaac / USD prim name | Unity map field |
|------|----------------------|-----------------|
| Base | | |
| … | | |
| EE | | |
| Gripper pads / tips | | |

Shared code:

- Isaac: `bridge/names_<robot>.py` (or task cfg body names)
- Unity: `*LinkMap.cs` + `OfflineJointDriver.Bind…`

- [ ] Names taken from **real USD**, not placeholders
- [ ] Bind assert: `__ / __` expected links bound
- [ ] Offline joint count matches policy action layout
- [ ] ROS pose follower uses the **same** names

## 3. Pose table (Isaac Z-up → Unity via XrFrameConverter)

| Object | Isaac (x, y, z) | Unity local (after converter) | Notes |
|--------|-----------------|-------------------------------|-------|
| Robot base | | | Forward axis: |
| Table / props | | | Centered in front of robot? |
| Place target / basket | | | |
| Piece / ball spawn | | | |

- [ ] Layout makes sense for the skill (not sideways / missing targets)
- [ ] Unity setup menu writes the **same** numbers as Isaac cfg
- [ ] Only `XrFrameConverter` used for frame changes

## 4. Obs / action contract

| Field | Isaac | Unity | `policy.json` |
|-------|-------|-------|---------------|
| `obs_dim` | | | |
| `action_dim` | | | |
| `action_scale` | | | |
| Control `dt` | | | |
| Obs layout (ordered list) | | | |

- [ ] Controller `ObsDim` / `ActionDim` match training
- [ ] Export + Unity copy use TorchScript / Unity-safe ONNX (see `ONNX_UNITY.md`)
- [ ] Changing obs/layout/curriculum updates Unity **in the same change**

## 5. ROS mirror

| Field | Value |
|-------|--------|
| Bridge script | `scripts/bridge/run_xr_bridge_….py` |
| Topics (robot / object / demo) | |
| Unity client / followers | |

- [ ] Connect → links track Isaac within visual tolerance
- [ ] Object / ball / piece topics match frames above

## 6. Offline ONNX

| Field | Value |
|-------|--------|
| Controller | `…OfflinePolicyController.cs` |
| Joint driver bind method | `BindKinova` / `BindUr10e` / `BindAgibotA2D` |
| Model asset path | `Policies/<Name>/policy.onnx` |

- [ ] Start from **default pose** → arm motion resembles Isaac `play.py`
- [ ] FK rest **not** re-captured every start (rebuild only via explicit menu)
- [ ] `policy.json` `checkpoint` points at the intended `.pt`

## 7. Verification order (do in order)

1. [ ] Setup menu / docs create station with robot USD + maps + poses
2. [ ] ROS mirror smoke (~10 s)
3. [ ] Offline ONNX smoke from default pose
4. [ ] Train / export only after 2–3 pass
5. [ ] This doc filled and linked from any launcher profile notes

## Lessons / do-not-repeat

- Wrong USD link names (placeholders) → silent bad maps  
- Re-capturing offline FK rest mid-motion → Unity arm freezes vs ONNX commands  
- Table/robot pose mismatch Isaac forward axis → “sideways” stations  
- Obs dim bump (e.g. 28→30) without Unity controller update → garbage actions  

## References

- Standards rule: `.cursor/rules/isaac-unity-training-station.mdc`
- ONNX: `docs/ONNX_UNITY.md`
- Bridge: `docs/UNITY_ISAAC_BRIDGE.md`
- Examples: `BALL_CATCH.md`, `CONVEYOR_COLOR.md`, `PICK_PLACE_TABLE.md`
