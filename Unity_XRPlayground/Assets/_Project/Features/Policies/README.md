# Unity offline policies (ONNX)

Trained **RSL-RL** policies export here as `policy.onnx` + `policy.json`.

| Folder | Task | Obs | Actions |
|--------|------|-----|---------|
| `Conveyor/` | Conveyor Color (UR10e) | 66 | 7 |
| `BallCatch/` | Ball Catch (Kinova) | 28 | 8 |

## Workflow

1. Isaac launcher: train **Conveyor** or **Ball Catch** with **Export ONNX after train/stop** (Y).
2. Or launcher **[E] Export ONNX** from an existing RSL-RL checkpoint.
3. Unity imports `policy.onnx` as an Inference Engine **ModelAsset**.
4. Drag it onto `OnnxPolicyRunner.modelAsset` on the station/robot.
5. Play → **Start Offline Policy** on the world UI (uses cube slots / ball refs already wired).

Package: local `Packages/com.unity.ai.inference` (embedded 2.3.0, patched for Unity 6.5 `EntityId`). Offline loops are kinematic approximations — good for demos, not bit-identical PhysX.
