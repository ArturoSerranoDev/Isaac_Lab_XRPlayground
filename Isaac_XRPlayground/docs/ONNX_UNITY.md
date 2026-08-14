# RSL-RL → ONNX → Unity (offline)

Train once in Isaac Lab with **RSL-RL**, export `policy.onnx`, run in Unity with **Inference Engine** — no Isaac at play time.

## Train + export

Launcher (`python -m scripts.launcher`):

1. Profile **Conveyor color** or **Ball catch**.
2. **[1] Train** → answer **Y** to *Export ONNX after train/stop*.
3. Stop with Ctrl+C or let training finish — writes:
   - `Isaac_XRPlayground/logs/rsl_rl/<experiment>/<run>/exported/policy.onnx`
   - Copy → `Unity_XRPlayground/Assets/_Project/Features/Policies/{Conveyor|BallCatch}/`

Or later: **[E] Export ONNX** from a checkpoint.

CLI:

```bat
cd Isaac_XRPlayground
python scripts\rsl_rl\train.py --task=Template-Xrplayground-Conveyor-Color-Direct-v0 --num_envs=64 --max_iterations=500 --headless --export_onnx
```

## Unity import

Inference Engine **2.3** only imports **classic ONNX opset 7–15** (TorchScript exporter).

Do **not** drop the default RSL-RL/PyTorch 2.11 file (dynamo, opset 18, `pkg.torch.onnx.fx_node`). That is what caused:

`ONNXModelConverter.Convert()` → NullReferenceException

Re-export with the launcher **[E] Export ONNX** (now uses TorchScript + opset 15), then copy `policy.onnx` into:

`Unity_XRPlayground/Assets/_Project/Features/Policies/BallCatch/` (or `Conveyor/`)

A healthy MLP ONNX is typically **100 KB+**. A ~14 KB file is usually a dynamo graph without embedded weights.

## Unity

1. Wait for Package Manager to resolve `com.unity.ai.inference`.
2. Re-run **XRPlayground → Setup Conveyor Color Station** (or **Setup XR Bridge Scene** for ball catch) so offline components + **Start Offline Policy** button exist.
3. Select the station/robot → `OnnxPolicyRunner` → assign `policy.onnx`.
4. Play → press **Start Offline Policy**.
   - Conveyor: uses the four cube slots; spawn R/G/B while offline also works.
   - Ball catch: uses the Ball transform/Rigidbody (throws on start).

## Limits

Offline Unity rebuilds the same observation layout and applies joint deltas locally. Belt/cubes/ball are kinematic / Rigidbody approximations — useful for XR demos, not identical to Isaac PhysX.
