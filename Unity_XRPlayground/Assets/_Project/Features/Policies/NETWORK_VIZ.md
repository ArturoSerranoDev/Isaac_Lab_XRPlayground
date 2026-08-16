# Policy network visualizer (Unity XR)

World-space **Network** panel beside each station’s Mirror / Offline Policy bridge UI. While **Offline Policy** runs, it draws the actor MLP and highlights neurons by activation magnitude each inference step.

## Stations

| Station | Robot | Bridge UI | Network panel | Typical arch (fallback → live from ONNX) | ONNX path |
|---------|-------|-----------|---------------|------------------------------------------|-----------|
| **BallCatch** | Kinova | `XR Bridge World UI` | `Policy Network World UI` | `30 → 256 → 128 → 64 → 8` | `Policies/BallCatch/policy.onnx` |
| **Conveyor** | UR10e | `Conveyor Bridge World UI` | `Policy Network Conveyor World UI` | `66 → 256 → 128 → 64 → 7` | `Policies/Conveyor/policy.onnx` |
| **PickPlace** | Agibot | `PickPlace Bridge World UI` | `Policy Network PickPlace World UI` | `30 → 256 → 128 → 64 → 8` | `Policies/PickPlace/policy.onnx` |
| **BalanceBot** | 2-DOF tray | `BalanceBot Bridge World UI` | `Policy Network BalanceBot World UI` | `20 → 256 → 128 → 64 → 2` | `Policies/BalanceBot/policy.onnx` |

Hidden widths (`256 → 128 → 64`) come from each model’s Dense bias shapes when a `policy.onnx` is assigned. Until then the panel shows the fallback sizes above (idle / placeholder).

## Open / place the panels

1. Open `SampleScene` (or a dedicated station scene with its bridge UI).
2. Menu (pick one or all):
   - **XRPlayground → Setup Policy Network Panel (BallCatch)**
   - **XRPlayground → Setup Policy Network Panel (Conveyor)**
   - **XRPlayground → Setup Policy Network Panel (PickPlace)**
   - **XRPlayground → Setup Policy Network Panel (BalanceBot)**
   - **XRPlayground → Setup Policy Network Panels (All Stations)**
3. Station setup menus also create the matching Network panel:
   - **Setup XR Bridge Scene** → BallCatch
   - **Setup Conveyor Color Station** → Conveyor
   - **Setup Pick Place Station** → PickPlace
   - **Setup Balance Bot Station** → BalanceBot

Each Network panel sits to the local +X of that station’s bridge (same facing, world-space uGUI).

Play Mode → on the station bridge tap **Start Offline Policy**. The Network panel’s **LIVE** badge turns on and columns animate (requires `policy.onnx` on that station’s `OnnxPolicyRunner`).

## What you see

| Column | Meaning |
|--------|---------|
| `in N` | Observation magnitudes `\|obs\|` |
| `h1` … `hK` | Hidden **ELU** post-activations (absolute value) |
| `out M` | Action magnitudes `\|mean action\|` |

Architecture label shows live sizes from ONNX (or the station fallback). Wide layers (&gt; 48 units) render as a compact heatmap strip; smaller layers use discrete dots. Optional thin edges connect top-k peaks between adjacent discrete layers.

**Color:** cold blue → warm yellow → orange peak. Intensity is **per-layer normalized** `|activation| / max(|a|)` that step, so a quiet layer still shows relative structure.

## Real intermediates vs approximate

| Mode | When | Hidden columns |
|------|------|----------------|
| **Real** | Default for RSL-RL Unity ONNX (`Sub/Div` norm → `Dense`→`Elu` ×3 → `Dense`) | Inference Engine `Model.AddOutput` on each **ELU** tensor; `PeekOutput` after `Schedule` |
| **Fallback** | Graph has no separate ELU (fused) or probe fails | Status line notes limitation; input/output still real; hidden may be empty or Dense pre-act probes |

Status footer text:

- `Highlight = |activation| · real ELU tensors` — trusted hidden viz
- `Highlight = |obs|/|act| · hidden approx unavailable` — structure only for middles
- `Assign policy.onnx — panel lights up when Offline Policy runs` — model not assigned yet

Obs normalization (`RunningMeanStd` as Sub/Div) is inside the ONNX; hidden ELUs are after that.

## Wiring

Same components on every station:

1. Station `OnnxPolicyRunner` (`captureActivations = true`)
2. Station offline controller (`BallCatch` / `Conveyor` / `PickPlace` / `BalanceBot`)
3. `PolicyNetworkPanel.BindBallCatch` / `BindConveyor` / `BindPickPlace` / `BindBalanceBot`

| Station | Controller | Runner host |
|---------|------------|-------------|
| BallCatch | `BallCatchOfflinePolicyController` | `Kinova_Jaco2_j2n7s300` |
| Conveyor | `ConveyorOfflinePolicyController` | `Station_B_Conveyor` |
| PickPlace | `PickPlaceOfflinePolicyController` | `Station_C_PickPlace` |
| BalanceBot | `BalanceBotOfflinePolicyController` | `Station_D_BalanceBot` |
| Spot | `SpotOfflinePolicyController` (loco runner) | `Station_E_Spot` |

## Files

| Asset | Role |
|-------|------|
| `Policies/Scripts/OnnxPolicyRunner.cs` | Loads ONNX, optional intermediate outputs |
| `Policies/Scripts/PolicyInferenceSnapshot.cs` | Per-step activation payload |
| `Policies/Scripts/PolicyNetworkVisualizer.cs` | Draws columns / strips / top-k edges |
| `Policies/Scripts/PolicyNetworkPanel.cs` | Live/idle chrome + wiring |
| `ROS/Editor/PolicyNetworkPanelSetup.cs` | Editor menus + world UI build |

Does not touch Isaac train scripts. Offline robot control is unchanged aside from enabling activation capture on each runner.
