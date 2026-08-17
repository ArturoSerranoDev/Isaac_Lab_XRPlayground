# XRPlayground Agent Rules

These rules apply to every agent working in this repository. Read this file and
`deployment/README.md` before modifying Isaac or Unity deployment code.

## Working conventions

1. Prefix every shell command with `rtk`, as required by the repository's
   `AGENTS.md` instructions.
2. Preserve unrelated user changes. The worktree may be dirty; inspect it before
   editing and never reset, discard, or overwrite changes that are not yours.
3. Use `apply_patch` for hand-written file changes. Use the project generators
   only for files explicitly marked as generated.
4. Treat `deployment/stations.json` as the single authored deployment catalog.
   Never hand-author a competing station list or policy dimension table.
5. The Unity resource catalog and `deployment/STATIONS.generated.md` are generated
   artifacts. Regenerate them from the authored catalog and make check mode pass.
6. Do not delete legacy scripts, controllers, panels, bridge runners, or batch
   files until all six stations pass the complete promotion and behavior suite.
7. When asked to play an Isaac Lab training, launch it with USD stage
   synchronization and CPU simulation (`--disable_fabric --device cpu`) unless
   the user explicitly overrides this.

## Deployment architecture

1. Every station must expose exactly three explicit modes: `Disabled`, `Mirror`,
   and `OfflinePolicy`. A mode must fail closed; it may never silently fall back
   to another mode.
2. Mirror mode is authoritative Isaac state rendered through plain visual
   transforms. It must not drive Unity physics and must never extrapolate stale
   Isaac state.
3. Offline mode owns its state in Unity and uses real Unity physics. Robot rigs
   must use `ArticulationBody`, a matching `RobotDefinition`, and an
   `OfflineRigIdentity`. At most one offline station may be active.
4. Station-specific code belongs only in `IStationAdapter` implementations:
   reset, observation construction, action application, telemetry, and scoring.
   Inference, timing, bundle loading, bridge state, mode switching, and UI stay
   generic.
5. Do not hard-code observation/action dimensions, term order, joint order,
   tensor names, or cadence in Unity. Read them from the catalog and
   `policy.contract.json` generated from Isaac descriptors.
6. Preserve units and coordinate frames explicitly. Revolute positions use
   radians in contracts/definitions and degrees only at Unity drive boundaries;
   prismatic positions remain metres. Apply each exported joint axis and anchor.
7. Do not add an unconditional `[-1, 1]` action clamp. Apply only the scaling,
   offsets, integration, target range, and optional clipping declared by the
   policy contract.
8. Physics profiles and policy cadence are contractual. Arm/ball/balance stations
   use 120 Hz physics and 60 Hz policy evaluation. Spot uses 250 Hz physics,
   50 Hz locomotion, and 5 Hz follow commands.

## Assets and provenance

1. Robot assets are fail-closed. Do not normalize, commit, or bundle a robot
   unless its exact immutable source revision and every redistribution notice are
   recorded in the catalog and under `deployment/licenses/`.
2. `redistribution_verified` means the intended source is redistributable; it
   does not mean a normalized prefab, calibration, or policy bundle exists.
3. Generated robot definitions must include the canonical definition SHA-256.
   Reject definitions whose physics payload, robot ID, hierarchy, link/joint
   names, or collider counts do not match.
4. Use Unity's pinned official URDF Importer only as an editor conversion tool.
   Strip importer-only components from normalized runtime prefabs.
5. Overlay physics truth exported from the actual Isaac articulation: mass, COM,
   inertia, anchors, axes, limits, gains, velocity limits, collision and material
   properties. Do not substitute guessed Unity defaults.
6. AgiBot remains blocked until the checked-in `A2D_physics.usd` has defensible
   source and redistribution terms. Do not mark it verified based on a different
   AgiBot repository's license.

## Bridge, policies, and promotion

1. Use only bridge schema v2: one length-prefixed JSON envelope with station ID,
   sequence, simulation time, frame ID, message type, and generic named payloads.
2. Reject out-of-order sequences. Mark data stale and freeze after 250 ms;
   disconnect after 2 s. Interpolate only rendering and never extrapolate
   authoritative state.
3. Production ONNX is classic TorchScript export, opset 15, for Unity Inference
   Engine 2.3. Validate real input/output tensor shapes and PyTorch/ONNX parity
   before atomic promotion.
4. Candidate or partial exports stay outside Unity's ready catalog. Promotion
   requires contract validation, hashes, robot definition, calibration,
   golden trace, Isaac evaluation, ONNX parity, Unity validation, and behavior
   gates.
5. Assistance must remain bounded and contact-driven. It may not attract objects
   before contact, translate Spot's root, hide falls, or teleport recovery.
   Record activation, force/torque, and break events and keep an unassisted toggle.
6. Never describe a station as behavior-validated merely because code or unit
   tests pass. Behavior validation requires the trained bundle and 100 seeded
   Isaac/Unity scenarios.

## Required verification

1. Run Python unit tests, bytecode compilation, catalog validation, generated
   artifact check, and `git diff --check` after relevant changes.
2. Let Unity finish compiling and confirm zero console errors before using new
   types or running tests.
3. Run both Unity EditMode and PlayMode deployment test assemblies after runtime,
   catalog, prefab, adapter, or policy changes.
4. Unity test runs can change
   `Unity_XRPlayground/ProjectSettings/EditorSettings.asset` field
   `m_EnterPlayModeOptions` from `0` to `1`. Restore it to `0` after every test
   run unless the user deliberately changed that setting.
5. Keep numerical gates intact: PyTorch/ONNX `<= 1e-5`, Python/Unity ONNX
   `<= 1e-4`, injected observations `<= 1e-4`, and mirror conversion within
   1 mm / 0.1 degrees.
6. Final acceptance also requires 100 identical seeds per policy, Unity score at
   least 80% of Isaac, no NaNs/invalid actions/missing joints/limit violations,
   and PCVR p95 frame time at or below 13.9 ms with one active station.

## Handoff discipline

1. Update `TODO.md` when a milestone, blocker, test result, or external dependency
   changes.
2. Record the exact command/test job and whether it ran before or after the latest
   edits. Do not present stale passing results as current verification.
3. Leave concise comments explaining deliberate fail-closed behavior; do not add
   compatibility wrappers during this clean-break migration.
