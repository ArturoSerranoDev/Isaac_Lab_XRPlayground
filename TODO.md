# Generalized Isaac Lab to Unity Deployment TODO

Last handoff update: 2026-08-17.

## Goal

Replace station-specific deployment duplication with one catalog-driven
framework for all six stations. Each station must support authoritative Isaac
mirror rendering and an offline ONNX policy running against Unity-owned physics.

## Implemented framework

- [x] Authored six-station catalog at `deployment/stations.json`, generated Unity
  runtime catalog, and generated station documentation.
- [x] Versioned `policy.contract.json` model with tensor, observation, action,
  frame/unit, cadence, physics/assist, hash, and evaluation metadata.
- [x] Isaac descriptor and Direct-environment descriptor support, exporter,
  ONNX validation, candidate/ready promotion, golden traces, calibration helpers,
  evaluation helpers, and unified `XRPlayground.cli` commands.
- [x] Bridge v2 framing, named robot/link/object payloads, sequence rejection,
  stale/disconnect handling, and Unity mirror runtime.
- [x] Generic Unity `StationMode`, catalog, contract, runtime, policy runtime,
  adapters, robot definitions, offline rig identity, assist profiles, telemetry,
  scoring, and evaluation runner.
- [x] Shared mirror/offline representation rules and exact hierarchy/collider/hash
  gates for normalized robot prefabs.
- [x] Robot-definition schema 6 with deterministic physics-payload SHA-256,
  revolute/prismatic units, joint axes, anchors, masses, inertias, limits, gains,
  collision and material properties.
- [x] Catalog semantic bindings for root, end effector, contacts, and Isaac robot
  offset; catalog-driven offline task authoring for Ball, Conveyor, Pick & Place,
  and Spot.
- [x] Physical, procedural Balance Bot station and tests.
- [x] Contact-only conveyor motion, corrected conveyor reject metric, deterministic
  object reset/spawn behavior, and corrected Pick & Place defaults.
- [x] Pinned Unity URDF Importer 0.5.2 as an editor-only conversion dependency.
- [x] Recorded immutable redistribution notices for UR10e, Robotiq 2F-85, and the
  selected RAI Spot description. Kinova and procedural Balance were already
  recorded. AgiBot remains blocked.

## Verification state

- [x] Physics-contract milestone: Python `32 passed`, Unity EditMode `19/19`,
  Unity PlayMode `8/8`, generated artifacts valid, and `git diff --check` clean.
- [x] Evidence/handoff/recurrent/normalization batch: Python `36 passed`, Unity
  compilation zero errors, and the two affected EditMode contract tests passed.
  The unchanged full PlayMode baseline remains `8/8`; complete Unity suites were
  intentionally not repeated after every edit.
- [x] Confirmed `m_EnterPlayModeOptions: 0` after the previous Unity test run.
- [x] Added negative tests for missing, empty, and duplicate `license_paths` in
  both Python and Unity catalog validation.

Suggested verification sequence:

1. From `Isaac_XRPlayground`, run the deployment pytest suite and `compileall`
   with `source/XRPlayground` on the Python path.
2. Run catalog validation and generated-artifact check.
3. From repository root, run `rtk git diff --check` with the repository supplied
   as a safe directory if necessary.
4. Wait for Unity compilation; inspect console errors.
5. Run deployment EditMode tests, then PlayMode tests.
6. Restore and verify the EditorSettings play-mode option.

## Immediate engineering work

- [x] Apply exported joint `max_velocity` to Unity
  `ArticulationBody.maxJointVelocity` and cover revolute and prismatic cases.
- [x] Generalize ordered joint and loop-closure telemetry field names/units and
  validate them in Python trace calibration.
- [x] Handle exported `restOffset` truthfully: Unity applies Isaac `contactOffset`,
  accepts absent/zero rest offset, and fails closed on a non-zero rest offset that
  `Collider` cannot represent. Resolve such a source during geometry preparation
  or nominal calibration rather than silently dropping it.
- [x] Tighten joint and collision-axis validation to require finite unit vectors.
- [x] Runtime-test that a contact sensor on an articulation child receives static
  ground collision callbacks.
- [ ] Exercise the authored contact-link bindings on every normalized robot during
  the station behavior/calibration runs.
- [x] Add a checkpoint-free `validate --handoff` JSON report with dependency-first
  training commands and truthful provenance, calibration, candidate, behavior,
  promotion, and legacy-removal blockers for all six policies.
- [x] Require at least 100 promotion golden-trace samples and validate schema,
  contact bindings, assistance profile IDs, and loop-closure identities/units.
- [x] Support optional static recurrent ONNX tensor pairs in Python and Unity;
  validate exact names/shapes, retain outputs as the next inputs, and zero state
  on every high- and low-level station reset.
- [x] Make observation normalization fail closed: read the actor config, require a
  stateful normalizer in the export graph when training uses one, record embedded
  normalization, and reject external/ambiguous normalization in Unity.

## Robot assets and offline rigs

- [ ] Fetch/import the pinned UR10e and Robotiq source revisions recorded in
  `deployment/licenses/SOURCES.md`; generate and commit the normalized combined
  prefab only after exact robot-definition binding passes.
- [ ] Fetch/import the pinned Spot source revision; generate one normalized Spot
  prefab shared by locomotion and follow; overlay actual Isaac physics truth.
- [ ] Resolve the exact license/provenance of the checked-in AgiBot
  `A2D_physics.usd`. Keep Pick & Place unavailable until resolved.
- [ ] Verify/rebuild the Kinova Jaco2 normalized articulation from the recorded
  source and Isaac truth.
- [ ] Install each normalized prefab into its catalog stations, run the offline
  task configurator, and audit exact root/end-effector/contact bindings.
- [ ] Check collider decomposition, self-collision, materials, joint limits,
  inertias, drive units, and reset pose for every rig.

## Physics calibration

- [ ] Export seeded Isaac open-loop traces for each nominal profile.
- [ ] Run the same processed action sequences in Unity.
- [ ] Fit drives, damping, friction, mass/inertia, contact materials, and latency
  with bounded coordinate search over normalized trajectory error.
- [ ] Mark a profile calibrated only after it passes its trace thresholds.
- [ ] Derive domain-randomization ranges from measured residuals plus 20%; do not
  use randomization to conceal a bad nominal model.

## Retraining and bundles

- [ ] Retrain Ball Catch from scratch through Wrap, Throw-A, and Throw-B.
- [ ] Retrain Conveyor Color, Pick & Place, and Balance Bot with standardized
  metrics and health gates.
- [ ] Retrain Spot Stand then Walk at 250 Hz physics / 50 Hz low-level cadence.
- [ ] Promote the new Spot locomotion bundle, then train Spot Follow at 5 Hz using
  the corrected 10-value observation and four-value pose command into locomotion.
- [ ] Export every policy as classic TorchScript ONNX opset 15.
- [ ] Validate real tensor names/shapes, recurrent tensors if present, hashes,
  PyTorch/ONNX parity, and Python/Unity inference parity.
- [ ] Produce a complete `golden_trace.jsonl` for every candidate bundle.
- [ ] Promote only bundles that pass catalog, contract, definition, calibration,
  parity, Unity import, and behavior gates. No production bundle is currently
  ready merely from framework completion.

## Station behavior gates

- [ ] Ball: physical thrown-ball catch and retained three-fingertip grasp, no
  pre-contact assistance.
- [ ] Conveyor: correct sorting and physical reject handling; report wrong-bin
  rate separately.
- [ ] Pick & Place: contacted grasp, lift, opening-command release, and stable
  placement.
- [ ] Balance: two-ball full-episode hold/drop metrics.
- [ ] Spot locomotion: ten-second stand, command tracking, valid foot contacts,
  and acceptable fall rate without translation or teleport assistance.
- [ ] Spot Follow: exact 10-value observation, 5/50 Hz hierarchy, distance/heading
  tracking, and acceptable fall rate.
- [ ] Run 100 identical seeds in Isaac and Unity for each policy; require Unity
  normalized score at least 80% of Isaac with no numerical/contract violations.

## Performance and migration completion

- [ ] Profile a Windows PCVR development build with CPU inference and one active
  station: target 11.1 ms and require p95 frame time no worse than 13.9 ms.
- [ ] Verify 90 Hz target and 72 Hz hard minimum for every active offline station.
- [ ] Run the complete six-station suite in one clean validation report.
- [ ] Only after every gate passes, remove duplicated legacy controllers, bridge
  runners, sidecars, panels, launchers, batch files, and stale documentation.
- [ ] Confirm no compatibility wrappers or old command syntax remain.
