# Isaac Lab to Unity deployment

`stations.json` is the only authored deployment catalog. Run generation in check mode in CI; do not edit `STATIONS.generated.md` or Unity's generated catalog by hand.

## Runtime contract

Every Unity station has three explicit modes:

- `Disabled`: the station owns no bridge or policy work.
- `Mirror`: Isaac owns simulation state. Unity renders version-2 named link/object poses, interpolates only between received samples, freezes after 250 ms, and disconnects after 2 s. It never extrapolates authoritative state.
- `OfflinePolicy`: Unity owns the articulation and task physics. A validated ONNX bundle drives the station through its generated `policy.contract.json`; Mirror is never used as a fallback.

Only one offline station may be active. Candidate bundles and uncalibrated physics profiles fail closed in production. This is deliberate: the current source assets are useful Mirror rigs, but they are not substitutes for normalized `ArticulationBody` prefabs with Isaac physics truth.

## Unified commands

Install `Isaac_XRPlayground` into the Python environment that contains Isaac Lab, then run from the repository root:

```text
python -m XRPlayground.cli train --policy ball_catch.throw --phase 1 -- --headless
python -m XRPlayground.cli play --policy spot.locomotion -- --checkpoint /path/to/model.pt
python -m XRPlayground.cli bridge --station ball_catch
python -m XRPlayground.cli bridge --station spot_follow -- --checkpoint /path/to/follow-model.pt
python -m XRPlayground.cli export --policy spot.locomotion --definition-only -- --device cpu --viz none
python -m XRPlayground.cli export --policy ball_catch.throw --checkpoint /path/to/model.pt
python -m XRPlayground.cli validate --catalog-only
python -m XRPlayground.cli validate --handoff --handoff-output /path/to/handoff.json
python -m XRPlayground.cli validate --behavior
python -m XRPlayground.cli validate --policy ball_catch.throw --build-evaluation \
  --isaac-results /path/to/isaac.scenarios.jsonl \
  --unity-results /path/to/unity.scenarios.jsonl \
  --parity-report /path/to/parity.samples.json
python -m XRPlayground.cli validate --policy balance_bot.two_ball \
  --write-open-loop-sequence /path/to/balance.actions.json
python -m XRPlayground.cli validate --build-calibration-comparison \
  --station balance_bot --physics-profile tray_120hz \
  --isaac-trace /path/to/isaac.golden_trace.jsonl \
  --unity-trace /path/to/unity.golden_trace.jsonl
python -m XRPlayground.cli validate --promote
```

Arguments after `--` are forwarded to the Isaac script. Spot Follow intentionally has no implicit high-level checkpoint: it is 10 to 3, while the nested locomotion policy is 48 to 12. The CLI injects the hashed locomotion candidate automatically, or accepts an explicit `XRPLAYGROUND_SPOT_LOCO_POLICY` path.

For local Spot iteration before the 100-seed Unity behavior gate is available, train locomotion first and export it as a statically validated candidate. Follow can then be trained with `train --policy spot.follow --allow-unvalidated-dependency -- ...`. This explicit switch affects dependency selection only: it does not mark either policy ready, calibrated, or promotion-eligible. In Unity, `XRPlayground/Deployment/Bind Local Spot Follow Candidates` binds the two candidate bundles with `allowUnpromotedCandidate` enabled so OfflinePolicy can be exercised locally; production builds still use `Bind Promoted Policy Release`.

Add `--reference-only` to `export` when probing an old checkpoint. The bundle is staged under Unity's `Bundles/References` tree with `promotion_eligible: false`; validation and promotion scan only `Bundles/Candidates`.

`export --definition-only` is the checkpoint-free preparation path. It launches one live Isaac environment, exports schema-6 articulation topology, collision geometry descriptors, masses/inertias/materials, and resolved actuator properties, stages only the Unity robot definition, and exits without loading or training a policy. Body-to-body USD constraints omitted by the PhysX DOF/fixed-tree views are retained as hashed `auxiliary_joints`, including source prim/type, anchors, axis, and a validated `tree_connector` or `loop_closure` role.

For a Mirror smoke test, start a station bridge, open `SampleScene`, and enter Play Mode. The generated shells default to Mirror (Spot Locomotion is disabled because it shares Spot's visual hierarchy with Follow). Select `Deployment Runtime`, set `selectedStationId`, then use the `DeploymentModeController` component context commands. `Launch Ball` sends the Isaac-coordinate launch parameters through bridge v2; Isaac remains authoritative.

## Raw evidence and golden traces

Each generated station shell has a `GoldenTraceRecorder`. Start and stop it from the component context menu while the station is in a healthy OfflinePolicy mode. It writes `golden_trace.jsonl` atomically under `Application.persistentDataPath/DeploymentTraces/<session>` and records the reset seed/state, ordered observations, raw and processed actions, contract-ordered joint state with per-joint position/velocity units, root and task-object state, contacts, assistance force/torque and breaks, and task score on every policy step. A recording with no samples is discarded instead of masquerading as evidence.

The same shell has a `UnityScenarioEvaluationRunner`. Its `Start 100-Seed Unity Evaluation` context command refuses to run unless OfflinePolicy is healthy, executes seeds 0 through 99 for the configured duration, and atomically publishes `unity.scenarios.jsonl` only after the complete set finishes. Aborting or losing the station deletes the partial output. Ball scenarios use the seeded Throw-B ballistic launcher and the same 0.55 gravity scale as training; manipulation objects and Spot roots return to cached or explicitly configured reset poses between seeds.

For physics calibration, assign a schema-1 action-sequence JSON asset to the shell's `OpenLoopActionPlayer` and use its context command in healthy OfflinePolicy mode. The sequence identifies the station, policy and reset seed and contains ordered `processed_action` arrays. Normal policy inference is suspended while these exact actions are applied; the shared golden-trace recorder captures Unity's response. Run the same processed actions in Isaac, then use `validate --build-calibration-comparison`. The comparison rejects any input mismatch, hashes both traces, and reports normalized joint, root and object response errors. Closed-chain rigs additionally record per-constraint anchor/axis error, derived position/velocity with explicit units, proxy mass, and health; malformed units or unhealthy closures reject the comparison, while healthy residuals contribute to its normalized error. Marking a catalog profile calibrated is still a deliberate authored change, and validation then requires passing raw-evidence reports for every station using that profile.

Isaac and Unity scenario files use one JSON object per line. Every file must contain exactly the same 100 unique seeds. Each row has `schema_version: 1`, `policy_id`, `seed`, `normalized_task_score`, station-specific `task_metrics`, and the boolean health fields `has_nan`, `invalid_actions`, `missing_joints`, and `joint_limit_violations`. The required station metric names are enforced by the evaluator.

The parity sample JSON has `schema_version: 1` and non-empty arrays named `python_unity_abs_errors`, `observation_abs_errors`, `mirror_position_errors_m`, `mirror_rotation_errors_deg`, and `frame_times_ms`; frame timing requires at least 100 samples. `validate --build-evaluation` copies these three raw inputs beside the candidate, hashes them, and writes `evaluation.json`. Promotion verifies the hashes, recomputes normalized scores from the standardized task metrics, and applies the task-specific thresholds authored in `stations.json`, so hand-edited summary scores cannot pass the gate.

`validate --handoff` is checkpoint-free and never launches Isaac, trains, exports, calibrates, or promotes anything. It emits a schema-1 JSON readiness report in dependency order with every definition, provenance, per-station calibration, candidate, behavior, and Spot hierarchy blocker plus the exact catalog-derived commands for the next step. `legacy_removal_allowed` becomes true only when the complete six-policy gate is genuinely ready.

Promotion requires at least 100 valid golden-trace policy samples. Each sample must carry schema/identity/timing fields, exact contract dimensions and joint units, the catalog-authored contact-sensor set, contract-matched assistance events, and the robot-definition-matched loop-closure count, identity, units, mass, and health. A short or structurally plausible trace cannot stand in for raw runtime evidence.

## Clean-break production sequence

1. Generate/check the catalog with `validate --catalog-only`.
2. Record and verify the robot source provenance and redistribution license, then import the pinned URDF sources in Unity. Approved immutable source revisions and notices are listed in [`licenses/SOURCES.md`](licenses/SOURCES.md); verification authorizes that source but does not bypass hierarchy or calibration gates.
3. Export each candidate from its live Isaac environment. Export writes a static opset-15 `policy.onnx`, hashed `policy.pt` for hierarchical Isaac dependencies, versioned `policy.contract.json`, and `robot.definition.json` atomically after PyTorch/ONNX parity passes. Feed-forward models expose one input/output pair. Recurrent models carry ordered, static, shape-matched state input/output pairs; Unity keeps those tensors between policy steps and zeros them on every station reset. The exporter reads the actor's authored `obs_normalization` setting and, when enabled, refuses the export unless a stateful normalizer is demonstrably part of the exported module. Unity accepts only `normalization_embedded: true`; it never silently feeds raw observations to a policy trained on normalized inputs.
4. Normalize and install the Unity articulation using that exact schema-6 robot definition. Offline activation requires a matching catalog robot ID, verified provenance, canonical live-physics hash, exact link/joint hierarchy, and exact per-link collider counts. The definition carries fixed-link topology, resolved Isaac actuator gains, model type, delay range, any angle-dependent effort curve, and explicit limited/continuous joint semantics instead of relying only on the raw PhysX drive. Joint axes are mapped into Unity anchor frames, revolute targets are converted from radians to drive degrees, and prismatic quantities remain meters. Axes must be finite unit vectors and exported maximum velocities are applied to each articulation degree of freedom. Unity applies Isaac contact offsets; because `Collider` has no per-shape rest-offset equivalent, absent/zero Isaac rest offsets are accepted and non-zero values fail closed for explicit geometry/profile resolution. Configure the catalog semantic task bindings, then calibrate the nominal profile against seeded open-loop traces.
5. Derive domain-randomization bounds from measured residuals, add 20 percent margin, and retrain from scratch in catalog order. Spot Follow training is refused until the locomotion candidate passes its behavior gate; its exact ONNX and TorchScript dependency hashes are recorded in the Follow contract.
6. Record the required `golden_trace.jsonl`, paired 100-seed scenario results, numerical parity samples, and Windows PCVR frame times. Build the evaluation report from those raw files, then run the contract and behavior gates.
7. Promote only with `validate --promote`. Promotion is a six-policy transaction and updates Unity's ready catalog atomically. Unity then binds that active release into every loaded station `PolicyRuntime` (including Spot Follow's nested locomotion runtime); the editor menu `Bind Promoted Policy Release` can repeat the idempotent binding before a build.
8. Remove the disabled legacy components only after that complete promoted release passes. There are intentionally no compatibility wrappers.

The Unity scene menu `XRPlayground/Deployment/Build V2 Runtime Shells` is idempotent and creates the generic station shells. `Build And Install USD Articulation Prefabs` reproducibly rebuilds and installs Jaco2, the prepared UR10e/Robotiq composite, and Spot from checked-in source assets plus the live definitions. AgiBot is registered in the same builder. Its topology planner keeps the four `*_2_Joint` edges in one reduced-coordinate tree and represents the four passive duplicate-child revolute edges with separately rooted, collision-free Rigidbody proxies joined to the relevant articulation links. Proxy binding order, external-joint telemetry, and bounded anchor drift are covered by PlayMode tests; the explicit 0.1 kg proxy mass remains part of later open-loop calibration. AgiBot prefab output is still skipped until redistribution provenance is verified. `Normalize Selected URDF Robot` remains the generic path for a newly imported robot, `Install Selected Normalized Robot Rig` installs an environment wrapper plus the catalog robot offset without overwriting an authored rig, and `Configure Selected Offline Station Rig` builds/wires the station task physics from catalog semantic links. `Audit Runtime Readiness` reports identity/provenance failures, missing adapters, and missing bundles without inventing replacements.

For UR10e/Robotiq, first run `Isaac_XRPlayground/scripts/deployment/prepare_unity_robot_sources.py` with the Isaac Lab environment Python. It verifies the two catalog-pinned Git revisions, expands the official Xacros, writes an exact 16-link URDF from the live Isaac topology, copies the licensed source meshes, and records a hash manifest. The generated URDF is compatible with the pinned Unity URDF Importer 0.5.2; the automated normalized-prefab builder uses Unity-imported versions of the same meshes to avoid an editor-runtime Assimp dependency.

## Current readiness boundary

Mirror infrastructure and all six station shells are implemented. Definition-bound physical rigs and task adapters are installed for Ball Catch, Conveyor Color, Balance Bot, Spot Locomotion, and Spot Follow. Production OfflinePolicy remains fail-closed because the nominal physics profiles are deliberately uncalibrated and no newly trained six-policy release has been promoted. Pick & Place additionally remains blocked by the unverified AgiBot redistribution license. Its live definition records the four revolute hand closures explicitly, and Unity now has a tested passive-loop proxy adapter, but no AgiBot physical prefab is emitted until provenance is verified; once emitted, its proxy mass and residual closure error must be included in nominal-profile calibration. Do not mark a profile calibrated or provenance verified merely to bypass these gates.
