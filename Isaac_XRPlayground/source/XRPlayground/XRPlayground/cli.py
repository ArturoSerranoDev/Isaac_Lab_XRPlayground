# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single public CLI for training, playing, bridging, exporting and validating."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .deployment.catalog import load_catalog, monorepo_root
from .deployment.generate import generate
from .deployment.validation import validate_all, validate_candidate


ISAAC_ROOT = monorepo_root() / "Isaac_XRPlayground"


def _run(script: str, arguments: list[str], *, environment: dict[str, str] | None = None) -> int:
    command = [sys.executable, str(ISAAC_ROOT / script), *arguments]
    return subprocess.run(command, cwd=ISAAC_ROOT, env=environment, check=False).returncode


def _forwarded(arguments: list[str]) -> list[str]:
    return arguments[1:] if arguments and arguments[0] == "--" else arguments


def _training_environment(
    policy_id: str, *, allow_unvalidated_dependency: bool = False
) -> dict[str, str] | None:
    policy = load_catalog().policy(policy_id)
    if "spot.locomotion" not in policy.depends_on_policy_ids:
        return None
    report = validate_candidate(
        "spot.locomotion", require_behavior=not allow_unvalidated_dependency
    )
    if not report.ok:
        qualification = (
            "statically validated local" if allow_unvalidated_dependency
            else "behavior-validated"
        )
        raise RuntimeError(
            f"Spot Follow training requires a {qualification} locomotion candidate:\n"
            + "\n".join(report.errors)
        )
    dependency = (
        ISAAC_ROOT.parent
        / "Unity_XRPlayground"
        / "Assets"
        / "_Project"
        / "Features"
        / "Deployment"
        / "Bundles"
        / "Candidates"
        / "spot.locomotion"
        / "policy.pt"
    )
    environment = os.environ.copy()
    environment["XRPLAYGROUND_SPOT_LOCO_POLICY"] = str(dependency)
    return environment


def _bridge_environment(station_id: str) -> dict[str, str] | None:
    if station_id != "spot_follow":
        return None
    environment = os.environ.copy()
    configured = environment.get("XRPLAYGROUND_SPOT_LOCO_POLICY", "").strip()
    if configured and Path(configured).is_file():
        return environment
    report = validate_candidate("spot.locomotion")
    if not report.ok:
        raise RuntimeError(
            "Spot Follow bridge requires an exported locomotion candidate or "
            "XRPLAYGROUND_SPOT_LOCO_POLICY:\n" + "\n".join(report.errors)
        )
    dependency = (
        ISAAC_ROOT.parent
        / "Unity_XRPlayground"
        / "Assets"
        / "_Project"
        / "Features"
        / "Deployment"
        / "Bundles"
        / "Candidates"
        / "spot.locomotion"
        / "policy.pt"
    )
    environment["XRPLAYGROUND_SPOT_LOCO_POLICY"] = str(dependency)
    return environment


def _policy_task(policy_id: str, phase: str | None) -> str:
    policy = load_catalog().policy(policy_id)
    if phase is None:
        return policy.source_task_id
    if phase.isdigit():
        index = int(phase)
        if index < 1 or index > len(policy.training_task_ids):
            raise ValueError(f"phase index must be within 1..{len(policy.training_task_ids)}")
        return policy.training_task_ids[index - 1]
    if phase not in policy.training_task_ids:
        raise ValueError(f"'{phase}' is not a training phase for {policy_id}")
    return phase


def _bridge_invocation(station_id: str) -> tuple[str, list[str]]:
    catalog = load_catalog()
    station = catalog.station(station_id)
    scripts = {
        "ball_catch": "scripts/bridge/run_xr_bridge.py",
        "conveyor_color": "scripts/bridge/run_xr_bridge_conveyor.py",
        "pick_place_table": "scripts/bridge/run_xr_bridge_pick_place.py",
        "balance_bot": "scripts/bridge/run_xr_bridge_balance_bot.py",
        "spot_loco": "scripts/bridge/run_xr_bridge_spot.py",
        "spot_follow": "scripts/bridge/run_xr_bridge_spot.py",
    }
    primary_policy = catalog.policy(station.policy_ids[0])
    arguments = [f"--port={station.port}", f"--task={primary_policy.source_task_id}"]
    spot_tasks = {
        "spot_loco": "Template-Xrplayground-Spot-Loco-Walk-Play-v0",
        "spot_follow": "Template-Xrplayground-Spot-Follow-Play-v0",
    }
    if station.station_id in spot_tasks:
        arguments[1] = f"--task={spot_tasks[station.station_id]}"
        arguments.append(f"--station-id={station.station_id}")
    return scripts[station.station_id], arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m XRPlayground.cli", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a catalog policy")
    train.add_argument("--policy", required=True)
    train.add_argument("--phase", help="1-based curriculum phase or exact Isaac task ID")
    train.add_argument(
        "--allow-unvalidated-dependency",
        action="store_true",
        help=(
            "allow Spot Follow training to consume a statically valid local locomotion "
            "candidate before its Unity behavior gate; never changes promotion eligibility"
        ),
    )
    train.add_argument("args", nargs=argparse.REMAINDER)

    play = commands.add_parser("play", help="play a policy checkpoint in Isaac")
    play.add_argument("--policy", required=True)
    play.add_argument("--phase")
    play.add_argument("args", nargs=argparse.REMAINDER)

    bridge = commands.add_parser("bridge", help="run a station's Isaac mirror bridge")
    bridge.add_argument("--station", required=True)
    bridge.add_argument("args", nargs=argparse.REMAINDER)

    export = commands.add_parser(
        "export", help="export a policy bundle or checkpoint-free robot definition"
    )
    export.add_argument("--policy", required=True)
    export.add_argument("--phase")
    export.add_argument("--checkpoint")
    export.add_argument(
        "--definition-only",
        action="store_true",
        help="export live robot physics to Unity without loading a policy checkpoint",
    )
    export.add_argument(
        "--reference-only",
        action="store_true",
        help="stage an old checkpoint outside the promotion candidate tree",
    )
    export.add_argument("args", nargs=argparse.REMAINDER)

    validate = commands.add_parser("validate", help="validate catalog and candidate bundles")
    validate.add_argument("--policy")
    validate.add_argument("--behavior", action="store_true", help="require full promotion artifacts")
    validate.add_argument("--catalog-only", action="store_true")
    validate.add_argument(
        "--handoff", action="store_true",
        help="report calibration, training, candidate and promotion readiness without running Isaac",
    )
    validate.add_argument("--handoff-output")
    validate.add_argument("--write-generated", action="store_true")
    validate.add_argument("--build-evaluation", action="store_true")
    validate.add_argument("--isaac-results")
    validate.add_argument("--unity-results")
    validate.add_argument("--parity-report")
    validate.add_argument("--build-calibration-comparison", action="store_true")
    validate.add_argument("--station")
    validate.add_argument("--physics-profile")
    validate.add_argument("--isaac-trace")
    validate.add_argument("--unity-trace")
    validate.add_argument("--calibration-output")
    validate.add_argument("--write-open-loop-sequence")
    validate.add_argument("--open-loop-frames", type=int)
    validate.add_argument("--open-loop-amplitude", type=float, default=0.2)
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument(
        "--promote", action="store_true", help="atomically promote only after all six behavior gates pass"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            task = _policy_task(args.policy, args.phase)
            return _run(
                "scripts/rsl_rl/train.py",
                [f"--task={task}", *_forwarded(args.args)],
                environment=_training_environment(
                    args.policy,
                    allow_unvalidated_dependency=args.allow_unvalidated_dependency,
                ),
            )
        if args.command == "play":
            task = _policy_task(args.policy, args.phase)
            return _run("scripts/rsl_rl/play.py", [f"--task={task}", *_forwarded(args.args)])
        if args.command == "export":
            task = _policy_task(args.policy, args.phase)
            if args.definition_only:
                if args.checkpoint or args.reference_only:
                    raise ValueError(
                        "--definition-only does not accept --checkpoint or --reference-only"
                    )
                return _run(
                    "scripts/rsl_rl/export_robot_definition.py",
                    [f"--task={task}", *_forwarded(args.args)],
                )
            if not args.checkpoint:
                raise ValueError("policy export requires --checkpoint")
            export_args = [f"--task={task}", f"--ckpt={args.checkpoint}"]
            if args.reference_only:
                export_args.append("--reference_only")
            return _run(
                "scripts/rsl_rl/export_onnx.py",
                [*export_args, *_forwarded(args.args)],
                environment=_training_environment(
                    args.policy, allow_unvalidated_dependency=True
                ),
            )
        if args.command == "bridge":
            script, bridge_args = _bridge_invocation(args.station)
            return _run(
                script,
                [*bridge_args, *_forwarded(args.args)],
                environment=_bridge_environment(args.station),
            )
        if args.command == "validate":
            if args.write_generated:
                generate(check=False)
            if args.catalog_only:
                generate(check=True)
                load_catalog()
                print("Catalog and generated artifacts are valid")
                return 0
            if args.handoff:
                from .deployment.handoff import build_handoff_report, write_handoff_report

                if args.handoff_output:
                    output = write_handoff_report(args.handoff_output)
                    print(f"Built deployment handoff report: {output}")
                else:
                    print(json.dumps(build_handoff_report(), indent=2, sort_keys=True))
                return 0
            if args.build_evaluation:
                if not args.policy or not args.isaac_results or not args.unity_results or not args.parity_report:
                    raise ValueError(
                        "--build-evaluation requires --policy, --isaac-results, "
                        "--unity-results and --parity-report"
                    )
                from .deployment.evaluation import build_evaluation_report

                output = build_evaluation_report(
                    args.policy, args.isaac_results, args.unity_results, args.parity_report
                )
                print(f"Built evaluation report: {output}")
                return 0
            if args.build_calibration_comparison:
                if not args.station or not args.physics_profile or not args.isaac_trace or not args.unity_trace:
                    raise ValueError(
                        "--build-calibration-comparison requires --station, --physics-profile, "
                        "--isaac-trace and --unity-trace"
                    )
                from .deployment.calibration import build_open_loop_comparison_report

                output = build_open_loop_comparison_report(
                    args.station,
                    args.physics_profile,
                    args.isaac_trace,
                    args.unity_trace,
                    output=args.calibration_output,
                )
                print(f"Built calibration comparison: {output}")
                return 0
            if args.write_open_loop_sequence:
                if not args.policy:
                    raise ValueError("--write-open-loop-sequence requires --policy")
                from .deployment.calibration import write_open_loop_action_sequence

                output = write_open_loop_action_sequence(
                    args.policy,
                    args.write_open_loop_sequence,
                    seed=args.seed,
                    frame_count=args.open_loop_frames,
                    amplitude=args.open_loop_amplitude,
                )
                print(f"Built open-loop action sequence: {output}")
                return 0
            if args.promote:
                if args.policy:
                    raise ValueError("promotion is an all-station transaction; omit --policy")
                from .deployment.promotion import promote_all

                release = promote_all()
                print(f"Promoted complete deployment release: {release}")
                return 0
            report = (
                validate_candidate(args.policy, require_behavior=args.behavior)
                if args.policy
                else validate_all(require_behavior=args.behavior)
            )
            for item in report.checked:
                print(f"[OK] {item}")
            for item in report.errors:
                print(f"[ERROR] {item}", file=sys.stderr)
            return 0 if report.ok else 1
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
