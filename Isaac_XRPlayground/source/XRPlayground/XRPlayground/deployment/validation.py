# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Static validation for deployment candidates; behavioral promotion is separate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import load_catalog
from .calibration import compare_open_loop_traces
from .contract import PolicyContract, PolicyContractError, sha256_file
from .evaluation import compute_evaluation_report
from .exporter import candidate_bundle_path
from .generate import generate
from .robot_definition import RobotDefinitionDocument, RobotDefinitionError
from .trace import load_golden_trace


@dataclass
class ValidationReport:
    checked: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _reject_reference_marker(bundle: Path) -> None:
    marker = bundle / "REFERENCE_ONLY.json"
    if marker.is_file():
        raise PolicyContractError(
            f"{marker}: reference-only exports can never be validated as candidates"
        )


def validate_candidate(policy_id: str, *, require_behavior: bool = False) -> ValidationReport:
    report = ValidationReport()
    try:
        policy = load_catalog().policy(policy_id)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(str(exc))
        return report
    bundle = candidate_bundle_path(policy_id)
    model_path = bundle / "policy.onnx"
    contract_path = bundle / "policy.contract.json"
    robot_definition_path = bundle / "robot.definition.json"
    torchscript_path = bundle / "policy.pt"
    try:
        _reject_reference_marker(bundle)
        contract = PolicyContract.load(contract_path, model_path)
        report.checked.extend([str(contract_path), str(model_path)])
        if contract.policy_id != policy.policy_id:
            raise PolicyContractError(
                f"Bundle policy_id '{contract.policy_id}' does not match '{policy.policy_id}'"
            )
        if contract.observation_dim != policy.observation_dim or contract.action_dim != policy.action_dim:
            raise PolicyContractError("Bundle dimensions differ from the authored catalog")
        station = load_catalog().station(policy.station_id)
        if (
            contract.source_task_id != policy.source_task_id
            or contract.station_id != policy.station_id
            or contract.robot_id != station.robot_id
            or contract.adapter_id != station.adapter_id
            or contract.physics_profile_id != station.physics_profile_id
            or contract.assist_profile_id != station.assist_profile_id
        ):
            raise PolicyContractError("Bundle IDs and profiles differ from the authored catalog")
        if (
            contract.timing.get("source_physics_hz") != policy.source_physics_hz
            or contract.timing.get("deployment_physics_hz") != policy.deployment_physics_hz
            or contract.timing.get("policy_hz") != policy.policy_hz
        ):
            raise PolicyContractError("Bundle cadence differs from the authored catalog")
        if (
            contract.evaluation.get("minimum_isaac_score") != policy.minimum_isaac_score
            or contract.evaluation.get("minimum_unity_relative_score") !=
            policy.minimum_unity_relative_score
            or contract.evaluation.get("task_thresholds") != list(policy.task_thresholds)
        ):
            raise PolicyContractError("Bundle evaluation thresholds differ from the authored catalog")
        dependency_ids = tuple(item["policy_id"] for item in contract.dependencies)
        if dependency_ids != policy.depends_on_policy_ids:
            raise PolicyContractError(
                f"Bundle dependencies {dependency_ids} differ from catalog {policy.depends_on_policy_ids}"
            )
        robot_definition = RobotDefinitionDocument.load(robot_definition_path)
        if robot_definition.robot_id != station.robot_id:
            raise RobotDefinitionError(
                f"Robot definition '{robot_definition.robot_id}' does not match '{station.robot_id}'"
            )
        report.checked.append(str(robot_definition_path))
        if not torchscript_path.is_file() or contract.torchscript_sha256 != _sha256(torchscript_path):
            raise PolicyContractError("policy.pt is missing or does not match torchscript_sha256")
        report.checked.append(str(torchscript_path))
        for dependency in contract.dependencies:
            dependency_bundle = candidate_bundle_path(dependency["policy_id"])
            dependency_contract = PolicyContract.load(
                dependency_bundle / "policy.contract.json",
                dependency_bundle / "policy.onnx",
            )
            dependency_torchscript = dependency_bundle / "policy.pt"
            if dependency_contract.model_sha256 != dependency["model_sha256"] or not dependency_torchscript.is_file() or _sha256(
                dependency_torchscript
            ) != dependency["torchscript_sha256"]:
                raise PolicyContractError(
                    f"dependency '{dependency['policy_id']}' does not match the recorded hashes"
                )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"{policy_id}: {exc}")
        return report

    if require_behavior:
        asset = load_catalog().robot_asset(station.robot_id)
        if not asset.redistribution_verified:
            report.errors.append(
                f"{policy_id}: robot asset '{station.robot_id}' redistribution license is unverified"
            )
        elif (
            robot_definition.provenance != asset.provenance or
            robot_definition.redistribution_license != asset.redistribution_license
        ):
            report.errors.append(
                f"{robot_definition_path}: provenance/license differs from deployment/stations.json"
            )
        physics_profile = next(
            item
            for item in load_catalog().physics_profiles
            if item["profile_id"] == contract.physics_profile_id
        )
        if not bool(physics_profile.get("calibrated", False)):
            report.errors.append(
                f"{policy_id}: physics profile '{contract.physics_profile_id}' is not calibrated"
            )
        else:
            try:
                _validate_calibration_profile(contract.physics_profile_id)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{policy_id}: {exc}")
        if not robot_definition.provenance.strip() or robot_definition.redistribution_license in {
            "", "UNVERIFIED"
        }:
            report.errors.append(
                f"{robot_definition_path}: provenance and redistribution license must be verified before promotion"
            )
        _validate_behavior_artifacts(
            bundle,
            policy.policy_id,
            policy.minimum_unity_relative_score,
            contract,
            robot_definition,
            report,
        )
    return report


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_behavior_artifacts(
    bundle: Path,
    policy_id: str,
    minimum_relative_score: float,
    contract: PolicyContract,
    robot_definition: RobotDefinitionDocument,
    report: ValidationReport,
) -> None:
    trace_path = bundle / "golden_trace.jsonl"
    evaluation_path = bundle / "evaluation.json"
    try:
        samples = load_golden_trace(trace_path)
        _validate_trace_samples(samples, policy_id, contract, robot_definition)
        report.checked.append(str(trace_path))
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"{trace_path}: {exc}")

    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation.get("schema_version") != 1 or evaluation.get("policy_id") != policy_id:
            raise ValueError("evaluation schema or policy_id is invalid")
        seeds = evaluation.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != 100 or len(set(seeds)) != 100:
            raise ValueError("requires exactly 100 unique paired seeds")
        if int(evaluation.get("seeded_scenarios", 0)) != 100:
            raise ValueError("seeded_scenarios must equal 100")
        evidence = evaluation.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("raw evaluation evidence is missing")
        evidence_paths: dict[str, Path] = {}
        for name in ("isaac_scenarios", "unity_scenarios", "parity_samples"):
            item = evidence.get(name)
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError(f"evaluation evidence '{name}' is invalid")
            path = (bundle / item["path"]).resolve()
            if bundle.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"evaluation evidence '{name}' escapes or is missing from bundle")
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"evaluation evidence '{name}' hash mismatch")
            evidence_paths[name] = path
        recomputed = compute_evaluation_report(
            policy_id,
            evidence_paths["isaac_scenarios"],
            evidence_paths["unity_scenarios"],
            evidence_paths["parity_samples"],
        )
        for name, value in recomputed.items():
            if evaluation.get(name) != value:
                raise ValueError(f"evaluation field '{name}' differs from raw evidence")
        isaac_score = float(evaluation["isaac_normalized_score"])
        unity_score = float(evaluation["unity_normalized_score"])
        if isaac_score <= 0 or unity_score / isaac_score < minimum_relative_score:
            raise ValueError(
                f"Unity/Isaac score {unity_score / max(isaac_score, 1e-12):.3f} "
                f"is below {minimum_relative_score:.3f}"
            )
        limits = {
            "python_unity_max_abs_error": 1.0e-4,
            "observation_max_abs_error": 1.0e-4,
            "mirror_position_max_m": 1.0e-3,
            "mirror_rotation_max_deg": 0.1,
            "p95_frame_time_ms": 13.9,
        }
        for name, limit in limits.items():
            value = float(evaluation[name])
            if value > limit:
                raise ValueError(f"{name}={value:g} exceeds {limit:g}")
        for flag in ("has_nan", "invalid_actions", "missing_joints", "joint_limit_violations"):
            if bool(evaluation.get(flag, True)):
                raise ValueError(f"{flag} must be false")
        if not bool(evaluation.get("task_gate_passed", False)):
            raise ValueError("task behavior gate did not pass")
        _validate_task_metrics(policy_id, evaluation)
        report.checked.append(str(evaluation_path))
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"{evaluation_path}: {exc}")


def _validate_trace_samples(
    samples: list[dict],
    policy_id: str,
    contract: PolicyContract,
    robot_definition: RobotDefinitionDocument,
) -> None:
    if len(samples) < 100:
        raise ValueError("golden trace requires at least 100 policy samples")
    catalog = load_catalog()
    station = catalog.station(contract.station_id)
    expected_joint_names = [item["name"] for item in robot_definition.joints]
    expected_joint_units = {
        item["name"]: (
            item["position_unit"],
            {
                "radian": "radian_per_second",
                "meter": "meter_per_second",
                "fixed": "fixed",
            }[item["position_unit"]],
        )
        for item in robot_definition.joints
    }
    expected_contact_ids = set(station.offline_bindings.contact_links)
    expected_loop_count = sum(
        item.get("topology_role") == "loop_closure"
        for item in robot_definition.auxiliary_joints
    )
    expected_objects = {
        "ball_catch.throw": 1,
        "conveyor_color.sort": 4,
        "pick_place_table.place": 4,
        "balance_bot.two_ball": 2,
        "spot.locomotion": 0,
        "spot.follow": 0,
    }[policy_id]
    last_time = -math.inf
    for index, sample in enumerate(samples, 1):
        if sample.get("schema_version") != 1:
            raise ValueError(f"line {index} golden trace schema must equal 1")
        if sample.get("policy_id") != policy_id or sample.get("station_id") != contract.station_id:
            raise ValueError(f"line {index} policy/station IDs do not match the contract")
        if not isinstance(sample.get("reset_seed"), int):
            raise ValueError(f"line {index} reset_seed must be an integer")
        sim_time = float(sample.get("sim_time_s", math.nan))
        if not math.isfinite(sim_time) or sim_time < last_time:
            raise ValueError(f"line {index} sim_time_s is invalid or out of order")
        last_time = sim_time
        for name, expected in (
            ("observations", contract.observation_dim),
            ("raw_actions", contract.action_dim),
            ("processed_actions", contract.action_dim),
        ):
            values = sample.get(name)
            if not isinstance(values, list) or len(values) != expected:
                raise ValueError(f"line {index} {name} length does not match the contract")
        joint_state = sample.get("joint_state")
        if not isinstance(joint_state, dict) or joint_state.get("names") != expected_joint_names:
            raise ValueError(f"line {index} joint names differ from robot.definition.json")
        if any(
            not isinstance(joint_state.get(name), list) or
            len(joint_state[name]) != len(expected_joint_names)
            for name in ("position_units", "velocity_units", "position", "velocity")
        ):
            raise ValueError(f"line {index} joint arrays have invalid lengths")
        expected_position_units = [joint["position_unit"] for joint in robot_definition.joints]
        expected_velocity_units = [
            {"radian": "radian_per_second", "meter": "meter_per_second", "fixed": "fixed"}[unit]
            for unit in expected_position_units
        ]
        if joint_state["position_units"] != expected_position_units or \
                joint_state["velocity_units"] != expected_velocity_units:
            raise ValueError(f"line {index} joint units differ from robot.definition.json")
        root = sample.get("root_state")
        if not isinstance(root, dict) or any(
            not isinstance(root.get(name), list) or len(root[name]) != length
            for name, length in (
                ("position", 3), ("orientation_xyzw", 4),
                ("linear_velocity", 3), ("angular_velocity", 3),
            )
        ):
            raise ValueError(f"line {index} root_state is invalid")
        objects = sample.get("object_state")
        if not isinstance(objects, list) or len(objects) != expected_objects:
            raise ValueError(f"line {index} object_state count must equal {expected_objects}")
        if any(not isinstance(item, dict) or not item.get("id") for item in objects):
            raise ValueError(f"line {index} contains an invalid object state")
        if not isinstance(sample.get("reset_state"), dict) or not isinstance(sample.get("contacts"), list) or not isinstance(sample.get("assist"), list):
            raise ValueError(f"line {index} reset/contact/assist telemetry is invalid")
        contacts = sample["contacts"]
        contact_ids = [item.get("sensor_id") for item in contacts if isinstance(item, dict)]
        if len(contact_ids) != len(contacts) or len(set(contact_ids)) != len(contact_ids) or \
                set(contact_ids) != expected_contact_ids:
            raise ValueError(f"line {index} contact sensors differ from catalog bindings")
        for contact in contacts:
            body_ids = contact.get("body_ids")
            if not isinstance(body_ids, list) or any(
                not isinstance(name, str) or not name.strip() for name in body_ids
            ) or len(set(body_ids)) != len(body_ids):
                raise ValueError(f"line {index} contains malformed contact telemetry")
        for assist in sample["assist"]:
            if not isinstance(assist, dict) or assist.get("profile_id") != contract.assist_profile_id:
                raise ValueError(f"line {index} assistance profile differs from the contract")
            if float(assist.get("force_n", -1.0)) < 0.0 or float(
                assist.get("torque_nm", -1.0)
            ) < 0.0:
                raise ValueError(f"line {index} assistance force/torque must be non-negative")
        loops = sample.get("loop_closures")
        if not isinstance(loops, list) or len(loops) != expected_loop_count:
            raise ValueError(
                f"line {index} loop-closure count must equal {expected_loop_count}"
            )
        loop_ids: set[str] = set()
        for loop in loops:
            constraint_id = loop.get("constraint_id") if isinstance(loop, dict) else None
            if not constraint_id or constraint_id in loop_ids or constraint_id not in expected_joint_units:
                raise ValueError(f"line {index} contains invalid loop-closure identity")
            loop_ids.add(constraint_id)
            expected_position_unit, expected_velocity_unit = expected_joint_units[constraint_id]
            if loop.get("position_unit") != expected_position_unit or \
                    loop.get("velocity_unit") != expected_velocity_unit:
                raise ValueError(f"line {index} loop-closure units differ from robot definition")
            if not bool(loop.get("healthy")) or float(loop.get("proxy_mass_kg", 0.0)) <= 0.0:
                raise ValueError(f"line {index} loop closure '{constraint_id}' is unhealthy")


def _validate_task_metrics(policy_id: str, evaluation: dict) -> None:
    required = {
        "ball_catch.throw": ["catch_rate", "retained_grasp_rate", "pre_contact_assist_events"],
        "conveyor_color.sort": ["correct_sort_rate", "wrong_bin_rate", "reject_rate"],
        "pick_place_table.place": [
            "contact_grasp_rate", "lift_rate", "release_rate", "stable_placement_rate"
        ],
        "balance_bot.two_ball": ["full_episode_hold_rate", "drop_rate"],
        "spot.locomotion": [
            "ten_second_stand_rate", "command_tracking_score", "foot_contact_valid_rate", "fall_rate"
        ],
        "spot.follow": [
            "follow_observation_dim", "high_level_policy_hz", "low_level_policy_hz",
            "target_distance_error_m", "heading_error_rad", "fall_rate"
        ],
    }[policy_id]
    for name in required:
        value = float(evaluation[name])
        if not math.isfinite(value):
            raise ValueError(f"task metric {name} must be finite")
    if policy_id == "ball_catch.throw" and int(evaluation["pre_contact_assist_events"]) != 0:
        raise ValueError("Ball assistance activated before valid fingertip contact")
    if policy_id == "spot.follow":
        expected = {
            "follow_observation_dim": 10.0,
            "high_level_policy_hz": 5.0,
            "low_level_policy_hz": 50.0,
        }
        for name, value in expected.items():
            if float(evaluation[name]) != value:
                raise ValueError(f"{name} must equal {value:g}")


def _validate_calibration_profile(profile_id: str) -> None:
    catalog = load_catalog()
    profile = next(
        (item for item in catalog.physics_profiles if item["profile_id"] == profile_id), None
    )
    if profile is None or not bool(profile.get("calibrated", False)):
        raise ValueError(f"physics profile '{profile_id}' is not authored as calibrated")
    stations = [item for item in catalog.stations if item.physics_profile_id == profile_id]
    root = Path(__file__).resolve().parents[5] / "deployment" / "calibration" / profile_id
    for station in stations:
        report_path = root / f"{station.station_id}.json"
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"calibration report '{report_path}' is missing or invalid: {exc}") from exc
        if (
            value.get("schema_version") != 1
            or value.get("station_id") != station.station_id
            or value.get("physics_profile_id") != profile_id
            or int(value.get("sample_count", 0)) < 100
        ):
            raise ValueError(f"calibration report '{report_path}' has invalid identity or coverage")
        evidence = value.get("evidence")
        paths: dict[str, Path] = {}
        if not isinstance(evidence, dict):
            raise ValueError(f"calibration report '{report_path}' has no raw evidence")
        for name in ("isaac_trace", "unity_trace"):
            item = evidence.get(name)
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError(f"calibration evidence '{name}' is invalid")
            path = (report_path.parent / item["path"]).resolve()
            if report_path.parent.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"calibration evidence '{name}' escapes or is missing")
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"calibration evidence '{name}' hash mismatch")
            paths[name] = path
        comparison = compare_open_loop_traces(paths["isaac_trace"], paths["unity_trace"])
        if not math.isclose(
            comparison.normalized_error,
            float(value.get("normalized_error", math.nan)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or comparison.components != value.get("components"):
            raise ValueError(f"calibration report '{report_path}' differs from raw evidence")
        maximum = float(value.get("maximum_normalized_error", math.nan))
        if not math.isfinite(maximum) or comparison.normalized_error > maximum or not bool(
            value.get("nominal_gate_passed", False)
        ):
            raise ValueError(f"calibration report '{report_path}' failed its nominal gate")


def validate_all(*, require_behavior: bool = False) -> ValidationReport:
    report = ValidationReport()
    try:
        generate(check=True)
        report.checked.append("generated deployment artifacts")
        catalog = load_catalog()
    except Exception as exc:  # noqa: BLE001
        report.errors.append(str(exc))
        return report
    for policy in catalog.policies:
        child = validate_candidate(policy.policy_id, require_behavior=require_behavior)
        report.checked.extend(child.checked)
        report.errors.extend(child.errors)
    for profile in catalog.physics_profiles:
        if bool(profile.get("calibrated", False)):
            try:
                _validate_calibration_profile(profile["profile_id"])
                report.checked.append(f"calibration:{profile['profile_id']}")
            except Exception as exc:  # noqa: BLE001
                report.errors.append(str(exc))
    return report
