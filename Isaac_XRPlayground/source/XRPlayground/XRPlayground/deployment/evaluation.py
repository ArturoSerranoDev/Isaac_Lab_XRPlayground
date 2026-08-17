# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Build promotion reports from paired seeded scenarios and raw parity samples."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .catalog import load_catalog
from .contract import sha256_file
from .exporter import candidate_bundle_path


SCENARIO_SCHEMA_VERSION = 1
REQUIRED_FLAGS = ("has_nan", "invalid_actions", "missing_joints", "joint_limit_violations")
TASK_METRICS = {
    "ball_catch.throw": ("catch_rate", "retained_grasp_rate", "pre_contact_assist_events"),
    "conveyor_color.sort": ("correct_sort_rate", "wrong_bin_rate", "reject_rate"),
    "pick_place_table.place": (
        "contact_grasp_rate", "lift_rate", "release_rate", "stable_placement_rate"
    ),
    "balance_bot.two_ball": ("full_episode_hold_rate", "drop_rate"),
    "spot.locomotion": (
        "ten_second_stand_rate", "command_tracking_score", "foot_contact_valid_rate", "fall_rate"
    ),
    "spot.follow": (
        "follow_observation_dim", "high_level_policy_hz", "low_level_policy_hz",
        "target_distance_error_m", "heading_error_rad", "fall_rate"
    ),
}
RATE_METRICS = {
    "catch_rate", "retained_grasp_rate", "correct_sort_rate", "wrong_bin_rate", "reject_rate",
    "contact_grasp_rate", "lift_rate", "release_rate", "stable_placement_rate",
    "full_episode_hold_rate", "drop_rate", "ten_second_stand_rate",
    "command_tracking_score", "foot_contact_valid_rate", "fall_rate",
}


class EvaluationError(ValueError):
    """Raised when raw evaluation evidence is incomplete or inconsistent."""


def _standardized_score(policy_id: str, metrics: dict[str, Any]) -> float:
    value = lambda name: float(metrics[name])
    if policy_id == "ball_catch.throw":
        return 0.5 * (value("catch_rate") + value("retained_grasp_rate"))
    if policy_id == "conveyor_color.sort":
        return max(0.0, min(1.0, 0.7 * value("correct_sort_rate") +
                            0.3 * value("reject_rate") - 0.5 * value("wrong_bin_rate")))
    if policy_id == "pick_place_table.place":
        return 0.25 * sum(value(name) for name in (
            "contact_grasp_rate", "lift_rate", "release_rate", "stable_placement_rate"))
    if policy_id == "balance_bot.two_ball":
        return value("full_episode_hold_rate")
    if policy_id == "spot.locomotion":
        return (value("ten_second_stand_rate") + value("command_tracking_score") +
                value("foot_contact_valid_rate")) / 3.0
    if policy_id == "spot.follow":
        if value("fall_rate") > 0.0:
            return 0.0
        return 1.0 / (1.0 + value("target_distance_error_m") + value("heading_error_rad"))
    raise EvaluationError(f"No standardized score for '{policy_id}'")


def _finite(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationError(f"{path} contains a non-finite value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
        return
    raise EvaluationError(f"{path} contains unsupported type {type(value).__name__}")


def load_scenarios(path: str | Path, policy_id: str) -> list[dict[str, Any]]:
    source = Path(path)
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"Could not load '{source}': {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{source}:{line_number}: {exc}") from exc
        if not isinstance(item, dict) or item.get("schema_version") != SCENARIO_SCHEMA_VERSION:
            raise EvaluationError(f"{source}:{line_number}: invalid scenario schema")
        if item.get("policy_id") != policy_id:
            raise EvaluationError(f"{source}:{line_number}: policy_id mismatch")
        seed = item.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed in seen:
            raise EvaluationError(f"{source}:{line_number}: seed must be a unique integer")
        seen.add(seed)
        score = item.get("normalized_task_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise EvaluationError(f"{source}:{line_number}: normalized_task_score is invalid")
        metrics = item.get("task_metrics")
        if not isinstance(metrics, dict):
            raise EvaluationError(f"{source}:{line_number}: task_metrics must be an object")
        missing = set(TASK_METRICS[policy_id]) - set(metrics)
        if missing:
            raise EvaluationError(f"{source}:{line_number}: missing task metrics {sorted(missing)}")
        for flag in REQUIRED_FLAGS:
            if not isinstance(item.get(flag), bool):
                raise EvaluationError(f"{source}:{line_number}: {flag} must be boolean")
        _finite(item, f"{source}:{line_number}")
        for name in RATE_METRICS.intersection(metrics):
            value = metrics[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise EvaluationError(f"{source}:{line_number}: {name} must be within [0, 1]")
        standardized = _standardized_score(policy_id, metrics)
        if not math.isclose(float(score), standardized, rel_tol=0.0, abs_tol=1.0e-5):
            raise EvaluationError(
                f"{source}:{line_number}: normalized_task_score {score} does not match "
                f"standardized task metrics {standardized}"
            )
        result.append(item)
    if len(result) < 100:
        raise EvaluationError(f"{source}: requires at least 100 seeded scenarios, found {len(result)}")
    return result


def _load_parity(path: str | Path) -> dict[str, list[float]]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not load '{source}': {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise EvaluationError("parity report schema must equal 1")
    required = {
        "python_unity_abs_errors",
        "observation_abs_errors",
        "mirror_position_errors_m",
        "mirror_rotation_errors_deg",
        "frame_times_ms",
    }
    missing = required - set(value)
    if missing:
        raise EvaluationError(f"parity report missing {sorted(missing)}")
    result: dict[str, list[float]] = {}
    for name in required:
        values = value[name]
        if not isinstance(values, list) or not values:
            raise EvaluationError(f"parity report {name} must be a non-empty array")
        result[name] = [float(item) for item in values]
        if not all(math.isfinite(item) and item >= 0.0 for item in result[name]):
            raise EvaluationError(f"parity report {name} contains invalid values")
    if len(result["frame_times_ms"]) < 100:
        raise EvaluationError("parity report requires at least 100 frame-time samples")
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _aggregate_metrics(policy_id: str, unity: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in TASK_METRICS[policy_id]:
        values = [float(item["task_metrics"][name]) for item in unity]
        if name == "pre_contact_assist_events":
            result[name] = sum(values)
        elif name in {"follow_observation_dim", "high_level_policy_hz", "low_level_policy_hz"}:
            if any(item != values[0] for item in values[1:]):
                raise EvaluationError(f"{name} changed between seeded scenarios")
            result[name] = values[0]
        else:
            result[name] = _mean(values)
    return result


def _thresholds_pass(policy: Any, metrics: dict[str, float]) -> bool:
    for threshold in policy.task_thresholds:
        name = threshold["metric"]
        actual = metrics[name]
        expected = float(threshold["value"])
        comparison = threshold["comparison"]
        if comparison == "min" and actual < expected:
            return False
        if comparison == "max" and actual > expected:
            return False
        if comparison == "equal" and not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1.0e-6
        ):
            return False
    return True


def compute_evaluation_report(
    policy_id: str,
    isaac_results: str | Path,
    unity_results: str | Path,
    parity_report: str | Path,
) -> dict[str, Any]:
    """Pair identical seeds and recompute every promotion metric from raw evidence."""
    policy = load_catalog().policy(policy_id)
    isaac = load_scenarios(isaac_results, policy_id)
    unity = load_scenarios(unity_results, policy_id)
    isaac_by_seed = {item["seed"]: item for item in isaac}
    unity_by_seed = {item["seed"]: item for item in unity}
    if set(isaac_by_seed) != set(unity_by_seed):
        missing_unity = sorted(set(isaac_by_seed) - set(unity_by_seed))
        missing_isaac = sorted(set(unity_by_seed) - set(isaac_by_seed))
        raise EvaluationError(
            f"Isaac/Unity seeds differ; missing Unity={missing_unity}, missing Isaac={missing_isaac}"
        )
    seeds = sorted(isaac_by_seed)
    if len(seeds) != 100:
        raise EvaluationError(f"evaluation requires exactly 100 paired seeds, found {len(seeds)}")

    parity = _load_parity(parity_report)
    isaac_score = _mean([float(isaac_by_seed[seed]["normalized_task_score"]) for seed in seeds])
    unity_score = _mean([float(unity_by_seed[seed]["normalized_task_score"]) for seed in seeds])
    metrics = _aggregate_metrics(policy_id, [unity_by_seed[seed] for seed in seeds])
    flags = {
        flag: any(bool(isaac_by_seed[seed][flag]) or bool(unity_by_seed[seed][flag]) for seed in seeds)
        for flag in REQUIRED_FLAGS
    }
    thresholds_passed = _thresholds_pass(policy, metrics)
    task_gate = isaac_score >= policy.minimum_isaac_score and (
        unity_score / max(isaac_score, 1.0e-12) >= policy.minimum_unity_relative_score
    ) and thresholds_passed
    task_gate = task_gate and not any(flags.values())

    return {
        "schema_version": 1,
        "policy_id": policy_id,
        "seeded_scenarios": len(seeds),
        "seeds": seeds,
        "isaac_normalized_score": isaac_score,
        "unity_normalized_score": unity_score,
        "python_unity_max_abs_error": max(parity["python_unity_abs_errors"]),
        "observation_max_abs_error": max(parity["observation_abs_errors"]),
        "mirror_position_max_m": max(parity["mirror_position_errors_m"]),
        "mirror_rotation_max_deg": max(parity["mirror_rotation_errors_deg"]),
        "p95_frame_time_ms": _percentile(parity["frame_times_ms"], 0.95),
        **flags,
        **metrics,
        "task_thresholds_passed": thresholds_passed,
        "task_gate_passed": task_gate,
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_evaluation_report(
    policy_id: str,
    isaac_results: str | Path,
    unity_results: str | Path,
    parity_report: str | Path,
    *,
    output: str | Path | None = None,
) -> Path:
    """Recompute the report and retain hash-addressed raw evidence beside it."""
    report = compute_evaluation_report(policy_id, isaac_results, unity_results, parity_report)
    destination = Path(output) if output is not None else candidate_bundle_path(policy_id) / "evaluation.json"
    evidence_dir = destination.parent / "evaluation_evidence"
    evidence_sources = {
        "isaac_scenarios": Path(isaac_results),
        "unity_scenarios": Path(unity_results),
        "parity_samples": Path(parity_report),
    }
    evidence: dict[str, dict[str, str]] = {}
    for name, source in evidence_sources.items():
        suffix = ".jsonl" if name.endswith("scenarios") else ".json"
        target = evidence_dir / f"{name}{suffix}"
        _atomic_copy(source, target)
        evidence[name] = {
            "path": str(target.relative_to(destination.parent)).replace("\\", "/"),
            "sha256": sha256_file(target),
        }
    report["evidence"] = evidence
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination
