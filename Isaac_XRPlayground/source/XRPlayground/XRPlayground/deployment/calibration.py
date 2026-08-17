# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded calibration utilities for seeded Isaac/Unity open-loop traces."""

from __future__ import annotations

import math
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .catalog import load_catalog, monorepo_root
from .contract import sha256_file
from .trace import load_golden_trace


@dataclass(frozen=True)
class ParameterBound:
    name: str
    lower: float
    upper: float
    initial: float
    step: float

    def validate(self) -> None:
        values = (self.lower, self.upper, self.initial, self.step)
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"Calibration parameter '{self.name}' contains non-finite values")
        if not self.name or self.lower > self.initial or self.initial > self.upper or self.step <= 0:
            raise ValueError(f"Calibration parameter '{self.name}' has invalid bounds")


@dataclass(frozen=True)
class CalibrationResult:
    parameters: dict[str, float]
    normalized_error: float
    evaluations: int


@dataclass(frozen=True)
class OpenLoopComparison:
    components: dict[str, float]
    normalized_error: float
    sample_count: int


def bounded_coordinate_search(
    parameters: list[ParameterBound],
    evaluate: Callable[[dict[str, float]], float],
    *,
    maximum_passes: int = 12,
    minimum_step_fraction: float = 1.0e-3,
) -> CalibrationResult:
    """Deterministic derivative-free fit suitable for expensive Unity trace evaluations."""
    if not parameters:
        raise ValueError("At least one calibration parameter is required")
    for parameter in parameters:
        parameter.validate()
    current = {parameter.name: parameter.initial for parameter in parameters}
    steps = {parameter.name: parameter.step for parameter in parameters}
    best = float(evaluate(dict(current)))
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("Calibration evaluator must return a finite non-negative error")
    evaluations = 1
    for _ in range(maximum_passes):
        improved = False
        for parameter in parameters:
            origin = current[parameter.name]
            local_value = origin
            local_error = best
            for direction in (-1.0, 1.0):
                candidate_value = min(
                    parameter.upper,
                    max(parameter.lower, origin + direction * steps[parameter.name]),
                )
                if candidate_value == origin:
                    continue
                candidate = dict(current)
                candidate[parameter.name] = candidate_value
                error = float(evaluate(candidate))
                evaluations += 1
                if not math.isfinite(error) or error < 0.0:
                    raise ValueError("Calibration evaluator returned an invalid error")
                if error < local_error:
                    local_error = error
                    local_value = candidate_value
            if local_error < best:
                current[parameter.name] = local_value
                best = local_error
                improved = True
        if not improved:
            for parameter in parameters:
                steps[parameter.name] *= 0.5
            if all(
                steps[parameter.name]
                <= max(parameter.upper - parameter.lower, 1.0) * minimum_step_fraction
                for parameter in parameters
            ):
                break
    return CalibrationResult(current, best, evaluations)


def derive_randomization_ranges(
    nominal: dict[str, float], residual_fraction: dict[str, float], *, margin: float = 0.2
) -> dict[str, tuple[float, float]]:
    """Expand measured cross-engine residuals by 20%, without hiding nominal mismatch."""
    if margin < 0.0:
        raise ValueError("Randomization margin must be non-negative")
    if set(nominal) != set(residual_fraction):
        raise ValueError("Nominal parameters and measured residuals must have identical keys")
    result: dict[str, tuple[float, float]] = {}
    for name, center in nominal.items():
        residual = residual_fraction[name]
        if not math.isfinite(center) or not math.isfinite(residual) or residual < 0.0:
            raise ValueError(f"Invalid nominal/residual value for '{name}'")
        radius = abs(center) * residual * (1.0 + margin)
        result[name] = (center - radius, center + radius)
    return result


def _rmse(pairs: list[tuple[float, float]], scale: float) -> float:
    if not pairs:
        return 0.0
    return math.sqrt(sum(((unity - isaac) / scale) ** 2 for isaac, unity in pairs) / len(pairs))


def _quaternion_error(isaac: list[float], unity: list[float]) -> float:
    if len(isaac) != 4 or len(unity) != 4:
        raise ValueError("orientation arrays must contain four values")
    dot = abs(sum(float(a) * float(b) for a, b in zip(isaac, unity)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot))) / math.pi


def compare_open_loop_traces(
    isaac_path: str | Path, unity_path: str | Path
) -> OpenLoopComparison:
    """Compare identical processed-action traces using normalized system-response error."""
    isaac = load_golden_trace(isaac_path)
    unity = load_golden_trace(unity_path)
    if len(isaac) != len(unity):
        raise ValueError(f"open-loop trace lengths differ: Isaac={len(isaac)}, Unity={len(unity)}")
    pairs: dict[str, list[tuple[float, float]]] = {
        "joint_position": [], "joint_velocity": [], "root_position": [],
        "root_linear_velocity": [], "root_angular_velocity": [],
        "object_position": [], "object_linear_velocity": [], "object_angular_velocity": [],
        "unity_loop_anchor_error": [], "unity_loop_axis_error": [],
    }
    rotation_errors: list[float] = []
    object_rotation_errors: list[float] = []
    for index, (source, deployment) in enumerate(zip(isaac, unity)):
        context = f"open-loop sample {index}"
        for name in ("policy_id", "station_id", "reset_seed"):
            if source.get(name) != deployment.get(name):
                raise ValueError(f"{context}: {name} differs")
        source_action = source.get("processed_actions")
        unity_action = deployment.get("processed_actions")
        if not isinstance(source_action, list) or not isinstance(unity_action, list) or (
            len(source_action) != len(unity_action)
        ):
            raise ValueError(f"{context}: processed action dimensions differ")
        if any(abs(float(a) - float(b)) > 1.0e-8 for a, b in zip(source_action, unity_action)):
            raise ValueError(f"{context}: processed actions are not identical")

        source_joint = source["joint_state"]
        unity_joint = deployment["joint_state"]
        if source_joint["names"] != unity_joint["names"]:
            raise ValueError(f"{context}: joint order differs")
        for units_field in ("position_units", "velocity_units"):
            if source_joint[units_field] != unity_joint[units_field]:
                raise ValueError(f"{context}: joint {units_field} differ")
        for field, component in (
            ("position", "joint_position"),
            ("velocity", "joint_velocity"),
        ):
            pairs[component].extend(
                (float(a), float(b))
                for a, b in zip(source_joint[field], unity_joint[field])
            )

        source_root = source["root_state"]
        unity_root = deployment["root_state"]
        for field, component in (
            ("position", "root_position"),
            ("linear_velocity", "root_linear_velocity"),
            ("angular_velocity", "root_angular_velocity"),
        ):
            pairs[component].extend(
                (float(a), float(b))
                for a, b in zip(source_root[field], unity_root[field])
            )
        rotation_errors.append(_quaternion_error(
            source_root["orientation_xyzw"], unity_root["orientation_xyzw"]
        ))

        source_objects = source["object_state"]
        unity_objects = deployment["object_state"]
        if [item["id"] for item in source_objects] != [item["id"] for item in unity_objects]:
            raise ValueError(f"{context}: object order differs")
        for source_object, unity_object in zip(source_objects, unity_objects):
            if bool(source_object["active"]) != bool(unity_object["active"]):
                raise ValueError(f"{context}: object active state differs")
            for field, component in (
                ("position", "object_position"),
                ("linear_velocity", "object_linear_velocity"),
                ("angular_velocity", "object_angular_velocity"),
            ):
                pairs[component].extend(
                    (float(a), float(b))
                    for a, b in zip(source_object[field], unity_object[field])
                )
            object_rotation_errors.append(_quaternion_error(
                source_object["orientation_xyzw"], unity_object["orientation_xyzw"]
            ))

        for loop in deployment.get("loop_closures", []):
            if not isinstance(loop, dict) or not loop.get("constraint_id"):
                raise ValueError(f"{context}: Unity loop-closure telemetry is malformed")
            position_unit = loop.get("position_unit")
            velocity_unit = loop.get("velocity_unit")
            expected_velocity_unit = {
                "radian": "radian_per_second",
                "meter": "meter_per_second",
            }.get(position_unit)
            if expected_velocity_unit is None or velocity_unit != expected_velocity_unit:
                raise ValueError(
                    f"{context}: Unity loop closure '{loop['constraint_id']}' has invalid units"
                )
            if not bool(loop.get("healthy")):
                raise ValueError(
                    f"{context}: Unity loop closure '{loop['constraint_id']}' is unhealthy"
                )
            pairs["unity_loop_anchor_error"].append(
                (0.0, float(loop["anchor_error_m"]))
            )
            pairs["unity_loop_axis_error"].append(
                (0.0, float(loop["axis_error_deg"]))
            )

    scales = {
        "joint_position": 1.0,
        "joint_velocity": 5.0,
        "root_position": 0.5,
        "root_linear_velocity": 1.0,
        "root_angular_velocity": 2.0,
        "object_position": 0.5,
        "object_linear_velocity": 1.0,
        "object_angular_velocity": 2.0,
        "unity_loop_anchor_error": 0.005,
        "unity_loop_axis_error": 2.0,
    }
    components = {name: _rmse(values, scales[name]) for name, values in pairs.items()}
    components["root_rotation"] = _rmse([(0.0, value) for value in rotation_errors], 1.0)
    components["object_rotation"] = _rmse(
        [(0.0, value) for value in object_rotation_errors], 1.0
    )
    populated = [value for name, value in components.items() if pairs.get(name) or (
        name == "root_rotation" and rotation_errors
    ) or (name == "object_rotation" and object_rotation_errors)]
    normalized_error = math.sqrt(sum(value * value for value in populated) / len(populated))
    return OpenLoopComparison(components, normalized_error, len(isaac))


def build_open_loop_comparison_report(
    station_id: str,
    physics_profile_id: str,
    isaac_path: str | Path,
    unity_path: str | Path,
    *,
    output: str | Path | None = None,
    maximum_normalized_error: float = 0.1,
) -> Path:
    """Retain hashed traces and atomically publish a nominal cross-engine comparison."""
    catalog = load_catalog()
    station = catalog.station(station_id)
    if station.physics_profile_id != physics_profile_id:
        raise ValueError(
            f"station '{station_id}' uses '{station.physics_profile_id}', not '{physics_profile_id}'"
        )
    if not math.isfinite(maximum_normalized_error) or maximum_normalized_error <= 0.0:
        raise ValueError("maximum_normalized_error must be finite and positive")
    comparison = compare_open_loop_traces(isaac_path, unity_path)
    destination = Path(output) if output else (
        monorepo_root() / "deployment" / "calibration" /
        physics_profile_id / f"{station_id}.json"
    )
    evidence_dir = destination.parent / f"{station_id}_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, dict[str, str]] = {}
    for name, source_value in (("isaac_trace", isaac_path), ("unity_trace", unity_path)):
        source = Path(source_value)
        target = evidence_dir / f"{name}.jsonl"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=evidence_dir
        )
        os.close(descriptor)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        evidence[name] = {
            "path": str(target.relative_to(destination.parent)).replace("\\", "/"),
            "sha256": sha256_file(target),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "station_id": station_id,
        "physics_profile_id": physics_profile_id,
        "sample_count": comparison.sample_count,
        "normalized_error": comparison.normalized_error,
        "maximum_normalized_error": maximum_normalized_error,
        "nominal_gate_passed": comparison.normalized_error <= maximum_normalized_error,
        "components": comparison.components,
        "evidence": evidence,
    }
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


def write_open_loop_action_sequence(
    policy_id: str,
    output: str | Path,
    *,
    seed: int = 0,
    frame_count: int | None = None,
    amplitude: float = 0.2,
) -> Path:
    """Write a deterministic, bounded multi-sine excitation for both engines."""
    policy = load_catalog().policy(policy_id)
    if frame_count is None:
        frame_count = max(100, policy.policy_hz * 10)
    if frame_count < 100:
        raise ValueError("open-loop calibration requires at least 100 policy frames")
    if not math.isfinite(amplitude) or not 0.0 < amplitude <= 1.0:
        raise ValueError("open-loop amplitude must be within (0, 1]")
    frames: list[dict[str, list[float]]] = []
    for frame in range(frame_count):
        phase = frame / max(frame_count - 1, 1)
        envelope = math.sin(math.pi * phase) ** 2
        action = [
            amplitude * envelope * math.sin(
                2.0 * math.pi * (1 + component % 5) * phase +
                component * math.pi / max(policy.action_dim, 1)
            )
            for component in range(policy.action_dim)
        ]
        frames.append({"processed_action": action})
    value = {
        "schema_version": 1,
        "station_id": policy.station_id,
        "policy_id": policy.policy_id,
        "seed": seed,
        "frames": frames,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
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
