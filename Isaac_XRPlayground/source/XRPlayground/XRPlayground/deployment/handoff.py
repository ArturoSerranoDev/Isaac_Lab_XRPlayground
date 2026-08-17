# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Truthful, checkpoint-free calibration and retraining handoff report."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .catalog import DeploymentCatalog, load_catalog, monorepo_root
from .contract import sha256_file
from .exporter import candidate_bundle_path
from .generate import generate
from .validation import validate_candidate


HANDOFF_SCHEMA_VERSION = 1


def _policy_order(catalog: DeploymentCatalog) -> list[str]:
    """Return a stable dependency-first order and reject catalog cycles."""
    policies = {item.policy_id: item for item in catalog.policies}
    remaining = [item.policy_id for item in catalog.policies]
    result: list[str] = []
    while remaining:
        ready = [
            policy_id
            for policy_id in remaining
            if set(policies[policy_id].depends_on_policy_ids).issubset(result)
        ]
        if not ready:
            raise ValueError(f"policy dependency cycle contains {sorted(remaining)}")
        result.extend(ready)
        remaining = [policy_id for policy_id in remaining if policy_id not in ready]
    return result


def _relative(path: Path) -> str:
    return str(path.relative_to(monorepo_root())).replace("\\", "/")


def _portable_errors(errors: list[str]) -> list[str]:
    root = str(monorepo_root())
    return [item.replace(root, ".").replace(root.replace("\\", "/"), ".") for item in errors]


def _commands(policy_id: str, phases: tuple[str, ...]) -> dict[str, Any]:
    train = [
        f"python -m XRPlayground.cli train --policy {policy_id} --phase {index} -- --headless"
        for index in range(1, len(phases) + 1)
    ]
    return {
        "definition_only": (
            f"python -m XRPlayground.cli export --policy {policy_id} "
            "--definition-only -- --device cpu --viz none"
        ),
        "open_loop_sequence": (
            f"python -m XRPlayground.cli validate --policy {policy_id} "
            f"--write-open-loop-sequence deployment/calibration/actions/{policy_id}.json"
        ),
        "train_phases": train,
        "export_candidate": (
            f"python -m XRPlayground.cli export --policy {policy_id} "
            "--checkpoint <checkpoint.pt>"
        ),
        "validate_candidate": (
            f"python -m XRPlayground.cli validate --policy {policy_id} --behavior"
        ),
    }


def build_handoff_report() -> dict[str, Any]:
    """Describe current readiness without training, exporting, or promoting anything."""
    generate(check=True)
    catalog = load_catalog()
    root = monorepo_root()
    order = _policy_order(catalog)
    policies_by_id = {item.policy_id: item for item in catalog.policies}
    behavior_ok: dict[str, bool] = {}
    entries: list[dict[str, Any]] = []

    for policy_id in order:
        policy = policies_by_id[policy_id]
        station = catalog.station(policy.station_id)
        robot = catalog.robot_asset(station.robot_id)
        profile = next(
            item for item in catalog.physics_profiles
            if item["profile_id"] == station.physics_profile_id
        )
        definition = (
            root / "Unity_XRPlayground" / "Assets" / "_Project" / "Features" /
            "Deployment" / "RobotDefinitions" / station.robot_id / "robot.definition.json"
        )
        calibration = (
            root / "deployment" / "calibration" / station.physics_profile_id /
            f"{station.station_id}.json"
        )
        bundle = candidate_bundle_path(policy_id)
        required_bundle_files = (
            "policy.onnx", "policy.pt", "policy.contract.json", "robot.definition.json"
        )
        missing_bundle_files = [
            name for name in required_bundle_files if not (bundle / name).is_file()
        ]
        static = validate_candidate(policy_id)
        behavior = validate_candidate(policy_id, require_behavior=True) if static.ok else None
        static_errors = _portable_errors(static.errors)
        behavior_errors = _portable_errors(behavior.errors) if behavior is not None else []
        behavior_ok[policy_id] = bool(behavior and behavior.ok)

        calibration_blockers: list[str] = []
        if not robot.redistribution_verified:
            calibration_blockers.append(
                f"robot '{robot.robot_id}' redistribution provenance is unverified"
            )
        if not definition.is_file():
            calibration_blockers.append("live robot definition has not been exported to Unity")
        if not calibration.is_file():
            calibration_blockers.append("paired Isaac/Unity open-loop report is missing")
        if not bool(profile.get("calibrated", False)):
            calibration_blockers.append(
                f"physics profile '{station.physics_profile_id}' is not authored as calibrated"
            )

        training_blockers = list(calibration_blockers)
        for dependency in policy.depends_on_policy_ids:
            if not behavior_ok.get(dependency, False):
                training_blockers.append(
                    f"dependency '{dependency}' has not passed its behavior gate"
                )

        promotion_blockers = list(training_blockers)
        if missing_bundle_files:
            promotion_blockers.append(
                "candidate bundle is missing " + ", ".join(missing_bundle_files)
            )
        promotion_blockers.extend(static_errors)
        if static.ok and behavior is not None:
            promotion_blockers.extend(behavior_errors)
        promotion_blockers = list(dict.fromkeys(promotion_blockers))

        entries.append(
            {
                "policy_id": policy_id,
                "station_id": station.station_id,
                "robot_id": station.robot_id,
                "physics_profile_id": station.physics_profile_id,
                "assist_profile_id": station.assist_profile_id,
                "training_task_ids": list(policy.training_task_ids),
                "depends_on_policy_ids": list(policy.depends_on_policy_ids),
                "robot_definition": {
                    "path": _relative(definition),
                    "present": definition.is_file(),
                },
                "provenance_verified": robot.redistribution_verified,
                "calibration": {
                    "profile_authored_calibrated": bool(profile.get("calibrated", False)),
                    "station_report": _relative(calibration),
                    "station_report_present": calibration.is_file(),
                    "blockers": calibration_blockers,
                },
                "candidate": {
                    "bundle": _relative(bundle),
                    "missing_files": missing_bundle_files,
                    "static_gate_passed": static.ok,
                    "static_errors": static_errors,
                    "behavior_gate_passed": behavior_ok[policy_id],
                    "behavior_errors": behavior_errors,
                },
                "training_ready": not training_blockers,
                "training_blockers": training_blockers,
                "promotion_ready": not promotion_blockers and behavior_ok[policy_id],
                "promotion_blockers": promotion_blockers,
                "commands": _commands(policy_id, policy.training_task_ids),
            }
        )

    profiles = []
    for profile in catalog.physics_profiles:
        stations = [
            station.station_id for station in catalog.stations
            if station.physics_profile_id == profile["profile_id"]
        ]
        profiles.append(
            {
                "profile_id": profile["profile_id"],
                "physics_hz": profile["physics_hz"],
                "calibrated": bool(profile.get("calibrated", False)),
                "required_station_reports": stations,
            }
        )

    complete = all(item["promotion_ready"] for item in entries)
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "catalog_sha256": sha256_file(root / "deployment" / "stations.json"),
        "policy_order": order,
        "physics_profiles": profiles,
        "policies": entries,
        "complete_six_policy_gate": complete,
        "legacy_removal_allowed": complete,
        "policies_retrained_by_this_command": False,
    }


def write_handoff_report(output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(build_handoff_report(), stream, indent=2, sort_keys=True)
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
