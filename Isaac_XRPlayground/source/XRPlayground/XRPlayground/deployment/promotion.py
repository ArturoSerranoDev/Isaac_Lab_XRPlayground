# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic, all-station promotion into an immutable Unity-ready release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .catalog import load_catalog, monorepo_root
from .contract import PolicyContract
from .exporter import candidate_bundle_path
from .validation import validate_all


class PromotionError(RuntimeError):
    """Raised when the complete six-station gate has not passed."""


def ready_root() -> Path:
    return (
        monorepo_root()
        / "Unity_XRPlayground"
        / "Assets"
        / "_Project"
        / "Features"
        / "Deployment"
        / "Bundles"
        / "Ready"
    )


def _release_id() -> str:
    catalog = load_catalog()
    digest = hashlib.sha256()
    for policy in sorted(catalog.policies, key=lambda item: item.policy_id):
        contract = PolicyContract.load(
            candidate_bundle_path(policy.policy_id) / "policy.contract.json",
            candidate_bundle_path(policy.policy_id) / "policy.onnx",
        )
        digest.update(policy.policy_id.encode("utf-8"))
        digest.update(contract.model_sha256.encode("ascii"))
        digest.update(contract.config_sha256.encode("ascii"))
    return digest.hexdigest()[:20]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def promote_all() -> Path:
    """Promote only when every policy passes the complete behavioral gate."""
    report = validate_all(require_behavior=True)
    if not report.ok:
        raise PromotionError("Promotion blocked:\n" + "\n".join(report.errors))

    catalog = load_catalog()
    release_id = _release_id()
    root = ready_root()
    releases = root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    release = releases / release_id
    policy_entries: list[dict[str, str]] = []
    if not release.exists():
        temporary = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=releases))
        try:
            for policy in catalog.policies:
                source = candidate_bundle_path(policy.policy_id)
                destination = temporary / "policies" / policy.policy_id
                shutil.copytree(source, destination)
                contract_path = destination / "policy.contract.json"
                contract = PolicyContract.load(contract_path, destination / "policy.onnx")
                contract.evaluation = {**contract.evaluation, "ready": True, "release_id": release_id}
                contract.write_atomic(contract_path)
            _atomic_json(
                temporary / "release.json",
                {
                    "schema_version": 1,
                    "release_id": release_id,
                    "policy_ids": [item.policy_id for item in catalog.policies],
                    "complete_station_gate": True,
                },
            )
            os.replace(temporary, release)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    for policy in catalog.policies:
        policy_entries.append(
            {
                "policy_id": policy.policy_id,
                "station_id": policy.station_id,
                "bundle": f"releases/{release_id}/policies/{policy.policy_id}",
            }
        )
    _atomic_json(
        root / "catalog.json",
        {
            "schema_version": 1,
            "active_release": release_id,
            "complete_station_gate": True,
            "policies": policy_entries,
        },
    )
    return release

