# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load and validate the monorepo deployment station catalog."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CATALOG_SCHEMA_VERSION = 3


class CatalogError(ValueError):
    """Raised when the authored deployment catalog is inconsistent."""


def monorepo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def default_catalog_path() -> Path:
    return monorepo_root() / "deployment" / "stations.json"


def _require(
    data: dict[str, Any], key: str, expected: type | tuple[type, ...], context: str
) -> Any:
    if key not in data:
        raise CatalogError(f"{context} is missing required field '{key}'")
    value = data[key]
    if not isinstance(value, expected):
        expected_name = (
            " or ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise CatalogError(
            f"{context}.{key} must be {expected_name}, got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True)
class OfflineBindings:
    root_link: str
    end_effector_link: str
    contact_links: tuple[str, ...]
    robot_position_isaac: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "OfflineBindings":
        position = _require(data, "robot_position_isaac", list, context)
        entry = cls(
            root_link=_require(data, "root_link", str, context),
            end_effector_link=_require(data, "end_effector_link", str, context),
            contact_links=tuple(_require(data, "contact_links", list, context)),
            robot_position_isaac=tuple(float(value) for value in position),
        )
        if not entry.root_link.strip():
            raise CatalogError(f"{context}.root_link must be non-empty")
        if any(not isinstance(name, str) or not name.strip() for name in entry.contact_links):
            raise CatalogError(f"{context}.contact_links must contain non-empty strings")
        _unique(entry.contact_links, f"{context}.contact_links")
        if len(entry.robot_position_isaac) != 3 or any(
            not math.isfinite(value) for value in entry.robot_position_isaac
        ):
            raise CatalogError(f"{context}.robot_position_isaac must contain three finite values")
        return entry


@dataclass(frozen=True)
class StationEntry:
    station_id: str
    display_name: str
    robot_id: str
    adapter_id: str
    port: int
    physics_profile_id: str
    assist_profile_id: str
    policy_ids: tuple[str, ...]
    bridge_payloads: tuple[str, ...]
    offline_bindings: OfflineBindings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StationEntry":
        context = f"station[{data.get('station_id', '?')}]"
        return cls(
            station_id=_require(data, "station_id", str, context),
            display_name=_require(data, "display_name", str, context),
            robot_id=_require(data, "robot_id", str, context),
            adapter_id=_require(data, "adapter_id", str, context),
            port=_require(data, "port", int, context),
            physics_profile_id=_require(data, "physics_profile_id", str, context),
            assist_profile_id=_require(data, "assist_profile_id", str, context),
            policy_ids=tuple(_require(data, "policy_ids", list, context)),
            bridge_payloads=tuple(_require(data, "bridge_payloads", list, context)),
            offline_bindings=OfflineBindings.from_dict(
                _require(data, "offline_bindings", dict, context),
                f"{context}.offline_bindings",
            ),
        )


@dataclass(frozen=True)
class RobotAssetEntry:
    robot_id: str
    provenance: str
    redistribution_license: str
    license_paths: tuple[str, ...]
    redistribution_verified: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotAssetEntry":
        context = f"robot_asset[{data.get('robot_id', '?')}]"
        entry = cls(
            robot_id=_require(data, "robot_id", str, context),
            provenance=_require(data, "provenance", str, context),
            redistribution_license=_require(data, "redistribution_license", str, context),
            license_paths=tuple(_require(data, "license_paths", list, context)),
            redistribution_verified=_require(data, "redistribution_verified", bool, context),
        )
        if not entry.provenance.strip():
            raise CatalogError(f"{context}.provenance must be non-empty")
        if entry.redistribution_verified and entry.redistribution_license == "UNVERIFIED":
            raise CatalogError(f"{context} cannot verify an UNVERIFIED license")
        if entry.redistribution_verified and entry.redistribution_license != "project-authored" and not entry.license_paths:
            raise CatalogError(f"{context}.license_paths must record every redistributable source")
        if len(set(entry.license_paths)) != len(entry.license_paths):
            raise CatalogError(f"{context}.license_paths contains duplicates")
        for license_path in entry.license_paths:
            if not isinstance(license_path, str) or not license_path.strip():
                raise CatalogError(f"{context}.license_paths must contain non-empty strings")
            path = monorepo_root() / license_path
            if not path.is_file():
                raise CatalogError(f"{context}.license_paths entry does not exist: {path}")
        return entry


@dataclass(frozen=True)
class PolicyEntry:
    policy_id: str
    station_id: str
    source_task_id: str
    training_task_ids: tuple[str, ...]
    log_dir: str
    unity_folder: str
    observation_dim: int
    action_dim: int
    source_physics_hz: int
    deployment_physics_hz: int
    policy_hz: int
    primary_metric: str
    minimum_isaac_score: float
    minimum_unity_relative_score: float
    task_thresholds: tuple[dict[str, Any], ...]
    depends_on_policy_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyEntry":
        context = f"policy[{data.get('policy_id', '?')}]"
        entry = cls(
            policy_id=_require(data, "policy_id", str, context),
            station_id=_require(data, "station_id", str, context),
            source_task_id=_require(data, "source_task_id", str, context),
            training_task_ids=tuple(_require(data, "training_task_ids", list, context)),
            log_dir=_require(data, "log_dir", str, context),
            unity_folder=_require(data, "unity_folder", str, context),
            observation_dim=_require(data, "observation_dim", int, context),
            action_dim=_require(data, "action_dim", int, context),
            source_physics_hz=_require(data, "source_physics_hz", int, context),
            deployment_physics_hz=_require(data, "deployment_physics_hz", int, context),
            policy_hz=_require(data, "policy_hz", int, context),
            primary_metric=_require(data, "primary_metric", str, context),
            minimum_isaac_score=float(_require(data, "minimum_isaac_score", (int, float), context)),
            minimum_unity_relative_score=float(
                _require(data, "minimum_unity_relative_score", (int, float), context)
            ),
            task_thresholds=tuple(_require(data, "task_thresholds", list, context)),
            depends_on_policy_ids=tuple(data.get("depends_on_policy_ids", [])),
        )
        if entry.observation_dim <= 0 or entry.action_dim <= 0:
            raise CatalogError(f"{context} dimensions must be positive")
        if entry.policy_hz <= 0 or entry.deployment_physics_hz % entry.policy_hz != 0:
            raise CatalogError(
                f"{context} policy_hz must evenly divide deployment_physics_hz"
            )
        for name, score in (
            ("minimum_isaac_score", entry.minimum_isaac_score),
            ("minimum_unity_relative_score", entry.minimum_unity_relative_score),
        ):
            if not 0.0 <= score <= 1.0:
                raise CatalogError(f"{context}.{name} must be within [0, 1]")
        seen_thresholds: set[str] = set()
        for index, threshold in enumerate(entry.task_thresholds):
            threshold_context = f"{context}.task_thresholds[{index}]"
            if not isinstance(threshold, dict):
                raise CatalogError(f"{threshold_context} must be an object")
            metric = _require(threshold, "metric", str, threshold_context)
            comparison = _require(threshold, "comparison", str, threshold_context)
            _require(threshold, "value", (int, float), threshold_context)
            if comparison not in {"min", "max", "equal"}:
                raise CatalogError(f"{threshold_context}.comparison is invalid")
            if metric in seen_thresholds:
                raise CatalogError(f"{context} has duplicate task threshold '{metric}'")
            seen_thresholds.add(metric)
        return entry


@dataclass(frozen=True)
class DeploymentCatalog:
    schema_version: int
    bridge: dict[str, Any]
    runtime: dict[str, Any]
    robot_assets: tuple[RobotAssetEntry, ...]
    physics_profiles: tuple[dict[str, Any], ...]
    assist_profiles: tuple[dict[str, Any], ...]
    stations: tuple[StationEntry, ...]
    policies: tuple[PolicyEntry, ...]

    def station(self, station_id: str) -> StationEntry:
        for station in self.stations:
            if station.station_id == station_id:
                return station
        raise CatalogError(f"Unknown station_id '{station_id}'")

    def robot_asset(self, robot_id: str) -> RobotAssetEntry:
        for asset in self.robot_assets:
            if asset.robot_id == robot_id:
                return asset
        raise CatalogError(f"Unknown robot asset '{robot_id}'")

    def policy(self, policy_id: str) -> PolicyEntry:
        for policy in self.policies:
            if policy.policy_id == policy_id:
                return policy
        raise CatalogError(f"Unknown policy_id '{policy_id}'")

    def policy_for_task(self, task_id: str) -> PolicyEntry:
        matches = [policy for policy in self.policies if task_id in policy.training_task_ids]
        if len(matches) != 1:
            raise CatalogError(
                f"Expected one deployment policy for task '{task_id}', found {len(matches)}"
            )
        return matches[0]

    def to_runtime_dict(self) -> dict[str, Any]:
        policies = {policy.policy_id: policy for policy in self.policies}
        return {
            "schema_version": self.schema_version,
            "bridge": self.bridge,
            "runtime": self.runtime,
            "physics_profiles": list(self.physics_profiles),
            "assist_profiles": list(self.assist_profiles),
            "robot_assets": [asset.__dict__ for asset in self.robot_assets],
            "stations": [
                {
                    **{key: value for key, value in station.__dict__.items()
                       if key != "offline_bindings"},
                    "policy_ids": list(station.policy_ids),
                    "bridge_payloads": list(station.bridge_payloads),
                    "offline_bindings": {
                        "root_link": station.offline_bindings.root_link,
                        "end_effector_link": station.offline_bindings.end_effector_link,
                        "contact_links": list(station.offline_bindings.contact_links),
                        "robot_position_isaac": list(
                            station.offline_bindings.robot_position_isaac
                        ),
                    },
                    "policies": [
                        {
                            "policy_id": policies[policy_id].policy_id,
                            "source_task_id": policies[policy_id].source_task_id,
                            "unity_folder": policies[policy_id].unity_folder,
                            "observation_dim": policies[policy_id].observation_dim,
                            "action_dim": policies[policy_id].action_dim,
                            "physics_hz": policies[policy_id].deployment_physics_hz,
                            "policy_hz": policies[policy_id].policy_hz,
                        }
                        for policy_id in station.policy_ids
                    ],
                }
                for station in self.stations
            ],
        }


def _unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise CatalogError(f"Duplicate {label}: {', '.join(sorted(duplicates))}")


def load_catalog(path: str | Path | None = None) -> DeploymentCatalog:
    catalog_path = Path(path) if path else default_catalog_path()
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Could not load catalog '{catalog_path}': {exc}") from exc

    version = _require(data, "schema_version", int, "catalog")
    if version != CATALOG_SCHEMA_VERSION:
        raise CatalogError(
            f"Unsupported catalog schema {version}; expected {CATALOG_SCHEMA_VERSION}"
        )
    stations = tuple(StationEntry.from_dict(item) for item in _require(data, "stations", list, "catalog"))
    policies = tuple(PolicyEntry.from_dict(item) for item in _require(data, "policies", list, "catalog"))
    physics_profiles = tuple(_require(data, "physics_profiles", list, "catalog"))
    assist_profiles = tuple(_require(data, "assist_profiles", list, "catalog"))
    robot_assets = tuple(
        RobotAssetEntry.from_dict(item)
        for item in _require(data, "robot_assets", list, "catalog")
    )
    _unique((entry.station_id for entry in stations), "station_id")
    _unique((entry.policy_id for entry in policies), "policy_id")
    _unique((entry.robot_id for entry in robot_assets), "robot asset_id")
    _unique((str(entry.get("profile_id", "")) for entry in physics_profiles), "physics profile_id")
    _unique((str(entry.get("profile_id", "")) for entry in assist_profiles), "assist profile_id")
    _unique((str(entry.port) for entry in stations if entry.station_id not in {"spot_loco", "spot_follow"}), "station port")

    station_ids = {entry.station_id for entry in stations}
    policy_ids = {entry.policy_id for entry in policies}
    physics_profile_ids = {str(entry.get("profile_id", "")) for entry in physics_profiles}
    assist_profile_ids = {str(entry.get("profile_id", "")) for entry in assist_profiles}
    if "" in physics_profile_ids or "" in assist_profile_ids:
        raise CatalogError("All physics and assist profiles require a non-empty profile_id")
    for profile in physics_profiles:
        profile_id = str(profile["profile_id"])
        physics_hz = _require(profile, "physics_hz", int, f"physics_profile[{profile_id}]")
        if physics_hz <= 0:
            raise CatalogError(f"physics_profile[{profile_id}].physics_hz must be positive")
        for field_name in ("solver_iterations", "solver_velocity_iterations"):
            if _require(profile, field_name, int, f"physics_profile[{profile_id}]") <= 0:
                raise CatalogError(f"physics_profile[{profile_id}].{field_name} must be positive")
        _require(profile, "calibrated", bool, f"physics_profile[{profile_id}]")
    for profile in assist_profiles:
        profile_id = str(profile["profile_id"])
        kind = _require(profile, "kind", str, f"assist_profile[{profile_id}]")
        if kind not in {"None", "ContactGrip", "SpotUprightDamping"}:
            raise CatalogError(f"assist_profile[{profile_id}].kind is unsupported")
        required = _require(
            profile, "required_distinct_contacts", int, f"assist_profile[{profile_id}]"
        )
        if required < 0:
            raise CatalogError(
                f"assist_profile[{profile_id}].required_distinct_contacts must be non-negative"
            )
    for policy in policies:
        if policy.station_id not in station_ids:
            raise CatalogError(
                f"Policy '{policy.policy_id}' references unknown station '{policy.station_id}'"
            )
        unknown_dependencies = set(policy.depends_on_policy_ids) - policy_ids
        if unknown_dependencies:
            raise CatalogError(
                f"Policy '{policy.policy_id}' has unknown dependencies {sorted(unknown_dependencies)}"
            )
    for station in stations:
        if station.physics_profile_id not in physics_profile_ids:
            raise CatalogError(
                f"Station '{station.station_id}' references unknown physics profile "
                f"'{station.physics_profile_id}'"
            )
        if station.assist_profile_id not in assist_profile_ids:
            raise CatalogError(
                f"Station '{station.station_id}' references unknown assist profile "
                f"'{station.assist_profile_id}'"
            )
        station_physics_hz = next(
            int(item["physics_hz"])
            for item in physics_profiles
            if item["profile_id"] == station.physics_profile_id
        )
        unknown_policies = set(station.policy_ids) - policy_ids
        if unknown_policies:
            raise CatalogError(
                f"Station '{station.station_id}' references unknown policies {sorted(unknown_policies)}"
            )
        for policy_id in station.policy_ids:
            station_policy = next(item for item in policies if item.policy_id == policy_id)
            if station_policy.deployment_physics_hz != station_physics_hz:
                raise CatalogError(
                    f"Station '{station.station_id}' physics profile is {station_physics_hz} Hz but "
                    f"policy '{policy_id}' requires {station_policy.deployment_physics_hz} Hz"
                )
        mismatched = [
            policy_id
            for policy_id in station.policy_ids
            if next(policy for policy in policies if policy.policy_id == policy_id).station_id
            not in {station.station_id, "spot_loco"}
        ]
        if mismatched:
            raise CatalogError(
                f"Station '{station.station_id}' references policies owned by another station: {mismatched}"
            )

    for station in stations:
        if station.robot_id not in {asset.robot_id for asset in robot_assets}:
            raise CatalogError(f"station '{station.station_id}' has no robot asset provenance")
    return DeploymentCatalog(
        schema_version=version,
        bridge=_require(data, "bridge", dict, "catalog"),
        runtime=_require(data, "runtime", dict, "catalog"),
        robot_assets=robot_assets,
        physics_profiles=physics_profiles,
        assist_profiles=assist_profiles,
        stations=stations,
        policies=policies,
    )
