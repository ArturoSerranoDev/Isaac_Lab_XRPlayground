"""Shared station metadata loaded from Unity's runtime manifest.

Keep task/port/topic facts in ``Unity_XRPlayground/Assets/Resources/XRPlayground/stations.json``.
The launcher only owns presentation and training defaults.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

MONOREPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = MONOREPO_ROOT / "Unity_XRPlayground" / "Assets" / "Resources" / "XRPlayground" / "stations.json"


@lru_cache(maxsize=1)
def load_stations() -> list[dict[str, Any]]:
    """Return validated station definitions, or an empty list when the manifest is unavailable."""
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    stations = payload.get("stations", [])
    return [station for station in stations if isinstance(station, dict) and station.get("id")]


def station_for_task(task_id: str) -> dict[str, Any] | None:
    for station in load_stations():
        if task_id in station.get("task_ids", []):
            return station
    return None


def apply_station_metadata(task: dict[str, Any]) -> dict[str, Any]:
    """Overlay operational bridge metadata from the canonical station record."""
    resolved = dict(task)
    station = station_for_task(str(resolved.get("task_id", "")))
    if station is None:
        return resolved
    resolved.update(
        station_id=station["id"],
        bridge_port=int(station["port"]),
        bridge_script=station["bridge_script"],
        unity_policy_folder=station["policy_folder"],
        policy_task_id=station["policy_task_id"],
        policy_obs_dim=int(station["obs_dim"]),
        policy_action_dim=int(station["action_dim"]),
    )
    return resolved
