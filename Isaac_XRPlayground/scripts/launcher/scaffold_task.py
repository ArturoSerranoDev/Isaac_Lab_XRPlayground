# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scaffold a new direct RL training task from the cartpole template."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .project_config import ISAAC_PROJECT_ROOT, TASKS_SOURCE

USER_TASKS_PATH = ISAAC_PROJECT_ROOT / ".xr_tasks.json"
TEMPLATE_KEY = "cartpole"


def _snake_to_pascal(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def _validate_task_name(name: str) -> str:
    cleaned = name.strip().lower().replace("-", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", cleaned):
        raise ValueError("Use snake_case letters, numbers, and underscores (e.g. my_robot_arm).")
    if cleaned == TEMPLATE_KEY:
        raise ValueError(f"'{TEMPLATE_KEY}' already exists as the base template.")
    return cleaned


def _load_user_tasks() -> dict[str, dict[str, Any]]:
    if not USER_TASKS_PATH.is_file():
        return {}
    try:
        with USER_TASKS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_user_task(name: str, metadata: dict[str, Any]) -> None:
    tasks = _load_user_tasks()
    tasks[name] = metadata
    with USER_TASKS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(tasks, handle, indent=2)


def _rename_mapping(template: str, task_name: str, pascal: str) -> dict[str, str]:
    template_pascal = _snake_to_pascal(template)
    return {
        template: task_name,
        template_pascal: pascal,
        f"xrplayground_{template}_direct": f"xrplayground_{task_name}_direct",
        f"Template-Xrplayground-{template_pascal}-Direct-v0": f"Template-Xrplayground-{pascal}-Direct-v0",
    }


def _apply_replacements(text: str, mapping: dict[str, str]) -> str:
    # Replace longer keys first to avoid partial overlaps.
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(old, new)
    return text


def _target_filename(filename: str, template: str, task_name: str) -> str:
    return filename.replace(template, task_name)


def scaffold_training_task(
    raw_name: str,
    *,
    label: str,
    description: str = "",
    tier: str = "Custom",
) -> dict[str, Any]:
    """Copy cartpole task folder and register user metadata."""
    task_name = _validate_task_name(raw_name)
    template_dir = TASKS_SOURCE / TEMPLATE_KEY
    target_dir = TASKS_SOURCE / task_name

    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template not found: {template_dir}")
    if target_dir.exists():
        raise FileExistsError(f"Task folder already exists: {target_dir}")

    pascal = _snake_to_pascal(task_name)
    mapping = _rename_mapping(TEMPLATE_KEY, task_name, pascal)

    shutil.copytree(template_dir, target_dir)

    for path in sorted(target_dir.rglob("*"), reverse=True):
        if path.is_dir():
            continue
        if path.suffix in {".pyc"} or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(_apply_replacements(text, mapping), encoding="utf-8")

    # Rename files that still contain the template name.
    for path in sorted(target_dir.rglob(f"*{TEMPLATE_KEY}*")):
        if path.is_dir():
            continue
        new_path = path.with_name(_target_filename(path.name, TEMPLATE_KEY, task_name))
        if new_path != path:
            path.rename(new_path)

    task_id = f"Template-Xrplayground-{pascal}-Direct-v0"
    metadata = {
        "label": label,
        "subtitle": "Custom · single-agent PPO",
        "tier": tier,
        "description": description or f"Custom training task scaffolded from {TEMPLATE_KEY}.",
    }
    _save_user_task(task_name, metadata)

    return {
        "task_key": task_name,
        "task_id": task_id,
        "folder": str(target_dir),
        "log_dir": f"xrplayground_{task_name}_direct",
        **metadata,
    }


def load_user_task_metadata() -> dict[str, dict[str, Any]]:
    return _load_user_tasks()
