# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discover and describe trainable XRPlayground tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .project_config import CHECKPOINT_SUFFIXES, ISAAC_PROJECT_ROOT, TASKS, TASKS_SOURCE
from .scaffold_task import load_user_task_metadata

GYM_ID_PATTERN = re.compile(r"""id\s*=\s*["']([^"']+)["']""")
LOG_DIR_PATTERN = re.compile(r"""directory:\s*["']([^"']+)["']""")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _infer_log_dir(task_folder: Path) -> str:
    for yaml_path in task_folder.rglob("skrl_*_cfg.yaml"):
        match = LOG_DIR_PATTERN.search(_read_text(yaml_path))
        if match:
            return match.group(1)
    return task_folder.name


def _infer_algorithms(task_folder: Path) -> list[str]:
    algos: list[str] = []
    for yaml_path in task_folder.glob("agents/skrl_*_cfg.yaml"):
        name = yaml_path.stem.replace("skrl_", "").replace("_cfg", "").upper()
        if name == "PPO":
            algos.append("PPO")
        elif name == "IPPO":
            algos.append("IPPO")
        elif name == "MAPPO":
            algos.append("MAPPO")
        else:
            algos.append(name)
    return algos or ["PPO"]


def discover_tasks() -> dict[str, dict[str, Any]]:
    """Merge manual TASKS metadata with folders found under tasks/direct/."""
    discovered: dict[str, dict[str, Any]] = {}

    if not TASKS_SOURCE.is_dir():
        return dict(TASKS)

    for folder in sorted(TASKS_SOURCE.iterdir()):
        if not folder.is_dir() or folder.name.startswith((".", "_")):
            continue
        init_path = folder / "__init__.py"
        if not init_path.is_file():
            continue

        key = folder.name
        init_text = _read_text(init_path)
        match = GYM_ID_PATTERN.search(init_text)
        if not match and key not in TASKS:
            continue

        manual = {**TASKS.get(key, {}), **load_user_task_metadata().get(key, {})}
        algorithms = manual.get("algorithms") or _infer_algorithms(folder)
        default_algorithm = manual.get("default_algorithm") or algorithms[0]
        task_id = manual.get("task_id") or (match.group(1) if match else key)

        has_rsl = (folder / "agents" / "rsl_rl_ppo_cfg.py").is_file()
        rl_library = manual.get("rl_library") or ("rsl_rl" if has_rsl else "skrl")
        discovered[key] = {
            "label": manual.get("label", key.replace("_", " ").title()),
            "subtitle": manual.get("subtitle", "XRPlayground task"),
            "tier": manual.get("tier", "Custom"),
            "task_id": task_id,
            "algorithms": algorithms,
            "default_algorithm": default_algorithm,
            "rl_library": rl_library,
            "log_dir": manual.get("log_dir", _infer_log_dir(folder)),
            "unity_policy_folder": manual.get("unity_policy_folder", ""),
            "source_dir": folder.name,
            "description": manual.get(
                "description",
                f"Training environment registered as {task_id}.",
            ),
        }
        for bridge_key in ("bridge_port", "bridge_script"):
            if bridge_key in manual:
                discovered[key][bridge_key] = manual[bridge_key]

    # Keep manually registered tasks even if folder naming differs
    for key, manual in TASKS.items():
        if key not in discovered and manual.get("task_id"):
            discovered[key] = dict(manual)

    return discovered


def find_latest_checkpoint(log_dir_name: str) -> str | None:
    runs = list_training_runs(log_dir_name)
    for run in runs:
        if run.get("checkpoint"):
            return run["checkpoint"]
    return None


def _collect_runs_under(logs_root: Path, backend: str) -> list[dict[str, Any]]:
    if not logs_root.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in logs_root.iterdir():
        if not run_dir.is_dir():
            continue
        checkpoint = None
        latest_mtime = run_dir.stat().st_mtime
        ckpts: list[tuple[float, Path]] = []
        # skrl: checkpoints/; rsl_rl: model_*.pt in run dir (and optional nested)
        search_roots = [run_dir]
        ckpt_subdir = run_dir / "checkpoints"
        if ckpt_subdir.is_dir():
            search_roots.append(ckpt_subdir)
        for root in search_roots:
            for ckpt in root.rglob("*"):
                if not ckpt.is_file() or ckpt.suffix not in CHECKPOINT_SUFFIXES:
                    continue
                # Prefer RSL-RL model_*.pt over exporter side artifacts
                name = ckpt.name.lower()
                if name in {"policy.pt", "policy.onnx"}:
                    continue
                ckpts.append((ckpt.stat().st_mtime, ckpt))
        if ckpts:
            latest_mtime, best = sorted(ckpts, key=lambda item: item[0])[-1]
            checkpoint = str(best)
        runs.append(
            {
                "name": f"{run_dir.name} [{backend}]",
                "folder": str(run_dir),
                "checkpoint": checkpoint,
                "mtime": latest_mtime,
                "backend": backend,
            }
        )
    return runs


def list_training_runs(log_dir_name: str) -> list[dict[str, Any]]:
    """List runs under logs/rsl_rl and logs/skrl for the experiment folder name."""
    runs = _collect_runs_under(ISAAC_PROJECT_ROOT / "logs" / "rsl_rl" / log_dir_name, "rsl_rl")
    runs.extend(_collect_runs_under(ISAAC_PROJECT_ROOT / "logs" / "skrl" / log_dir_name, "skrl"))
    runs.sort(key=lambda item: item["mtime"], reverse=True)
    return runs
