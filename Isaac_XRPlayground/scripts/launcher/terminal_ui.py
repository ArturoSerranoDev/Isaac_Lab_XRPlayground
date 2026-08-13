# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminal UI helpers for the XRPlayground launcher."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .project_config import (
    BANNER,
    HARDWARE_PROFILE,
    ISAACLAB_ROOT,
    ISAAC_PROJECT_ROOT,
    MONOREPO_ROOT,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    SHARED_ASSETS_ROOT,
    TASKS_SOURCE,
    UNITY_PROJECT_ROOT,
    WORKSPACE_OWNER,
)
from .project_config import physics_sync_from_config


class Style:
    """Minimal ANSI styling (Windows VT enabled when possible)."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    WHITE = "\033[97m"

    @classmethod
    def enable(cls) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    @classmethod
    def paint(cls, text: str, *codes: str) -> str:
        if not codes:
            return text
        return "".join(codes) + text + cls.RESET


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pause(message: str = "\nPress Enter to return to the menu...") -> None:
    input(message)


def shorten_path(path: Path | str, max_len: int = 58) -> str:
    text = str(path)
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3) :]


def format_timestamp(ts: float | None = None) -> str:
    when = datetime.fromtimestamp(ts or datetime.now().timestamp())
    return when.strftime("%Y-%m-%d %H:%M")


def open_path(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(target)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(target)], check=False)
    else:
        subprocess.run(["xdg-open", str(target)], check=False)


def copy_to_clipboard(text: str) -> bool:
    if os.name == "nt":
        try:
            subprocess.run("clip", input=text, text=True, check=True, shell=True)
            return True
        except (subprocess.CalledProcessError, OSError):
            return False
    if shutil.which("clip.exe"):
        try:
            subprocess.run(["clip.exe"], input=text, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, OSError):
            return False
    return False


def run_command(command: list[str], cwd: Path) -> int:
    print(Style.paint("\n  >> Running command", Style.BOLD, Style.CYAN))
    print(Style.paint("  " + format_command(command), Style.DIM))
    print()
    completed = subprocess.run(command, cwd=cwd)
    code = completed.returncode
    if code == 0:
        print(Style.paint("\n  OK — finished successfully.", Style.GREEN))
    else:
        print(Style.paint(f"\n  Exit code: {code}", Style.RED))
    return code


def format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def print_banner() -> None:
    print(Style.paint(BANNER, Style.CYAN, Style.BOLD))
    print(Style.paint(f"  {PROJECT_NAME}", Style.WHITE, Style.BOLD))
    print(Style.paint(f"  {PROJECT_TAGLINE}", Style.DIM))
    print(Style.paint(f"  Workspace: {WORKSPACE_OWNER}  ·  {HARDWARE_PROFILE}", Style.DIM))


def print_paths_block() -> None:
    rows = [
        ("Monorepo", MONOREPO_ROOT),
        ("Isaac project", ISAAC_PROJECT_ROOT),
        ("Shared assets", SHARED_ASSETS_ROOT),
        ("Isaac Lab install", ISAACLAB_ROOT),
        ("Unity (future)", UNITY_PROJECT_ROOT),
    ]
    print(Style.paint("\n  Paths", Style.YELLOW, Style.BOLD))
    for label, path in rows:
        exists = path.exists()
        marker = Style.paint("[ok]", Style.GREEN) if exists else Style.paint("[--]", Style.DIM)
        print(f"  {marker} {label:<16} {Style.paint(shorten_path(path), Style.DIM)}")


def print_status_block(python: Path, extension_ok: bool) -> None:
    print(Style.paint("\n  Environment", Style.YELLOW, Style.BOLD))
    py_marker = Style.paint("[ok]", Style.GREEN) if python.is_file() else Style.paint("[!!]", Style.RED)
    ext_marker = Style.paint("[ok]", Style.GREEN) if extension_ok else Style.paint("[!!]", Style.RED)
    print(f"  {py_marker} Python         {Style.paint(shorten_path(python), Style.DIM)}")
    print(f"  {ext_marker} XRPlayground   {'installed in env' if extension_ok else 'run: pip install -e source/XRPlayground'}")


def print_task_card(task: dict[str, Any], algo: str, active: bool = True) -> None:
    tone = Style.CYAN if active else Style.DIM
    print(Style.paint("\n  Active task", Style.YELLOW, Style.BOLD))
    print(Style.paint(f"  {task['label']}", Style.BOLD, tone))
    print(f"  {task['subtitle']}  ·  {task['tier']}")
    print(Style.paint(f"  {task['description']}", Style.DIM))
    print(f"  ID: {Style.paint(task['task_id'], Style.MAGENTA)}")
    print(f"  Algorithm: {algo}")


def print_config_summary(config: dict[str, Any], profile_label: str) -> None:
    print(Style.paint("\n  Run profile", Style.YELLOW, Style.BOLD))
    print(f"  {profile_label}")
    visual_envs = config.get("num_envs_train_visual", 32)
    print(
        f"  Train {config['num_envs_train']} envs (headless)"
        f"  ·  {visual_envs} envs (visual)"
        f"  ·  {config['max_iterations']} iters"
    )
    print(f"  Play {config['num_envs_play']} envs  ·  Demo {config['num_envs_demo']} envs  ·  seed={config['seed']}")
    train_phys = physics_sync_from_config(config, "train")
    play_phys = physics_sync_from_config(config, "play")
    train_dev = config.get("device_train", config.get("device", "cuda:0"))
    play_dev = config.get("device_play", config.get("device", "cuda:0"))
    print(
        f"  Train {train_phys}/{train_dev}"
        f"  ·  Play {play_phys}/{play_dev}"
        f"  ·  real-time play={'on' if config.get('real_time_play') else 'off'}"
    )
    mode = config.get("train_visual_mode", "ask")
    print(f"  Train default view: {mode}")
    bridge_mode = str(config.get("bridge_mode", "mirror"))
    bridge_label = "Mirror Isaac" if bridge_mode == "mirror" else "Await player throw"
    print(
        f"  XR Bridge: {bridge_label}"
        f"  ·  {config.get('bridge_host', '127.0.0.1')}:{config.get('bridge_port', 9090)}"
        f"  ·  real-time={'on' if config.get('real_time_bridge', True) else 'off'}"
    )
    if config.get("checkpoint"):
        print(f"  Checkpoint: {Style.paint(shorten_path(config['checkpoint']), Style.DIM)}")


def print_recent_runs(recent: list[dict[str, Any]]) -> None:
    if not recent:
        return
    print(Style.paint("\n  Recent runs", Style.YELLOW, Style.BOLD))
    for entry in recent[:5]:
        status = Style.paint("ok", Style.GREEN) if entry.get("exit_code") == 0 else Style.paint("fail", Style.RED)
        print(
            f"  [{status}] {entry.get('timestamp', '?')}  "
            f"{entry.get('action', '?')}  ·  {entry.get('task_label', '?')}"
        )


def print_main_menu() -> None:
    print(Style.paint("\n  Train & evaluate", Style.GREEN, Style.BOLD))
    print("  [1] Train policy        (task + run config)")
    print("  [2] Play trained policy (task + checkpoint + run config)")
    print("  [3] Demo — random actions (task + run config)")
    print("  [4] Demo — zero actions   (task + run config)")
    print("  [B] XR Bridge → Unity     (Mirror Isaac / Await throw)")
    print(Style.paint("\n  Project shortcuts", Style.BLUE, Style.BOLD))
    print("  [5] Open shared assets folder")
    print("  [6] Open active task source code")
    print("  [7] Open training logs")
    print("  [8] Open monorepo in Explorer")
    print("  [9] Launch Isaac Sim (empty GUI)")
    print("  [0] Launch Isaac Lab empty scene")
    print("  [N] Create new training (from cartpole template)")
    print(Style.paint("\n  More", Style.YELLOW, Style.BOLD))
    print("  [L] List registered environments")
    print("  [P] Apply hardware profile preset")
    print("  [H] Simulation speed guide")
    print("  [S] Settings")
    print("  [C] Copy last command to clipboard")
    print("  [Q] Quit")
