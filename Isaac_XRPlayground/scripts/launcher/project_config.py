# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-specific constants for the XRPlayground launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Monorepo layout (Isaac_Lab_XRPlayground git root)
# ---------------------------------------------------------------------------
MONOREPO_ROOT = Path(__file__).resolve().parents[2].parent
ISAAC_PROJECT_ROOT = MONOREPO_ROOT / "Isaac_XRPlayground"
UNITY_PROJECT_ROOT = MONOREPO_ROOT / "Unity_XRPlayground"
SHARED_ASSETS_ROOT = MONOREPO_ROOT / "assets"
ISAACLAB_ROOT = MONOREPO_ROOT.parent / "IsaacLab"

CONFIG_PATH = ISAAC_PROJECT_ROOT / ".xr_launcher.json"
TASKS_SOURCE = (
    ISAAC_PROJECT_ROOT / "source" / "XRPlayground" / "XRPlayground" / "tasks" / "direct"
)

# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------
PROJECT_NAME = "Isaac Lab XRPlayground"
PROJECT_TAGLINE = "Sim-to-XR robotics playground"
WORKSPACE_OWNER = "Arturo"
HARDWARE_PROFILE = "RTX 3060 Ti · 8 GB VRAM"

BANNER = r"""
  ___  __   ____  _             _
 / _ \ \ \ / / _ \| | __ _ _   _| |__   ___ _ __ ___  ___
| | | | \ V / | | | |/ _` | | | | '_ \ / _ \ '__/ _ \/ __|
| |_| |  | || |_| | | (_| | |_| | |_) |  __/ | |  __/\__ \
 \___/   |_| \__\_\_|\__,_|\__,_|_.__/ \___|_|  \___||___/
""".strip("\n")

# ---------------------------------------------------------------------------
# Tasks registry — extend here when you add new environments
# ---------------------------------------------------------------------------
TASKS: dict[str, dict[str, Any]] = {
    "cartpole": {
        "label": "Cartpole Balance",
        "subtitle": "Single-agent · classic RL starter",
        "tier": "Beginner",
        "task_id": "Template-Xrplayground-Cartpole-Direct-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "log_dir": "xrplayground_cartpole_direct",
        "source_dir": "cartpole",
        "description": "Keep an inverted pendulum upright by pushing the cart. Best first training task.",
    },
    "marl": {
        "label": "Cart + Double Pendulum",
        "subtitle": "Multi-agent · template demo",
        "tier": "Advanced",
        "task_id": "Template-Xrplayground-Marl-Direct-v0",
        "algorithms": ["IPPO"],
        "default_algorithm": "IPPO",
        "log_dir": "cart_double_pendulum_direct",
        "source_dir": "xrplayground_marl",
        "description": "Two agents control cart force and pendulum torque. Useful to study MARL, not your main XR task yet.",
    },
}

# GPU-tuned presets for RTX 3060 Ti (8 GB)
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "label": "Smoke test",
        "hint": "~1 min · sanity check that training runs",
        "num_envs_train": 64,
        "num_envs_train_visual": 16,
        "num_envs_play": 8,
        "num_envs_demo": 8,
        "max_iterations": 50,
        "headless_train": True,
    },
    "learn": {
        "label": "Learning session",
        "hint": "Recommended default for your GPU",
        "num_envs_train": 512,
        "num_envs_train_visual": 32,
        "num_envs_play": 16,
        "num_envs_demo": 16,
        "max_iterations": 500,
        "headless_train": True,
    },
    "visual": {
        "label": "Visual debug",
        "hint": "See the sim while training (slower)",
        "num_envs_train": 64,
        "num_envs_train_visual": 32,
        "num_envs_play": 16,
        "num_envs_demo": 16,
        "max_iterations": 200,
        "headless_train": False,
    },
    "heavy": {
        "label": "Heavy train",
        "hint": "May hit VRAM limits — use if stable",
        "num_envs_train": 1024,
        "num_envs_train_visual": 64,
        "num_envs_play": 32,
        "num_envs_demo": 32,
        "max_iterations": 2000,
        "headless_train": True,
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "profile": "learn",
    "task_key": "cartpole",
    "num_envs_train": 512,
    "num_envs_train_visual": 32,
    "num_envs_play": 16,
    "num_envs_demo": 16,
    "device": "cuda:0",
    "device_train": "cuda:0",
    "device_play": "cuda:0",
    "device_demo": "cuda:0",
    "physics_sync": "fabric",
    "physics_sync_train": "fabric",
    "physics_sync_play": "fabric",
    "physics_sync_demo": "fabric",
    "headless_train": True,
    "train_visual_mode": "ask",  # ask | headless | visual
    "max_iterations": 500,
    "seed": 42,
    "algorithm": "",
    "checkpoint": "",
    "record_video": False,
    "real_time_play": False,
    "real_time_demo": False,
    "confirm_before_run": True,
    "show_command_preview": True,
    "recent_runs": [],
}

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".agent"}


def physics_sync_from_config(config: dict[str, Any], action: str) -> str:
    key = f"physics_sync_{action}"
    value = str(config.get(key, config.get("physics_sync", "fabric"))).lower()
    return "usd" if value == "usd" else "fabric"


def device_from_config(config: dict[str, Any], action: str) -> str:
    key = f"device_{action}"
    return str(config.get(key, config.get("device", "cuda:0")))
