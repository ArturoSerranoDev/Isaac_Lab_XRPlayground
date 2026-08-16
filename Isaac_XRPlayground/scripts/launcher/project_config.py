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
    "ball_catch": {
        "label": "Ball Catch (Kinova Jaco2)",
        "subtitle": "7-DoF arm · Kinova Jaco2 + 3-finger gripper",
        "tier": "Intermediate",
        "task_id": "Template-Xrplayground-Ball-Catch-Direct-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_ball_catch_direct",
        "unity_policy_folder": "BallCatch",
        "source_dir": "ball_catch",
        "description": "Kinova Jaco2 with 3-finger gripper learns to intercept tossed balls. Stepping stone toward XR player throws.",
        "bridge_port": 9090,
        "bridge_script": "scripts/bridge/run_xr_bridge.py",
    },
    "ball_catch_wrap": {
        "label": "Ball Catch — Phase Wrap",
        "subtitle": "In-hand enclose + hold (phase 1)",
        "tier": "Intermediate",
        "task_id": "Template-Xrplayground-Ball-Catch-Wrap-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_ball_catch_2phase",
        "unity_policy_folder": "BallCatch",
        "source_dir": "ball_catch",
        "description": "Phase Wrap: mostly in-hand until soft_grasp ≥ ~0.9. Then Throw-A → Throw-B.",
        "bridge_port": 9090,
        "bridge_script": "scripts/bridge/run_xr_bridge.py",
    },
    "ball_catch_throw_a": {
        "label": "Ball Catch — Throw-A (drift)",
        "subtitle": "Close-under-motion, arm frozen (phase 2a)",
        "tier": "Intermediate",
        "task_id": "Template-Xrplayground-Ball-Catch-Throw-A-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_ball_catch_2phase",
        "unity_policy_folder": "BallCatch",
        "source_dir": "ball_catch",
        "description": "Throw-A: resume wrap; in-aperture drift ramp with arm frozen. Gate: throw_rolling ≥ ~0.75.",
        "bridge_port": 9090,
        "bridge_script": "scripts/bridge/run_xr_bridge.py",
    },
    "ball_catch_throw_b": {
        "label": "Ball Catch — Throw-B (reach)",
        "subtitle": "Free-arm near throws → parabolas (phase 2b)",
        "tier": "Intermediate",
        "task_id": "Template-Xrplayground-Ball-Catch-Throw-B-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_ball_catch_2phase",
        "unity_policy_folder": "BallCatch",
        "source_dir": "ball_catch",
        "description": "Throw-B: resume Throw-A; free arm. Gate Unity on throw_soft_grasp / throw_rolling ≥ ~0.8–0.95.",
        "bridge_port": 9090,
        "bridge_script": "scripts/bridge/run_xr_bridge.py",
    },
    "ball_catch_throw": {
        "label": "Ball Catch — Phase Throw (legacy→B)",
        "subtitle": "Alias of Throw-B",
        "tier": "Intermediate",
        "task_id": "Template-Xrplayground-Ball-Catch-Throw-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_ball_catch_2phase",
        "unity_policy_folder": "BallCatch",
        "source_dir": "ball_catch",
        "description": "Legacy Throw gym ID; prefers Throw-A then Throw-B pipeline.",
        "bridge_port": 9090,
        "bridge_script": "scripts/bridge/run_xr_bridge.py",
    },
    "conveyor_color": {
        "label": "Conveyor Color Detection",
        "subtitle": "UR10e + Robotiq · pick target color from belt",
        "tier": "Intermediate",
        "task_id": "Template-Xrplayground-Conveyor-Color-Direct-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_conveyor_color_direct",
        "unity_policy_folder": "Conveyor",
        "source_dir": "conveyor_color",
        "description": "UR10e picks red/green/blue cubes from a conveyor into a bin. State-based color ID (vision later). XR spawn + mirror bridge.",
        "bridge_port": 9091,
        "bridge_script": "scripts/bridge/run_xr_bridge_conveyor.py",
    },
    "pick_place_table": {
        "label": "Table Pick & Place (Agibot A2D)",
        "subtitle": "Wall-mounted humanoid · pick pieces → bucket",
        "tier": "Intermediate",
        "task_id": "Template-Xrplayground-Pick-Place-Table-Direct-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_pick_place_table_direct",
        "unity_policy_folder": "PickPlace",
        "source_dir": "pick_place_table",
        "description": "Agibot A2D right arm picks random table pieces and places them in a front bucket. IL demo recording scaffold for later.",
        "bridge_port": 9092,
        "bridge_script": "scripts/bridge/run_xr_bridge_pick_place.py",
    },
    "balance_bot": {
        "label": "Balance Bot (Ball-on-Plate)",
        "subtitle": "2-DoF tray · hold 1 then 2 hand-sized balls",
        "tier": "Beginner",
        "task_id": "Template-Xrplayground-Balance-Bot-Direct-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_balance_bot_direct",
        "unity_policy_folder": "BalanceBot",
        "source_dir": "balance_bot",
        "description": "Kinematic 2-DOF tilting tray keeps tennis-scale balls from falling. Curriculum: 1 ball → 2 balls.",
        "bridge_port": 9093,
        "bridge_script": "scripts/bridge/run_xr_bridge_balance_bot.py",
    },
    "spot_loco": {
        "label": "Spot Locomotion (Stand→Walk)",
        "subtitle": "Manager-based · Spot flat velocity stand then walk",
        "tier": "Advanced",
        "task_id": "Template-Xrplayground-Spot-Loco-Stand-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_spot_loco",
        "unity_policy_folder": "SpotLoco",
        "source_dir": "spot_loco",
        "tasks_root": "manager_based",
        "description": "Phase A/B: Spot stand-upright then velocity-track walk. Low-level skill for follow-me.",
        "bridge_port": 9094,
        "bridge_script": "scripts/bridge/run_xr_bridge_spot.py",
        "play_task_id": "Template-Xrplayground-Spot-Loco-Walk-Play-v0",
        "walk_task_id": "Template-Xrplayground-Spot-Loco-Walk-v0",
    },
    "spot_follow": {
        "label": "Spot Follow-Me (HMD)",
        "subtitle": "Manager-based · pose nav over pretrained Spot loco",
        "tier": "Advanced",
        "task_id": "Template-Xrplayground-Spot-Follow-v0",
        "algorithms": ["PPO"],
        "default_algorithm": "PPO",
        "rl_library": "rsl_rl",
        "log_dir": "xrplayground_spot_follow",
        "unity_policy_folder": "SpotFollow",
        "source_dir": "spot_follow",
        "tasks_root": "manager_based",
        "description": "Phase C: Anymal-nav clone on Spot. High-level pose track uses PreTrainedPolicyAction loco. Unity HMD via XrFrameConverter.",
        "bridge_port": 9094,
        "bridge_script": "scripts/bridge/run_xr_bridge_spot.py",
        "play_task_id": "Template-Xrplayground-Spot-Follow-Play-v0",
        "requires_loco_policy": True,
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
    "catch": {
        "label": "Ball catch",
        "hint": "Kinova catch — low visual env count, longer training",
        "num_envs_train": 128,
        "num_envs_train_visual": 8,
        "num_envs_play": 4,
        "num_envs_demo": 4,
        "max_iterations": 2000,
        "headless_train": True,
    },
    "conveyor": {
        "label": "Conveyor color",
        "hint": "UR10e pick — moderate envs, longer horizon",
        "num_envs_train": 64,
        "num_envs_train_visual": 4,
        "num_envs_play": 2,
        "num_envs_demo": 2,
        "max_iterations": 2000,
        "headless_train": True,
    },
    "pick_place": {
        "label": "Agibot pick-place",
        "hint": "A2D table pick — moderate envs, humanoid right arm",
        "num_envs_train": 128,
        "num_envs_train_visual": 4,
        "num_envs_play": 2,
        "num_envs_demo": 2,
        "max_iterations": 2500,
        "headless_train": True,
    },
    "balance": {
        "label": "Balance bot",
        "hint": "2-DOF tray — many envs, short horizon, curriculum 1→2 balls",
        "num_envs_train": 256,
        "num_envs_train_visual": 8,
        "num_envs_play": 4,
        "num_envs_demo": 4,
        "max_iterations": 1000,
        "headless_train": True,
    },
    "spot_loco": {
        "label": "Spot loco",
        "hint": "Spot stand/walk — 128 envs, 50 Hz control, long loco train",
        "num_envs_train": 128,
        "num_envs_train_visual": 8,
        "num_envs_play": 4,
        "num_envs_demo": 4,
        "max_iterations": 2000,
        "headless_train": True,
    },
    "spot_follow": {
        "label": "Spot follow",
        "hint": "Spot nav over loco — fewer envs; set XRPLAYGROUND_SPOT_LOCO_POLICY first",
        "num_envs_train": 64,
        "num_envs_train_visual": 4,
        "num_envs_play": 2,
        "num_envs_demo": 2,
        "max_iterations": 1500,
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
    "real_time_bridge": False,
    "bridge_mode": "mirror",  # mirror | await_throw
    "bridge_action_mode": "zero",  # zero | random | policy (fallback without ckpt)
    "export_onnx_on_train": True,
    "bridge_host": "127.0.0.1",
    "bridge_port": 9090,
    "num_envs_bridge": 1,
    "device_bridge": "cuda:0",
    "physics_sync_bridge": "fabric",
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
