# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export RSL-RL policy to ONNX (+ Unity sidecar metadata)."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # noqa: E402
from unity_onnx import export_policy_onnx_for_unity  # noqa: E402

import XRPlayground.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

parser = argparse.ArgumentParser(description="Export RSL-RL policy to ONNX for Unity.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--ckpt", type=str, required=True, help="Path to RSL-RL checkpoint (.pt).")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for policy.onnx / policy.json. Defaults next to checkpoint/exported.",
)
parser.add_argument(
    "--no_copy_to_unity",
    action="store_true",
    default=False,
    help="Skip copying exports into Unity_XRPlayground Assets.",
)
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

installed_version = metadata.version("rsl-rl-lib")

TASK_META = {
    "Template-Xrplayground-Conveyor-Color-Direct-v0": {
        "unity_folder": "Conveyor",
        "obs_dim": 66,
        "action_dim": 7,
        "action_scale": 4.0,
        "dt": 1.0 / 60.0,
    },
    "Template-Xrplayground-Ball-Catch-Direct-v0": {
        "unity_folder": "BallCatch",
        "obs_dim": 30,
        "action_dim": 8,
        "action_scale": 5.0,
        "dt": 1.0 / 60.0,
    },
    "Template-Xrplayground-Ball-Catch-Wrap-v0": {
        "unity_folder": "BallCatch",
        "obs_dim": 30,
        "action_dim": 8,
        "action_scale": 5.0,
        "dt": 1.0 / 60.0,
    },
    "Template-Xrplayground-Ball-Catch-Throw-A-v0": {
        "unity_folder": "BallCatch",
        "obs_dim": 30,
        "action_dim": 8,
        "action_scale": 5.0,
        "dt": 1.0 / 60.0,
    },
    "Template-Xrplayground-Ball-Catch-Throw-B-v0": {
        "unity_folder": "BallCatch",
        "obs_dim": 30,
        "action_dim": 8,
        "action_scale": 5.0,
        "dt": 1.0 / 60.0,
    },
    "Template-Xrplayground-Ball-Catch-Throw-v0": {
        "unity_folder": "BallCatch",
        "obs_dim": 30,
        "action_dim": 8,
        "action_scale": 5.0,
        "dt": 1.0 / 60.0,
    },
    "Template-Xrplayground-Balance-Bot-Direct-v0": {
        "unity_folder": "BalanceBot",
        "obs_dim": 20,
        "action_dim": 2,
        "action_scale": 1.5,
        "dt": 1.0 / 60.0,
    },
    "Template-Xrplayground-Spot-Loco-Stand-v0": {
        "unity_folder": "SpotLoco",
        "obs_dim": 48,
        "action_dim": 12,
        "action_scale": 0.2,
        "dt": 0.02,
    },
    "Template-Xrplayground-Spot-Loco-Walk-v0": {
        "unity_folder": "SpotLoco",
        "obs_dim": 48,
        "action_dim": 12,
        "action_scale": 0.2,
        "dt": 0.02,
    },
    "Template-Xrplayground-Spot-Loco-Walk-Play-v0": {
        "unity_folder": "SpotLoco",
        "obs_dim": 48,
        "action_dim": 12,
        "action_scale": 0.2,
        "dt": 0.02,
    },
    "Template-Xrplayground-Spot-Follow-v0": {
        "unity_folder": "SpotFollow",
        "obs_dim": 9,
        "action_dim": 3,
        "action_scale": 1.0,
        "dt": 0.2,
    },
    "Template-Xrplayground-Spot-Follow-Play-v0": {
        "unity_folder": "SpotFollow",
        "obs_dim": 9,
        "action_dim": 3,
        "action_scale": 1.0,
        "dt": 0.2,
    },
}


def _export_runner(runner, export_dir: str) -> str:
    os.makedirs(export_dir, exist_ok=True)
    # Always use Unity-safe TorchScript/opset-15 export. RSL-RL 5 + PyTorch 2.9+
    # default dynamo/opset-18 ONNX crashes the Unity importer.
    try:
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] JIT export skipped: {exc}")
    return export_policy_onnx_for_unity(runner, export_dir, filename="policy.onnx")


def _write_sidecar(path: str, task: str, meta: dict, checkpoint: str) -> None:
    payload = {
        "task_id": task,
        "obs_dim": meta["obs_dim"],
        "action_dim": meta["action_dim"],
        "action_scale": meta["action_scale"],
        "dt": meta["dt"],
        "frame": "isaac_env",
        "checkpoint": os.path.abspath(checkpoint),
        "runtime": "unity_inference_engine",
        "notes": "Obs must match Isaac DirectRL policy observation layout. Actions are mean (deterministic).",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _copy_to_unity(onnx_path: str, json_path: str, unity_folder: str) -> Path | None:
    monorepo = Path(__file__).resolve().parents[3]
    dest = monorepo / "Unity_XRPlayground" / "Assets" / "_Project" / "Features" / "Policies" / unity_folder
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dest / "policy.onnx")
    shutil.copy2(json_path, dest / "policy.json")
    print(f"[INFO] Copied ONNX -> {dest}")
    return dest


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    meta = TASK_META.get(args_cli.task, {"unity_folder": "Unknown", "obs_dim": -1, "action_dim": -1, "action_scale": 1.0, "dt": 0.0167})
    ckpt = os.path.abspath(args_cli.ckpt)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    export_dir = args_cli.output_dir or os.path.join(os.path.dirname(ckpt), "exported")
    export_dir = os.path.abspath(export_dir)

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(agent_cfg.class_name)

        configure_seed(env_cfg.seed, False)
        print(f"[INFO] Loading checkpoint: {ckpt}")
        runner.load(ckpt)

        onnx_path = _export_runner(runner, export_dir)
        json_path = os.path.join(export_dir, "policy.json")
        _write_sidecar(json_path, args_cli.task, meta, ckpt)
        print(f"[INFO] Wrote {onnx_path}")
        print(f"[INFO] Wrote {json_path}")

        if not args_cli.no_copy_to_unity:
            _copy_to_unity(onnx_path, json_path, meta["unity_folder"])

        env.close()


if __name__ == "__main__":
    main()
