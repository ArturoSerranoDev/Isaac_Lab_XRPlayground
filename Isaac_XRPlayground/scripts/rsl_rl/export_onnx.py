# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export RSL-RL policy to ONNX (+ Unity sidecar metadata)."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys

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
from XRPlayground.deployment.exporter import export_runner_bundle  # noqa: E402

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
    help="Directory for policy.onnx, policy.pt, contract and robot definition. Defaults next to checkpoint/exported.",
)
staging = parser.add_mutually_exclusive_group()
staging.add_argument(
    "--no_copy_to_unity",
    action="store_true",
    default=False,
    help="Skip copying exports into Unity_XRPlayground Assets.",
)
staging.add_argument(
    "--reference_only",
    action="store_true",
    default=False,
    help=(
        "Stage under Unity Bundles/References for physics/inference inspection. "
        "Reference bundles are never eligible for promotion."
    ),
)
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

installed_version = metadata.version("rsl-rl-lib")

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
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
        descriptor_env = env
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

        onnx_path, contract_path, parity_error, staged = export_runner_bundle(
            runner=runner,
            descriptor_env=descriptor_env,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            task_id=args_cli.task,
            checkpoint_path=ckpt,
            export_dir=export_dir,
            copy_to_unity=not args_cli.no_copy_to_unity,
            reference_only=args_cli.reference_only,
        )
        print(f"[INFO] Wrote {onnx_path}")
        print(f"[INFO] Wrote {contract_path}")
        print(f"[INFO] PyTorch/ONNX max abs error: {parity_error:.9g}")
        if staged is not None:
            kind = "reference" if args_cli.reference_only else "candidate"
            print(f"[INFO] Staged {kind} bundle -> {staged}")

        env.close()


if __name__ == "__main__":
    main()
