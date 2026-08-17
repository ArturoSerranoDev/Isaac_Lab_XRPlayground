# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export policy-independent Unity robot physics from a live Isaac task."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

from XRPlayground.deployment.exporter import export_robot_definition  # noqa: E402

import XRPlayground.tasks  # noqa: F401, E402

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(
    description="Export a Unity robot definition without a policy checkpoint."
)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory; defaults to logs/deployment/robot_definitions/<task>.",
)
parser.add_argument(
    "--no_copy_to_unity",
    action="store_true",
    default=False,
    help="Do not stage the definition in Unity's canonical RobotDefinitions tree.",
)
add_launcher_args(parser)
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg):
    output_dir = args_cli.output_dir or os.path.join(
        "logs", "deployment", "robot_definitions", args_cli.task
    )
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            definition_path, staged = export_robot_definition(
                descriptor_env=env,
                task_id=args_cli.task,
                export_dir=output_dir,
                copy_to_unity=not args_cli.no_copy_to_unity,
            )
            print(f"[INFO] Wrote {definition_path}")
            if staged is not None:
                print(f"[INFO] Staged Unity robot definition -> {staged}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
