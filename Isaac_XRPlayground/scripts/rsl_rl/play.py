# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL checkpoint for XRPlayground tasks (Kit visualizer)."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402

import XRPlayground.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

parser = argparse.ArgumentParser(description="Play XRPlayground RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O (CPU-friendly stage sync).",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
# Kit window by default for visual play.
parser.set_defaults(visualizer=["kit"])
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.device is not None:
            agent_cfg.device = args_cli.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False
            print("[INFO] Fabric disabled — using USD stage sync.")

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading checkpoint: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(agent_cfg.class_name)
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        dt = env.unwrapped.step_dt
        obs = env.get_observations()

        print("[INFO] Playing — close the Kit window or Ctrl+C to stop.")
        try:
            while True:
                start = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                sleep_t = dt - (time.time() - start)
                if args_cli.real_time and sleep_t > 0:
                    time.sleep(sleep_t)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()


if __name__ == "__main__":
    main()
