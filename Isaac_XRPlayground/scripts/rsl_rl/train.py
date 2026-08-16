# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train XRPlayground tasks with RSL-RL. Optional --export_onnx after finish / Ctrl+C."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# Ensure local cli_args import works when launched as a file path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402
from unity_onnx import export_policy_onnx_for_unity  # noqa: E402

import XRPlayground.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

RSL_RL_VERSION = "5.0.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description="Train XRPlayground with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument(
    "--export_onnx",
    action="store_true",
    default=False,
    help="After training finishes or on Ctrl+C, export policy.onnx (+ Unity copy).",
)
parser.add_argument(
    "--no_copy_to_unity",
    action="store_true",
    default=False,
    help="With --export_onnx, skip copying into Unity_XRPlayground Assets.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args

if args_cli.video:
    args_cli.enable_cameras = True

installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(f"Please install rsl-rl-lib=={RSL_RL_VERSION} (found {installed_version})")
    sys.exit(1)

TASK_UNITY = {
    "Template-Xrplayground-Conveyor-Color-Direct-v0": ("Conveyor", 66, 7, 4.0),
    "Template-Xrplayground-Ball-Catch-Direct-v0": ("BallCatch", 30, 8, 5.0),
    "Template-Xrplayground-Ball-Catch-Wrap-v0": ("BallCatch", 30, 8, 5.0),
    "Template-Xrplayground-Ball-Catch-Throw-A-v0": ("BallCatch", 30, 8, 5.0),
    "Template-Xrplayground-Ball-Catch-Throw-B-v0": ("BallCatch", 30, 8, 5.0),
    "Template-Xrplayground-Ball-Catch-Throw-v0": ("BallCatch", 30, 8, 5.0),
    "Template-Xrplayground-Pick-Place-Table-Direct-v0": ("PickPlace", 30, 8, 5.0),
}


def _export_onnx(runner, log_dir: str, task: str) -> None:
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)
    try:
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] JIT export skipped: {exc}")
    export_policy_onnx_for_unity(runner, export_dir, filename="policy.onnx")

    unity_folder, obs_dim, action_dim, action_scale = TASK_UNITY.get(task, ("Unknown", -1, -1, 1.0))
    sidecar = {
        "task_id": task,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "action_scale": action_scale,
        "dt": 1.0 / 60.0,
        "frame": "isaac_env",
        "runtime": "unity_inference_engine",
    }
    json_path = os.path.join(export_dir, "policy.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    onnx_path = os.path.join(export_dir, "policy.onnx")
    print(f"[INFO] ONNX exported -> {onnx_path}")

    if not args_cli.no_copy_to_unity and unity_folder != "Unknown":
        monorepo = Path(__file__).resolve().parents[3]
        dest = monorepo / "Unity_XRPlayground" / "Assets" / "_Project" / "Features" / "Policies" / unity_folder
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, dest / "policy.onnx")
        shutil.copy2(json_path, dest / "policy.json")
        print(f"[INFO] Copied to Unity -> {dest}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        if not args_cli.distributed:
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        resume_path = None
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        if getattr(args_cli, "deterministic", False):
            configure_seed(env_cfg.seed, True)
        runner.add_git_repo_to_log(__file__)
        if resume_path is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        except KeyboardInterrupt:
            print("[INFO] Training interrupted — exporting if --export_onnx was set.")
        finally:
            if args_cli.export_onnx:
                try:
                    _export_onnx(runner, log_dir, args_cli.task)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] ONNX export failed: {exc}")
            env.close()


if __name__ == "__main__":
    main()
