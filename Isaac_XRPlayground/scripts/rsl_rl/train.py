# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train XRPlayground tasks with RSL-RL. Optional --export_onnx after finish / Ctrl+C."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

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
from XRPlayground.deployment.exporter import export_runner_bundle  # noqa: E402

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
parser.add_argument(
    "--curriculum_alpha",
    type=float,
    default=None,
    help="Optionally train at a fixed Ball Catch curriculum difficulty in [0, 1].",
)
parser.add_argument(
    "--throw_mode",
    choices=("drift", "lob"),
    default=None,
    help="Optionally force Ball Catch moving episodes to drift or true parabolic lobs.",
)
parser.add_argument(
    "--in_hand_spawn_p",
    type=float,
    default=None,
    help="Optionally override Ball Catch in-hand rehearsal probability in [0, 1].",
)
parser.add_argument(
    "--policy_std_override",
    type=float,
    default=None,
    help=(
        "After loading a checkpoint, replace the Gaussian policy std and clear "
        "its optimizer moments. Useful for deterministic-policy fine-tuning."
    ),
)
parser.add_argument(
    "--zero_arm_outputs_on_resume",
    action="store_true",
    default=False,
    help=(
        "Zero only the seven arm rows of the resumed actor output layer. "
        "Preserves the learned gripper row while relearning moving-arm catches."
    ),
)
parser.add_argument(
    "--zero_gripper_output_on_resume",
    action="store_true",
    default=False,
    help="Zero the resumed actor gripper-output row so closure is relearned from the current curriculum.",
)
parser.add_argument(
    "--arm_policy_std_override",
    type=float,
    default=None,
    help="Optionally override only the seven arm-action standard deviations after resume.",
)
parser.add_argument(
    "--gripper_policy_std_override",
    type=float,
    default=None,
    help="Optionally override only the final gripper-action standard deviation after resume.",
)
parser.add_argument(
    "--behavior_clone_teacher_steps",
    type=int,
    default=0,
    help=(
        "Before PPO, train Ball Catch's actor directly on the validated catch teacher "
        "for this many simulator steps. The teacher is used only to produce labels/rollouts."
    ),
)
parser.add_argument(
    "--behavior_clone_learning_rate",
    type=float,
    default=3.0e-4,
    help="Actor-only learning rate for --behavior_clone_teacher_steps.",
)
parser.add_argument(
    "--behavior_clone_policy_mix",
    type=float,
    default=0.15,
    help=(
        "Final fraction of learner action mixed into teacher data collection, in [0, 1]. "
        "The mix ramps from zero to expose the actor to small off-teacher errors."
    ),
)
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

def _latest_checkpoint(log_dir: str) -> Path:
    checkpoints = sorted(Path(log_dir).glob("model_*.pt"), key=lambda item: item.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {log_dir}")
    return checkpoints[-1]


def _export_onnx(
    runner, descriptor_env, env_cfg, agent_cfg, log_dir: str, task: str
) -> None:
    export_dir = os.path.join(log_dir, "exported")
    checkpoint = _latest_checkpoint(log_dir)
    model_path, contract_path, parity_error, candidate = export_runner_bundle(
        runner=runner,
        descriptor_env=descriptor_env,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        task_id=task,
        checkpoint_path=checkpoint,
        export_dir=export_dir,
        copy_to_unity=not args_cli.no_copy_to_unity,
    )
    print(f"[INFO] Validated ONNX -> {model_path}")
    print(f"[INFO] Contract -> {contract_path}")
    print(f"[INFO] PyTorch/ONNX max abs error: {parity_error:.9g}")
    if candidate is not None:
        print(f"[INFO] Staged candidate bundle -> {candidate}")


def _behavior_clone_ball_catch(runner, env, log_dir: str) -> Path | None:
    """Supervise the actor on physically validated teacher rollouts before PPO."""
    steps = int(args_cli.behavior_clone_teacher_steps)
    if steps <= 0:
        return None
    base_env = env.unwrapped
    if not hasattr(base_env, "expert_catch_actions"):
        raise ValueError("--behavior_clone_teacher_steps is only supported by Ball Catch")
    learning_rate = float(args_cli.behavior_clone_learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("--behavior_clone_learning_rate must be positive")
    policy_mix_final = float(args_cli.behavior_clone_policy_mix)
    if not 0.0 <= policy_mix_final <= 1.0:
        raise ValueError("--behavior_clone_policy_mix must be within [0, 1]")

    actor = runner.alg.actor
    actor.train()
    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=learning_rate)
    obs = env.get_observations().to(runner.device)
    replay_capacity = min(steps * env.num_envs, 131072)
    obs_dim = int(obs["policy"].shape[-1])
    replay_obs = torch.empty((replay_capacity, obs_dim), device=runner.device)
    replay_targets = torch.empty((replay_capacity, env.num_actions), device=runner.device)
    replay_count = 0
    replay_cursor = 0
    report_every = max(steps // 10, 1)
    loss_sum = 0.0
    print(
        f"[INFO] Behavior cloning Ball Catch teacher for {steps} steps "
        f"(lr={learning_rate:g}, final policy mix={policy_mix_final:g})"
    )
    for step in range(steps):
        with torch.no_grad():
            actor.update_normalization(obs)
            runner.alg.critic.update_normalization(obs)
            teacher_actions = base_env.expert_catch_actions().to(runner.device)
            batch_count = int(teacher_actions.shape[0])
            first_count = min(batch_count, replay_capacity - replay_cursor)
            replay_obs[replay_cursor : replay_cursor + first_count] = obs["policy"][:first_count]
            replay_targets[replay_cursor : replay_cursor + first_count] = teacher_actions[:first_count]
            remaining = batch_count - first_count
            if remaining > 0:
                replay_obs[:remaining] = obs["policy"][first_count:]
                replay_targets[:remaining] = teacher_actions[first_count:]
            replay_cursor = (replay_cursor + batch_count) % replay_capacity
            replay_count = min(replay_count + batch_count, replay_capacity)
            sample_count = min(1024, replay_count)
            sample_ids = torch.randint(replay_count, (sample_count,), device=runner.device)
            train_obs = TensorDict(
                {"policy": replay_obs[sample_ids]}, batch_size=[sample_count]
            )
            train_targets = replay_targets[sample_ids]

        predicted_actions = actor(train_obs, stochastic_output=False)
        arm_error = (predicted_actions[:, :-1] - train_targets[:, :-1]).square().mean(dim=-1)
        grip_error = (predicted_actions[:, -1] - train_targets[:, -1]).square()
        # Active closing/unloading frames are much rarer than the long steady
        # hold. Preserve them in every replay update instead of learning a
        # deceptively good all-zero grip output.
        grip_weight = 1.0 + 3.0 * (train_targets[:, -1].abs() > 0.10).float()
        arm_loss = arm_error.mean()
        grip_loss = (grip_weight * grip_error).mean()
        # Arm actions integrate into joint-position targets. A seemingly tiny
        # 0.005 bias accumulates into a large cup displacement over a six-second
        # hold, so arm imitation needs much tighter tolerance than grip timing.
        loss = 1000.0 * arm_loss + 2.0 * grip_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.detach())

        with torch.inference_mode():
            current_prediction = actor(obs, stochastic_output=False)
            policy_mix = policy_mix_final * float(step + 1) / float(steps)
            rollout_actions = teacher_actions * (1.0 - policy_mix)
            rollout_actions = rollout_actions + torch.clamp(
                current_prediction, -1.0, 1.0
            ) * policy_mix
            obs, _, _, _ = env.step(rollout_actions.to(env.device))
            obs = obs.to(runner.device)

        if (step + 1) % report_every == 0 or step + 1 == steps:
            mean_loss = loss_sum / float(report_every if (step + 1) % report_every == 0 else 1)
            print(
                f"[INFO] BC step {step + 1}/{steps}: loss={mean_loss:.6f}, "
                f"arm={float(arm_loss.detach()):.6f}, grip={float(grip_loss.detach()):.6f}"
            )
            loss_sum = 0.0

    # PPO's resumed Adam moments correspond to the pre-cloned actor. Clear only
    # actor state so the first PPO update cannot undo the supervised weights.
    for parameter in actor.parameters():
        for value in runner.alg.optimizer.state.get(parameter, {}).values():
            if torch.is_tensor(value):
                value.zero_()
    checkpoint = Path(log_dir) / f"model_{runner.current_learning_iteration}_bc.pt"
    if not hasattr(runner.logger, "writer"):
        runner.logger.writer = None
    runner.save(str(checkpoint))
    print(f"[INFO] Saved behavior-cloned checkpoint -> {checkpoint}")
    return checkpoint


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        if args_cli.curriculum_alpha is not None:
            alpha = float(args_cli.curriculum_alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("--curriculum_alpha must be within [0, 1]")
            if not hasattr(env_cfg, "forced_curriculum_alpha"):
                raise ValueError("--curriculum_alpha is only supported by curriculum-aware environments")
            env_cfg.forced_curriculum_alpha = alpha
        if args_cli.throw_mode is not None:
            if not hasattr(env_cfg, "force_throw_mode"):
                raise ValueError("--throw_mode is only supported by Ball Catch environments")
            env_cfg.force_throw_mode = args_cli.throw_mode
        if args_cli.in_hand_spawn_p is not None:
            in_hand_p = float(args_cli.in_hand_spawn_p)
            if not 0.0 <= in_hand_p <= 1.0:
                raise ValueError("--in_hand_spawn_p must be within [0, 1]")
            if not hasattr(env_cfg, "in_hand_spawn_p"):
                raise ValueError("--in_hand_spawn_p is only supported by Ball Catch environments")
            env_cfg.in_hand_spawn_p = in_hand_p
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
        descriptor_env = env
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
        if args_cli.zero_arm_outputs_on_resume or args_cli.zero_gripper_output_on_resume:
            if resume_path is None:
                raise ValueError("Actor output resets require --resume")
            output_layers = [
                module
                for module in runner.alg.actor.mlp.modules()
                if isinstance(module, torch.nn.Linear) and module.out_features == env.num_actions
            ]
            if len(output_layers) != 1:
                raise RuntimeError(
                    f"Expected one {env.num_actions}-output actor layer, found {len(output_layers)}"
                )
            output_layer = output_layers[0]
            with torch.no_grad():
                if args_cli.zero_arm_outputs_on_resume:
                    output_layer.weight[:-1].zero_()
                    if output_layer.bias is not None:
                        output_layer.bias[:-1].zero_()
                if args_cli.zero_gripper_output_on_resume:
                    output_layer.weight[-1].zero_()
                    if output_layer.bias is not None:
                        output_layer.bias[-1].zero_()
            # Clear Adam moments for the modified layer; the gripper row is
            # unchanged, but stale shared tensors would otherwise restore arm
            # outputs on the first optimizer step.
            for parameter in (output_layer.weight, output_layer.bias):
                if parameter is None:
                    continue
                for value in runner.alg.optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value):
                        value.zero_()
            reset_groups = []
            if args_cli.zero_arm_outputs_on_resume:
                reset_groups.append("arm")
            if args_cli.zero_gripper_output_on_resume:
                reset_groups.append("gripper")
            print(f"[INFO]: Zeroed resumed actor output groups: {', '.join(reset_groups)}")
        std_overrides_requested = any(
            value is not None
            for value in (
                args_cli.policy_std_override,
                args_cli.arm_policy_std_override,
                args_cli.gripper_policy_std_override,
            )
        )
        if std_overrides_requested:
            distribution = runner.alg.actor.distribution
            if not hasattr(distribution, "log_std_param"):
                raise RuntimeError("Policy std override requires a log-std Gaussian distribution")
            std_param = distribution.log_std_param
            with torch.no_grad():
                if args_cli.policy_std_override is not None:
                    std = float(args_cli.policy_std_override)
                    if not 0.0 < std <= 1.0:
                        raise ValueError("--policy_std_override must be within (0, 1]")
                    std_param.fill_(math.log(std))
                if args_cli.arm_policy_std_override is not None:
                    arm_std = float(args_cli.arm_policy_std_override)
                    if not 0.0 < arm_std <= 1.0:
                        raise ValueError("--arm_policy_std_override must be within (0, 1]")
                    std_param[:-1].fill_(math.log(arm_std))
                if args_cli.gripper_policy_std_override is not None:
                    grip_std = float(args_cli.gripper_policy_std_override)
                    if not 0.0 < grip_std <= 1.0:
                        raise ValueError("--gripper_policy_std_override must be within (0, 1]")
                    std_param[-1].fill_(math.log(grip_std))
            # A resumed Adam state can immediately undo the override. Preserve
            # all other optimizer state while clearing moments for log(std).
            optimizer_state = runner.alg.optimizer.state.get(std_param, {})
            for value in optimizer_state.values():
                if torch.is_tensor(value):
                    value.zero_()
            effective_std = std_param.detach().exp().cpu().tolist()
            print(f"[INFO]: Overrode resumed policy std -> {effective_std}")

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        _behavior_clone_ball_catch(runner, env, log_dir)

        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        except KeyboardInterrupt:
            print("[INFO] Training interrupted — exporting if --export_onnx was set.")
        finally:
            if args_cli.export_onnx:
                try:
                    _export_onnx(
                        runner, descriptor_env, env_cfg, agent_cfg, log_dir, args_cli.task
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] ONNX export failed: {exc}")
            env.close()


if __name__ == "__main__":
    main()
