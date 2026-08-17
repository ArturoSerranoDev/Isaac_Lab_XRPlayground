# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministically evaluate a Ball Catch checkpoint on true parabolic lobs.

Unlike training summaries, this runner forces zero in-hand/drift spawns and
reports full-episode gentle retention. It exits non-zero when the requested
gate is missed, which makes checkpoint selection objective and repeatable.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import json
import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402

import XRPlayground.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Evaluate Ball Catch gentle hold on true lobs.")
parser.add_argument("--task", default="Template-Xrplayground-Ball-Catch-Throw-B-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--difficulty", type=float, default=0.50)
parser.add_argument("--throw_mode", choices=("in_hand", "drift", "lob"), default="lob")
parser.add_argument(
    "--in_hand_along",
    type=float,
    default=None,
    help="Diagnostic: force an in-hand spawn fraction along the EE-to-tip axis.",
)
parser.add_argument(
    "--in_hand_offset",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Diagnostic world-space XYZ offset for an in-hand spawn.",
)
parser.add_argument(
    "--arm_joint_offsets",
    type=float,
    nargs=7,
    metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    default=None,
    help="Diagnostic reset offsets for the seven arm joints.",
)
parser.add_argument(
    "--controller",
    choices=("policy", "timed_grip", "cartesian_catch", "expert_teacher", "open_grip"),
    default="policy",
    help="Use the checkpoint policy or a non-exported diagnostic controller.",
)
parser.add_argument(
    "--grip_lead_time",
    type=float,
    default=0.18,
    help="Seconds before closest approach when the diagnostic controller begins closing.",
)
parser.add_argument(
    "--grip_target_close",
    type=float,
    default=0.56,
    help="Normalized closure target for the diagnostic timing controller.",
)
parser.add_argument(
    "--grip_feedback",
    choices=("mean", "base"),
    default="mean",
    help="Diagnostic scalar feedback: all six finger joints or the three proximal tendon joints.",
)
parser.add_argument(
    "--unload_on_contact",
    action="store_true",
    help="Diagnostic: remove accumulated finger drive error as soon as enclosure begins.",
)
parser.add_argument(
    "--hold_actual_on_contact",
    action="store_true",
    help="Diagnostic: hold the measured joint position at first enclosure instead of jumping to a preset closure.",
)
parser.add_argument(
    "--grip_hold_close",
    type=float,
    default=0.38,
    help="Normalized fixed finger target used after diagnostic contact unloading.",
)
parser.add_argument(
    "--grip_final_hold_close",
    type=float,
    default=None,
    help="Optional firmer normalized hold target reached gradually after contact.",
)
parser.add_argument(
    "--grip_hold_ramp_steps",
    type=float,
    default=30.0,
    help="Control steps used to ramp from contact closure to final hold closure.",
)
parser.add_argument(
    "--firm_hold_below_speed",
    type=float,
    default=None,
    help="Diagnostic: switch to the final hold once hand/ball relative speed is below this value.",
)
parser.add_argument(
    "--cushion_lead_time",
    type=float,
    default=0.12,
    help="Seconds over which the Cartesian diagnostic prepares and cushions impact.",
)
parser.add_argument(
    "--cushion_damping",
    type=float,
    default=0.08,
    help="Damped-least-squares regularization for the Cartesian diagnostic.",
)
parser.add_argument(
    "--cushion_velocity_gain",
    type=float,
    default=1.0,
    help="Velocity feed-forward gain for the Cartesian diagnostic.",
)
parser.add_argument(
    "--cushion_settle_steps",
    type=float,
    default=60.0,
    help="Control steps over which caught hand/ball velocity is gently removed.",
)
parser.add_argument(
    "--cushion_post_gain",
    type=float,
    default=1.0,
    help="Actuator-lag compensation applied to hand velocity after latch.",
)
parser.add_argument(
    "--cushion_no_downward_follow",
    action="store_true",
    help="Diagnostic: after latch, carry incoming momentum but do not chase a ball downward toward the floor.",
)
parser.add_argument("--min_gentle_rate", type=float, default=0.80)
parser.add_argument("--max_drop_rate", type=float, default=0.20)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--capture_dir",
    type=Path,
    default=None,
    help="Diagnostic: save three close camera views at reset, latch, and escape.",
)
parser.add_argument(
    "--print_robot_prims",
    action="store_true",
    help="Diagnostic: print the env_0 robot USD hierarchy and applied physics schemas.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
parser.set_defaults(headless=True, visualizer=["none"])
args_cli, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> int:
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.scene.num_envs = int(args_cli.num_envs)
        env_cfg.seed = int(args_cli.seed)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.device is not None:
            agent_cfg.device = args_cli.device

        if not isinstance(env_cfg, DirectRLEnvCfg):
            raise TypeError("Ball Catch evaluation requires a DirectRLEnvCfg")
        if not 0.0 <= float(args_cli.difficulty) <= 1.0:
            raise ValueError("--difficulty must be within [0, 1]")

        # Environment state cannot change the requested evaluation profile.
        env_cfg.in_hand_spawn_p = 1.0 if args_cli.throw_mode == "in_hand" else 0.0
        env_cfg.force_throw_mode = None if args_cli.throw_mode == "in_hand" else args_cli.throw_mode
        env_cfg.forced_curriculum_alpha = float(args_cli.difficulty)
        if args_cli.in_hand_along is not None:
            along = float(args_cli.in_hand_along)
            env_cfg.spawn_in_hand_along = (along, along)
            env_cfg.spawn_in_hand_jitter = 0.0
            env_cfg.spawn_in_hand_speed = 0.0
        if args_cli.in_hand_offset is not None:
            env_cfg.spawn_in_hand_offset = tuple(float(value) for value in args_cli.in_hand_offset)
        if args_cli.arm_joint_offsets is not None:
            env_cfg.reset_arm_joint_offsets = tuple(
                float(value) for value in args_cli.arm_joint_offsets
            )
        env_cfg.diagnostic_camera = args_cli.capture_dir is not None

        env = gym.make(args_cli.task, cfg=env_cfg)
        if args_cli.print_robot_prims:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith("/World/envs/env_0/Robot"):
                    continue
                schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
                print(
                    "ROBOT_PRIM",
                    path,
                    f"type={prim.GetTypeName()}",
                    f"instance={prim.IsInstance()}",
                    f"proxy={prim.IsInstanceProxy()}",
                    f"schemas={schemas}",
                )
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        initial_tip_center = base_env._tip_center_w()[0]
        initial_ee_position = base_env.robot.data.body_pos_w[0, base_env._ee_body_idx]
        initial_ee_position = (
            initial_ee_position.torch
            if hasattr(initial_ee_position, "torch")
            else initial_ee_position
        )
        initial_grasp_axis = initial_tip_center - initial_ee_position
        initial_grasp_axis = initial_grasp_axis / torch.linalg.norm(initial_grasp_axis).clamp(min=1e-6)
        if args_cli.print_robot_prims:
            initial_ee_quat = base_env.robot.data.body_quat_w[0, base_env._ee_body_idx]
            initial_ee_quat = (
                initial_ee_quat.torch if hasattr(initial_ee_quat, "torch") else initial_ee_quat
            )
            link_7_idx = int(base_env.robot.find_bodies("j2n7s300_link_7")[0][0])
            initial_link_7_position = base_env.robot.data.body_pos_w[0, link_7_idx]
            initial_link_7_quat = base_env.robot.data.body_quat_w[0, link_7_idx]
            initial_link_7_position = (
                initial_link_7_position.torch
                if hasattr(initial_link_7_position, "torch")
                else initial_link_7_position
            )
            initial_link_7_quat = (
                initial_link_7_quat.torch
                if hasattr(initial_link_7_quat, "torch")
                else initial_link_7_quat
            )
            print(
                "EE_FRAME",
                f"position={[float(value) for value in initial_ee_position.tolist()]}",
                f"quat_xyzw={[float(value) for value in initial_ee_quat.tolist()]}",
                f"axis_world={[float(value) for value in initial_grasp_axis.tolist()]}",
            )
            print(
                "LINK7_FRAME",
                f"position={[float(value) for value in initial_link_7_position.tolist()]}",
                f"quat_xyzw={[float(value) for value in initial_link_7_quat.tolist()]}",
            )
            from isaaclab.utils.math import quat_apply

            support_translation = torch.tensor(
                base_env.cfg.palm_support_translation,
                device=initial_link_7_position.device,
                dtype=initial_link_7_position.dtype,
            )
            support_xyzw = base_env.cfg.palm_support_orientation
            support_quat = torch.tensor(
                support_xyzw,
                device=initial_link_7_position.device,
                dtype=initial_link_7_position.dtype,
            )
            local_normal = quat_apply(
                support_quat.unsqueeze(0),
                torch.tensor(
                    ((0.0, 0.0, 1.0),),
                    device=initial_link_7_position.device,
                    dtype=initial_link_7_position.dtype,
                ),
            )[0]
            support_center = initial_link_7_position + quat_apply(
                initial_link_7_quat.unsqueeze(0), support_translation.unsqueeze(0)
            )[0]
            support_normal = quat_apply(
                initial_link_7_quat.unsqueeze(0), local_normal.unsqueeze(0)
            )[0]
            initial_ball_position = base_env.ball.data.root_pos_w[0]
            initial_ball_position = (
                initial_ball_position.torch
                if hasattr(initial_ball_position, "torch")
                else initial_ball_position
            )
            ball_height = torch.dot(initial_ball_position - support_center, support_normal)
            print(
                "PALM_SUPPORT_FRAME",
                f"center={[float(value) for value in support_center.tolist()]}",
                f"normal={[float(value) for value in support_normal.tolist()]}",
                f"ball={[float(value) for value in initial_ball_position.tolist()]}",
                f"ball_height={float(ball_height)}",
            )

        capture_dir = args_cli.capture_dir
        diagnostic_camera = getattr(base_env, "diagnostic_camera", None)
        captured_phases: set[str] = set()
        if capture_dir is not None:
            if diagnostic_camera is None:
                raise RuntimeError("Diagnostic camera was requested but not created")
            capture_dir.mkdir(parents=True, exist_ok=True)
            ee_position = base_env.robot.data.body_pos_w[0, base_env._ee_body_idx]
            ee_position = ee_position.torch if hasattr(ee_position, "torch") else ee_position
            camera_positions = torch.stack(
                (
                    ee_position + torch.tensor((0.32, 0.0, 0.16), device=ee_position.device),
                    ee_position + torch.tensor((0.0, 0.32, 0.16), device=ee_position.device),
                    ee_position + torch.tensor((-0.24, -0.24, 0.12), device=ee_position.device),
                )
            )
            camera_targets = ee_position.unsqueeze(0).repeat(3, 1)
            diagnostic_camera.set_world_poses_from_view(camera_positions, camera_targets)

            def capture_phase(label: str) -> None:
                from PIL import Image

                ball_position = base_env.ball.data.root_pos_w[0]
                ball_position = (
                    ball_position.torch if hasattr(ball_position, "torch") else ball_position
                )
                camera_positions = torch.stack(
                    (
                        ball_position + torch.tensor((0.20, 0.20, 0.10), device=ball_position.device),
                        ball_position + torch.tensor((0.20, -0.20, 0.10), device=ball_position.device),
                        ball_position + torch.tensor((-0.20, 0.0, 0.10), device=ball_position.device),
                    )
                )
                camera_targets = ball_position.unsqueeze(0).repeat(3, 1)
                diagnostic_camera.set_world_poses_from_view(camera_positions, camera_targets)
                for _ in range(3):
                    base_env.sim.render()
                    diagnostic_camera.update(dt=base_env.dt, force_recompute=True)
                frames = diagnostic_camera.data.output["rgb"].detach().cpu().numpy()
                for camera_index, frame in enumerate(frames):
                    Image.fromarray(frame[:, :, :3]).save(
                        capture_dir / f"{label}_view_{camera_index}.png"
                    )
                captured_phases.add(label)

        checkpoint = None
        policy = None
        if args_cli.controller == "policy":
            if not args_cli.checkpoint:
                raise ValueError("--checkpoint is required with --controller=policy")
            checkpoint = retrieve_file_path(args_cli.checkpoint)
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(agent_cfg.class_name)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

        requested = int(args_cli.episodes)
        completed = 0
        metric_sums = {
            "gentle_catch_rate": 0.0,
            "retained_grasp_rate": 0.0,
            "latch_rate": 0.0,
            "drop_rate": 0.0,
            "grasp_loss_rate": 0.0,
            "mean_hold_drive_error": 0.0,
            "mean_hold_closing": 0.0,
            "soft_grasp_any_rate": 0.0,
            "min_tip_distance": 0.0,
            "max_grasp_alignment": 0.0,
            "closest_tip_max": 0.0,
            "closest_closing": 0.0,
            "closest_ball_speed": 0.0,
            "closest_relative_speed": 0.0,
            "closest_alignment": 0.0,
            "closest_along": 0.0,
            "closest_radial": 0.0,
            "closest_tip_spread": 0.0,
            "loss_along": 0.0,
            "loss_radial": 0.0,
            "loss_tip_dist": 0.0,
            "loss_ball_speed": 0.0,
            "loss_relative_speed": 0.0,
        }
        tag_map = {
            "gentle_catch_rate": "Metrics/gentle_grasp",
            "retained_grasp_rate": "Metrics/soft_grasp",
            "latch_rate": "Metrics/soft_grasp_latch",
            "drop_rate": "Metrics/drop_rate",
            "grasp_loss_rate": "Metrics/grasp_loss_rate",
            "mean_hold_drive_error": "Metrics/mean_hold_drive_error",
            "mean_hold_closing": "Metrics/mean_hold_closing",
            "soft_grasp_any_rate": "Metrics/soft_grasp_any",
            "min_tip_distance": "Metrics/min_tip_dist",
            "max_grasp_alignment": "Metrics/max_grasp_align",
            "closest_tip_max": "Metrics/closest_tip_max",
            "closest_closing": "Metrics/closest_closing",
            "closest_ball_speed": "Metrics/closest_ball_speed",
            "closest_relative_speed": "Metrics/closest_relative_speed",
            "closest_alignment": "Metrics/closest_align",
            "closest_along": "Metrics/closest_along",
            "closest_radial": "Metrics/closest_radial",
            "closest_tip_spread": "Metrics/closest_tip_spread",
            "loss_along": "Metrics/loss_along",
            "loss_radial": "Metrics/loss_radial",
            "loss_tip_dist": "Metrics/loss_tip_dist",
            "loss_ball_speed": "Metrics/loss_ball_speed",
            "loss_relative_speed": "Metrics/loss_relative_speed",
        }

        obs = env.get_observations()
        if capture_dir is not None:
            ball_position = base_env.ball.data.root_pos_w[0]
            ball_position = (
                ball_position.torch if hasattr(ball_position, "torch") else ball_position
            )
            camera_positions = torch.stack(
                (
                    ball_position + torch.tensor((0.20, 0.20, 0.10), device=ball_position.device),
                    ball_position + torch.tensor((0.20, -0.20, 0.10), device=ball_position.device),
                    ball_position + torch.tensor((-0.20, 0.0, 0.10), device=ball_position.device),
                )
            )
            camera_targets = ball_position.unsqueeze(0).repeat(3, 1)
            diagnostic_camera.set_world_poses_from_view(camera_positions, camera_targets)
            capture_phase("reset")
        diagnostic_hold_target = torch.full(
            (int(args_cli.num_envs),),
            float("nan"),
            device=env.unwrapped.device,
        )
        diagnostic_contact_target = torch.full_like(diagnostic_hold_target, float("nan"))
        diagnostic_contact_steps = torch.zeros(
            int(args_cli.num_envs), device=env.unwrapped.device
        )
        max_steps = math.ceil(requested / int(args_cli.num_envs)) * (env.unwrapped.max_episode_length + 8)
        steps = 0
        try:
            while completed < requested and steps < max_steps:
                if capture_dir is not None:
                    if "latched" not in captured_phases and bool(base_env._episode_caught.any()):
                        capture_phase("latched")
                    step_ball_radial = getattr(
                        base_env,
                        "_step_ball_radial",
                        torch.zeros_like(base_env._episode_caught, dtype=torch.float),
                    )
                    escaping = base_env._episode_caught & (step_ball_radial > 0.05)
                    if "escaping" not in captured_phases and bool(escaping.any()):
                        capture_phase("escaping")
                with torch.inference_mode():
                    if policy is not None:
                        actions = policy(obs)
                    else:
                        # Diagnostic only: command a gentle close from observed
                        # time-to-closest-approach. The Cartesian variant also
                        # uses the simulator Jacobian to cushion impact. Neither
                        # controller is exported or injected into policy actions.
                        policy_obs = obs["policy"]
                        actions = torch.zeros(
                            (policy_obs.shape[0], env.num_actions),
                            device=policy_obs.device,
                            dtype=policy_obs.dtype,
                        )
                        if args_cli.controller == "expert_teacher":
                            actions = base_env.expert_catch_actions()
                        elif args_cli.controller == "open_grip":
                            actions[:, -1] = -1.0
                        else:
                            to_tips = policy_obs[:, 25:28]
                            ball_vel = policy_obs[:, 19:22]
                            closing_rate = (to_tips * ball_vel).sum(dim=-1)
                            time_to_contact = torch.clamp(
                                -closing_rate / ball_vel.square().sum(dim=-1).clamp(min=1e-4),
                                min=0.0,
                                max=1.0,
                            )
                            near = torch.linalg.norm(to_tips, dim=-1) < 0.13
                            approaching = closing_rate < 0.0
                            lead = max(float(args_cli.grip_lead_time), 0.04)
                            target_close = float(args_cli.grip_target_close)
                            desired_close = target_close * torch.clamp(
                                (lead - time_to_contact) / max(lead - 0.035, 1e-3), 0.0, 1.0
                            )
                            desired_close = torch.where(
                                approaching | near, desired_close, torch.zeros_like(desired_close)
                            )
                            desired_close = torch.where(
                                near, torch.full_like(desired_close, target_close), desired_close
                            )
                            if args_cli.grip_feedback == "base":
                                joint_pos_data = base_env.robot.data.joint_pos
                                joint_pos_data = (
                                    joint_pos_data.torch
                                    if hasattr(joint_pos_data, "torch")
                                    else joint_pos_data
                                )
                                grip_pos = joint_pos_data[
                                    :, base_env._gripper_base_joint_ids
                                ].mean(dim=-1)
                            else:
                                grip_pos = policy_obs[:, 14]
                            actual_close = (grip_pos - 0.04) / (1.10 - 0.04)
                            actions[:, -1] = torch.clamp(
                                8.0 * (desired_close - actual_close), -1.0, 1.0
                            )
                            if args_cli.unload_on_contact:
                                contact = getattr(
                                    base_env,
                                    "_step_soft_grasp",
                                    torch.zeros_like(base_env._episode_caught),
                                )
                                # Do not unload merely because the ball entered
                                # the aperture: the fingers still need a quick
                                # wrap.  The single-frame soft-grasp geometry is
                                # early enough to relax before the latch gate,
                                # without mistaking proximity for contact.
                                unload = contact | base_env._episode_caught
                                joint_pos = base_env.robot.data.joint_pos
                                joint_pos = joint_pos.torch if hasattr(joint_pos, "torch") else joint_pos
                                feedback_ids = (
                                    base_env._gripper_base_joint_ids
                                    if args_cli.grip_feedback == "base"
                                    else base_env._gripper_ids
                                )
                                grip_actual_raw = joint_pos[:, feedback_ids].mean(dim=-1)
                                grip_target_raw = base_env.robot_dof_targets[:, feedback_ids].mean(dim=-1)
                                # Keep only a tiny positive drive margin: enough
                                # to prevent passive opening, not enough to turn
                                # symmetric contact into a ball-launching wedge.
                                first_contact = contact & torch.isnan(diagnostic_hold_target)
                                diagnostic_contact_target = torch.where(
                                    first_contact,
                                    grip_actual_raw,
                                    diagnostic_contact_target,
                                )
                                diagnostic_hold_target = torch.where(
                                    first_contact,
                                    grip_actual_raw,
                                    diagnostic_hold_target,
                                )
                                unload = ~torch.isnan(diagnostic_hold_target)
                                final_hold_close = (
                                    float(args_cli.grip_final_hold_close)
                                    if args_cli.grip_final_hold_close is not None
                                    else float(args_cli.grip_hold_close)
                                )
                                hold_blend = torch.clamp(
                                    diagnostic_contact_steps
                                    / max(float(args_cli.grip_hold_ramp_steps), 1.0),
                                    0.0,
                                    1.0,
                                )
                                if args_cli.firm_hold_below_speed is not None:
                                    ball_velocity_data = base_env.ball.data.root_lin_vel_w
                                    ball_velocity = (
                                        ball_velocity_data.torch
                                        if hasattr(ball_velocity_data, "torch")
                                        else ball_velocity_data
                                    )
                                    body_velocity_data = base_env.robot.data.body_lin_vel_w
                                    body_velocity = (
                                        body_velocity_data.torch
                                        if hasattr(body_velocity_data, "torch")
                                        else body_velocity_data
                                    )
                                    tip_velocity = body_velocity[:, base_env._tip_body_ids].mean(dim=1)
                                    ee_velocity = body_velocity[:, base_env._ee_body_idx]
                                    hand_velocity = 0.5 * (tip_velocity + ee_velocity)
                                    relative_speed = torch.linalg.norm(
                                        ball_velocity - hand_velocity, dim=-1
                                    )
                                    firm = unload & (
                                        relative_speed < float(args_cli.firm_hold_below_speed)
                                    )
                                    hold_blend = torch.where(
                                        firm, torch.ones_like(hold_blend), hold_blend
                                    )
                                if args_cli.hold_actual_on_contact:
                                    contact_target = torch.where(
                                        torch.isnan(diagnostic_contact_target),
                                        grip_actual_raw,
                                        diagnostic_contact_target,
                                    )
                                    if args_cli.grip_final_hold_close is None:
                                        hold_target_raw = contact_target
                                    else:
                                        final_target_raw = 0.04 + final_hold_close * (1.10 - 0.04)
                                        hold_target_raw = contact_target * (1.0 - hold_blend)
                                        hold_target_raw = hold_target_raw + final_target_raw * hold_blend
                                else:
                                    hold_close = float(args_cli.grip_hold_close) * (1.0 - hold_blend)
                                    hold_close = hold_close + final_hold_close * hold_blend
                                    hold_target_raw = 0.04 + hold_close * (1.10 - 0.04)
                                diagnostic_hold_target = torch.where(
                                    unload, hold_target_raw, diagnostic_hold_target
                                )
                                target_delta_per_action = (
                                    0.75 * float(base_env.dt) * float(base_env.cfg.action_scale)
                                )
                                unload_action = torch.clamp(
                                    (diagnostic_hold_target - grip_target_raw)
                                    / max(target_delta_per_action, 1e-4),
                                    -1.0,
                                    1.0,
                                )
                                actions[:, -1] = torch.where(
                                    unload, unload_action, actions[:, -1]
                                )
                            if args_cli.controller == "cartesian_catch":
                                # Move very slightly toward the ball, then begin
                                # yielding early enough for the position drives
                                # to reach velocity before impact. f(s)=1.5s^2-0.5s
                                # starts and ends smoothly enough for the arm,
                                # crosses into cushioning after one third of the
                                # lead window and displaces the cup by v*T/4.
                                cushion_lead = max(float(args_cli.cushion_lead_time), 0.04)
                                phase = torch.clamp(
                                    (cushion_lead - time_to_contact) / cushion_lead, 0.0, 1.0
                                )
                                velocity_profile = 1.5 * phase.square() - 0.5 * phase
                                velocity_profile = velocity_profile * approaching.float()
                                desired_hand_vel = (
                                    float(args_cli.cushion_velocity_gain)
                                    * velocity_profile.unsqueeze(-1)
                                    * ball_vel
                                )

                                # After a real three-frame enclosure, bleed the
                                # shared hand/ball velocity away gently instead
                                # of abruptly stopping and ejecting the ball.
                                caught = base_env._episode_caught
                                settle = torch.exp(
                                    -base_env._post_latch_steps.float()
                                    / max(float(args_cli.cushion_settle_steps), 1.0)
                                )
                                post_ball_vel = ball_vel
                                if args_cli.cushion_no_downward_follow:
                                    post_ball_vel = ball_vel.clone()
                                    post_ball_vel[:, 2] = torch.clamp(post_ball_vel[:, 2], min=0.0)
                                desired_hand_vel = torch.where(
                                    caught.unsqueeze(-1),
                                    float(args_cli.cushion_post_gain)
                                    * settle.unsqueeze(-1)
                                    * post_ball_vel,
                                    desired_hand_vel,
                                )

                                jacobian_data = base_env.robot.data.body_link_jacobian_w
                                jacobians = (
                                    jacobian_data.torch
                                    if hasattr(jacobian_data, "torch")
                                    else jacobian_data
                                )
                                ee_jacobian_idx = (
                                    base_env._ee_body_idx - 1
                                    if base_env.robot.is_fixed_base
                                    else base_env._ee_body_idx
                                )
                                jacobian = jacobians[:, ee_jacobian_idx, :3, :]
                                jacobian = jacobian[:, :, base_env._arm_ids]
                                damping = max(float(args_cli.cushion_damping), 1e-4)
                                identity = torch.eye(
                                    3, device=jacobian.device, dtype=jacobian.dtype
                                ).unsqueeze(0)
                                task_matrix = jacobian @ jacobian.transpose(1, 2)
                                task_matrix = task_matrix + damping * damping * identity
                                joint_velocity = jacobian.transpose(1, 2) @ torch.linalg.solve(
                                    task_matrix, desired_hand_vel.unsqueeze(-1)
                                )
                                joint_velocity = joint_velocity.squeeze(-1)

                                alpha = float(base_env._curriculum_alpha())
                                easy = float(base_env.cfg.arm_motion_scale_easy)
                                hard = float(base_env.cfg.arm_motion_scale_hard)
                                power = float(base_env.cfg.arm_motion_scale_power)
                                motion_scale = easy * (1.0 - alpha**power) + hard * alpha**power
                                action_velocity = float(base_env.cfg.action_scale) * motion_scale
                                actions[:, :-1] = torch.clamp(
                                    joint_velocity / max(action_velocity, 1e-4), -1.0, 1.0
                                )
                    obs, _, dones, extras = env.step(actions)
                    diagnostic_contact_steps = torch.where(
                        torch.isnan(diagnostic_hold_target),
                        diagnostic_contact_steps,
                        diagnostic_contact_steps + 1.0,
                    )
                    diagnostic_hold_target = torch.where(
                        dones.bool(),
                        torch.full_like(diagnostic_hold_target, float("nan")),
                        diagnostic_hold_target,
                    )
                    diagnostic_contact_target = torch.where(
                        dones.bool(),
                        torch.full_like(diagnostic_contact_target, float("nan")),
                        diagnostic_contact_target,
                    )
                    diagnostic_contact_steps = torch.where(
                        dones.bool(), torch.zeros_like(diagnostic_contact_steps), diagnostic_contact_steps
                    )
                    if policy is not None and version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                done_count = int(dones.sum().item())
                if done_count > 0:
                    weight = min(done_count, requested - completed)
                    log = extras.get("log", {})
                    for key, tag in tag_map.items():
                        value = log.get(tag, float("nan"))
                        value = float(value.item()) if hasattr(value, "item") else float(value)
                        if not math.isnan(value):
                            metric_sums[key] += value * weight
                    completed += weight
                steps += 1
        finally:
            env.close()

        denominator = max(completed, 1)
        result = {
            "task": args_cli.task,
            "controller": args_cli.controller,
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else None,
            "seed": int(args_cli.seed),
            "difficulty": float(args_cli.difficulty),
            "throw_mode": args_cli.throw_mode,
            "episodes": completed,
            "grasp_axis_world": [float(value) for value in initial_grasp_axis.tolist()],
            **{key: value / denominator for key, value in metric_sums.items()},
        }
        result["passed"] = (
            completed >= requested
            and result["gentle_catch_rate"] >= float(args_cli.min_gentle_rate)
            and result["drop_rate"] <= float(args_cli.max_drop_rate)
        )
        payload = json.dumps(result, indent=2)
        print(payload)
        if args_cli.output is not None:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(payload + "\n", encoding="utf-8")
        return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
