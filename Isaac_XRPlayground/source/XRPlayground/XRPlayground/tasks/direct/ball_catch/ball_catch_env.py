# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL env: Kinova Jaco2 + 3-finger gripper learns to catch thrown balls."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .ball_catch_env_cfg import BallCatchEnvCfg


def _as_tensor(data) -> torch.Tensor:
    """Read articulation/rigid-object buffers across Isaac Lab API variants."""
    return data.torch if hasattr(data, "torch") else data


class BallCatchEnv(DirectRLEnv):
    cfg: BallCatchEnvCfg

    def __init__(self, cfg: BallCatchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        joint_limits = _as_tensor(self.robot.data.soft_joint_pos_limits)[0]
        self.robot_dof_lower_limits = joint_limits[:, 0].to(self.device)
        self.robot_dof_upper_limits = joint_limits[:, 1].to(self.device)

        self._arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._gripper_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
        if len(self._arm_ids) != len(self.cfg.arm_joint_names):
            raise RuntimeError(
                f"Expected {len(self.cfg.arm_joint_names)} Kinova arm joints, found {len(self._arm_ids)}."
            )
        if len(self._gripper_ids) < 3:
            raise RuntimeError(f"Expected Kinova finger joints, found {len(self._gripper_ids)}.")
        self._arm_ids = torch.tensor(self._arm_ids, device=self.device, dtype=torch.long)
        self._gripper_ids = torch.tensor(self._gripper_ids, device=self.device, dtype=torch.long)

        self.robot_dof_speed_scales = torch.ones(self.robot.num_joints, device=self.device)
        self.robot_dof_speed_scales[self._gripper_ids] = 1.2

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.left_finger_idx, self.right_finger_idx = self._resolve_gripper_body_indices()
        self._tip_body_ids = self._resolve_finger_tip_body_indices()
        self._arm_body_ids = self._resolve_arm_body_indices()
        self._ee_body_idx = self._resolve_ee_body_index()

        self._episode_caught = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._just_caught = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_dist = torch.ones(self.num_envs, device=self.device)
        self._grasp_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cup_balance_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Drive targets must satisfy both soft joint limits and PhysX revolute ±2π.
        two_pi = self.cfg.physx_drive_angle_limit
        self.robot_dof_drive_lower = torch.maximum(
            self.robot_dof_lower_limits, torch.full_like(self.robot_dof_lower_limits, -two_pi)
        )
        self.robot_dof_drive_upper = torch.minimum(
            self.robot_dof_upper_limits, torch.full_like(self.robot_dof_upper_limits, two_pi)
        )

    def _resolve_gripper_body_indices(self) -> tuple[int, int]:
        candidates = (
            self.cfg.gripper_body_names,
            [".*finger.*"],
            [self.cfg.ee_body_name],
        )
        for keys in candidates:
            ids, _ = self.robot.find_bodies(keys)
            if len(ids) >= 2:
                return ids[0], ids[1]
            if len(ids) == 1:
                return ids[0], ids[0]
        raise RuntimeError("Could not find gripper or end-effector bodies on the Kinova articulation.")

    def _resolve_arm_body_indices(self) -> torch.Tensor:
        ids, _ = self.robot.find_bodies(self.cfg.arm_body_names)
        if len(ids) == 0:
            ids, _ = self.robot.find_bodies([".*link_[1-7].*"])
        if len(ids) == 0:
            raise RuntimeError("Could not find arm link bodies for body-contact detection.")
        return torch.tensor(ids, device=self.device, dtype=torch.long)

    def _resolve_ee_body_index(self) -> int:
        ids, _ = self.robot.find_bodies([self.cfg.ee_body_name])
        if len(ids) == 0:
            ids, _ = self.robot.find_bodies([".*end_effector.*"])
        if len(ids) == 0:
            raise RuntimeError(f"Could not find EE body '{self.cfg.ee_body_name}'.")
        return int(ids[0])

    def _resolve_finger_tip_body_indices(self) -> torch.Tensor:
        ids, _ = self.robot.find_bodies(self.cfg.finger_tip_body_names, preserve_order=True)
        if len(ids) < 3:
            ids, _ = self.robot.find_bodies([".*finger_tip.*"])
        if len(ids) < 3:
            raise RuntimeError(
                f"Expected 3 Kinova finger-tip bodies, found {len(ids)} "
                f"(cfg names={self.cfg.finger_tip_body_names})."
            )
        return torch.tensor(ids[:3], device=self.device, dtype=torch.long)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.ball = RigidObject(self.cfg.ball_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Freeze arm only after a confirmed catch — early freeze on grasp_hold_count
        # locked cupping poses before the ball was truly between the fingers.
        freeze_arm = self._episode_caught

        arm_delta = self.robot_dof_speed_scales[self._arm_ids] * self.dt * self.actions[:, :-1] * self.cfg.action_scale
        grip_delta = (
            self.robot_dof_speed_scales[self._gripper_ids]
            * self.dt
            * self.actions[:, -1:]
            * self.cfg.action_scale
        )
        arm_delta = torch.where(freeze_arm.unsqueeze(-1), torch.zeros_like(arm_delta), arm_delta)

        self.robot_dof_targets[:, self._arm_ids] += arm_delta
        self.robot_dof_targets[:, self._gripper_ids] += grip_delta

        joint_pos = _as_tensor(self.robot.data.joint_pos)
        if freeze_arm.any():
            # Lock PD targets to current arm configuration so the arm stays put.
            self.robot_dof_targets[:, self._arm_ids] = torch.where(
                freeze_arm.unsqueeze(-1),
                joint_pos[:, self._arm_ids],
                self.robot_dof_targets[:, self._arm_ids],
            )
            # Hold a firm close; ignore open commands while grasping.
            close = torch.full(
                (self.num_envs, self._gripper_ids.numel()),
                self.cfg.gripper_close_target,
                device=self.device,
                dtype=self.robot_dof_targets.dtype,
            )
            self.robot_dof_targets[:, self._gripper_ids] = torch.where(
                freeze_arm.unsqueeze(-1),
                close,
                self.robot_dof_targets[:, self._gripper_ids],
            )
            # Zero reported actions for frozen envs so action penalty reflects hold-still intent.
            self.actions = torch.where(freeze_arm.unsqueeze(-1), torch.zeros_like(self.actions), self.actions)

        self.robot_dof_targets[:] = torch.clamp(
            self.robot_dof_targets, self.robot_dof_drive_lower, self.robot_dof_drive_upper
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _tip_center_w(self) -> torch.Tensor:
        """Mean world position of the three finger-tip bodies."""
        tip_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._tip_body_ids]
        return tip_pos.mean(dim=1)

    def _gripper_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tip_center (world), gripper_pos, closing, tip_span."""
        tip_center = self._tip_center_w()
        tip_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._tip_body_ids]
        tip_span = torch.linalg.norm(tip_pos[:, 0] - tip_pos[:, 1], dim=-1)
        joint_pos = _as_tensor(self.robot.data.joint_pos)
        gripper_pos = joint_pos[:, self._gripper_ids].mean(dim=-1)
        closing = (gripper_pos - self.cfg.gripper_open_pos) / (
            self.cfg.gripper_close_target - self.cfg.gripper_open_pos
        )
        return tip_center, gripper_pos, closing, tip_span

    def _ball_to_arm_dist(self, ball_pos_w: torch.Tensor) -> torch.Tensor:
        """Min distance from ball to non-finger arm links."""
        body_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._arm_body_ids]
        # (N, num_links, 3)
        delta = body_pos - ball_pos_w.unsqueeze(1)
        return torch.linalg.norm(delta, dim=-1).min(dim=-1).values

    def _get_observations(self) -> dict:
        joint_pos = _as_tensor(self.robot.data.joint_pos)
        joint_vel = _as_tensor(self.robot.data.joint_vel)
        arm_pos = joint_pos[:, self._arm_ids]
        arm_vel = joint_vel[:, self._arm_ids]
        gripper_pos = joint_pos[:, self._gripper_ids].mean(dim=-1, keepdim=True)
        gripper_vel = joint_vel[:, self._gripper_ids].mean(dim=-1, keepdim=True)
        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)

        tip_center = self._tip_center_w() - self.scene.env_origins
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx] - self.scene.env_origins
        to_ball = ball_pos - ee_pos
        to_ball_fingers = ball_pos - tip_center

        obs = torch.cat(
            (
                arm_pos,
                self.cfg.dof_velocity_scale * arm_vel,
                gripper_pos,
                self.cfg.dof_velocity_scale * gripper_vel,
                ball_pos,
                ball_vel,
                to_ball,
                to_ball_fingers,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _update_grasp_and_body_flags(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update hold / body / cup-balance counters.

        Success requires the ball in the finger aperture along the EE→tip_center
        grasp axis (not resting on the dorsal/outer gripper housing).
        """
        ball_pos_w = _as_tensor(self.ball.data.root_pos_w)
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)
        ball_speed = torch.linalg.norm(ball_vel, dim=-1)
        tip_center, _, closing, _ = self._gripper_state()
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]

        ee_to_tips = tip_center - ee_pos
        grasp_axis = ee_to_tips / torch.linalg.norm(ee_to_tips, dim=-1, keepdim=True).clamp(min=1e-6)
        ee_to_ball = ball_pos_w - ee_pos
        ball_along = (ee_to_ball * grasp_axis).sum(dim=-1)
        ball_radial_vec = ee_to_ball - ball_along.unsqueeze(-1) * grasp_axis
        ball_radial = torch.linalg.norm(ball_radial_vec, dim=-1)

        dist_ee = torch.linalg.norm(ee_to_ball, dim=-1)
        dist_tips = torch.linalg.norm(ball_pos_w - tip_center, dim=-1)
        dist_palm = dist_ee  # EE body ≈ palm / housing reference
        # Cos similarity: EE→ball vs EE→tips (1 = approaching along grasp axis)
        ee_to_ball_n = ee_to_ball / dist_ee.unsqueeze(-1).clamp(min=1e-6)
        grasp_align = (ee_to_ball_n * grasp_axis).sum(dim=-1)

        # Primary shaping distance: tip cluster (not raw EE — that rewarded top-down cups)
        dist = dist_tips
        arm_dist = self._ball_to_arm_dist(ball_pos_w)

        near_ee = dist_ee < self.cfg.gripper_near_dist * 1.15
        # Cup / housing: near EE but poorly aligned, sitting behind the tips, or closer
        # to the palm than to the tip cluster (works when the gripper is tilted).
        cup_balance = near_ee & (
            (grasp_align < self.cfg.cup_align_max)
            | (ball_along < self.cfg.grasp_along_min)
            | ((dist_palm + 0.01) < dist_tips)
        ) & (closing < self.cfg.success_close_min + 0.12)

        in_grasp = (
            (ball_along > self.cfg.grasp_along_min)
            & (ball_along < self.cfg.grasp_along_max)
            & (ball_radial < self.cfg.grasp_radial_max)
            & (dist_tips < self.cfg.success_tip_dist)
            & (closing > self.cfg.success_close_min)
            & (ball_speed < self.cfg.success_speed_threshold)
            & (grasp_align > self.cfg.grasp_align_min)
            & (ball_pos_w[:, 2] > self.cfg.fall_height_threshold + 0.05)
            & (~cup_balance)
        )
        # Ball resting on forearm/links, not inside the gripper volume
        on_body = (arm_dist < self.cfg.body_contact_radius) & (dist_ee > self.cfg.gripper_near_dist) & (~in_grasp)

        self._grasp_hold_count = torch.where(
            in_grasp, self._grasp_hold_count + 1, torch.zeros_like(self._grasp_hold_count)
        )
        self._body_contact_count = torch.where(
            on_body, self._body_contact_count + 1, torch.zeros_like(self._body_contact_count)
        )
        self._cup_balance_count = torch.where(
            cup_balance, self._cup_balance_count + 1, torch.zeros_like(self._cup_balance_count)
        )
        newly_caught = (self._grasp_hold_count >= self.cfg.grasp_hold_steps) & (~self._episode_caught)
        self._just_caught = newly_caught
        self._episode_caught |= newly_caught
        self._body_fail |= self._body_contact_count >= self.cfg.body_fail_steps
        self._body_fail |= self._cup_balance_count >= self.cfg.cup_fail_steps

        # Cache for rewards (dones run before rewards in DirectRLEnv.step)
        self._step_dist = dist
        self._step_dist_ee = dist_ee
        self._step_dist_tips = dist_tips
        self._step_closing = closing
        self._step_in_grasp = in_grasp
        self._step_on_body = on_body
        self._step_cup_balance = cup_balance
        self._step_grasp_align = grasp_align
        return dist, closing, in_grasp, on_body

    def _get_rewards(self) -> torch.Tensor:
        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)
        ball_speed = torch.linalg.norm(ball_vel, dim=-1)

        # Prefer flags computed in _get_dones this step; fall back if needed
        if hasattr(self, "_step_dist"):
            dist = self._step_dist
            dist_ee = self._step_dist_ee
            dist_tips = self._step_dist_tips
            closing = self._step_closing
            in_grasp = self._step_in_grasp
            on_body = self._step_on_body
            cup_balance = self._step_cup_balance
            grasp_align = self._step_grasp_align
        else:
            dist, closing, in_grasp, on_body = self._update_grasp_and_body_flags()
            dist_ee = self._step_dist_ee
            dist_tips = self._step_dist_tips
            cup_balance = self._step_cup_balance
            grasp_align = self._step_grasp_align

        joint_pos = _as_tensor(self.robot.data.joint_pos)
        joint_vel = _as_tensor(self.robot.data.joint_vel)
        gripper_pos = joint_pos[:, self._gripper_ids].mean(dim=-1)
        arm_speed = torch.linalg.norm(joint_vel[:, self._arm_ids], dim=-1)

        dropped = ball_pos[:, 2] < self.cfg.fall_height_threshold
        self._dropped |= dropped

        reward = compute_rewards(
            dist_ee,
            dist_tips,
            self._prev_dist,
            ball_speed,
            arm_speed,
            gripper_pos,
            closing,
            grasp_align,
            in_grasp.float(),
            on_body.float(),
            cup_balance.float(),
            dropped.float(),
            self._just_caught.float(),
            self._episode_caught.float(),
            self.cfg.dist_reward_scale,
            self.cfg.approach_reward_scale,
            self.cfg.catch_reward_scale,
            self.cfg.grasp_reward_scale,
            self.cfg.hold_still_reward_scale,
            self.cfg.hold_action_penalty_scale,
            self.cfg.body_contact_penalty,
            self.cfg.cup_balance_penalty,
            self.cfg.drop_penalty,
            self.cfg.action_penalty_scale,
            self.cfg.success_tip_dist,
            self.cfg.gripper_near_dist,
            self.cfg.success_close_min,
            self.cfg.grasp_align_min,
            self.cfg.cup_align_max,
            self.actions,
        )
        self._prev_dist = dist.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_grasp_and_body_flags()

        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        dropped = ball_pos[:, 2] < self.cfg.fall_height_threshold
        self._dropped |= dropped

        terminated = dropped | self._body_fail
        if self.cfg.terminate_on_catch:
            terminated = terminated | self._episode_caught
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            log = self.extras.setdefault("log", {})
            log["Metrics/catch_rate"] = self._episode_caught[env_ids].float().mean().item()
            log["Metrics/body_fail_rate"] = self._body_fail[env_ids].float().mean().item()
            log["Metrics/drop_rate"] = self._dropped[env_ids].float().mean().item()

        self._episode_caught[env_ids] = False
        self._just_caught[env_ids] = False
        self._body_fail[env_ids] = False
        self._dropped[env_ids] = False
        self._grasp_hold_count[env_ids] = 0
        self._body_contact_count[env_ids] = 0
        self._cup_balance_count[env_ids] = 0

        super()._reset_idx(env_ids)

        joint_pos = _as_tensor(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        root_state = _as_tensor(self.robot.data.default_root_state)[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot_dof_targets[env_ids] = joint_pos
        self._prev_dist[env_ids] = 1.0

        self._launch_ball(env_ids)

    def _curriculum_alpha(self) -> float:
        steps = max(int(self.cfg.curriculum_steps), 1)
        return min(float(self.common_step_counter) / float(steps), 1.0)

    def _mix_range(self, easy: tuple[float, float], hard: tuple[float, float], alpha: float) -> tuple[float, float]:
        return (
            easy[0] * (1.0 - alpha) + hard[0] * alpha,
            easy[1] * (1.0 - alpha) + hard[1] * alpha,
        )

    def _launch_ball(self, env_ids: torch.Tensor) -> None:
        """Sample throws: easy at first, then lerp toward wider XR-like ranges."""
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        alpha = self._curriculum_alpha()
        self.extras.setdefault("log", {})["Metrics/curriculum"] = alpha

        release = torch.zeros((n, 3), device=self.device)
        release[:, 0] = sample_uniform(
            *self._mix_range(self.cfg.throw_pos_x_easy, self.cfg.throw_pos_x_hard, alpha), (n,), device=self.device
        )
        release[:, 1] = sample_uniform(
            *self._mix_range(self.cfg.throw_pos_y_easy, self.cfg.throw_pos_y_hard, alpha), (n,), device=self.device
        )
        release[:, 2] = sample_uniform(
            *self._mix_range(self.cfg.throw_pos_z_easy, self.cfg.throw_pos_z_hard, alpha), (n,), device=self.device
        )

        aim = torch.zeros((n, 3), device=self.device)
        aim[:, 0] = sample_uniform(
            *self._mix_range(self.cfg.aim_pos_x_easy, self.cfg.aim_pos_x_hard, alpha), (n,), device=self.device
        )
        aim[:, 1] = sample_uniform(
            *self._mix_range(self.cfg.aim_pos_y_easy, self.cfg.aim_pos_y_hard, alpha), (n,), device=self.device
        )
        aim[:, 2] = sample_uniform(
            *self._mix_range(self.cfg.aim_pos_z_easy, self.cfg.aim_pos_z_hard, alpha), (n,), device=self.device
        )

        direction = aim - release
        direction = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6))

        speed = sample_uniform(
            *self._mix_range(self.cfg.throw_speed_easy, self.cfg.throw_speed_hard, alpha), (n,), device=self.device
        )
        lin_vel = direction * speed.unsqueeze(-1)

        pos = release + origins
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(n, 1)
        root_pose = torch.cat((pos, quat), dim=-1)

        root_vel = torch.zeros((n, 6), device=self.device)
        root_vel[:, :3] = lin_vel
        root_vel[:, 3:] = sample_uniform(*self.cfg.throw_ang_vel, (n, 3), device=self.device)

        self.ball.write_root_pose_to_sim(root_pose, env_ids)
        self.ball.write_root_velocity_to_sim(root_vel, env_ids)


@torch.jit.script
def compute_rewards(
    dist_ee: torch.Tensor,
    dist_tips: torch.Tensor,
    prev_dist_tips: torch.Tensor,
    ball_speed: torch.Tensor,
    arm_speed: torch.Tensor,
    gripper_pos: torch.Tensor,
    closing: torch.Tensor,
    grasp_align: torch.Tensor,
    in_grasp: torch.Tensor,
    on_body: torch.Tensor,
    cup_balance: torch.Tensor,
    dropped: torch.Tensor,
    just_caught: torch.Tensor,
    episode_caught: torch.Tensor,
    dist_reward_scale: float,
    approach_reward_scale: float,
    catch_reward_scale: float,
    grasp_reward_scale: float,
    hold_still_reward_scale: float,
    hold_action_penalty_scale: float,
    body_contact_penalty: float,
    cup_balance_penalty: float,
    drop_penalty: float,
    action_penalty_scale: float,
    success_tip_dist: float,
    gripper_near_dist: float,
    success_close_min: float,
    grasp_align_min: float,
    cup_align_max: float,
    actions: torch.Tensor,
):
    # Prefer tip-cluster approach. Gate EE proximity by grasp-axis alignment so
    # top-down / dorsal approaches are not rewarded.
    align_gate = torch.clamp((grasp_align - cup_align_max) / max(grasp_align_min - cup_align_max, 1e-3), 0.0, 1.0)
    near = (dist_tips < gripper_near_dist * 1.8).float()
    tip_rew = near * torch.exp(-dist_reward_scale * dist_tips)
    # Soft EE shaping only when already roughly aligned with the finger aperture
    ee_rew = near * align_gate * 0.35 * torch.exp(-dist_reward_scale * dist_ee)
    approach_rew = approach_reward_scale * near * align_gate * torch.clamp(prev_dist_tips - dist_tips, -0.03, 0.06)

    closing_clamped = torch.clamp(closing, 0.0, 1.0)
    close_rew = (
        5.0
        * closing_clamped
        * (dist_tips < gripper_near_dist).float()
        * align_gate
        * (1.0 - episode_caught)
        * (1.0 - cup_balance)
    )
    grasp_rew = grasp_reward_scale * in_grasp
    center_bonus = 6.0 * in_grasp * torch.exp(-20.0 * dist_tips)
    success_bonus = catch_reward_scale * just_caught

    holding = torch.clamp(in_grasp + episode_caught, 0.0, 1.0)
    hold_still_rew = hold_still_reward_scale * holding * torch.exp(-1.5 * arm_speed)

    body_pen = body_contact_penalty * on_body
    cup_pen = cup_balance_penalty * cup_balance
    drop_pen = drop_penalty * dropped
    alive_pen = 0.02 * (1.0 - holding)
    action_penalty = action_penalty_scale * torch.sum(actions * actions, dim=-1)
    hold_action_pen = hold_action_penalty_scale * holding * torch.sum(actions * actions, dim=-1)

    return (
        tip_rew
        + ee_rew
        + approach_rew
        + close_rew
        + grasp_rew
        + center_bonus
        + success_bonus
        + hold_still_rew
        - body_pen
        - cup_pen
        - drop_pen
        - alive_pen
        - action_penalty
        - hold_action_pen
    )
