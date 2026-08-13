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
        self.robot_dof_speed_scales[self._gripper_ids] = 0.35

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.left_finger_idx, self.right_finger_idx = self._resolve_gripper_body_indices()

        self._episode_caught = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        arm_delta = self.robot_dof_speed_scales[self._arm_ids] * self.dt * self.actions[:, :-1] * self.cfg.action_scale
        grip_delta = (
            self.robot_dof_speed_scales[self._gripper_ids]
            * self.dt
            * self.actions[:, -1:]
            * self.cfg.action_scale
        )
        self.robot_dof_targets[:, self._arm_ids] += arm_delta
        self.robot_dof_targets[:, self._gripper_ids] += grip_delta
        self.robot_dof_targets[:] = torch.clamp(
            self.robot_dof_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self) -> dict:
        joint_pos = _as_tensor(self.robot.data.joint_pos)
        joint_vel = _as_tensor(self.robot.data.joint_vel)
        arm_pos = joint_pos[:, self._arm_ids]
        arm_vel = joint_vel[:, self._arm_ids]
        gripper_pos = joint_pos[:, self._gripper_ids].mean(dim=-1, keepdim=True)
        gripper_vel = joint_vel[:, self._gripper_ids].mean(dim=-1, keepdim=True)
        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)

        left_tip = _as_tensor(self.robot.data.body_pos_w)[:, self.left_finger_idx] - self.scene.env_origins
        right_tip = _as_tensor(self.robot.data.body_pos_w)[:, self.right_finger_idx] - self.scene.env_origins
        gripper_mid = 0.5 * (left_tip + right_tip)
        to_ball = ball_pos - gripper_mid

        obs = torch.cat(
            (
                arm_pos,
                self.cfg.dof_velocity_scale * arm_vel,
                gripper_pos,
                self.cfg.dof_velocity_scale * gripper_vel,
                ball_pos,
                ball_vel,
                to_ball,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)
        ball_speed = torch.linalg.norm(ball_vel, dim=-1)

        left_tip = _as_tensor(self.robot.data.body_pos_w)[:, self.left_finger_idx] - self.scene.env_origins
        right_tip = _as_tensor(self.robot.data.body_pos_w)[:, self.right_finger_idx] - self.scene.env_origins
        gripper_mid = 0.5 * (left_tip + right_tip)
        dist = torch.linalg.norm(ball_pos - gripper_mid, dim=-1)

        joint_pos = _as_tensor(self.robot.data.joint_pos)
        gripper_pos = joint_pos[:, self._gripper_ids].mean(dim=-1)

        reward = compute_rewards(
            dist,
            ball_speed,
            gripper_pos,
            self.cfg.dist_reward_scale,
            self.cfg.catch_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.gripper_open_pos,
            self.cfg.gripper_close_target,
            self.cfg.success_dist_threshold,
            self.cfg.success_speed_threshold,
            self.actions,
        )

        caught_now = (
            (dist < self.cfg.success_dist_threshold)
            & (ball_speed < self.cfg.success_speed_threshold)
            & (gripper_pos > self.cfg.gripper_open_pos + 0.35 * (self.cfg.gripper_close_target - self.cfg.gripper_open_pos))
            & (ball_pos[:, 2] > self.cfg.fall_height_threshold + 0.05)
        )
        self._episode_caught |= caught_now
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        dropped = ball_pos[:, 2] < self.cfg.fall_height_threshold
        caught = self._episode_caught
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = dropped | caught
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            self.extras.setdefault("log", {})["Metrics/catch_rate"] = self._episode_caught[env_ids].float().mean().item()
        self._episode_caught[env_ids] = False

        super()._reset_idx(env_ids)

        joint_pos = _as_tensor(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        root_state = _as_tensor(self.robot.data.default_root_state)[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        self._launch_ball(env_ids)

    def _launch_ball(self, env_ids: torch.Tensor) -> None:
        """Sample XR-like throws: random release pose, speed, aim point, and spin."""
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        release = torch.zeros((n, 3), device=self.device)
        release[:, 0] = sample_uniform(*self.cfg.throw_pos_x, (n,), device=self.device)
        release[:, 1] = sample_uniform(*self.cfg.throw_pos_y, (n,), device=self.device)
        release[:, 2] = sample_uniform(*self.cfg.throw_pos_z, (n,), device=self.device)

        aim = torch.zeros((n, 3), device=self.device)
        aim[:, 0] = sample_uniform(*self.cfg.aim_pos_x, (n,), device=self.device)
        aim[:, 1] = sample_uniform(*self.cfg.aim_pos_y, (n,), device=self.device)
        aim[:, 2] = sample_uniform(*self.cfg.aim_pos_z, (n,), device=self.device)

        direction = aim - release
        direction = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6))

        speed = sample_uniform(*self.cfg.throw_speed, (n,), device=self.device)
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
    dist: torch.Tensor,
    ball_speed: torch.Tensor,
    gripper_pos: torch.Tensor,
    dist_reward_scale: float,
    catch_reward_scale: float,
    action_penalty_scale: float,
    gripper_open_pos: float,
    gripper_close_target: float,
    success_dist_threshold: float,
    success_speed_threshold: float,
    actions: torch.Tensor,
):
    dist_rew = torch.exp(-dist_reward_scale * dist)
    closing = torch.clamp(
        (gripper_pos - gripper_open_pos) / (gripper_close_target - gripper_open_pos), 0.0, 1.0
    )
    catch_rew = catch_reward_scale * closing * torch.exp(-4.0 * dist) * (ball_speed < success_speed_threshold).float()
    success_bonus = catch_reward_scale * 2.0 * (dist < success_dist_threshold).float() * (ball_speed < success_speed_threshold).float()
    action_penalty = action_penalty_scale * torch.sum(actions * actions, dim=-1)
    return dist_rew + catch_rew + success_bonus - action_penalty
