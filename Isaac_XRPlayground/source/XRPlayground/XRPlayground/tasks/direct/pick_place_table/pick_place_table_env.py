# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL: Agibot A2D right arm picks table pieces and places them in a bucket."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .pick_place_table_env_cfg import PickPlaceTableEnvCfg


def _as_tensor(data) -> torch.Tensor:
    return data.torch if hasattr(data, "torch") else data


class PickPlaceTableEnv(DirectRLEnv):
    cfg: PickPlaceTableEnvCfg

    def __init__(self, cfg: PickPlaceTableEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        joint_limits = _as_tensor(self.robot.data.soft_joint_pos_limits)[0]
        self.robot_dof_lower_limits = joint_limits[:, 0].to(self.device)
        self.robot_dof_upper_limits = joint_limits[:, 1].to(self.device)

        self._arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._gripper_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
        self._gripper_driver_idx = int(self.robot.find_joints([self.cfg.gripper_driver_name])[0][0])
        self._frozen_ids, _ = self.robot.find_joints(self.cfg.frozen_joint_names)

        if len(self._arm_ids) != 7:
            raise RuntimeError(f"Expected 7 right-arm joints, found {len(self._arm_ids)}.")
        self._arm_ids = torch.tensor(self._arm_ids, device=self.device, dtype=torch.long)
        self._gripper_ids = torch.tensor(self._gripper_ids, device=self.device, dtype=torch.long)
        self._frozen_ids = torch.tensor(self._frozen_ids, device=self.device, dtype=torch.long)

        self._ee_body_idx = self._resolve_ee_body_index()
        self._default_joint_pos = _as_tensor(self.robot.data.default_joint_pos)[0].clone()

        two_pi = self.cfg.physx_drive_angle_limit
        self.robot_dof_drive_lower = torch.maximum(
            self.robot_dof_lower_limits, torch.full_like(self.robot_dof_lower_limits, -two_pi)
        )
        self.robot_dof_drive_upper = torch.minimum(
            self.robot_dof_upper_limits, torch.full_like(self.robot_dof_upper_limits, two_pi)
        )

        self.robot_dof_targets = self._default_joint_pos.unsqueeze(0).repeat(self.num_envs, 1).clone()
        self.robot_dof_speed_scales = torch.ones(self.robot.num_joints, device=self.device)

        n_slots = self.cfg.num_pieces
        self._piece_active = torch.zeros((self.num_envs, n_slots), dtype=torch.bool, device=self.device)
        self._piece_grasped = torch.zeros((self.num_envs, n_slots), dtype=torch.bool, device=self.device)
        self._piece_color = torch.zeros((self.num_envs, n_slots), dtype=torch.long, device=self.device)
        self._prev_piece_dist = torch.ones(self.num_envs, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._placed_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._table_surface_z = float(self.cfg.table_pos[2]) + 0.5 * float(self.cfg.table_size[2]) + float(
            self.cfg.cube_half_size
        )
        self._bucket_target = torch.tensor(
            [
                float(self.cfg.bucket_pos[0]),
                float(self.cfg.bucket_pos[1]),
                float(self.cfg.bucket_pos[2]) + 0.5 * float(self.cfg.bucket_size[2]) + float(self.cfg.cube_half_size),
            ],
            device=self.device,
        )

    def _resolve_ee_body_index(self) -> int:
        ids, _ = self.robot.find_bodies([self.cfg.ee_body_name])
        if len(ids) == 0:
            ids, _ = self.robot.find_bodies([".*gripper_center.*"])
        if len(ids) == 0:
            raise RuntimeError(f"Could not find EE body '{self.cfg.ee_body_name}'.")
        return int(ids[0])

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.pieces: list[RigidObject] = [
            RigidObject(getattr(self.cfg, f"piece_{i}_cfg")) for i in range(self.cfg.num_pieces)
        ]
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        table_cfg = sim_utils.CuboidCfg(
            size=self.cfg.table_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        )
        bucket_cfg = sim_utils.CuboidCfg(
            size=self.cfg.bucket_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.40)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=self.cfg.table_pos)
        bucket_cfg.func("/World/envs/env_.*/Bucket", bucket_cfg, translation=self.cfg.bucket_pos)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        for i, piece in enumerate(self.pieces):
            self.scene.rigid_objects[f"piece_{i}"] = piece

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _piece_on_table_z(self) -> float:
        return self._table_surface_z

    def _spawn_piece(self, env_ids: torch.Tensor, slot: int) -> None:
        if env_ids.numel() == 0:
            return
        obj = self.pieces[slot]
        n = len(env_ids)
        origins = _as_tensor(self.scene.env_origins)[env_ids]
        pos = torch.zeros((n, 3), device=self.device)
        pos[:, 0] = sample_uniform(self.cfg.spawn_x_range[0], self.cfg.spawn_x_range[1], (n,), self.device)
        pos[:, 1] = sample_uniform(self.cfg.spawn_y_range[0], self.cfg.spawn_y_range[1], (n,), self.device)
        pos[:, 2] = self._piece_on_table_z()
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 3] = 1.0
        pose = torch.cat((pos + origins, quat), dim=-1)
        vel = torch.zeros((n, 6), device=self.device)
        obj.write_root_pose_to_sim(pose, env_ids)
        obj.write_root_velocity_to_sim(vel, env_ids)
        self._piece_active[env_ids, slot] = True
        self._piece_grasped[env_ids, slot] = False
        colors = torch.randint(0, self.cfg.num_colors, (n,), device=self.device)
        self._piece_color[env_ids, slot] = colors

    def _activate_random_piece(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._piece_active[env_ids] = False
        self._piece_grasped[env_ids] = False
        self._spawn_piece(env_ids, 0)

    def spawn_piece_external(
        self,
        env_id: int,
        color: int | None = None,
        position_xyz: Sequence[float] | None = None,
    ) -> int | None:
        """Bridge / Unity helper: spawn one piece on the table."""
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        self._piece_active[env_ids] = False
        self._piece_grasped[env_ids] = False
        slot = 0
        if position_xyz is None:
            self._spawn_piece(env_ids, slot)
        else:
            obj = self.pieces[slot]
            origins = _as_tensor(self.scene.env_origins)[env_ids]
            pos = torch.tensor(
                [[float(position_xyz[0]), float(position_xyz[1]), self._piece_on_table_z()]],
                device=self.device,
            )
            quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device)
            pose = torch.cat((pos + origins, quat), dim=-1)
            vel = torch.zeros((1, 6), device=self.device)
            obj.write_root_pose_to_sim(pose, env_ids)
            obj.write_root_velocity_to_sim(vel, env_ids)
            self._piece_active[env_ids, slot] = True
            self._piece_grasped[env_ids, slot] = False
            c = int(color) % self.cfg.num_colors if color is not None else 0
            self._piece_color[env_ids, slot] = c
        return slot

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        arm_delta = self.robot_dof_speed_scales[self._arm_ids] * self.dt * self.actions[:, :-1] * self.cfg.action_scale
        grip_cmd = self.actions[:, -1:]
        grip_delta = (
            self.robot_dof_speed_scales[self._gripper_ids] * self.dt * grip_cmd * self.cfg.action_scale
        )

        self.robot_dof_targets[:, self._arm_ids] += arm_delta
        self.robot_dof_targets[:, self._gripper_ids] += grip_delta
        self.robot_dof_targets[:, self._frozen_ids] = self._default_joint_pos[self._frozen_ids]

        self.robot_dof_targets[:] = torch.clamp(
            self.robot_dof_targets, self.robot_dof_drive_lower, self.robot_dof_drive_upper
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _update_grasp_flags(self) -> None:
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]
        grip = _as_tensor(self.robot.data.joint_pos)[:, self._gripper_driver_idx]
        closing = grip < 0.55 * (self.cfg.gripper_open + self.cfg.gripper_close)
        for slot, obj in enumerate(self.pieces):
            active = self._piece_active[:, slot]
            if not torch.any(active):
                self._piece_grasped[:, slot] = False
                continue
            obj_pos = _as_tensor(obj.data.root_pos_w)
            dist = torch.norm(obj_pos - ee_pos, dim=-1)
            grasped = active & closing & (dist < self.cfg.grasp_dist)
            self._piece_grasped[:, slot] = grasped
            if torch.any(grasped):
                ids = grasped.nonzero(as_tuple=False).flatten()
                quat = _as_tensor(obj.data.root_quat_w)[ids]
                pose = torch.cat((ee_pos[ids], quat), dim=-1)
                vel = torch.zeros((len(ids), 6), device=self.device)
                obj.write_root_pose_to_sim(pose, ids)
                obj.write_root_velocity_to_sim(vel, ids)

    def _nearest_active_piece_local(self, ee_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origins = _as_tensor(self.scene.env_origins)
        best = torch.full((self.num_envs,), 1e6, device=self.device)
        nearest_pos = ee_local.clone()
        grasped_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for slot, obj in enumerate(self.pieces):
            active = self._piece_active[:, slot]
            if not torch.any(active):
                continue
            pos_l = _as_tensor(obj.data.root_pos_w) - origins
            d = torch.norm(pos_l - ee_local, dim=-1)
            d = torch.where(active, d, torch.full_like(d, 1e6))
            better = d < best
            best = torch.where(better, d, best)
            nearest_pos = torch.where(better.unsqueeze(-1), pos_l, nearest_pos)
            grasped_any |= self._piece_grasped[:, slot]
        return best, nearest_pos, grasped_any

    def _get_observations(self) -> dict:
        self._update_grasp_flags()

        jp = _as_tensor(self.robot.data.joint_pos)
        jv = _as_tensor(self.robot.data.joint_vel)
        origins = _as_tensor(self.scene.env_origins)
        ee_w = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]
        ee_local = ee_w - origins

        arm_pos = jp[:, self._arm_ids]
        arm_vel = jv[:, self._arm_ids] * self.cfg.dof_velocity_scale
        grip_pos = jp[:, self._gripper_driver_idx].unsqueeze(-1)
        grip_vel = jv[:, self._gripper_driver_idx].unsqueeze(-1) * self.cfg.dof_velocity_scale

        dist, piece_local, grasped = self._nearest_active_piece_local(ee_local)
        piece_vel = torch.zeros((self.num_envs, 3), device=self.device)
        for slot, obj in enumerate(self.pieces):
            active = self._piece_active[:, slot]
            if torch.any(active):
                vel = _as_tensor(obj.data.root_lin_vel_w)
                piece_vel = torch.where(active.unsqueeze(-1), vel, piece_vel)

        ee_to_piece = piece_local - ee_local
        piece_to_bucket = self._bucket_target.unsqueeze(0) - piece_local
        lifted = (piece_local[:, 2] > self.cfg.lift_height).float().unsqueeze(-1)
        grasped_f = grasped.float().unsqueeze(-1)

        obs = torch.cat(
            [
                arm_pos,
                arm_vel,
                grip_pos,
                grip_vel,
                piece_local,
                piece_vel,
                ee_to_piece,
                piece_to_bucket,
                grasped_f,
                lifted,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _in_bucket(self, pos_l: torch.Tensor) -> torch.Tensor:
        center = self._bucket_target
        in_xy = torch.norm(pos_l[:, :2] - center[:2], dim=-1) < self.cfg.success_xy_radius
        in_z = (pos_l[:, 2] >= self.cfg.bucket_z_min) & (pos_l[:, 2] <= self.cfg.success_z_max)
        return in_xy & in_z

    def _get_rewards(self) -> torch.Tensor:
        origins = _as_tensor(self.scene.env_origins)
        ee_local = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx] - origins
        dist, piece_local, grasped = self._nearest_active_piece_local(ee_local)

        approach = (self._prev_piece_dist - dist) * self.cfg.approach_reward_scale
        self._prev_piece_dist = dist.clamp(max=2.0)

        dist_r = (-dist) * self.cfg.dist_reward_scale * 0.1
        grasp_r = grasped.float() * self.cfg.grasp_reward_scale
        lift_r = (grasped & (piece_local[:, 2] > self.cfg.lift_height)).float() * self.cfg.lift_reward_scale

        bucket_delta = piece_local - self._bucket_target.unsqueeze(0)
        bucket_dist = torch.norm(bucket_delta[:, :2], dim=-1)
        bucket_r = grasped.float() * (-bucket_dist) * self.cfg.bucket_approach_scale * 0.2

        placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for slot, obj in enumerate(self.pieces):
            active = self._piece_active[:, slot]
            if not torch.any(active):
                continue
            pos_l = _as_tensor(obj.data.root_pos_w) - origins
            in_bucket = self._in_bucket(pos_l) & active & ~self._piece_grasped[:, slot]
            placed |= in_bucket
            dropped |= active & (pos_l[:, 2] < self.cfg.floor_z) & ~in_bucket

        place_r = placed.float() * self.cfg.place_reward_scale
        drop_r = dropped.float() * (-self.cfg.drop_penalty)
        action_pen = torch.sum(self.actions**2, dim=-1) * (-self.cfg.action_penalty_scale)

        self._episode_success |= placed
        if torch.any(placed):
            self._placed_count[placed] += 1

        reward = dist_r + approach + grasp_r + lift_r + bucket_r + place_r + drop_r + action_pen

        self.extras.setdefault("log", {}).update(
            {
                "Metrics/place_rate": placed.float().mean().item(),
                "Metrics/grasp_rate": grasped.float().mean().item(),
                "Metrics/mean_piece_dist": dist.mean().item(),
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        origins = _as_tensor(self.scene.env_origins)
        for slot, obj in enumerate(self.pieces):
            active = self._piece_active[:, slot]
            pos_l = _as_tensor(obj.data.root_pos_w) - origins
            dropped |= active & (pos_l[:, 2] < self.cfg.floor_z)
        success_done = self._episode_success
        terminated = dropped | success_done
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        self.robot.reset(env_ids)
        for obj in self.pieces:
            obj.reset(env_ids)

        self.robot_dof_targets[env_ids] = self._default_joint_pos.unsqueeze(0).expand(len(env_ids), -1)
        self._piece_active[env_ids] = False
        self._piece_grasped[env_ids] = False
        self._prev_piece_dist[env_ids] = 1.0
        self._episode_success[env_ids] = False
        self._placed_count[env_ids] = 0

        self._activate_random_piece(env_ids)
        self._update_grasp_flags()

        self.episode_length_buf[env_ids] = 0
