# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL: UR10e sorts target-colored cubes to a side table; rejects go to end trash."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .conveyor_color_env_cfg import COLOR_RGB, ConveyorColorEnvCfg


def _as_tensor(data) -> torch.Tensor:
    return data.torch if hasattr(data, "torch") else data


class ConveyorColorEnv(DirectRLEnv):
    cfg: ConveyorColorEnvCfg

    def __init__(self, cfg: ConveyorColorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        joint_limits = _as_tensor(self.robot.data.soft_joint_pos_limits)[0]
        self.robot_dof_lower_limits = joint_limits[:, 0].to(self.device)
        self.robot_dof_upper_limits = joint_limits[:, 1].to(self.device)

        self._arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._gripper_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names, preserve_order=True)
        if len(self._arm_ids) != 6:
            raise RuntimeError(f"Expected 6 UR arm joints, found {len(self._arm_ids)}.")
        if len(self._gripper_ids) < 1:
            raise RuntimeError("Expected Robotiq finger_joint, found none.")
        self._arm_ids = torch.tensor(self._arm_ids, device=self.device, dtype=torch.long)
        self._gripper_ids = torch.tensor(self._gripper_ids, device=self.device, dtype=torch.long)

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self._ee_body_idx = self._resolve_ee_body_index()

        two_pi = self.cfg.physx_drive_angle_limit
        self.robot_dof_drive_lower = torch.maximum(
            self.robot_dof_lower_limits, torch.full_like(self.robot_dof_lower_limits, -two_pi)
        )
        self.robot_dof_drive_upper = torch.minimum(
            self.robot_dof_upper_limits, torch.full_like(self.robot_dof_upper_limits, two_pi)
        )

        n = self.cfg.num_object_slots
        self._obj_active = torch.zeros((self.num_envs, n), dtype=torch.bool, device=self.device)
        self._obj_color = torch.zeros((self.num_envs, n), dtype=torch.long, device=self.device)
        self._obj_grasped = torch.zeros((self.num_envs, n), dtype=torch.bool, device=self.device)
        self._target_color = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._spawn_timer = torch.zeros(self.num_envs, device=self.device)
        self._prev_target_dist = torch.ones(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Bridge can suppress auto-spawn (await_spawn mode)
        self.auto_spawn_enabled = True

    def _resolve_ee_body_index(self) -> int:
        for key in (
            [self.cfg.ee_body_name],
            [".*wrist_3.*"],
            [".*tool0.*"],
            [".*robotiq.*base.*"],
            [".*end_effector.*"],
        ):
            ids, _ = self.robot.find_bodies(key)
            if len(ids) >= 1:
                return int(ids[0])
        # Fall back to last body
        return int(self.robot.num_bodies - 1)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.belt = RigidObject(self.cfg.belt_cfg)
        self.sort_table = RigidObject(self.cfg.sort_table_cfg)
        self.trash = RigidObject(self.cfg.trash_cfg)
        self.objects: list[RigidObject] = [
            RigidObject(self.cfg.object_0_cfg),
            RigidObject(self.cfg.object_1_cfg),
            RigidObject(self.cfg.object_2_cfg),
            RigidObject(self.cfg.object_3_cfg),
        ]

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["belt"] = self.belt
        self.scene.rigid_objects["sort_table"] = self.sort_table
        self.scene.rigid_objects["trash"] = self.trash
        for i, obj in enumerate(self.objects):
            self.scene.rigid_objects[f"object_{i}"] = obj

        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.78, 0.78))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self.robot_dof_targets.clone()
        # Arm deltas
        arm_delta = self.actions[:, :6] * self.cfg.action_scale * self.dt
        targets[:, self._arm_ids] = targets[:, self._arm_ids] + arm_delta
        # Gripper: -1 open … +1 close
        g = 0.5 * (self.actions[:, 6] + 1.0)
        g_cmd = self.cfg.gripper_open + g * (self.cfg.gripper_close - self.cfg.gripper_open)
        targets[:, self._gripper_ids[0]] = g_cmd

        targets = torch.clamp(targets, self.robot_dof_drive_lower, self.robot_dof_drive_upper)
        self.robot_dof_targets[:] = targets
        self.robot.set_joint_position_target(targets)

    # ------------------------------------------------------------------ belt / spawn
    def _park_slot(self, env_ids: torch.Tensor, slot: int) -> None:
        obj = self.objects[slot]
        n = len(env_ids)
        pos = torch.zeros((n, 3), device=self.device)
        pos[:, 2] = -0.5
        origins = _as_tensor(self.scene.env_origins)[env_ids]
        pos_w = pos + origins
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 3] = 1.0
        pose = torch.cat((pos_w, quat), dim=-1)
        vel = torch.zeros((n, 6), device=self.device)
        obj.write_root_pose_to_sim(pose, env_ids)
        obj.write_root_velocity_to_sim(vel, env_ids)
        self._obj_active[env_ids, slot] = False
        self._obj_grasped[env_ids, slot] = False

    def _belt_surface_z(self) -> float:
        """World/env-local Z of the belt top face."""
        return float(self.cfg.belt_pos[2] + 0.5 * self.cfg.belt_size[2])

    def _cube_on_belt_z(self) -> float:
        """Cube center Z when resting on the belt."""
        return self._belt_surface_z() + float(self.cfg.cube_half_size)

    def _sample_belt_spawn_xy(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Random X/Y on the belt surface (env-local), within margins."""
        cx = float(self.cfg.belt_pos[0])
        hx = float(self.cfg.belt_half_x)
        x = cx + (torch.rand(n, device=self.device) * 2.0 - 1.0) * hx
        y = self.cfg.spawn_y_min + torch.rand(n, device=self.device) * (
            self.cfg.spawn_y_max - self.cfg.spawn_y_min
        )
        return x, y

    def _clamp_to_belt_xy(self, local_xy: torch.Tensor) -> torch.Tensor:
        """Clamp env-local XY onto the usable belt footprint."""
        out = local_xy.clone()
        cx = float(self.cfg.belt_pos[0])
        hx = float(self.cfg.belt_half_x)
        out[:, 0] = torch.clamp(out[:, 0], cx - hx, cx + hx)
        out[:, 1] = torch.clamp(out[:, 1], float(self.cfg.belt_y_min), float(self.cfg.belt_y_max))
        return out

    def _activate_slot(
        self,
        env_ids: torch.Tensor,
        slot: int,
        color: torch.Tensor,
        *,
        y: float | None = None,
        x: float | None = None,
        z: float | None = None,
    ) -> None:
        """Place an object on the belt for the given envs (random on-belt if x/y omitted)."""
        obj = self.objects[slot]
        n = len(env_ids)
        pos = torch.zeros((n, 3), device=self.device)
        if x is None or y is None:
            sx, sy = self._sample_belt_spawn_xy(n)
            pos[:, 0] = sx if x is None else float(x)
            pos[:, 1] = sy if y is None else float(y)
        else:
            pos[:, 0] = float(x)
            pos[:, 1] = float(y)
        clamped = self._clamp_to_belt_xy(pos[:, :2])
        pos[:, 0] = clamped[:, 0]
        pos[:, 1] = clamped[:, 1]
        # Always pin height to belt surface so cubes spawn and stay on the belt
        pos[:, 2] = self._cube_on_belt_z() if z is None else float(z)
        origins = _as_tensor(self.scene.env_origins)[env_ids]
        pos_w = pos + origins
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 3] = 1.0
        pose = torch.cat((pos_w, quat), dim=-1)
        vel = torch.zeros((n, 6), device=self.device)
        vel[:, 1] = self.cfg.belt_speed
        obj.write_root_pose_to_sim(pose, env_ids)
        obj.write_root_velocity_to_sim(vel, env_ids)
        self._obj_active[env_ids, slot] = True
        self._obj_grasped[env_ids, slot] = False
        self._obj_color[env_ids, slot] = color.to(dtype=torch.long)

    def spawn_object_external(
        self,
        env_id: int,
        color: int,
        position_xyz: Sequence[float] | None = None,
    ) -> int | None:
        """Bridge helper: activate first free slot. Returns slot index or None."""
        free = (~self._obj_active[env_id]).nonzero(as_tuple=False).flatten()
        if free.numel() == 0:
            return None
        slot = int(free[0].item())
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        color_t = torch.tensor([int(color) % self.cfg.num_colors], device=self.device, dtype=torch.long)
        if position_xyz is None:
            self._activate_slot(env_ids, slot, color_t)
        else:
            # Keep Unity-requested XY if on-belt; always force Z onto the belt surface
            self._activate_slot(
                env_ids,
                slot,
                color_t,
                x=float(position_xyz[0]),
                y=float(position_xyz[1]),
                z=None,
            )
        return slot

    def _maybe_spawn(self) -> None:
        if not self.auto_spawn_enabled:
            return
        self._spawn_timer += self.dt
        ready = self._spawn_timer >= self.cfg.spawn_interval_s
        if not torch.any(ready):
            return
        env_ids = ready.nonzero(as_tuple=False).flatten()
        self._spawn_timer[env_ids] = 0.0
        for env_i in env_ids.tolist():
            free = (~self._obj_active[env_i]).nonzero(as_tuple=False).flatten()
            if free.numel() == 0:
                continue
            slot = int(free[0].item())
            color = torch.randint(0, self.cfg.num_colors, (1,), device=self.device)
            self._activate_slot(
                torch.tensor([env_i], device=self.device, dtype=torch.long),
                slot,
                color,
            )

    def _deposit_on_trash(self, env_ids: torch.Tensor, slot: int) -> None:
        """Place cubes that left the belt onto the trash platform surface."""
        if env_ids.numel() == 0:
            return
        obj = self.objects[slot]
        n = len(env_ids)
        pos = torch.zeros((n, 3), device=self.device)
        pos[:, 0] = float(self.cfg.trash_pos[0])
        pos[:, 1] = float(self.cfg.trash_pos[1])
        pos[:, 2] = float(self.cfg.trash_pos[2]) + 0.5 * float(self.cfg.trash_size[2]) + float(
            self.cfg.cube_half_size
        )
        origins = _as_tensor(self.scene.env_origins)[env_ids]
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 3] = 1.0
        pose = torch.cat((pos + origins, quat), dim=-1)
        vel = torch.zeros((n, 6), device=self.device)
        obj.write_root_pose_to_sim(pose, env_ids)
        obj.write_root_velocity_to_sim(vel, env_ids)
        self._obj_grasped[env_ids, slot] = False

    def _in_zone(
        self,
        pos_l: torch.Tensor,
        center: tuple[float, float, float],
        xy_radius: float,
        z_min: float,
        z_max: float,
    ) -> torch.Tensor:
        center_t = torch.tensor(center, device=self.device)
        in_xy = torch.norm(pos_l[:, :2] - center_t[:2], dim=-1) < xy_radius
        in_z = (pos_l[:, 2] >= z_min) & (pos_l[:, 2] < z_max)
        return in_xy & in_z

    def _apply_belt_motion(self) -> None:
        """Integrate +Y motion and keep cubes seated on the belt (no fall-off)."""
        surface_z = self._cube_on_belt_z()
        for slot, obj in enumerate(self.objects):
            active = self._obj_active[:, slot] & ~self._obj_grasped[:, slot]
            if not torch.any(active):
                continue
            env_ids = active.nonzero(as_tuple=False).flatten()
            pos_w = _as_tensor(obj.data.root_pos_w)[env_ids].clone()
            origins = _as_tensor(self.scene.env_origins)[env_ids]
            local = pos_w - origins

            # Skip cubes already off the belt (trash / sort table) — avoids re-driving them
            # every frame after deposit (PhysX write spam can hard-crash Kit).
            on_belt = local[:, 1] <= float(self.cfg.belt_y_max)
            if not torch.any(on_belt):
                continue
            env_ids = env_ids[on_belt]
            local = local[on_belt]
            origins = origins[on_belt]

            # Advance along belt
            local[:, 1] += self.cfg.belt_speed * self.dt

            # End of belt → deposit on trash platform (scored in rewards)
            past = local[:, 1] > self.cfg.belt_y_max
            if torch.any(past):
                past_ids = env_ids[past]
                self._deposit_on_trash(past_ids, slot)
                keep = ~past
                env_ids = env_ids[keep]
                local = local[keep]
                origins = origins[keep]
                if env_ids.numel() == 0:
                    continue

            # Stay on belt footprint + surface
            local[:, :2] = self._clamp_to_belt_xy(local[:, :2])
            local[:, 2] = surface_z
            pos_w = local + origins

            quat = torch.zeros((len(env_ids), 4), device=self.device)
            quat[:, 3] = 1.0
            pose = torch.cat((pos_w, quat), dim=-1)
            vel = torch.zeros((len(env_ids), 6), device=self.device)
            vel[:, 1] = self.cfg.belt_speed
            obj.write_root_pose_to_sim(pose, env_ids)
            obj.write_root_velocity_to_sim(vel, env_ids)

    def _update_grasp_flags(self) -> None:
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]
        grip = _as_tensor(self.robot.data.joint_pos)[:, self._gripper_ids[0]]
        closing = grip > 0.35 * (self.cfg.gripper_open + self.cfg.gripper_close)
        for slot, obj in enumerate(self.objects):
            active = self._obj_active[:, slot]
            if not torch.any(active):
                self._obj_grasped[:, slot] = False
                continue
            obj_pos = _as_tensor(obj.data.root_pos_w)
            dist = torch.norm(obj_pos - ee_pos, dim=-1)
            grasped = active & closing & (dist < self.cfg.grasp_dist)
            self._obj_grasped[:, slot] = grasped
            # Magnet-lite: hold grasped object at EE
            if torch.any(grasped):
                ids = grasped.nonzero(as_tuple=False).flatten()
                quat = _as_tensor(obj.data.root_quat_w)[ids]
                pose = torch.cat((ee_pos[ids], quat), dim=-1)
                vel = torch.zeros((len(ids), 6), device=self.device)
                obj.write_root_pose_to_sim(pose, ids)
                obj.write_root_velocity_to_sim(vel, ids)

    # ------------------------------------------------------------------ obs / reward / done
    def _get_observations(self) -> dict:
        self._maybe_spawn()
        self._apply_belt_motion()
        self._update_grasp_flags()

        jp = _as_tensor(self.robot.data.joint_pos)
        jv = _as_tensor(self.robot.data.joint_vel)
        origins = _as_tensor(self.scene.env_origins)
        ee_w = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]
        ee_local = ee_w - origins
        bin_local = torch.tensor(self.cfg.bin_pos, device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        arm_pos = jp[:, self._arm_ids]
        arm_vel = jv[:, self._arm_ids] * self.cfg.dof_velocity_scale
        grip_pos = jp[:, self._gripper_ids[0]].unsqueeze(-1)
        grip_vel = jv[:, self._gripper_ids[0]].unsqueeze(-1) * self.cfg.dof_velocity_scale

        target_oh = torch.nn.functional.one_hot(self._target_color, self.cfg.num_colors).float()

        obj_feats = []
        for slot, obj in enumerate(self.objects):
            pos_w = _as_tensor(obj.data.root_pos_w)
            vel_w = _as_tensor(obj.data.root_lin_vel_w)
            pos_l = pos_w - origins
            active = self._obj_active[:, slot].float().unsqueeze(-1)
            color_oh = torch.nn.functional.one_hot(self._obj_color[:, slot], self.cfg.num_colors).float()
            # Zero out inactive slots
            pos_l = pos_l * active
            vel_w = vel_w * active
            color_oh = color_oh * active
            obj_feats.extend([pos_l, vel_w, color_oh, active])

        # Nearest target-colored active object
        nearest_delta = torch.zeros((self.num_envs, 3), device=self.device)
        best = torch.full((self.num_envs,), 1e6, device=self.device)
        for slot, obj in enumerate(self.objects):
            mask = self._obj_active[:, slot] & (self._obj_color[:, slot] == self._target_color)
            if not torch.any(mask):
                continue
            pos_l = _as_tensor(obj.data.root_pos_w) - origins
            d = torch.norm(pos_l - ee_local, dim=-1)
            d = torch.where(mask, d, torch.full_like(d, 1e6))
            better = d < best
            best = torch.where(better, d, best)
            nearest_delta = torch.where(better.unsqueeze(-1), pos_l - ee_local, nearest_delta)

        obs = torch.cat(
            [
                arm_pos,
                arm_vel,
                grip_pos,
                grip_vel,
                target_oh,
                *obj_feats,
                ee_local,
                nearest_delta,
                bin_local - ee_local,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        origins = _as_tensor(self.scene.env_origins)
        ee_local = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx] - origins
        sort_local = torch.tensor(self.cfg.sort_table_pos, device=self.device)

        # Distance to nearest target object
        best = torch.full((self.num_envs,), 1e6, device=self.device)
        nearest_pos = ee_local.clone()
        holding_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        holding_wrong = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for slot, obj in enumerate(self.objects):
            active = self._obj_active[:, slot]
            is_target = active & (self._obj_color[:, slot] == self._target_color)
            pos_l = _as_tensor(obj.data.root_pos_w) - origins
            d = torch.norm(pos_l - ee_local, dim=-1)
            d = torch.where(is_target, d, torch.full_like(d, 1e6))
            better = d < best
            best = torch.where(better, d, best)
            nearest_pos = torch.where(better.unsqueeze(-1), pos_l, nearest_pos)

            holding_target |= self._obj_grasped[:, slot] & is_target
            holding_wrong |= self._obj_grasped[:, slot] & active & ~is_target

        approach = (self._prev_target_dist - best) * self.cfg.approach_reward_scale
        self._prev_target_dist = best.clamp(max=2.0)

        grasp_r = holding_target.float() * self.cfg.grasp_reward_scale
        wrong_r = holding_wrong.float() * (-self.cfg.wrong_grasp_penalty)
        lift_r = (holding_target & (nearest_pos[:, 2] > self.cfg.lift_height)).float() * self.cfg.lift_reward_scale

        sort_delta = nearest_pos - sort_local.unsqueeze(0)
        sort_dist = torch.norm(sort_delta[:, :2], dim=-1)
        bin_r = holding_target.float() * (-sort_dist) * self.cfg.bin_approach_scale * 0.15

        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reject_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        wrong_on_table = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        target_trashed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for slot, obj in enumerate(self.objects):
            active = self._obj_active[:, slot]
            if not torch.any(active):
                continue
            is_target = active & (self._obj_color[:, slot] == self._target_color)
            is_reject = active & ~is_target
            pos_l = _as_tensor(obj.data.root_pos_w) - origins
            released = ~self._obj_grasped[:, slot]

            on_sort = self._in_zone(
                pos_l,
                self.cfg.sort_table_pos,
                self.cfg.bin_success_xy,
                self.cfg.bin_success_z_min,
                self.cfg.bin_success_z_max,
            )
            on_trash = self._in_zone(
                pos_l,
                self.cfg.trash_pos,
                self.cfg.trash_success_xy,
                self.cfg.trash_success_z_min,
                self.cfg.trash_success_z_max,
            )

            # Target on side sort table → success
            ok = is_target & on_sort & released
            success |= ok
            if torch.any(ok):
                self._park_slot(ok.nonzero(as_tuple=False).flatten(), slot)

            # Non-target on trash → correct reject
            rej = is_reject & on_trash & released
            reject_ok |= rej
            if torch.any(rej):
                self._park_slot(rej.nonzero(as_tuple=False).flatten(), slot)

            # Non-target on sort table → wrong
            bad_table = is_reject & on_sort & released
            wrong_on_table |= bad_table
            if torch.any(bad_table):
                self._park_slot(bad_table.nonzero(as_tuple=False).flatten(), slot)

            # Target on trash → missed / fail
            miss = is_target & on_trash & released
            target_trashed |= miss
            if torch.any(miss):
                self._park_slot(miss.nonzero(as_tuple=False).flatten(), slot)

        self._success |= success
        success_r = success.float() * self.cfg.success_reward
        reject_r = reject_ok.float() * self.cfg.reject_reward
        wrong_table_r = wrong_on_table.float() * (-self.cfg.wrong_on_table_penalty)
        trash_miss_r = target_trashed.float() * (-self.cfg.target_to_trash_penalty)
        self._failed |= target_trashed | wrong_on_table

        # Fail: target fell
        fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for slot, obj in enumerate(self.objects):
            is_target = self._obj_active[:, slot] & (self._obj_color[:, slot] == self._target_color)
            pos_l = _as_tensor(obj.data.root_pos_w) - origins
            fell |= is_target & (pos_l[:, 2] < self.cfg.fall_height) & ~self._obj_grasped[:, slot]
        self._failed |= fell
        drop_r = fell.float() * (-self.cfg.drop_penalty)

        action_pen = -self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)
        return (
            approach
            + grasp_r
            + wrong_r
            + lift_r
            + bin_r
            + success_r
            + reject_r
            + wrong_table_r
            + trash_miss_r
            + drop_r
            + action_pen
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._success | self._failed
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        joint_pos = _as_tensor(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        root = _as_tensor(self.robot.data.default_root_state)[env_ids].clone()
        root[:, :3] += _as_tensor(self.scene.env_origins)[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)

        for slot in range(self.cfg.num_object_slots):
            self._park_slot(env_ids, slot)

        self._target_color[env_ids] = torch.randint(
            0, self.cfg.num_colors, (len(env_ids),), device=self.device
        )
        self._spawn_timer[env_ids] = 0.0
        self._prev_target_dist[env_ids] = 1.0
        self._success[env_ids] = False
        self._failed[env_ids] = False

        # Seed one object of (sometimes) target color for denser learning
        if self.auto_spawn_enabled and len(env_ids) > 0:
            colors = self._target_color[env_ids].clone()
            # 50% chance exact target, else random
            rand = torch.randint(0, self.cfg.num_colors, (len(env_ids),), device=self.device)
            use_target = torch.rand(len(env_ids), device=self.device) < 0.5
            colors = torch.where(use_target, colors, rand)
            for i, env_i in enumerate(env_ids.tolist()):
                self._activate_slot(
                    torch.tensor([env_i], device=self.device, dtype=torch.long),
                    0,
                    colors[i : i + 1],
                )
