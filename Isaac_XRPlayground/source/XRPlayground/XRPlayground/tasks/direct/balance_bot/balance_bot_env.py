# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL: 2-DOF tilting tray keeps hand-sized ball(s) from falling (ball-on-plate)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from .balance_bot_env_cfg import BalanceBotEnvCfg


def _as_tensor(data) -> torch.Tensor:
    return data.torch if hasattr(data, "torch") else data


class BalanceBotEnv(DirectRLEnv):
    cfg: BalanceBotEnvCfg

    def __init__(self, cfg: BalanceBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        n = self.num_envs
        self._roll = torch.zeros(n, device=self.device)
        self._pitch = torch.zeros(n, device=self.device)
        self._roll_vel = torch.zeros(n, device=self.device)
        self._pitch_vel = torch.zeros(n, device=self.device)
        self._n_balls = torch.ones(n, dtype=torch.long, device=self.device)
        self._ball_active = torch.zeros((n, self.cfg.num_ball_slots), dtype=torch.bool, device=self.device)
        self._episode_held = torch.ones(n, dtype=torch.bool, device=self.device)
        self._dropped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._hold_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((n, self.cfg.action_space), device=self.device)
        # Bridge may suppress auto ball drops later; kept for API parity with other stations
        self.auto_spawn_enabled = True
        self._hold_ema = 0.0
        self._unlock_good_batches = 0
        self._curriculum_unlocked = int(getattr(self.cfg, "curriculum_start_stage", 0)) >= 1

    def _setup_scene(self):
        self.pedestal = RigidObject(self.cfg.pedestal_cfg)
        self.tray = RigidObject(self.cfg.tray_cfg)
        self.balls = [
            RigidObject(self.cfg.ball_0_cfg),
            RigidObject(self.cfg.ball_1_cfg),
        ]
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["base_link"] = self.pedestal
        self.scene.rigid_objects["tray_link"] = self.tray
        self.scene.rigid_objects["Ball_0"] = self.balls[0]
        self.scene.rigid_objects["Ball_1"] = self.balls[1]
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _curriculum_stage(self) -> int:
        if int(getattr(self.cfg, "curriculum_start_stage", 0)) >= 1:
            return 1
        if self._curriculum_unlocked:
            return 1
        steps = max(int(self.cfg.curriculum_steps), 1)
        if int(self.common_step_counter) >= steps:
            self._curriculum_unlocked = True
            return 1
        return 0

    def _n_balls_for_stage(self) -> int:
        stage = self._curriculum_stage()
        if stage >= 1:
            return int(self.cfg.curriculum_phase2_balls)
        return int(self.cfg.curriculum_phase1_balls)

    def _update_curriculum_from_hold(self, mean_hold: float) -> None:
        """Advance to 2-ball once hold looks solid (or after curriculum_steps)."""
        if self._curriculum_stage() >= 1:
            return
        alpha = 0.2
        self._hold_ema = (1.0 - alpha) * self._hold_ema + alpha * float(mean_hold)
        unlock = float(getattr(self.cfg, "curriculum_unlock_hold_steps", 350.0))
        patience = max(int(getattr(self.cfg, "curriculum_unlock_patience", 40)), 1)
        if self._hold_ema >= unlock:
            self._unlock_good_batches += 1
        else:
            self._unlock_good_batches = 0
        if self._unlock_good_batches >= patience:
            self._curriculum_unlocked = True

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Rate command → integrate tilt with soft clamp
        cmd = self.actions * self.cfg.action_scale
        self._roll_vel = cmd[:, 0]
        self._pitch_vel = cmd[:, 1]
        self._roll = (self._roll + self._roll_vel * self.dt).clamp(-self.cfg.max_tilt_rad, self.cfg.max_tilt_rad)
        self._pitch = (self._pitch + self._pitch_vel * self.dt).clamp(-self.cfg.max_tilt_rad, self.cfg.max_tilt_rad)
        # Zero commanded rate at soft stops
        at_roll_lim = (self._roll.abs() >= self.cfg.max_tilt_rad - 1e-4) & (self._roll * self._roll_vel > 0)
        at_pitch_lim = (self._pitch.abs() >= self.cfg.max_tilt_rad - 1e-4) & (self._pitch * self._pitch_vel > 0)
        self._roll_vel = torch.where(at_roll_lim, torch.zeros_like(self._roll_vel), self._roll_vel)
        self._pitch_vel = torch.where(at_pitch_lim, torch.zeros_like(self._pitch_vel), self._pitch_vel)

    def _apply_action(self) -> None:
        self._write_tray_pose(None)

    def _write_tray_pose(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.tray._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids]
        offset = torch.tensor(self.cfg.tray_center, device=self.device, dtype=origins.dtype)
        pos = origins + offset.unsqueeze(0)
        quat = quat_from_euler_xyz(
            self._roll[env_ids], self._pitch[env_ids], torch.zeros_like(self._roll[env_ids])
        )
        pose = torch.cat((pos, quat), dim=-1)
        # Kinematic tray: pose only — write_root_velocity_to_sim raises on CPU PhysX.
        self.tray.write_root_pose_to_sim(pose, env_ids)

    def _get_observations(self) -> dict:
        obs_parts = [
            self._roll.unsqueeze(-1),
            self._pitch.unsqueeze(-1),
            self._roll_vel.unsqueeze(-1) * self.cfg.dof_velocity_scale,
            self._pitch_vel.unsqueeze(-1) * self.cfg.dof_velocity_scale,
        ]
        tray_c = torch.tensor(self.cfg.tray_center, device=self.device)
        for i, ball in enumerate(self.balls):
            pos = _as_tensor(ball.data.root_pos_w) - self.scene.env_origins
            vel = _as_tensor(ball.data.root_lin_vel_w)
            active = self._ball_active[:, i].float().unsqueeze(-1)
            rel = (pos - tray_c.unsqueeze(0)) * active
            vel_m = vel * active
            obs_parts.extend([rel, vel_m, active])
        stage = float(self._curriculum_stage())
        n_balls = self._n_balls.float().unsqueeze(-1)
        stage_t = torch.full((self.num_envs, 1), stage, device=self.device)
        obs_parts.extend([n_balls, stage_t])
        # 4 + 2*(3+3+1) + 2 = 20
        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs}

    def _ball_off_tray(self, ball: RigidObject, active: torch.Tensor) -> torch.Tensor:
        pos = _as_tensor(ball.data.root_pos_w) - self.scene.env_origins
        cx, cy, cz = self.cfg.tray_center
        dx = (pos[:, 0] - cx).abs()
        dy = (pos[:, 1] - cy).abs()
        z = pos[:, 2]
        off_xy = (dx > self.cfg.tray_half_xy) | (dy > self.cfg.tray_half_xy)
        fell = z < self.cfg.fall_z
        return active & (off_xy | fell)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i, ball in enumerate(self.balls):
            dropped |= self._ball_off_tray(ball, self._ball_active[:, i])
        self._dropped = dropped
        self._episode_held &= ~dropped

        # Hold metric: all active balls near center and slow
        holding = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        tray_c = torch.tensor(self.cfg.tray_center, device=self.device)
        for i, ball in enumerate(self.balls):
            active = self._ball_active[:, i]
            pos = _as_tensor(ball.data.root_pos_w) - self.scene.env_origins
            vel = _as_tensor(ball.data.root_lin_vel_w)
            rel_xy = torch.linalg.norm((pos - tray_c.unsqueeze(0))[:, :2], dim=-1)
            speed = torch.linalg.norm(vel, dim=-1)
            ok = (~active) | ((rel_xy < 0.12) & (speed < 0.35))
            holding &= ok
        self._hold_steps = torch.where(holding, self._hold_steps + 1, torch.zeros_like(self._hold_steps))

        terminated = dropped
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        tray_c = torch.tensor(self.cfg.tray_center, device=self.device)
        center_rew = torch.zeros(self.num_envs, device=self.device)
        vel_pen = torch.zeros(self.num_envs, device=self.device)
        n_active = torch.clamp(self._n_balls.float(), min=1.0)
        for i, ball in enumerate(self.balls):
            active = self._ball_active[:, i].float()
            pos = _as_tensor(ball.data.root_pos_w) - self.scene.env_origins
            vel = _as_tensor(ball.data.root_lin_vel_w)
            dist_xy = torch.linalg.norm((pos - tray_c.unsqueeze(0))[:, :2], dim=-1)
            speed = torch.linalg.norm(vel, dim=-1)
            center_rew = center_rew + active * torch.exp(-6.0 * dist_xy)
            vel_pen = vel_pen + active * speed
            hold_near = active * (dist_xy < 0.12).float() * (speed < 0.35).float()
            center_rew = center_rew + self.cfg.rew_hold_bonus * hold_near

        center_rew = center_rew / n_active
        vel_pen = vel_pen / n_active
        tilt = self._roll.abs() + self._pitch.abs()
        alive = self.cfg.rew_alive * (~self._dropped).float()
        action_pen = self.cfg.rew_action_penalty * torch.sum(self.actions * self.actions, dim=-1)
        drop_pen = self.cfg.rew_drop_penalty * self._dropped.float()
        return (
            alive
            + self.cfg.rew_center * center_rew
            - self.cfg.rew_vel_penalty * vel_pen
            - self.cfg.rew_tilt_penalty * tilt
            - action_pen
            - drop_pen
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.tray._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            mean_hold = self._hold_steps[env_ids].float().mean().item()
            self._update_curriculum_from_hold(mean_hold)
            log = self.extras.setdefault("log", {})
            log["Metrics/hold_rate"] = self._episode_held[env_ids].float().mean().item()
            log["Metrics/drop_rate"] = self._dropped[env_ids].float().mean().item()
            log["Metrics/n_balls"] = self._n_balls[env_ids].float().mean().item()
            log["Metrics/curriculum_stage"] = float(self._curriculum_stage())
            log["Metrics/mean_hold_steps"] = mean_hold
            log["Metrics/hold_ema"] = float(self._hold_ema)

        n_balls = self._n_balls_for_stage()
        self._n_balls[env_ids] = n_balls
        self._roll[env_ids] = 0.0
        self._pitch[env_ids] = 0.0
        self._roll_vel[env_ids] = 0.0
        self._pitch_vel[env_ids] = 0.0
        self._episode_held[env_ids] = True
        self._dropped[env_ids] = False
        self._hold_steps[env_ids] = 0
        self._ball_active[env_ids] = False

        self._write_tray_pose(env_ids)
        self._spawn_balls(env_ids, n_balls)
        self.extras.setdefault("log", {})["Metrics/curriculum_stage"] = float(self._curriculum_stage())

    def _spawn_balls(self, env_ids: torch.Tensor, n_balls: int) -> None:
        tray_c = torch.tensor(self.cfg.tray_center, device=self.device)
        z = tray_c[2] + 0.5 * self.cfg.tray_size[2] + self.cfg.ball_radius + self.cfg.spawn_height_above_tray
        half = self.cfg.spawn_xy_half
        offsets = (
            (0.0, 0.0),
            (0.09, 0.06),
        )
        for i, ball in enumerate(self.balls):
            active = i < n_balls
            self._ball_active[env_ids, i] = active
            if not active:
                # Park under floor
                pose = torch.zeros((len(env_ids), 7), device=self.device)
                pose[:, 0:3] = self.scene.env_origins[env_ids] + torch.tensor(
                    (0.0, 0.0, -0.5), device=self.device
                )
                pose[:, 6] = 1.0  # identity quat (x,y,z,w)
                ball.write_root_pose_to_sim(pose, env_ids)
                ball.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)
                continue

            ox, oy = offsets[i]
            jitter = sample_uniform(-half * 0.35, half * 0.35, (len(env_ids), 2), self.device)
            pos = self.scene.env_origins[env_ids].clone()
            pos[:, 0] += tray_c[0] + ox + jitter[:, 0]
            pos[:, 1] += tray_c[1] + oy + jitter[:, 1]
            pos[:, 2] += z
            quat = torch.zeros((len(env_ids), 4), device=self.device)
            quat[:, 3] = 1.0  # identity (x, y, z, w)
            pose = torch.cat((pos, quat), dim=-1)
            ball.write_root_pose_to_sim(pose, env_ids)
            ball.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)

    # --- Bridge helpers (env-local poses) ---
    def get_joint_state(self, env_id: int = 0) -> tuple[list[str], list[float]]:
        i = int(env_id)
        return list(self.cfg.joint_names), [float(self._roll[i].item()), float(self._pitch[i].item())]

    def get_link_poses_env_local(self, env_id: int = 0) -> list[dict]:
        """Named link poses in isaac_env frame (matches Unity BalanceBotLinkMap)."""
        i = int(env_id)
        origin = self.scene.env_origins[i]
        bx, by, bz = self.cfg.pedestal_pos
        tx, ty, tz = self.cfg.tray_center
        roll = float(self._roll[i].item())
        pitch = float(self._pitch[i].item())
        q_roll = quat_from_euler_xyz(
            torch.tensor([roll], device=self.device),
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
        )[0]
        # Prefer live tray pose from sim
        tray_pos = (_as_tensor(self.tray.data.root_pos_w)[i] - origin)
        tray_q = _as_tensor(self.tray.data.root_quat_w)[i]

        def _pose(name: str, pos_xyz, quat_xyzw):
            x, y, z, w = [float(v) for v in quat_xyzw.tolist()]
            return {
                "name": name,
                "position": [float(v) for v in pos_xyz],
                "orientation_xyzw": [x, y, z, w],
            }

        identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        return [
            _pose("base_link", (bx, by, bz), identity),
            _pose("roll_link", (tx, ty, tz), q_roll),
            _pose("tray_link", tray_pos.tolist(), tray_q),
        ]
