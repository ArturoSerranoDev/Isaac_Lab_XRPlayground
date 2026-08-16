# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL env: Kinova Jaco2 + 3-finger gripper learns to catch thrown balls."""

from __future__ import annotations

from collections import deque
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

        default_joint = _as_tensor(self.robot.data.default_joint_pos)[0]
        self._default_arm_pos = default_joint[self._arm_ids].clone()

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
        self._grasp_miss_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._soft_grasp_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cup_balance_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Gate throw hardness on throw-only rolling success (not mixed in-hand soft_grasp).
        self._catch_ema = 0.0
        self._throw_catch_ema = 0.0
        self._throw_rolling = 0.0
        self._throw_gate_streak = 0
        hist_len = int(getattr(self.cfg, "throw_hist_len", 256))
        self._throw_hist: deque[float] = deque(maxlen=max(hist_len, 32))
        self._frac_ema = 0.0
        phase = str(getattr(self.cfg, "training_phase", "mixed")).lower()
        # Wrap: stay easy. Throw-A/B: start near-zero hardness.
        self._curriculum_cap = 0.05 if phase in ("throw", "throw_a", "throw_b") else 0.15
        self._min_tip_dist = torch.ones(self.num_envs, device=self.device) * 10.0
        self._max_grasp_align = torch.zeros(self.num_envs, device=self.device)
        # Per-env: last reset used in-hand spawn (for throw-only metrics).
        self._spawned_in_hand = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Buoyancy matches Unity offline (physics contract — not a catch magnet).
        ball_mass = float(self.cfg.ball_cfg.spawn.mass_props.mass)
        buoyancy = ball_mass * self.cfg.gravity_full * (1.0 - self.cfg.ball_gravity_scale)
        self._ball_ext_force = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._ball_ext_force[..., 2] = buoyancy
        self._ball_ext_torque = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._ball_assist_force = torch.zeros((self.num_envs, 3), device=self.device)

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

    def _phase(self) -> str:
        return str(getattr(self.cfg, "training_phase", "mixed")).lower()

    def _is_throw_like(self) -> bool:
        return self._phase() in ("throw", "throw_a", "throw_b")

    def _in_drift_mode(self) -> bool:
        """True while non-in-hand episodes use in-aperture drift (arm usually frozen)."""
        phase = self._phase()
        if phase == "throw_a":
            return True
        if phase in ("throw", "throw_b"):
            drift_until = float(getattr(self.cfg, "throw_drift_until_alpha", 0.40))
            return self._curriculum_alpha() < drift_until
        return False

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Freeze arm only after a confirmed catch — early freeze on grasp_hold_count
        # locked cupping poses before the ball was truly between the fingers.
        # Wrap / Throw-A: freeze arm (gripper free until latch).
        # Throw-B: freeze arm only on in-hand + brief early drift; free arm for lobs.
        freeze_arm = self._episode_caught
        freeze_arm_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        phase = self._phase()
        if phase in ("wrap", "throw_a"):
            freeze_arm_only = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        elif phase in ("throw", "throw_b"):
            in_drift = self._in_drift_mode()
            freeze_arm_only = self._spawned_in_hand | (
                (~self._spawned_in_hand)
                & torch.full((self.num_envs,), in_drift, dtype=torch.bool, device=self.device)
            )

        arm_delta = self.robot_dof_speed_scales[self._arm_ids] * self.dt * self.actions[:, :-1] * self.cfg.action_scale
        grip_delta = (
            self.robot_dof_speed_scales[self._gripper_ids]
            * self.dt
            * self.actions[:, -1:]
            * self.cfg.action_scale
        )
        arm_frozen = freeze_arm | freeze_arm_only
        arm_delta = torch.where(arm_frozen.unsqueeze(-1), torch.zeros_like(arm_delta), arm_delta)

        self.robot_dof_targets[:, self._arm_ids] += arm_delta
        self.robot_dof_targets[:, self._gripper_ids] += grip_delta

        joint_pos = _as_tensor(self.robot.data.joint_pos)
        if arm_frozen.any():
            # Lock PD targets to current arm configuration so the arm stays put.
            self.robot_dof_targets[:, self._arm_ids] = torch.where(
                arm_frozen.unsqueeze(-1),
                joint_pos[:, self._arm_ids],
                self.robot_dof_targets[:, self._arm_ids],
            )
        if freeze_arm.any():
            # Hold a firm close; milder after latch to avoid ejecting the ball.
            close_tgt = float(self.cfg.gripper_close_target)
            phase = self._phase()
            if phase in ("wrap", "throw_a"):
                # Gentle hold — full slam ejects the ball after latch.
                close_tgt = (
                    0.30 * float(self.cfg.gripper_open_pos)
                    + 0.70 * float(self.cfg.gripper_close_target)
                )
            elif phase in ("throw", "throw_b"):
                close_tgt = (
                    0.22 * float(self.cfg.gripper_open_pos)
                    + 0.78 * float(self.cfg.gripper_close_target)
                )
            close = torch.full(
                (self.num_envs, self._gripper_ids.numel()),
                close_tgt,
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
            # Keep arm action zeros in wrap even before latch (gripper action still learned).
            if freeze_arm_only.any():
                self.actions[:, :-1] = torch.where(
                    freeze_arm_only.unsqueeze(-1),
                    torch.zeros_like(self.actions[:, :-1]),
                    self.actions[:, :-1],
                )

        self.robot_dof_targets[:] = torch.clamp(
            self.robot_dof_targets, self.robot_dof_drive_lower, self.robot_dof_drive_upper
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)
        # Buoyancy only — catch assist disabled (no spring / blend / finger drive).
        self.ball.set_external_force_and_torque(self._ball_ext_force, self._ball_ext_torque)

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

        # Grasp geometry: cup facing + off-axis radial error (kills sideways tip-chase farming).
        ee_to_tips = tip_center - ee_pos
        grasp_axis = ee_to_tips / torch.linalg.norm(ee_to_tips, dim=-1, keepdim=True).clamp(min=1e-6)
        dist_ee = torch.linalg.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-6)
        grasp_align = (to_ball / dist_ee * grasp_axis).sum(dim=-1, keepdim=True)
        ball_along = (to_ball * grasp_axis).sum(dim=-1, keepdim=True)
        ball_radial = torch.linalg.norm(to_ball - ball_along * grasp_axis, dim=-1, keepdim=True)

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
                grasp_align,
                ball_radial,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _update_grasp_and_body_flags(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update hold / body / cup-balance counters.

        Success requires the ball in the finger aperture along the EE→tip_center
        grasp axis (between palm and tips, not resting on a closed fingertip platform).
        """
        ball_pos_w = _as_tensor(self.ball.data.root_pos_w)
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)
        ball_speed = torch.linalg.norm(ball_vel, dim=-1)
        tip_center, _, closing, _ = self._gripper_state()
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]

        ee_to_tips = tip_center - ee_pos
        ee_to_tips_len = torch.linalg.norm(ee_to_tips, dim=-1)
        grasp_axis = ee_to_tips / ee_to_tips_len.unsqueeze(-1).clamp(min=1e-6)
        ee_to_ball = ball_pos_w - ee_pos
        ball_along = (ee_to_ball * grasp_axis).sum(dim=-1)
        ball_radial_vec = ee_to_ball - ball_along.unsqueeze(-1) * grasp_axis
        ball_radial = torch.linalg.norm(ball_radial_vec, dim=-1)

        dist_ee = torch.linalg.norm(ee_to_ball, dim=-1)
        dist_tips = torch.linalg.norm(ball_pos_w - tip_center, dim=-1)
        dist_palm = dist_ee
        ee_to_ball_n = ee_to_ball / dist_ee.unsqueeze(-1).clamp(min=1e-6)
        grasp_align = (ee_to_ball_n * grasp_axis).sum(dim=-1)

        # Past the fingertip plane (incoming throw OR resting on the closed tip cluster).
        beyond_tips = ball_along > (ee_to_tips_len - self.cfg.grasp_beyond_tips_margin)
        # Cheat: ball parked on a closed tip cluster while NOT facing/wrapping.
        # Do not flag a facing wrap that settles slightly past the tip plane (assist close path).
        near_ee = dist_ee < self.cfg.gripper_near_dist * 1.15
        near_tips = dist_tips < self.cfg.success_tip_dist * 1.5
        tip_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._tip_body_ids]
        tip_dists = torch.linalg.norm(tip_pos - ball_pos_w.unsqueeze(1), dim=-1)
        tip_max = tip_dists.max(dim=-1).values
        tip_mean = tip_dists.mean(dim=-1)
        tip_spread = tip_dists.max(dim=-1).values - tip_dists.min(dim=-1).values
        poke_spread = getattr(self.cfg, "poke_tip_spread", self.cfg.tip_spread_max * 1.3)
        tip_max_lim = float(getattr(self.cfg, "success_tip_max", self.cfg.success_tip_dist * 1.15))
        poke = near_tips & (tip_spread > poke_spread)
        wrap_sym = tip_spread < self.cfg.tip_spread_max
        enclosed = tip_max < tip_max_lim
        wrap_phase = self._phase() == "wrap"
        # Throw-A / early drift: same relaxed enclose gates as Wrap.
        in_drift = self._in_drift_mode()
        drift_episode = (~self._spawned_in_hand) & torch.full(
            (self.num_envs,), in_drift, dtype=torch.bool, device=self.device
        )
        in_hand_episode = self._spawned_in_hand
        relaxed_enclose = wrap_phase | in_hand_episode | drift_episode
        tip_platform = (
            beyond_tips
            & near_tips
            & (ball_speed < self.cfg.success_speed_threshold)
            & (closing > 0.50)
            & ((grasp_align < self.cfg.grasp_align_min) | (~wrap_sym) | (~enclosed))
            & (~self._episode_caught)  # after latch, tip settle is hold — not a cheat penalty
        )
        # Wrap / in-hand: ball starts in aperture; tip-platform / dorsal cup heuristics fight honest enclose.
        tip_platform = tip_platform & (~relaxed_enclose)

        dist = dist_tips
        arm_dist = self._ball_to_arm_dist(ball_pos_w)

        # Settled cheat only — do NOT mark every misaligned approach as cup (that made the arm flee).
        dorsal_cup = (
            near_ee
            & (grasp_align < self.cfg.cup_align_max * 0.55)
            & (ball_speed < self.cfg.success_speed_threshold)
            & (closing > 0.35)
            & (~self._episode_caught)
        )
        dorsal_cup = dorsal_cup & (~relaxed_enclose)
        cup_balance = tip_platform | dorsal_cup
        # Forearm-cup cheat only on true throw episodes (not wrap / in-hand handoff).
        throw_eps = ~relaxed_enclose
        cup_balance = cup_balance | (
            throw_eps
            & ((dist_palm + 0.008) < dist_tips)
            & near_tips
            & (ball_speed < 0.6)
            & (~self._episode_caught)
        )
        misaligned_near = near_tips & (grasp_align < self.cfg.cup_align_max) & (~tip_platform)

        # Soft grasp = real enclosure. After latch, keep success if still enclosed+near+closed
        # (force-close can push ball slightly past tip plane without being a drop).
        along_lo = self.cfg.grasp_along_min * 0.5
        along_hi = torch.maximum(
            torch.full_like(ee_to_tips_len, self.cfg.grasp_along_max),
            ee_to_tips_len * 0.92,
        )
        radial_lim = self.cfg.grasp_radial_max * 1.35
        align_min = self.cfg.grasp_align_min
        # Per-env: relax along/radial/align for wrap + throw in-hand handoff episodes.
        along_lo_t = torch.full_like(ball_along, along_lo)
        along_hi_t = along_hi.clone()
        radial_lim_t = torch.full_like(ball_radial, radial_lim)
        align_min_t = torch.full_like(grasp_align, align_min)
        if relaxed_enclose.any():
            along_lo_t = torch.where(relaxed_enclose, torch.full_like(along_lo_t, -0.05), along_lo_t)
            along_hi_t = torch.where(
                relaxed_enclose,
                torch.maximum(along_hi_t, ee_to_tips_len + 0.06),
                along_hi_t,
            )
            radial_lim_t = torch.where(
                relaxed_enclose,
                torch.full_like(radial_lim_t, self.cfg.grasp_radial_max * 2.0),
                radial_lim_t,
            )
            align_min_t = torch.where(relaxed_enclose, torch.full_like(align_min_t, -1.0), align_min_t)
        soft_core = (
            (dist_tips < self.cfg.success_tip_dist)
            & enclosed
            & (closing > self.cfg.success_close_min)
            & (ball_speed < self.cfg.success_speed_threshold)
            & (grasp_align > align_min_t)
            & wrap_sym
            & (ball_along > along_lo_t)
            & (ball_along < along_hi_t)
            & (ball_radial < radial_lim_t)
            & (ball_pos_w[:, 2] > self.cfg.fall_height_threshold + 0.05)
            & (~dorsal_cup)
            & (~poke)
            & (~tip_platform)
        )
        soft_hold = (
            self._episode_caught
            & (dist_tips < self.cfg.success_tip_dist * 1.35)
            & (tip_max < tip_max_lim * (1.40 if self._phase() == "throw_a" else 1.25))
            & (closing > self.cfg.success_close_min * 0.80)
            & (ball_pos_w[:, 2] > self.cfg.fall_height_threshold + 0.05)
            & (~poke)
        )
        soft_grasp = soft_core | soft_hold
        in_grasp = soft_core & (
            (ball_along > self.cfg.grasp_along_min)
            & (ball_along < self.cfg.grasp_along_max)
            & (ball_radial < self.cfg.grasp_radial_max)
        )
        # Ball resting on forearm/links, not inside the gripper volume
        on_body = (arm_dist < self.cfg.body_contact_radius) & (dist_ee > self.cfg.gripper_near_dist) & (~soft_grasp)

        self._episode_steps += 1
        self._soft_grasp_steps = torch.where(
            soft_grasp, self._soft_grasp_steps + 1, self._soft_grasp_steps
        )
        self._min_tip_dist = torch.minimum(self._min_tip_dist, dist_tips.detach())
        self._max_grasp_align = torch.maximum(self._max_grasp_align, grasp_align.detach())
        # Hysteresis: allow 2-frame soft_grasp flicker without zeroing the streak (PhysX jitter).
        self._grasp_miss_count = torch.where(
            soft_grasp, torch.zeros_like(self._grasp_miss_count), self._grasp_miss_count + 1
        )
        self._grasp_hold_count = torch.where(
            soft_grasp,
            self._grasp_hold_count + 1,
            torch.where(
                self._grasp_miss_count <= 2,
                self._grasp_hold_count,
                torch.zeros_like(self._grasp_hold_count),
            ),
        )
        self._body_contact_count = torch.where(
            on_body, self._body_contact_count + 1, torch.zeros_like(self._body_contact_count)
        )
        self._cup_balance_count = torch.where(
            cup_balance, self._cup_balance_count + 1, torch.zeros_like(self._cup_balance_count)
        )

        # Cache for rewards (dones run before rewards in DirectRLEnv.step)
        self._step_dist = dist
        self._step_dist_ee = dist_ee
        self._step_dist_tips = dist_tips
        self._step_tip_max = tip_max
        self._step_tip_mean = tip_mean
        self._step_closing = closing
        self._step_in_grasp = in_grasp
        self._step_soft_grasp = soft_grasp
        self._step_on_body = on_body
        self._step_cup_balance = cup_balance
        self._step_misaligned_near = misaligned_near
        self._step_grasp_align = grasp_align
        self._step_beyond_tips = beyond_tips
        self._step_tip_platform = tip_platform
        self._step_ball_along = ball_along
        self._step_ee_to_tips_len = ee_to_tips_len
        self._step_tip_spread = tip_spread
        self._step_poke = poke
        self._step_wrap_sym = wrap_sym
        self._step_ball_radial = ball_radial
        self._step_enclosed = enclosed

        # Latch quickly once wrap is real; hold reward + freeze teach sustain.
        cum_ok = self._soft_grasp_steps >= max(int(self.cfg.grasp_hold_steps), 3)
        streak_ok = self._grasp_hold_count >= self.cfg.grasp_hold_steps
        newly_caught = (cum_ok | streak_ok) & soft_grasp & (~self._episode_caught)
        self._just_caught = newly_caught
        self._episode_caught |= newly_caught
        self._body_fail |= self._body_contact_count >= self.cfg.body_fail_steps
        self._body_fail |= self._cup_balance_count >= self.cfg.cup_fail_steps
        return dist, closing, in_grasp, on_body

    def _assist_fade(self) -> float:
        """Assist disabled — always 0 (documented for logs / curriculum compatibility)."""
        return 0.0

    def _apply_catch_assist(
        self,
        tip_center: torch.Tensor,
        ee_pos: torch.Tensor,
        dist_tips: torch.Tensor,
        grasp_align: torch.Tensor,
        closing: torch.Tensor,
        tip_spread: torch.Tensor,
    ) -> torch.Tensor:
        """Disabled: no spring, blend, or finger drive (forbidden catch cheats)."""
        self._ball_assist_force.zero_()
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)
        ball_speed = torch.linalg.norm(ball_vel, dim=-1)

        # Prefer flags computed in _get_dones this step; fall back if needed
        if hasattr(self, "_step_dist"):
            dist = self._step_dist
            dist_ee = self._step_dist_ee
            dist_tips = self._step_dist_tips
            tip_max = self._step_tip_max
            closing = self._step_closing
            soft_grasp = self._step_soft_grasp
            on_body = self._step_on_body
            cup_balance = self._step_cup_balance
            grasp_align = self._step_grasp_align
            beyond_tips = self._step_beyond_tips
            tip_platform = self._step_tip_platform
            tip_spread = self._step_tip_spread
            poke = self._step_poke
            wrap_sym = self._step_wrap_sym
            ball_radial = self._step_ball_radial
            ball_along = self._step_ball_along
        else:
            dist, closing, _, on_body = self._update_grasp_and_body_flags()
            dist_ee = self._step_dist_ee
            dist_tips = self._step_dist_tips
            tip_max = self._step_tip_max
            soft_grasp = self._step_soft_grasp
            cup_balance = self._step_cup_balance
            grasp_align = self._step_grasp_align
            beyond_tips = self._step_beyond_tips
            tip_platform = self._step_tip_platform
            tip_spread = self._step_tip_spread
            poke = self._step_poke
            wrap_sym = self._step_wrap_sym
            ball_radial = self._step_ball_radial
            ball_along = self._step_ball_along

        joint_vel = _as_tensor(self.robot.data.joint_vel)
        arm_speed = torch.linalg.norm(joint_vel[:, self._arm_ids], dim=-1)

        dropped = ball_pos[:, 2] < self.cfg.fall_height_threshold
        self._dropped |= dropped

        hold_scale = float(getattr(self.cfg, "hold_reward_scale", 20.0))
        enc_scale = float(getattr(self.cfg, "enclosure_reward_scale", 14.0))
        tip_max_lim = float(getattr(self.cfg, "success_tip_max", self.cfg.success_tip_dist * 1.15))

        # Dense velocity-match near contact (tip/palm ↔ ball); keep 30-D obs unchanged.
        tip_vel = _as_tensor(self.robot.data.body_lin_vel_w)[:, self._tip_body_ids].mean(dim=1)
        ee_vel = _as_tensor(self.robot.data.body_lin_vel_w)[:, self._ee_body_idx]
        hand_vel = 0.5 * (tip_vel + ee_vel)
        vel_err = torch.linalg.norm(hand_vel - ball_vel, dim=-1)
        vel_match_dist = float(getattr(self.cfg, "vel_match_dist", 0.18))
        vel_near = (dist_tips < vel_match_dist).float() * (1.0 - self._episode_caught.float())
        vel_match_scale = float(getattr(self.cfg, "vel_match_reward_scale", 6.0))
        vel_match_rew = vel_match_scale * vel_near * torch.exp(-1.8 * vel_err)

        reward = compute_rewards(
            dist_ee,
            dist_tips,
            tip_max,
            self._prev_dist,
            ball_speed,
            arm_speed,
            closing,
            grasp_align,
            soft_grasp.float(),
            on_body.float(),
            cup_balance.float(),
            beyond_tips.float(),
            tip_platform.float(),
            poke.float(),
            wrap_sym.float(),
            tip_spread,
            ball_radial,
            ball_along,
            dropped.float(),
            self._just_caught.float(),
            self._episode_caught.float(),
            self.cfg.dist_reward_scale,
            self.cfg.approach_reward_scale,
            self.cfg.catch_reward_scale,
            self.cfg.grasp_reward_scale,
            hold_scale,
            self.cfg.hold_still_reward_scale,
            self.cfg.hold_action_penalty_scale,
            self.cfg.face_ball_reward_scale,
            self.cfg.aperture_reward_scale,
            enc_scale,
            self.cfg.side_miss_penalty,
            self.cfg.wrap_reward_scale,
            self.cfg.poke_penalty,
            self.cfg.early_close_penalty,
            self.cfg.close_reward_scale,
            self.cfg.body_contact_penalty,
            self.cfg.cup_balance_penalty,
            self.cfg.drop_penalty,
            float(getattr(self.cfg, "drop_after_latch_penalty", 120.0)),
            self.cfg.action_penalty_scale,
            self.cfg.success_tip_dist,
            tip_max_lim,
            self.cfg.gripper_near_dist,
            self.cfg.grasp_align_min,
            self.cfg.cup_align_max,
            self.cfg.tip_spread_max,
            self.cfg.grasp_along_min,
            self.cfg.grasp_along_max,
            self.actions,
        )
        reward = reward + vel_match_rew
        # Sparse intercept enclose: only on throw-spawned episodes (not in-hand wrap practice).
        intercept_bonus = float(getattr(self.cfg, "intercept_enclose_bonus", 180.0))
        throw_latch = self._just_caught & (~self._spawned_in_hand)
        reward = reward + intercept_bonus * throw_latch.float()
        # Sparse jackpot: still soft_grasping at timeout without ever dropping.
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        end_bonus = float(getattr(self.cfg, "end_hold_bonus", 400.0))
        reward = reward + end_bonus * (
            time_out & soft_grasp & self._episode_caught & (~self._dropped)
        ).float()
        self._prev_dist = dist.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_grasp_and_body_flags()

        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        dropped = ball_pos[:, 2] < self.cfg.fall_height_threshold
        self._dropped |= dropped

        # Episode ends only when the ball hits the floor, or after episode_length_s.
        # Catch / body-contact / tip-platform do NOT reset — otherwise near-grip looks like
        # a flicker and you never see a real hold.
        terminated = dropped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            log = self.extras.setdefault("log", {})
            # Honest success: sustained soft_grasp latch, never dropped, still holding at end.
            soft_ok = self._episode_caught[env_ids] & (~self._dropped[env_ids])
            if hasattr(self, "_step_soft_grasp"):
                soft_ok = soft_ok & self._step_soft_grasp[env_ids]
            soft_rate = soft_ok.float().mean().item()
            latch_rate = self._episode_caught[env_ids].float().mean().item()
            log["Metrics/soft_grasp"] = soft_rate  # PRIMARY gate metric
            log["Metrics/catch_rate"] = soft_rate  # alias — must match soft_grasp (no fake catch)
            log["Metrics/soft_grasp_latch"] = latch_rate  # latch without requiring end-hold
            log["Metrics/held_no_drop"] = (
                self._episode_caught[env_ids] & (~self._dropped[env_ids])
            ).float().mean().item()
            # Throw-only soft_grasp (excludes in-hand spawn episodes) — Throw phase gate.
            throw_ids = env_ids[~self._spawned_in_hand[env_ids]]
            throw_rate = None
            if len(throw_ids) > 0:
                throw_ok = self._episode_caught[throw_ids] & (~self._dropped[throw_ids])
                if hasattr(self, "_step_soft_grasp"):
                    throw_ok = throw_ok & self._step_soft_grasp[throw_ids]
                throw_rate = throw_ok.float().mean().item()
                log["Metrics/throw_soft_grasp"] = throw_rate
                # Rolling window over individual throw episodes (stable gate signal).
                for ok in throw_ok.detach().cpu().tolist():
                    self._throw_hist.append(1.0 if ok else 0.0)
                if len(self._throw_hist) >= int(getattr(self.cfg, "throw_hist_min", 48)):
                    self._throw_rolling = float(sum(self._throw_hist) / len(self._throw_hist))
            else:
                # Do not alias in-hand success as throw success (misleading for the gate).
                log["Metrics/throw_soft_grasp"] = float("nan")
            log["Metrics/throw_episode_frac"] = float(len(throw_ids)) / float(max(len(env_ids), 1))
            log["Metrics/throw_rolling"] = float(self._throw_rolling)
            log["Metrics/throw_hist_n"] = float(len(self._throw_hist))
            log["Metrics/assist_near_rate"] = 0.0
            log["Metrics/assist_fade"] = 0.0
            log["Metrics/body_fail_rate"] = self._body_fail[env_ids].float().mean().item()
            log["Metrics/drop_rate"] = self._dropped[env_ids].float().mean().item()
            log["Metrics/cup_hold"] = self._cup_balance_count[env_ids].float().mean().item()
            steps = self._episode_steps[env_ids].float().clamp(min=1.0)
            soft_frac = (self._soft_grasp_steps[env_ids].float() / steps).mean().item()
            hold_mean = self._grasp_hold_count[env_ids].float().mean().item()
            log["Metrics/soft_grasp_frac"] = soft_frac
            log["Metrics/hold_streak"] = hold_mean
            log["Metrics/min_tip_dist"] = self._min_tip_dist[env_ids].mean().item()
            log["Metrics/max_grasp_align"] = self._max_grasp_align[env_ids].mean().item()
            # Diagnostics: which soft_grasp gates fire on the last step (debug only).
            if hasattr(self, "_step_soft_grasp"):
                log["Metrics/diag_enclosed"] = self._step_enclosed[env_ids].float().mean().item()
                log["Metrics/diag_wrap"] = self._step_wrap_sym[env_ids].float().mean().item()
                log["Metrics/diag_closing"] = (
                    (self._step_closing[env_ids] > self.cfg.success_close_min).float().mean().item()
                )
                log["Metrics/diag_align"] = (
                    (self._step_grasp_align[env_ids] > self.cfg.grasp_align_min).float().mean().item()
                )
                log["Metrics/diag_near"] = (
                    (self._step_dist_tips[env_ids] < self.cfg.success_tip_dist).float().mean().item()
                )
                log["Metrics/diag_platform"] = self._step_tip_platform[env_ids].float().mean().item()
                log["Metrics/diag_radial"] = (
                    (self._step_ball_radial[env_ids] < self.cfg.grasp_radial_max * 1.35).float().mean().item()
                )
                log["Metrics/diag_along"] = (
                    (
                        (self._step_ball_along[env_ids] > self.cfg.grasp_along_min * 0.5)
                        & (self._step_ball_along[env_ids] < self.cfg.grasp_along_max * 1.5)
                    )
                    .float()
                    .mean()
                    .item()
                )
                # Any-step success proxy for debugging (not the gate metric).
                log["Metrics/soft_grasp_any"] = (self._soft_grasp_steps[env_ids] > 0).float().mean().item()
            self._update_curriculum_gate(
                soft_rate,
                soft_grasp_frac=soft_frac,
                hold_mean=hold_mean,
                throw_soft_grasp=throw_rate,
            )
            log["Metrics/curriculum_cap"] = float(self._curriculum_cap)
            log["Metrics/catch_ema"] = float(self._catch_ema)
            log["Metrics/throw_catch_ema"] = float(self._throw_catch_ema)
            log["Metrics/frac_ema"] = float(self._frac_ema)
            in_hand_p = self._in_hand_spawn_probability()
            log["Metrics/in_hand_spawn_p"] = in_hand_p
            log["Metrics/training_phase"] = {
                "wrap": 0.0,
                "throw": 1.0,
                "throw_a": 0.85,
                "throw_b": 1.0,
                "mixed": 0.5,
            }.get(self._phase(), 0.5)
            if hasattr(self, "_step_tip_platform"):
                log["Metrics/tip_platform"] = self._step_tip_platform[env_ids].float().mean().item()
            if hasattr(self, "_step_dist"):
                log["Metrics/mean_tip_dist"] = self._step_dist[env_ids].mean().item()
            if hasattr(self, "_step_tip_max"):
                log["Metrics/mean_tip_max"] = self._step_tip_max[env_ids].mean().item()
            if hasattr(self, "_step_grasp_align"):
                log["Metrics/mean_grasp_align"] = self._step_grasp_align[env_ids].mean().item()
            if hasattr(self, "_step_tip_spread"):
                log["Metrics/mean_tip_spread"] = self._step_tip_spread[env_ids].mean().item()
            if hasattr(self, "_step_poke"):
                log["Metrics/poke_rate"] = self._step_poke[env_ids].float().mean().item()

        self._episode_caught[env_ids] = False
        self._just_caught[env_ids] = False
        self._body_fail[env_ids] = False
        self._dropped[env_ids] = False
        self._grasp_hold_count[env_ids] = 0
        self._grasp_miss_count[env_ids] = 0
        self._soft_grasp_steps[env_ids] = 0
        self._episode_steps[env_ids] = 0
        self._body_contact_count[env_ids] = 0
        self._cup_balance_count[env_ids] = 0
        self._ball_assist_force[env_ids] = 0.0
        self._min_tip_dist[env_ids] = 10.0
        self._max_grasp_align[env_ids] = 0.0

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

        # Preclose: in-hand uses spawn_in_hand_preclose; drift handoff uses lighter frac.
        in_hand_mask = self._spawned_in_hand[env_ids]
        preclose_frac = float(getattr(self.cfg, "spawn_in_hand_preclose", 0.72))
        drift_band = self._in_drift_mode()
        drift_mask = (~in_hand_mask) & drift_band
        drift_preclose = float(getattr(self.cfg, "throw_drift_preclose", 0.35))
        if in_hand_mask.any() and preclose_frac > 1e-6:
            ih_local = in_hand_mask.nonzero(as_tuple=False).flatten()
            ih_env = env_ids[ih_local]
            preclose = (
                self.cfg.gripper_open_pos * (1.0 - preclose_frac)
                + self.cfg.gripper_close_target * preclose_frac
            )
            joint_pos[ih_local[:, None], self._gripper_ids] = preclose
            self.robot.write_joint_state_to_sim(
                joint_pos[ih_local], torch.zeros_like(joint_pos[ih_local]), None, ih_env
            )
            self.robot_dof_targets[ih_env[:, None], self._gripper_ids] = preclose
        if drift_mask.any() and drift_preclose > 1e-6:
            d_local = drift_mask.nonzero(as_tuple=False).flatten()
            d_env = env_ids[d_local]
            preclose = (
                self.cfg.gripper_open_pos * (1.0 - drift_preclose)
                + self.cfg.gripper_close_target * drift_preclose
            )
            joint_pos[d_local[:, None], self._gripper_ids] = preclose
            self.robot.write_joint_state_to_sim(
                joint_pos[d_local], torch.zeros_like(joint_pos[d_local]), None, d_env
            )
            self.robot_dof_targets[d_env[:, None], self._gripper_ids] = preclose

    def _in_hand_spawn_probability(self) -> float:
        """Forced phase probability, else fade 1→0 by spawn_in_hand_until."""
        forced = getattr(self.cfg, "in_hand_spawn_p", None)
        if forced is not None:
            return float(max(0.0, min(1.0, forced)))
        fade_until = float(getattr(self.cfg, "spawn_in_hand_until", 0.30))
        p = max(0.0, 1.0 - self._curriculum_alpha() / max(fade_until, 1e-6))
        # Throw-like: reserve motion/lob episodes while still in soft handoff.
        if self._is_throw_like():
            if self._in_drift_mode() or self._phase() == "throw_a":
                min_throw = float(getattr(self.cfg, "throw_min_frac", 0.18))
                p = min(p, max(0.0, 1.0 - min_throw))
            elif self._curriculum_alpha() < float(getattr(self.cfg, "throw_drift_until_alpha", 0.40)) + 0.25:
                min_throw = float(getattr(self.cfg, "throw_min_frac", 0.18)) * 0.6
                p = min(p, max(0.0, 1.0 - min_throw))
        return p

    def _curriculum_alpha(self) -> float:
        """Throw hardness in [0,1]. Step schedule capped by catch-performance gate."""
        steps = max(int(self.cfg.curriculum_steps), 1)
        step_alpha = min(float(self.common_step_counter) / float(steps), 1.0)
        return min(step_alpha, float(self._curriculum_cap))

    def _update_curriculum_gate(
        self,
        batch_soft_grasp: float,
        soft_grasp_frac: float | None = None,
        hold_mean: float | None = None,
        throw_soft_grasp: float | None = None,
    ) -> None:
        """Raise throw difficulty while catch metrics look healthy.

        Throw phases advance on rolling throw success (not mixed in-hand soft_grasp).
        """
        alpha_ema = 0.12
        self._catch_ema = (1.0 - alpha_ema) * self._catch_ema + alpha_ema * float(batch_soft_grasp)
        if throw_soft_grasp is not None:
            self._throw_catch_ema = (1.0 - alpha_ema) * self._throw_catch_ema + alpha_ema * float(
                throw_soft_grasp
            )
        if soft_grasp_frac is not None:
            self._frac_ema = (1.0 - alpha_ema) * self._frac_ema + alpha_ema * float(soft_grasp_frac)
        phase = self._phase()
        if phase == "wrap":
            self._curriculum_cap = min(float(self._curriculum_cap), 0.20)
            return
        unlock = float(getattr(self.cfg, "curriculum_unlock_catch", 0.60))
        unlock_frac = float(getattr(self.cfg, "curriculum_unlock_frac", 0.35))
        unlock_hold = float(getattr(self.cfg, "curriculum_unlock_hold_steps", 6))
        rate = float(getattr(self.cfg, "curriculum_advance_rate", 0.004))
        hold_ok = hold_mean is not None and float(hold_mean) >= unlock_hold
        frac_ok = self._frac_ema >= unlock_frac
        hist_min = int(getattr(self.cfg, "throw_hist_min", 48))
        rolling = float(self._throw_rolling) if len(self._throw_hist) >= hist_min else 0.0
        leave_ema = float(getattr(self.cfg, "throw_leave_drift_ema", 0.55))
        streak_need = int(getattr(self.cfg, "throw_gate_streak", 8))

        if self._is_throw_like():
            gate = rolling if len(self._throw_hist) >= hist_min else float(self._throw_catch_ema)
            catch_ok = gate >= unlock
            if gate >= leave_ema:
                self._throw_gate_streak += 1
            else:
                self._throw_gate_streak = 0
            if catch_ok:
                next_cap = min(1.0, self._curriculum_cap + rate)
                if phase == "throw_a":
                    # Throw-A: harden drift only (spawn stays drift forever).
                    self._curriculum_cap = next_cap
                else:
                    drift_until = float(getattr(self.cfg, "throw_drift_until_alpha", 0.40))
                    if self._curriculum_cap < drift_until:
                        self._curriculum_cap = min(drift_until, next_cap)
                    elif self._throw_gate_streak >= streak_need:
                        self._curriculum_cap = next_cap
            elif gate < unlock * 0.55:
                self._curriculum_cap = max(0.03, self._curriculum_cap - rate * 0.6)
            return

        catch_ok = self._catch_ema >= unlock
        gate_ema = self._catch_ema
        if catch_ok or (frac_ok and hold_ok):
            self._curriculum_cap = min(1.0, self._curriculum_cap + rate)
        elif gate_ema < unlock * 0.55 and self._frac_ema < unlock_frac * 0.5:
            self._curriculum_cap = max(0.03, self._curriculum_cap - rate * 0.6)

    def _mix_range(self, easy: tuple[float, float], hard: tuple[float, float], alpha: float) -> tuple[float, float]:
        return (
            easy[0] * (1.0 - alpha) + hard[0] * alpha,
            easy[1] * (1.0 - alpha) + hard[1] * alpha,
        )

    def _launch_ball(self, env_ids: torch.Tensor) -> None:
        """Curriculum: early in-hand wrap practice (fades to 0), then parabolic throws."""
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        alpha = self._curriculum_alpha()
        self.extras.setdefault("log", {})["Metrics/curriculum"] = alpha

        tip_center = self._tip_center_w()[env_ids] - origins
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[env_ids, self._ee_body_idx] - origins
        cup_aim = 0.55 * tip_center + 0.45 * ee_pos

        # Phase Wrap forces ~in-hand; Phase Throw fades in-hand→near-lobs; mixed fades with curriculum.
        in_hand_p = self._in_hand_spawn_probability()
        use_in_hand = torch.rand(n, device=self.device) < in_hand_p
        self._spawned_in_hand[env_ids] = use_in_hand

        pos = torch.zeros((n, 3), device=self.device)
        lin_vel = torch.zeros((n, 3), device=self.device)

        if torch.any(use_in_hand):
            ih = use_in_hand.nonzero(as_tuple=False).flatten()
            jitter = float(getattr(self.cfg, "spawn_in_hand_jitter", 0.012))
            speed = float(getattr(self.cfg, "spawn_in_hand_speed", 0.08))
            # Closer to palm so fingers can wrap; tiny residual speed only.
            along = sample_uniform(0.20, 0.48, (len(ih), 1), device=self.device)
            aperture = ee_pos[ih] + along * (tip_center[ih] - ee_pos[ih])
            aperture += sample_uniform(-jitter, jitter, (len(ih), 3), device=self.device)
            pos[ih] = aperture + origins[ih]
            lin_vel[ih] = sample_uniform(-speed, speed, (len(ih), 3), device=self.device)

        if torch.any(~use_in_hand):
            th = (~use_in_hand).nonzero(as_tuple=False).flatten()
            nt = len(th)
            front = self._mix_range(self.cfg.throw_front_offset_easy, self.cfg.throw_front_offset, alpha)
            side = self._mix_range(self.cfg.throw_side_offset_easy, self.cfg.throw_side_offset, alpha)
            boost = self._mix_range(self.cfg.throw_height_boost_easy, self.cfg.throw_height_boost, alpha)
            flight = self._mix_range(self.cfg.throw_flight_time_easy, self.cfg.throw_flight_time, alpha)
            jit_xy = self.cfg.aim_jitter_xy_easy * (1.0 - alpha) + self.cfg.aim_jitter_xy * alpha
            jit_z = self.cfg.aim_jitter_z_easy * (1.0 - alpha) + self.cfg.aim_jitter_z * alpha

            target = cup_aim[th].clone()
            target[:, 0] += sample_uniform(-jit_xy, jit_xy, (nt,), device=self.device)
            target[:, 1] += sample_uniform(-jit_xy, jit_xy, (nt,), device=self.device)
            target[:, 2] += sample_uniform(-jit_z, jit_z, (nt,), device=self.device)

            # Early throw handoff / Throw-A: in-aperture drift with curriculum-ramped speed/along.
            use_drift = self._in_drift_mode()
            if use_drift:
                along_easy = getattr(self.cfg, "throw_drift_along_easy", (0.55, 0.90))
                along_hard = getattr(self.cfg, "throw_drift_along_hard", (0.30, 0.60))
                speed_easy = getattr(self.cfg, "throw_drift_speed_easy", (0.04, 0.10))
                speed_hard = getattr(self.cfg, "throw_drift_speed_hard", getattr(self.cfg, "throw_drift_speed", (0.08, 0.20)))
                along_r = self._mix_range(along_easy, along_hard, alpha)
                speed_r = self._mix_range(speed_easy, speed_hard, alpha)
                along = sample_uniform(*along_r, (nt, 1), device=self.device)
                aperture = ee_pos[th] + along * (tip_center[th] - ee_pos[th])
                jitter = float(getattr(self.cfg, "spawn_in_hand_jitter", 0.006))
                aperture += sample_uniform(-jitter, jitter, (nt, 3), device=self.device)
                speed = sample_uniform(*speed_r, (nt, 1), device=self.device)
                axis = tip_center[th] - ee_pos[th]
                axis_n = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-4)
                pos[th] = aperture + origins[th]
                lin_vel[th] = -axis_n * speed + sample_uniform(-0.03, 0.03, (nt, 3), device=self.device)
            else:
                release = cup_aim[th].clone()
                release[:, 0] += sample_uniform(*front, (nt,), device=self.device)
                release[:, 1] += sample_uniform(*side, (nt,), device=self.device)
                release[:, 2] -= sample_uniform(*self.cfg.throw_below_offset, (nt,), device=self.device)

                mid = 0.5 * (release + target)
                mid[:, 2] += sample_uniform(*boost, (nt,), device=self.device)

                flight_t = sample_uniform(*flight, (nt, 1), device=self.device)
                g_eff = self.cfg.gravity_full * self.cfg.ball_gravity_scale
                d = target - release
                linear_z = release[:, 2:3] + d[:, 2:3] * 0.5
                loft = 2.0 * (mid[:, 2:3] - linear_z) / flight_t
                g = torch.zeros((nt, 3), device=self.device)
                g[:, 2] = -g_eff
                v = d / flight_t - 0.5 * g * flight_t
                v[:, 2:3] += loft
                pos[th] = release + origins[th]
                lin_vel[th] = v

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
    tip_max: torch.Tensor,
    prev_dist_tips: torch.Tensor,
    ball_speed: torch.Tensor,
    arm_speed: torch.Tensor,
    closing: torch.Tensor,
    grasp_align: torch.Tensor,
    soft_grasp: torch.Tensor,
    on_body: torch.Tensor,
    cup_balance: torch.Tensor,
    beyond_tips: torch.Tensor,
    tip_platform: torch.Tensor,
    poke: torch.Tensor,
    wrap_sym: torch.Tensor,
    tip_spread: torch.Tensor,
    ball_radial: torch.Tensor,
    ball_along: torch.Tensor,
    dropped: torch.Tensor,
    just_caught: torch.Tensor,
    episode_caught: torch.Tensor,
    dist_reward_scale: float,
    approach_reward_scale: float,
    catch_reward_scale: float,
    grasp_reward_scale: float,
    hold_reward_scale: float,
    hold_still_reward_scale: float,
    hold_action_penalty_scale: float,
    face_ball_reward_scale: float,
    aperture_reward_scale: float,
    enclosure_reward_scale: float,
    side_miss_penalty: float,
    wrap_reward_scale: float,
    poke_penalty: float,
    early_close_penalty: float,
    close_reward_scale: float,
    body_contact_penalty: float,
    cup_balance_penalty: float,
    drop_penalty: float,
    drop_after_latch_penalty: float,
    action_penalty_scale: float,
    success_tip_dist: float,
    success_tip_max: float,
    gripper_near_dist: float,
    grasp_align_min: float,
    cup_align_max: float,
    tip_spread_max: float,
    grasp_along_min: float,
    grasp_along_max: float,
    actions: torch.Tensor,
):
    # Stage shaping (impact-aware / DexCatch pattern):
    #   1) face + approach tip center
    #   2) aperture center + fingertip enclosure
    #   3) close wrap → dense soft_grasp → latch bonus → hold without falling
    align_gate = torch.clamp((grasp_align - cup_align_max) / max(grasp_align_min - cup_align_max, 1e-3), 0.0, 1.0)
    near = (dist_tips < gripper_near_dist * 2.0).float()
    # After latch, stop paying approach/enclosure — only hold/drop matter.
    pre_latch = (1.0 - episode_caught)
    tip_rew = 0.12 * near * align_gate * pre_latch * torch.exp(-dist_reward_scale * dist_tips)
    ee_rew = 0.05 * near * align_gate * pre_latch * torch.exp(-dist_reward_scale * dist_ee)
    approach_rew = 0.65 * approach_reward_scale * near * align_gate * pre_latch * torch.clamp(
        prev_dist_tips - dist_tips, -0.02, 0.05
    )
    face_rew = face_ball_reward_scale * pre_latch * torch.clamp(grasp_align, 0.0, 1.0) * torch.exp(-1.4 * dist_tips)

    along_ok = ((ball_along > grasp_along_min) & (ball_along < grasp_along_max)).float()
    aperture_rew = (
        aperture_reward_scale * near * align_gate * along_ok * pre_latch * torch.exp(-16.0 * ball_radial)
    )
    enclosure_rew = (
        enclosure_reward_scale
        * near
        * align_gate
        * wrap_sym
        * pre_latch
        * torch.exp(-10.0 * tip_max)
        * torch.exp(-6.0 * tip_spread)
    )
    side_pen = side_miss_penalty * near * (1.0 - align_gate) * pre_latch * torch.exp(-2.5 * dist_tips)

    closing_clamped = torch.clamp(closing, 0.0, 1.0)
    # Close when tip_center is near + facing — do NOT gate on tip_max (that deadlocks open→enclose).
    close_gate = align_gate * (1.0 - tip_platform) * (1.0 - poke) * (1.0 - episode_caught)
    near_close = (dist_tips < gripper_near_dist).float()
    very_near = (dist_tips < success_tip_dist * 1.2).float()
    close_rew = close_reward_scale * closing_clamped * near_close * close_gate
    close_rew = close_rew + 1.6 * close_reward_scale * closing_clamped * very_near * close_gate
    # Extra payoff for closing while tips are still spreading around the ball.
    close_rew = close_rew + 1.0 * close_reward_scale * closing_clamped * near_close * close_gate * (
        1.0 - torch.clamp(tip_max / max(success_tip_max * 2.0, 1e-3), 0.0, 1.0)
    )

    wrap_rew = wrap_reward_scale * near * align_gate * wrap_sym * pre_latch * torch.exp(-8.0 * tip_spread)
    poke_pen = poke_penalty * poke
    early_close_pen = early_close_penalty * closing_clamped * (dist_tips > gripper_near_dist * 1.5).float() * (
        1.0 - episode_caught
    )

    grasp_rew = grasp_reward_scale * soft_grasp
    success_bonus = catch_reward_scale * just_caught
    # Hold without falling (DexCatch): continuous reward after latch while still wrapped.
    hold_rew = hold_reward_scale * soft_grasp * episode_caught
    hold_still_rew = hold_still_reward_scale * soft_grasp * torch.exp(-1.5 * arm_speed) * torch.exp(
        -0.8 * ball_speed
    )
    # Sparse end-of-episode hold is applied outside the JIT (needs time_out flag).

    body_pen = body_contact_penalty * on_body
    cup_pen = cup_balance_penalty * cup_balance
    platform_pen = 1.0 * cup_balance_penalty * tip_platform
    drop_pen = drop_penalty * dropped
    # Latch-then-drop must be a net loss vs never latching.
    drop_after_latch_pen = drop_after_latch_penalty * dropped * episode_caught
    action_penalty = action_penalty_scale * torch.sum(actions * actions, dim=-1)
    hold_action_pen = hold_action_penalty_scale * episode_caught * soft_grasp * torch.sum(
        actions * actions, dim=-1
    )

    return (
        tip_rew
        + ee_rew
        + approach_rew
        + face_rew
        + aperture_rew
        + enclosure_rew
        + close_rew
        + wrap_rew
        + grasp_rew
        + success_bonus
        + hold_rew
        + hold_still_rew
        - side_pen
        - poke_pen
        - early_close_pen
        - body_pen
        - cup_pen
        - platform_pen
        - drop_pen
        - drop_after_latch_pen
        - action_penalty
        - hold_action_pen
    )
