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
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from XRPlayground.deployment.descriptors import direct_descriptors

from .ball_catch_env_cfg import BallCatchEnvCfg


def _as_tensor(data) -> torch.Tensor:
    """Read articulation/rigid-object buffers across Isaac Lab API variants."""
    return data.torch if hasattr(data, "torch") else data


class BallCatchEnv(DirectRLEnv):
    cfg: BallCatchEnvCfg

    def get_deployment_descriptors(self) -> dict:
        """Return the ordered IO contract used by this environment and Unity."""
        return direct_descriptors("ball_catch.throw", self.cfg, self)

    def __init__(self, cfg: BallCatchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        joint_limits = _as_tensor(self.robot.data.soft_joint_pos_limits)[0]
        self.robot_dof_lower_limits = joint_limits[:, 0].to(self.device)
        self.robot_dof_upper_limits = joint_limits[:, 1].to(self.device)

        self._arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._gripper_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
        tip_joint_ids, _ = self.robot.find_joints(
            ["j2n7s300_joint_finger_tip_[1-3]"], preserve_order=True
        )
        base_joint_ids, _ = self.robot.find_joints(
            ["j2n7s300_joint_finger_[1-3]"], preserve_order=True
        )
        if len(self._arm_ids) != len(self.cfg.arm_joint_names):
            raise RuntimeError(
                f"Expected {len(self.cfg.arm_joint_names)} Kinova arm joints, found {len(self._arm_ids)}."
            )
        if len(self._gripper_ids) < 3:
            raise RuntimeError(f"Expected Kinova finger joints, found {len(self._gripper_ids)}.")
        self._arm_ids = torch.tensor(self._arm_ids, device=self.device, dtype=torch.long)
        self._gripper_ids = torch.tensor(self._gripper_ids, device=self.device, dtype=torch.long)
        self._gripper_tip_joint_ids = torch.tensor(
            tip_joint_ids, device=self.device, dtype=torch.long
        )
        self._gripper_base_joint_ids = torch.tensor(
            base_joint_ids, device=self.device, dtype=torch.long
        )

        default_joint = _as_tensor(self.robot.data.default_joint_pos)[0]
        self._default_arm_pos = default_joint[self._arm_ids].clone()

        self.robot_dof_speed_scales = torch.ones(self.robot.num_joints, device=self.device)
        # The original 1.2 multiplier let one saturated action slam the fingers
        # through a large fraction of their range in a few control frames.
        self.robot_dof_speed_scales[self._gripper_ids] = 0.75

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.left_finger_idx, self.right_finger_idx = self._resolve_gripper_body_indices()
        self._tip_body_ids = self._resolve_finger_tip_body_indices()
        self._arm_body_ids = self._resolve_arm_body_indices()
        self._ee_body_idx = self._resolve_ee_body_index()

        self._episode_caught = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._just_caught = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_dist = torch.ones(self.num_envs, device=self.device)
        self._prev_intercept_dist = torch.ones(self.num_envs, device=self.device)
        self._grasp_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_miss_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._soft_grasp_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cup_balance_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lost_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_drive_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._hold_drive_error_max = torch.zeros(self.num_envs, device=self.device)
        self._hold_closing_sum = torch.zeros(self.num_envs, device=self.device)
        self._post_latch_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._expert_contact_target = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self._expert_contact_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # Gate throw hardness on throw-only rolling success (not mixed in-hand soft_grasp).
        self._catch_ema = 0.0
        self._throw_catch_ema = 0.0
        self._throw_rolling = 0.0
        self._throw_gate_streak = 0
        hist_len = int(getattr(self.cfg, "throw_hist_len", 256))
        self._throw_hist: deque[float] = deque(maxlen=max(hist_len, 32))
        self._lob_gentle_rolling = 0.0
        self._lob_catch_ema = 0.0
        self._lob_hist: deque[float] = deque(maxlen=max(hist_len, 32))
        self._frac_ema = 0.0
        phase = str(getattr(self.cfg, "training_phase", "mixed")).lower()
        # Wrap: stay easy. Throw-A/B: start near-zero hardness.
        self._curriculum_cap = 0.05 if phase in ("throw", "throw_a", "throw_b") else 0.15
        self._min_tip_dist = torch.ones(self.num_envs, device=self.device) * 10.0
        self._max_grasp_align = torch.zeros(self.num_envs, device=self.device)
        # Closest-approach diagnostics distinguish an uncatchable launch from
        # a policy timing failure without relaxing the real grasp gate.
        self._tip_max_at_min = torch.ones(self.num_envs, device=self.device) * 10.0
        self._closing_at_min = torch.zeros(self.num_envs, device=self.device)
        self._speed_at_min = torch.zeros(self.num_envs, device=self.device)
        self._rel_speed_at_min = torch.zeros(self.num_envs, device=self.device)
        self._align_at_min = torch.zeros(self.num_envs, device=self.device)
        self._along_at_min = torch.zeros(self.num_envs, device=self.device)
        self._radial_at_min = torch.ones(self.num_envs, device=self.device) * 10.0
        self._spread_at_min = torch.ones(self.num_envs, device=self.device) * 10.0
        self._loss_along = torch.zeros(self.num_envs, device=self.device)
        self._loss_radial = torch.zeros(self.num_envs, device=self.device)
        self._loss_tip_dist = torch.zeros(self.num_envs, device=self.device)
        self._loss_speed = torch.zeros(self.num_envs, device=self.device)
        self._loss_relative_speed = torch.zeros(self.num_envs, device=self.device)
        # Per-env: last reset used in-hand spawn (for throw-only metrics).
        self._spawned_in_hand = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._spawned_drift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        self.cfg.palm_support_cfg.func(
            "/World/envs/env_0/Robot/j2n7s300_link_7/ball_catch_palm_support",
            self.cfg.palm_support_cfg,
            translation=self.cfg.palm_support_translation,
            orientation=self.cfg.palm_support_orientation,
        )
        for index, (translation, orientation) in enumerate(self.cfg.palm_rim_transforms):
            self.cfg.palm_rim_cfg.func(
                f"/World/envs/env_0/Robot/j2n7s300_link_7/ball_catch_palm_rim_{index}",
                self.cfg.palm_rim_cfg,
                translation=translation,
                orientation=orientation,
            )
        self.ball = RigidObject(self.cfg.ball_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.diagnostic_camera = None
        if bool(getattr(self.cfg, "diagnostic_camera", False)):
            for index in range(3):
                sim_utils.create_prim(f"/World/DiagnosticCameras/Origin_{index}", "Xform")
            self.diagnostic_camera = Camera(
                CameraCfg(
                    prim_path="/World/DiagnosticCameras/Origin_.*/CameraSensor",
                    update_period=0.0,
                    height=480,
                    width=640,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=28.0,
                        focus_distance=0.5,
                        horizontal_aperture=20.955,
                        clipping_range=(0.01, 10.0),
                    ),
                )
            )
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _phase(self) -> str:
        return str(getattr(self.cfg, "training_phase", "mixed")).lower()

    def _is_throw_like(self) -> bool:
        return self._phase() in ("throw", "throw_a", "throw_b")

    def _in_drift_mode(self) -> bool:
        """True while non-in-hand episodes use in-aperture drift (arm usually frozen)."""
        forced = getattr(self.cfg, "force_throw_mode", None)
        if forced is not None:
            mode = str(forced).lower()
            if mode not in ("drift", "lob"):
                raise ValueError(f"force_throw_mode must be 'drift' or 'lob', got {forced!r}")
            return mode == "drift"
        phase = self._phase()
        if phase == "throw_a":
            return True
        if phase in ("throw", "throw_b"):
            drift_until = float(getattr(self.cfg, "throw_drift_until_alpha", 0.40))
            return self._curriculum_alpha() < drift_until
        return False

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Wrap isolates finger closure. Every moving-ball episode leaves the arm
        # policy-driven so randomized offsets can be corrected through positioning.
        # In-hand handoffs stay frozen because their sole purpose is grip retention.
        freeze_arm_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        phase = self._phase()
        if phase == "wrap":
            freeze_arm_only = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        elif phase in ("throw", "throw_a", "throw_b"):
            freeze_arm_only = self._spawned_in_hand

        arm_delta = self.robot_dof_speed_scales[self._arm_ids] * self.dt * self.actions[:, :-1] * self.cfg.action_scale
        # Arm outputs are untrained after the frozen Wrap stage. Release their
        # authority with curriculum difficulty so the first moving episodes do
        # not jerk a valid cup pose away from the ball, while hard lobs retain
        # the full action range needed for interception.
        alpha = self._curriculum_alpha()
        arm_scale_easy = float(getattr(self.cfg, "arm_motion_scale_easy", 0.0))
        arm_scale_hard = float(getattr(self.cfg, "arm_motion_scale_hard", 1.0))
        arm_scale_power = float(getattr(self.cfg, "arm_motion_scale_power", 3.0))
        arm_blend = alpha**arm_scale_power
        arm_motion_scale = arm_scale_easy * (1.0 - arm_blend) + arm_scale_hard * arm_blend
        arm_delta = arm_delta * arm_motion_scale
        grip_delta = (
            self.robot_dof_speed_scales[self._gripper_ids]
            * self.dt
            * self.actions[:, -1:]
            * self.cfg.action_scale
        )
        arm_delta = torch.where(freeze_arm_only.unsqueeze(-1), torch.zeros_like(arm_delta), arm_delta)

        self.robot_dof_targets[:, self._arm_ids] += arm_delta
        self.robot_dof_targets[:, self._gripper_ids] += grip_delta
        base_grip_target = self.robot_dof_targets[:, self._gripper_base_joint_ids].mean(
            dim=-1, keepdim=True
        )
        tip_grip_target = float(self.cfg.gripper_open_pos) + float(
            self.cfg.gripper_tip_target_ratio
        ) * (base_grip_target - float(self.cfg.gripper_open_pos))
        self.robot_dof_targets[:, self._gripper_tip_joint_ids] = tip_grip_target

        joint_pos = _as_tensor(self.robot.data.joint_pos)
        if freeze_arm_only.any():
            # Lock PD targets to current arm configuration so the arm stays put.
            self.robot_dof_targets[:, self._arm_ids] = torch.where(
                freeze_arm_only.unsqueeze(-1),
                joint_pos[:, self._arm_ids],
                self.robot_dof_targets[:, self._arm_ids],
            )
        if freeze_arm_only.any():
            self.actions[:, :-1] = torch.where(
                freeze_arm_only.unsqueeze(-1),
                torch.zeros_like(self.actions[:, :-1]),
                self.actions[:, :-1],
            )

        self.robot_dof_targets[:] = torch.clamp(
            self.robot_dof_targets, self.robot_dof_drive_lower, self.robot_dof_drive_upper
        )
        safe_close = (
            float(self.cfg.gripper_open_pos) * (1.0 - float(self.cfg.gripper_safe_close_fraction))
            + float(self.cfg.gripper_close_target) * float(self.cfg.gripper_safe_close_fraction)
        )
        self.robot_dof_targets[:, self._gripper_base_joint_ids] = torch.clamp(
            self.robot_dof_targets[:, self._gripper_base_joint_ids],
            min=float(self.cfg.gripper_open_pos),
            max=safe_close,
        )
        self.robot_dof_targets[:, self._gripper_tip_joint_ids] = torch.clamp(
            self.robot_dof_targets[:, self._gripper_tip_joint_ids],
            min=float(self.cfg.gripper_open_pos),
            max=float(self.cfg.gripper_tip_safe_close),
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)
        # Buoyancy only — catch assist disabled (no spring / blend / finger drive).
        self.ball.set_external_force_and_torque(self._ball_ext_force, self._ball_ext_torque)

    def _tip_center_w(self) -> torch.Tensor:
        """Virtual world-space center of the physical finger aperture.

        The tip rigid-body origins are joint frames, not distal contact points.
        Their lateral offsets cancel in the arithmetic mean, placing that mean
        only about 1.3 cm from the wrist.  Use the mean solely to recover the
        grasp-axis direction, then project to the actual distal aperture.
        """
        tip_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._tip_body_ids]
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]
        raw_axis = tip_pos.mean(dim=1) - ee_pos
        grasp_axis = raw_axis / torch.linalg.norm(raw_axis, dim=-1, keepdim=True).clamp(min=1e-6)
        return ee_pos + float(self.cfg.grasp_center_distance) * grasp_axis

    def _tip_aperture_center_w(self) -> torch.Tensor:
        """Distal aperture center with a radial three-finger correction.

        The raw tip circumcenter contains a large, invalid axial displacement
        because the link origins sit at different depths.  Preserve the known
        distal depth and apply only its lateral correction.
        """
        tips = _as_tensor(self.robot.data.body_pos_w)[:, self._tip_body_ids]
        a, b, c = tips[:, 0], tips[:, 1], tips[:, 2]
        u = b - a
        v = c - a
        normal = torch.linalg.cross(u, v, dim=-1)
        normal_sq = normal.square().sum(dim=-1, keepdim=True)
        numerator = (
            u.square().sum(dim=-1, keepdim=True) * torch.linalg.cross(v, normal, dim=-1)
            + v.square().sum(dim=-1, keepdim=True) * torch.linalg.cross(normal, u, dim=-1)
        )
        circumcenter = a + numerator / (2.0 * normal_sq.clamp(min=1e-10))
        raw_mean = tips.mean(dim=1)
        valid = (normal_sq.squeeze(-1) > 1e-8) & (
            torch.linalg.norm(circumcenter - raw_mean, dim=-1) < 0.20
        )
        virtual_center = self._tip_center_w()
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[:, self._ee_body_idx]
        grasp_axis = virtual_center - ee_pos
        grasp_axis = grasp_axis / torch.linalg.norm(
            grasp_axis, dim=-1, keepdim=True
        ).clamp(min=1e-6)
        offset = circumcenter - raw_mean
        radial_offset = offset - (offset * grasp_axis).sum(dim=-1, keepdim=True) * grasp_axis
        radial_norm = torch.linalg.norm(radial_offset, dim=-1, keepdim=True)
        radial_offset = radial_offset * torch.clamp(0.04 / radial_norm.clamp(min=1e-6), max=1.0)
        corrected = virtual_center + radial_offset
        return torch.where(valid.unsqueeze(-1), corrected, virtual_center)

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
        tip_vel = _as_tensor(self.robot.data.body_lin_vel_w)[:, self._tip_body_ids].mean(dim=1)
        ee_vel = _as_tensor(self.robot.data.body_lin_vel_w)[:, self._ee_body_idx]
        hand_vel = 0.5 * (tip_vel + ee_vel)
        rel_ball_speed = torch.linalg.norm(ball_vel - hand_vel, dim=-1)
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

        # Soft grasp = real enclosure. Retention remains geometry-based after
        # latch; it is not relaxed into a broad near-hand sphere.
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
            & (dist_tips < self.cfg.success_tip_dist * 1.15)
            & (tip_max < tip_max_lim * 1.15)
            & (closing > self.cfg.success_close_min * 0.80)
            & (ball_speed < self.cfg.success_speed_threshold * 1.25)
            & wrap_sym
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
        closer = dist_tips < self._min_tip_dist
        self._min_tip_dist = torch.minimum(self._min_tip_dist, dist_tips.detach())
        self._tip_max_at_min = torch.where(closer, tip_max.detach(), self._tip_max_at_min)
        self._closing_at_min = torch.where(closer, closing.detach(), self._closing_at_min)
        self._speed_at_min = torch.where(closer, ball_speed.detach(), self._speed_at_min)
        self._rel_speed_at_min = torch.where(closer, rel_ball_speed.detach(), self._rel_speed_at_min)
        self._align_at_min = torch.where(closer, grasp_align.detach(), self._align_at_min)
        self._along_at_min = torch.where(closer, ball_along.detach(), self._along_at_min)
        self._radial_at_min = torch.where(closer, ball_radial.detach(), self._radial_at_min)
        self._spread_at_min = torch.where(closer, tip_spread.detach(), self._spread_at_min)
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

        # Require sustained enclosure. A brief brush never switches the task
        # into the retained-catch state.
        streak_ok = self._grasp_hold_count >= self.cfg.grasp_hold_steps
        was_caught = self._episode_caught.clone()
        newly_caught = streak_ok & soft_grasp & (~was_caught)
        self._just_caught = newly_caught
        self._episode_caught |= newly_caught
        lost_grasp = was_caught & (self._grasp_miss_count >= int(self.cfg.grasp_loss_steps))
        self._lost_grasp |= lost_grasp
        self._loss_along = torch.where(lost_grasp, ball_along.detach(), self._loss_along)
        self._loss_radial = torch.where(lost_grasp, ball_radial.detach(), self._loss_radial)
        self._loss_tip_dist = torch.where(lost_grasp, dist_tips.detach(), self._loss_tip_dist)
        self._loss_speed = torch.where(lost_grasp, ball_speed.detach(), self._loss_speed)
        self._loss_relative_speed = torch.where(
            lost_grasp, rel_ball_speed.detach(), self._loss_relative_speed
        )

        # Positive commanded-vs-actual finger error approximates drive pressure.
        # Track it only after latch so the quick closing transient is not
        # mistaken for an excessively forceful hold.
        joint_pos = _as_tensor(self.robot.data.joint_pos)
        grip_actual = joint_pos[:, self._gripper_ids].mean(dim=-1)
        grip_target = self.robot_dof_targets[:, self._gripper_ids].mean(dim=-1)
        grip_drive_error = torch.clamp(grip_target - grip_actual, min=0.0)
        post_latch = self._episode_caught & (~self._lost_grasp)
        self._post_latch_steps += post_latch.long()
        self._hold_drive_error_sum += torch.where(
            post_latch, grip_drive_error, torch.zeros_like(grip_drive_error)
        )
        self._hold_drive_error_max = torch.where(
            post_latch,
            torch.maximum(self._hold_drive_error_max, grip_drive_error),
            self._hold_drive_error_max,
        )
        self._hold_closing_sum += torch.where(post_latch, closing, torch.zeros_like(closing))
        self._step_grip_drive_error = grip_drive_error
        self._step_lost_grasp = lost_grasp
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

    def expert_catch_actions(self) -> torch.Tensor:
        """Return the observation-conditioned catch teacher used for imitation.

        The returned tensor is a training target only. Calling this method
        does not apply an action, move the ball, or alter simulator state.
        """
        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        ball_vel = _as_tensor(self.ball.data.root_lin_vel_w)
        tip_center, _, closing, _ = self._gripper_state()
        tip_center = tip_center - self.scene.env_origins
        to_tips = ball_pos - tip_center
        body_vel = _as_tensor(self.robot.data.body_lin_vel_w)
        tip_vel = body_vel[:, self._tip_body_ids].mean(dim=1)
        ee_vel = body_vel[:, self._ee_body_idx]
        hand_vel = 0.5 * (tip_vel + ee_vel)

        teacher_actions = torch.zeros_like(self.actions)
        # Search closest approach on the actual ballistic path. A linear
        # p/v estimate closes far too late while a longer lob is still rising.
        # Sixty-five samples give <= 7.8 ms timing error over the 1 s horizon,
        # below one 60 Hz policy frame after nearest-sample rounding.
        sample_times = torch.linspace(0.0, 1.0, 65, device=self.device, dtype=ball_vel.dtype)
        ballistic_offsets = to_tips.unsqueeze(1) + ball_vel.unsqueeze(1) * sample_times.view(1, -1, 1)
        gravity_drop = 0.5 * float(
            self.cfg.gravity_full * self.cfg.ball_gravity_scale
        ) * sample_times.square()
        ballistic_offsets[:, :, 2] -= gravity_drop.unsqueeze(0)
        closest_index = ballistic_offsets.square().sum(dim=-1).argmin(dim=-1)
        teacher_ttc = sample_times[closest_index]
        teacher_closest = ballistic_offsets[
            torch.arange(self.num_envs, device=self.device), closest_index
        ]
        teacher_approaching = closest_index > 0
        position_velocity = float(self.cfg.expert_position_gain) * teacher_closest
        position_speed = torch.linalg.norm(position_velocity, dim=-1, keepdim=True)
        position_velocity = position_velocity * torch.clamp(
            float(self.cfg.expert_position_velocity_max) / position_speed.clamp(min=1e-6),
            max=1.0,
        )
        position_gate = teacher_approaching & (
            teacher_ttc <= float(self.cfg.expert_position_horizon)
        )
        position_velocity = position_velocity * position_gate.unsqueeze(-1)

        cushion_lead = max(float(self.cfg.expert_cushion_lead_time), 0.04)
        cushion_phase = torch.clamp(
            (cushion_lead - teacher_ttc) / cushion_lead, 0.0, 1.0
        )
        velocity_profile = 1.5 * cushion_phase.square() - 0.5 * cushion_phase
        velocity_profile = velocity_profile * teacher_approaching.float()
        desired_hand_velocity = position_velocity + (
            float(self.cfg.expert_cushion_velocity_gain)
            * velocity_profile.unsqueeze(-1)
            * ball_vel
        )

        post_ball_velocity = ball_vel.clone()
        post_ball_velocity[:, 2] = torch.clamp(post_ball_velocity[:, 2], min=0.0)
        settle = torch.exp(
            -self._post_latch_steps.float()
            / max(float(self.cfg.expert_settle_steps), 1.0)
        )
        desired_hand_velocity = torch.where(
            self._episode_caught.unsqueeze(-1),
            settle.unsqueeze(-1) * post_ball_velocity,
            desired_hand_velocity,
        )

        jacobian_data = self.robot.data.body_link_jacobian_w
        jacobians = jacobian_data.torch if hasattr(jacobian_data, "torch") else jacobian_data
        ee_jacobian_idx = self._ee_body_idx - 1 if self.robot.is_fixed_base else self._ee_body_idx
        jacobian = jacobians[:, ee_jacobian_idx, :3, :]
        jacobian = jacobian[:, :, self._arm_ids]
        damping = max(float(self.cfg.expert_cushion_damping), 1e-4)
        identity = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
        task_matrix = jacobian @ jacobian.transpose(1, 2) + damping * damping * identity
        teacher_joint_velocity = jacobian.transpose(1, 2) @ torch.linalg.solve(
            task_matrix, desired_hand_velocity.unsqueeze(-1)
        )
        teacher_joint_velocity = teacher_joint_velocity.squeeze(-1)
        alpha = float(self._curriculum_alpha())
        arm_blend = alpha ** float(self.cfg.arm_motion_scale_power)
        arm_motion_scale = (
            float(self.cfg.arm_motion_scale_easy) * (1.0 - arm_blend)
            + float(self.cfg.arm_motion_scale_hard) * arm_blend
        )
        teacher_actions[:, :-1] = torch.clamp(
            teacher_joint_velocity
            / max(float(self.cfg.action_scale) * arm_motion_scale, 1e-4),
            -1.0,
            1.0,
        )

        teacher_lead = max(float(self.cfg.expert_grip_lead_time), 0.04)
        teacher_target_close = float(self.cfg.expert_grip_target_close)
        teacher_desired_close = teacher_target_close * torch.clamp(
            (teacher_lead - teacher_ttc)
            / max(teacher_lead - float(self.cfg.expert_grip_end_time), 1e-3),
            0.0,
            1.0,
        )
        teacher_near = torch.linalg.norm(to_tips, dim=-1) < 0.13
        teacher_desired_close = torch.where(
            teacher_approaching | teacher_near,
            teacher_desired_close,
            torch.zeros_like(teacher_desired_close),
        )
        teacher_desired_close = torch.where(
            teacher_near,
            torch.full_like(teacher_desired_close, teacher_target_close),
            teacher_desired_close,
        )
        teacher_grip_action = torch.clamp(
            8.0 * (teacher_desired_close - closing), -1.0, 1.0
        )
        joint_pos = _as_tensor(self.robot.data.joint_pos)
        grip_actual_raw = joint_pos[:, self._gripper_ids].mean(dim=-1)
        grip_target_raw = self.robot_dof_targets[:, self._gripper_ids].mean(dim=-1)
        relative_speed = torch.linalg.norm(ball_vel - hand_vel, dim=-1)
        final_hold_raw = float(self.cfg.gripper_open_pos) + float(
            self.cfg.expert_final_hold_close
        ) * float(self.cfg.gripper_close_target - self.cfg.gripper_open_pos)
        soft_grasp = getattr(
            self, "_step_soft_grasp", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        first_contact = soft_grasp & torch.isnan(self._expert_contact_target)
        self._expert_contact_target = torch.where(
            first_contact, grip_actual_raw, self._expert_contact_target
        )
        self._expert_contact_step = torch.where(
            first_contact, self._episode_steps, self._expert_contact_step
        )
        contact_age = torch.clamp(
            self._episode_steps - self._expert_contact_step, min=0
        ).float()
        hold_blend = torch.clamp(
            contact_age / max(float(self.cfg.expert_hold_ramp_steps), 1.0),
            0.0,
            1.0,
        )
        unload_target = self._expert_contact_target * (1.0 - hold_blend)
        unload_target = unload_target + final_hold_raw * hold_blend
        target_delta_per_action = 0.75 * float(self.dt) * float(self.cfg.action_scale)
        unload_action = torch.clamp(
            (unload_target - grip_target_raw) / max(target_delta_per_action, 1e-4),
            -1.0,
            1.0,
        )
        contact_or_latched = (~torch.isnan(self._expert_contact_target)) | self._episode_caught
        teacher_actions[:, -1] = torch.where(
            contact_or_latched, unload_action, teacher_grip_action
        )
        return teacher_actions

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

        floor_dropped = ball_pos[:, 2] < self.cfg.fall_height_threshold
        dropped = floor_dropped | self._lost_grasp
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
        vel_match_rew = vel_match_rew - (
            float(getattr(self.cfg, "vel_match_error_penalty_scale", 0.0))
            * vel_near
            * torch.clamp(vel_err - 0.15, min=0.0, max=2.0)
        )

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
        # Moving episodes need a direct arm-positioning signal before contact.
        # Track a short-horizon ballistic intercept point; this is reward
        # shaping only (no force, pose blend, IK, or scripted catch action).
        intercept_horizon = float(getattr(self.cfg, "intercept_prediction_horizon", 0.20))
        future_ball = ball_pos + ball_vel * intercept_horizon
        future_ball[:, 2] -= (
            0.5
            * float(self.cfg.gravity_full * self.cfg.ball_gravity_scale)
            * intercept_horizon
            * intercept_horizon
        )
        tip_center = self._tip_center_w() - self.scene.env_origins
        intercept_dist = torch.linalg.norm(future_ball - tip_center, dim=-1)
        intercept_progress = torch.clamp(self._prev_intercept_dist - intercept_dist, -0.02, 0.02)
        intercept_progress = torch.where(
            self._episode_steps > 1, intercept_progress, torch.zeros_like(intercept_progress)
        )
        moving_pre_latch = ((~self._spawned_in_hand) & (~self._episode_caught)).float()
        reward = reward + moving_pre_latch * (
            float(self.cfg.intercept_tracking_reward_scale) * torch.exp(-9.0 * intercept_dist)
            + float(self.cfg.intercept_progress_reward_scale) * intercept_progress
        )
        # Teach the policy the same quick-but-gentle timing required at
        # deployment. This is computed only from observed relative motion and
        # contributes reward; it never overwrites a gripper action.
        to_tips = ball_pos - tip_center
        rel_vel = ball_vel - tip_vel
        closing_rate = (to_tips * rel_vel).sum(dim=-1)
        time_to_contact = torch.clamp(
            -closing_rate / rel_vel.square().sum(dim=-1).clamp(min=1e-4),
            min=0.0,
            max=1.0,
        )
        closest_offset = to_tips + rel_vel * time_to_contact.unsqueeze(-1)
        closest_dist = torch.linalg.norm(closest_offset, dim=-1)
        lead = float(self.cfg.grip_timing_lead_time)
        end = float(self.cfg.grip_timing_end_time)
        desired_close = float(self.cfg.grip_timing_target) * torch.clamp(
            (lead - time_to_contact) / max(lead - end, 1e-3), 0.0, 1.0
        )
        timing_error = closing - desired_close
        timing_gate = (
            moving_pre_latch
            * (closing_rate < 0.0).float()
            * torch.exp(-18.0 * closest_dist)
        )
        reward = reward + timing_gate * (
            float(self.cfg.grip_timing_reward_scale) * torch.exp(-14.0 * timing_error.square())
            - float(self.cfg.grip_timing_error_penalty_scale) * timing_error.square()
        )
        # Give PPO an immediate, action-space signal for the catch motion that
        # has been validated against the real contact dynamics.  This is
        # teacher-shaped reward only: the policy action above remains the sole
        # command applied to the robot, and no ball force or pose is modified.
        expert_arm_scale = float(getattr(self.cfg, "expert_arm_action_penalty_scale", 0.0))
        expert_grip_scale = float(getattr(self.cfg, "expert_grip_action_penalty_scale", 0.0))
        if expert_arm_scale > 0.0 or expert_grip_scale > 0.0:
            teacher_actions = torch.zeros_like(self.actions)

            # Predict where the current ballistic path passes the cup and make
            # a bounded lateral/vertical correction.  In the final lead window,
            # match ball velocity smoothly so the palm yields instead of
            # stopping the ball abruptly.
            teacher_closing_rate = (to_tips * ball_vel).sum(dim=-1)
            teacher_ttc = torch.clamp(
                -teacher_closing_rate / ball_vel.square().sum(dim=-1).clamp(min=1e-4),
                min=0.0,
                max=1.0,
            )
            teacher_approaching = teacher_closing_rate < 0.0
            teacher_closest = to_tips + ball_vel * teacher_ttc.unsqueeze(-1)
            position_velocity = float(self.cfg.expert_position_gain) * teacher_closest
            position_speed = torch.linalg.norm(position_velocity, dim=-1, keepdim=True)
            position_velocity = position_velocity * torch.clamp(
                float(self.cfg.expert_position_velocity_max) / position_speed.clamp(min=1e-6),
                max=1.0,
            )
            position_gate = teacher_approaching & (
                teacher_ttc <= float(self.cfg.expert_position_horizon)
            )
            position_velocity = position_velocity * position_gate.unsqueeze(-1)

            cushion_lead = max(float(self.cfg.expert_cushion_lead_time), 0.04)
            cushion_phase = torch.clamp(
                (cushion_lead - teacher_ttc) / cushion_lead, 0.0, 1.0
            )
            velocity_profile = 1.5 * cushion_phase.square() - 0.5 * cushion_phase
            velocity_profile = velocity_profile * teacher_approaching.float()
            desired_hand_velocity = position_velocity + (
                float(self.cfg.expert_cushion_velocity_gain)
                * velocity_profile.unsqueeze(-1)
                * ball_vel
            )

            # Once enclosed, follow the shared upward/horizontal velocity and
            # bleed it away.  Never command further downward following while
            # holding, which keeps the retained ball above the drop plane.
            post_ball_velocity = ball_vel.clone()
            post_ball_velocity[:, 2] = torch.clamp(post_ball_velocity[:, 2], min=0.0)
            settle = torch.exp(
                -self._post_latch_steps.float()
                / max(float(self.cfg.expert_settle_steps), 1.0)
            )
            desired_hand_velocity = torch.where(
                self._episode_caught.unsqueeze(-1),
                settle.unsqueeze(-1) * post_ball_velocity,
                desired_hand_velocity,
            )

            jacobian_data = self.robot.data.body_link_jacobian_w
            jacobians = jacobian_data.torch if hasattr(jacobian_data, "torch") else jacobian_data
            ee_jacobian_idx = self._ee_body_idx - 1 if self.robot.is_fixed_base else self._ee_body_idx
            jacobian = jacobians[:, ee_jacobian_idx, :3, :]
            jacobian = jacobian[:, :, self._arm_ids]
            damping = max(float(self.cfg.expert_cushion_damping), 1e-4)
            identity = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
            task_matrix = jacobian @ jacobian.transpose(1, 2) + damping * damping * identity
            teacher_joint_velocity = jacobian.transpose(1, 2) @ torch.linalg.solve(
                task_matrix, desired_hand_velocity.unsqueeze(-1)
            )
            teacher_joint_velocity = teacher_joint_velocity.squeeze(-1)
            alpha = float(self._curriculum_alpha())
            arm_blend = alpha ** float(self.cfg.arm_motion_scale_power)
            arm_motion_scale = (
                float(self.cfg.arm_motion_scale_easy) * (1.0 - arm_blend)
                + float(self.cfg.arm_motion_scale_hard) * arm_blend
            )
            teacher_actions[:, :-1] = torch.clamp(
                teacher_joint_velocity
                / max(float(self.cfg.action_scale) * arm_motion_scale, 1e-4),
                -1.0,
                1.0,
            )

            # Close quickly just before closest approach, then unload the
            # integrated finger target at first enclosure.  Once relative
            # motion is small, converge to the experimentally stable hold band.
            teacher_lead = max(float(self.cfg.expert_grip_lead_time), 0.04)
            teacher_target_close = float(self.cfg.expert_grip_target_close)
            teacher_desired_close = teacher_target_close * torch.clamp(
                (teacher_lead - teacher_ttc) / max(teacher_lead - 0.035, 1e-3),
                0.0,
                1.0,
            )
            teacher_near = torch.linalg.norm(to_tips, dim=-1) < 0.13
            teacher_desired_close = torch.where(
                teacher_approaching | teacher_near,
                teacher_desired_close,
                torch.zeros_like(teacher_desired_close),
            )
            teacher_desired_close = torch.where(
                teacher_near,
                torch.full_like(teacher_desired_close, teacher_target_close),
                teacher_desired_close,
            )
            teacher_grip_action = torch.clamp(
                8.0 * (teacher_desired_close - closing), -1.0, 1.0
            )
            joint_pos = _as_tensor(self.robot.data.joint_pos)
            grip_actual_raw = joint_pos[:, self._gripper_ids].mean(dim=-1)
            grip_target_raw = self.robot_dof_targets[:, self._gripper_ids].mean(dim=-1)
            relative_speed = torch.linalg.norm(ball_vel - hand_vel, dim=-1)
            final_hold_raw = float(self.cfg.gripper_open_pos) + float(
                self.cfg.expert_final_hold_close
            ) * float(self.cfg.gripper_close_target - self.cfg.gripper_open_pos)
            unload_target = torch.where(
                relative_speed < float(self.cfg.expert_firm_hold_below_speed),
                torch.full_like(grip_actual_raw, final_hold_raw),
                grip_actual_raw,
            )
            target_delta_per_action = (
                0.75 * float(self.dt) * float(self.cfg.action_scale)
            )
            unload_action = torch.clamp(
                (unload_target - grip_target_raw) / max(target_delta_per_action, 1e-4),
                -1.0,
                1.0,
            )
            contact_or_latched = soft_grasp | self._episode_caught
            teacher_actions[:, -1] = torch.where(
                contact_or_latched, unload_action, teacher_grip_action
            )

            # Keep PPO's shaping target identical to the public ballistic
            # teacher used for behavior cloning and deterministic diagnostics.
            teacher_actions = self.expert_catch_actions()

            throw_gate = ((~self._spawned_in_hand) & (~dropped)).float()
            arm_error = (self.actions[:, :-1] - teacher_actions[:, :-1]).square().mean(dim=-1)
            grip_error = (self.actions[:, -1] - teacher_actions[:, -1]).square()
            reward = reward - throw_gate * (
                expert_arm_scale * arm_error + expert_grip_scale * grip_error
            )
        self._prev_intercept_dist = intercept_dist.detach()
        # Retention must be gentle as well as geometrically valid. Drive error
        # captures continued squeezing against contact, while a positive close
        # action after latch catches the policy behavior that would keep
        # integrating toward a hard pinch in deployment.
        grip_drive_error = getattr(
            self, "_step_grip_drive_error", torch.zeros(self.num_envs, device=self.device)
        )
        grip_range = max(float(self.cfg.gripper_close_target - self.cfg.gripper_open_pos), 1e-4)
        drive_excess = torch.clamp(
            grip_drive_error - float(self.cfg.gentle_drive_error_tolerance), min=0.0
        ) / grip_range
        close_target = float(self.cfg.gentle_close_target)
        close_min = float(self.cfg.gentle_close_min)
        underclose = torch.clamp(close_min - closing, min=0.0)
        overclose = torch.clamp(closing - float(self.cfg.gentle_close_max), min=0.0)
        post_latch = self._episode_caught.float()
        gentle_quality = (
            torch.exp(-6.0 * grip_drive_error / grip_range)
            * torch.exp(-8.0 * underclose)
            * torch.exp(-8.0 * overclose)
        )
        reward = reward + (
            float(self.cfg.gentle_hold_reward_scale)
            * soft_grasp.float()
            * post_latch
            * gentle_quality
        )
        reward = reward - post_latch * (
            float(self.cfg.gentle_drive_error_penalty_scale) * drive_excess
            + float(self.cfg.gentle_underclose_penalty_scale) * underclose
            + float(self.cfg.gentle_overclose_penalty_scale) * overclose
            + float(self.cfg.post_latch_close_action_penalty_scale)
            * torch.clamp(self.actions[:, -1], min=0.0)
            * torch.clamp((closing - close_target) / 0.10, min=0.0, max=1.0)
        )
        # Sparse intercept enclose: only on throw-spawned episodes (not in-hand wrap practice).
        intercept_bonus = float(getattr(self.cfg, "intercept_enclose_bonus", 180.0))
        throw_latch = self._just_caught & (~self._spawned_in_hand)
        reward = reward + intercept_bonus * throw_latch.float()
        # Sparse jackpot: still gently grasping at timeout without ever dropping.
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        end_bonus = float(getattr(self.cfg, "end_hold_bonus", 400.0))
        gentle_now = (
            (grip_drive_error <= float(self.cfg.gentle_drive_error_max))
            & (closing >= float(self.cfg.gentle_close_min))
            & (closing <= float(self.cfg.gentle_close_max) + 0.08)
            & (~self._lost_grasp)
        )
        reward = reward + end_bonus * (
            time_out & soft_grasp & gentle_now & self._episode_caught & (~self._dropped)
        ).float()
        self._prev_dist = dist.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_grasp_and_body_flags()

        ball_pos = _as_tensor(self.ball.data.root_pos_w) - self.scene.env_origins
        floor_dropped = ball_pos[:, 2] < self.cfg.fall_height_threshold
        dropped = floor_dropped | self._lost_grasp
        self._dropped |= dropped

        # Default: episode ends only on floor or timeout (never on brief latch flicker).
        terminated = dropped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        reset_env_ids = env_ids
        completed_env_ids = env_ids[self._episode_steps[env_ids] > 0]
        if len(completed_env_ids) > 0:
            env_ids = completed_env_ids
            log = self.extras.setdefault("log", {})
            # Honest success: sustained soft_grasp latch, never dropped, still holding at end.
            soft_ok = self._episode_caught[env_ids] & (~self._dropped[env_ids])
            if hasattr(self, "_step_soft_grasp"):
                soft_ok = soft_ok & self._step_soft_grasp[env_ids]
            soft_rate = soft_ok.float().mean().item()
            hold_steps = self._post_latch_steps[env_ids].float().clamp(min=1.0)
            mean_drive_error = self._hold_drive_error_sum[env_ids] / hold_steps
            mean_hold_closing = self._hold_closing_sum[env_ids] / hold_steps
            gentle_ok = (
                soft_ok
                & (self._post_latch_steps[env_ids] > 0)
                & (mean_drive_error <= float(self.cfg.gentle_drive_error_max))
                & (self._hold_drive_error_max[env_ids] <= float(self.cfg.gentle_drive_error_max) * 2.0)
                & (mean_hold_closing >= float(self.cfg.gentle_close_min))
                & (mean_hold_closing <= float(self.cfg.gentle_close_max) + 0.08)
            )
            gentle_rate = gentle_ok.float().mean().item()
            latch_rate = self._episode_caught[env_ids].float().mean().item()
            log["Metrics/soft_grasp"] = soft_rate
            log["Metrics/gentle_grasp"] = gentle_rate  # PRIMARY gate metric
            log["Metrics/catch_rate"] = gentle_rate
            log["Metrics/retained_grasp_rate"] = soft_rate
            log["Metrics/soft_grasp_latch"] = latch_rate  # latch without requiring end-hold
            log["Metrics/held_no_drop"] = (
                self._episode_caught[env_ids] & (~self._dropped[env_ids])
            ).float().mean().item()
            log["Metrics/grasp_loss_rate"] = self._lost_grasp[env_ids].float().mean().item()
            log["Metrics/mean_hold_drive_error"] = mean_drive_error.mean().item()
            log["Metrics/max_hold_drive_error"] = self._hold_drive_error_max[env_ids].mean().item()
            log["Metrics/mean_hold_closing"] = mean_hold_closing.mean().item()
            # Motion-only metrics exclude in-hand spawns. Curriculum and
            # promotion use gentle retention, not a transient geometric latch.
            gentle_by_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            gentle_by_env[env_ids] = gentle_ok
            throw_ids = env_ids[~self._spawned_in_hand[env_ids]]
            throw_rate = None
            if len(throw_ids) > 0:
                throw_ok = self._episode_caught[throw_ids] & (~self._dropped[throw_ids])
                if hasattr(self, "_step_soft_grasp"):
                    throw_ok = throw_ok & self._step_soft_grasp[throw_ids]
                throw_soft_rate = throw_ok.float().mean().item()
                throw_gentle = gentle_by_env[throw_ids]
                throw_rate = throw_gentle.float().mean().item()
                log["Metrics/throw_soft_grasp"] = throw_soft_rate
                log["Metrics/throw_gentle_grasp"] = throw_rate
                # Rolling window over individual throw episodes (stable gate signal).
                for ok in throw_gentle.detach().cpu().tolist():
                    self._throw_hist.append(1.0 if ok else 0.0)
                if len(self._throw_hist) >= int(getattr(self.cfg, "throw_hist_min", 48)):
                    self._throw_rolling = float(sum(self._throw_hist) / len(self._throw_hist))
            else:
                # Do not alias in-hand success as throw success (misleading for the gate).
                log["Metrics/throw_soft_grasp"] = float("nan")
                log["Metrics/throw_gentle_grasp"] = float("nan")

            lob_ids = throw_ids[~self._spawned_drift[throw_ids]]
            lob_rate = None
            if len(lob_ids) > 0:
                lob_gentle = gentle_by_env[lob_ids]
                lob_rate = lob_gentle.float().mean().item()
                log["Metrics/lob_gentle_grasp"] = lob_rate
                for ok in lob_gentle.detach().cpu().tolist():
                    self._lob_hist.append(1.0 if ok else 0.0)
                if len(self._lob_hist) >= int(getattr(self.cfg, "throw_hist_min", 48)):
                    self._lob_gentle_rolling = float(sum(self._lob_hist) / len(self._lob_hist))
            else:
                log["Metrics/lob_gentle_grasp"] = float("nan")
            log["Metrics/throw_episode_frac"] = float(len(throw_ids)) / float(max(len(env_ids), 1))
            log["Metrics/lob_episode_frac"] = float(len(lob_ids)) / float(max(len(env_ids), 1))
            log["Metrics/throw_rolling"] = float(self._throw_rolling)
            log["Metrics/throw_gentle_rolling"] = float(self._throw_rolling)
            log["Metrics/throw_hist_n"] = float(len(self._throw_hist))
            log["Metrics/lob_gentle_rolling"] = float(self._lob_gentle_rolling)
            log["Metrics/lob_hist_n"] = float(len(self._lob_hist))
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
            log["Metrics/closest_tip_max"] = self._tip_max_at_min[env_ids].mean().item()
            log["Metrics/closest_closing"] = self._closing_at_min[env_ids].mean().item()
            log["Metrics/closest_ball_speed"] = self._speed_at_min[env_ids].mean().item()
            log["Metrics/closest_relative_speed"] = self._rel_speed_at_min[env_ids].mean().item()
            log["Metrics/closest_align"] = self._align_at_min[env_ids].mean().item()
            log["Metrics/closest_along"] = self._along_at_min[env_ids].mean().item()
            log["Metrics/closest_radial"] = self._radial_at_min[env_ids].mean().item()
            log["Metrics/closest_tip_spread"] = self._spread_at_min[env_ids].mean().item()
            log["Metrics/loss_along"] = self._loss_along[env_ids].mean().item()
            log["Metrics/loss_radial"] = self._loss_radial[env_ids].mean().item()
            log["Metrics/loss_tip_dist"] = self._loss_tip_dist[env_ids].mean().item()
            log["Metrics/loss_ball_speed"] = self._loss_speed[env_ids].mean().item()
            log["Metrics/loss_relative_speed"] = self._loss_relative_speed[env_ids].mean().item()
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
                gentle_rate,
                soft_grasp_frac=soft_frac,
                hold_mean=hold_mean,
                throw_soft_grasp=throw_rate,
                lob_gentle_grasp=lob_rate,
            )
            log["Metrics/curriculum_cap"] = float(self._curriculum_cap)
            log["Metrics/catch_ema"] = float(self._catch_ema)
            log["Metrics/throw_catch_ema"] = float(self._throw_catch_ema)
            log["Metrics/lob_catch_ema"] = float(self._lob_catch_ema)
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

        env_ids = reset_env_ids
        self._episode_caught[env_ids] = False
        self._just_caught[env_ids] = False
        self._body_fail[env_ids] = False
        self._dropped[env_ids] = False
        self._lost_grasp[env_ids] = False
        self._grasp_hold_count[env_ids] = 0
        self._grasp_miss_count[env_ids] = 0
        self._soft_grasp_steps[env_ids] = 0
        self._episode_steps[env_ids] = 0
        self._body_contact_count[env_ids] = 0
        self._cup_balance_count[env_ids] = 0
        self._hold_drive_error_sum[env_ids] = 0.0
        self._hold_drive_error_max[env_ids] = 0.0
        self._hold_closing_sum[env_ids] = 0.0
        self._post_latch_steps[env_ids] = 0
        self._expert_contact_target[env_ids] = float("nan")
        self._expert_contact_step[env_ids] = -1
        self._ball_assist_force[env_ids] = 0.0
        self._min_tip_dist[env_ids] = 10.0
        self._max_grasp_align[env_ids] = 0.0
        self._tip_max_at_min[env_ids] = 10.0
        self._closing_at_min[env_ids] = 0.0
        self._speed_at_min[env_ids] = 0.0
        self._rel_speed_at_min[env_ids] = 0.0
        self._align_at_min[env_ids] = 0.0
        self._along_at_min[env_ids] = 0.0
        self._radial_at_min[env_ids] = 10.0
        self._spread_at_min[env_ids] = 10.0
        self._loss_along[env_ids] = 0.0
        self._loss_radial[env_ids] = 0.0
        self._loss_tip_dist[env_ids] = 0.0
        self._loss_speed[env_ids] = 0.0
        self._loss_relative_speed[env_ids] = 0.0

        super()._reset_idx(env_ids)

        joint_pos = _as_tensor(self.robot.data.default_joint_pos)[env_ids].clone()
        arm_offsets = torch.tensor(
            self.cfg.reset_arm_joint_offsets, device=self.device, dtype=joint_pos.dtype
        )
        joint_pos[:, self._arm_ids] += arm_offsets
        joint_vel = torch.zeros_like(joint_pos)
        root_state = _as_tensor(self.robot.data.default_root_state)[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot_dof_targets[env_ids] = joint_pos
        self._prev_dist[env_ids] = 1.0
        self._prev_intercept_dist[env_ids] = 1.0

        # Select the episode type before posing the fingers. The launcher must
        # query link transforms only after the matching pre-close is written;
        # otherwise every zero-jitter trajectory targets a stale aperture.
        in_hand_p = self._in_hand_spawn_probability()
        use_in_hand = torch.rand(len(env_ids), device=self.device) < in_hand_p
        self._spawned_in_hand[env_ids] = use_in_hand
        self._spawned_drift[env_ids] = (~use_in_hand) & self._in_drift_mode()

        # Preclose: in-hand uses spawn_in_hand_preclose; drift handoff uses lighter frac.
        in_hand_mask = self._spawned_in_hand[env_ids]
        preclose_frac = float(getattr(self.cfg, "spawn_in_hand_preclose", 0.72))
        drift_band = self._in_drift_mode()
        drift_mask = (~in_hand_mask) & drift_band
        drift_preclose = float(getattr(self.cfg, "throw_drift_preclose", 0.35))
        lob_mask = (~in_hand_mask) & (~self._spawned_drift[env_ids])
        alpha = self._curriculum_alpha()
        lob_preclose_easy = float(getattr(self.cfg, "throw_lob_preclose_easy", 0.0))
        lob_preclose_hard = float(getattr(self.cfg, "throw_lob_preclose_hard", 0.0))
        lob_preclose = lob_preclose_easy * (1.0 - alpha) + lob_preclose_hard * alpha

        def set_preclose(local_ids: torch.Tensor, sim_env_ids: torch.Tensor, fraction: float) -> None:
            """Apply the same tendon curve used by the exported scalar grip action."""
            base_preclose = (
                self.cfg.gripper_open_pos * (1.0 - fraction)
                + self.cfg.gripper_close_target * fraction
            )
            safe_base = (
                float(self.cfg.gripper_open_pos)
                * (1.0 - float(self.cfg.gripper_safe_close_fraction))
                + float(self.cfg.gripper_close_target)
                * float(self.cfg.gripper_safe_close_fraction)
            )
            base_preclose = min(float(base_preclose), safe_base)
            tip_preclose = float(self.cfg.gripper_open_pos) + float(
                self.cfg.gripper_tip_target_ratio
            ) * (base_preclose - float(self.cfg.gripper_open_pos))
            tip_preclose = min(tip_preclose, float(self.cfg.gripper_tip_safe_close))
            joint_pos[local_ids[:, None], self._gripper_base_joint_ids] = base_preclose
            joint_pos[local_ids[:, None], self._gripper_tip_joint_ids] = tip_preclose
            self.robot.write_joint_state_to_sim(
                joint_pos[local_ids], torch.zeros_like(joint_pos[local_ids]), None, sim_env_ids
            )
            self.robot_dof_targets[sim_env_ids[:, None], self._gripper_base_joint_ids] = base_preclose
            self.robot_dof_targets[sim_env_ids[:, None], self._gripper_tip_joint_ids] = tip_preclose

        if in_hand_mask.any() and preclose_frac > 1e-6:
            ih_local = in_hand_mask.nonzero(as_tuple=False).flatten()
            ih_env = env_ids[ih_local]
            set_preclose(ih_local, ih_env, preclose_frac)
        if drift_mask.any() and drift_preclose > 1e-6:
            d_local = drift_mask.nonzero(as_tuple=False).flatten()
            d_env = env_ids[d_local]
            set_preclose(d_local, d_env, drift_preclose)
        if lob_mask.any() and lob_preclose > 1e-6:
            lob_local = lob_mask.nonzero(as_tuple=False).flatten()
            lob_env = env_ids[lob_local]
            set_preclose(lob_local, lob_env, lob_preclose)

        # Body-pose buffers are invalidated by write_joint_state_to_sim, so the
        # following link queries now reflect this episode's actual finger pose.
        self._launch_ball(env_ids)

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
        forced = getattr(self.cfg, "forced_curriculum_alpha", None)
        if forced is not None:
            return float(max(0.0, min(1.0, forced)))
        steps = max(int(self.cfg.curriculum_steps), 1)
        step_alpha = min(float(self.common_step_counter) / float(steps), 1.0)
        return min(step_alpha, float(self._curriculum_cap))

    def _update_curriculum_gate(
        self,
        batch_soft_grasp: float,
        soft_grasp_frac: float | None = None,
        hold_mean: float | None = None,
        throw_soft_grasp: float | None = None,
        lob_gentle_grasp: float | None = None,
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
        if lob_gentle_grasp is not None:
            self._lob_catch_ema = (1.0 - alpha_ema) * self._lob_catch_ema + alpha_ema * float(
                lob_gentle_grasp
            )
        if getattr(self.cfg, "forced_curriculum_alpha", None) is not None:
            return
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
            drift_until = float(getattr(self.cfg, "throw_drift_until_alpha", 0.40))
            lob_stage = phase in ("throw", "throw_b") and not self._in_drift_mode()
            if lob_stage:
                gate = (
                    float(self._lob_gentle_rolling)
                    if len(self._lob_hist) >= hist_min
                    else float(self._lob_catch_ema)
                )
            else:
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
                    if self._curriculum_cap < drift_until:
                        self._curriculum_cap = min(drift_until, next_cap)
                    elif self._throw_gate_streak >= streak_need:
                        self._curriculum_cap = next_cap
            elif gate < unlock * 0.55:
                floor = drift_until if lob_stage else 0.03
                self._curriculum_cap = max(floor, self._curriculum_cap - rate * 0.6)
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

        # Correct toward the equidistant three-tip point without leaving the
        # interior of this obtuse tip triangle. Policy observations retain
        # their deployed 30-D arithmetic-mean tip contract.
        mean_tip_center = self._tip_center_w()[env_ids] - origins
        circumcenter = self._tip_aperture_center_w()[env_ids] - origins
        circum_easy = float(getattr(self.cfg, "throw_aperture_circumcenter_blend_easy", 0.0))
        circum_hard = float(getattr(self.cfg, "throw_aperture_circumcenter_blend_hard", circum_easy))
        circum_blend = circum_easy * (1.0 - alpha) + circum_hard * alpha
        tip_center = mean_tip_center + circum_blend * (circumcenter - mean_tip_center)
        ee_pos = _as_tensor(self.robot.data.body_pos_w)[env_ids, self._ee_body_idx] - origins
        # Aim inside the distal aperture, not halfway back toward the palm.
        # Throw-B starts nearer the fingertip hooks so the ball is enclosed
        # before its momentum can carry it through the curved finger wedge.
        aim_easy = float(getattr(self.cfg, "throw_cup_aim_fraction_easy", 0.68))
        aim_hard = float(getattr(self.cfg, "throw_cup_aim_fraction_hard", aim_easy))
        aim_fraction = aim_easy * (1.0 - alpha) + aim_hard * alpha
        cup_aim = aim_fraction * tip_center + (1.0 - aim_fraction) * ee_pos

        # Episode type was sampled in _reset_idx before applying its matching
        # finger pose, so launch geometry and pre-close cannot disagree.
        use_in_hand = self._spawned_in_hand[env_ids]

        pos = torch.zeros((n, 3), device=self.device)
        lin_vel = torch.zeros((n, 3), device=self.device)

        if torch.any(use_in_hand):
            ih = use_in_hand.nonzero(as_tuple=False).flatten()
            jitter = float(getattr(self.cfg, "spawn_in_hand_jitter", 0.012))
            speed = float(getattr(self.cfg, "spawn_in_hand_speed", 0.08))
            # Closer to palm so fingers can wrap; tiny residual speed only.
            in_hand_along = getattr(self.cfg, "spawn_in_hand_along", (0.20, 0.48))
            along = sample_uniform(*in_hand_along, (len(ih), 1), device=self.device)
            aperture = ee_pos[ih] + along * (tip_center[ih] - ee_pos[ih])
            aperture += torch.tensor(
                getattr(self.cfg, "spawn_in_hand_offset", (0.0, 0.0, 0.0)),
                device=self.device,
                dtype=aperture.dtype,
            )
            aperture += sample_uniform(-jitter, jitter, (len(ih), 3), device=self.device)
            pos[ih] = aperture + origins[ih]
            lin_vel[ih] = sample_uniform(-speed, speed, (len(ih), 3), device=self.device)

        if torch.any(~use_in_hand):
            th = (~use_in_hand).nonzero(as_tuple=False).flatten()
            nt = len(th)
            front = self._mix_range(self.cfg.throw_front_offset_easy, self.cfg.throw_front_offset, alpha)
            side = self._mix_range(self.cfg.throw_side_offset_easy, self.cfg.throw_side_offset, alpha)
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
                self._spawned_drift[env_ids[th]] = True
                along_easy = getattr(self.cfg, "throw_drift_along_easy", (0.55, 0.90))
                along_hard = getattr(self.cfg, "throw_drift_along_hard", (0.30, 0.60))
                speed_easy = getattr(self.cfg, "throw_drift_speed_easy", (0.04, 0.10))
                speed_hard = getattr(self.cfg, "throw_drift_speed_hard", getattr(self.cfg, "throw_drift_speed", (0.08, 0.20)))
                along_r = self._mix_range(along_easy, along_hard, alpha)
                speed_r = self._mix_range(speed_easy, speed_hard, alpha)
                along = sample_uniform(*along_r, (nt, 1), device=self.device)
                aperture = ee_pos[th] + along * (tip_center[th] - ee_pos[th])
                jitter_easy = float(getattr(self.cfg, "throw_drift_jitter_easy", 0.004))
                jitter_hard = float(getattr(self.cfg, "throw_drift_jitter_hard", 0.035))
                jitter = jitter_easy * (1.0 - alpha) + jitter_hard * alpha
                aperture += sample_uniform(-jitter, jitter, (nt, 3), device=self.device)
                speed = sample_uniform(*speed_r, (nt, 1), device=self.device)
                axis = tip_center[th] - ee_pos[th]
                axis_n = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-4)
                pos[th] = aperture + origins[th]
                lin_vel[th] = -axis_n * speed + sample_uniform(-0.03, 0.03, (nt, 3), device=self.device)
            else:
                # Spawn beyond the fingertip plane along the actual grasp
                # axis. A world-X offset can enter this rotated Jaco hand
                # sideways, striking one finger even though the endpoint is
                # the cup center.
                axis = tip_center[th] - ee_pos[th]
                axis_n = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-4)
                world_up = torch.zeros_like(axis_n)
                world_up[:, 2] = 1.0
                side_axis = torch.linalg.cross(world_up, axis_n, dim=-1)
                side_norm = torch.linalg.norm(side_axis, dim=-1, keepdim=True)
                fallback = torch.zeros_like(axis_n)
                fallback[:, 1] = 1.0
                fallback = fallback - (fallback * axis_n).sum(dim=-1, keepdim=True) * axis_n
                fallback = fallback / torch.linalg.norm(fallback, dim=-1, keepdim=True).clamp(min=1e-4)
                side_axis = torch.where(
                    side_norm > 1e-4,
                    side_axis / side_norm.clamp(min=1e-4),
                    fallback,
                )
                release = cup_aim[th].clone()
                release += axis_n * sample_uniform(*front, (nt, 1), device=self.device)
                release += side_axis * sample_uniform(*side, (nt, 1), device=self.device)
                release[:, 2] -= sample_uniform(*self.cfg.throw_below_offset, (nt,), device=self.device)

                flight_t = sample_uniform(*flight, (nt, 1), device=self.device)
                g_eff = self.cfg.gravity_full * self.cfg.ball_gravity_scale
                d = target - release
                g = torch.zeros((nt, 3), device=self.device)
                g[:, 2] = -g_eff
                # Exact constant-gravity endpoint solution. The previous extra
                # loft term was added after this solve, so it necessarily made
                # the ball overshoot the cup vertically at flight_t.
                v = d / flight_t - 0.5 * g * flight_t
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
