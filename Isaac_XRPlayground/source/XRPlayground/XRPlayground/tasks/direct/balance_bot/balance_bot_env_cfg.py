# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Balance Bot (2-DOF ball-on-plate tray)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass


@configclass
class BalanceBotEnvCfg(DirectRLEnvCfg):
    """Keep hand-sized ball(s) on a tilting tray. Curriculum: 1 ball → 2 balls."""

    decimation = 2
    episode_length_s = 8.0
    # roll + pitch rate commands
    action_space = 2
    # See balance_bot_env._get_observations for layout (20-D)
    observation_space = 20
    state_space = 0

    action_scale = 1.5  # rad/s at |action|=1
    max_tilt_rad = 0.40
    dof_velocity_scale = 0.25

    # Shared link / joint names (must match Unity BalanceBotLinkMap + bridge names_balance_bot)
    link_names = ["base_link", "roll_link", "tray_link"]
    joint_names = ["roll_joint", "pitch_joint"]
    default_joint_pos = (0.0, 0.0)

    # Layout (Isaac Z-up, env-local)
    base_pos = (0.0, 0.0, 0.0)
    tray_center = (0.0, 0.0, 0.75)
    tray_size = (0.60, 0.60, 0.02)  # x, y, thickness (1.5× prior plate)
    pedestal_size = (0.12, 0.12, 0.70)
    pedestal_pos = (0.0, 0.0, 0.35)

    ball_radius = 0.035  # half prior radius
    # Keep tennis-ball mass on a smaller sphere → denser / still heavy feel
    ball_mass = 0.057
    num_ball_slots = 2
    # Drop slightly above tray surface
    spawn_height_above_tray = 0.08
    spawn_xy_half = 0.12
    # Fail if ball center leaves this half-extent from tray center (xy) or falls below
    tray_half_xy = 0.27
    fall_z = 0.35

    # Curriculum uses common_step_counter (1 per env.step, NOT summed across envs).
    # With num_steps_per_env=16, 1000 iters ≈ 16k steps — old 80k never unlocked stage 1.
    curriculum_steps = 10000  # ~625 iters fallback if hold gate is slow
    # Also unlock earlier once rolling mean hold looks solid (episode len ~480 at dt=1/60, 8s)
    curriculum_unlock_hold_steps = 350.0
    curriculum_unlock_patience = 40  # reset-batches with mean_hold >= unlock before stage 1
    # Set >=1 to force 2-ball from the first reset (e.g. resume after proven 1-ball).
    # Keep 0 for cold-start; hold-gate / curriculum_steps then unlock stage 1.
    # Keep 1 for play of the exported 2-ball policy; set 0 for cold-start train.
    curriculum_start_stage = 1
    # Phase 1 = 1 ball; phase 2 = 2 balls
    curriculum_phase1_balls = 1
    curriculum_phase2_balls = 2

    # Rewards
    rew_alive = 0.5
    rew_center = 2.5
    rew_vel_penalty = 0.15
    rew_tilt_penalty = 0.08
    rew_action_penalty = 0.01
    rew_drop_penalty = 5.0
    rew_hold_bonus = 0.35  # near-center + slow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.85,
            dynamic_friction=0.75,
            restitution=0.15,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=False)

    pedestal_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/base_link",
        spawn=sim_utils.CuboidCfg(
            size=pedestal_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pedestal_pos, rot=(0.0, 0.0, 0.0, 1.0)),
    )

    tray_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/tray_link",
        spawn=sim_utils.CuboidCfg(
            size=tray_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.28)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.95,
                dynamic_friction=0.85,
                restitution=0.12,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tray_center, rot=(0.0, 0.0, 0.0, 1.0)),
    )

    ball_0_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball_0",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.55, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.6,
                restitution=0.35,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass),
        ),
        # On-tray so Kit viewport shows balls even before first reset / if Fabric sync lags.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, tray_center[2] + 0.5 * tray_size[2] + ball_radius + spawn_height_above_tray),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    ball_1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball_1",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.55, 0.85)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.6,
                restitution=0.35,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.09, 0.06, tray_center[2] + 0.5 * tray_size[2] + ball_radius + spawn_height_above_tray),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
