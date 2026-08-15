# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ball-catch industrial manipulator task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.kinova import KINOVA_JACO2_N7S300_CFG

# Kinova Jaco2 7-DoF + 3-finger gripper.


@configclass
class BallCatchEnvCfg(DirectRLEnvCfg):
    # env — 7 arm joints + 1 shared gripper command
    decimation = 2
    episode_length_s = 4.5
    action_space = 8
    observation_space = 30  # + grasp_align, ball_radial
    state_space = 0

    # Delta joint commands — match cup-hold baseline (stable, not wild).
    action_scale = 5.0
    dof_velocity_scale = 0.1
    # PhysX revolute drive targets must stay in [-2π, 2π]
    physx_drive_angle_limit = 6.283185307179586
    arm_joint_names = [
        "j2n7s300_joint_1",
        "j2n7s300_joint_2",
        "j2n7s300_joint_3",
        "j2n7s300_joint_4",
        "j2n7s300_joint_5",
        "j2n7s300_joint_6",
        "j2n7s300_joint_7",
    ]
    gripper_joint_names = [
        "j2n7s300_joint_finger_1",
        "j2n7s300_joint_finger_2",
        "j2n7s300_joint_finger_3",
        "j2n7s300_joint_finger_tip_1",
        "j2n7s300_joint_finger_tip_2",
        "j2n7s300_joint_finger_tip_3",
    ]
    gripper_body_names = [
        "j2n7s300_link_finger_1",
        "j2n7s300_link_finger_2",
        "j2n7s300_link_finger_3",
    ]
    finger_tip_body_names = [
        "j2n7s300_link_finger_tip_1",
        "j2n7s300_link_finger_tip_2",
        "j2n7s300_link_finger_tip_3",
    ]
    # Non-finger links used to detect "balance ball on arm" cheating (exclude EE/palm)
    arm_body_names = [
        "j2n7s300_link_1",
        "j2n7s300_link_2",
        "j2n7s300_link_3",
        "j2n7s300_link_4",
        "j2n7s300_link_5",
        "j2n7s300_link_6",
        "j2n7s300_link_7",
    ]
    ee_body_name = "j2n7s300_end_effector"
    gripper_open_pos = 0.04
    gripper_close_target = 1.10

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
    )

    # scene — tuned for RTX 3060 Ti (raise num_envs once stable)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.5, replicate_physics=True)

    robot_cfg: ArticulationCfg = KINOVA_JACO2_N7S300_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=KINOVA_JACO2_N7S300_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "j2n7s300_joint_1": 0.0,
                "j2n7s300_joint_2": 2.35,
                "j2n7s300_joint_3": 0.25,
                "j2n7s300_joint_4": 1.65,
                "j2n7s300_joint_5": 1.40,
                "j2n7s300_joint_6": 0.35,
                "j2n7s300_joint_7": 0.0,
                "j2n7s300_joint_finger_[1-3]": 0.04,
                "j2n7s300_joint_finger_tip_[1-3]": 0.04,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint_[1-7]"],
                effort_limit_sim={
                    ".*_joint_[1-2]": 220.0,
                    ".*_joint_[3-4]": 140.0,
                    ".*_joint_[5-7]": 80.0,
                },
                velocity_limit_sim=3.5,
                stiffness={
                    ".*_joint_[1-4]": 280.0,
                    ".*_joint_[5-7]": 120.0,
                },
                damping={
                    ".*_joint_[1-4]": 12.0,
                    ".*_joint_[5-7]": 6.0,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*_finger_[1-3]", ".*_finger_tip_[1-3]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=4.0,
                stiffness=40.0,
                damping=2.0,
            ),
        },
    )

    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.04125,  # 75% of previous 0.055
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.25, 0.15)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.8, dynamic_friction=1.4, restitution=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, 0.0, 0.70), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Player-style parabolic toss with curriculum: learnable → farther / more reaction time.
    curriculum_steps = 28000  # ~full hard throws by ~iter 1000 (vectorized step counter)
    throw_front_offset = (0.30, 0.50)       # hard (final)
    throw_front_offset_easy = (0.16, 0.28)  # still off the palm, random
    throw_side_offset = (-0.18, 0.18)
    throw_side_offset_easy = (-0.10, 0.10)
    throw_below_offset = (0.02, 0.10)
    throw_height_boost = (0.12, 0.22)
    throw_height_boost_easy = (0.08, 0.14)
    throw_flight_time = (0.75, 1.05)        # hard
    throw_flight_time_easy = (0.55, 0.75)
    aim_jitter_xy = 0.05
    aim_jitter_xy_easy = 0.03
    aim_jitter_z = 0.04
    aim_jitter_z_easy = 0.025
    gravity_full = 9.81
    ball_gravity_scale = 0.50
    # Legacy fields
    throw_pos_x_easy = (0.55, 0.85)
    throw_pos_x_hard = (0.55, 0.85)
    throw_pos_y_easy = (-0.20, 0.20)
    throw_pos_y_hard = (-0.20, 0.20)
    throw_pos_z_easy = (0.45, 0.70)
    throw_pos_z_hard = (0.45, 0.70)
    aim_pos_x_easy = (0.28, 0.36)
    aim_pos_x_hard = (0.28, 0.36)
    aim_pos_y_easy = (-0.08, 0.08)
    aim_pos_y_hard = (-0.08, 0.08)
    aim_pos_z_easy = (0.50, 0.58)
    aim_pos_z_hard = (0.50, 0.58)
    throw_speed_easy = (0.45, 0.75)
    throw_speed_hard = (0.45, 0.75)
    throw_ang_vel = (-1.5, 1.5)

    dist_reward_scale = 4.0
    approach_reward_scale = 2.5
    catch_reward_scale = 200.0
    grasp_reward_scale = 14.0
    hold_still_reward_scale = 4.0
    hold_action_penalty_scale = 0.08
    face_ball_reward_scale = 4.0
    aperture_reward_scale = 10.0
    side_miss_penalty = 8.0
    wrap_reward_scale = 8.0
    poke_penalty = 12.0
    early_close_penalty = 2.5
    # Keep assist latch through the full 1k run — soft_grasp alone never bootstrapped.
    assist_catch_until_curriculum = 1.01
    # Soft assist when facing + near; tip-spread gate is loose so open claws can still settle.
    catch_assist_enabled = True
    catch_assist_dist = 0.13
    catch_assist_align_min = 0.25
    catch_assist_close_min = 0.15
    catch_assist_spring_kp = 38.0
    catch_assist_spring_kd = 4.0
    catch_assist_force_clip = 2.6
    catch_assist_blend_dist = 0.085
    catch_assist_blend_alpha = 0.24
    catch_assist_tip_spread_max = 0.10
    body_contact_penalty = 6.0
    cup_balance_penalty = 14.0
    drop_penalty = 5.0
    action_penalty_scale = 0.01
    gripper_near_dist = 0.14
    grasp_along_min = 0.018
    grasp_along_max = 0.110
    grasp_radial_max = 0.055
    grasp_beyond_tips_margin = 0.012
    success_tip_dist = 0.085
    success_ee_dist = 0.12
    success_finger_dist = 0.085
    success_dist_threshold = 0.085
    success_speed_threshold = 1.10
    success_close_min = 0.20
    grasp_align_min = 0.28
    cup_align_max = 0.18
    cup_height_margin = 0.022
    # Open 3-finger claw tip-spread is ~0.09; wrap latch must tolerate that, poke is worse.
    tip_spread_max = 0.095
    poke_tip_spread = 0.120
    grasp_hold_steps = 3
    terminate_on_catch = False
    body_contact_radius = 0.10
    body_fail_steps = 14
    cup_fail_steps = 14
    fall_height_threshold = 0.06