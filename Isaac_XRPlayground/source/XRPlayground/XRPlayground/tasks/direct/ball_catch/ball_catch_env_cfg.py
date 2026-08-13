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

# Kinova Jaco2 7-DoF + 3-finger gripper. Arcade-style catch assists (not 1:1 real).


@configclass
class BallCatchEnvCfg(DirectRLEnvCfg):
    # env — 7 arm joints + 1 shared gripper command
    decimation = 2
    episode_length_s = 3.5
    action_space = 8
    observation_space = 25
    state_space = 0

    # Delta joint commands; keep moderate so continuous joints (esp. base) don't wind up.
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
    # Non-finger links used to detect "balance ball on arm" cheating
    arm_body_names = [
        "j2n7s300_link_1",
        "j2n7s300_link_2",
        "j2n7s300_link_3",
        "j2n7s300_link_4",
        "j2n7s300_link_5",
        "j2n7s300_link_6",
        "j2n7s300_link_7",
        "j2n7s300_end_effector",
    ]
    ee_body_name = "j2n7s300_end_effector"
    gripper_open_pos = 0.2
    gripper_close_target = 1.2

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
                "j2n7s300_joint_finger_[1-3]": 0.2,
                "j2n7s300_joint_finger_tip_[1-3]": 0.2,
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
            radius=0.055,
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

    # Easy throws first; lerp toward hard ranges over curriculum_steps
    curriculum_steps = 12000
    throw_pos_x_easy = (0.62, 0.88)
    throw_pos_x_hard = (0.70, 1.10)
    throw_pos_y_easy = (-0.16, 0.16)
    throw_pos_y_hard = (-0.38, 0.38)
    throw_pos_z_easy = (0.58, 0.78)
    throw_pos_z_hard = (0.52, 0.85)
    throw_speed_easy = (0.55, 1.05)
    throw_speed_hard = (1.10, 2.00)
    aim_pos_x_easy = (0.28, 0.42)
    aim_pos_x_hard = (0.18, 0.52)
    aim_pos_y_easy = (-0.10, 0.10)
    aim_pos_y_hard = (-0.28, 0.28)
    aim_pos_z_easy = (0.48, 0.62)
    aim_pos_z_hard = (0.40, 0.72)
    throw_ang_vel = (-2.0, 2.0)

    # Arcade grasp assist: only inside the finger volume while closing
    assist_capture_radius = 0.10
    assist_min_close = 0.25
    assist_vel_damping = 0.18
    assist_pull = 0.55

    # reward / success scales — grasp-centric (body balancing is penalized)
    dist_reward_scale = 4.0
    approach_reward_scale = 6.0
    catch_reward_scale = 50.0
    grasp_reward_scale = 8.0
    # Reward staying still while gripping (joint-speed + action quiet)
    hold_still_reward_scale = 3.0
    hold_action_penalty_scale = 0.08
    body_contact_penalty = 8.0
    drop_penalty = 5.0
    action_penalty_scale = 0.006
    gripper_near_dist = 0.12
    success_dist_threshold = 0.09
    success_speed_threshold = 1.2
    success_close_min = 0.35
    grasp_hold_steps = 8
    # After a confirmed catch, freeze and hold until episode timeout (do not end early).
    terminate_on_catch = False
    body_contact_radius = 0.11
    body_fail_steps = 12
    fall_height_threshold = 0.06
