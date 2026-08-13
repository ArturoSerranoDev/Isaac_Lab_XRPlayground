# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ball-catch industrial manipulator task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.kinova import KINOVA_JACO2_N7S300_CFG

# Kinova Jaco2 7-DoF + 3-finger gripper. Uses an instanceable USD (no Robotiq variants).


@configclass
class BallCatchEnvCfg(DirectRLEnvCfg):
    # env — 7 arm joints + 1 shared gripper command
    decimation = 2
    episode_length_s = 6.0
    action_space = 8
    observation_space = 25
    state_space = 0

    action_scale = 5.0
    dof_velocity_scale = 0.1
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
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.05,
        ),
    )

    # scene — tuned for RTX 3060 Ti (raise num_envs once stable)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.5, replicate_physics=True)

    robot_cfg: ArticulationCfg = KINOVA_JACO2_N7S300_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "j2n7s300_joint_1": 0.0,
                "j2n7s300_joint_2": 2.76,
                "j2n7s300_joint_3": 0.0,
                "j2n7s300_joint_4": 2.0,
                "j2n7s300_joint_5": 2.0,
                "j2n7s300_joint_6": 0.0,
                "j2n7s300_joint_7": 0.0,
                "j2n7s300_joint_finger_[1-3]": 0.2,
                "j2n7s300_joint_finger_tip_[1-3]": 0.2,
            },
        ),
    )

    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.25, 0.15)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.4, restitution=0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.057),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.95, 0.0, 0.72), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Player release zone (env frame: robot near origin, player stands at +X)
    throw_pos_x = (0.70, 1.20)
    throw_pos_y = (-0.50, 0.50)
    throw_pos_z = (0.55, 0.88)

    # Throw speed (m/s) — slower casual toss, not a fast pitch
    throw_speed = (1.4, 2.8)

    # Random aim point in the robot workspace (controls angle + arc)
    aim_pos_x = (0.15, 0.55)
    aim_pos_y = (-0.35, 0.35)
    aim_pos_z = (0.40, 0.78)

    # Optional ball spin (rad/s) for visual realism
    throw_ang_vel = (-4.0, 4.0)

    # reward / success scales
    dist_reward_scale = 2.0
    catch_reward_scale = 25.0
    action_penalty_scale = 0.02
    success_dist_threshold = 0.10
    success_speed_threshold = 0.75
    fall_height_threshold = 0.06
