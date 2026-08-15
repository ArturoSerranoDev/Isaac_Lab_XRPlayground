# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Agibot A2D table pick-and-place (wall-mounted, right arm)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG

# Piece colors (Unity / bridge)
COLOR_RGB = (
    (0.90, 0.15, 0.12),
    (0.15, 0.75, 0.25),
    (0.15, 0.35, 0.90),
    (0.85, 0.75, 0.15),
)


def _piece_cfg(idx: int, color_rgb: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Piece_{idx}",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_rgb),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5), rot=(0.0, 0.0, 0.0, 1.0)),
    )


@configclass
class PickPlaceTableEnvCfg(DirectRLEnvCfg):
    """Pick random table pieces and drop them in a front-center bucket."""

    decimation = 2
    episode_length_s = 10.0
    action_space = 8
    observation_space = 30
    state_space = 0

    action_scale = 5.0
    dof_velocity_scale = 0.1
    physx_drive_angle_limit = 6.283185307179586

    # Right arm + parallel gripper driver (matches IsaacLab Agibot place tasks)
    arm_joint_names = [f"right_arm_joint{i}" for i in range(1, 8)]
    gripper_joint_names = ["right_hand_joint1", "right_Right_Support_Joint", "right_Left_Support_Joint"]
    gripper_driver_name = "right_hand_joint1"
    ee_body_name = "right_gripper_center"
    gripper_open = 0.994
    gripper_close = 0.20

    # Joints held at default (body, head, left arm + gripper)
    frozen_joint_names = [
        "joint_lift_body",
        "joint_body_pitch",
        "joint_head_yaw",
        "joint_head_pitch",
        *[f"left_arm_joint{i}" for i in range(1, 8)],
        "left_hand_joint1",
        "left_Right_Support_Joint",
        "left_Left_Support_Joint",
    ]

    num_pieces = 4
    num_colors = len(COLOR_RGB)
    max_active_pieces = 1

    # Imitation-learning hook (VR demos in Unity → Isaac datagen later)
    imitation_learning_enabled = False
    demo_record_rate_hz = 30.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)

    # Wall-mounted pose: robot behind table along -Y, facing +Y
    robot_cfg: ArticulationCfg = AGIBOT_A2D_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=AGIBOT_A2D_CFG.spawn.replace(
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
            pos=(0.0, -0.78, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=AGIBOT_A2D_CFG.init_state.joint_pos,
        ),
        actuators={
            **AGIBOT_A2D_CFG.actuators,
            "body": AGIBOT_A2D_CFG.actuators["body"].replace(
                stiffness=10000000.0,
                damping=500.0,
            ),
            "head": AGIBOT_A2D_CFG.actuators["head"].replace(stiffness=800.0, damping=40.0),
            "left_arm": AGIBOT_A2D_CFG.actuators["left_arm"].replace(stiffness=80000.0, damping=80.0),
        },
    )

    table_pos = (0.45, 0.0, 0.38)
    table_size = (0.75, 0.60, 0.04)
    # Front of table toward robot (-Y)
    bucket_pos = (0.45, -0.18, 0.38)
    bucket_size = (0.14, 0.14, 0.10)
    bucket_xy_radius = 0.09
    bucket_z_min = 0.395
    bucket_z_max = 0.52

    spawn_x_range = (0.30, 0.60)
    spawn_y_range = (-0.22, 0.22)
    cube_half_size = 0.02

    grasp_dist = 0.12
    lift_height = 0.46
    success_xy_radius = 0.10
    success_z_max = 0.50

    dist_reward_scale = 4.0
    approach_reward_scale = 12.0
    grasp_reward_scale = 18.0
    lift_reward_scale = 8.0
    bucket_approach_scale = 10.0
    place_reward_scale = 80.0
    action_penalty_scale = 0.02
    drop_penalty = 6.0
    floor_z = 0.15

    piece_0_cfg: RigidObjectCfg = _piece_cfg(0, COLOR_RGB[0])
    piece_1_cfg: RigidObjectCfg = _piece_cfg(1, COLOR_RGB[1])
    piece_2_cfg: RigidObjectCfg = _piece_cfg(2, COLOR_RGB[2])
    piece_3_cfg: RigidObjectCfg = _piece_cfg(3, COLOR_RGB[3])
