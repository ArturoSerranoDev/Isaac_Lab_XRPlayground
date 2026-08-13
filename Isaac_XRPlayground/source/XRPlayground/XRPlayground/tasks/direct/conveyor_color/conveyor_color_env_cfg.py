# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Conveyor Color Detection (UR10e + Robotiq 2F-85)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG


# RGB for PreviewSurface (matches Unity: red / green / blue)
COLOR_RGB = (
    (0.90, 0.15, 0.12),
    (0.15, 0.75, 0.25),
    (0.15, 0.35, 0.90),
)


def _cube_cfg(idx: int, color_rgb: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_{idx}",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        # Parked under the floor until activated
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5), rot=(0.0, 0.0, 0.0, 1.0)),
    )


@configclass
class ConveyorColorEnvCfg(DirectRLEnvCfg):
    """Sort conveyor cubes: target color → side table; rejects → end trash."""

    decimation = 2
    episode_length_s = 12.0
    # 6 arm + 1 gripper command
    action_space = 7
    # See conveyor_color_env._get_observations for layout
    observation_space = 66
    state_space = 0

    action_scale = 4.0
    dof_velocity_scale = 0.1
    physx_drive_angle_limit = 6.283185307179586

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    gripper_joint_names = ["finger_joint"]
    # Resolved flexibly if USD names differ
    ee_body_name = "wrist_3_link"
    gripper_body_names = ["left_inner_finger", "right_inner_finger"]

    num_colors = 3
    num_object_slots = 4
    cube_half_size = 0.025
    spawn_interval_s = 2.5
    belt_speed = 0.22
    # Belt center + size in Isaac Z-up (length along +Y)
    belt_pos = (0.55, 0.0, 0.40)
    belt_size = (0.30, 1.40, 0.04)
    # Usable belt surface extents (margin from edges so cubes stay on)
    belt_half_x = 0.11
    belt_y_min = -0.62
    belt_y_max = 0.62
    # Spawn band near the start of the belt (random x/y on surface)
    spawn_y_min = -0.60
    spawn_y_max = -0.40

    # Side sort table (CORRECT / target color) — beside the robot, belt height − a bit
    # belt top z = 0.42
    sort_table_pos = (-0.40, 0.15, 0.405)
    sort_table_size = (0.36, 0.36, 0.03)
    # Alias used by obs / approach shaping (success target)
    bin_pos = sort_table_pos
    bin_half_extents = (0.18, 0.18, 0.015)
    bin_success_xy = 0.18
    bin_success_z_max = 0.48
    bin_success_z_min = 0.38

    # End trash platform (REJECT / non-target) — end of conveyor
    trash_pos = (0.55, 0.78, 0.405)
    trash_size = (0.34, 0.28, 0.03)
    trash_success_xy = 0.16
    trash_success_z_max = 0.48
    trash_success_z_min = 0.38

    gripper_open = 0.0
    gripper_close = 0.785  # ~45 deg for 2F-85 drive

    grasp_dist = 0.06
    lift_height = 0.55
    fall_height = 0.12

    # rewards
    approach_reward_scale = 6.0
    grasp_reward_scale = 8.0
    lift_reward_scale = 4.0
    bin_approach_scale = 6.0
    success_reward = 40.0
    reject_reward = 8.0  # non-target reaches trash
    wrong_on_table_penalty = 10.0  # non-target placed on sort table
    target_to_trash_penalty = 12.0  # target missed → trash
    wrong_grasp_penalty = 6.0
    drop_penalty = 4.0
    action_penalty_scale = 0.008

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.5, replicate_physics=True)

    robot_cfg: ArticulationCfg = UR10e_ROBOTIQ_2F_85_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=UR10e_ROBOTIQ_2F_85_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # Kinematic belt visual + collision
    belt_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Belt",
        spawn=sim_utils.CuboidCfg(
            size=belt_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=belt_pos, rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Side table for target-colored cubes
    sort_table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SortTable",
        spawn=sim_utils.CuboidCfg(
            size=sort_table_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.55, 0.30)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=sort_table_pos, rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # End-of-belt trash / reject platform
    trash_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Trash",
        spawn=sim_utils.CuboidCfg(
            size=trash_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.18, 0.12)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=trash_pos, rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Object pool — materials are placeholders; colors are assigned at runtime via buffers
    object_0_cfg: RigidObjectCfg = _cube_cfg(0, COLOR_RGB[0])
    object_1_cfg: RigidObjectCfg = _cube_cfg(1, COLOR_RGB[1])
    object_2_cfg: RigidObjectCfg = _cube_cfg(2, COLOR_RGB[2])
    object_3_cfg: RigidObjectCfg = _cube_cfg(3, COLOR_RGB[0])
