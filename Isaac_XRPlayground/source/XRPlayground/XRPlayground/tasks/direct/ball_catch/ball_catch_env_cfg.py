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
    episode_length_s = 6.0
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
            static_friction=2.0,
            dynamic_friction=1.6,
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
                effort_limit_sim=28.0,
                velocity_limit_sim=6.0,
                stiffness=120.0,
                damping=6.0,
            ),
        },
    )

    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.04125,  # 75% of previous 0.055
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.25, 0.15)),
            # Higher grip friction (Unity-matched buoyancy unchanged).
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=3.5, dynamic_friction=3.0, restitution=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.055),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, 0.0, 0.70), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Two-phase training: "wrap" (in-hand enclose) → "throw" (in_hand_p=0). "mixed" = legacy fade.
    training_phase: str = "mixed"
    # If set, forces in-hand spawn probability (overrides fade). Wrap=0.95, Throw=0.0.
    in_hand_spawn_p: float | None = None

    # Throw curriculum: start with in-hand wrap practice, then reachable throws.
    # Schedule: alpha = min(step/curriculum_steps, curriculum_cap); cap rises after unlock.
    curriculum_steps = 50000  # ~3100 iters @ 16 steps/env — soft upper bound
    curriculum_unlock_catch = 0.60  # unlock harder throws sooner once wraps are real
    curriculum_advance_rate = 0.004
    # Also advance when soft_grasp_frac / hold streak look healthy (not only end-episode).
    curriculum_unlock_frac = 0.35
    curriculum_unlock_hold_steps = 6
    # In-hand spawn probability fades 1→0 by alpha=spawn_in_hand_until (then pure throws).
    spawn_in_hand_until = 0.30
    spawn_in_hand_jitter = 0.006
    spawn_in_hand_speed = 0.01
    spawn_in_hand_preclose = 0.85  # fraction toward close at reset (fades with in_hand_p)
    # Reachable throws: always aimed at cup volume; hard = wider within workspace.
    throw_front_offset = (0.16, 0.32)
    throw_front_offset_easy = (0.06, 0.14)
    throw_side_offset = (-0.14, 0.14)
    throw_side_offset_easy = (-0.03, 0.03)
    throw_below_offset = (0.02, 0.05)
    throw_height_boost = (0.06, 0.14)
    throw_height_boost_easy = (0.03, 0.07)
    throw_flight_time = (0.60, 0.95)
    throw_flight_time_easy = (0.70, 0.90)
    aim_jitter_xy = 0.035
    aim_jitter_xy_easy = 0.008
    aim_jitter_z = 0.028
    aim_jitter_z_easy = 0.008
    gravity_full = 9.81
    # Matched to Unity offline physics (not a catch cheat — same buoyancy in deployment).
    ball_gravity_scale = 0.55
    # Legacy fields (unused by parabolic launcher; kept for serialized configs)
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
    throw_ang_vel = (-1.0, 1.0)

    # Dense shaping is SMALL so hold/end-success dominate (prevents reward farming).
    dist_reward_scale = 4.0
    approach_reward_scale = 1.2
    catch_reward_scale = 80.0  # modest latch; end_hold_bonus is the real jackpot
    grasp_reward_scale = 12.0
    hold_reward_scale = 55.0  # primary dense signal after latch
    end_hold_bonus = 400.0  # sparse: soft_grasp at timeout with no drop
    # Sparse: first sustained enclose at intercept (separate from end_hold jackpot).
    intercept_enclose_bonus = 180.0
    # Dense: match tip/palm velocity to ball near contact (impact-aware catching).
    vel_match_reward_scale = 6.0
    vel_match_dist = 0.18
    drop_after_latch_penalty = 120.0  # harsh — latch then drop must not pay
    hold_still_reward_scale = 10.0
    hold_action_penalty_scale = 0.04
    face_ball_reward_scale = 2.5
    aperture_reward_scale = 6.0
    enclosure_reward_scale = 8.0
    side_miss_penalty = 12.0
    wrap_reward_scale = 5.0
    poke_penalty = 18.0
    early_close_penalty = 3.0
    close_reward_scale = 8.0
    # NO catch assist — spring/blend/finger-drive was reward-hacking soft_grasp_catch.
    assist_catch_until_curriculum = -1.0
    catch_assist_enabled = False
    catch_assist_fade_curriculum = 0.0  # unused when disabled
    catch_assist_dist = 0.12
    catch_assist_align_min = 0.16
    catch_assist_close_min = 0.08
    catch_assist_spring_kp = 0.0
    catch_assist_spring_kd = 0.0
    catch_assist_force_clip = 0.0
    catch_assist_blend_dist = 0.0
    catch_assist_blend_alpha = 0.0
    catch_assist_tip_spread_max = 0.13
    body_contact_penalty = 10.0
    cup_balance_penalty = 16.0
    drop_penalty = 20.0
    action_penalty_scale = 0.01
    gripper_near_dist = 0.14
    grasp_along_min = 0.008
    grasp_along_max = 0.140
    grasp_radial_max = 0.060
    grasp_beyond_tips_margin = 0.014
    success_tip_dist = 0.105
    success_tip_max = 0.125
    success_ee_dist = 0.15
    success_finger_dist = 0.105
    success_dist_threshold = 0.105
    success_speed_threshold = 1.35
    success_close_min = 0.12
    grasp_align_min = 0.18
    cup_align_max = 0.08
    cup_height_margin = 0.022
    tip_spread_max = 0.12
    poke_tip_spread = 0.15
    grasp_hold_steps = 3
    terminate_on_catch = False
    body_contact_radius = 0.10
    body_fail_steps = 14
    cup_fail_steps = 14
    fall_height_threshold = 0.06


@configclass
class BallCatchWrapEnvCfg(BallCatchEnvCfg):
    """Phase Wrap: mostly in-hand / near-zero speed → learn enclose + hold."""

    training_phase: str = "wrap"
    in_hand_spawn_p: float | None = 1.0
    spawn_in_hand_until = 1.0
    curriculum_unlock_catch = 0.55
    curriculum_unlock_frac = 0.40
    curriculum_advance_rate = 0.002
    # Dominate with hold / end-hold; avoid latch-farming (intercept bonus is throw-only).
    catch_reward_scale = 40.0
    hold_reward_scale = 90.0
    drop_after_latch_penalty = 250.0
    end_hold_bonus = 600.0
    intercept_enclose_bonus = 0.0  # wrap = in-hand; throw phase re-enables
    spawn_in_hand_speed = 0.005
    spawn_in_hand_preclose = 0.65
    episode_length_s = 4.0  # shorter: less room to drop after a lucky latch


@configclass
class BallCatchThrowAEnvCfg(BallCatchEnvCfg):
    """Throw-A: close-under-motion with arm frozen (in-aperture drift ramp). Resume wrap."""

    training_phase: str = "throw_a"
    in_hand_spawn_p: float | None = None
    spawn_in_hand_until = 0.70
    spawn_in_hand_preclose = 0.65  # match wrap so enclose transfers
    spawn_in_hand_speed = 0.008
    # Motion practice without starving wrap hold.
    throw_min_frac = 0.40
    throw_drift_until_alpha = 1.01
    throw_drift_speed_easy = (0.015, 0.045)
    throw_drift_speed_hard = (0.08, 0.20)
    throw_drift_along_easy = (0.75, 0.95)
    throw_drift_along_hard = (0.35, 0.65)
    throw_drift_preclose = 0.65
    throw_hist_len = 256
    throw_hist_min = 48
    curriculum_unlock_catch = 0.35
    curriculum_advance_rate = 0.0010
    throw_leave_drift_ema = 0.70
    throw_gate_streak = 10
    intercept_enclose_bonus = 160.0
    vel_match_reward_scale = 8.0
    vel_match_dist = 0.20
    catch_reward_scale = 50.0
    hold_reward_scale = 110.0
    end_hold_bonus = 650.0
    drop_after_latch_penalty = 280.0
    success_speed_threshold = 1.55
    grasp_hold_steps = 2
    episode_length_s = 4.5


@configclass
class BallCatchThrowBEnvCfg(BallCatchEnvCfg):
    """Throw-B: free-arm near approaches → parabolas. Resume Throw-A."""

    training_phase: str = "throw_b"
    in_hand_spawn_p: float | None = None
    spawn_in_hand_until = 0.35
    spawn_in_hand_preclose = 0.25
    spawn_in_hand_speed = 0.02
    throw_min_frac = 0.35
    # Brief ultra-near drift then free-arm lobs.
    throw_drift_until_alpha = 0.18
    throw_drift_speed_easy = (0.04, 0.10)
    throw_drift_speed_hard = (0.08, 0.18)
    throw_drift_along_easy = (0.45, 0.80)
    throw_drift_along_hard = (0.30, 0.55)
    throw_drift_preclose = 0.40
    throw_hist_len = 256
    throw_hist_min = 48
    curriculum_unlock_catch = 0.45
    curriculum_advance_rate = 0.0018
    throw_leave_drift_ema = 0.55
    throw_gate_streak = 10
    throw_front_offset_easy = (0.03, 0.08)
    throw_side_offset_easy = (-0.02, 0.02)
    throw_height_boost_easy = (0.015, 0.04)
    throw_flight_time_easy = (0.95, 1.25)
    aim_jitter_xy_easy = 0.005
    aim_jitter_z_easy = 0.005
    intercept_enclose_bonus = 300.0
    vel_match_reward_scale = 14.0
    vel_match_dist = 0.26
    catch_reward_scale = 100.0
    hold_reward_scale = 90.0
    end_hold_bonus = 560.0
    drop_after_latch_penalty = 250.0
    success_speed_threshold = 1.85
    grasp_hold_steps = 2
    episode_length_s = 6.0


@configclass
class BallCatchThrowEnvCfg(BallCatchThrowBEnvCfg):
    """Legacy alias → Throw-B (prefer Throw-A then Throw-B gym IDs)."""

    pass
