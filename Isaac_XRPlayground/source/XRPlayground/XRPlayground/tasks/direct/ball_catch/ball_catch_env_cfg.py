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
    diagnostic_camera = False
    reset_arm_joint_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Delta joint commands — match cup-hold baseline (stable, not wild).
    action_scale = 5.0
    # Wrap leaves arm outputs untrained. Moving-ball curricula smoothly release
    # their authority so initial random means cannot destroy the cup pose.
    arm_motion_scale_easy = 0.0
    arm_motion_scale_hard = 1.0
    arm_motion_scale_power = 2.0
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
    # The finger-tip rigid-body origins are joint frames whose lateral offsets
    # cancel when averaged, leaving the result near the wrist.  Place the
    # virtual grasp center at the physical distal aperture instead.
    grasp_center_distance = 0.105
    gripper_open_pos = 0.04
    gripper_close_target = 1.10
    # One exported grip action drives the Jaco's six finger joints. Map its
    # proximal target to a stronger distal curl, like a tendon-coupled hand.
    gripper_tip_target_ratio = 2.0
    # Prevent the finger drives from accumulating enough closing error to
    # squeeze a rigid sphere back out of the three-finger cup.  The distal
    # links need their own, larger limit so the tendon map can form a hook;
    # clamping all six joints to the proximal limit silently erased that curl.
    gripper_safe_close_fraction = 0.68
    gripper_tip_safe_close = 1.50
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
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
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
                # Cup-up deployment pose: the palm support is horizontal and
                # the open aperture faces the incoming parabolic ball.
                "j2n7s300_joint_6": -0.05,
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
                stiffness=55.0,
                damping=4.0,
            ),
        },
    )

    # The stock asset has no collision geometry on j2n7s300_end_effector; the
    # visible palm ends at a narrow wrist rim.  Add a thin, passive collision
    # surface to link_7 itself so it is part of the articulation (and therefore
    # cannot kinematically fight the adjacent finger links).  Its top surface is
    # 64 mm along the open-hand grasp axis from the EE origin, placing the
    # 41.25 mm-radius ball at the 105 mm distal aperture center when seated.
    palm_support_translation = (0.00126, -0.04817, -0.12772)
    palm_support_orientation = (0.42864, 0.00089, 0.0, 0.90348)
    palm_support_cfg = sim_utils.CylinderCfg(
        radius=0.050,
        height=0.012,
        axis="Z",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.12, 0.12, 0.14),
            roughness=0.8,
        ),
    )
    # A low four-rail lip prevents a correctly cushioned ball from rolling off
    # the finite palm before the fingers finish wrapping.  These are ordinary
    # passive colliders on link_7, not a constraint or catch assist.
    palm_rim_transforms = (
        ((0.001319, -0.026612, -0.080770), (0.42864, 0.00089, 0.0, 0.90348)),
        ((0.001243, -0.089866, -0.158224), (0.42864, 0.00089, 0.0, 0.90348)),
        ((0.051281, -0.058201, -0.119577), (0.30372, -0.30246, 0.63886, 0.63886)),
        ((-0.048719, -0.058277, -0.119417), (0.30372, -0.30246, 0.63886, 0.63886)),
    )
    palm_rim_cfg = sim_utils.CuboidCfg(
        size=(0.108, 0.008, 0.026),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.12, 0.12, 0.14),
            roughness=0.8,
        ),
    )

    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.04125,  # 75% of previous 0.055
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.25, 0.15)),
            # High rubber-like friction is required for the horizontal cup to
            # support the ball after impact.  The distal aperture aim prevents
            # the palm penetration that previously made this setting unstable.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=3.0,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
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
    # Evaluation-only overrides. Training leaves both unset.
    forced_curriculum_alpha: float | None = None
    force_throw_mode: str | None = None  # None | "drift" | "lob"

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
    spawn_in_hand_along = (0.65, 0.85)
    spawn_in_hand_offset = (0.0, 0.0, 0.0)
    spawn_in_hand_preclose = 0.85  # fraction toward close at reset (fades with in_hand_p)
    # Reachable throws: always aimed at cup volume; hard = wider within workspace.
    throw_front_offset = (0.16, 0.32)
    throw_front_offset_easy = (0.06, 0.14)
    throw_side_offset = (-0.14, 0.14)
    throw_side_offset_easy = (-0.03, 0.03)
    throw_below_offset = (0.02, 0.05)
    throw_height_boost = (0.06, 0.14)
    throw_height_boost_easy = (0.03, 0.07)
    # Short, catchable arcs. The exact ballistic solve reaches the sampled cup
    # target at flight_t; longer times create excessive downward impact speed.
    throw_flight_time = (0.45, 0.75)
    throw_flight_time_easy = (0.30, 0.42)
    throw_cup_aim_fraction_easy = 0.68
    throw_cup_aim_fraction_hard = 0.68
    throw_aperture_circumcenter_blend_easy = 0.0
    throw_aperture_circumcenter_blend_hard = 0.0
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
    hold_reward_scale = 8.0  # primary dense signal after latch
    end_hold_bonus = 300.0  # sparse: gentle retained grasp at timeout
    # Sparse: first sustained enclose at intercept (separate from end_hold jackpot).
    intercept_enclose_bonus = 100.0
    # Dense: match tip/palm velocity to ball near contact (impact-aware catching).
    vel_match_reward_scale = 6.0
    vel_match_dist = 0.18
    vel_match_error_penalty_scale = 0.0
    # Dense arm learning toward a short-horizon ballistic intercept. This never
    # modifies physics or actions; it only rewards policy-driven positioning.
    intercept_prediction_horizon = 0.20
    intercept_tracking_reward_scale = 5.0
    intercept_progress_reward_scale = 45.0
    # Observation-derived closure timing (reward only): remain open on a far
    # approach, then reach the gentle band shortly before closest approach.
    grip_timing_lead_time = 0.45
    grip_timing_end_time = 0.035
    grip_timing_target = 0.56
    grip_timing_reward_scale = 6.0
    grip_timing_error_penalty_scale = 8.0
    # Optional immediate action-space teaching signal.  The target is computed
    # from the same policy observations and robot Jacobian available during
    # training; it is reward shaping only and never replaces the policy action.
    expert_arm_action_penalty_scale = 0.0
    expert_grip_action_penalty_scale = 0.0
    expert_cushion_lead_time = 0.30
    expert_cushion_velocity_gain = 1.6
    expert_cushion_damping = 0.08
    expert_position_gain = 6.0
    expert_position_velocity_max = 0.60
    expert_position_horizon = 0.85
    expert_grip_lead_time = 0.45
    expert_grip_end_time = 0.16
    expert_grip_target_close = 0.55
    expert_final_hold_close = 0.58
    expert_firm_hold_below_speed = 0.15
    # Hold the measured contact aperture. The passive palm/lip carries the
    # settled ball; tightening toward a preset after asymmetric contact builds
    # drive error and wedges the sphere back out.
    expert_hold_ramp_steps = 1.0e9
    expert_settle_steps = 100.0
    drop_after_latch_penalty = 350.0  # latch then drop must be a net loss
    hold_still_reward_scale = 2.0
    hold_action_penalty_scale = 0.35
    # Commanded-vs-actual finger closing error is a stable proxy for contact
    # force from the implicit drives.
    gentle_drive_error_tolerance = 0.035
    gentle_drive_error_max = 0.12
    gentle_drive_error_penalty_scale = 22.0
    gentle_underclose_penalty_scale = 14.0
    gentle_overclose_penalty_scale = 12.0
    post_latch_close_action_penalty_scale = 1.5
    gentle_hold_reward_scale = 4.0
    gentle_close_target = 0.52
    gentle_close_min = 0.38
    gentle_close_max = 0.72
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
    # Ten control steps is 1/6 s at 60 Hz. A two-frame brush is not a catch.
    grasp_hold_steps = 10
    # A retained catch may flicker briefly, but a sustained loss resets early.
    grasp_loss_steps = 6
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
    hold_reward_scale = 10.0
    drop_after_latch_penalty = 400.0
    end_hold_bonus = 350.0
    intercept_enclose_bonus = 0.0  # wrap = in-hand; throw phase re-enables
    spawn_in_hand_speed = 0.005
    spawn_in_hand_preclose = 0.65
    episode_length_s = 4.0  # shorter: less room to drop after a lucky latch


@configclass
class BallCatchThrowAEnvCfg(BallCatchEnvCfg):
    """Throw-A: close and reposition under an in-aperture drift ramp. Resume wrap."""

    training_phase: str = "throw_a"
    # Reach the unlocked drift range within ~750 PPO updates (16 steps/update).
    curriculum_steps = 12000
    in_hand_spawn_p: float | None = None
    spawn_in_hand_until = 0.70
    spawn_in_hand_preclose = 0.65  # match wrap so enclose transfers
    spawn_in_hand_speed = 0.008
    # Motion practice without starving wrap hold.
    # Positioning must dominate PPO batches; retain 20% in-hand rehearsal to
    # avoid forgetting the gentle full-timeout hold.
    throw_min_frac = 0.80
    throw_drift_until_alpha = 1.01
    throw_drift_speed_easy = (0.010, 0.030)
    throw_drift_speed_hard = (0.08, 0.20)
    throw_drift_along_easy = (0.75, 0.95)
    throw_drift_along_hard = (0.35, 0.65)
    # Moving-ball episodes must exercise arm positioning, not just closure
    # timing. The offset grows only after the easy retention gate is cleared.
    throw_drift_jitter_easy = 0.004
    throw_drift_jitter_hard = 0.045
    throw_drift_preclose = 0.35
    throw_hist_len = 256
    throw_hist_min = 48
    curriculum_unlock_catch = 0.25
    # Gate runs on asynchronous reset batches, not PPO iterations. Keep each
    # increment small enough that many early drops cannot skip difficulty bands.
    curriculum_advance_rate = 0.00010
    throw_leave_drift_ema = 0.40
    throw_gate_streak = 10
    intercept_enclose_bonus = 100.0
    vel_match_reward_scale = 8.0
    vel_match_dist = 0.20
    catch_reward_scale = 45.0
    hold_reward_scale = 10.0
    end_hold_bonus = 350.0
    drop_after_latch_penalty = 400.0
    success_close_min = 0.24
    success_speed_threshold = 1.55
    episode_length_s = 4.5


@configclass
class BallCatchThrowBEnvCfg(BallCatchEnvCfg):
    """Throw-B: free-arm near approaches → parabolas. Resume Throw-A."""

    training_phase: str = "throw_b"
    # A small amount of authority lets the hand cushion the first centered
    # lob. Training resets inherited arm outputs to zero, then learns motion
    # from the velocity/intercept rewards; authority grows with difficulty.
    arm_motion_scale_easy = 1.0
    arm_motion_scale_power = 2.0
    # Slightly slower than Throw-A because the drift-to-lob transition also
    # releases the full arm range and parabolic target variation.
    curriculum_steps = 12000
    in_hand_spawn_p: float | None = None
    spawn_in_hand_until = 0.35
    spawn_in_hand_preclose = 0.25
    spawn_in_hand_speed = 0.02
    throw_min_frac = 0.70
    # Brief ultra-near drift then free-arm lobs.
    throw_drift_until_alpha = 0.12
    throw_drift_speed_easy = (0.04, 0.10)
    throw_drift_speed_hard = (0.08, 0.18)
    throw_drift_along_easy = (0.45, 0.80)
    throw_drift_along_hard = (0.30, 0.55)
    throw_drift_jitter_easy = 0.008
    throw_drift_jitter_hard = 0.050
    throw_drift_preclose = 0.40
    # Bridge drift to a fully open deployed catch. Early lobs begin ready but
    # remain policy-driven; the reset preclose fades to zero at hard difficulty.
    throw_lob_preclose_easy = 0.25
    throw_lob_preclose_hard = 0.25
    throw_hist_len = 256
    throw_hist_min = 48
    # Promotion is exposure, not acceptance: deterministic lob evaluation is
    # the final gate. These values let a modest drift seed begin true lobs.
    curriculum_unlock_catch = 0.12
    curriculum_advance_rate = 0.00020
    throw_leave_drift_ema = 0.18
    throw_gate_streak = 5
    # Release outside the hand collision volume. The former 3--8 cm offset was
    # comparable to the ball diameter and caused reset depenetration, not a lob.
    throw_front_offset_easy = (0.11, 0.14)
    # Deployment envelope: genuine parabolas that remain inside the arm/cup's
    # validated reachable volume. The inherited generic hard endpoint (32 cm
    # deep, +/-14 cm lateral, 0.75 s) produced edge-on impacts rather than
    # meaningful catch attempts for this hand geometry.
    throw_front_offset = (0.12, 0.16)
    throw_side_offset = (-0.02, 0.02)
    throw_height_boost = (0.02, 0.04)
    # First true lobs must enter the open aperture cleanly. Lateral release
    # and target variation ramp later, when the arm has authority to correct.
    throw_side_offset_easy = (-0.002, 0.002)
    throw_height_boost_easy = (0.015, 0.04)
    # A 0.4--0.52 s arc rises 11--18 cm above the cup line under the matched
    # gravity scale and enters this fixed hand almost vertically. Begin with
    # a shallow but genuinely ballistic lob; longer/steeper arcs ramp in with
    # difficulty and arm authority.
    throw_flight_time_easy = (0.16, 0.24)
    throw_flight_time = (0.20, 0.28)
    # Catch near the distal hook of the fingers. A deep-palm intercept lets
    # the incoming ball ride the curved finger surfaces through the palm and
    # converts symmetric closing pressure into a high-speed sideways ejection.
    throw_cup_aim_fraction_easy = 1.00
    throw_cup_aim_fraction_hard = 1.00
    throw_aperture_circumcenter_blend_easy = 0.0
    throw_aperture_circumcenter_blend_hard = 0.0
    # Easy lobs pass close to the initial cup; hard lobs require a genuine,
    # reachable intercept instead of always entering a stationary hand.
    aim_jitter_xy_easy = 0.002
    aim_jitter_z_easy = 0.002
    aim_jitter_xy = 0.006
    aim_jitter_z = 0.006
    intercept_enclose_bonus = 140.0
    # The former velocity/progress terms dominated the sparse retained-catch
    # objective and rewarded aggressive arm oscillation.  Keep them as light
    # shaping while directly rewarding the proven cushion/close action.
    vel_match_reward_scale = 4.0
    vel_match_dist = 0.26
    vel_match_error_penalty_scale = 0.0
    intercept_progress_reward_scale = 15.0
    grip_timing_reward_scale = 3.0
    grip_timing_error_penalty_scale = 4.0
    expert_arm_action_penalty_scale = 35.0
    expert_grip_action_penalty_scale = 20.0
    catch_reward_scale = 60.0
    hold_reward_scale = 8.0
    end_hold_bonus = 350.0
    drop_after_latch_penalty = 400.0
    # Give brief but real lob enclosure a learning signal; accepted success is
    # still an end-of-episode gentle hold, never this transient latch alone.
    grasp_hold_steps = 2
    # Finger-tip body origins sit behind their collision surfaces. The former
    # 12.5 cm max-origin radius rejected a centered, symmetric three-finger
    # intercept (mean tip-center distance ~8.3 cm) before retention could be
    # learned. All along/radial/symmetry/speed gates remain strict.
    success_tip_max = 0.160
    success_close_min = 0.24
    success_speed_threshold = 1.85
    episode_length_s = 6.0


@configclass
class BallCatchThrowEnvCfg(BallCatchThrowBEnvCfg):
    """Legacy alias → Throw-B (prefer Throw-A then Throw-B gym IDs)."""

    pass
