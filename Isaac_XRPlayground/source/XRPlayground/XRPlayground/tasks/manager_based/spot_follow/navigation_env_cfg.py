# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Spot pose-follow navigation over a pretrained Spot loco policy (Anymal-nav pattern)."""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.navigation.mdp as mdp
from XRPlayground.tasks.manager_based.spot_loco.flat_env_cfg import SpotLocoWalkEnvCfg

LOW_LEVEL_ENV_CFG = SpotLocoWalkEnvCfg()

# Default: latest exported loco under logs; override via env or train --agent overrides.
_DEFAULT_LOCO = (
    Path(__file__).resolve().parents[6]
    / "logs"
    / "rsl_rl"
    / "xrplayground_spot_loco"
    / "exported"
    / "policy.pt"
)


def _loco_policy_path() -> str:
    import os

    override = os.environ.get("XRPLAYGROUND_SPOT_LOCO_POLICY", "").strip()
    if override:
        return override
    if _DEFAULT_LOCO.is_file():
        return str(_DEFAULT_LOCO)
    # Fallback: Isaac Lab Anymal path shape — user must train Spot loco first.
    return str(_DEFAULT_LOCO)


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=_loco_policy_path(),
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Moving / resampled Pose2d goals for continuous follow training."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(4.0, 6.0),
        debug_vis=True,
        position_success_threshold=0.45,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-2.5, 2.5), pos_y=(-2.5, 2.5), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"), "threshold": 1.0},
    )


@configclass
class SpotFollowEnvCfg(ManagerBasedRLEnvCfg):
    """Follow a moving 2D pose using Spot loco as low-level skill."""

    sim: SimulationCfg = LOW_LEVEL_ENV_CFG.sim
    scene = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = max(self.commands.pose_command.resampling_time_range[1] * 2.0, 12.0)
        self.scene.num_envs = 64
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # Refresh loco path after train exports
        self.actions.pre_trained_policy_action.policy_path = _loco_policy_path()


@configclass
class SpotFollowEnvCfg_PLAY(SpotFollowEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
