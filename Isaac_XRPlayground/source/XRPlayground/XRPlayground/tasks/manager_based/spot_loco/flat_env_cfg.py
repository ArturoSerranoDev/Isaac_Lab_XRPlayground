# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""XRPlayground Spot locomotion: stand → walk curriculum over Isaac Spot flat velocity."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotCommandsCfg,
    SpotFlatEnvCfg,
    SpotFlatEnvCfg_PLAY,
)


@configclass
class SpotLocoStandCommandsCfg(SpotCommandsCfg):
    """Phase A: mostly standing, tiny velocity commands."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.85,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 0.35), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.4, 0.4)
        ),
    )


@configclass
class SpotLocoWalkCommandsCfg(SpotCommandsCfg):
    """Phase B: full workspace velocity track (within Spot flat ranges)."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.5, 1.5)
        ),
    )


@configclass
class SpotLocoStandEnvCfg(SpotFlatEnvCfg):
    """Stand-upright curriculum (Phase A)."""

    commands: SpotLocoStandCommandsCfg = SpotLocoStandCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128


@configclass
class SpotLocoWalkEnvCfg(SpotFlatEnvCfg):
    """Walk / velocity track curriculum (Phase B)."""

    commands: SpotLocoWalkCommandsCfg = SpotLocoWalkCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128


@configclass
class SpotLocoWalkEnvCfg_PLAY(SpotFlatEnvCfg_PLAY):
    commands: SpotLocoWalkCommandsCfg = SpotLocoWalkCommandsCfg()
