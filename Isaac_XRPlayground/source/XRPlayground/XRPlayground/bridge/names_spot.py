# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Spot (Boston Dynamics) name tables and bridge topics.

Link / joint names match Nucleus `Robots/BostonDynamics/spot/spot.usd` prims.
"""

from __future__ import annotations

# Articulation body / Xform names under /spot (Unity SpotLinkMap expectedLinks).
SPOT_LINK_NAMES: list[str] = [
    "body",
    "fl_hip",
    "fl_uleg",
    "fl_lleg",
    "fl_foot",
    "fr_hip",
    "fr_uleg",
    "fr_lleg",
    "fr_foot",
    "hl_hip",
    "hl_uleg",
    "hl_lleg",
    "hl_foot",
    "hr_hip",
    "hr_uleg",
    "hr_lleg",
    "hr_foot",
]

# 12 actuated DOFs — order used by OfflineJointDriver.BindSpot / policy actions.
SPOT_JOINT_NAMES: list[str] = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]

# Default standing pose (matches isaaclab_assets.robots.spot.SPOT_CFG.init_state).
SPOT_DEFAULT_JOINT_POS: list[float] = [
    0.1,   # fl_hx
    -0.1,  # fr_hx
    0.1,   # hl_hx
    -0.1,  # hr_hx
    0.9,   # fl_hy
    0.9,   # fr_hy
    1.1,   # hl_hy
    1.1,   # hr_hy
    -1.5,  # fl_kn
    -1.5,  # fr_kn
    -1.5,  # hl_kn
    -1.5,  # hr_kn
]

BASE_BODY_NAME = "body"

TOPIC_ROBOT_STATE = "/xr/spot/robot_state"
TOPIC_PLAYER_POSE = "/xr/spot/player_pose"
TOPIC_HEARTBEAT = "/xr/heartbeat"
TOPIC_SESSION_COMMAND = "/xr/session_command"
TOPIC_SESSION_STATUS = "/xr/session_status"

MODE_MIRROR = "mirror"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9094

# Loco policy contract (Isaac-Velocity-Flat-Spot / XRPlayground Spot Loco).
LOCO_OBS_DIM = 48
LOCO_ACTION_DIM = 12
LOCO_ACTION_SCALE = 0.2
LOCO_DT = 1.0 / 50.0  # SpotFlat typically 0.02s physics * decimation; Unity uses controlDt

# Follow (nav) policy: lin_vel, gravity, pose_cmd → vel cmd for loco.
FOLLOW_OBS_DIM = 9
FOLLOW_ACTION_DIM = 3
FOLLOW_ACTION_SCALE = 1.0
FOLLOW_DT = 0.2  # loco step_dt (0.02) * high-level decimation 10
