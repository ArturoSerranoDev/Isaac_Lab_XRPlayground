# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Kinova name tables for Unity ↔ Isaac bridge."""

from __future__ import annotations

# Order matches ball_catch_env_cfg arm + gripper joint lists.
KINOVA_ARM_JOINT_NAMES: list[str] = [
    "j2n7s300_joint_1",
    "j2n7s300_joint_2",
    "j2n7s300_joint_3",
    "j2n7s300_joint_4",
    "j2n7s300_joint_5",
    "j2n7s300_joint_6",
    "j2n7s300_joint_7",
]

KINOVA_GRIPPER_JOINT_NAMES: list[str] = [
    "j2n7s300_joint_finger_1",
    "j2n7s300_joint_finger_2",
    "j2n7s300_joint_finger_3",
    "j2n7s300_joint_finger_tip_1",
    "j2n7s300_joint_finger_tip_2",
    "j2n7s300_joint_finger_tip_3",
]

KINOVA_JOINT_NAMES: list[str] = KINOVA_ARM_JOINT_NAMES + KINOVA_GRIPPER_JOINT_NAMES

# Flat Unity USD link names (must exist under Kinova_Jaco2_j2n7s300).
KINOVA_LINK_NAMES: list[str] = [
    "j2n7s300_link_base",
    "j2n7s300_link_1",
    "j2n7s300_link_2",
    "j2n7s300_link_3",
    "j2n7s300_link_4",
    "j2n7s300_link_5",
    "j2n7s300_link_6",
    "j2n7s300_link_7",
    "j2n7s300_end_effector",
    "j2n7s300_link_finger_1",
    "j2n7s300_link_finger_2",
    "j2n7s300_link_finger_3",
    "j2n7s300_link_finger_tip_1",
    "j2n7s300_link_finger_tip_2",
    "j2n7s300_link_finger_tip_3",
]

EE_BODY_NAME = "j2n7s300_end_effector"

TOPIC_BALL_STATE = "/xr/ball_state"
TOPIC_ROBOT_STATE = "/xr/robot_state"
TOPIC_HEARTBEAT = "/xr/heartbeat"
TOPIC_SESSION_COMMAND = "/xr/session_command"
TOPIC_SESSION_STATUS = "/xr/session_status"

# Session modes (Unity UI ↔ Isaac runner)
MODE_MIRROR = "mirror"
MODE_AWAIT_THROW = "await_throw"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9090
