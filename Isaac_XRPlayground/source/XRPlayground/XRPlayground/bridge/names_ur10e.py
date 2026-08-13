# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + Robotiq name tables and conveyor bridge topics."""

from __future__ import annotations

# Isaac Lab UR10e + Robotiq_2f_85 body names (must match Unity RobotLinkMap).
UR10E_ARM_LINK_NAMES: list[str] = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]

# Robotiq 2F-85 links commonly present on the Nucleus ur10e Gripper=Robotiq_2f_85 variant.
UR10E_GRIPPER_LINK_NAMES: list[str] = [
    "robotiq_base_link",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_knuckle",
]

UR10E_LINK_NAMES: list[str] = UR10E_ARM_LINK_NAMES + UR10E_GRIPPER_LINK_NAMES

UR10E_ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR10E_GRIPPER_JOINT_NAMES: list[str] = [
    "finger_joint",
]

UR10E_JOINT_NAMES: list[str] = UR10E_ARM_JOINT_NAMES + UR10E_GRIPPER_JOINT_NAMES

EE_BODY_NAME = "wrist_3_link"

COLOR_NAMES = ("red", "green", "blue")

# Topics (namespaced so Ball Catch on :9090 is untouched)
TOPIC_ROBOT_STATE = "/xr/conveyor/robot_state"
TOPIC_OBJECTS_STATE = "/xr/conveyor/objects_state"
TOPIC_SPAWN = "/xr/conveyor/spawn"
TOPIC_HEARTBEAT = "/xr/heartbeat"
TOPIC_SESSION_COMMAND = "/xr/session_command"
TOPIC_SESSION_STATUS = "/xr/session_status"

MODE_MIRROR = "mirror"
MODE_AWAIT_SPAWN = "await_spawn"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9091
