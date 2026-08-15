# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Agibot A2D name tables and pick-place bridge topics."""

from __future__ import annotations

# Bodies streamed to Unity (right arm + gripper + torso anchor).
# Names match IsaacLab Agibot A2D USD / URDF (verify after USD import with Rebuild Link Map).
AGIBOT_TORSO_LINK_NAMES: list[str] = [
    "base_link",
    "body_link",
    "head_link",
]

AGIBOT_RIGHT_ARM_LINK_NAMES: list[str] = [
    "right_arm_link1",
    "right_arm_link2",
    "right_arm_link3",
    "right_arm_link4",
    "right_arm_link5",
    "right_arm_link6",
    "right_arm_link7",
]

AGIBOT_RIGHT_GRIPPER_LINK_NAMES: list[str] = [
    "right_gripper_base",
    "right_gripper_center",
    "right_Left_Pad_Link",
    "right_Right_Pad_Link",
]

AGIBOT_LINK_NAMES: list[str] = (
    AGIBOT_TORSO_LINK_NAMES + AGIBOT_RIGHT_ARM_LINK_NAMES + AGIBOT_RIGHT_GRIPPER_LINK_NAMES
)

AGIBOT_RIGHT_ARM_JOINT_NAMES: list[str] = [f"right_arm_joint{i}" for i in range(1, 8)]
AGIBOT_GRIPPER_JOINT_NAMES: list[str] = ["right_hand_joint1"]
AGIBOT_JOINT_NAMES: list[str] = AGIBOT_RIGHT_ARM_JOINT_NAMES + AGIBOT_GRIPPER_JOINT_NAMES

EE_BODY_NAME = "right_gripper_center"

COLOR_NAMES = ("red", "green", "blue", "yellow")

TOPIC_ROBOT_STATE = "/xr/pick_place/robot_state"
TOPIC_OBJECTS_STATE = "/xr/pick_place/objects_state"
TOPIC_SPAWN = "/xr/pick_place/spawn"
TOPIC_DEMO_RECORD = "/xr/pick_place/demo_record"
TOPIC_HEARTBEAT = "/xr/heartbeat"
TOPIC_SESSION_COMMAND = "/xr/session_command"
TOPIC_SESSION_STATUS = "/xr/session_status"

MODE_MIRROR = "mirror"
MODE_AWAIT_SPAWN = "await_spawn"
MODE_RECORD_DEMO = "record_demo"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9092
