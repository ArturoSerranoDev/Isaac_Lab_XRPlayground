# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Balance Bot (2-DOF tray) name tables and bridge topics."""

from __future__ import annotations

# Must match Unity BalanceBotLinkMap / OfflineJointDriver.BindBalanceBot / Isaac cfg.
BALANCE_BOT_LINK_NAMES: list[str] = [
    "base_link",
    "roll_link",
    "tray_link",
]

BALANCE_BOT_JOINT_NAMES: list[str] = [
    "roll_joint",
    "pitch_joint",
]

EE_BODY_NAME = "tray_link"

TOPIC_ROBOT_STATE = "/xr/balance_bot/robot_state"
TOPIC_BALLS_STATE = "/xr/balance_bot/balls_state"
TOPIC_HEARTBEAT = "/xr/heartbeat"
TOPIC_SESSION_COMMAND = "/xr/session_command"
TOPIC_SESSION_STATUS = "/xr/session_status"

MODE_MIRROR = "mirror"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9093
