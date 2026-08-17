# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge adapter for BalanceBotEnv (Unity ↔ Isaac)."""

from __future__ import annotations

from typing import Any

import torch

from .names_balance_bot import (
    BALANCE_BOT_JOINT_NAMES,
    BALANCE_BOT_LINK_NAMES,
    EE_BODY_NAME,
    TOPIC_BALLS_STATE,
    TOPIC_ROBOT_STATE,
)
from .protocol import make_envelope


def _as_tensor(data) -> torch.Tensor:
    return data.torch if hasattr(data, "torch") else data


class BalanceBotBridgeAdapter:
    def __init__(self, env, env_id: int = 0):
        self.env = env
        self.env_id = int(env_id)
        print(
            f"[XR BalanceBot Bridge] Links={BALANCE_BOT_LINK_NAMES} "
            f"joints={BALANCE_BOT_JOINT_NAMES} env_id={self.env_id}"
        )

    def build_robot_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        i = self.env_id
        names, positions = self.env.get_joint_state(i)
        links = self.env.get_link_poses_env_local(i)
        ee = next((lnk for lnk in links if lnk.get("name") == EE_BODY_NAME), None)
        if ee is None and links:
            ee = {k: links[-1][k] for k in ("position", "orientation_xyzw")}
        elif ee is None:
            ee = {"position": [0.0, 0.0, 0.75], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        else:
            ee = {"position": ee["position"], "orientation_xyzw": ee["orientation_xyzw"]}

        return make_envelope(
            TOPIC_ROBOT_STATE,
            {
                "joints": [
                    {"name": name, "position": position, "velocity": 0.0}
                    for name, position in zip(names, positions, strict=True)
                ],
                "ee": ee,
                "links": links,
                "n_balls": int(self.env._n_balls[i].item()),
                "curriculum_stage": int(self.env._curriculum_stage()),
            },
            station_id="balance_bot",
            frame_id="isaac_env",
            sim_time_s=stamp_s,
        )

    def build_balls_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        i = self.env_id
        origin = _as_tensor(self.env.scene.env_origins)[i]
        balls = []
        for slot, ball in enumerate(self.env.balls):
            active = bool(self.env._ball_active[i, slot].item())
            pos = _as_tensor(ball.data.root_pos_w)[i] - origin
            quat = _as_tensor(ball.data.root_quat_w)[i]
            vel = _as_tensor(ball.data.root_lin_vel_w)[i]
            ang = _as_tensor(ball.data.root_ang_vel_w)[i]
            x, y, z, w = [float(v) for v in quat.tolist()]
            balls.append(
                {
                    "id": f"ball_{slot}",
                    "active": active,
                    "position": [float(v) for v in pos.tolist()],
                    "orientation_xyzw": [x, y, z, w],
                    "linear_velocity": [float(v) for v in vel.tolist()],
                    "angular_velocity": [float(v) for v in ang.tolist()],
                }
            )
        return make_envelope(
            TOPIC_BALLS_STATE,
            {
                "objects": balls,
                "n_balls": int(self.env._n_balls[i].item()),
                "curriculum_stage": int(self.env._curriculum_stage()),
                "source": "isaac",
            },
            station_id="balance_bot",
            frame_id="isaac_env",
            sim_time_s=stamp_s,
        )
