# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Read/write helpers for BallCatchEnv used by the XR bridge (no sockets here)."""

from __future__ import annotations

from typing import Any

import torch

from .names import (
    EE_BODY_NAME,
    KINOVA_JOINT_NAMES,
    KINOVA_LINK_NAMES,
    TOPIC_ROBOT_STATE,
)
from .protocol import make_envelope


def _as_tensor(data) -> torch.Tensor:
    return data.torch if hasattr(data, "torch") else data


def _pose_dict(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> dict[str, list[float]]:
    # Isaac Lab stores quaternion as (w, x, y, z); wire format is xyzw.
    w, x, y, z = [float(v) for v in quat_wxyz.tolist()]
    return {
        "position": [float(v) for v in pos.tolist()],
        "orientation_xyzw": [x, y, z, w],
    }


class BallCatchBridgeAdapter:
    """Thin adapter over a live BallCatchEnv instance (env index 0 by default)."""

    def __init__(self, env, env_id: int = 0):
        self.env = env
        self.env_id = int(env_id)
        self._link_ids: dict[str, int] = {}
        self._joint_pairs: list[tuple[str, int]] = []
        self._ee_id: int | None = None
        self._resolve_indices()

    def _resolve_indices(self) -> None:
        robot = self.env.robot
        for name in KINOVA_LINK_NAMES:
            ids, found = robot.find_bodies([name])
            if len(ids) == 1:
                self._link_ids[name] = int(ids[0])
        ids, found = robot.find_bodies([EE_BODY_NAME])
        if len(ids) >= 1:
            self._ee_id = int(ids[0])
            self._link_ids.setdefault(EE_BODY_NAME, self._ee_id)

        for name in KINOVA_JOINT_NAMES:
            jids, _ = robot.find_joints([name], preserve_order=True)
            if len(jids) == 1:
                self._joint_pairs.append((name, int(jids[0])))
            else:
                print(f"[XR Bridge] Warning: joint not found: {name}")

    def build_robot_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        robot = self.env.robot
        i = self.env_id
        origin = _as_tensor(self.env.scene.env_origins)[i]

        joint_pos = _as_tensor(robot.data.joint_pos)[i]
        names: list[str] = []
        positions: list[float] = []
        for name, jid in self._joint_pairs:
            names.append(name)
            positions.append(float(joint_pos[jid].item()))

        body_pos = _as_tensor(robot.data.body_pos_w)[i]
        body_quat = _as_tensor(robot.data.body_quat_w)[i]

        links: list[dict[str, Any]] = []
        for name, bid in self._link_ids.items():
            # env-local position for Unity rootOffset alignment
            pos_local = body_pos[bid] - origin
            links.append({"name": name, **_pose_dict(pos_local, body_quat[bid])})

        ee: dict[str, Any]
        if self._ee_id is not None:
            ee = _pose_dict(body_pos[self._ee_id] - origin, body_quat[self._ee_id])
        elif links:
            ee = {k: links[-1][k] for k in ("position", "orientation_xyzw")}
        else:
            ee = {"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}

        return make_envelope(
            TOPIC_ROBOT_STATE,
            {
                "joint_names": names,
                "joint_positions": positions,
                "ee": ee,
                "links": links,
            },
            frame_id="isaac_env",
            stamp_s=stamp_s,
        )

    def apply_ball_state(self, data: dict[str, Any]) -> None:
        """Write Unity-provided ball pose/vel into Isaac (env-local → world)."""
        ball = self.env.ball
        i = self.env_id
        device = self.env.device
        origin = _as_tensor(self.env.scene.env_origins)[i]

        pos = data.get("position", [0.0, 0.0, 0.5])
        quat_xyzw = data.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
        lin = data.get("linear_velocity", [0.0, 0.0, 0.0])
        ang = data.get("angular_velocity", [0.0, 0.0, 0.0])

        x, y, z, w = [float(v) for v in quat_xyzw]
        quat_wxyz = torch.tensor([[w, x, y, z]], device=device, dtype=torch.float32)
        pos_w = torch.tensor([[float(pos[0]), float(pos[1]), float(pos[2])]], device=device, dtype=torch.float32)
        pos_w = pos_w + origin.unsqueeze(0)

        pose = torch.cat((pos_w, quat_wxyz), dim=-1)
        vel = torch.tensor(
            [[float(lin[0]), float(lin[1]), float(lin[2]), float(ang[0]), float(ang[1]), float(ang[2])]],
            device=device,
            dtype=torch.float32,
        )
        env_ids = torch.tensor([i], device=device, dtype=torch.long)
        ball.write_root_pose_to_sim(pose, env_ids)
        ball.write_root_velocity_to_sim(vel, env_ids)

    def park_ball(self) -> None:
        """Park ball out of the workspace while waiting for a player throw."""
        self.apply_ball_state(
            {
                "position": [1.15, 0.55, 0.12],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "grasped": False,
            }
        )

    def reset_robot_hold(self) -> None:
        """Clear grasp flags and re-hold default joint targets for waiting pose."""
        i = self.env_id
        env = self.env
        if hasattr(env, "_episode_caught"):
            env._episode_caught[i] = False
            env._just_caught[i] = False
            env._grasp_hold_count[i] = 0
            env._body_fail[i] = False
            env._dropped[i] = False
        joint_pos = _as_tensor(env.robot.data.default_joint_pos)[i : i + 1].clone()
        joint_vel = torch.zeros_like(joint_pos)
        env_ids = torch.tensor([i], device=env.device, dtype=torch.long)
        env.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env.robot_dof_targets[i] = joint_pos[0]
        self.park_ball()
