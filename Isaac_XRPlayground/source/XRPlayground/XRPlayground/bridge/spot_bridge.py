# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge adapter for Spot loco / follow (Unity ↔ Isaac ManagerBasedRLEnv)."""

from __future__ import annotations

from typing import Any

import torch

from .names_spot import (
    BASE_BODY_NAME,
    SPOT_JOINT_NAMES,
    SPOT_LINK_NAMES,
    TOPIC_PLAYER_POSE,
    TOPIC_ROBOT_STATE,
)
from .protocol import make_envelope


def _as_tensor(data) -> torch.Tensor:
    return data.torch if hasattr(data, "torch") else data


def _pose_dict(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> dict[str, list[float]]:
    # Isaac body_quat_w is (w, x, y, z); Unity / bridge use xyzw.
    w, x, y, z = [float(v) for v in quat_wxyz.tolist()]
    return {
        "position": [float(v) for v in pos.tolist()],
        "orientation_xyzw": [x, y, z, w],
    }


class SpotBridgeAdapter:
    """Publish Spot link/joint state; track latest Unity HMD pose in Isaac env frame."""

    def __init__(self, env, env_id: int = 0):
        self.env = env
        self.env_id = int(env_id)
        self._link_ids: dict[str, int] = {}
        self._joint_pairs: list[tuple[str, int]] = []
        self._base_id: int | None = None
        self.player_pose_isaac: dict[str, list[float]] | None = None
        self._resolve_indices()

    def _robot(self):
        if hasattr(self.env, "robot"):
            return self.env.robot
        return self.env.scene["robot"]

    def _resolve_indices(self) -> None:
        robot = self._robot()
        body_names = list(robot.body_names)
        missing_links: list[str] = []
        for name in SPOT_LINK_NAMES:
            if name not in body_names:
                missing_links.append(name)
                continue
            try:
                ids, _ = robot.find_bodies([name])
            except ValueError:
                missing_links.append(name)
                continue
            if len(ids) == 1:
                self._link_ids[name] = int(ids[0])
            else:
                missing_links.append(name)

        if BASE_BODY_NAME in self._link_ids:
            self._base_id = self._link_ids[BASE_BODY_NAME]
        elif BASE_BODY_NAME in body_names:
            try:
                ids, _ = robot.find_bodies([BASE_BODY_NAME])
                if len(ids) >= 1:
                    self._base_id = int(ids[0])
                    self._link_ids.setdefault(BASE_BODY_NAME, self._base_id)
            except ValueError:
                pass

        joint_names = list(robot.joint_names)
        for name in SPOT_JOINT_NAMES:
            if name not in joint_names:
                print(f"[XR Spot Bridge] Warning: joint not found: {name}")
                continue
            try:
                jids, _ = robot.find_joints([name], preserve_order=True)
            except ValueError:
                print(f"[XR Spot Bridge] Warning: joint not found: {name}")
                continue
            if len(jids) == 1:
                self._joint_pairs.append((name, int(jids[0])))
            else:
                print(f"[XR Spot Bridge] Warning: joint not found: {name}")

        if missing_links:
            print(f"[XR Spot Bridge] Missing link names (check USD): {missing_links}")
        print(
            f"[XR Spot Bridge] Resolved {len(self._link_ids)}/{len(SPOT_LINK_NAMES)} links, "
            f"{len(self._joint_pairs)}/{len(SPOT_JOINT_NAMES)} joints."
        )

    def handle_player_pose(self, data: dict[str, Any]) -> None:
        pos = data.get("position")
        quat = data.get("orientation_xyzw")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            return
        if not isinstance(quat, (list, tuple)) or len(quat) < 4:
            quat = [0.0, 0.0, 0.0, 1.0]
        self.player_pose_isaac = {
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "orientation_xyzw": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
        }

    def build_robot_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        robot = self._robot()
        i = self.env_id
        origin = _as_tensor(self.env.scene.env_origins)[i]
        joint_pos = _as_tensor(robot.data.joint_pos)[i]
        names: list[str] = []
        positions: list[float] = []
        for name, jid in self._joint_pairs:
            if jid < 0 or jid >= joint_pos.shape[0]:
                continue
            names.append(name)
            positions.append(float(joint_pos[jid].item()))

        data = robot.data
        if hasattr(data, "body_pos_w"):
            body_pos = _as_tensor(data.body_pos_w)[i]
            body_quat = _as_tensor(data.body_quat_w)[i]
        else:
            body_pos = _as_tensor(data.body_link_pos_w)[i]
            body_quat = _as_tensor(data.body_link_quat_w)[i]

        n_bodies = int(body_pos.shape[0])
        links = []
        for name, bid in self._link_ids.items():
            if bid < 0 or bid >= n_bodies:
                continue
            links.append({"name": name, **_pose_dict(body_pos[bid] - origin, body_quat[bid])})

        if self._base_id is not None and 0 <= self._base_id < n_bodies:
            ee = _pose_dict(body_pos[self._base_id] - origin, body_quat[self._base_id])
        elif links:
            ee = {k: links[0][k] for k in ("position", "orientation_xyzw")}
        else:
            ee = {"position": [0.0, 0.0, 0.5], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}

        payload: dict[str, Any] = {
            "joint_names": names,
            "joint_positions": positions,
            "ee": ee,
            "links": links,
        }
        if self.player_pose_isaac is not None:
            payload["player"] = self.player_pose_isaac

        return make_envelope(TOPIC_ROBOT_STATE, payload, frame_id="isaac_env", stamp_s=stamp_s)

    def build_player_ack_envelope(self, stamp_s: float | None = None) -> dict[str, Any] | None:
        if self.player_pose_isaac is None:
            return None
        return make_envelope(
            TOPIC_PLAYER_POSE,
            {**self.player_pose_isaac, "source": "isaac_echo"},
            frame_id="isaac_env",
            stamp_s=stamp_s,
        )
