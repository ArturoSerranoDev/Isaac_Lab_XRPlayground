# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge adapter for PickPlaceTableEnv (Unity ↔ Isaac)."""

from __future__ import annotations

from typing import Any

import torch

from .names_agibot import (
    AGIBOT_JOINT_NAMES,
    AGIBOT_LINK_NAMES,
    COLOR_NAMES,
    EE_BODY_NAME,
    TOPIC_OBJECTS_STATE,
    TOPIC_ROBOT_STATE,
)
from .protocol import make_envelope


def _as_tensor(data) -> torch.Tensor:
    return data.torch if hasattr(data, "torch") else data


def _pose_dict(pos: torch.Tensor, quat_xyzw: torch.Tensor) -> dict[str, list[float]]:
    x, y, z, w = [float(v) for v in quat_xyzw.tolist()]
    return {
        "position": [float(v) for v in pos.tolist()],
        "orientation_xyzw": [x, y, z, w],
    }


class PickPlaceBridgeAdapter:
    def __init__(self, env, env_id: int = 0):
        self.env = env
        self.env_id = int(env_id)
        self._link_ids: dict[str, int] = {}
        self._joint_pairs: list[tuple[str, int]] = []
        self._ee_id: int | None = None
        self._resolve_indices()

    def _resolve_indices(self) -> None:
        robot = self.env.robot
        body_names = list(robot.body_names)
        for name in AGIBOT_LINK_NAMES:
            if name not in body_names:
                continue
            try:
                ids, _ = robot.find_bodies([name])
            except ValueError:
                continue
            if len(ids) == 1:
                self._link_ids[name] = int(ids[0])

        if EE_BODY_NAME in body_names:
            try:
                ids, _ = robot.find_bodies([EE_BODY_NAME])
                if len(ids) >= 1:
                    self._ee_id = int(ids[0])
            except ValueError:
                pass
        if self._ee_id is None and hasattr(self.env, "_ee_body_idx"):
            self._ee_id = int(self.env._ee_body_idx)

        joint_names = list(robot.joint_names)
        for name in AGIBOT_JOINT_NAMES:
            if name not in joint_names:
                continue
            try:
                jids, _ = robot.find_joints([name], preserve_order=True)
            except ValueError:
                continue
            if len(jids) == 1:
                self._joint_pairs.append((name, int(jids[0])))

        print(
            f"[XR PickPlace Bridge] Resolved {len(self._link_ids)} links, {len(self._joint_pairs)} joints."
        )

    def build_robot_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        robot = self.env.robot
        i = self.env_id
        origin = _as_tensor(self.env.scene.env_origins)[i]
        joint_pos = _as_tensor(robot.data.joint_pos)[i]
        joint_vel = _as_tensor(robot.data.joint_vel)[i]
        names: list[str] = []
        positions: list[float] = []
        velocities: list[float] = []
        for name, jid in self._joint_pairs:
            if jid < 0 or jid >= joint_pos.shape[0]:
                continue
            names.append(name)
            positions.append(float(joint_pos[jid].item()))
            velocities.append(float(joint_vel[jid].item()))

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

        if self._ee_id is not None and 0 <= self._ee_id < n_bodies:
            ee = _pose_dict(body_pos[self._ee_id] - origin, body_quat[self._ee_id])
        elif links:
            ee = {k: links[-1][k] for k in ("position", "orientation_xyzw")}
        else:
            ee = {"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}

        return make_envelope(
            TOPIC_ROBOT_STATE,
            {
                "joints": [
                    {"name": name, "position": position, "velocity": velocity}
                    for name, position, velocity in zip(names, positions, velocities, strict=True)
                ],
                "ee": ee,
                "links": links,
            },
            station_id="pick_place_table",
            frame_id="isaac_env",
            sim_time_s=stamp_s,
        )

    def build_objects_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        i = self.env_id
        origin = _as_tensor(self.env.scene.env_origins)[i]
        objects = []
        for slot, obj in enumerate(self.env.pieces):
            active = bool(self.env._piece_active[i, slot].item())
            color = int(self.env._piece_color[i, slot].item())
            pos_w = _as_tensor(obj.data.root_pos_w)[i]
            quat = _as_tensor(obj.data.root_quat_w)[i]
            lin = _as_tensor(obj.data.root_lin_vel_w)[i]
            objects.append(
                {
                    "id": f"piece_{slot}",
                    "active": active,
                    "color": color if active else -1,
                    "color_name": COLOR_NAMES[color % len(COLOR_NAMES)] if active else "none",
                    "grasped": bool(self.env._piece_grasped[i, slot].item()),
                    "position": [float(v) for v in (pos_w - origin).tolist()],
                    "orientation_xyzw": [float(v) for v in quat.tolist()],
                    "linear_velocity": [float(v) for v in lin.tolist()],
                    "angular_velocity": [0.0, 0.0, 0.0],
                }
            )
        return make_envelope(
            TOPIC_OBJECTS_STATE,
            {"objects": objects, "source": "isaac"},
            station_id="pick_place_table",
            frame_id="isaac_env",
            sim_time_s=stamp_s,
        )

    def handle_spawn(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        color = payload.get("color_id", payload.get("color"))
        pos = payload.get("position")
        slot = self.env.spawn_piece_external(self.env_id, color=color, position_xyz=pos)
        if slot is None:
            return None
        return {"slot": slot, "ok": True}
