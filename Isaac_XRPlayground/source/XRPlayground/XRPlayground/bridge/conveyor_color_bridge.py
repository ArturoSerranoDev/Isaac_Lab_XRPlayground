# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge adapter for ConveyorColorEnv (Unity ↔ Isaac)."""

from __future__ import annotations

from typing import Any

import torch

from .names_ur10e import (
    COLOR_NAMES,
    EE_BODY_NAME,
    TOPIC_OBJECTS_STATE,
    TOPIC_ROBOT_STATE,
    UR10E_JOINT_NAMES,
    UR10E_LINK_NAMES,
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


class ConveyorColorBridgeAdapter:
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
        missing_links: list[str] = []
        # Match against the live body list — find_bodies() raises if a name is absent.
        for name in UR10E_LINK_NAMES:
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

        if EE_BODY_NAME in body_names:
            try:
                ids, _ = robot.find_bodies([EE_BODY_NAME])
                if len(ids) >= 1:
                    self._ee_id = int(ids[0])
                    self._link_ids.setdefault(EE_BODY_NAME, self._ee_id)
            except ValueError:
                pass
        if self._ee_id is None and hasattr(self.env, "_ee_body_idx"):
            self._ee_id = int(self.env._ee_body_idx)

        joint_names = list(robot.joint_names)
        for name in UR10E_JOINT_NAMES:
            if name not in joint_names:
                print(f"[XR Conveyor Bridge] Warning: joint not found: {name}")
                continue
            try:
                jids, _ = robot.find_joints([name], preserve_order=True)
            except ValueError:
                print(f"[XR Conveyor Bridge] Warning: joint not found: {name}")
                continue
            if len(jids) == 1:
                self._joint_pairs.append((name, int(jids[0])))
            else:
                print(f"[XR Conveyor Bridge] Warning: joint not found: {name}")

        if missing_links:
            print(
                "[XR Conveyor Bridge] Optional / alternate link names not exact-matched "
                f"(OK if USD uses different Robotiq labels): {missing_links}"
            )
        print(
            f"[XR Conveyor Bridge] Resolved {len(self._link_ids)} links, "
            f"{len(self._joint_pairs)} joints."
        )

    def build_robot_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        robot = self.env.robot
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
        # Prefer body_pos_w (same index space as find_bodies); fall back to body_link_*.
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
                "joint_names": names,
                "joint_positions": positions,
                "ee": ee,
                "links": links,
                "target_color": int(self.env._target_color[i].item()),
                "target_color_name": COLOR_NAMES[int(self.env._target_color[i].item()) % len(COLOR_NAMES)],
            },
            frame_id="isaac_env",
            stamp_s=stamp_s,
        )

    def build_objects_state_envelope(self, stamp_s: float | None = None) -> dict[str, Any]:
        i = self.env_id
        origin = _as_tensor(self.env.scene.env_origins)[i]
        objects = []
        for slot, obj in enumerate(self.env.objects):
            active = bool(self.env._obj_active[i, slot].item())
            color = int(self.env._obj_color[i, slot].item())
            pos_w = _as_tensor(obj.data.root_pos_w)[i]
            quat = _as_tensor(obj.data.root_quat_w)[i]
            lin = _as_tensor(obj.data.root_lin_vel_w)[i]
            ang = _as_tensor(obj.data.root_ang_vel_w)[i]
            objects.append(
                {
                    "id": slot,
                    "active": active,
                    "color": color,
                    "color_name": COLOR_NAMES[color % len(COLOR_NAMES)],
                    "grasped": bool(self.env._obj_grasped[i, slot].item()),
                    **_pose_dict(pos_w - origin, quat),
                    "linear_velocity": [float(v) for v in lin.tolist()],
                    "angular_velocity": [float(v) for v in ang.tolist()],
                }
            )
        return make_envelope(
            TOPIC_OBJECTS_STATE,
            {
                "objects": objects,
                "target_color": int(self.env._target_color[i].item()),
                "source": "isaac",
            },
            frame_id="isaac_env",
            stamp_s=stamp_s,
        )

    def apply_spawn(self, data: dict[str, Any]) -> int | None:
        """Unity spawn request → activate a free object slot."""
        color = int(data.get("color", 0)) % 3
        pos = data.get("position")
        return self.env.spawn_object_external(self.env_id, color, pos)

    def set_auto_spawn(self, enabled: bool) -> None:
        self.env.auto_spawn_enabled = bool(enabled)

    def reset_hold(self) -> None:
        env = self.env
        i = self.env_id
        env_ids = torch.tensor([i], device=env.device, dtype=torch.long)
        env._reset_idx(env_ids)
