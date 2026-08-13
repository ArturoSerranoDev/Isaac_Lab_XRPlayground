# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Frame helpers (Isaac Z-up ↔ Unity Y-up). Payloads on the wire stay Isaac Z-up."""

from __future__ import annotations

from typing import Sequence


def isaac_pos_to_unity(p: Sequence[float]) -> list[float]:
    """Isaac (x, y, z) Z-up → Unity (x, z, y) Y-up."""
    return [float(p[0]), float(p[2]), float(p[1])]


def unity_pos_to_isaac(p: Sequence[float]) -> list[float]:
    """Unity (x, y, z) Y-up → Isaac (x, z, y) Z-up."""
    return [float(p[0]), float(p[2]), float(p[1])]


def isaac_quat_xyzw_to_unity(q: Sequence[float]) -> list[float]:
    """Quaternion xyzw Isaac Z-up → Unity Y-up."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [x, z, y, -w]


def unity_quat_xyzw_to_isaac(q: Sequence[float]) -> list[float]:
    """Quaternion xyzw Unity Y-up → Isaac Z-up."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [x, z, y, -w]
