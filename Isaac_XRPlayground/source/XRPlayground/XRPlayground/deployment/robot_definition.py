# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export Unity robot physics from the live Isaac articulation, not a hand-written sidecar."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROBOT_DEFINITION_SCHEMA_VERSION = 6


class RobotDefinitionError(ValueError):
    """Raised when live Isaac physics cannot be exported without guessing."""


def _values(value: Any) -> Any:
    if hasattr(value, "torch"):
        value = value.torch() if callable(value.torch) else value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _row(value: Any, index: int) -> Any:
    result = _values(value)
    if not isinstance(result, list) or not result:
        raise RobotDefinitionError("Isaac physics view returned unreadable or empty data")
    if index >= len(result):
        raise RobotDefinitionError(f"Isaac environment index {index} is outside the physics view")
    return result[index]


def _finite_vector(value: Any, count: int, label: str) -> list[float]:
    values = _values(value)
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise RobotDefinitionError(f"{label} must contain {count} values")
    result = [float(item) for item in values]
    if not all(math.isfinite(item) for item in result):
        raise RobotDefinitionError(f"{label} contains a non-finite value")
    return result


def _position_to_unity(value: Any) -> list[float]:
    x, y, z = _finite_vector(value, 3, "position")
    return [x, z, y]


def _quat_xyzw_to_unity(value: Any) -> list[float]:
    x, y, z, w = _finite_vector(value, 4, "quaternion")
    return [x, z, y, -w]


def _axis_to_unity(axis: str) -> list[float]:
    source = {
        "X": [1.0, 0.0, 0.0],
        "Y": [0.0, 1.0, 0.0],
        "Z": [0.0, 0.0, 1.0],
    }.get(str(axis).upper())
    if source is None:
        raise RobotDefinitionError(f"Unsupported USD joint axis '{axis}'")
    # Joint axes are axial vectors.  The Isaac -> Unity Y/Z swap changes
    # handedness, so an axis needs the determinant sign in addition to the
    # ordinary position-vector conversion.  Without this negation every
    # revolute joint turns in the opposite physical direction in Unity even
    # though its scalar joint coordinate appears numerically correct.
    return [-value for value in _position_to_unity(source)]


def _matrix_to_quaternion(matrix: Any) -> list[float]:
    """Return a normalized Unity xyzw quaternion from a proper 3x3 rotation."""
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = [(m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale]
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        result = [0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale]
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        result = [(m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale]
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        result = [(m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale]
    length = math.sqrt(sum(item * item for item in result))
    if length <= 1.0e-12:
        raise RobotDefinitionError("Could not diagonalize an inertia tensor")
    return [item / length for item in result]


def _principal_inertia(flat: Any) -> tuple[list[float], list[float]]:
    """Convert Isaac actor-frame inertia to Unity coordinates and principal axes."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RobotDefinitionError("numpy is required to export inertia tensors") from exc

    values = np.asarray(_values(flat), dtype=np.float64)
    if values.shape == (9,):
        matrix = values.reshape(3, 3)
    elif values.shape == (3, 3):
        matrix = values
    else:
        raise RobotDefinitionError(f"Unexpected Isaac inertia shape {values.shape}")
    mapping = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    unity_matrix = mapping @ (0.5 * (matrix + matrix.T)) @ mapping.T
    eigenvalues, eigenvectors = np.linalg.eigh(unity_matrix)
    if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvalues)):
        raise RobotDefinitionError(f"Inertia tensor is not positive definite: {eigenvalues.tolist()}")
    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, 2] *= -1.0
    return eigenvalues.tolist(), _matrix_to_quaternion(eigenvectors.tolist())


def _source_asset(robot: Any) -> str:
    spawn = getattr(getattr(robot, "cfg", None), "spawn", None)
    return str(getattr(spawn, "usd_path", "") or getattr(spawn, "asset_path", "") or "")


def _source_hash(source: str) -> str:
    path = Path(source)
    if not source or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _definition_hash(
    robot_id: str,
    source_asset: str,
    links: list[dict[str, Any]],
    joints: list[dict[str, Any]],
    fixed_joints: list[dict[str, Any]],
    auxiliary_joints: list[dict[str, Any]] | None = None,
) -> str:
    value = {
        "robot_id": robot_id,
        "source_asset": source_asset,
        "links": links,
        "joints": joints,
        "fixed_joints": fixed_joints,
    }
    # Preserve the canonical hashes of definitions exported before auxiliary
    # constraints were introduced. An empty list carries no additional physics.
    if auxiliary_joints:
        value["auxiliary_joints"] = auxiliary_joints

    def require_finite(item: Any, path: str) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise RobotDefinitionError(f"Robot definition contains {item} at {path}")
        if isinstance(item, dict):
            for key, child in item.items():
                require_finite(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                require_finite(child, f"{path}[{index}]")

    require_finite(value, "definition")
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _actuator_joint_properties(
    robot: Any,
    joint_names: list[str],
    env_index: int,
    physx_stiffnesses: list[Any],
    physx_dampings: list[Any],
    physx_force_limits: list[Any],
    physx_velocity_limits: list[Any],
) -> list[dict[str, Any]]:
    """Resolve Isaac actuator values which may intentionally differ from PhysX drives."""
    properties = [
        {
            "actuator_group": "",
            "actuator_model": "PhysXDrive",
            "actuator_is_implicit": True,
            "actuator_delay_min_steps": 0,
            "actuator_delay_max_steps": 0,
            "actuator_nominal_delay_steps": 0,
            "effort_limit_curve": [],
            "stiffness": float(physx_stiffnesses[index]),
            "damping": float(physx_dampings[index]),
            "force_limit": float(physx_force_limits[index]),
            "max_velocity": float(physx_velocity_limits[index]),
        }
        for index in range(len(joint_names))
    ]
    actuator_groups = getattr(robot, "actuators", None)
    if not actuator_groups:
        return properties
    joint_indices = {name: index for index, name in enumerate(joint_names)}
    assigned: set[str] = set()
    for group_name, actuator in actuator_groups.items():
        names = [str(item) for item in actuator.joint_names]
        stiffnesses = _row(actuator.stiffness, env_index)
        dampings = _row(actuator.damping, env_index)
        effort_limits = _row(actuator.effort_limit, env_index)
        velocity_limits = _row(actuator.velocity_limit, env_index)
        if not all(
            len(values) == len(names)
            for values in (stiffnesses, dampings, effort_limits, velocity_limits)
        ):
            raise RobotDefinitionError(
                f"Actuator group '{group_name}' property counts do not match its joint names"
            )
        cfg = getattr(actuator, "cfg", None)
        delay_min = int(getattr(cfg, "min_delay", 0) or 0)
        delay_max = int(getattr(cfg, "max_delay", 0) or 0)
        if delay_min < 0 or delay_max < delay_min:
            raise RobotDefinitionError(f"Actuator group '{group_name}' has an invalid delay range")
        lookup = getattr(cfg, "joint_parameter_lookup", None) or []
        effort_curve = [
            {
                "position": float(row[0]),
                "max_effort": float(row[2]),
            }
            for row in lookup
        ]
        curve_force_limit = max(
            (sample["max_effort"] for sample in effort_curve), default=-1.0
        )
        model_name = actuator.__class__.__name__
        is_implicit = bool(getattr(actuator, "is_implicit_model", False))
        for local_index, name in enumerate(names):
            if name not in joint_indices:
                raise RobotDefinitionError(
                    f"Actuator group '{group_name}' references unknown joint '{name}'"
                )
            if name in assigned:
                raise RobotDefinitionError(f"Joint '{name}' belongs to multiple actuator groups")
            assigned.add(name)
            index = joint_indices[name]
            force_limit = (
                curve_force_limit if effort_curve else float(effort_limits[local_index])
            )
            if not math.isfinite(force_limit) or force_limit <= 0.0:
                force_limit = float(physx_force_limits[index])
            max_velocity = float(velocity_limits[local_index])
            if not math.isfinite(max_velocity) or max_velocity <= 0.0:
                max_velocity = float(physx_velocity_limits[index])
            properties[index] = {
                "actuator_group": str(group_name),
                "actuator_model": model_name,
                "actuator_is_implicit": is_implicit,
                "actuator_delay_min_steps": delay_min,
                "actuator_delay_max_steps": delay_max,
                "actuator_nominal_delay_steps": (delay_min + delay_max) // 2,
                "effort_limit_curve": effort_curve,
                "stiffness": float(stiffnesses[local_index]),
                "damping": float(dampings[local_index]),
                "force_limit": force_limit,
                "max_velocity": max_velocity,
            }
    return properties


def _gf_vec(value: Any, count: int) -> list[float]:
    return [float(value[index]) for index in range(count)]


def _gf_quat_xyzw(value: Any) -> list[float]:
    imaginary = value.GetImaginary()
    return [float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(value.GetReal())]


def _stage_prims_including_instance_proxies(stage: Any) -> list[Any]:
    """Traverse loaded prims and collision children hidden behind instance proxies."""
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RobotDefinitionError("USD traversal must run inside Isaac Sim") from exc
    queue = list(stage.GetPseudoRoot().GetFilteredChildren(Usd.TraverseInstanceProxies()))
    result: list[Any] = []
    while queue:
        prim = queue.pop(0)
        result.append(prim)
        queue.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    return result


def _body_name_aliases(name: str) -> tuple[str, ...]:
    aliases = [name]
    head, separator, suffix = name.rpartition("_")
    if separator and head and suffix.isdigit():
        aliases.append(head)
    return tuple(aliases)


def _usd_body_prims(body_names: list[str]) -> dict[str, Any]:
    """Bind PhysX link names to USD rigid prims, including merged-root suffixes."""
    try:
        import omni.usd
        from pxr import UsdPhysics
    except ImportError as exc:
        raise RobotDefinitionError("USD body export must run inside Isaac Sim") from exc
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RobotDefinitionError("Isaac USD stage is not available")
    candidates = [
        prim
        for prim in _stage_prims_including_instance_proxies(stage)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]

    def preference(prim: Any) -> tuple[int, int, str]:
        path = str(prim.GetPath())
        return (
            0 if "/env_0/Robot" in path else 1,
            0 if "/env_0/" in path else 1,
            path,
        )

    result: dict[str, Any] = {}
    used_paths: set[str] = set()
    for body_name in body_names:
        matches = [
            prim
            for prim in candidates
            if prim.GetName() in _body_name_aliases(body_name)
            and str(prim.GetPath()) not in used_paths
        ]
        if not matches:
            raise RobotDefinitionError(
                f"USD stage has no rigid body prim for PhysX body '{body_name}'"
            )
        selected = sorted(matches, key=preference)[0]
        result[body_name] = selected
        used_paths.add(str(selected.GetPath()))
    return result


def _usd_joint_topology(
    joint_names: list[str], body_prims: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    try:
        import omni.usd
        from pxr import UsdPhysics
    except ImportError as exc:
        raise RobotDefinitionError("USD topology export must run inside Isaac Sim") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RobotDefinitionError("Isaac USD stage is not available")
    requested = set(joint_names)
    body_by_path = {str(prim.GetPath()): name for name, prim in body_prims.items()}

    def mapped_body(target: Any, joint_name: str, role: str) -> str:
        direct = body_by_path.get(str(target))
        if direct is not None:
            return direct
        matches = [
            body_name
            for body_name, prim in body_prims.items()
            if prim.GetName() == target.name
            or target.name in _body_name_aliases(body_name)
        ]
        if len(matches) != 1:
            raise RobotDefinitionError(
                f"Joint '{joint_name}' {role} body '{target}' cannot be mapped to PhysX names"
            )
        return matches[0]
    candidates: dict[str, list[Any]] = {name: [] for name in joint_names}
    for prim in _stage_prims_including_instance_proxies(stage):
        name = prim.GetName()
        if name in requested and prim.IsA(UsdPhysics.Joint):
            candidates[name].append(prim)

    result: dict[str, dict[str, Any]] = {}
    for name in joint_names:
        matches = candidates[name]
        if not matches:
            raise RobotDefinitionError(f"USD stage has no joint prim named '{name}'")
        prim = next((item for item in matches if "/env_0/" in str(item.GetPath())), matches[0])
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body1) != 1:
            raise RobotDefinitionError(f"Joint '{name}' must have exactly one child body")
        joint_type = "FixedJoint"
        axis = "X"
        if prim.IsA(UsdPhysics.RevoluteJoint):
            schema = UsdPhysics.RevoluteJoint(prim)
            joint_type = "RevoluteJoint"
            axis = str(schema.GetAxisAttr().Get() or "X")
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            schema = UsdPhysics.PrismaticJoint(prim)
            joint_type = "PrismaticJoint"
            axis = str(schema.GetAxisAttr().Get() or "X")
        local_pos0 = joint.GetLocalPos0Attr().Get()
        local_pos1 = joint.GetLocalPos1Attr().Get()
        local_rot0 = joint.GetLocalRot0Attr().Get()
        local_rot1 = joint.GetLocalRot1Attr().Get()
        result[name] = {
            "parent_link": mapped_body(body0[0], name, "parent") if len(body0) == 1 else "",
            "child_link": mapped_body(body1[0], name, "child"),
            "joint_type": joint_type,
            "axis": _axis_to_unity(axis),
            "parent_anchor_position": _position_to_unity(_gf_vec(local_pos0, 3)),
            "parent_anchor_rotation": _quat_xyzw_to_unity(_gf_quat_xyzw(local_rot0)),
            "anchor_position": _position_to_unity(_gf_vec(local_pos1, 3)),
            "anchor_rotation": _quat_xyzw_to_unity(_gf_quat_xyzw(local_rot1)),
        }
    return result


def _usd_fixed_joint_topology(body_prims: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fixed edges between exported PhysX bodies.

    PhysX exposes only DOF joints through ``robot.joint_names``. These fixed
    edges are still required to rebuild one reduced-coordinate hierarchy in
    Unity (for example Spot's feet and Jaco's end-effector body).
    """
    try:
        import omni.usd
        from pxr import UsdPhysics
    except ImportError as exc:
        raise RobotDefinitionError("USD fixed topology export must run inside Isaac Sim") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RobotDefinitionError("Isaac USD stage is not available")
    body_by_path = {str(prim.GetPath()): name for name, prim in body_prims.items()}

    def mapped_body(target: Any) -> str | None:
        direct = body_by_path.get(str(target))
        if direct is not None:
            return direct
        exact = [
            body_name
            for body_name, prim in body_prims.items()
            if prim.GetName() == target.name
        ]
        if len(exact) == 1:
            return exact[0]
        aliases = [
            body_name
            for body_name in body_prims
            if target.name in _body_name_aliases(body_name)
        ]
        return aliases[0] if len(aliases) == 1 else None

    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for prim in _stage_prims_including_instance_proxies(stage):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            continue
        parent_link = mapped_body(body0[0])
        child_link = mapped_body(body1[0])
        if parent_link is None or child_link is None or parent_link == child_link:
            continue
        item = {
            "name": str(prim.GetName()),
            "parent_link": parent_link,
            "child_link": child_link,
            "parent_anchor_position": _position_to_unity(
                _gf_vec(joint.GetLocalPos0Attr().Get(), 3)
            ),
            "parent_anchor_rotation": _quat_xyzw_to_unity(
                _gf_quat_xyzw(joint.GetLocalRot0Attr().Get())
            ),
            "anchor_position": _position_to_unity(
                _gf_vec(joint.GetLocalPos1Attr().Get(), 3)
            ),
            "anchor_rotation": _quat_xyzw_to_unity(
                _gf_quat_xyzw(joint.GetLocalRot1Attr().Get())
            ),
        }
        candidates.setdefault(child_link, []).append((str(prim.GetPath()), item))

    result: list[dict[str, Any]] = []
    for child_link in body_prims:
        matches = candidates.get(child_link, [])
        if not matches:
            continue
        matches.sort(key=lambda pair: (0 if "/env_0/" in pair[0] else 1, pair[0]))
        selected_path, selected = matches[0]
        equivalent = [item for path, item in matches if "/env_0/" in path]
        if len(equivalent) > 1 and any(item != selected for item in equivalent[1:]):
            raise RobotDefinitionError(
                f"Body '{child_link}' has ambiguous fixed-joint topology at '{selected_path}'"
            )
        result.append(selected)
    name_counts: dict[str, int] = {}
    for item in result:
        name_counts[item["name"]] = name_counts.get(item["name"], 0) + 1
    for item in result:
        if name_counts[item["name"]] > 1:
            item["name"] = f"{item['name']}__{item['child_link']}"
    return result


def _usd_auxiliary_joint_topology(
    body_prims: dict[str, Any],
    joint_topology: dict[str, dict[str, Any]],
    fixed_joints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return body-to-body USD constraints omitted by the PhysX DOF view.

    Closed-chain mechanisms cannot be represented by a single reduced-coordinate
    tree. Isaac may therefore expose a joint in USD without including it in
    ``robot.joint_names`` (or in the selected fixed-tree edges). Keeping these
    constraints explicit prevents Unity prefab preparation from silently turning
    a four-bar gripper into several disconnected articulation roots.
    """
    try:
        import omni.usd
        from pxr import UsdPhysics
    except ImportError as exc:
        raise RobotDefinitionError("USD auxiliary topology export must run inside Isaac Sim") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RobotDefinitionError("Isaac USD stage is not available")
    body_by_path = {str(prim.GetPath()): name for name, prim in body_prims.items()}

    def mapped_body(target: Any) -> str | None:
        direct = body_by_path.get(str(target))
        if direct is not None:
            return direct
        exact = [
            body_name
            for body_name, prim in body_prims.items()
            if prim.GetName() == target.name
        ]
        if len(exact) == 1:
            return exact[0]
        aliases = [
            body_name
            for body_name in body_prims
            if target.name in _body_name_aliases(body_name)
        ]
        return aliases[0] if len(aliases) == 1 else None

    represented_names = set(joint_topology)
    represented_fixed = {
        (item["name"], item["parent_link"], item["child_link"])
        for item in fixed_joints
    }
    candidates: list[tuple[str, dict[str, Any]]] = []
    for prim in _stage_prims_including_instance_proxies(stage):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        path = str(prim.GetPath())
        if "/env_0/" not in path:
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            continue
        parent_link = mapped_body(body0[0])
        child_link = mapped_body(body1[0])
        if parent_link is None or child_link is None or parent_link == child_link:
            continue
        name = str(prim.GetName())
        if name in represented_names or (name, parent_link, child_link) in represented_fixed:
            continue
        joint_type = "GenericJoint"
        axis = [1.0, 0.0, 0.0]
        if prim.IsA(UsdPhysics.FixedJoint):
            joint_type = "FixedJoint"
        elif prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "RevoluteJoint"
            axis = _axis_to_unity(str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() or "X"))
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "PrismaticJoint"
            axis = _axis_to_unity(str(UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get() or "X"))
        item = {
            "name": name,
            "source_prim_path": path,
            "source_joint_type": str(prim.GetTypeName()),
            "joint_type": joint_type,
            "parent_link": parent_link,
            "child_link": child_link,
            "axis": axis,
            "parent_anchor_position": _position_to_unity(
                _gf_vec(joint.GetLocalPos0Attr().Get(), 3)
            ),
            "parent_anchor_rotation": _quat_xyzw_to_unity(
                _gf_quat_xyzw(joint.GetLocalRot0Attr().Get())
            ),
            "anchor_position": _position_to_unity(
                _gf_vec(joint.GetLocalPos1Attr().Get(), 3)
            ),
            "anchor_rotation": _quat_xyzw_to_unity(
                _gf_quat_xyzw(joint.GetLocalRot1Attr().Get())
            ),
        }
        candidates.append((path, item))

    # USD instances can expose equivalent constraints more than once. Keep a
    # stable env_0 edge per name and endpoint pair, then classify whether it
    # joins disconnected trees or closes an already-connected cycle.
    unique: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
    for path, item in sorted(candidates, key=lambda pair: pair[0]):
        key = (item["name"], item["parent_link"], item["child_link"])
        unique.setdefault(key, (path, item))

    parent = {name: name for name in body_prims}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(first: str, second: str) -> bool:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        parent[second_root] = first_root
        return True

    for item in joint_topology.values():
        if item["parent_link"]:
            union(item["parent_link"], item["child_link"])
    for item in fixed_joints:
        union(item["parent_link"], item["child_link"])

    result: list[dict[str, Any]] = []
    for _, item in sorted(unique.values(), key=lambda pair: pair[0]):
        item["topology_role"] = (
            "tree_connector"
            if union(item["parent_link"], item["child_link"])
            else "loop_closure"
        )
        result.append(item)
    name_counts: dict[str, int] = {}
    for item in result:
        name_counts[item["name"]] = name_counts.get(item["name"], 0) + 1
    for item in result:
        if name_counts[item["name"]] > 1:
            item["name"] = f"{item['name']}__{item['child_link']}"
    return result


def _usd_link_collision_properties(
    body_names: list[str], body_prims: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Read effective PhysX material/contact properties from collision prims per body."""
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RobotDefinitionError("USD collision export must run inside Isaac Sim") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RobotDefinitionError("Isaac USD stage is not available")
    stage_prims = _stage_prims_including_instance_proxies(stage)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    path_to_body = {str(prim.GetPath()): name for name, prim in body_prims.items()}
    shapes: dict[str, list[Any]] = {name: [] for name in body_names}
    for prim in stage_prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        current = prim
        while current and current.IsValid():
            owner = path_to_body.get(str(current.GetPath()))
            if owner is not None:
                shapes[owner].append(prim)
                break
            current = current.GetParent()

    def inherited_material(shape: Any) -> Any | None:
        current = shape
        while current and current.IsValid():
            relationship = current.GetRelationship("material:binding:physics")
            targets = relationship.GetTargets() if relationship and relationship.IsValid() else []
            if targets:
                material = stage.GetPrimAtPath(targets[0])
                if material and material.IsValid():
                    return material
            current = current.GetParent()
        return None

    def number(prim: Any | None, attribute: str, default: float) -> float:
        if prim is None:
            return default
        attr = prim.GetAttribute(attribute)
        value = attr.Get() if attr and attr.IsValid() else None
        if value is None:
            return default
        result = float(value)
        if math.isfinite(result):
            return result
        if attribute in {"physxCollision:contactOffset", "physxCollision:restOffset"}:
            return default
        raise RobotDefinitionError(
            f"Collision/material attribute '{attribute}' is non-finite"
        )

    def common(values: list[float], label: str, body_name: str) -> float:
        if not values:
            return -1.0
        first = values[0]
        if any(not math.isclose(item, first, rel_tol=0.0, abs_tol=1.0e-7) for item in values[1:]):
            raise RobotDefinitionError(
                f"Body '{body_name}' has heterogeneous {label}; normalize it before Unity export"
            )
        return first

    def collision_descriptor(shape: Any, body_prim: Any) -> dict[str, Any]:
        relative, _ = xform_cache.ComputeRelativeTransform(shape, body_prim)
        transform = Gf.Transform(relative)
        scale = [abs(float(value)) for value in transform.GetScale()]
        descriptor: dict[str, Any] = {
            "type": "mesh",
            "local_position": _position_to_unity(_gf_vec(transform.GetTranslation(), 3)),
            "local_rotation": _quat_xyzw_to_unity(
                _gf_quat_xyzw(transform.GetRotation().GetQuat())
            ),
            "scale": _position_to_unity(scale),
            "source_prim_path": str(shape.GetPath()),
        }
        if shape.IsA(UsdGeom.Sphere):
            radius = float(UsdGeom.Sphere(shape).GetRadiusAttr().Get())
            descriptor.update(type="sphere", radius=radius * max(scale))
        elif shape.IsA(UsdGeom.Cube):
            size = float(UsdGeom.Cube(shape).GetSizeAttr().Get())
            descriptor.update(type="box", size=_position_to_unity([size * item for item in scale]))
        elif shape.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(shape)
            axis = str(capsule.GetAxisAttr().Get() or "Z")
            source_axis = {"X": 0, "Y": 1, "Z": 2}[axis]
            perpendicular = [scale[index] for index in range(3) if index != source_axis]
            descriptor.update(
                type="capsule",
                radius=float(capsule.GetRadiusAttr().Get()) * max(perpendicular),
                height=float(capsule.GetHeightAttr().Get()) * scale[source_axis],
                axis=_axis_to_unity(axis),
            )
        elif shape.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(shape)
            axis = str(cylinder.GetAxisAttr().Get() or "Z")
            source_axis = {"X": 0, "Y": 1, "Z": 2}[axis]
            perpendicular = [scale[index] for index in range(3) if index != source_axis]
            descriptor.update(
                type="cylinder",
                radius=float(cylinder.GetRadiusAttr().Get()) * max(perpendicular),
                height=float(cylinder.GetHeightAttr().Get()) * scale[source_axis],
                axis=_axis_to_unity(axis),
            )
        return descriptor

    result: dict[str, dict[str, Any]] = {}
    for body_name, collision_prims in shapes.items():
        enabled: list[bool] = []
        static_friction: list[float] = []
        dynamic_friction: list[float] = []
        restitution: list[float] = []
        contact_offsets: list[float] = []
        rest_offsets: list[float] = []
        for shape in collision_prims:
            enabled_attr = UsdPhysics.CollisionAPI(shape).GetCollisionEnabledAttr().Get()
            enabled.append(True if enabled_attr is None else bool(enabled_attr))
            material = inherited_material(shape)
            # Isaac/PhysX defaults when no physics material is authored.
            static_friction.append(number(material, "physics:staticFriction", 0.5))
            dynamic_friction.append(number(material, "physics:dynamicFriction", 0.5))
            restitution.append(number(material, "physics:restitution", 0.0))
            contact_offsets.append(number(shape, "physxCollision:contactOffset", -1.0))
            rest_offsets.append(number(shape, "physxCollision:restOffset", -1.0))
        if enabled and any(item != enabled[0] for item in enabled[1:]):
            raise RobotDefinitionError(
                f"Body '{body_name}' mixes enabled and disabled collision shapes"
            )
        result[body_name] = {
            "collision_shape_count": len(collision_prims),
            "collision_shapes": [
                collision_descriptor(shape, body_prims[body_name])
                for shape in collision_prims
            ],
            "collision_enabled": bool(enabled[0]) if enabled else False,
            "static_friction": common(static_friction, "static friction", body_name),
            "dynamic_friction": common(dynamic_friction, "dynamic friction", body_name),
            "restitution": common(restitution, "restitution", body_name),
            "contact_offset": common(contact_offsets, "contact offset", body_name),
            "rest_offset": common(rest_offsets, "rest offset", body_name),
        }
    return result


@dataclass
class RobotDefinitionDocument:
    robot_id: str
    source_asset: str
    source_sha256: str
    definition_sha256: str
    provenance: str
    redistribution_license: str
    links: list[dict[str, Any]]
    joints: list[dict[str, Any]]
    fixed_joints: list[dict[str, Any]]
    auxiliary_joints: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = ROBOT_DEFINITION_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != ROBOT_DEFINITION_SCHEMA_VERSION or not self.robot_id:
            raise RobotDefinitionError("Robot definition schema or robot_id is invalid")
        if self.source_sha256 and (
            len(self.source_sha256) != 64 or
            any(character not in "0123456789abcdefABCDEF" for character in self.source_sha256)
        ):
            raise RobotDefinitionError("Robot source SHA-256 is invalid")
        link_names = [item.get("name") for item in self.links]
        if len(link_names) != len(set(link_names)) or any(not item for item in link_names):
            raise RobotDefinitionError("Robot link names must be unique and non-empty")
        joint_names = [item.get("name") for item in self.joints]
        if len(joint_names) != len(set(joint_names)) or any(not item for item in joint_names):
            raise RobotDefinitionError("Robot joint names must be unique and non-empty")
        fixed_joint_names = [item.get("name") for item in self.fixed_joints]
        if len(fixed_joint_names) != len(set(fixed_joint_names)) or any(
            not item for item in fixed_joint_names
        ):
            raise RobotDefinitionError("Robot fixed-joint names must be unique and non-empty")
        auxiliary_joint_names = [item.get("name") for item in self.auxiliary_joints]
        if len(auxiliary_joint_names) != len(set(auxiliary_joint_names)) or any(
            not item for item in auxiliary_joint_names
        ):
            raise RobotDefinitionError(
                "Robot auxiliary-joint names must be unique and non-empty"
            )
        for link in self.links:
            if not math.isfinite(float(link["mass"])) or float(link["mass"]) <= 0.0:
                raise RobotDefinitionError(f"Link '{link['name']}' has invalid mass")
            if any(float(item) <= 0.0 for item in link["inertia_tensor"]):
                raise RobotDefinitionError(f"Link '{link['name']}' has invalid inertia")
            collision_count = int(link.get("collision_shape_count", -1))
            if collision_count < 0:
                raise RobotDefinitionError(
                    f"Link '{link['name']}' has no collision shape count"
                )
            collision_shapes = link.get("collision_shapes")
            if collision_shapes is not None:
                if len(collision_shapes) != collision_count:
                    raise RobotDefinitionError(
                        f"Link '{link['name']}' collision descriptor count differs from PhysX"
                    )
                for index, shape in enumerate(collision_shapes):
                    shape_type = shape.get("type")
                    if shape_type not in {"mesh", "box", "sphere", "capsule", "cylinder"}:
                        raise RobotDefinitionError(
                            f"Link '{link['name']}' collision {index} has an invalid type"
                        )
                    for key, length in (("local_position", 3), ("local_rotation", 4), ("scale", 3)):
                        values = [float(value) for value in shape.get(key, [])]
                        if len(values) != length or not all(math.isfinite(value) for value in values):
                            raise RobotDefinitionError(
                                f"Link '{link['name']}' collision {index} has invalid {key}"
                            )
                        if key == "scale" and any(value <= 0.0 for value in values):
                            raise RobotDefinitionError(
                                f"Link '{link['name']}' collision {index} has non-positive scale"
                            )
                    rotation = [float(value) for value in shape["local_rotation"]]
                    if not math.isclose(
                        sum(value * value for value in rotation),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1.0e-3,
                    ):
                        raise RobotDefinitionError(
                            f"Link '{link['name']}' collision {index} has a non-unit rotation"
                        )
                    if shape_type == "box":
                        size = [float(value) for value in shape.get("size", [])]
                        if len(size) != 3 or any(
                            not math.isfinite(value) or value <= 0.0 for value in size
                        ):
                            raise RobotDefinitionError(
                                f"Link '{link['name']}' collision {index} has invalid box size"
                            )
                    if shape_type in {"sphere", "capsule", "cylinder"}:
                        radius = float(shape.get("radius", -1.0))
                        if not math.isfinite(radius) or radius <= 0.0:
                            raise RobotDefinitionError(
                                f"Link '{link['name']}' collision {index} has invalid radius"
                            )
                    if shape_type in {"capsule", "cylinder"}:
                        height = float(shape.get("height", -1.0))
                        axis = [float(value) for value in shape.get("axis", [])]
                        if (
                            not math.isfinite(height)
                            or height < 0.0
                            or len(axis) != 3
                            or not math.isclose(
                                sum(value * value for value in axis),
                                1.0,
                                rel_tol=0.0,
                                abs_tol=1.0e-3,
                            )
                        ):
                            raise RobotDefinitionError(
                                f"Link '{link['name']}' collision {index} has invalid axis/height"
                            )
            if collision_count > 0:
                for key in ("static_friction", "dynamic_friction", "restitution"):
                    value = float(link.get(key, -1.0))
                    if not math.isfinite(value) or value < 0.0:
                        raise RobotDefinitionError(
                            f"Link '{link['name']}' has invalid {key}"
                        )
        known_links = set(link_names)
        for joint in self.joints:
            if joint["child_link"] not in known_links:
                raise RobotDefinitionError(f"Joint '{joint['name']}' has an unknown child link")
            if joint["parent_link"] and joint["parent_link"] not in known_links:
                raise RobotDefinitionError(f"Joint '{joint['name']}' has an unknown parent link")
            expected_unit = {
                "RevoluteJoint": "radian",
                "PrismaticJoint": "meter",
                "FixedJoint": "fixed",
            }.get(joint["joint_type"])
            if expected_unit is None or joint.get("position_unit") != expected_unit:
                raise RobotDefinitionError(
                    f"Joint '{joint['name']}' has an invalid type/unit combination"
                )
            limit_mode = joint.get("limit_mode")
            if (
                (joint["joint_type"] == "FixedJoint" and limit_mode != "fixed")
                or (
                    joint["joint_type"] == "PrismaticJoint"
                    and limit_mode != "limited"
                )
                or (
                    joint["joint_type"] == "RevoluteJoint"
                    and limit_mode not in {"limited", "continuous"}
                )
            ):
                raise RobotDefinitionError(
                    f"Joint '{joint['name']}' has an invalid limit mode"
                )
            axis = [float(component) for component in joint.get("axis", [])]
            if len(axis) != 3 or not all(math.isfinite(component) for component in axis) or not math.isclose(
                sum(component * component for component in axis), 1.0, rel_tol=0.0, abs_tol=1.0e-3
            ):
                raise RobotDefinitionError(f"Joint '{joint['name']}' has a non-unit axis")
            values = [
                float(joint[key])
                for key in (
                    "lower_limit",
                    "upper_limit",
                    "default_position",
                    "stiffness",
                    "damping",
                    "force_limit",
                    "max_velocity",
                )
            ]
            if (
                not all(math.isfinite(value) for value in values)
                or values[0] > values[1]
                or values[3] < 0.0
                or values[4] < 0.0
                or values[5] <= 0.0
            ):
                raise RobotDefinitionError(
                    f"Joint '{joint['name']}' has invalid limits or actuator properties"
                )
            if joint["joint_type"] != "FixedJoint" and values[6] <= 0.0:
                raise RobotDefinitionError(f"Joint '{joint['name']}' has a non-positive velocity limit")
            if not joint.get("actuator_model"):
                raise RobotDefinitionError(f"Joint '{joint['name']}' has no actuator model")
            delay_min = int(joint.get("actuator_delay_min_steps", -1))
            delay_max = int(joint.get("actuator_delay_max_steps", -1))
            nominal_delay = int(joint.get("actuator_nominal_delay_steps", -1))
            if delay_min < 0 or delay_max < delay_min or not delay_min <= nominal_delay <= delay_max:
                raise RobotDefinitionError(f"Joint '{joint['name']}' has an invalid actuator delay")
            previous_position = -math.inf
            for sample in joint.get("effort_limit_curve", []):
                position = float(sample["position"])
                effort = float(sample["max_effort"])
                if (
                    not math.isfinite(position)
                    or not math.isfinite(effort)
                    or effort <= 0.0
                    or position <= previous_position
                ):
                    raise RobotDefinitionError(
                        f"Joint '{joint['name']}' has an invalid effort limit curve"
                    )
                previous_position = position
        for joint in self.fixed_joints:
            if joint.get("child_link") not in known_links:
                raise RobotDefinitionError(
                    f"Fixed joint '{joint.get('name')}' has an unknown child link"
                )
            if joint.get("parent_link") not in known_links:
                raise RobotDefinitionError(
                    f"Fixed joint '{joint.get('name')}' has an unknown parent link"
                )
        topology_parent = {name: name for name in known_links}

        def topology_find(name: str) -> str:
            while topology_parent[name] != name:
                topology_parent[name] = topology_parent[topology_parent[name]]
                name = topology_parent[name]
            return name

        def topology_union(first: str, second: str) -> bool:
            first_root = topology_find(first)
            second_root = topology_find(second)
            if first_root == second_root:
                return False
            topology_parent[second_root] = first_root
            return True

        for joint in self.joints:
            if joint["parent_link"]:
                topology_union(joint["parent_link"], joint["child_link"])
        for joint in self.fixed_joints:
            topology_union(joint["parent_link"], joint["child_link"])
        for joint in self.auxiliary_joints:
            if joint.get("child_link") not in known_links:
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' has an unknown child link"
                )
            if joint.get("parent_link") not in known_links:
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' has an unknown parent link"
                )
            if joint.get("parent_link") == joint.get("child_link"):
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' connects a link to itself"
                )
            if not joint.get("source_prim_path") or not joint.get("source_joint_type"):
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' has no USD source identity"
                )
            if joint.get("joint_type") not in {
                "FixedJoint",
                "RevoluteJoint",
                "PrismaticJoint",
                "GenericJoint",
            }:
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' has an invalid type"
                )
            if joint.get("topology_role") not in {"tree_connector", "loop_closure"}:
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' has an invalid topology role"
                )
            expected_role = (
                "tree_connector"
                if topology_union(joint["parent_link"], joint["child_link"])
                else "loop_closure"
            )
            if joint["topology_role"] != expected_role:
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' is mislabeled; "
                    f"expected {expected_role}"
                )
            axis = [float(component) for component in joint.get("axis", [])]
            if (
                len(axis) != 3
                or not all(math.isfinite(component) for component in axis)
                or not math.isclose(
                    sum(component * component for component in axis),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-3,
                )
            ):
                raise RobotDefinitionError(
                    f"Auxiliary joint '{joint.get('name')}' has a non-unit axis"
                )
            for key, length in (
                ("parent_anchor_position", 3),
                ("parent_anchor_rotation", 4),
                ("anchor_position", 3),
                ("anchor_rotation", 4),
            ):
                values = [float(value) for value in joint.get(key, [])]
                if len(values) != length or not all(math.isfinite(value) for value in values):
                    raise RobotDefinitionError(
                        f"Auxiliary joint '{joint.get('name')}' has invalid {key}"
                    )
                if key.endswith("rotation") and not math.isclose(
                    sum(value * value for value in values),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-3,
                ):
                    raise RobotDefinitionError(
                        f"Auxiliary joint '{joint.get('name')}' has non-unit {key}"
                    )
        expected_hash = _definition_hash(
            self.robot_id,
            self.source_asset,
            self.links,
            self.joints,
            self.fixed_joints,
            self.auxiliary_joints,
        )
        if self.definition_sha256 != expected_hash:
            raise RobotDefinitionError(
                "Robot definition SHA-256 does not match its canonical physics payload"
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "robot_id": self.robot_id,
            "source_asset": self.source_asset,
            "source_sha256": self.source_sha256,
            "definition_sha256": self.definition_sha256,
            "provenance": self.provenance,
            "redistribution_license": self.redistribution_license,
            "links": self.links,
            "joints": self.joints,
            "fixed_joints": self.fixed_joints,
        }
        if self.auxiliary_joints:
            result["auxiliary_joints"] = self.auxiliary_joints
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotDefinitionDocument":
        try:
            document = cls(**data)
        except TypeError as exc:
            raise RobotDefinitionError(f"Invalid robot definition fields: {exc}") from exc
        document.validate()
        return document

    @classmethod
    def load(cls, path: str | Path) -> "RobotDefinitionDocument":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RobotDefinitionError(f"Could not load '{source}': {exc}") from exc
        if not isinstance(data, dict):
            raise RobotDefinitionError("Robot definition root must be an object")
        return cls.from_dict(data)

    def write_atomic(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise


def _balance_definition(
    robot_id: str,
    provenance: str,
    redistribution_license: str,
) -> RobotDefinitionDocument:
    links = [
        {
            "name": "tray_root",
            "mass": 5.0,
            "center_of_mass": [0.0, 0.0, 0.0],
            "inertia_tensor": [0.25, 0.25, 0.25],
            "inertia_tensor_rotation": [0.0, 0.0, 0.0, 1.0],
            "collision_shape_count": 0,
            "collision_enabled": False,
            "static_friction": -1.0,
            "dynamic_friction": -1.0,
            "restitution": -1.0,
            "contact_offset": -1.0,
            "rest_offset": -1.0,
        },
        {
            "name": "roll_link",
            "mass": 1.0,
            "center_of_mass": [0.0, 0.0, 0.0],
            "inertia_tensor": [0.05, 0.05, 0.05],
            "inertia_tensor_rotation": [0.0, 0.0, 0.0, 1.0],
            "collision_shape_count": 0,
            "collision_enabled": False,
            "static_friction": -1.0,
            "dynamic_friction": -1.0,
            "restitution": -1.0,
            "contact_offset": -1.0,
            "rest_offset": -1.0,
        },
        {
            "name": "tray",
            "mass": 2.0,
            "center_of_mass": [0.0, 0.0, 0.0],
            "inertia_tensor": [0.1, 0.1, 0.1],
            "inertia_tensor_rotation": [0.0, 0.0, 0.0, 1.0],
            "collision_shape_count": 1,
            "collision_enabled": True,
            "static_friction": 0.5,
            "dynamic_friction": 0.5,
            "restitution": 0.0,
            "contact_offset": 0.005,
            "rest_offset": 0.0,
        },
    ]
    joints = []
    for name, parent, child, axis in (
        ("roll_joint", "tray_root", "roll_link", [1.0, 0.0, 0.0]),
        ("pitch_joint", "roll_link", "tray", [0.0, 0.0, 1.0]),
    ):
        joints.append(
            {
                "name": name,
                "parent_link": parent,
                "child_link": child,
                "joint_type": "RevoluteJoint",
                "position_unit": "radian",
                "limit_mode": "limited",
                "axis": axis,
                "parent_anchor_position": [0.0, 0.0, 0.0],
                "parent_anchor_rotation": [0.0, 0.0, 0.0, 1.0],
                "anchor_position": [0.0, 0.0, 0.0],
                "anchor_rotation": [0.0, 0.0, 0.0, 1.0],
                "lower_limit": -0.35,
                "upper_limit": 0.35,
                "default_position": 0.0,
                "stiffness": 0.0,
                "damping": 0.0,
                "force_limit": 100.0,
                "max_velocity": 2.0,
                "actuator_group": "procedural_tray",
                "actuator_model": "UnityKinematicTarget",
                "actuator_is_implicit": True,
                "actuator_delay_min_steps": 0,
                "actuator_delay_max_steps": 0,
                "actuator_nominal_delay_steps": 0,
                "effort_limit_curve": [],
            }
        )
    source_asset = "procedural://balance_tray_2dof"
    return RobotDefinitionDocument(
        robot_id=robot_id,
        source_asset=source_asset,
        source_sha256="",
        definition_sha256=_definition_hash(robot_id, source_asset, links, joints, []),
        provenance=provenance or "XRPlayground procedural definition",
        redistribution_license=redistribution_license,
        links=links,
        joints=joints,
        fixed_joints=[],
    )


def definition_from_environment(
    env: Any,
    robot_id: str,
    *,
    env_index: int = 0,
    provenance: str = "",
    redistribution_license: str = "UNVERIFIED",
) -> RobotDefinitionDocument:
    if robot_id == "balance_tray_2dof":
        return _balance_definition(robot_id, provenance, redistribution_license)
    unwrapped = env.unwrapped
    robot = getattr(unwrapped, "robot", None)
    if robot is None and hasattr(unwrapped, "scene"):
        robot = unwrapped.scene["robot"]
    if robot is None:
        raise RobotDefinitionError(f"Environment has no articulation for robot '{robot_id}'")

    view = robot.root_physx_view
    body_names = [str(item) for item in robot.body_names]
    joint_names = [str(item) for item in robot.joint_names]
    masses = _row(view.get_masses(), env_index)
    inertias = _row(view.get_inertias(), env_index)
    coms = _row(view.get_coms(), env_index)
    limits = _row(view.get_dof_limits(), env_index)
    stiffnesses = _row(view.get_dof_stiffnesses(), env_index)
    dampings = _row(view.get_dof_dampings(), env_index)
    force_limits = _row(view.get_dof_max_forces(), env_index)
    velocity_limits = _row(view.get_dof_max_velocities(), env_index)
    default_positions = _row(robot.data.default_joint_pos, env_index)
    actuator_properties = _actuator_joint_properties(
        robot,
        joint_names,
        env_index,
        stiffnesses,
        dampings,
        force_limits,
        velocity_limits,
    )
    body_prims = _usd_body_prims(body_names)
    topology = _usd_joint_topology(joint_names, body_prims)
    fixed_joints = _usd_fixed_joint_topology(body_prims)
    auxiliary_joints = _usd_auxiliary_joint_topology(
        body_prims, topology, fixed_joints
    )
    collision_properties = _usd_link_collision_properties(body_names, body_prims)

    if not (len(body_names) == len(masses) == len(inertias) == len(coms)):
        raise RobotDefinitionError("Isaac body property counts do not match body_names")
    if not all(
        len(values) == len(joint_names)
        for values in (limits, stiffnesses, dampings, force_limits, velocity_limits, default_positions)
    ):
        raise RobotDefinitionError("Isaac joint property counts do not match joint_names")

    links: list[dict[str, Any]] = []
    for index, name in enumerate(body_names):
        inertia, rotation = _principal_inertia(inertias[index])
        com = _finite_vector(coms[index], len(coms[index]), f"COM for {name}")
        if len(com) < 3:
            raise RobotDefinitionError(f"COM for '{name}' has no position")
        links.append(
            {
                "name": name,
                "mass": float(masses[index]),
                "center_of_mass": _position_to_unity(com[:3]),
                "inertia_tensor": inertia,
                "inertia_tensor_rotation": rotation,
                **collision_properties[name],
            }
        )

    joints: list[dict[str, Any]] = []
    for index, name in enumerate(joint_names):
        lower, upper = _finite_vector(limits[index], 2, f"limits for {name}")
        joint_type = topology[name]["joint_type"]
        position_unit = {
            "RevoluteJoint": "radian",
            "PrismaticJoint": "meter",
            "FixedJoint": "fixed",
        }[joint_type]
        limit_mode = "fixed"
        if joint_type == "PrismaticJoint":
            limit_mode = "limited"
        elif joint_type == "RevoluteJoint":
            limit_mode = (
                "continuous"
                if lower <= -1.0e20 and upper >= 1.0e20
                else "limited"
            )
        joints.append(
            {
                "name": name,
                **topology[name],
                "position_unit": position_unit,
                "limit_mode": limit_mode,
                "lower_limit": lower,
                "upper_limit": upper,
                "default_position": float(default_positions[index]),
                **actuator_properties[index],
            }
        )
    source = _source_asset(robot)
    return RobotDefinitionDocument(
        robot_id=robot_id,
        source_asset=source,
        source_sha256=_source_hash(source),
        definition_sha256=_definition_hash(
            robot_id, source, links, joints, fixed_joints, auxiliary_joints
        ),
        provenance=provenance or source,
        redistribution_license=redistribution_license,
        links=links,
        joints=joints,
        fixed_joints=fixed_joints,
        auxiliary_joints=auxiliary_joints,
    )
