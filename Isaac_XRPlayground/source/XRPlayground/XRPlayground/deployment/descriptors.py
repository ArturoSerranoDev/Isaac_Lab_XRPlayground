# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact ordered observation/action descriptors used by export and Unity."""

from __future__ import annotations

import copy
import math
from typing import Any


class DescriptorError(ValueError):
    """Raised when environment IO cannot be represented without ambiguity."""


def _term(
    name: str,
    size: int,
    *,
    frame: str = "none",
    units: str = "unitless",
    scale: float | list[float] = 1.0,
    offset: float | list[float] = 0.0,
    names: list[str] | None = None,
    history: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": name,
        "shape": [size],
        "dtype": "float32",
        "frame": frame,
        "units": units,
        "scale": scale,
        "offset": offset,
        "history": history,
    }
    if names:
        value["names"] = names
    value.update(extra)
    return value


def _action(
    name: str,
    size: int,
    *,
    target_type: str,
    integration: str,
    scale: float | list[float] = 1.0,
    offset: float | list[float] = 0.0,
    clip: list[float] | None = None,
    names: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    value = _term(
        name,
        size,
        units="normalized" if clip else "policy_output",
        scale=scale,
        offset=offset,
        names=names,
        target_type=target_type,
        integration=integration,
        **extra,
    )
    if clip is not None:
        value["clip"] = clip
    return value


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_float_list(item))
        return result
    return [float(value)]


def _selected_speed_scales(env: Any | None, ids_name: str, count: int) -> list[float]:
    """Read the exact Direct-env actuator multiplier without importing torch here."""
    if env is None:
        return [1.0] * count
    scales = getattr(env, "robot_dof_speed_scales", None)
    ids = getattr(env, ids_name, None)
    if scales is None or ids is None:
        return [1.0] * count
    try:
        selected = scales[ids]
    except (IndexError, KeyError, TypeError) as exc:
        raise DescriptorError(f"Could not resolve runtime speed scales for {ids_name}") from exc
    result = _float_list(selected)
    if len(result) != count:
        raise DescriptorError(
            f"Runtime speed scale count {len(result)} does not match {ids_name} count {count}"
        )
    return result


def direct_descriptors(
    policy_id: str, cfg: Any, env: Any | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Emit descriptors beside the Direct environment configuration that builds the tensors."""
    if policy_id == "ball_catch.throw":
        arm = list(cfg.arm_joint_names)
        observations = [
            _term("arm_joint_position", 7, units="rad", names=arm),
            _term("arm_joint_velocity", 7, units="rad/s", scale=float(cfg.dof_velocity_scale), names=arm),
            _term("gripper_position_mean", 1, units="rad", names=list(cfg.gripper_joint_names)),
            _term("gripper_velocity_mean", 1, units="rad/s", scale=float(cfg.dof_velocity_scale), names=list(cfg.gripper_joint_names)),
            _term("ball_position", 3, frame="environment", units="m", names=["x", "y", "z"]),
            _term("ball_linear_velocity", 3, frame="world", units="m/s", names=["x", "y", "z"]),
            _term("end_effector_to_ball", 3, frame="environment", units="m", names=["x", "y", "z"]),
            _term("fingertip_center_to_ball", 3, frame="environment", units="m", names=["x", "y", "z"]),
            _term("grasp_alignment", 1),
            _term("ball_radial_error", 1, units="m"),
        ]
        actions = [
            _action("arm_joint_position_delta", 7, target_type="joint_position", integration="velocity_scaled_delta", scale=float(cfg.action_scale), clip=[-1.0, 1.0], names=arm, velocity_scale=_selected_speed_scales(env, "_arm_ids", len(arm))),
            _action("gripper_position_delta", 1, target_type="joint_position", integration="velocity_scaled_delta", scale=float(cfg.action_scale), clip=[-1.0, 1.0], names=list(cfg.gripper_joint_names), velocity_scale=_selected_speed_scales(env, "_gripper_ids", len(cfg.gripper_joint_names))),
        ]
    elif policy_id == "conveyor_color.sort":
        arm = list(cfg.arm_joint_names)
        observations = [
            _term("arm_joint_position", 6, units="rad", names=arm),
            _term("arm_joint_velocity", 6, units="rad/s", scale=float(cfg.dof_velocity_scale), names=arm),
            _term("gripper_position", 1, units="rad", names=list(cfg.gripper_joint_names)),
            _term("gripper_velocity", 1, units="rad/s", scale=float(cfg.dof_velocity_scale), names=list(cfg.gripper_joint_names)),
            _term("target_color_one_hot", int(cfg.num_colors)),
        ]
        for slot in range(int(cfg.num_object_slots)):
            prefix = f"object_{slot}"
            observations.extend(
                [
                    _term(f"{prefix}_position", 3, frame="environment", units="m"),
                    _term(f"{prefix}_linear_velocity", 3, frame="world", units="m/s"),
                    _term(f"{prefix}_color_one_hot", int(cfg.num_colors)),
                    _term(f"{prefix}_active", 1),
                ]
            )
        observations.extend(
            [
                _term("end_effector_position", 3, frame="environment", units="m"),
                _term("nearest_target_delta", 3, frame="environment", units="m"),
                _term("bin_delta", 3, frame="environment", units="m"),
            ]
        )
        actions = [
            _action("arm_joint_position_delta", 6, target_type="joint_position", integration="delta", scale=float(cfg.action_scale), clip=[-1.0, 1.0], names=arm),
            _action("gripper_normalized_target", 1, target_type="joint_position", integration="absolute_lerp", clip=[-1.0, 1.0], names=[str(cfg.gripper_joint_names[0])], target_range=[float(cfg.gripper_open), float(cfg.gripper_close)]),
        ]
    elif policy_id == "pick_place_table.place":
        arm = list(cfg.arm_joint_names)
        observations = [
            _term("arm_joint_position", 7, units="rad", names=arm),
            _term("arm_joint_velocity", 7, units="rad/s", scale=float(cfg.dof_velocity_scale), names=arm),
            _term("gripper_position", 1, units="rad", names=[cfg.gripper_driver_name]),
            _term("gripper_velocity", 1, units="rad/s", scale=float(cfg.dof_velocity_scale), names=[cfg.gripper_driver_name]),
            _term("piece_position", 3, frame="environment", units="m"),
            _term("piece_linear_velocity", 3, frame="world", units="m/s"),
            _term("end_effector_to_piece", 3, frame="environment", units="m"),
            _term("piece_to_bucket", 3, frame="environment", units="m"),
            _term("piece_grasped", 1),
            _term("piece_lifted", 1),
        ]
        actions = [
            _action("arm_joint_position_delta", 7, target_type="joint_position", integration="velocity_scaled_delta", scale=float(cfg.action_scale), clip=[-1.0, 1.0], names=arm, velocity_scale=_selected_speed_scales(env, "_arm_ids", len(arm))),
            _action("gripper_position_delta", 1, target_type="joint_position", integration="velocity_scaled_delta", scale=float(cfg.action_scale), clip=[-1.0, 1.0], names=list(cfg.gripper_joint_names), velocity_scale=_selected_speed_scales(env, "_gripper_ids", len(cfg.gripper_joint_names))),
        ]
    elif policy_id == "balance_bot.two_ball":
        observations = [
            _term("tray_roll", 1, units="rad"),
            _term("tray_pitch", 1, units="rad"),
            _term("tray_roll_velocity", 1, units="rad/s", scale=float(cfg.dof_velocity_scale)),
            _term("tray_pitch_velocity", 1, units="rad/s", scale=float(cfg.dof_velocity_scale)),
        ]
        for slot in range(2):
            observations.extend(
                [
                    _term(f"ball_{slot}_relative_position", 3, frame="tray_origin", units="m"),
                    _term(f"ball_{slot}_linear_velocity", 3, frame="world", units="m/s"),
                    _term(f"ball_{slot}_active", 1),
                ]
            )
        observations.extend([_term("active_ball_count", 1), _term("curriculum_stage", 1)])
        actions = [
            _action("tray_angular_velocity", 2, target_type="kinematic_angle", integration="angular_velocity", scale=float(cfg.action_scale), clip=[-1.0, 1.0], names=list(cfg.joint_names))
        ]
    else:
        raise DescriptorError(f"No Direct descriptor generator for policy '{policy_id}'")
    return {"observations": observations, "actions": actions}


def manager_descriptors(env: Any, policy_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Read Isaac Lab ManagerBased IO descriptors without re-declaring term order."""
    unwrapped = env.unwrapped
    source = getattr(unwrapped, "get_IO_descriptors", None)
    if source is None:
        raise DescriptorError("ManagerBased environment does not expose get_IO_descriptors")
    raw = source() if callable(source) else source
    observations = raw.get("observations", {}).get("policy", [])
    actions = copy.deepcopy(raw.get("actions", []))
    if not observations or not actions:
        raise DescriptorError("Isaac Lab returned empty policy IO descriptors")
    # GenericActionIODescriptor leaves shape unset unless an action term
    # explicitly overrides it. PreTrainedPolicyAction is one such upstream
    # term, even though ActionManager owns the exact, ordered runtime sizes.
    # Backfill only from that manager-owned metadata so the contract remains
    # generated from the live environment rather than a duplicated constant.
    action_manager = getattr(unwrapped, "action_manager", None)
    action_dims = getattr(action_manager, "action_term_dim", None)
    if action_dims is not None:
        if len(action_dims) != len(actions):
            raise DescriptorError(
                "Isaac Lab action descriptors and manager term dimensions differ"
            )
        for item, dimension in zip(actions, action_dims, strict=True):
            if item.get("shape") is None and item.get("data_shape") is None:
                item["shape"] = [int(dimension)]
    result = {
        "observations": [_normalize_manager_term(item, f"observation_{index}") for index, item in enumerate(observations)],
        "actions": [_normalize_manager_term(item, f"action_{index}") for index, item in enumerate(actions)],
    }
    if policy_id is not None:
        _enrich_manager_semantics(policy_id, result)
    return result


def _normalize_manager_term(term: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    result = copy.deepcopy(term)
    result["name"] = str(result.get("name") or result.get("term_name") or fallback_name)
    shape = result.get("shape") or result.get("data_shape")
    if isinstance(shape, int):
        shape = [shape]
    if not isinstance(shape, (list, tuple)) or not shape:
        raise DescriptorError(f"Manager descriptor '{result['name']}' has no concrete shape")
    result["shape"] = [int(item) for item in shape if int(item) > 0]
    result["dtype"] = str(result.get("dtype", "float32")).replace("torch.", "")
    extras = result.get("extras") or {}
    overloads = result.get("overloads") or {}
    result.setdefault("frame", _manager_frame(result))
    result.setdefault("units", str(extras.get("units") or "unitless"))
    _normalize_affine(result, "scale", overloads.get("scale"), default=1.0)
    _normalize_affine(result, "offset", None, default=0.0)
    if "clip" not in result and overloads.get("clip") is not None:
        result["clip"] = _float_list(overloads["clip"])
    if "joint_names" in result and "names" not in result:
        result["names"] = [str(item) for item in result["joint_names"]]
    history_length = int(overloads.get("history_length") or 0)
    result.setdefault("history", max(1, history_length))
    return result


def _manager_frame(term: dict[str, Any]) -> str:
    name = str(term.get("name") or "").lower()
    if name in {"base_lin_vel", "base_ang_vel", "projected_gravity"} or "command" in name:
        return "robot_base"
    return "none"


def _normalize_affine(
    result: dict[str, Any], key: str, fallback: Any, *, default: float
) -> None:
    value = result.get(key, fallback)
    values = _float_list(value)
    if not values:
        result[key] = default
    elif all(abs(item - values[0]) <= 1.0e-12 for item in values):
        result[key] = values[0]
    else:
        result[key] = default
        result[f"{key}_values"] = values


def _enrich_manager_semantics(
    policy_id: str, descriptors: dict[str, list[dict[str, Any]]]
) -> None:
    """Add deployment behavior while preserving Isaac's authored term ordering."""
    if policy_id == "spot.locomotion":
        if len(descriptors["actions"]) != 1:
            raise DescriptorError("Spot locomotion must expose exactly one action term")
        action = descriptors["actions"][0]
        full_path = str(action.get("full_path") or "")
        if "JointPositionAction" not in full_path and action.get("action_type") != "JointAction":
            raise DescriptorError(
                f"Spot locomotion action is not a JointPositionAction: {full_path or action['name']}"
            )
        if len(action.get("names", [])) != math.prod(action["shape"]):
            raise DescriptorError("Spot locomotion action descriptor is missing ordered joint names")
        action.update(target_type="joint_position", integration="absolute", units="rad")
    elif policy_id == "spot.follow":
        if len(descriptors["actions"]) != 1 or math.prod(descriptors["actions"][0]["shape"]) != 3:
            raise DescriptorError("Spot Follow must expose one three-value high-level action")
        descriptors["actions"][0].update(
            target_type="velocity_command",
            integration="absolute_command",
            units="m/s,m/s,rad/s",
            names=["linear_x", "linear_y", "yaw_rate"],
        )
        for term in descriptors["observations"]:
            name = term["name"].lower()
            if "pose_command" in name or name == "generated_commands":
                if math.prod(term["shape"]) != 4:
                    raise DescriptorError("Spot Follow pose command must contain x, y, z and heading")
                term.update(
                    frame="robot_base",
                    units="m,m,m,rad",
                    names=["position_x", "position_y", "position_z", "heading"],
                )
    else:
        raise DescriptorError(f"No ManagerBased deployment semantics for policy '{policy_id}'")


def flat_dimension(terms: list[dict[str, Any]]) -> int:
    return sum(math.prod(term["shape"]) for term in terms)
