# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validated Isaac Lab policy export with isolated candidate/reference staging."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .catalog import PolicyEntry, load_catalog, monorepo_root
from .contract import PolicyContract, PolicyContractError, sha256_file
from .descriptors import flat_dimension, manager_descriptors
from .robot_definition import definition_from_environment


ONNX_PARITY_LIMIT = 1.0e-5
UNITY_PARITY_LIMIT = 1.0e-4
OBSERVATION_PARITY_LIMIT = 1.0e-4


def validate_normalization_embedding(torch_model: Any, agent_cfg: Any) -> bool:
    """Prove required actor normalization is part of the exported module."""
    actor_cfg = getattr(agent_cfg, "actor", None)
    if actor_cfg is None or not hasattr(actor_cfg, "obs_normalization"):
        raise PolicyContractError(
            "agent config must explicitly declare actor.obs_normalization"
        )
    if not bool(actor_cfg.obs_normalization):
        return True
    try:
        modules = list(torch_model.named_modules())
        state_keys = list(torch_model.state_dict())
    except (AttributeError, TypeError) as exc:
        raise PolicyContractError(
            "exported policy cannot prove embedded observation normalization"
        ) from exc
    normalizer_prefixes = [
        name
        for name, module in modules
        if "normal" in module.__class__.__name__.lower() or
        "normal" in name.lower()
    ]
    if not normalizer_prefixes or not any(
        any(
            not prefix or key == prefix or key.startswith(prefix + ".")
            for prefix in normalizer_prefixes
        )
        for key in state_keys
    ):
        raise PolicyContractError(
            "actor.obs_normalization is enabled but the ONNX export module has no "
            "stateful normalizer"
        )
    return True


def _shape(value_info: Any) -> list[int]:
    dims: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
            raise PolicyContractError(
                f"ONNX tensor '{value_info.name}' must have static positive dimensions"
            )
        dims.append(int(dim.dim_value))
    return dims


def inspect_onnx(model_path: str | Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise PolicyContractError("The 'onnx' package is required to validate exports") from exc

    path = Path(model_path)
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    outputs = list(model.graph.output)
    if not inputs or not outputs:
        raise PolicyContractError("ONNX graph must have at least one input and output")
    opsets = [int(item.version) for item in model.opset_import if item.domain in {"", "ai.onnx"}]
    if opsets != [15]:
        raise PolicyContractError(f"Expected one default ONNX opset 15 import, got {opsets}")
    signature = {
        "sha256": sha256_file(path),
        "opset": 15,
        "inputs": [{"name": item.name, "shape": _shape(item)} for item in inputs],
        "outputs": [{"name": item.name, "shape": _shape(item)} for item in outputs],
        "input_name": inputs[0].name,
        "input_shape": _shape(inputs[0]),
        "output_name": outputs[0].name,
        "output_shape": _shape(outputs[0]),
    }
    return signature


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def configuration_sha256(env_cfg: Any, agent_cfg: Any) -> str:
    payload = json.dumps(
        {"environment": _jsonable(env_cfg), "agent": _jsonable(agent_cfg)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def descriptors_from_environment(
    env: Any, policy: PolicyEntry
) -> dict[str, list[dict[str, Any]]]:
    unwrapped = env.unwrapped
    direct = getattr(unwrapped, "get_deployment_descriptors", None)
    return direct() if direct is not None else manager_descriptors(env, policy.policy_id)


def validate_descriptors(
    policy: PolicyEntry, descriptors: dict[str, list[dict[str, Any]]]
) -> None:
    observation_dim = flat_dimension(descriptors["observations"])
    action_dim = flat_dimension(descriptors["actions"])
    if observation_dim != policy.observation_dim:
        raise PolicyContractError(
            f"{policy.policy_id}: descriptor observation dimension {observation_dim} "
            f"does not match catalog {policy.observation_dim}"
        )
    if action_dim != policy.action_dim:
        raise PolicyContractError(
            f"{policy.policy_id}: descriptor action dimension {action_dim} "
            f"does not match catalog {policy.action_dim}"
        )


def validate_torch_onnx(
    torch_model: Any, model_path: str | Path, *, samples: int = 16, seed: int = 47
) -> float:
    """Compare classic TorchScript exporter inputs against ONNX Runtime on CPU."""
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise PolicyContractError(
            "torch, numpy and onnxruntime are required for strict ONNX parity validation"
        ) from exc

    dummy = torch_model.get_dummy_inputs()
    template = list(dummy) if isinstance(dummy, (tuple, list)) else [dummy]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    session_inputs = session.get_inputs()
    if len(template) != len(session_inputs):
        raise PolicyContractError(
            f"PyTorch exposes {len(template)} inputs but ONNX exposes {len(session_inputs)}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    maximum = 0.0
    with torch.inference_mode():
        for _ in range(samples):
            torch_inputs = [
                torch.randn(item.shape, dtype=item.dtype, generator=generator)
                if index == 0
                else torch.zeros_like(item)
                for index, item in enumerate(template)
            ]
            torch_result = torch_model(*torch_inputs)
            if isinstance(torch_result, (tuple, list)):
                torch_outputs = list(torch_result)
            else:
                torch_outputs = [torch_result]
            feed = {
                info.name: value.detach().cpu().numpy()
                for info, value in zip(session_inputs, torch_inputs, strict=True)
            }
            ort_outputs = session.run(None, feed)
            if len(torch_outputs) != len(ort_outputs):
                raise PolicyContractError("PyTorch and ONNX output counts differ")
            for expected, actual in zip(torch_outputs, ort_outputs, strict=True):
                error = float(
                    np.max(np.abs(expected.detach().cpu().numpy().astype(np.float64) - actual))
                )
                maximum = max(maximum, error)
    if maximum > ONNX_PARITY_LIMIT:
        raise PolicyContractError(
            f"PyTorch/ONNX parity failed: max abs error {maximum:.9g} > {ONNX_PARITY_LIMIT:g}"
        )
    return maximum


def build_contract(
    *,
    policy: PolicyEntry,
    descriptors: dict[str, list[dict[str, Any]]],
    checkpoint_path: str | Path,
    model_path: str | Path,
    config_sha256: str,
    parity_error: float,
    torchscript_path: str | Path,
    normalization_embedded: bool,
) -> PolicyContract:
    catalog = load_catalog()
    station = catalog.station(policy.station_id)
    onnx = inspect_onnx(model_path)
    recurrent = onnx["inputs"][1:] + onnx["outputs"][1:]
    dependencies: list[dict[str, str]] = []
    for dependency_id in policy.depends_on_policy_ids:
        dependency_bundle = candidate_bundle_path(dependency_id)
        dependency_contract = PolicyContract.load(
            dependency_bundle / "policy.contract.json",
            dependency_bundle / "policy.onnx",
        )
        dependency_torchscript = dependency_bundle / "policy.pt"
        if not dependency_torchscript.is_file():
            raise PolicyContractError(
                f"{policy.policy_id}: dependency '{dependency_id}' has no policy.pt"
            )
        dependency_torchscript_hash = sha256_file(dependency_torchscript)
        if dependency_contract.torchscript_sha256 != dependency_torchscript_hash:
            raise PolicyContractError(
                f"{policy.policy_id}: dependency '{dependency_id}' TorchScript hash mismatch"
            )
        dependencies.append(
            {
                "policy_id": dependency_id,
                "model_sha256": dependency_contract.model_sha256,
                "torchscript_sha256": dependency_torchscript_hash,
            }
        )
    contract = PolicyContract(
        policy_id=policy.policy_id,
        source_task_id=policy.source_task_id,
        station_id=station.station_id,
        robot_id=station.robot_id,
        adapter_id=station.adapter_id,
        checkpoint=Path(checkpoint_path).name,
        checkpoint_sha256=sha256_file(checkpoint_path),
        config_sha256=config_sha256,
        model_sha256=onnx["sha256"],
        onnx=onnx,
        normalization_embedded=normalization_embedded,
        recurrent_state=recurrent,
        torchscript_sha256=sha256_file(torchscript_path),
        dependencies=dependencies,
        observations=descriptors["observations"],
        actions=descriptors["actions"],
        timing={
            "source_physics_hz": policy.source_physics_hz,
            "deployment_physics_hz": policy.deployment_physics_hz,
            "policy_hz": policy.policy_hz,
            "source_sim_dt": 1.0 / policy.source_physics_hz,
            "deployment_sim_dt": 1.0 / policy.deployment_physics_hz,
            "policy_dt": 1.0 / policy.policy_hz,
            "deployment_decimation": policy.deployment_physics_hz // policy.policy_hz,
        },
        frame={
            "source": "isaac_rh_z_up",
            "deployment": "unity_lh_y_up",
            "position_mapping": ["x", "z", "y"],
            "quaternion_xyzw_mapping": ["x", "z", "y", "-w"],
            "never_extrapolate_authoritative_state": True,
        },
        physics_profile_id=station.physics_profile_id,
        assist_profile_id=station.assist_profile_id,
        evaluation={
            "primary_metric": policy.primary_metric,
            "minimum_isaac_score": policy.minimum_isaac_score,
            "minimum_unity_relative_score": policy.minimum_unity_relative_score,
            "task_thresholds": list(policy.task_thresholds),
            "torch_onnx_max_abs_error": parity_error,
            "torch_onnx_limit": ONNX_PARITY_LIMIT,
            "python_unity_limit": UNITY_PARITY_LIMIT,
            "observation_limit": OBSERVATION_PARITY_LIMIT,
            "seeded_scenarios": 100,
            "ready": False,
        },
    )
    contract.validate(model_path)
    return contract


def candidate_bundle_path(policy_id: str) -> Path:
    return (
        monorepo_root()
        / "Unity_XRPlayground"
        / "Assets"
        / "_Project"
        / "Features"
        / "Deployment"
        / "Bundles"
        / "Candidates"
        / policy_id
    )


def reference_bundle_path(policy_id: str) -> Path:
    """Return a Unity import path which promotion deliberately never scans."""
    return (
        monorepo_root()
        / "Unity_XRPlayground"
        / "Assets"
        / "_Project"
        / "Features"
        / "Deployment"
        / "Bundles"
        / "References"
        / policy_id
    )


def unity_robot_definition_path(robot_id: str) -> Path:
    """Canonical policy-independent Unity physics definition import path."""
    return (
        monorepo_root()
        / "Unity_XRPlayground"
        / "Assets"
        / "_Project"
        / "Features"
        / "Deployment"
        / "RobotDefinitions"
        / robot_id
        / "robot.definition.json"
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def stage_candidate_bundle(
    model_path: str | Path,
    contract_path: str | Path,
    robot_definition_path: str | Path,
    torchscript_path: str | Path,
    policy_id: str,
) -> Path:
    destination = candidate_bundle_path(policy_id)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_copy(Path(model_path), destination / "policy.onnx")
    _atomic_copy(Path(contract_path), destination / "policy.contract.json")
    _atomic_copy(Path(robot_definition_path), destination / "robot.definition.json")
    _atomic_copy(Path(torchscript_path), destination / "policy.pt")
    return destination


def stage_reference_bundle(
    model_path: str | Path,
    contract_path: str | Path,
    robot_definition_path: str | Path,
    torchscript_path: str | Path,
    policy_id: str,
) -> Path:
    """Stage old checkpoints for physics/inference inspection, never promotion."""
    destination = reference_bundle_path(policy_id)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_copy(Path(model_path), destination / "policy.onnx")
    _atomic_copy(Path(contract_path), destination / "policy.contract.json")
    _atomic_copy(Path(robot_definition_path), destination / "robot.definition.json")
    _atomic_copy(Path(torchscript_path), destination / "policy.pt")
    marker = destination / "REFERENCE_ONLY.json"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{marker.name}.", suffix=".tmp", dir=destination
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "policy_id": policy_id,
                    "promotion_eligible": False,
                    "purpose": "reference checkpoint physics and inference inspection only",
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def stage_robot_definition(robot_definition_path: str | Path, robot_id: str) -> Path:
    destination = unity_robot_definition_path(robot_id)
    _atomic_copy(Path(robot_definition_path), destination)
    return destination


def export_robot_definition(
    *,
    descriptor_env: Any,
    task_id: str,
    export_dir: str | Path,
    copy_to_unity: bool,
) -> tuple[Path, Path | None]:
    """Export live articulation physics without requiring or loading a checkpoint."""
    catalog = load_catalog()
    policy = catalog.policy_for_task(task_id)
    station = catalog.station(policy.station_id)
    robot_asset = catalog.robot_asset(station.robot_id)
    output = Path(export_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    definition_path = output / "robot.definition.json"
    definition_from_environment(
        descriptor_env,
        station.robot_id,
        provenance=robot_asset.provenance,
        redistribution_license=(
            robot_asset.redistribution_license
            if robot_asset.redistribution_verified
            else "UNVERIFIED"
        ),
    ).write_atomic(definition_path)
    staged = (
        stage_robot_definition(definition_path, station.robot_id)
        if copy_to_unity
        else None
    )
    return definition_path, staged


def export_runner_bundle(
    *,
    runner: Any,
    descriptor_env: Any,
    env_cfg: Any,
    agent_cfg: Any,
    task_id: str,
    checkpoint_path: str | Path,
    export_dir: str | Path,
    copy_to_unity: bool,
    reference_only: bool = False,
) -> tuple[Path, Path, float, Path | None]:
    """Export and parity-check, optionally staging an isolated reference or candidate."""
    from unity_onnx import (
        build_policy_onnx_module,
        export_policy_onnx_for_unity,
        export_policy_torchscript_for_isaac_dependency,
    )

    catalog = load_catalog()
    policy = catalog.policy_for_task(task_id)
    station = catalog.station(policy.station_id)
    descriptors = descriptors_from_environment(descriptor_env, policy)
    validate_descriptors(policy, descriptors)
    output = Path(export_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch_model = build_policy_onnx_module(runner)
    model_path = Path(
        export_policy_onnx_for_unity(
            runner, str(output), filename="policy.onnx", onnx_model=torch_model
        )
    )
    torchscript_path = Path(
        export_policy_torchscript_for_isaac_dependency(torch_model, str(output))
    )
    parity_error = validate_torch_onnx(torch_model, model_path)
    normalization_embedded = validate_normalization_embedding(torch_model, agent_cfg)
    contract = build_contract(
        policy=policy,
        descriptors=descriptors,
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        config_sha256=configuration_sha256(env_cfg, agent_cfg),
        parity_error=parity_error,
        torchscript_path=torchscript_path,
        normalization_embedded=normalization_embedded,
    )
    contract_path = output / "policy.contract.json"
    contract.write_atomic(contract_path)
    robot_definition_path = output / "robot.definition.json"
    robot_asset = catalog.robot_asset(station.robot_id)
    definition_from_environment(
        descriptor_env,
        station.robot_id,
        provenance=robot_asset.provenance,
        redistribution_license=(
            robot_asset.redistribution_license
            if robot_asset.redistribution_verified
            else "UNVERIFIED"
        ),
    ).write_atomic(robot_definition_path)
    staged = None
    if copy_to_unity:
        stage = stage_reference_bundle if reference_only else stage_candidate_bundle
        staged = stage(
            model_path,
            contract_path,
            robot_definition_path,
            torchscript_path,
            policy.policy_id,
        )
    return model_path, contract_path, parity_error, staged
