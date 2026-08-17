# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned, fail-closed ONNX deployment contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


POLICY_CONTRACT_SCHEMA_VERSION = 2


class PolicyContractError(ValueError):
    """Raised when a policy bundle cannot be trusted for deployment."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PolicyContractError(f"{label} must be a finite number")
    return float(value)


def _validate_term(term: dict[str, Any], index: int, kind: str) -> None:
    for key in ("name", "shape", "dtype"):
        if key not in term:
            raise PolicyContractError(f"{kind}[{index}] is missing '{key}'")
    if not isinstance(term["name"], str) or not term["name"]:
        raise PolicyContractError(f"{kind}[{index}].name must be non-empty")
    shape = term["shape"]
    if not isinstance(shape, list) or not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise PolicyContractError(f"{kind}[{index}].shape must contain positive integers")
    size = math.prod(shape)
    for key, expected in (
        ("scale_values", size),
        ("offset_values", size),
        ("velocity_scale", len(term.get("names", []))),
    ):
        if key not in term:
            continue
        values = term[key]
        if not isinstance(values, list) or len(values) != expected:
            raise PolicyContractError(
                f"{kind}[{index}].{key} must contain exactly {expected} values"
            )
        for item_index, value in enumerate(values):
            _finite_number(value, f"{kind}[{index}].{key}[{item_index}]")
    for key in ("scale", "offset"):
        if key in term:
            _finite_number(term[key], f"{kind}[{index}].{key}")
    if kind == "actions":
        for key in ("target_type", "integration"):
            if not isinstance(term.get(key), str) or not term[key]:
                raise PolicyContractError(f"{kind}[{index}] is missing '{key}'")


@dataclass
class PolicyContract:
    policy_id: str
    source_task_id: str
    station_id: str
    robot_id: str
    adapter_id: str
    checkpoint: str
    checkpoint_sha256: str
    config_sha256: str
    model_sha256: str
    onnx: dict[str, Any]
    observations: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    timing: dict[str, Any]
    frame: dict[str, Any]
    physics_profile_id: str
    assist_profile_id: str
    evaluation: dict[str, Any]
    normalization_embedded: bool = True
    recurrent_state: list[dict[str, Any]] = field(default_factory=list)
    torchscript_sha256: str = ""
    dependencies: list[dict[str, str]] = field(default_factory=list)
    schema_version: int = POLICY_CONTRACT_SCHEMA_VERSION

    @property
    def observation_dim(self) -> int:
        return sum(math.prod(term["shape"]) for term in self.observations)

    @property
    def action_dim(self) -> int:
        return sum(math.prod(term["shape"]) for term in self.actions)

    def validate(self, model_path: str | Path | None = None) -> None:
        if self.schema_version != POLICY_CONTRACT_SCHEMA_VERSION:
            raise PolicyContractError(
                f"Unsupported policy contract schema {self.schema_version}; "
                f"expected {POLICY_CONTRACT_SCHEMA_VERSION}"
            )
        for label in ("policy_id", "source_task_id", "station_id", "robot_id", "adapter_id"):
            if not getattr(self, label):
                raise PolicyContractError(f"{label} must be non-empty")
        if self.normalization_embedded is not True:
            raise PolicyContractError(
                "Unity deployment requires observation normalization embedded in ONNX"
            )
        for label, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("config_sha256", self.config_sha256),
            ("model_sha256", self.model_sha256),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise PolicyContractError(f"{label} must be a lowercase SHA-256")
        if self.torchscript_sha256 and (
            len(self.torchscript_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.torchscript_sha256)
        ):
            raise PolicyContractError("torchscript_sha256 must be a lowercase SHA-256")
        dependency_ids: set[str] = set()
        for dependency in self.dependencies:
            dependency_id = dependency.get("policy_id", "")
            if not dependency_id or dependency_id in dependency_ids:
                raise PolicyContractError("dependency policy IDs must be unique and non-empty")
            dependency_ids.add(dependency_id)
            for key in ("model_sha256", "torchscript_sha256"):
                value = dependency.get(key, "")
                if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                    raise PolicyContractError(f"dependency {dependency_id}.{key} is not a SHA-256")

        for index, term in enumerate(self.observations):
            _validate_term(term, index, "observations")
        for index, term in enumerate(self.actions):
            _validate_term(term, index, "actions")

        input_shape = self.onnx.get("input_shape")
        output_shape = self.onnx.get("output_shape")
        if self.onnx.get("sha256") != self.model_sha256:
            raise PolicyContractError("ONNX signature hash does not match model_sha256")
        if not self.onnx.get("input_name") or not self.onnx.get("output_name"):
            raise PolicyContractError("ONNX input and output tensor names must be explicit")
        if input_shape != [1, self.observation_dim]:
            raise PolicyContractError(
                f"ONNX input shape {input_shape} does not match observations [1, {self.observation_dim}]"
            )
        if output_shape != [1, self.action_dim]:
            raise PolicyContractError(
                f"ONNX output shape {output_shape} does not match actions [1, {self.action_dim}]"
            )
        inputs = self.onnx.get("inputs")
        outputs = self.onnx.get("outputs")
        for label, tensors in (("inputs", inputs), ("outputs", outputs)):
            if not isinstance(tensors, list) or not tensors:
                raise PolicyContractError(f"ONNX {label} must be a non-empty tensor list")
            names: set[str] = set()
            for index, tensor in enumerate(tensors):
                if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str) or \
                        not tensor["name"] or tensor["name"] in names:
                    raise PolicyContractError(f"ONNX {label}[{index}] has an invalid name")
                shape = tensor.get("shape")
                if not isinstance(shape, list) or not shape or any(
                    not isinstance(value, int) or value <= 0 for value in shape
                ):
                    raise PolicyContractError(
                        f"ONNX {label}[{index}] must have a static positive shape"
                    )
                names.add(tensor["name"])
        if inputs[0] != {"name": self.onnx["input_name"], "shape": input_shape} or \
                outputs[0] != {"name": self.onnx["output_name"], "shape": output_shape}:
            raise PolicyContractError("ONNX primary tensor descriptors are inconsistent")
        if len(inputs) != len(outputs):
            raise PolicyContractError(
                "Recurrent ONNX must expose one state output for every state input"
            )
        for index, (state_input, state_output) in enumerate(
            zip(inputs[1:], outputs[1:], strict=True), 1
        ):
            if state_input["shape"] != state_output["shape"]:
                raise PolicyContractError(
                    f"Recurrent ONNX state pair {index} has different input/output shapes"
                )
        if self.recurrent_state != inputs[1:] + outputs[1:]:
            raise PolicyContractError(
                "recurrent_state must exactly describe the ordered extra ONNX inputs and outputs"
            )
        if self.onnx.get("opset") != 15:
            raise PolicyContractError("Unity Inference Engine 2.3 bundles must use ONNX opset 15")

        source_dt = _finite_number(self.timing.get("source_sim_dt"), "timing.source_sim_dt")
        deployment_dt = _finite_number(
            self.timing.get("deployment_sim_dt"), "timing.deployment_sim_dt"
        )
        policy_dt = _finite_number(self.timing.get("policy_dt"), "timing.policy_dt")
        if source_dt <= 0 or deployment_dt <= 0 or policy_dt <= 0:
            raise PolicyContractError("All timing intervals must be positive")
        ratio = policy_dt / deployment_dt
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-6):
            raise PolicyContractError("policy_dt must be an integer multiple of deployment_sim_dt")

        if self.evaluation.get("seeded_scenarios") != 100:
            raise PolicyContractError("evaluation.seeded_scenarios must equal 100")
        thresholds = self.evaluation.get("task_thresholds")
        if not isinstance(thresholds, list) or not thresholds:
            raise PolicyContractError("evaluation.task_thresholds must be a non-empty list")
        threshold_names: set[str] = set()
        for index, threshold in enumerate(thresholds):
            if not isinstance(threshold, dict):
                raise PolicyContractError(f"evaluation.task_thresholds[{index}] must be an object")
            metric = threshold.get("metric")
            comparison = threshold.get("comparison")
            if not isinstance(metric, str) or not metric or metric in threshold_names:
                raise PolicyContractError("evaluation threshold metrics must be unique and non-empty")
            if comparison not in {"min", "max", "equal"}:
                raise PolicyContractError(f"evaluation threshold '{metric}' comparison is invalid")
            _finite_number(threshold.get("value"), f"evaluation threshold '{metric}'.value")
            threshold_names.add(metric)

        if model_path is not None:
            actual = sha256_file(model_path)
            if actual != self.model_sha256:
                raise PolicyContractError(
                    f"ONNX SHA-256 mismatch: contract={self.model_sha256}, actual={actual}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "source_task_id": self.source_task_id,
            "station_id": self.station_id,
            "robot_id": self.robot_id,
            "adapter_id": self.adapter_id,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "model_sha256": self.model_sha256,
            "onnx": self.onnx,
            "normalization_embedded": self.normalization_embedded,
            "recurrent_state": self.recurrent_state,
            "torchscript_sha256": self.torchscript_sha256,
            "dependencies": self.dependencies,
            "observations": self.observations,
            "actions": self.actions,
            "timing": self.timing,
            "frame": self.frame,
            "physics_profile_id": self.physics_profile_id,
            "assist_profile_id": self.assist_profile_id,
            "evaluation": self.evaluation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyContract":
        try:
            contract = cls(**data)
        except TypeError as exc:
            raise PolicyContractError(f"Invalid policy contract fields: {exc}") from exc
        contract.validate()
        return contract

    @classmethod
    def load(cls, path: str | Path, model_path: str | Path | None = None) -> "PolicyContract":
        contract_path = Path(path)
        try:
            data = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyContractError(f"Could not load '{contract_path}': {exc}") from exc
        contract = cls.from_dict(data)
        if model_path is not None:
            contract.validate(model_path)
        return contract

    def write_atomic(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
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
