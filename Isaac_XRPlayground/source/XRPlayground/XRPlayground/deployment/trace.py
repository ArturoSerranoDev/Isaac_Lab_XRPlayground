# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed golden trace writer shared by evaluation runners."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


GOLDEN_TRACE_FIELDS = {
    "schema_version",
    "station_id",
    "policy_id",
    "reset_seed",
    "sim_time_s",
    "reset_state",
    "observations",
    "raw_actions",
    "processed_actions",
    "joint_state",
    "root_state",
    "object_state",
    "contacts",
    "assist",
    "loop_closures",
    "task_score",
}


class GoldenTraceError(ValueError):
    """Raised when an evaluation sample is incomplete or non-numeric."""


def _validate_finite(value: Any, path: str = "sample") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GoldenTraceError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def _validate_sample(sample: Any, path: str = "sample") -> None:
    if not isinstance(sample, dict):
        raise GoldenTraceError(f"{path} must be an object")
    missing = GOLDEN_TRACE_FIELDS - set(sample)
    if missing:
        raise GoldenTraceError(f"{path} is missing {sorted(missing)}")
    if sample.get("schema_version") != 1:
        raise GoldenTraceError(f"{path}.schema_version must equal 1")
    for name in ("station_id", "policy_id"):
        if not isinstance(sample.get(name), str) or not sample[name].strip():
            raise GoldenTraceError(f"{path}.{name} must be a non-empty string")
    if isinstance(sample.get("reset_seed"), bool) or not isinstance(
        sample.get("reset_seed"), int
    ):
        raise GoldenTraceError(f"{path}.reset_seed must be an integer")
    if not isinstance(sample.get("reset_state"), dict) or not isinstance(
        sample.get("joint_state"), dict
    ) or not isinstance(sample.get("root_state"), dict):
        raise GoldenTraceError(f"{path} reset/joint/root state must be objects")
    for name in (
        "observations", "raw_actions", "processed_actions", "object_state",
        "contacts", "assist", "loop_closures",
    ):
        if not isinstance(sample.get(name), list):
            raise GoldenTraceError(f"{path}.{name} must be an array")
    _validate_finite(sample, path)


class GoldenTraceWriter:
    """Write JSONL atomically so partial evaluations cannot look promotable."""

    def __init__(self, output: str | Path):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            prefix=f".{self.output.name}.", suffix=".tmp", dir=self.output.parent
        )
        self._temporary = Path(temporary)
        self._stream = os.fdopen(handle, "w", encoding="utf-8", newline="\n")
        self.count = 0

    def append(self, sample: dict[str, Any]) -> None:
        _validate_sample(sample)
        self._stream.write(json.dumps(sample, sort_keys=True, separators=(",", ":")))
        self._stream.write("\n")
        self.count += 1

    def close(self, *, promote: bool = True) -> None:
        if self._stream.closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        if promote and self.count > 0:
            os.replace(self._temporary, self.output)
        else:
            self._temporary.unlink(missing_ok=True)

    def __enter__(self) -> "GoldenTraceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(promote=exc is None)


def load_golden_trace(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    samples: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            sample = json.loads(line)
            _validate_sample(sample, f"line {line_number}")
            samples.append(sample)
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenTraceError(f"Could not load '{source}': {exc}") from exc
    if not samples:
        raise GoldenTraceError("golden trace is empty")
    return samples
