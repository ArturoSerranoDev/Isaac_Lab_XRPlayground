# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Version-2 length-prefixed JSON protocol shared by every station."""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from collections import defaultdict
from typing import Any

SCHEMA_VERSION = 2
HEADER = struct.Struct("<I")
MAX_MESSAGE_BYTES = 1 << 20

_sequence_lock = threading.Lock()
_sequences: defaultdict[str, int] = defaultdict(int)


def message_type_from_topic(topic: str) -> str:
    """Collapse station-specific topic names into the v2 generic message taxonomy."""
    tail = topic.rsplit("/", 1)[-1].lower()
    if tail == "robot_state":
        return "robot_state"
    if tail in {"ball_state", "balls_state", "objects_state"}:
        return "objects_state"
    if tail in {"heartbeat", "session_status"}:
        return "session_status"
    return tail


def _next_sequence(station_id: str) -> int:
    with _sequence_lock:
        value = _sequences[station_id]
        _sequences[station_id] = value + 1
    return value


def make_envelope(
    message_type: str,
    payload: dict[str, Any],
    *,
    station_id: str,
    sequence: int | None = None,
    sim_time_s: float | None = None,
    frame_id: str = "isaac_env",
) -> dict[str, Any]:
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "station_id": station_id,
        "sequence": _next_sequence(station_id) if sequence is None else int(sequence),
        "sim_time_s": float(time.monotonic() if sim_time_s is None else sim_time_s),
        "frame_id": frame_id,
        "message_type": message_type_from_topic(message_type),
        "payload": payload,
    }
    validate_envelope(envelope)
    return envelope


def validate_envelope(envelope: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "station_id",
        "sequence",
        "sim_time_s",
        "frame_id",
        "message_type",
        "payload",
    }
    missing = required - set(envelope)
    if missing:
        raise ValueError(f"Bridge v2 envelope missing {sorted(missing)}")
    if set(envelope) != required:
        raise ValueError(f"Bridge v2 envelope has unsupported fields {sorted(set(envelope) - required)}")
    if envelope["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bridge schema {envelope['schema_version']}; expected {SCHEMA_VERSION}"
        )
    for key in ("station_id", "frame_id", "message_type"):
        if not isinstance(envelope[key], str) or not envelope[key]:
            raise ValueError(f"Bridge field '{key}' must be a non-empty string")
    if not isinstance(envelope["sequence"], int) or envelope["sequence"] < 0:
        raise ValueError("Bridge sequence must be a non-negative integer")
    sim_time = envelope["sim_time_s"]
    if not isinstance(sim_time, (int, float)) or not math.isfinite(float(sim_time)):
        raise ValueError("Bridge sim_time_s must be finite")
    if not isinstance(envelope["payload"], dict):
        raise ValueError("Bridge payload must be an object")


class SequenceGate:
    """Reject duplicate and out-of-order authoritative messages per station."""

    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def accept(self, envelope: dict[str, Any]) -> bool:
        validate_envelope(envelope)
        station_id = envelope["station_id"]
        sequence = envelope["sequence"]
        previous = self._last.get(station_id, -1)
        if sequence <= previous:
            return False
        self._last[station_id] = sequence
        return True


def encode_message(envelope: dict[str, Any], max_message_bytes: int = MAX_MESSAGE_BYTES) -> bytes:
    validate_envelope(envelope)
    payload = json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > max_message_bytes:
        raise ValueError(f"XR bridge message is {len(payload)} bytes; limit is {max_message_bytes}")
    return HEADER.pack(len(payload)) + payload


def try_decode_buffer(
    buffer: bytearray, max_message_bytes: int = MAX_MESSAGE_BYTES
) -> tuple[dict[str, Any] | None, bytearray]:
    """Pop and validate one message if a complete frame is available."""
    if len(buffer) < HEADER.size:
        return None, buffer
    (length,) = HEADER.unpack_from(buffer, 0)
    if length <= 0 or length > max_message_bytes:
        buffer.clear()
        raise ValueError(f"Invalid XR bridge frame length: {length}")
    total = HEADER.size + length
    if len(buffer) < total:
        return None, buffer
    raw = bytes(buffer[HEADER.size:total])
    del buffer[:total]
    envelope = json.loads(raw.decode("utf-8"))
    validate_envelope(envelope)
    return envelope, buffer
