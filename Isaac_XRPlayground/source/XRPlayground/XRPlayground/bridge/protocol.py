# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Length-prefixed JSON envelope protocol for the XR bridge."""

from __future__ import annotations

import json
import struct
import time
from typing import Any

HEADER = struct.Struct("<I")
MAX_MESSAGE_BYTES = 1 << 20


def make_envelope(topic: str, data: dict[str, Any], frame_id: str = "isaac_env", stamp_s: float | None = None) -> dict[str, Any]:
    return {
        "topic": topic,
        "stamp_s": float(time.time() if stamp_s is None else stamp_s),
        "frame_id": frame_id,
        "data": data,
    }


def encode_message(envelope: dict[str, Any], max_message_bytes: int = MAX_MESSAGE_BYTES) -> bytes:
    payload = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    if len(payload) > max_message_bytes:
        raise ValueError(f"XR bridge message is {len(payload)} bytes; limit is {max_message_bytes}")
    return HEADER.pack(len(payload)) + payload


def try_decode_buffer(
    buffer: bytearray, max_message_bytes: int = MAX_MESSAGE_BYTES
) -> tuple[dict[str, Any] | None, bytearray]:
    """Pop one message from buffer if a full frame is available."""
    if len(buffer) < HEADER.size:
        return None, buffer
    (length,) = HEADER.unpack_from(buffer, 0)
    if length <= 0 or length > max_message_bytes:
        buffer.clear()
        raise ValueError(f"Invalid XR bridge frame length: {length}")
    total = HEADER.size + length
    if len(buffer) < total:
        return None, buffer
    raw = bytes(buffer[HEADER.size : total])
    del buffer[:total]
    return json.loads(raw.decode("utf-8")), buffer
