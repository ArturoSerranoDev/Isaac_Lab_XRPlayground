"""Station contract smoke test for the Unity ↔ Isaac TCP bridge.

Examples:
    python scripts/bridge/health_check.py --station conveyor_color --timeout 5
    python scripts/bridge/health_check.py --station ball_catch --offline
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

MONOREPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = MONOREPO_ROOT / "Unity_XRPlayground" / "Assets" / "Resources" / "XRPlayground" / "stations.json"
POLICIES_ROOT = MONOREPO_ROOT / "Unity_XRPlayground" / "Assets" / "_Project" / "Features" / "Policies"


def load_station(station_id: str) -> dict[str, Any]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for station in payload.get("stations", []):
        if station.get("id") == station_id:
            return station
    raise ValueError(f"unknown station '{station_id}'")


def check_policy(station: dict[str, Any]) -> list[str]:
    folder = station["policy_folder"]
    policy_dir = POLICIES_ROOT / folder
    sidecar = policy_dir / "policy.json"
    onnx = policy_dir / "policy.onnx"
    failures: list[str] = []
    if not sidecar.is_file():
        return [f"sidecar missing: {sidecar}"]
    # Unity can save TextAssets with a UTF-8 BOM on Windows.
    metadata = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    expected = (station["policy_task_id"], station["obs_dim"], station["action_dim"])
    actual = (metadata.get("task_id"), metadata.get("obs_dim"), metadata.get("action_dim"))
    if actual != expected:
        failures.append(f"policy.json contract {actual} != manifest {expected}")
    if not onnx.is_file() or onnx.stat().st_size == 0:
        failures.append(f"ONNX assignment unavailable: {onnx}")
    return failures


def send_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack("<I", len(encoded)) + encoded)


def receive_frames(sock: socket.socket, deadline: float) -> set[str]:
    buffer = bytearray()
    topics: set[str] = set()
    while time.monotonic() < deadline:
        try:
            data = sock.recv(65536)
        except socket.timeout:
            continue
        if not data:
            break
        buffer.extend(data)
        while len(buffer) >= 4:
            length = struct.unpack_from("<I", buffer)[0]
            if length <= 0 or length > 1 << 20:
                raise RuntimeError(f"invalid bridge frame length {length}")
            if len(buffer) < 4 + length:
                break
            frame = bytes(buffer[4 : 4 + length])
            del buffer[: 4 + length]
            try:
                topic = json.loads(frame).get("topic")
            except json.JSONDecodeError:
                continue
            if topic:
                topics.add(topic)
    return topics


def check_live(host: str, station: dict[str, Any], timeout: float) -> list[str]:
    failures: list[str] = []
    try:
        with socket.create_connection((host, int(station["port"])), timeout=timeout) as sock:
            sock.settimeout(0.25)
            send_frame(
                sock,
                {
                    "topic": "/xr/heartbeat",
                    "stamp_s": time.time(),
                    "frame_id": "health_check",
                    "data": {"role": "health_check", "unity_time": time.time()},
                },
            )
            topics = receive_frames(sock, time.monotonic() + timeout)
    except OSError as error:
        return [f"port {host}:{station['port']} unreachable: {error}"]

    for topic in station["required_topics"]:
        if topic not in topics:
            failures.append(f"no flow on required topic {topic}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one XRPlayground bridge station contract.")
    parser.add_argument("--station", required=True, help="manifest station ID, e.g. ball_catch")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--offline", action="store_true", help="only validate the manifest and ONNX sidecar package")
    args = parser.parse_args()

    try:
        station = load_station(args.station)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"FAIL manifest: {error}")
        return 2

    failures = check_policy(station)
    if not args.offline:
        failures.extend(check_live(args.host, station, args.timeout))

    if failures:
        print(f"FAIL {station['id']}")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"PASS {station['id']}: port={args.host}:{station['port']}; "
        f"topics={len(station['required_topics'])}; obs={station['obs_dim']}; action={station['action_dim']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
