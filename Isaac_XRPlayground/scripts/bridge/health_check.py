"""Smoke-test a generated station contract and its bridge-v2 data flow."""

from __future__ import annotations

import argparse
import socket
import sys
import time

from XRPlayground.bridge.protocol import encode_message, make_envelope, try_decode_buffer
from XRPlayground.deployment.catalog import load_catalog
from XRPlayground.deployment.validation import validate_candidate


def receive_types(sock: socket.socket, deadline: float) -> set[str]:
    buffer = bytearray()
    message_types: set[str] = set()
    while time.monotonic() < deadline:
        try:
            data = sock.recv(65536)
        except socket.timeout:
            continue
        if not data:
            break
        buffer.extend(data)
        while True:
            envelope, buffer = try_decode_buffer(buffer)
            if envelope is None:
                break
            message_types.add(envelope["message_type"])
    return message_types


def check_live(host: str, station_id: str, timeout: float) -> list[str]:
    station = load_catalog().station(station_id)
    try:
        with socket.create_connection((host, station.port), timeout=timeout) as sock:
            sock.settimeout(0.25)
            envelope = make_envelope(
                "session_status",
                {"role": "health_check"},
                station_id=station.station_id,
                sequence=0,
                sim_time_s=0.0,
                frame_id="health_check",
            )
            sock.sendall(encode_message(envelope))
            received = receive_types(sock, time.monotonic() + timeout)
    except OSError as exc:
        return [f"port {host}:{station.port} unreachable: {exc}"]
    required = {"robot_state", "session_status"}
    if "objects_state" in station.bridge_payloads:
        required.add("objects_state")
    return [f"no flow for message_type '{item}'" for item in sorted(required - received)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    try:
        station = load_catalog().station(args.station)
    except ValueError as exc:
        print(f"FAIL catalog: {exc}")
        return 2
    failures: list[str] = []
    for policy_id in station.policy_ids:
        report = validate_candidate(policy_id)
        failures.extend(report.errors)
    if not args.offline:
        failures.extend(check_live(args.host, station.station_id, args.timeout))
    if failures:
        print(f"FAIL {station.station_id}")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS {station.station_id}: {args.host}:{station.port} schema=2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
