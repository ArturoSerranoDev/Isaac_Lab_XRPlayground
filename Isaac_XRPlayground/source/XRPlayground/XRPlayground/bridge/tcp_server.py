# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Non-blocking TCP server that fans out / collects JSON envelopes."""

from __future__ import annotations

import select
import socket
import threading
from collections import deque
from typing import Any, Callable

from .protocol import encode_message, try_decode_buffer


class RosTcpServer:
    """Simple multi-client TCP hub (Isaac side)."""

    def __init__(self, host: str, port: int, on_message: Callable[[dict[str, Any]], None] | None = None):
        self.host = host
        self.port = port
        self.on_message = on_message
        self._sock: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._buffers: dict[socket.socket, bytearray] = {}
        self._lock = threading.Lock()
        self._inbox: deque[dict[str, Any]] = deque(maxlen=256)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(4)
        sock.setblocking(False)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="xr-bridge-tcp", daemon=True)
        self._thread.start()
        print(f"[XR Bridge] Listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()
            self._buffers.clear()
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def broadcast(self, envelope: dict[str, Any]) -> None:
        raw = encode_message(envelope)
        dead: list[socket.socket] = []
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                c.sendall(raw)
            except OSError:
                dead.append(c)
        if dead:
            with self._lock:
                for c in dead:
                    if c in self._clients:
                        self._clients.remove(c)
                    self._buffers.pop(c, None)
                    try:
                        c.close()
                    except OSError:
                        pass

    def pop_messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            while self._inbox:
                out.append(self._inbox.popleft())
        return out

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running:
            with self._lock:
                clients = list(self._clients)
            readable: list[socket.socket] = [self._sock] + clients
            try:
                ready, _, _ = select.select(readable, [], [], 0.05)
            except (OSError, ValueError):
                continue
            for s in ready:
                if s is self._sock:
                    self._accept()
                else:
                    self._recv(s)

    def _accept(self) -> None:
        assert self._sock is not None
        try:
            conn, addr = self._sock.accept()
        except BlockingIOError:
            return
        conn.setblocking(False)
        with self._lock:
            self._clients.append(conn)
            self._buffers[conn] = bytearray()
        print(f"[XR Bridge] Client connected: {addr}")

    def _recv(self, conn: socket.socket) -> None:
        try:
            chunk = conn.recv(65536)
        except BlockingIOError:
            return
        except OSError:
            chunk = b""
        if not chunk:
            with self._lock:
                if conn in self._clients:
                    self._clients.remove(conn)
                self._buffers.pop(conn, None)
            try:
                conn.close()
            except OSError:
                pass
            print("[XR Bridge] Client disconnected")
            return

        with self._lock:
            buf = self._buffers.setdefault(conn, bytearray())
            buf.extend(chunk)
            while True:
                msg, buf = try_decode_buffer(buf)
                self._buffers[conn] = buf
                if msg is None:
                    break
                self._inbox.append(msg)
                if self.on_message is not None:
                    self.on_message(msg)
