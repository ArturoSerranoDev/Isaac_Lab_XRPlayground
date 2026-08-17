# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""XR TCP bridge for Balance Bot (Unity ↔ Isaac).

Mode:
  - mirror: Isaac drops balls; Unity visualizes tray + balls (port 9093)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="XR TCP bridge for Balance Bot (Unity ↔ Isaac).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Template-Xrplayground-Balance-Bot-Direct-v0")
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Cap to wall-clock step_dt.",
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=9093)
parser.add_argument("--publish_hz", type=float, default=60.0)
parser.add_argument("--log_robot", action="store_true")
parser.add_argument("--mode", type=str, default="mirror", choices=["mirror"])
parser.add_argument("--action_mode", type=str, default="zero", choices=["zero", "random"])
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import XRPlayground.tasks  # noqa: F401
from XRPlayground.bridge.balance_bot_bridge import BalanceBotBridgeAdapter
from XRPlayground.bridge.names_balance_bot import (
    MODE_MIRROR,
    TOPIC_HEARTBEAT,
    TOPIC_SESSION_COMMAND,
    TOPIC_SESSION_STATUS,
)
from XRPlayground.bridge.protocol import make_envelope, message_type_from_topic
from XRPlayground.bridge.tcp_server import RosTcpServer


class SessionState:
    def __init__(self, mode: str):
        self.mode = mode
        self.phase = "mirroring"
        self.policy_loaded = False

    def set_mode(self, mode: str) -> None:
        if mode != MODE_MIRROR:
            print(f"[XR BalanceBot Bridge] Unknown mode '{mode}', ignoring.")
            return
        self.mode = mode
        self.phase = "mirroring"
        print(f"[XR BalanceBot Bridge] Session mode → {self.mode}")


def _fallback_actions(base_env, action_mode: str):
    shape = getattr(base_env.action_space, "shape", None)
    if shape is None:
        shape = (base_env.num_envs, int(base_env.cfg.action_space))
    elif len(shape) == 1:
        shape = (base_env.num_envs, int(shape[0]))
    if action_mode == "random":
        return 2.0 * torch.rand(shape, device=base_env.device) - 1.0
    return torch.zeros(shape, device=base_env.device)


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = gym_env.unwrapped
        adapter = BalanceBotBridgeAdapter(base_env, env_id=0)
        session = SessionState(args_cli.mode)

        server = RosTcpServer(args_cli.host, args_cli.port)
        server.start()
        gym_env.reset()

        sim = base_env.sim
        step_dt = float(base_env.step_dt)
        publish_period = 1.0 / max(args_cli.publish_hz, 1.0)
        last_publish = 0.0
        last_heartbeat = 0.0
        last_status = 0.0

        print(
            f"[XR BalanceBot Bridge] task={args_cli.task} mode={session.mode} port={args_cli.port}"
        )

        try:
            while True:
                if sim.visualizers and hasattr(sim, "is_playing") and not sim.is_playing():
                    time.sleep(0.01)
                    continue

                for msg in server.pop_messages():
                    topic = msg.get("message_type")
                    data = msg.get("payload") or {}
                    if topic == message_type_from_topic(TOPIC_SESSION_COMMAND):
                        mode = data.get("mode")
                        if mode:
                            session.set_mode(str(mode))
                            gym_env.reset()

                t0 = time.perf_counter()
                actions = _fallback_actions(base_env, args_cli.action_mode)
                _, _, term, trunc, _ = gym_env.step(actions)
                done = bool(term.any() if hasattr(term, "any") else term) or bool(
                    trunc.any() if hasattr(trunc, "any") else trunc
                )
                if done:
                    gym_env.reset()

                now = time.perf_counter()
                if now - last_publish >= publish_period:
                    try:
                        sim_time_s = float(base_env.episode_length_buf[0].item()) * step_dt
                        server.broadcast(adapter.build_robot_state_envelope(stamp_s=sim_time_s))
                        server.broadcast(adapter.build_balls_state_envelope(stamp_s=sim_time_s))
                    except Exception as exc:  # noqa: BLE001
                        print(f"[XR BalanceBot Bridge] publish failed: {exc}")
                    last_publish = now
                    if args_cli.log_robot:
                        print(f"[XR BalanceBot Bridge] {session.mode}/{session.phase}")

                if now - last_heartbeat >= 1.0:
                    server.broadcast(
                        make_envelope(
                            TOPIC_HEARTBEAT,
                            {
                                "role": "isaac_balance_bot",
                                "sim_time": float(base_env.episode_length_buf[0].item()) * step_dt,
                            },
                            station_id="balance_bot",
                            sim_time_s=float(base_env.episode_length_buf[0].item()) * step_dt,
                        )
                    )
                    last_heartbeat = now

                if now - last_status >= 0.5:
                    server.broadcast(
                        make_envelope(
                            TOPIC_SESSION_STATUS,
                            {
                                "mode": session.mode,
                                "phase": session.phase,
                                "policy_loaded": session.policy_loaded,
                                "clients": server.client_count(),
                                "task": "balance_bot",
                            },
                            station_id="balance_bot",
                            sim_time_s=float(base_env.episode_length_buf[0].item()) * step_dt,
                        )
                    )
                    last_status = now

                if args_cli.real_time:
                    sleep_t = step_dt - (time.perf_counter() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("[XR BalanceBot Bridge] Interrupted.")
        finally:
            server.stop()
            gym_env.close()


if __name__ == "__main__":
    main()
