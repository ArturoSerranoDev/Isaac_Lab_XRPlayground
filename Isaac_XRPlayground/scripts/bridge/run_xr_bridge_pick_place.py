# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""XR TCP bridge for Agibot table pick-and-place (Unity ↔ Isaac, port 9092)."""

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
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="XR TCP bridge for Pick-Place Table (Unity ↔ Isaac).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Template-Xrplayground-Pick-Place-Table-Direct-v0")
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=9092)
parser.add_argument("--publish_hz", type=float, default=60.0)
parser.add_argument("--action_mode", type=str, default="zero", choices=["zero", "random"])
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import XRPlayground.tasks  # noqa: F401
from XRPlayground.bridge.names_agibot import (
    MODE_AWAIT_SPAWN,
    MODE_MIRROR,
    TOPIC_HEARTBEAT,
    TOPIC_SESSION_COMMAND,
    TOPIC_SESSION_STATUS,
    TOPIC_SPAWN,
)
from XRPlayground.bridge.pick_place_bridge import PickPlaceBridgeAdapter
from XRPlayground.bridge.protocol import make_envelope, message_type_from_topic
from XRPlayground.bridge.tcp_server import RosTcpServer


def _actions(base_env, mode: str) -> torch.Tensor:
    shape = (base_env.num_envs, int(base_env.cfg.action_space))
    if mode == "random":
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
        adapter = PickPlaceBridgeAdapter(base_env, env_id=0)
        session_mode = MODE_MIRROR
        gym_env.reset()

        server = RosTcpServer(args_cli.host, args_cli.port)
        server.start()

        sim = base_env.sim
        step_dt = float(base_env.step_dt)
        publish_period = 1.0 / max(args_cli.publish_hz, 1.0)
        last_publish = 0.0
        last_heartbeat = 0.0

        print(f"[XR PickPlace Bridge] task={args_cli.task} port={args_cli.port}")

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
                        if mode in (MODE_MIRROR, MODE_AWAIT_SPAWN):
                            session_mode = str(mode)
                            print(f"[XR PickPlace Bridge] mode → {session_mode}")
                            gym_env.reset()
                    elif topic == message_type_from_topic(TOPIC_SPAWN):
                        adapter.handle_spawn(data)

                actions = _actions(base_env, args_cli.action_mode)
                _, _, _, _, _ = gym_env.step(actions)

                now = time.time()
                if now - last_publish >= publish_period:
                    sim_time_s = float(base_env.episode_length_buf[0].item()) * float(base_env.cfg.sim.dt)
                    server.publish(adapter.build_robot_state_envelope(stamp_s=sim_time_s))
                    server.publish(adapter.build_objects_state_envelope(stamp_s=sim_time_s))
                    last_publish = now
                if now - last_heartbeat >= 1.0:
                    server.publish(
                        make_envelope(
                            TOPIC_HEARTBEAT,
                            {"role": "isaac_pick_place", "sim_time": float(base_env.episode_length_buf[0].item())},
                            station_id="pick_place_table",
                            sim_time_s=float(base_env.episode_length_buf[0].item()) * float(base_env.cfg.sim.dt),
                        )
                    )
                    server.publish(
                        make_envelope(
                            TOPIC_SESSION_STATUS,
                            {
                                "mode": session_mode,
                                "phase": "mirroring",
                                "policy_loaded": False,
                                "task": args_cli.task,
                            },
                            station_id="pick_place_table",
                            sim_time_s=float(base_env.episode_length_buf[0].item()) * float(base_env.cfg.sim.dt),
                        )
                    )
                    last_heartbeat = now

                if args_cli.real_time:
                    time.sleep(max(0.0, step_dt - (time.time() - now)))
        except KeyboardInterrupt:
            print("[XR PickPlace Bridge] stopped.")
        finally:
            server.stop()
            gym_env.close()


if __name__ == "__main__":
    main()
