# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""XR TCP bridge for Conveyor Color Detection (Unity ↔ Isaac).

Modes:
  - mirror: Isaac auto-spawns colors; Unity visualizes robot + objects
  - await_spawn: Unity sends /xr/conveyor/spawn; Isaac runs policy on belt objects
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from typing import Any

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="XR TCP bridge for Conveyor Color (Unity ↔ Isaac).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Template-Xrplayground-Conveyor-Color-Direct-v0")
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Cap to wall-clock step_dt. Use --no-real-time for max speed.",
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=9091)
parser.add_argument("--publish_hz", type=float, default=60.0)
parser.add_argument("--log_robot", action="store_true")
parser.add_argument(
    "--mode",
    type=str,
    default="mirror",
    choices=["mirror", "await_spawn"],
)
parser.add_argument("--action_mode", type=str, default="zero", choices=["zero", "random", "policy"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--agent", type=str, default="skrl_cfg_entry_point")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import XRPlayground.tasks  # noqa: F401
from XRPlayground.bridge.conveyor_color_bridge import ConveyorColorBridgeAdapter
from XRPlayground.bridge.names_ur10e import (
    MODE_AWAIT_SPAWN,
    MODE_MIRROR,
    TOPIC_HEARTBEAT,
    TOPIC_SESSION_COMMAND,
    TOPIC_SESSION_STATUS,
    TOPIC_SPAWN,
)
from XRPlayground.bridge.protocol import make_envelope
from XRPlayground.bridge.tcp_server import RosTcpServer


class SessionState:
    def __init__(self, mode: str):
        self.mode = mode
        self.phase = "mirroring" if mode == MODE_MIRROR else "waiting"
        self.policy_loaded = False

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_MIRROR, MODE_AWAIT_SPAWN):
            print(f"[XR Conveyor Bridge] Unknown mode '{mode}', ignoring.")
            return
        self.mode = mode
        self.phase = "mirroring" if mode == MODE_MIRROR else "waiting"
        print(f"[XR Conveyor Bridge] Session mode → {self.mode} (phase={self.phase})")


def _try_load_policy(env, args_cli):
    ckpt = args_cli.checkpoint
    if not ckpt and args_cli.action_mode != "policy":
        return env, None, None, None
    try:
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from packaging import version  # noqa: F401

        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        else:
            from skrl.utils.runner.jax import Runner
    except Exception as exc:  # noqa: BLE001
        print(f"[XR Conveyor Bridge] skrl unavailable ({exc}).")
        return env, None, None, None

    env_cfg2, experiment_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    _ = env_cfg2
    if not ckpt:
        log_root = os.path.abspath(
            os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
        )
        try:
            ckpt = get_checkpoint_path(log_root, run_dir=".*_ppo_torch", other_dirs=["checkpoints"])
        except Exception as exc:  # noqa: BLE001
            print(f"[XR Conveyor Bridge] No checkpoint ({exc}).")
            return env, None, None, None
    ckpt = os.path.abspath(ckpt)
    if not os.path.isfile(ckpt):
        print(f"[XR Conveyor Bridge] Checkpoint missing: {ckpt}")
        return env, None, None, None

    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    wrapped = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(wrapped, experiment_cfg)
    print(f"[XR Conveyor Bridge] Loading policy: {ckpt}")
    runner.agent.load(ckpt)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    obs, _ = wrapped.reset()
    states = wrapped.state()
    return wrapped, runner, obs, states


def _fallback_actions(base_env, action_mode: str) -> torch.Tensor:
    if action_mode == "random":
        return 2.0 * torch.rand(base_env.action_space.shape, device=base_env.device) - 1.0
    return torch.zeros(base_env.action_space.shape, device=base_env.device)


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
        adapter = ConveyorColorBridgeAdapter(base_env, env_id=0)

        step_env, runner, obs, states = _try_load_policy(gym_env, args_cli)
        session = SessionState(args_cli.mode)
        session.policy_loaded = runner is not None
        adapter.set_auto_spawn(session.mode == MODE_MIRROR)

        server = RosTcpServer(args_cli.host, args_cli.port)
        server.start()

        if runner is None:
            gym_env.reset()
        if session.mode == MODE_AWAIT_SPAWN:
            adapter.reset_hold()
            adapter.set_auto_spawn(False)

        sim = base_env.sim
        step_dt = float(base_env.step_dt)
        publish_period = 1.0 / max(args_cli.publish_hz, 1.0)
        last_publish = 0.0
        last_heartbeat = 0.0
        last_status = 0.0

        print(
            f"[XR Conveyor Bridge] task={args_cli.task} mode={session.mode} "
            f"policy={'yes' if session.policy_loaded else 'no'} port={args_cli.port}"
        )

        try:
            while True:
                if sim.visualizers and hasattr(sim, "is_playing") and not sim.is_playing():
                    time.sleep(0.01)
                    continue

                for msg in server.pop_messages():
                    topic = msg.get("topic")
                    data = msg.get("data") or {}
                    if topic == TOPIC_SESSION_COMMAND:
                        mode = data.get("mode")
                        if mode:
                            session.set_mode(str(mode))
                            adapter.set_auto_spawn(session.mode == MODE_MIRROR)
                            if runner is not None:
                                obs, _ = step_env.reset()
                                states = step_env.state()
                            else:
                                gym_env.reset()
                            if session.mode == MODE_AWAIT_SPAWN:
                                adapter.reset_hold()
                                adapter.set_auto_spawn(False)
                    elif topic == TOPIC_SPAWN:
                        if session.mode != MODE_AWAIT_SPAWN:
                            continue
                        slot = adapter.apply_spawn(data)
                        session.phase = "running"
                        print(f"[XR Conveyor Bridge] Unity spawn → slot={slot} color={data.get('color')}")

                t0 = time.perf_counter()
                if runner is not None:
                    with torch.inference_mode():
                        outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                        actions = outputs[-1].get("mean_actions", outputs[0])
                        obs, _, terminated, truncated, _ = step_env.step(actions)
                        states = step_env.state()
                    done = bool(terminated.any() or truncated.any()) if hasattr(terminated, "any") else False
                    if done:
                        obs, _ = step_env.reset()
                        states = step_env.state()
                        if session.mode == MODE_AWAIT_SPAWN:
                            session.phase = "waiting"
                            adapter.set_auto_spawn(False)
                else:
                    actions = _fallback_actions(
                        base_env, args_cli.action_mode if args_cli.action_mode != "policy" else "zero"
                    )
                    _, _, term, trunc, _ = gym_env.step(actions)
                    done = bool(term.any() if hasattr(term, "any") else term) or bool(
                        trunc.any() if hasattr(trunc, "any") else trunc
                    )
                    if done:
                        gym_env.reset()
                        if session.mode == MODE_AWAIT_SPAWN:
                            session.phase = "waiting"

                now = time.perf_counter()
                if now - last_publish >= publish_period:
                    server.broadcast(adapter.build_robot_state_envelope(stamp_s=now))
                    server.broadcast(adapter.build_objects_state_envelope(stamp_s=now))
                    last_publish = now
                    if args_cli.log_robot:
                        print(f"[XR Conveyor Bridge] {session.mode}/{session.phase}")

                if now - last_heartbeat >= 1.0:
                    server.broadcast(
                        make_envelope(
                            TOPIC_HEARTBEAT,
                            {"role": "isaac_conveyor", "sim_time": float(base_env.episode_length_buf[0].item()) * step_dt},
                            stamp_s=now,
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
                                "task": "conveyor_color",
                            },
                            stamp_s=now,
                        )
                    )
                    last_status = now

                if args_cli.real_time:
                    sleep_t = step_dt - (time.perf_counter() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("[XR Conveyor Bridge] Interrupted.")
        finally:
            server.stop()
            try:
                step_env.close()
            except Exception:  # noqa: BLE001
                gym_env.close()


if __name__ == "__main__":
    main()
