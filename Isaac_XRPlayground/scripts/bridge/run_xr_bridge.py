# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run ball-catch with a TCP ROS-like bridge for Unity XR.

Modes (Unity /xr/session_command):
  - mirror: stream Isaac robot; Isaac owns throws / policy (Unity ball ignored)
  - await_throw: hold ready until Unity throw_event, then run policy on that ball
"""

from __future__ import annotations

import argparse
import contextlib
import json
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

parser = argparse.ArgumentParser(description="XR TCP bridge for Ball Catch (Unity ↔ Isaac).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (use 1 for bridge).")
parser.add_argument("--task", type=str, default="Template-Xrplayground-Ball-Catch-Direct-v0")
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Cap sim to wall-clock step_dt (default on). Use --no-real-time to run as fast as Kit allows.",
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=9090)
parser.add_argument("--publish_hz", type=float, default=60.0, help="Robot/ball state publish rate.")
parser.add_argument("--log_robot", action="store_true", help="Print robot_state summaries.")
parser.add_argument(
    "--mode",
    type=str,
    default="mirror",
    choices=["mirror", "await_throw"],
    help="Initial session mode (Unity UI can change it).",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="zero",
    choices=["zero", "random", "policy"],
    help="Fallback actions when no skrl policy is loaded.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Optional skrl checkpoint for policy mode.")
parser.add_argument("--agent", type=str, default="skrl_cfg_entry_point")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import XRPlayground.tasks  # noqa: F401
from XRPlayground.bridge.ball_catch_bridge import BallCatchBridgeAdapter
from XRPlayground.bridge.names import (
    MODE_AWAIT_THROW,
    MODE_MIRROR,
    TOPIC_BALL_STATE,
    TOPIC_HEARTBEAT,
    TOPIC_SESSION_COMMAND,
    TOPIC_SESSION_STATUS,
)
from XRPlayground.bridge.protocol import make_envelope, message_type_from_topic
from XRPlayground.bridge.tcp_server import RosTcpServer


class SessionState:
    def __init__(self, mode: str):
        self.mode = mode
        # await_throw phases: waiting | catching
        self.phase = "mirroring" if mode == MODE_MIRROR else "waiting"
        self.policy_loaded = False
        self._was_grasped = False

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_MIRROR, MODE_AWAIT_THROW):
            print(f"[XR Bridge] Unknown mode '{mode}', ignoring.")
            return
        self.mode = mode
        self.phase = "mirroring" if mode == MODE_MIRROR else "waiting"
        print(f"[XR Bridge] Session mode → {self.mode} (phase={self.phase})")


def _try_load_policy(env, env_cfg, args_cli):
    """Return (wrapped_env, runner, obs, states) or (env, None, None, None)."""
    ckpt = args_cli.checkpoint
    if not ckpt and args_cli.action_mode != "policy":
        return env, None, None, None

    try:
        import skrl
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from packaging import version

        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        else:
            from skrl.utils.runner.jax import Runner
    except Exception as exc:  # noqa: BLE001
        print(f"[XR Bridge] skrl policy unavailable ({exc}); using {args_cli.action_mode} actions.")
        return env, None, None, None

    agent_ep = args_cli.agent
    env_cfg2, experiment_cfg = resolve_task_config(args_cli.task, agent_ep)
    # Keep live env; only need experiment cfg for Runner
    _ = env_cfg2

    if not ckpt:
        log_root = os.path.abspath(
            os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
        )
        try:
            ckpt = get_checkpoint_path(log_root, run_dir=".*_ppo_torch", other_dirs=["checkpoints"])
        except Exception as exc:  # noqa: BLE001
            print(f"[XR Bridge] No checkpoint found ({exc}).")
            return env, None, None, None

    ckpt = os.path.abspath(ckpt)
    if not os.path.isfile(ckpt):
        print(f"[XR Bridge] Checkpoint not found: {ckpt}")
        return env, None, None, None

    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    wrapped = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(wrapped, experiment_cfg)
    print(f"[XR Bridge] Loading policy: {ckpt}")
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
        adapter = BallCatchBridgeAdapter(base_env, env_id=0)

        step_env, runner, obs, states = _try_load_policy(gym_env, env_cfg, args_cli)
        session = SessionState(args_cli.mode)
        session.policy_loaded = runner is not None

        server = RosTcpServer(args_cli.host, args_cli.port)
        server.start()

        if runner is None:
            gym_env.reset()
        if session.mode == MODE_AWAIT_THROW:
            adapter.reset_robot_hold()

        sim = base_env.sim
        step_dt = float(base_env.step_dt)
        publish_period = 1.0 / max(args_cli.publish_hz, 1.0)
        last_publish = 0.0
        last_heartbeat = 0.0
        last_status = 0.0

        print(
            f"[XR Bridge] task={args_cli.task} mode={session.mode} "
            f"policy={'yes' if session.policy_loaded else 'no'} port={args_cli.port}"
        )

        try:
            while True:
                if sim.visualizers and hasattr(sim, "is_playing") and not sim.is_playing():
                    time.sleep(0.01)
                    continue

                # --- ingest Unity messages ---
                pending_throw: dict[str, Any] | None = None
                for msg in server.pop_messages():
                    topic = msg.get("message_type")
                    data = msg.get("payload") or {}
                    if topic == message_type_from_topic(TOPIC_SESSION_COMMAND):
                        mode = data.get("mode")
                        if mode:
                            session.set_mode(str(mode))
                            if session.mode == MODE_AWAIT_THROW:
                                if runner is not None:
                                    obs, _ = step_env.reset()
                                    states = step_env.state()
                                else:
                                    gym_env.reset()
                                adapter.reset_robot_hold()
                            elif session.mode == MODE_MIRROR:
                                if runner is not None:
                                    obs, _ = step_env.reset()
                                    states = step_env.state()
                                else:
                                    gym_env.reset()
                        command = str(data.get("command") or "")
                        if command == "reset":
                            if not mode:
                                if runner is not None:
                                    obs, _ = step_env.reset()
                                    states = step_env.state()
                                else:
                                    gym_env.reset()
                            adapter.reset_robot_hold()
                        elif command == "launch_ball":
                            try:
                                parameters = json.loads(data.get("parameters_json") or "{}")
                                position = parameters["position"]
                                velocity = parameters["velocity"]
                                if len(position) != 3 or len(velocity) != 3:
                                    raise ValueError("position and velocity must contain three values")
                                adapter.apply_ball_state(
                                    {
                                        "position": position,
                                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                                        "linear_velocity": velocity,
                                        "angular_velocity": [0.0, 0.0, 0.0],
                                        "grasped": False,
                                    }
                                )
                            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                                print(f"[XR Bridge] rejected launch_ball command: {exc}")
                    elif topic == message_type_from_topic(TOPIC_BALL_STATE):
                        # Ignore our own echo / Isaac-sourced packets from other tools
                        if str(data.get("source", "unity")).lower() == "isaac":
                            continue
                        if data.get("objects"):
                            data = next(
                                (item for item in data["objects"] if item.get("id") == "ball"),
                                data["objects"][0],
                            )
                        if session.mode == MODE_MIRROR:
                            continue  # Isaac owns the ball in mirror mode
                        grasped = bool(data.get("grasped", False))
                        throw_event = bool(data.get("throw_event", False))
                        released = (session._was_grasped and not grasped) or throw_event
                        session._was_grasped = grasped
                        if session.phase == "waiting":
                            # Replicate Unity ball into Isaac while player holds / aims
                            try:
                                adapter.apply_ball_state(data)
                            except Exception as exc:  # noqa: BLE001
                                print(f"[XR Bridge] ball replicate failed: {exc}")
                            if released and not grasped:
                                pending_throw = data
                        # catching: Isaac physics owns the ball
                    elif topic == message_type_from_topic(TOPIC_HEARTBEAT):
                        pass

                if pending_throw is not None:
                    try:
                        adapter.apply_ball_state(pending_throw)
                        session.phase = "catching"
                        print("[XR Bridge] Player throw received → catching")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[XR Bridge] throw apply failed: {exc}")

                # --- actions ---
                t0 = time.perf_counter()
                if session.mode == MODE_AWAIT_THROW and session.phase == "waiting":
                    actions = _fallback_actions(base_env, "zero")
                    if runner is not None:
                        # still step wrapped env with zeros via agent bypass
                        with torch.inference_mode():
                            obs, _, terminated, truncated, _ = step_env.step(actions)
                            states = step_env.state()
                            done = bool(terminated.any() or truncated.any()) if hasattr(terminated, "any") else False
                        if done:
                            obs, _ = step_env.reset()
                            states = step_env.state()
                            adapter.reset_robot_hold()
                    else:
                        obs_g, _, term, trunc, _ = gym_env.step(actions)
                        _ = obs_g
                        if bool(term.any() if hasattr(term, "any") else term) or bool(
                            trunc.any() if hasattr(trunc, "any") else trunc
                        ):
                            gym_env.reset()
                            adapter.reset_robot_hold()
                    # Ball pose comes from Unity replicate above (park only on mode/reset).
                else:
                    # mirror OR catching
                    if runner is not None:
                        with torch.inference_mode():
                            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                            actions = outputs[-1].get("mean_actions", outputs[0])
                            obs, _, terminated, truncated, _ = step_env.step(actions)
                            states = step_env.state()
                        done = False
                        if hasattr(terminated, "any"):
                            done = bool(terminated.any() or truncated.any())
                        if done:
                            obs, _ = step_env.reset()
                            states = step_env.state()
                            if session.mode == MODE_AWAIT_THROW:
                                session.phase = "waiting"
                                adapter.reset_robot_hold()
                                print("[XR Bridge] Episode done → waiting for next throw")
                    else:
                        actions = _fallback_actions(base_env, args_cli.action_mode if args_cli.action_mode != "policy" else "zero")
                        _, _, term, trunc, _ = gym_env.step(actions)
                        done = bool(term.any() if hasattr(term, "any") else term) or bool(
                            trunc.any() if hasattr(trunc, "any") else trunc
                        )
                        if done:
                            gym_env.reset()
                            if session.mode == MODE_AWAIT_THROW:
                                session.phase = "waiting"
                                adapter.reset_robot_hold()

                now = time.perf_counter()
                if now - last_publish >= publish_period:
                    sim_time_s = float(base_env.episode_length_buf[0].item()) * step_dt
                    envelope = adapter.build_robot_state_envelope(stamp_s=sim_time_s)
                    server.broadcast(envelope)
                    # Stream Isaac ball so Unity can visualize (mirror + post-throw catch)
                    try:
                        server.broadcast(adapter.build_ball_state_envelope(stamp_s=sim_time_s))
                    except Exception as exc:  # noqa: BLE001
                        print(f"[XR Bridge] ball publish failed: {exc}")
                    last_publish = now
                    if args_cli.log_robot:
                        ee = envelope["payload"]["ee"]["position"]
                        print(
                            f"[XR Bridge] {session.mode}/{session.phase} "
                            f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
                        )

                if now - last_heartbeat >= 1.0:
                    server.broadcast(
                        make_envelope(
                            TOPIC_HEARTBEAT,
                            {
                                "role": "isaac",
                                "sim_time": float(base_env.episode_length_buf[0].item()) * step_dt,
                            },
                            station_id="ball_catch",
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
                            },
                            station_id="ball_catch",
                            sim_time_s=float(base_env.episode_length_buf[0].item()) * step_dt,
                        )
                    )
                    last_status = now

                if args_cli.real_time:
                    sleep_t = step_dt - (time.perf_counter() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("[XR Bridge] Interrupted.")
        finally:
            server.stop()
            try:
                step_env.close()
            except Exception:  # noqa: BLE001
                gym_env.close()


if __name__ == "__main__":
    main()
