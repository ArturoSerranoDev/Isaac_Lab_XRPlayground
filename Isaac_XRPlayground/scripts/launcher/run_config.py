# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run configuration helpers for the XRPlayground launcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .project_config import device_from_config, physics_sync_from_config

ActionKind = Literal["train", "play", "demo"]
PhysicsSync = Literal["fabric", "usd"]


def _style():
    from .terminal_ui import Style

    return Style


@dataclass
class RunConfig:
    """Resolved settings for a single train / play / demo launch."""

    num_envs: int
    device: str
    physics_sync: PhysicsSync
    headless: bool | None = None
    real_time: bool = False
    max_iterations: int | None = None
    seed: int | None = None
    record_video: bool = False

    @property
    def use_fabric(self) -> bool:
        return self.physics_sync == "fabric"

    def physics_label(self) -> str:
        return "Fabric (GPU fast path)" if self.use_fabric else "USD (CPU, stage sync)"

    def summary_lines(self) -> list[str]:
        lines = [
            f"  num_envs     : {self.num_envs}",
            f"  device       : {self.device}",
            f"  physics      : {self.physics_label()}",
        ]
        if self.headless is not None:
            lines.append(f"  visualization: {'headless' if self.headless else 'Kit window'}")
        if self.max_iterations is not None:
            lines.append(f"  max_iters    : {self.max_iterations}")
        if self.seed is not None:
            lines.append(f"  seed         : {self.seed}")
        if self.real_time:
            lines.append("  real-time    : on")
        if self.record_video:
            lines.append("  record video : on")
        return lines


def default_num_envs(config: dict[str, Any], action: ActionKind, *, headless: bool | None = None) -> int:
    if action == "train":
        if headless is False:
            return int(config.get("num_envs_train_visual", 32))
        return int(config.get("num_envs_train", 512))
    if action == "play":
        return int(config.get("num_envs_play", 16))
    return int(config.get("num_envs_demo", 16))


def build_run_config(
    config: dict[str, Any],
    action: ActionKind,
    *,
    headless: bool | None = None,
) -> RunConfig:
    """Build a RunConfig from persisted launcher settings."""
    physics = physics_sync_from_config(config, action)
    device = device_from_config(config, action)
    if physics == "usd" and device.startswith("cuda"):
        device = "cpu"

    resolved_headless = headless
    if action == "train" and resolved_headless is None:
        resolved_headless = bool(config.get("headless_train", True))

    run = RunConfig(
        num_envs=default_num_envs(config, action, headless=resolved_headless),
        device=device,
        physics_sync=physics,  # type: ignore[arg-type]
        headless=resolved_headless if action == "train" else None,
        real_time=bool(config.get("real_time_play" if action == "play" else "real_time_demo", False)),
        max_iterations=int(config["max_iterations"]) if action == "train" else None,
        seed=int(config.get("seed", 42)),
        record_video=bool(config.get("record_video", False)) if action == "train" else False,
    )
    return run


def persist_run_config(config: dict[str, Any], action: ActionKind, run: RunConfig) -> None:
    """Write a RunConfig back into launcher JSON settings."""
    if action == "train":
        if run.headless is False:
            config["num_envs_train_visual"] = run.num_envs
            config["train_visual_mode"] = "visual"
            config["headless_train"] = False
        else:
            config["num_envs_train"] = run.num_envs
            config["train_visual_mode"] = "headless"
            config["headless_train"] = True
        if run.max_iterations is not None:
            config["max_iterations"] = run.max_iterations
        config["record_video"] = run.record_video
    elif action == "play":
        config["num_envs_play"] = run.num_envs
        config["real_time_play"] = run.real_time
    else:
        config["num_envs_demo"] = run.num_envs
        config["real_time_demo"] = run.real_time

    config[f"device_{action}"] = run.device
    config[f"physics_sync_{action}"] = run.physics_sync
    config["profile"] = "custom"


def append_sim_flags(command: list[str], run: RunConfig) -> list[str]:
    """Append --device and --disable_fabric when needed."""
    command = command + [f"--device={run.device}"]
    if not run.use_fabric:
        command.append("--disable_fabric")
    return command


def validate_run_config(run: RunConfig) -> list[str]:
    """Return user-facing warnings; may mutate run (e.g. force CPU for USD)."""
    warnings: list[str] = []
    if run.physics_sync == "usd" and run.device.startswith("cuda"):
        run.device = "cpu"
        warnings.append("USD mode requires CPU on this setup — switched device to cpu.")
    if run.physics_sync == "usd" and run.num_envs > 8:
        warnings.append("USD mode is slow — consider num_envs ≤ 8.")
    if run.physics_sync == "usd" and run.headless is False:
        warnings.append("USD mode is for inspecting the stage; Fabric is better for visual training.")
    if run.use_fabric and run.device == "cpu" and run.num_envs > 64:
        warnings.append("CPU + Fabric with many envs will be very slow.")
    return warnings


def print_run_config_summary(run: RunConfig, title: str) -> None:
    Style = _style()
    print(Style.paint(f"\n  {title}", Style.YELLOW, Style.BOLD))
    for line in run.summary_lines():
        print(line)


@dataclass
class RunConfigPrompts:
    """Callbacks used by the interactive run-config wizard."""

    prompt_int: Any
    prompt_yes_no: Any
    prompt_choice: Any


def select_run_config_interactive(
    config: dict[str, Any],
    action: ActionKind,
    prompts: RunConfigPrompts,
    *,
    title: str,
    headless: bool | None = None,
) -> RunConfig | None:
    """Ask whether to use saved defaults or customize; return RunConfig or None."""
    from .terminal_ui import clear, print_banner, pause

    Style = _style()
    saved = build_run_config(config, action, headless=headless)

    clear()
    print_banner()
    print(Style.paint(f"\n  {title}", Style.YELLOW, Style.BOLD))
    print(Style.paint("  Saved defaults for this action:", Style.DIM))
    for line in saved.summary_lines():
        print(line)
    print("\n  [1] Use saved defaults")
    print("  [2] Customize for this run")
    print("  [0] Cancel")

    raw = input("\n  Choice [1]: ").strip() or "1"
    if raw == "0":
        return None
    if raw == "1":
        run = RunConfig(
            num_envs=saved.num_envs,
            device=saved.device,
            physics_sync=saved.physics_sync,
            headless=saved.headless,
            real_time=saved.real_time,
            max_iterations=saved.max_iterations,
            seed=saved.seed,
            record_video=saved.record_video,
        )
        for warning in validate_run_config(run):
            print(Style.paint(f"\n  ! {warning}", Style.YELLOW))
        return run

    # --- customize ---
    run = RunConfig(
        num_envs=saved.num_envs,
        device=saved.device,
        physics_sync=saved.physics_sync,
        headless=saved.headless,
        real_time=saved.real_time,
        max_iterations=saved.max_iterations,
        seed=saved.seed,
        record_video=saved.record_video,
    )

    print(Style.paint("\n  Customize run", Style.CYAN, Style.BOLD))

    if action == "train" and run.headless is not None:
        view = prompts.prompt_choice(
            "Training window:",
            [("headless", "Headless — fastest"), ("visual", "Visual — Kit window")],
            "headless" if run.headless else "visual",
        )
        run.headless = view == "headless"
        run.num_envs = default_num_envs(config, action, headless=run.headless)

    run.num_envs = prompts.prompt_int("  num_envs", run.num_envs, minimum=1)

    run.physics_sync = prompts.prompt_choice(  # type: ignore[assignment]
        "Physics sync:",
        [
            ("fabric", "Fabric — GPU fast path (recommended)"),
            ("usd", "USD — updates stage transforms (CPU only, slow)"),
        ],
        run.physics_sync,
    )

    if run.physics_sync == "usd":
        run.device = "cpu"
        print(Style.paint("  device locked to cpu for USD mode.", Style.DIM))
    else:
        run.device = prompts.prompt_choice(
            "Device:",
            [("cuda:0", "cuda:0 — GPU (recommended)"), ("cpu", "cpu — slower")],
            "cuda:0" if saved.device.startswith("cuda") else "cpu",
        )

    if action == "train" and run.max_iterations is not None:
        run.max_iterations = prompts.prompt_int("  max_iterations", run.max_iterations, minimum=1)
        run.record_video = prompts.prompt_yes_no("  Record video while training?", run.record_video)

    if action == "play":
        run.real_time = prompts.prompt_yes_no("  Real-time playback?", run.real_time)
    elif action == "demo":
        run.real_time = prompts.prompt_yes_no("  Real-time demo?", run.real_time)

    if run.seed is not None:
        run.seed = prompts.prompt_int("  seed", run.seed, minimum=0)

    for warning in validate_run_config(run):
        print(Style.paint(f"\n  ! {warning}", Style.YELLOW))

    if run.physics_sync == "usd":
        if not prompts.prompt_yes_no("  USD mode can crash with GPU — continue with CPU?", default=True):
            return None

    persist_run_config(config, action, run)
    print_run_config_summary(run, "Final run configuration")
    pause("\n  Press Enter to continue...")
    return run


def edit_full_config_menu(config: dict[str, Any], prompts: RunConfigPrompts) -> None:
    """Settings sub-menu: edit all run parameters in one place."""
    from .terminal_ui import clear, pause, print_banner

    Style = _style()
    while True:
        clear()
        print_banner()
        print(Style.paint("\n  Full run configuration", Style.YELLOW, Style.BOLD))
        print(Style.paint("  Values used as defaults for Train / Play / Demo.\n", Style.DIM))

        rows = [
            ("Train headless num_envs", str(config.get("num_envs_train", 512))),
            ("Train visual num_envs", str(config.get("num_envs_train_visual", 32))),
            ("Play num_envs", str(config.get("num_envs_play", 16))),
            ("Demo num_envs", str(config.get("num_envs_demo", 16))),
            ("Train device", config.get("device_train", config.get("device", "cuda:0"))),
            ("Play device", config.get("device_play", config.get("device", "cuda:0"))),
            ("Demo device", config.get("device_demo", config.get("device", "cuda:0"))),
            ("Train physics", physics_sync_from_config(config, "train")),
            ("Play physics", physics_sync_from_config(config, "play")),
            ("Demo physics", physics_sync_from_config(config, "demo")),
            ("Max iterations", str(config.get("max_iterations", 500))),
            ("Seed", str(config.get("seed", 42))),
            ("Real-time play", str(config.get("real_time_play", False))),
            ("Real-time demo", str(config.get("real_time_demo", False))),
            ("Record video (train)", str(config.get("record_video", False))),
        ]
        for label, value in rows:
            print(f"  {label:<26} {Style.paint(value, Style.CYAN)}")

        print("\n  [1] Train settings")
        print("  [2] Play settings")
        print("  [3] Demo settings")
        print("  [4] Global seed & max iterations")
        print("  [0] Back")
        choice = input("\n  Choice: ").strip().lower()

        if choice == "0":
            return
        if choice == "1":
            _edit_action_block(config, "train", prompts)
        elif choice == "2":
            _edit_action_block(config, "play", prompts)
        elif choice == "3":
            _edit_action_block(config, "demo", prompts)
        elif choice == "4":
            config["seed"] = prompts.prompt_int("  seed", int(config.get("seed", 42)), minimum=0)
            config["max_iterations"] = prompts.prompt_int(
                "  max_iterations (train)", int(config.get("max_iterations", 500)), minimum=1
            )
            config["profile"] = "custom"
        else:
            print("  Invalid choice.")
            pause()


def _edit_action_block(config: dict[str, Any], action: ActionKind, prompts: RunConfigPrompts) -> None:
    from .terminal_ui import pause

    Style = _style()
    label = action.capitalize()
    if action == "train":
        config["num_envs_train"] = prompts.prompt_int(
            f"  {label} num_envs (headless)", int(config.get("num_envs_train", 512)), minimum=1
        )
        config["num_envs_train_visual"] = prompts.prompt_int(
            f"  {label} num_envs (visual)", int(config.get("num_envs_train_visual", 32)), minimum=1
        )
        config["headless_train"] = prompts.prompt_yes_no("  Default to headless training?", config["headless_train"])
        config["record_video"] = prompts.prompt_yes_no("  Record video while training?", config.get("record_video", False))
    elif action == "play":
        config["num_envs_play"] = prompts.prompt_int(
            f"  {label} num_envs", int(config.get("num_envs_play", 16)), minimum=1
        )
        config["real_time_play"] = prompts.prompt_yes_no("  Real-time playback?", config.get("real_time_play", False))
    else:
        config["num_envs_demo"] = prompts.prompt_int(
            f"  {label} num_envs", int(config.get("num_envs_demo", 16)), minimum=1
        )
        config["real_time_demo"] = prompts.prompt_yes_no("  Real-time demo?", config.get("real_time_demo", False))

    physics = prompts.prompt_choice(
        f"  {label} physics sync:",
        [("fabric", "Fabric (GPU)"), ("usd", "USD (CPU, stage updates)")],
        physics_sync_from_config(config, action),
    )
    config[f"physics_sync_{action}"] = physics

    if physics == "usd":
        config[f"device_{action}"] = "cpu"
        print(Style.paint("  device set to cpu for USD mode.", Style.DIM))
    else:
        config[f"device_{action}"] = prompts.prompt_choice(
            f"  {label} device:",
            [("cuda:0", "cuda:0"), ("cpu", "cpu")],
            str(config.get(f"device_{action}", config.get("device", "cuda:0"))),
        )

    config["profile"] = "custom"
    pause()
