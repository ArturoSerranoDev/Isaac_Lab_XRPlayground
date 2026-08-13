# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRPlayground interactive launcher."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .project_config import (
    CONFIG_PATH,
    DEFAULT_CONFIG,
    ISAACLAB_ROOT,
    ISAAC_PROJECT_ROOT,
    MONOREPO_ROOT,
    PROFILES,
    SHARED_ASSETS_ROOT,
    TASKS_SOURCE,
)
from .scaffold_task import scaffold_training_task
from .run_config import (
    RunConfig,
    RunConfigPrompts,
    append_sim_flags,
    edit_full_config_menu,
    select_run_config_interactive,
)
from .tasks_registry import discover_tasks, find_latest_checkpoint, list_training_runs
from .terminal_ui import (
    Style,
    clear,
    copy_to_clipboard,
    format_command,
    format_timestamp,
    open_path,
    pause,
    print_banner,
    print_config_summary,
    print_main_menu,
    print_paths_block,
    print_recent_runs,
    print_status_block,
    print_task_card,
    run_command,
    shorten_path,
)


class XRLauncher:
    def __init__(self) -> None:
        Style.enable()
        self.config = self.load_config()
        self.python = self.resolve_python()
        self.last_command: list[str] = []
        self.extension_ok = self.check_extension_installed()

    # ------------------------------------------------------------------ config
    def load_config(self) -> dict[str, Any]:
        if CONFIG_PATH.is_file():
            try:
                with CONFIG_PATH.open(encoding="utf-8") as handle:
                    stored = json.load(handle)
                merged = DEFAULT_CONFIG.copy()
                merged.update(stored)
                if "recent_runs" not in merged or not isinstance(merged["recent_runs"], list):
                    merged["recent_runs"] = []
                return merged
            except (json.JSONDecodeError, OSError):
                pass
        return DEFAULT_CONFIG.copy()

    def save_config(self) -> None:
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2)

    def apply_profile(self, profile_key: str) -> None:
        profile = PROFILES[profile_key]
        for key in (
            "num_envs_train",
            "num_envs_train_visual",
            "num_envs_play",
            "num_envs_demo",
            "max_iterations",
            "headless_train",
        ):
            if key in profile:
                self.config[key] = profile[key]
        self.config["profile"] = profile_key
        self.save_config()

    def profile_label(self) -> str:
        key = self.config.get("profile", "learn")
        if key not in PROFILES:
            return "Custom"
        profile = PROFILES[key]
        return f"{profile['label']} — {profile['hint']}"

    def run_prompts(self) -> RunConfigPrompts:
        return RunConfigPrompts(
            prompt_int=prompt_int,
            prompt_yes_no=prompt_yes_no,
            prompt_choice=prompt_choice,
        )

    # ------------------------------------------------------------------ helpers
    def resolve_python(self) -> Path:
        candidates = [
            ISAACLAB_ROOT / "env_isaaclab" / "Scripts" / "python.exe",
            Path(sys.executable),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Could not find Isaac Lab Python.\n"
            f"Expected: {ISAACLAB_ROOT / 'env_isaaclab' / 'Scripts' / 'python.exe'}"
        )

    def check_extension_installed(self) -> bool:
        try:
            result = subprocess.run(
                [str(self.python), "-c", "import XRPlayground; print(XRPlayground.__file__)"],
                cwd=ISAAC_PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def all_tasks(self) -> dict[str, dict[str, Any]]:
        return discover_tasks()

    def get_task(self, task_key: str | None = None) -> dict[str, Any]:
        tasks = self.all_tasks()
        key = task_key or self.config.get("task_key", "cartpole")
        if key not in tasks:
            key = next(iter(tasks), "cartpole")
        self.config["task_key"] = key
        return tasks[key]

    def get_algorithm(self, task: dict[str, Any]) -> str:
        algo = str(self.config.get("algorithm", "")).strip().upper()
        if algo in task["algorithms"]:
            return algo
        return task["default_algorithm"]

    def find_latest_checkpoint(self, log_dir_name: str) -> str | None:
        return find_latest_checkpoint(log_dir_name)

    def record_run(self, action: str, exit_code: int, task_key: str | None = None) -> None:
        task = self.get_task(task_key)
        entry = {
            "timestamp": format_timestamp(),
            "action": action,
            "task_key": task_key or self.config["task_key"],
            "task_label": task["label"],
            "exit_code": exit_code,
        }
        recent = self.config.setdefault("recent_runs", [])
        recent.insert(0, entry)
        self.config["recent_runs"] = recent[:8]
        self.save_config()

    def base_cmd(self) -> list[str]:
        return [str(self.python)]

    def maybe_confirm(self, command: list[str]) -> bool:
        if self.config.get("show_command_preview", True):
            print(Style.paint("\n  Command preview:", Style.YELLOW))
            print(Style.paint("  " + format_command(command), Style.DIM))
        if self.config.get("confirm_before_run", True):
            return prompt_yes_no("Run this command?", default=True)
        return True

    def execute(self, command: list[str], action: str, task_key: str | None = None) -> None:
        self.last_command = command
        if not self.maybe_confirm(command):
            print("Cancelled.")
            pause()
            return
        exit_code = run_command(command, ISAAC_PROJECT_ROOT)
        self.record_run(action, exit_code, task_key=task_key)
        pause()

    def select_task(self, title: str, default_key: str | None = None) -> str | None:
        tasks = self.all_tasks()
        if not tasks:
            print(Style.paint("\n  No trainable tasks found in tasks/direct/.", Style.RED))
            pause()
            return None

        clear()
        print_banner()
        print(Style.paint(f"\n  {title}", Style.YELLOW, Style.BOLD))
        print(Style.paint("  Pick which training environment to use:\n", Style.DIM))

        ordered = list(tasks.items())
        default_key = default_key or self.config.get("task_key", ordered[0][0])

        for index, (key, task) in enumerate(ordered, start=1):
            runs = list_training_runs(task["log_dir"])
            run_info = f"{len(runs)} run(s)" if runs else "no runs yet"
            default_mark = "  <<" if key == default_key else ""
            print(
                f"  [{index}] {Style.paint(task['label'], Style.CYAN)}"
                f"  ·  {task['default_algorithm']}"
                f"  ·  {run_info}{default_mark}"
            )
            print(f"       {task['subtitle']}")
            print(f"       {Style.paint(task['task_id'], Style.DIM)}")
            if runs and runs[0].get("name"):
                print(f"       latest: {Style.paint(runs[0]['name'], Style.DIM)}")

        print("  [0] Cancel")
        while True:
            raw = input(f"\n  Training [{default_key}]: ").strip()
            if raw == "0":
                return None
            if not raw:
                return default_key if default_key in tasks else ordered[0][0]
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(ordered):
                    return ordered[idx - 1][0]
            if raw in tasks:
                return raw
            print("  Invalid choice.")

    def select_checkpoint(self, task: dict[str, Any]) -> str | None:
        runs = list_training_runs(task["log_dir"])
        clear()
        print_banner()
        print(Style.paint(f"\n  Checkpoints — {task['label']}", Style.YELLOW, Style.BOLD))

        if not runs:
            print("  No training runs found for this task yet.")
            pause()
            return None

        for index, run in enumerate(runs[:8], start=1):
            ckpt = run.get("checkpoint") or "(no checkpoint file)"
            print(f"  [{index}] {run['name']}")
            print(f"       {Style.paint(shorten_path(str(ckpt)), Style.DIM)}")

        print("  [C] Custom checkpoint path")
        print("  [0] Cancel")

        while True:
            raw = input("\n  Checkpoint [1 = latest]: ").strip().lower()
            if raw in {"", "1"}:
                return runs[0].get("checkpoint")
            if raw == "0":
                return None
            if raw == "c":
                return prompt_text("  Checkpoint path", runs[0].get("checkpoint") or "")
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= min(len(runs), 8):
                    return runs[idx - 1].get("checkpoint")
            print("  Invalid choice.")

    def build_train_command(
        self,
        task: dict[str, Any],
        run: RunConfig,
        checkpoint: str = "",
    ) -> list[str]:
        algo = self.get_algorithm(task)
        command = self.base_cmd() + [
            "scripts/skrl/train.py",
            f"--task={task['task_id']}",
            f"--num_envs={run.num_envs}",
            f"--algorithm={algo}",
            f"--seed={run.seed}",
            f"--max_iterations={run.max_iterations}",
        ]
        if run.headless:
            command.append("--headless")
        else:
            command.extend(["--viz", "kit"])
        if checkpoint:
            command.append(f"--checkpoint={checkpoint}")
        if run.record_video:
            command.append("--video")
        return append_sim_flags(command, run)

    # ------------------------------------------------------------------ screens
    def render(self) -> None:
        clear()
        print_banner()
        print_paths_block()
        print_status_block(self.python, self.extension_ok)
        task = self.get_task()
        print_task_card(task, self.get_algorithm(task))
        tasks = self.all_tasks()
        print(Style.paint(f"\n  Available trainings: {len(tasks)}", Style.DIM))
        for key, meta in tasks.items():
            runs = list_training_runs(meta["log_dir"])
            marker = " <<" if key == self.config.get("task_key") else ""
            print(f"    · {meta['label']} ({len(runs)} runs){marker}")
        print_config_summary(self.config, self.profile_label())
        latest = self.find_latest_checkpoint(task["log_dir"])
        if latest and not self.config.get("checkpoint"):
            print(f"  Latest ckpt: {Style.paint(shorten_path(latest), Style.DIM)}")
        print_recent_runs(self.config.get("recent_runs", []))
        print_main_menu()

    def profiles_menu(self) -> None:
        while True:
            self.render()
            print(Style.paint("\n  Hardware profiles (RTX 3060 Ti)", Style.YELLOW, Style.BOLD))
            options = [(key, f"{meta['label']} — {meta['hint']}") for key, meta in PROFILES.items()]
            for key, label in options:
                active = " <<" if self.config.get("profile") == key else ""
                print(f"  [{key}] {label}{active}")
            print("  [0] Back")
            choice = input("\nProfile: ").strip().lower()
            if choice == "0":
                return
            if choice in PROFILES:
                self.apply_profile(choice)
                print(Style.paint(f"\n  Applied profile: {PROFILES[choice]['label']}", Style.GREEN))
                pause()
            else:
                print("Invalid profile.")
                pause()

    def settings_menu(self) -> None:
        while True:
            self.render()
            print(Style.paint("\n  Settings", Style.YELLOW, Style.BOLD))
            print("  [1] Change task")
            print("  [2] Full run configuration (envs, device, Fabric/USD)")
            print("  [3] Algorithm")
            print("  [4] Checkpoint (play / resume)")
            print("  [5] Confirm before run")
            print("  [6] Show command preview")
            print("  [R] Reset to 'Learning session' profile")
            print("  [0] Back")
            choice = input("\nChoice: ").strip().lower()

            task = self.get_task()
            if choice == "1":
                picked = self.select_task("Settings — default task", self.config["task_key"])
                if picked:
                    self.config["task_key"] = picked
                    self.config["algorithm"] = ""
                    self.config["profile"] = "custom"
            elif choice == "2":
                edit_full_config_menu(self.config, self.run_prompts())
            elif choice == "3":
                self.config["algorithm"] = prompt_choice(
                    "Algorithm:",
                    [(algo, algo) for algo in task["algorithms"]],
                    self.get_algorithm(task),
                )
                self.config["profile"] = "custom"
            elif choice == "4":
                suggested = self.config.get("checkpoint") or self.find_latest_checkpoint(task["log_dir"]) or ""
                self.config["checkpoint"] = prompt_text("Checkpoint path", suggested)
            elif choice == "5":
                self.config["confirm_before_run"] = prompt_yes_no(
                    "Ask before running commands?", self.config["confirm_before_run"]
                )
            elif choice == "6":
                self.config["show_command_preview"] = prompt_yes_no(
                    "Show command preview?", self.config["show_command_preview"]
                )
            elif choice == "r":
                self.apply_profile("learn")
                print(Style.paint("\n  Reset to Learning session profile.", Style.GREEN))
                pause()
                continue
            elif choice == "0":
                self.save_config()
                return
            else:
                print("Invalid choice.")
                pause()
                continue
            self.save_config()

    # ------------------------------------------------------------------ actions
    def action_train(self) -> None:
        task_key = self.select_task("Train policy", self.config.get("task_key"))
        if not task_key:
            return
        self.config["task_key"] = task_key
        task = self.get_task(task_key)

        run = select_run_config_interactive(
            self.config,
            "train",
            self.run_prompts(),
            title="Run configuration — Train",
        )
        if run is None:
            return
        self.save_config()

        resume = prompt_yes_no("  Resume from a checkpoint?", default=False)
        checkpoint = ""
        if resume:
            picked = self.select_checkpoint(task)
            if picked:
                checkpoint = picked

        command = self.build_train_command(task, run, checkpoint=checkpoint)
        self.execute(command, "train", task_key=task_key)

    def action_play(self) -> None:
        task_key = self.select_task("Play trained policy", self.config.get("task_key"))
        if not task_key:
            return
        self.config["task_key"] = task_key
        task = self.get_task(task_key)

        checkpoint = self.select_checkpoint(task)
        if not checkpoint:
            print(Style.paint("\n  No checkpoint selected.", Style.RED))
            pause()
            return

        run = select_run_config_interactive(
            self.config,
            "play",
            self.run_prompts(),
            title="Run configuration — Play",
        )
        if run is None:
            return
        self.save_config()

        algo = self.get_algorithm(task)
        command = self.base_cmd() + [
            "scripts/skrl/play.py",
            f"--task={task['task_id']}",
            f"--num_envs={run.num_envs}",
            f"--algorithm={algo}",
            f"--seed={run.seed}",
            f"--checkpoint={checkpoint}",
            "--viz",
            "kit",
        ]
        if run.real_time:
            command.append("--real-time")
        command = append_sim_flags(command, run)
        self.execute(command, "play", task_key=task_key)

    def action_demo(self, mode: str) -> None:
        task_key = self.select_task("Environment demo", self.config.get("task_key"))
        if not task_key:
            return
        self.config["task_key"] = task_key
        task = self.get_task(task_key)

        run = select_run_config_interactive(
            self.config,
            "demo",
            self.run_prompts(),
            title=f"Run configuration — Demo ({mode})",
        )
        if run is None:
            return
        self.save_config()

        script = "scripts/random_agent.py" if mode == "random" else "scripts/zero_agent.py"
        command = self.base_cmd() + [
            script,
            f"--task={task['task_id']}",
            f"--num_envs={run.num_envs}",
            "--viz",
            "kit",
        ]
        if run.real_time:
            command.append("--real-time")
        command = append_sim_flags(command, run)
        self.execute(command, f"demo-{mode}", task_key=task_key)

    def action_list_envs(self) -> None:
        command = self.base_cmd() + ["scripts/list_envs.py"]
        self.execute(command, "list-envs")

    def action_open_assets(self) -> None:
        open_path(SHARED_ASSETS_ROOT)
        pause("\nOpened shared assets folder.")

    def action_open_task_source(self) -> None:
        task = self.get_task()
        open_path(TASKS_SOURCE / task["source_dir"])
        pause("\nOpened task source folder.")

    def action_open_logs(self) -> None:
        task = self.get_task()
        open_path(ISAAC_PROJECT_ROOT / "logs" / "skrl" / task["log_dir"])
        pause("\nOpened training logs.")

    def action_open_monorepo(self) -> None:
        open_path(MONOREPO_ROOT)
        pause("\nOpened monorepo root.")

    def action_isaac_sim_empty(self) -> None:
        isaacsim = ISAACLAB_ROOT / "env_isaaclab" / "Scripts" / "isaacsim.exe"
        if not isaacsim.is_file():
            # fallback: python -m isaacsim via env
            command = self.base_cmd() + ["-m", "isaacsim"]
        else:
            command = [str(isaacsim)]
        print(Style.paint("\n  Launching Isaac Sim (empty GUI)...", Style.CYAN))
        subprocess.Popen(command, cwd=ISAAC_PROJECT_ROOT)
        pause("\nIsaac Sim started in a separate process.")

    def action_lab_empty_scene(self) -> None:
        command = self.base_cmd() + [
            str(ISAACLAB_ROOT / "scripts" / "tutorials" / "00_sim" / "create_empty.py"),
            "--viz",
            "kit",
        ]
        if not (ISAACLAB_ROOT / "scripts" / "tutorials" / "00_sim" / "create_empty.py").is_file():
            print(Style.paint("\n  create_empty.py not found in Isaac Lab install.", Style.RED))
            pause()
            return
        self.execute(command, "lab-empty")

    def action_create_training(self) -> None:
        clear()
        print_banner()
        print(Style.paint("\n  Create new training", Style.YELLOW, Style.BOLD))
        print("  Scaffolds a new single-agent PPO task from the cartpole template.")
        print(Style.paint("  You can edit rewards, robot, and scene in the generated folder.\n", Style.DIM))

        raw_name = input("  Folder name (snake_case, e.g. my_arm): ").strip()
        if not raw_name:
            print("  Cancelled.")
            pause()
            return

        label = prompt_text("  Display name", raw_name.replace("_", " ").title())
        description = prompt_text("  Short description (optional)", "")

        try:
            result = scaffold_training_task(raw_name, label=label, description=description)
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            print(Style.paint(f"\n  {exc}", Style.RED))
            pause()
            return

        self.config["task_key"] = result["task_key"]
        self.save_config()

        print(Style.paint("\n  Created successfully!", Style.GREEN))
        print(f"  Task ID : {Style.paint(result['task_id'], Style.MAGENTA)}")
        print(f"  Folder  : {result['folder']}")
        print("\n  Next steps:")
        print("    1. Edit *_env_cfg.py (robot, rewards, num_envs default)")
        print("    2. Edit *_env.py if you need custom scene logic")
        print("    3. Launcher → Train → pick your new task")

        if prompt_yes_no("\n  Open the new task folder now?", default=True):
            open_path(Path(result["folder"]))
        if prompt_yes_no("  Run a smoke train on it now?", default=False):
            self.action_train()
        else:
            pause()

    def action_sim_speed_help(self) -> None:
        clear()
        print_banner()
        print(Style.paint("\n  Simulation speed — quick guide", Style.YELLOW, Style.BOLD))
        print(
            """
  WATCHING (play / demo) — wall-clock speed
  ─────────────────────────────────────────
  • Settings → Real-time playback / demo → ON
    Uses --real-time so 1 sim second ≈ 1 real second.
  • OFF = runs as fast as your GPU allows (often very fast).

  TRAINING throughput — not the same as playback speed
  ────────────────────────────────────────────────────
  • Headless train = maximum learning speed (no window).
  • Visual train = slower, but you see the robots.
  • More num_envs (headless) = more steps/sec on GPU.

  PHYSICS timestep — edit your task *_env_cfg.py
  ───────────────────────────────────────────────
  • sim.dt           physics step (default 1/120 s)
  • decimation       physics steps per env step
  • env step time    = sim.dt × decimation

  Examples in cartpole_env_cfg.py:
    decimation = 2, sim.dt = 1/120  →  env step ≈ 16.7 ms
    decimation = 4                  →  slower env steps (easier to follow visually)
    decimation = 1                  →  faster env steps

  RENDERING cost
  ──────────────
  • render_interval in SimulationCfg — higher = fewer GPU renders
  • Fewer num_envs when using Visual mode

  FABRIC vs USD (Settings → Full run configuration)
  ─────────────────────────────────────────────────
  • Fabric + cuda:0 — default, fast training & viewport motion
  • USD + cpu — stage Transform panel updates (slow, num_envs ≤ 8)
  • USD + cuda on Windows often crashes — launcher forces CPU
"""
        )
        pause()

    def action_copy_command(self) -> None:
        if not self.last_command:
            print("\n  No command run yet in this session.")
        elif copy_to_clipboard(format_command(self.last_command)):
            print(Style.paint("\n  Copied last command to clipboard.", Style.GREEN))
        else:
            print("\n  Could not copy to clipboard.")
            print(f"  {format_command(self.last_command)}")
        pause()

    def run(self) -> int:
        try:
            _ = self.python
        except FileNotFoundError as exc:
            print(exc)
            pause()
            return 1

        while True:
            self.render()
            choice = input("\n  Choice: ").strip().lower()

            if choice == "1":
                self.action_train()
            elif choice == "2":
                self.action_play()
            elif choice == "3":
                self.action_demo("random")
            elif choice == "4":
                self.action_demo("zero")
            elif choice == "5":
                self.action_open_assets()
            elif choice == "6":
                self.action_open_task_source()
            elif choice == "7":
                self.action_open_logs()
            elif choice == "8":
                self.action_open_monorepo()
            elif choice == "9":
                self.action_isaac_sim_empty()
            elif choice == "0":
                self.action_lab_empty_scene()
            elif choice in {"l", "list"}:
                self.action_list_envs()
            elif choice in {"p", "profile", "profiles"}:
                self.profiles_menu()
            elif choice in {"s", "settings"}:
                self.settings_menu()
            elif choice in {"n", "new", "create"}:
                self.action_create_training()
            elif choice in {"h", "help", "?"}:
                self.action_sim_speed_help()
            elif choice in {"c", "copy"}:
                self.action_copy_command()
            elif choice in {"q", "quit", "exit"}:
                self.save_config()
                clear()
                print(Style.paint("\n  See you next session.\n", Style.CYAN))
                return 0
            else:
                print("Invalid choice.")
                time.sleep(0.6)


# ------------------------------------------------------------------ prompts (kept local to avoid circular imports)
def prompt_choice(title: str, options: list[tuple[str, str]], default: str) -> str:
    print(f"\n{title}")
    for key, label in options:
        marker = " (default)" if key == default else ""
        print(f"  [{key}] {label}{marker}")
    while True:
        choice = input(f"Choice [{default}]: ").strip() or default
        if any(key == choice for key, _ in options):
            return choice
        print("Invalid choice.")


def prompt_int(title: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{title} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if value < minimum:
            print(f"Must be at least {minimum}.")
            continue
        return value


def prompt_yes_no(title: str, default: bool) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{title} [{default_label}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "1", "true"}:
            return True
        if raw in {"n", "no", "0", "false"}:
            return False
        print("Answer y or n.")


def prompt_text(title: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else " (leave empty to skip)"
    raw = input(f"{title}{suffix}: ").strip()
    return raw or default


def main() -> int:
    return XRLauncher().run()
