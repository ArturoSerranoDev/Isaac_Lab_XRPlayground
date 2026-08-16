# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Train BallCatch Throw-A then Throw-B until throw_rolling gates pass; copy Unity on B gate.

Usage (from Isaac_XRPlayground):
  python scripts/rsl_rl/train_ball_catch_throw_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs" / "rsl_rl" / "xrplayground_ball_catch_2phase"
UNITY_DIR = (
    ROOT.parent
    / "Unity_XRPlayground"
    / "Assets"
    / "_Project"
    / "Features"
    / "Policies"
    / "BallCatch"
)
DEFAULT_PY = Path(r"C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe")
WRAP_RUN = "2026-08-16_06-05-50"
WRAP_CKPT = "model_5449.pt"

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _parse_metrics(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    text = ANSI.sub("", log_path.read_text(encoding="utf-8", errors="ignore"))
    pat = re.compile(
        r"Learning iteration\s+(\d+)/(\d+)"
        r"[\s\S]*?Metrics/throw_soft_grasp:\s+([0-9.nanNAN]+)"
        r"[\s\S]*?Metrics/throw_rolling:\s+([0-9.]+)"
        r"[\s\S]*?Metrics/throw_catch_ema:\s+([0-9.]+)"
        r"[\s\S]*?Metrics/curriculum_cap:\s+([0-9.]+)",
        re.MULTILINE,
    )
    rows = []
    for m in pat.finditer(text):
        rows.append(
            {
                "iter": int(m.group(1)),
                "max_iter": int(m.group(2)),
                "tsg": m.group(3),
                "rolling": float(m.group(4)),
                "tema": float(m.group(5)),
                "cap": float(m.group(6)),
            }
        )
    return rows


def _latest_run() -> Path | None:
    if not LOG_ROOT.exists():
        return None
    runs = sorted(LOG_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _latest_ckpt(run: Path) -> str | None:
    ckpts = sorted(run.glob("model_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0].name if ckpts else None


def _gate_ok(rows: list[dict], threshold: float, window: int = 40) -> bool:
    if len(rows) < window:
        return False
    last = rows[-window:]
    avg_roll = sum(r["rolling"] for r in last) / len(last)
    return avg_roll >= threshold and last[-1]["rolling"] >= threshold * 0.9


def _run_train(
    py: Path,
    task: str,
    log_file: Path,
    max_iters: int,
    load_run: str,
    checkpoint: str,
    copy_unity: bool,
):
    cmd = [
        str(py),
        "scripts/rsl_rl/train.py",
        f"--task={task}",
        "--num_envs=64",
        "--viz=none",
        f"--max_iterations={max_iters}",
        "--resume",
        f"--load_run={load_run}",
        f"--checkpoint={checkpoint}",
        "--export_onnx",
    ]
    if not copy_unity:
        cmd.append("--no_copy_to_unity")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fout = open(log_file, "w", encoding="utf-8", errors="replace")
    print(f"[pipeline] starting: {' '.join(cmd)}", flush=True)
    print(f"[pipeline] log: {log_file}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=fout,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc, fout


def _wait_gate(
    proc: subprocess.Popen,
    fout,
    log_file: Path,
    threshold: float,
    label: str,
    poll_s: float = 45.0,
    max_hours: float = 6.0,
) -> tuple[bool, Path | None, str | None]:
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            fout.close()
            rows = _parse_metrics(log_file)
            run = _latest_run()
            ckpt = _latest_ckpt(run) if run else None
            ok = _gate_ok(rows, threshold) if rows else False
            print(f"[pipeline] {label} process exited code={proc.returncode} gate_ok={ok}")
            return ok, run, ckpt
        rows = _parse_metrics(log_file)
        if rows:
            r = rows[-1]
            print(
                f"[pipeline] {label} iter={r['iter']}/{r['max_iter']} "
                f"rolling={r['rolling']:.3f} tema={r['tema']:.3f} cap={r['cap']:.3f}",
                flush=True,
            )
            if _gate_ok(rows, threshold):
                print(f"[pipeline] {label} GATE PASS (rolling≥{threshold})")
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                fout.close()
                run = _latest_run()
                return True, run, _latest_ckpt(run) if run else None
        if (time.time() - t0) > max_hours * 3600:
            print(f"[pipeline] {label} timeout after {max_hours}h — stopping")
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            fout.close()
            run = _latest_run()
            return False, run, _latest_ckpt(run) if run else None
        time.sleep(poll_s)


def _copy_unity(run: Path) -> None:
    src = run / "exported"
    if not src.exists():
        print(f"[pipeline] no exported/ in {run}")
        return
    UNITY_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("policy.onnx", "policy.json", "policy.pt"):
        f = src / name
        if f.exists():
            shutil.copy2(f, UNITY_DIR / name)
            print(f"[pipeline] copied {name} → Unity BallCatch/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=Path, default=DEFAULT_PY)
    ap.add_argument("--wrap-run", default=WRAP_RUN)
    ap.add_argument("--wrap-ckpt", default=WRAP_CKPT)
    ap.add_argument("--a-iters", type=int, default=2500)
    ap.add_argument("--b-iters", type=int, default=3500)
    ap.add_argument("--a-gate", type=float, default=0.75)
    ap.add_argument("--b-gate", type=float, default=0.80)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--a-run", default="", help="Existing Throw-A run to resume into B")
    ap.add_argument("--a-ckpt", default="")
    args = ap.parse_args()

    py = args.python
    if not py.exists():
        print(f"python not found: {py}", file=sys.stderr)
        return 1

    a_run_name = args.a_run
    a_ckpt = args.a_ckpt

    if not args.skip_a:
        log_a = ROOT / "logs" / "ball_catch_throw_a_pipeline.txt"
        proc, fout = _run_train(
            py,
            "Template-Xrplayground-Ball-Catch-Throw-A-v0",
            log_a,
            args.a_iters,
            args.wrap_run,
            args.wrap_ckpt,
            copy_unity=False,
        )
        ok, run, ckpt = _wait_gate(proc, fout, log_a, args.a_gate, "Throw-A")
        if run is None or ckpt is None:
            print("[pipeline] Throw-A produced no checkpoint", file=sys.stderr)
            return 2
        a_run_name, a_ckpt = run.name, ckpt
        print(f"[pipeline] Throw-A ckpt: {a_run_name}/{a_ckpt} gate={ok}")
        if not ok:
            print("[pipeline] Throw-A gate not met — continuing to B from best A anyway")
    else:
        if not a_run_name or not a_ckpt:
            print("--skip-a requires --a-run and --a-ckpt", file=sys.stderr)
            return 2

    log_b = ROOT / "logs" / "ball_catch_throw_b_pipeline.txt"
    proc, fout = _run_train(
        py,
        "Template-Xrplayground-Ball-Catch-Throw-B-v0",
        log_b,
        args.b_iters,
        a_run_name,
        a_ckpt,
        copy_unity=False,
    )
    ok, run, ckpt = _wait_gate(proc, fout, log_b, args.b_gate, "Throw-B")
    if run is None:
        print("[pipeline] Throw-B produced no run", file=sys.stderr)
        return 3
    print(f"[pipeline] Throw-B done gate={ok} run={run.name} ckpt={ckpt}")
    if ok:
        # Ensure export exists (train already exports at end; early stop may need re-export).
        exported = run / "exported" / "policy.onnx"
        if not exported.exists() and ckpt:
            print("[pipeline] re-exporting ONNX for early-stop gate…")
            ckpt_path = run / ckpt
            out_dir = run / "exported"
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                [
                    str(py),
                    "scripts/rsl_rl/export_onnx.py",
                    "--task=Template-Xrplayground-Ball-Catch-Throw-B-v0",
                    f"--ckpt={ckpt_path}",
                    f"--output_dir={out_dir}",
                    "--no_copy_to_unity",
                    "--headless",
                ],
                cwd=str(ROOT),
            )
        _copy_unity(run)
        print("[pipeline] SUCCESS — Unity BallCatch updated")
        return 0
    print("[pipeline] Throw-B gate not met — Unity NOT updated")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
