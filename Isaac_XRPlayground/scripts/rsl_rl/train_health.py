#!/usr/bin/env python3
"""Summarize latest RSL-RL ball-catch (or any) run for agent health checks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    print(json.dumps({"error": f"tensorboard not installed: {exc}"}))
    sys.exit(1)


def _latest_run(log_root: Path) -> Path | None:
    if not log_root.is_dir():
        return None
    runs = [p for p in log_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _scalar_at(ea: EventAccumulator, tag: str, step: int | None = None) -> dict | None:
    if tag not in ea.Tags().get("scalars", []):
        return None
    series = ea.Scalars(tag)
    if not series:
        return None
    if step is None:
        pt = series[-1]
    else:
        pt = min(series, key=lambda x: abs(x.step - step))
    return {"step": int(pt.step), "value": float(pt.value)}


def _max_scalar(ea: EventAccumulator, tag: str) -> float | None:
    if tag not in ea.Tags().get("scalars", []):
        return None
    series = ea.Scalars(tag)
    if not series:
        return None
    return max(float(x.value) for x in series)


def summarize_run(run_dir: Path) -> dict:
    events = list(run_dir.glob("events.out.tfevents*"))
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    out: dict = {
        "run": run_dir.name,
        "run_path": str(run_dir),
        "has_events": bool(events),
        "event_mtime": max((e.stat().st_mtime for e in events), default=0.0),
        "checkpoint_count": len(ckpts),
        "latest_checkpoint": ckpts[-1].name if ckpts else None,
        "alive_seconds_ago": time.time() - max((e.stat().st_mtime for e in events), default=0.0) if events else None,
    }
    if not events:
        return out

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()

    tags = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/catch_rate",
        "Metrics/drop_rate",
        "Metrics/cup_hold",
        "Metrics/tip_platform",
        "Metrics/curriculum",
        "Policy/mean_std",
        "Loss/learning_rate",
    ]
    for tag in tags:
        key = tag.replace("/", "_").replace("Metrics_", "metric_").replace("Train_", "train_").replace("Policy_", "policy_").replace("Loss_", "loss_")
        val = _scalar_at(ea, tag)
        if val is not None:
            out[key] = val

    catch_max = _max_scalar(ea, "Metrics/catch_rate")
    if catch_max is not None:
        out["metric_catch_rate_max"] = catch_max

    iteration = out.get("train_mean_reward", {}).get("step", 0)
    out["iteration"] = iteration

    # Health gates (ball catch)
    catch_last = out.get("metric_catch_rate", {}).get("value", 0.0)
    catch_max = out.get("metric_catch_rate_max", 0.0)
    reward_last = out.get("train_mean_reward", {}).get("value", 0.0)
    std_last = out.get("policy_mean_std", {}).get("value", 99.0)
    lr_last = out.get("loss_learning_rate", {}).get("value", 0.0)

    status = "ok"
    reasons: list[str] = []
    if not events:
        status = "no_events"
        reasons.append("no tensorboard events")
    elif out.get("alive_seconds_ago", 9999) > 600 and iteration < 50:
        status = "stalled"
        reasons.append("events stale >10min early in run")
    if iteration >= 400 and catch_max < 0.01:
        status = "unhealthy"
        reasons.append("catch_rate still 0 after 400 iters")
    if std_last > 3.5:
        status = "unhealthy"
        reasons.append(f"policy std exploded ({std_last:.2f})")
    if lr_last < 1e-6 and iteration > 100:
        status = "unhealthy"
        reasons.append("learning rate collapsed")
    if catch_max >= 0.05:
        status = "learning_catch"
        reasons.append(f"catch_rate peaked at {catch_max:.3f}")
    if catch_last >= 0.08 and iteration >= 800:
        status = "ready"
        reasons.append(f"catch_rate {catch_last:.3f} at iter {iteration}")

    out["status"] = status
    out["reasons"] = reasons
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RSL-RL training health JSON")
    parser.add_argument(
        "--experiment",
        default="xrplayground_ball_catch_direct",
        help="Experiment folder under logs/rsl_rl/",
    )
    parser.add_argument("--run", default=None, help="Specific run folder name (default: latest)")
    parser.add_argument("--logs-root", default=None, help="Override logs/rsl_rl root")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    log_root = Path(args.logs_root) if args.logs_root else repo / "logs" / "rsl_rl" / args.experiment

    if args.run:
        run_dir = log_root / args.run
    else:
        run_dir = _latest_run(log_root)

    if run_dir is None or not run_dir.is_dir():
        print(json.dumps({"status": "no_runs", "log_root": str(log_root)}))
        sys.exit(0)

    print(json.dumps(summarize_run(run_dir), indent=2))


if __name__ == "__main__":
    main()
