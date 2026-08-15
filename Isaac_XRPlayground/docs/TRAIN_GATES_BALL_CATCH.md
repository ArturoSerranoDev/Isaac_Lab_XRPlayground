# Ball catch training health gates

Use with:

```bat
python scripts\rsl_rl\train_health.py
```

## Launch (headless)

```bat
cd Isaac_XRPlayground
python scripts\rsl_rl\train.py --task=Template-Xrplayground-Ball-Catch-Direct-v0 --num_envs=128 --viz none --max_iterations=2000 --export_onnx --seed=44
```

## Gates

| Iter | Healthy | Unhealthy |
|------|---------|-----------|
| 0–100 | reward rising, `policy_mean_std` ~0.8–1.2, LR ≈ 3e-4 | no events / crash |
| 400+ | `metric_catch_rate_max` > 0 or `Metrics/soft_grasp` > 0 | catch still 0 and only proximity reward |
| 800+ | `metric_catch_rate` trending up | std > 3.5 with catch back at 0 |
| Done | export checkpoint with highest catch batch (not always final iter) | — |

## Latest good run

- **Run:** `2026-08-14_14-53-03`
- **Best checkpoint for Unity:** `model_1900.pt` (peak catch batches ~iter 1900)
- **Unity ONNX:** `Unity_XRPlayground/Assets/_Project/Features/Policies/BallCatch/policy.onnx`

Train from scratch after reward/throw changes; do not resume older runs.
