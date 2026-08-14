# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export RSL-RL policies as Unity Inference Engine–compatible ONNX.

PyTorch 2.9+ ``torch.onnx.export`` defaults to the dynamo/FX exporter (opset 18,
onnxscript). Unity Inference Engine 2.3 only imports classic ONNX opset 7–15 and
crashes on dynamo graphs. Always use the TorchScript exporter + opset 15.
"""

from __future__ import annotations

import inspect
import os

import torch


UNITY_ONNX_OPSET = 15


def export_policy_onnx_for_unity(runner, export_dir: str, filename: str = "policy.onnx") -> str:
    """Write a single-file ONNX that Unity Sentis/Inference Engine can import."""
    os.makedirs(export_dir, exist_ok=True)
    save_path = os.path.join(export_dir, filename)

    policy = runner.alg.get_policy()
    onnx_model = policy.as_onnx(verbose=False)
    onnx_model.to("cpu")
    onnx_model.eval()

    dummy = onnx_model.get_dummy_inputs()
    kwargs = {
        "export_params": True,
        "opset_version": UNITY_ONNX_OPSET,
        "do_constant_folding": True,
        "input_names": list(onnx_model.input_names),
        "output_names": list(onnx_model.output_names),
        "dynamic_axes": {},
    }
    # PyTorch 2.9+: default dynamo=True produces graphs Unity cannot parse.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    torch.onnx.export(onnx_model, dummy, save_path, **kwargs)

    size = os.path.getsize(save_path)
    print(
        f"[INFO] Unity ONNX (TorchScript, opset {UNITY_ONNX_OPSET}) → {save_path} ({size} bytes)"
    )
    if size < 50_000:
        print(
            "[WARN] ONNX is unusually small for an MLP policy. If Unity import fails, "
            "confirm weights are embedded (no companion .onnx.data file)."
        )
    return save_path
