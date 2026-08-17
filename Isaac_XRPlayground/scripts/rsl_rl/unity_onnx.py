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
from pathlib import Path

import torch


UNITY_ONNX_OPSET = 15


def build_policy_onnx_module(runner):
    """Build the exact deterministic module passed to the classic ONNX exporter."""
    policy = runner.alg.get_policy()
    onnx_model = policy.as_onnx(verbose=False)
    onnx_model.to("cpu")
    onnx_model.eval()
    return onnx_model


def export_policy_onnx_for_unity(
    runner, export_dir: str, filename: str = "policy.onnx", *, onnx_model=None
) -> str:
    """Write a single-file ONNX that Unity Sentis/Inference Engine can import."""
    os.makedirs(export_dir, exist_ok=True)
    save_path = os.path.join(export_dir, filename)

    onnx_model = onnx_model or build_policy_onnx_module(runner)

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
        f"[INFO] Unity ONNX (TorchScript, opset {UNITY_ONNX_OPSET}) -> {save_path} ({size} bytes)"
    )
    if size < 50_000:
        print(
            "[WARN] ONNX is unusually small for an MLP policy. If Unity import fails, "
            "confirm weights are embedded (no companion .onnx.data file)."
        )
    return save_path


def export_policy_torchscript_for_isaac_dependency(
    onnx_model, export_dir: str, filename: str = "policy.pt"
) -> str:
    """Export the same deterministic module for hierarchical Isaac policies.

    Do not freeze this graph. ``torch.jit.freeze`` folds module parameters into
    CPU constants, and Isaac's ``PreTrainedPolicyAction`` must be able to move
    the loaded dependency to the environment device with ``module.to(device)``.
    """
    output = Path(export_dir) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = onnx_model.get_dummy_inputs()
    inputs = tuple(dummy) if isinstance(dummy, (tuple, list)) else (dummy,)
    traced = torch.jit.trace(onnx_model, inputs, strict=True).eval()
    if not tuple(traced.named_parameters()):
        raise RuntimeError(
            "Isaac dependency TorchScript has no movable parameters; "
            "do not freeze or constant-fold this graph"
        )
    torch.jit.save(traced, str(output))
    print(f"[INFO] Isaac dependency TorchScript -> {output} ({output.stat().st_size} bytes)")
    return str(output)
