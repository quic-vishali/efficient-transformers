# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
TorchScript symbolic wrappers for the three FP8 DequantizeLinear granularities.

Used via ``select_interface`` in FP8DeQuantLinear and FP8BlockWiseDequantLinear
forward methods so that the legacy TorchScript export path emits standard ONNX
``DequantizeLinear`` nodes instead of ``Cast + Mul``.  Weights remain in FP8
(dtype=17) in the ONNX initializers.

Three granularities:

* ``FP8DequantizePerTensorFunc``  — scalar scale, DequantizeLinear (no axis).
* ``FP8DequantizePerAxisFunc``    — (out_features,) scale, DequantizeLinear(axis=0).
* ``FP8DequantizeBlockedFunc``    — compact (ceil(out/R), ceil(in/C)) scale.
                                    Aligned weights use blocked DequantizeLinear;
                                    partial edge blocks use a cropped expanded scale.
"""

import onnxscript
import torch

from QEfficient.utils import constants

# Dynamo ONNX translations use the standard-domain DequantizeLinear operator
# directly.  Keep these separate from the legacy symbolics below: the Dynamo
# exporter expects an ONNXScript function in its custom translation table.
_DYNAMO_OPS = getattr(onnxscript, f"opset{constants.ONNX_DYNAMO_EXPORT_OPSET}")


@onnxscript.script()
def FP8DequantizePerTensorDynamo(weight: onnxscript.FLOAT8E4M3FN, scale: onnxscript.FLOAT) -> onnxscript.FLOAT:
    return _DYNAMO_OPS.DequantizeLinear(weight, scale)


@onnxscript.script()
def FP8DequantizePerAxisDynamo(weight: onnxscript.FLOAT8E4M3FN, scale: onnxscript.FLOAT) -> onnxscript.FLOAT:
    return _DYNAMO_OPS.DequantizeLinear(weight, scale, axis=0)


@onnxscript.script()
def FP8DequantizeBlocked1x128Dynamo(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
    row_block_size: int,
    col_block_size: int,
) -> onnxscript.FLOAT:
    # ONNX blocked quantization expands only along `axis`; the row dimension
    # must therefore be expanded explicitly before DequantizeLinear.  The
    # compact scale remains reduced along the last dimension and is described
    # by block_size there.
    scale = _DYNAMO_OPS.Unsqueeze(scale, [1])
    scale = _DYNAMO_OPS.Tile(scale, [1, 1, 1])
    scale = _DYNAMO_OPS.Flatten(scale, axis=2)
    return _DYNAMO_OPS.DequantizeLinear(weight, scale, axis=-1, block_size=128)


@onnxscript.script()
def FP8DequantizeBlocked128x128Dynamo(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
    row_block_size: int,
    col_block_size: int,
) -> onnxscript.FLOAT:
    scale = _DYNAMO_OPS.Unsqueeze(scale, [1])
    scale = _DYNAMO_OPS.Tile(scale, [1, 128, 1])
    scale = _DYNAMO_OPS.Flatten(scale, axis=2)
    weight_rows = _DYNAMO_OPS.Slice(_DYNAMO_OPS.Shape(weight), starts=[0], ends=[1], axes=[0])
    scale = _DYNAMO_OPS.Slice(scale, starts=[0], ends=weight_rows, axes=[0])
    return _DYNAMO_OPS.DequantizeLinear(weight, scale, axis=-1, block_size=128)


def get_blocked_fn(row_block_size: int, col_block_size: int):
    """Return the Dynamo translator for a concrete blocked FP8 block size."""
    blocked_functions = {
        (1, 128): FP8DequantizeBlocked1x128Dynamo,
        (128, 128): FP8DequantizeBlocked128x128Dynamo,
    }
    if (row_block_size, col_block_size) in blocked_functions:
        return blocked_functions[(row_block_size, col_block_size)]
    raise ValueError(f"Unsupported blocked FP8 block size ({row_block_size}, {col_block_size})")


# ── TorchScript-path symbolic wrappers ───────────────────────────────────────


class FP8DequantizePerTensorFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_per_tensor."""

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return weight.to(scale.dtype) * scale

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight: torch.Value, scale: torch.Value) -> torch.Value:
        return g.op("DequantizeLinear", weight, scale)


class FP8DequantizePerAxisFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_per_axis."""

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1)
        return weight.to(scale.dtype) * scale

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight: torch.Value, scale: torch.Value) -> torch.Value:
        return g.op("DequantizeLinear", weight, scale, axis_i=0)


class FP8DequantizeBlockedFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_blocked.

    Emits Unsqueeze(scale) + Tile([1, row_bs, 1]) + Flatten(axis=2) +
    DequantizeLinear(axis=-1, block_size=col_bs).
    """

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor, row_block_size: int, col_block_size: int) -> torch.Tensor:
        scale_row = scale.repeat_interleave(row_block_size, dim=0)
        scale_full = scale_row.repeat_interleave(col_block_size, dim=-1)
        scale_full = scale_full[: weight.shape[-2], : weight.shape[-1]]
        return weight.to(scale_full.dtype) * scale_full

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        weight: torch.Value,
        scale: torch.Value,
        row_block_size,
        col_block_size,
    ) -> torch.Value:
        import torch.onnx.symbolic_helper as sym_help

        row_bs = sym_help._maybe_get_const(row_block_size, "i")
        col_bs = sym_help._maybe_get_const(col_block_size, "i")
        axes = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        scale_unsq = g.op("Unsqueeze", scale, axes)

        weight_rows = sym_help._get_tensor_dim_size(weight, 0)
        weight_cols = sym_help._get_tensor_dim_size(weight, 1)
        if (
            weight_rows is not None
            and weight_cols is not None
            and (weight_rows % row_bs != 0 or weight_cols % col_bs != 0)
        ):
            repeats = g.op("Constant", value_t=torch.tensor([1, row_bs, col_bs], dtype=torch.int64))
            tiled = g.op("Tile", scale_unsq, repeats)
            scale_full = g.op("Flatten", tiled, axis_i=2)
            starts = g.op("Constant", value_t=torch.tensor([0, 0], dtype=torch.int64))
            ends = g.op("Constant", value_t=torch.tensor([weight_rows, weight_cols], dtype=torch.int64))
            slice_axes = g.op("Constant", value_t=torch.tensor([0, 1], dtype=torch.int64))
            scale_full = g.op("Slice", scale_full, starts, ends, slice_axes)
            return g.op("Mul", g.op("CastLike", weight, scale), scale_full)

        repeats = g.op("Constant", value_t=torch.tensor([1, row_bs, 1], dtype=torch.int64))
        tiled = g.op("Tile", scale_unsq, repeats)
        scale_row = g.op("Flatten", tiled, axis_i=2)
        return g.op("DequantizeLinear", weight, scale_row, axis_i=-1, block_size_i=col_bs)


# ── Register symbolics for the legacy TorchScript export path ────────────────
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_per_tensor",
    FP8DequantizePerTensorFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_per_axis",
    FP8DequantizePerAxisFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_blocked",
    FP8DequantizeBlockedFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
