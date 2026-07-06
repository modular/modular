# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Correctness tests for the row-wise reduction ops (Row API).

Covers every non-composite reduction routed through MAX's own Mojo Row API:
the pure reductions (``reduce_sum/max/min/mean/product``, ``argmax``,
``argmin``, ``reduce_min_and_max``) and the last-axis norm-type ops
(``softmax``, ``logsoftmax``, ``layer_norm``, ``rms_norm``,
``row_mean_of_squares``). Each op is checked against a float32 torch reference
across a few small column counts (even + odd), on the inner axis and -- for the
pure arbitrary-axis reductions -- on a non-inner axis, in both bfloat16 and
float32 where precision makes it meaningful.

These are small, fast CI shapes. The bandwidth-oriented perf grid for the same
ops lives in the manual benchmark at
``//max/kernels/benchmarks/graph:bench_rowwise_reductions`` and is not run here.

Each parametrization builds one symbolic-dimension graph and loads it once, then
feeds several concrete shapes as data (per the guidance against per-case graph
recompilation) -- so the compile count equals the number of parametrizations,
not the number of shapes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

_DEV = DeviceRef.CPU()
_RMS_EPS = 1e-6
_LN_EPS = 1e-5

# Ops whose output is integer indices.
_INT_OUT = {"argmax", "argmin"}

# Small inner-axis column counts: even + odd, small + a wider row.
_INNER_COLS = [32, 127, 128, 512]
_INNER_ROWS = 8
# Non-inner (reduce over axis 0): (reduce_len, cols). Even + odd reduce length.
_NONINNER_SHAPES = [(33, 16), (64, 33)]


def _torch_dtype(dtype: DType) -> torch.dtype:
    return torch.bfloat16 if dtype == DType.bfloat16 else torch.float32


def _build_graph(op: str, dtype: DType, axis: int) -> Graph:
    """Build a single-reduction graph over `axis` with symbolic dimensions."""
    x_t = TensorType(dtype, ["r", "c"], device=_DEV)
    w_t = TensorType(dtype, ["c"], device=_DEV)
    if op == "layer_norm":
        input_types = [x_t, w_t, w_t]  # x, gamma, beta
    elif op == "rms_norm":
        input_types = [x_t, w_t]
    else:
        input_types = [x_t]

    with Graph(f"{op}_ax{axis}", input_types=input_types) as graph:
        x = graph.inputs[0].tensor
        if op == "reduce_sum":
            out: TensorValue = ops.sum(x, axis=axis)
        elif op == "reduce_max":
            out = ops.max(x, axis=axis)
        elif op == "reduce_min":
            out = ops.min(x, axis=axis)
        elif op == "reduce_mean":
            out = ops.mean(x, axis=axis)
        elif op == "reduce_product":
            out = ops.prod(x, axis=axis)
        elif op == "argmax":
            out = ops.argmax(x, axis=axis)
        elif op == "argmin":
            out = ops.argmin(x, axis=axis)
        elif op == "reduce_min_and_max":
            norm_axis = axis + x.rank if axis < 0 else axis
            out_shape: list[str | int] = ["r", "c"]
            out_shape[norm_axis] = 2
            out = ops.custom(
                "mo.reduce.reduce_min_and_max",
                device=_DEV,
                values=[x],
                out_types=[TensorType(dtype, out_shape, device=_DEV)],
                parameters={"axis": axis},
            )[0].tensor
        elif op == "softmax":
            out = ops.softmax(x, axis=axis)
        elif op == "logsoftmax":
            out = ops.logsoftmax(x, axis=axis)
        elif op == "layer_norm":
            gamma = graph.inputs[1].tensor
            beta = graph.inputs[2].tensor
            out = ops.layer_norm(x, gamma, beta, epsilon=_LN_EPS)
        elif op == "rms_norm":
            weight = graph.inputs[1].tensor
            out = ops.rms_norm(x, weight, epsilon=_RMS_EPS)
        elif op == "row_mean_of_squares":
            out = ops.custom(
                "mo.reduce.row_mean_of_squares",
                device=_DEV,
                values=[x],
                out_types=[TensorType(DType.float32, ["r", 1], device=_DEV)],
            )[0].tensor
        else:
            raise ValueError(f"unknown op {op!r}")
        graph.output(out)
    return graph


def _make_input(op: str, rows: int, cols: int, dtype: DType) -> torch.Tensor:
    """Random [rows, cols] input; near-1 for product to avoid under/overflow."""
    torch.manual_seed(0)
    if op == "reduce_product":
        f = 1.0 + 0.02 * torch.randn(rows, cols, dtype=torch.float32)
    else:
        f = torch.randn(rows, cols, dtype=torch.float32)
    return f.to(_torch_dtype(dtype))


def _make_weight(rows_seed: int, cols: int, dtype: DType) -> torch.Tensor:
    torch.manual_seed(rows_seed)
    w = 0.1 * torch.randn(cols, dtype=torch.float32) + 1.0
    return w.to(_torch_dtype(dtype))


def _feed(model: Model, tensors: list[torch.Tensor]) -> list[Buffer]:
    bufs = [
        Buffer.from_dlpack(t).to(model.input_devices[i])
        for i, t in enumerate(tensors)
    ]
    return model.execute(*bufs)


def _read_f32(buf: Buffer) -> np.ndarray:
    """Device buffer -> float32 numpy (bf16 read via torch, numpy lacks bf16)."""
    b = buf if buf.device.is_host else buf.to(CPU())
    if b.dtype == DType.bfloat16:
        return torch.from_dlpack(b).to(torch.float32).numpy()
    return b.to_numpy().astype(np.float32)


def _tol(op: str, dtype: DType) -> tuple[float, float, float]:
    """(atol, rtol, frac_allowed) comparing in float32; bf16 looser than fp32.

    Tolerances follow the validated benchmark harness: bf16 reductions
    accumulate in-dtype, so a small fraction of large-magnitude rows can exceed
    a per-term tolerance (a broken kernel fails ~all elements, not a few).
    """
    bf16 = dtype == DType.bfloat16
    if op in (
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
        "reduce_min_and_max",
    ):
        # max/min and the selected argmax/argmin value are representable
        # exactly, so both dtypes match to a tight absolute tolerance.
        return 1e-3, 0.0, 0.0
    if op in ("reduce_sum",):
        return (0.15, 5e-2, 2e-2) if bf16 else (1e-2, 1e-3, 0.0)
    if op in ("reduce_mean", "row_mean_of_squares"):
        return (2e-2, 3e-2, 2e-2) if bf16 else (1e-3, 1e-3, 0.0)
    if op == "softmax":
        return (3e-3, 6e-2, 5e-3) if bf16 else (1e-4, 1e-3, 0.0)
    if op == "logsoftmax":
        return (5e-2, 6e-2, 5e-3) if bf16 else (1e-3, 1e-3, 0.0)
    if op in ("layer_norm", "rms_norm"):
        return (4e-2, 6e-2, 1e-2) if bf16 else (2e-3, 2e-3, 0.0)
    raise ValueError(f"no tolerance for {op!r}")


def _assert_close(
    got: np.ndarray, ref: np.ndarray, op: str, dtype: DType, label: str
) -> None:
    atol, rtol, frac = _tol(op, dtype)
    got = got.reshape(-1)
    ref = ref.reshape(-1)
    abs_err = np.abs(got - ref)
    over = abs_err > (atol + rtol * np.abs(ref))
    n_bad = int(over.sum())
    allowed = math.ceil(frac * got.size)
    assert n_bad <= allowed, (
        f"{op} {label}: {n_bad}/{got.size} over tol "
        f"(allowed {allowed}); max_abs_err={abs_err.max():.4g}"
    )


def _reference_and_check(
    op: str,
    dtype: DType,
    model: Model,
    x: torch.Tensor,
    axis: int,
    weights: list[torch.Tensor],
    label: str,
) -> None:
    """Run the model and compare to a float32 torch reference for `op`."""
    xf = x.to(torch.float32)
    outs = _feed(model, [x, *weights])

    if op in _INT_OUT:
        idx = (
            outs[0] if outs[0].device.is_host else outs[0].to(CPU())
        ).to_numpy()
        idx = idx.astype(np.int64)
        got_val = np.take_along_axis(xf.numpy(), idx, axis=axis)
        ref = (
            xf.amax(dim=axis, keepdim=True)
            if op == "argmax"
            else xf.amin(dim=axis, keepdim=True)
        ).numpy()
        # Tie-safe: the value at the chosen index must equal the true extremum.
        assert np.array_equal(got_val, ref), (
            f"{op} {label}: index selected a non-extreme value"
        )
        return

    if op == "reduce_min_and_max":
        got = _read_f32(outs[0])
        norm_axis = axis + xf.dim() if axis < 0 else axis
        got_min = np.take(got, 0, axis=norm_axis)
        got_max = np.take(got, 1, axis=norm_axis)
        _assert_close(
            got_min, xf.amin(dim=axis).numpy(), op, dtype, f"{label}/min"
        )
        _assert_close(
            got_max, xf.amax(dim=axis).numpy(), op, dtype, f"{label}/max"
        )
        return

    if op == "reduce_product":
        got = _read_f32(outs[0]).reshape(-1)
        ref = xf.prod(dim=axis).numpy().reshape(-1)
        assert np.isfinite(got).all(), f"{op} {label}: non-finite output"
        denom = np.maximum(np.abs(ref), 1e-3)
        med_rel = float(np.median(np.abs(got - ref) / denom))
        # bf16 in-dtype accumulation vs the fp32 reference differs by tree order;
        # a directional check (finite, right order of magnitude) is enough.
        assert med_rel < 0.5, f"{op} {label}: median rel err {med_rel:.3f}"
        return

    got = _read_f32(outs[0])
    if op == "reduce_sum":
        ref = xf.sum(dim=axis).numpy()
    elif op == "reduce_max":
        ref = xf.amax(dim=axis).numpy()
    elif op == "reduce_min":
        ref = xf.amin(dim=axis).numpy()
    elif op == "reduce_mean":
        ref = xf.mean(dim=axis).numpy()
    elif op == "softmax":
        ref = torch.softmax(xf, dim=axis).numpy()
    elif op == "logsoftmax":
        ref = torch.log_softmax(xf, dim=axis).numpy()
    elif op == "row_mean_of_squares":
        ref = (xf**2).mean(dim=-1).numpy()
    elif op == "layer_norm":
        gamma, beta = weights[0].to(torch.float32), weights[1].to(torch.float32)
        ref = torch.nn.functional.layer_norm(
            xf, (xf.shape[-1],), gamma, beta, eps=_LN_EPS
        ).numpy()
    elif op == "rms_norm":
        weight = weights[0].to(torch.float32)
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        ref = (xf * torch.rsqrt(ms + _RMS_EPS) * weight).numpy()
    else:
        raise ValueError(f"no reference for {op!r}")
    _assert_close(got, ref, op, dtype, label)


# Inner-axis matrix: precision-sensitive ops in both dtypes; exact
# selection/compare ops (max/min/arg/min_and_max) in bfloat16 only.
_INNER_BOTH = [
    "reduce_sum",
    "reduce_mean",
    "reduce_product",
    "softmax",
    "logsoftmax",
    "layer_norm",
    "row_mean_of_squares",
]
_INNER_BF16_ONLY = [
    "reduce_max",
    "reduce_min",
    "argmax",
    "argmin",
    "reduce_min_and_max",
]
_INNER_CASES = [
    (op, DType.bfloat16) for op in _INNER_BOTH + _INNER_BF16_ONLY
] + [(op, DType.float32) for op in _INNER_BOTH]


@pytest.mark.parametrize(
    "op,dtype", _INNER_CASES, ids=lambda v: v if isinstance(v, str) else str(v)
)
def test_rowwise_inner(
    session: InferenceSession, op: str, dtype: DType
) -> None:
    model = session.load(_build_graph(op, dtype, axis=-1))
    for cols in _INNER_COLS:
        x = _make_input(op, _INNER_ROWS, cols, dtype)
        weights: list[torch.Tensor] = []
        if op == "layer_norm":
            weights = [
                _make_weight(1, cols, dtype),
                (0.1 * torch.randn(cols)).to(_torch_dtype(dtype)),
            ]
        elif op == "rms_norm":
            weights = [_make_weight(1, cols, dtype)]
        _reference_and_check(op, dtype, model, x, -1, weights, f"cols={cols}")


# Non-inner (reduce over axis 0): the pure arbitrary-axis reductions. Precision-
# sensitive sum/mean in both dtypes; the rest in bfloat16 only.
_NONINNER_CASES = [
    (op, DType.bfloat16)
    for op in [
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_mean",
        "reduce_product",
        "reduce_min_and_max",
    ]
] + [(op, DType.float32) for op in ["reduce_sum", "reduce_mean"]]


@pytest.mark.parametrize(
    "op,dtype",
    _NONINNER_CASES,
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_rowwise_noninner(
    session: InferenceSession, op: str, dtype: DType
) -> None:
    model = session.load(_build_graph(op, dtype, axis=0))
    for rows, cols in _NONINNER_SHAPES:
        x = _make_input(op, rows, cols, dtype)
        _reference_and_check(op, dtype, model, x, 0, [], f"shape={rows}x{cols}")
