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
pure arbitrary-axis reductions -- on a non-inner axis, in bfloat16, float16 and
float32 where precision makes it meaningful, plus a CPU-only 8-byte-element
group (see :data:`WIDE_CPU_SPECS`) for what only the widest element can reach.

Every case runs on both the CPU and the GPU -- the GPU is where the cooperative
tiers, the cross-thread combine and the split-K path live, so it is the priority
target rather than an afterthought. Two further inner-axis groups are GPU-only:
the ops whose monoid state is narrower than the 4-byte word that combine
exchanges (:data:`GPU_SUBWORD_SPECS`) and the split-K cross-block join
(:data:`GPU_SPLITK_SPECS`), neither of which any CPU case can reach.

These are small, fast CI shapes. The bandwidth-oriented perf grid for the same
ops lives in the manual benchmark at
``//utils/benchmarking/kepler/graph:reductions`` and is not run here.

The graphs are compiled to MEFs by CPU-only build actions
(``:rowwise_reduction_mefs`` via ``mef_precompile.bzl``); this test does NOT
compile. It initializes one symbolic-dimension MEF per parametrization and
feeds several concrete shapes as data, so the GPU worker only ever initializes
and executes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from _rowwise_reduction_specs import (
    GPU_SPLITK_SPECS,
    GPU_SUBWORD_SPECS,
    INNER_SPECS,
    NONINNER_SPECS,
    WIDE_CPU_SPECS,
    RowwiseSpec,
)
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from test_common.mef_precompile import init_from_mef, mefs_from_env
from test_common.reduction_graphs import LAYER_NORM_EPS, RMS_NORM_EPS

# Ops whose output is integer indices.
_INT_OUT = {"argmax", "argmin"}

# Small inner-axis column counts: even + odd, small + a wider row.
_INNER_COLS = [32, 127, 128, 512]
_INNER_ROWS = 8
# Non-inner (reduce over axis 0): (reduce_len, cols). Even + odd reduce length,
# even + odd cols, and short reduce axes crossed with SIMD-divisible and
# non-SIMD-divisible cols at small and large output counts.
_NONINNER_SHAPES = [(33, 16), (64, 33), (8, 8192), (8, 8193), (8, 9)]


@pytest.fixture(scope="module")
def mefs() -> dict[str, Path]:
    return mefs_from_env("ROWWISE_MEF_RLOCATIONS")


def _load(
    session: InferenceSession, mefs: dict[str, Path], spec: RowwiseSpec
) -> Model:
    return init_from_mef(session, mefs[f"{spec.name}.mef"])


def _torch_dtype(dtype: DType) -> torch.dtype:
    return {
        DType.bfloat16: torch.bfloat16,
        DType.float16: torch.float16,
        DType.bool: torch.bool,
        DType.int8: torch.int8,
        DType.int16: torch.int16,
        DType.int64: torch.int64,
        DType.float64: torch.float64,
    }.get(dtype, torch.float32)


def _make_input(op: str, rows: int, cols: int, dtype: DType) -> torch.Tensor:
    """Random [rows, cols] input; near-1 for product to avoid under/overflow."""
    torch.manual_seed(0)
    if dtype == DType.bool:
        # Sparse enough that a dropped element flips the row's max/min: most
        # rows are all-False with a single True (and vice versa for min).
        f = torch.rand(rows, cols, dtype=torch.float32)
        return (f > 1.0 - 2.0 / cols) | (f < 1.0 / cols)
    if dtype in (DType.int8, DType.int16, DType.int64):
        return torch.randint(-100, 100, (rows, cols), dtype=torch.int32).to(
            _torch_dtype(dtype)
        )
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
    """(atol, rtol, frac_allowed) comparing in float32; half looser than fp32.

    Tolerances follow the validated benchmark harness: bf16 reductions
    accumulate in-dtype, so a small fraction of large-magnitude rows can exceed
    a per-term tolerance (a broken kernel fails ~all elements, not a few). fp16
    shares the bf16 budget -- it accumulates in-dtype just the same, with a
    wider mantissa, so the bf16 numbers bound it.
    """
    half = dtype in (DType.bfloat16, DType.float16)
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
    if op == "reduce_sum":
        return (0.15, 5e-2, 2e-2) if half else (1e-2, 1e-3, 0.0)
    if op in ("reduce_mean", "row_mean_of_squares"):
        return (2e-2, 3e-2, 2e-2) if half else (1e-3, 1e-3, 0.0)
    if op == "softmax":
        return (3e-3, 6e-2, 5e-3) if half else (1e-4, 1e-3, 0.0)
    if op == "logsoftmax":
        return (5e-2, 6e-2, 5e-3) if half else (1e-3, 1e-3, 0.0)
    if op in ("layer_norm", "rms_norm"):
        return (4e-2, 6e-2, 1e-2) if half else (2e-3, 2e-3, 0.0)
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
            xf, (xf.shape[-1],), gamma, beta, eps=LAYER_NORM_EPS
        ).numpy()
    elif op == "rms_norm":
        weight = weights[0].to(torch.float32)
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        ref = (xf * torch.rsqrt(ms + RMS_NORM_EPS) * weight).numpy()
    else:
        raise ValueError(f"no reference for {op!r}")
    _assert_close(got, ref, op, dtype, label)


@pytest.mark.parametrize("spec", INNER_SPECS, ids=lambda s: s.name)
def test_rowwise_inner(
    session: InferenceSession, mefs: dict[str, Path], spec: RowwiseSpec
) -> None:
    model = _load(session, mefs, spec)
    for cols in _INNER_COLS:
        x = _make_input(spec.op, _INNER_ROWS, cols, spec.dtype)
        weights: list[torch.Tensor] = []
        if spec.op == "layer_norm":
            weights = [
                _make_weight(1, cols, spec.dtype),
                (0.1 * torch.randn(cols)).to(_torch_dtype(spec.dtype)),
            ]
        elif spec.op == "rms_norm":
            weights = [_make_weight(1, cols, spec.dtype)]
        _reference_and_check(
            spec.op, spec.dtype, model, x, -1, weights, f"cols={cols}"
        )


@pytest.mark.parametrize("spec", NONINNER_SPECS, ids=lambda s: s.name)
def test_rowwise_noninner(
    session: InferenceSession, mefs: dict[str, Path], spec: RowwiseSpec
) -> None:
    model = _load(session, mefs, spec)
    for rows, cols in _NONINNER_SHAPES:
        x = _make_input(spec.op, rows, cols, spec.dtype)
        _reference_and_check(
            spec.op, spec.dtype, model, x, 0, [], f"shape={rows}x{cols}"
        )


@pytest.mark.parametrize("spec", WIDE_CPU_SPECS, ids=lambda s: s.name)
def test_rowwise_wide_element_cpu(
    session: InferenceSession, mefs: dict[str, Path], spec: RowwiseSpec
) -> None:
    model = _load(session, mefs, spec)
    shapes = (
        [(_INNER_ROWS, cols) for cols in _INNER_COLS]
        if spec.axis == -1
        else _NONINNER_SHAPES
    )
    for rows, cols in shapes:
        x = _make_input(spec.op, rows, cols, spec.dtype)
        _reference_and_check(
            spec.op,
            spec.dtype,
            model,
            x,
            spec.axis,
            [],
            f"shape={rows}x{cols}",
        )


@pytest.mark.parametrize("spec", GPU_SUBWORD_SPECS, ids=lambda s: s.name)
def test_rowwise_inner_gpu_subword_state(
    session: InferenceSession, mefs: dict[str, Path], spec: RowwiseSpec
) -> None:
    model = _load(session, mefs, spec)
    for cols in _INNER_COLS:
        x = _make_input(spec.op, _INNER_ROWS, cols, spec.dtype)
        _reference_and_check(
            spec.op, spec.dtype, model, x, -1, [], f"cols={cols}"
        )


# A few-row, long-row shape is what routes the inner-axis reduction through the
# split-K tier (`num_rows` under the SM count and at least `_SPLITK_MIN_ROW` =
# 32768 elements per row at `simd_width <= 4`, i.e. float32 here), whose
# cross-block finish in `rowwise.pjoin` is the only place the scaffolder combines
# whole per-block monoid states. Every other case in this file is orders of
# magnitude below that element floor, so nothing else reaches it.
_SPLITK_ROWS = 8
_SPLITK_COLS = 40960


@pytest.mark.parametrize("spec", GPU_SPLITK_SPECS, ids=lambda s: s.name)
def test_rowwise_inner_gpu_splitk(
    session: InferenceSession, mefs: dict[str, Path], spec: RowwiseSpec
) -> None:
    model = _load(session, mefs, spec)
    x = _make_input(spec.op, _SPLITK_ROWS, _SPLITK_COLS, spec.dtype)
    _reference_and_check(
        spec.op, spec.dtype, model, x, -1, [], f"cols={_SPLITK_COLS}"
    )
