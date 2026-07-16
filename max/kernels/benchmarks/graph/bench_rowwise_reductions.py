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
"""Graph-level row-wise reduction benchmark (manual perf tool, not CI).

Builds a `Graph` per row-wise reduction op via the public `max.graph` API,
`session.load`s it, checks correctness against an fp32 numpy reference
(`--check-only`), and times `model.execute()` on a large bandwidth-bound grid.
Every op runs on MAX's own Mojo row-wise kernels (`max.kernels` Row API); this
tool exists to track their bandwidth, not to gate CI. The CI correctness test
for the same ops lives at
`max/tests/integration/graph/test_rowwise_reductions.py`.

Timing protocol: build + `session.load` + input alloc + H2D + warmup are OUTSIDE
the timer; only the `model.execute(buf)` loop + one trailing
`device.synchronize()` is inside. Norm weights (gamma/beta/weight) are baked as
bf16 constants so each timed `execute` is a single-input call.

Reductions
----------
Pure (true reductions, arbitrary axis -> inner + non-inner measured):
  reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_product,
  argmax, argmin, reduce_min_and_max
Norm-type (last-axis-defined -> non-inner is n/a):
  softmax, logsoftmax, layer_norm, rms_norm, row_mean_of_squares, group_norm
Fused composites (last-axis-defined, GPU-only GraphCompiler fusion products,
built as their unfused public-op pattern + verified fused via kernel_summaries):
  rms_norm_rope, rms_norm_fused_quantize_fp8

Internal ops (row_mean_of_squares, reduce_min_and_max) are built directly via
`ops.custom("mo.reduce.X", ...)` so this stays on the clean driver/dtype/engine/
graph dep set.

Grid
----
Inner (last) axis, 2D [rows, cols], even+odd cols in
{128,129,256,255,512,511,1024,1023,2048,2047}; rows = round(512e6/cols) so each
tensor is ~512M elements (fixed per-execute overhead < ~3%). ALL reductions run
the inner grid. Pure reductions ALSO run a 3D non-inner grid
([1024,512,1024] ~= 537M) reducing over axis 0 and axis 1.

reduce_product uses a near-1 input (1 + 0.02*randn) instead of randn: the
product of hundreds of standard-normal magnitudes underflows bf16/f32 to ~0 and
makes the correctness check degenerate. Near-1 values keep the product O(1) and
read the identical byte volume, so timing stays comparable.

Deps: `max` + `numpy` only (bf16 host<->device via uint16 bit-reinterpret). Run:

  ./bazelw run //max/kernels/benchmarks/graph:bench_rowwise_reductions \
    --curses=no --noshow_progress -- --device gpu
  taskset -c 0-7 ./bazelw run \
    //max/kernels/benchmarks/graph:bench_rowwise_reductions \
    --curses=no --noshow_progress -- --device cpu

Legacy-vs-new A/B is done by `git checkout`ing the baseline commit, building,
running this tool, then repeating on the branch tip (the tool always benchmarks
whatever Row-API kernels the current checkout compiles).
"""

from __future__ import annotations

import argparse
import statistics
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

_TARGET_ELEMS = 512_000_000
_INNER_COLS = [
    32,
    128,
    129,
    256,
    255,
    512,
    511,
    1024,
    1023,
    2048,
    2047,
    4096,
    8192,
    16384,
]
# Non-inner grid: one 3D tensor (~537M elem); reduce over axis 0 and axis 1.
_NONINNER_SHAPE: tuple[int, int, int] = (1024, 512, 1024)
_NONINNER_AXES = [0, 1]

_LAYER_NORM_EPS = 1e-5
_RMS_NORM_EPS = 1e-6
_GROUP_NORM_EPS = 1e-5

# Rows compared vs the numpy reference for full-size-output ops
# (softmax/logsoftmax/layer_norm/rms_norm/group_norm). Same kernel per row, so a
# leading slice is representative and bounds the host-side reference cost.
_N_CHECK_ROWS = 16384

# Pure = true reductions, defined on an arbitrary axis.
_PURE = [
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_mean",
    "reduce_product",
    "argmax",
    "argmin",
    "reduce_min_and_max",
]
# Norm-type = last-axis-defined; non-inner is n/a.
_NORM = [
    "softmax",
    "logsoftmax",
    "layer_norm",
    "rms_norm",
    "row_mean_of_squares",
]
# Fused composites (last-axis-defined). These are NOT directly-constructable
# custom ops -- they are GraphCompiler *fusion products* (`mo.composite.*`).
# The harness builds the exact unfused public-op pattern the fusion matches and
# relies on `session.load` to fuse it into the single composite GPU kernel;
# `check_correctness` verifies via `model.kernel_summaries` that the composite
# kernel actually fired (else we would be timing the unfused fallback). Both are
# GPU-only.
#   rms_norm_rope: rms_norm (f32 sandwich) -> split -> concat(-x2, x1) ->
#                  normed*cos + rotated*sin. Requires an EVEN last dim.
#   rms_norm_fused_quantize_fp8: rms_norm(multiply_before_cast=True) ->
#                  mo.quantize_dynamic_scaled_float8 (per-token). Two outputs
#                  (fp8 tensor + per-row scale).
_COMPOSITE = [
    "rms_norm_rope",
    "rms_norm_fused_quantize_fp8",
]
_ALL = _PURE + _NORM + _COMPOSITE
# Same-shape output (read + write full volume) vs a shape-shrinking reduction.
# rms_norm_rope writes a same-shape bf16 output; the fp8 composite writes a
# 1-byte fp8 output (+ tiny per-row scale), so it stays read-dominated (~1x).
_FULL_OUTPUT = {
    "softmax",
    "logsoftmax",
    "layer_norm",
    "rms_norm",
    "rms_norm_rope",
}
_FP8_SCALE_UB = 1200.0


def _f32_to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit-pattern as uint16, round-to-nearest-even."""
    u32 = f32.astype(np.float32).view(np.uint32)
    bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u32 + bias) >> 16).astype(np.uint16)


def _bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    """bfloat16 bit-pattern (uint16) -> float32, high-half placement."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _round_bf16(f32: np.ndarray) -> np.ndarray:
    """Round a float32 array to the nearest bf16 value (as float32)."""
    return _bf16_u16_to_f32(_f32_to_bf16_u16(f32.astype(np.float32)))


def _e4m3fn_u8_to_f32(u8: np.ndarray) -> np.ndarray:
    """Decode float8_e4m3fn bit patterns (uint8) to float32.

    e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa; no infinities. Max finite
    is 448 (exp=15, man=6); the sole NaN is S.1111.111. numpy has no fp8 dtype,
    so the fp8 output buffer is read as raw uint8 and decoded here (keeps the
    harness on the numpy-only dep set).
    """
    b = u8.astype(np.uint32)
    sign = np.where((b >> 7) & 1, -1.0, 1.0).astype(np.float32)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float32)
    subnormal = man * np.float32(2.0**-9)
    normal = (1.0 + man / 8.0) * np.exp2((exp - 7).astype(np.float32))
    val = np.where(exp == 0, subnormal, normal).astype(np.float32)
    val = np.where((exp == 0xF) & (man == 7.0), np.nan, val)
    return (sign * val).astype(np.float32)


def _bf16_accum_prod(x: np.ndarray, axis: int) -> np.ndarray:
    """Product over `axis` emulating bf16 in-dtype accumulation.

    The reduction runs in the input dtype (bf16), so a plain f32 numpy product
    diverges by the accumulated bf16 rounding. This rounds to bf16 after each
    multiply, capturing the dominant precision loss (GPU tree order still differs
    slightly, hence product's tolerance stays loose).
    """
    xm = np.moveaxis(x, axis, -1)
    acc = np.ones(xm.shape[:-1], dtype=np.float32)
    for i in range(xm.shape[-1]):
        acc = _round_bf16(acc * xm[..., i])
    return acc


def _group_norm_num_groups(cols: int) -> int:
    """Largest of {32,16,8,4,2,1} dividing cols with per-group >= 8.

    The group_norm GPU kernel requires the per-group reduce width
    (cols // num_groups) to be >= simd_width (8), so cap accordingly
    (e.g. cols=128 -> 16 groups of 8, not 32 groups of 4). Odd cols -> 1.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if cols % g == 0 and cols // g >= 8:
            return g
    return 1


@dataclass
class Aux:
    """Baked-constant weights for the norms (bf16), plus their exact f32 values.

    The uint16 arrays are retained because `Buffer.from_dlpack` views their
    memory zero-copy; they must stay alive until `ops.constant` copies the data
    into the graph attribute at build time.
    """

    bf16_bufs: dict[str, Buffer] = field(default_factory=dict)
    f32_vals: dict[str, np.ndarray] = field(default_factory=dict)
    dtypes: dict[str, DType] = field(default_factory=dict)
    num_groups: int = 1
    _keepalive: list[np.ndarray] = field(default_factory=list)

    def add(
        self, name: str, vals_f32: np.ndarray, dtype: DType = DType.bfloat16
    ) -> None:
        """Bake a weight/table constant at `dtype`; keep its exact host f32 view.

        `dtype=bfloat16` reinterprets via uint16; `float32` bakes verbatim. The
        recorded dtype lets `build_graph` emit a matching `ops.constant`.
        """
        self.dtypes[name] = dtype
        if dtype == DType.float32:
            f32 = np.ascontiguousarray(vals_f32.astype(np.float32))
            self._keepalive.append(f32)
            self.bf16_bufs[name] = Buffer.from_dlpack(f32)
            self.f32_vals[name] = f32
            return
        u16 = np.ascontiguousarray(_f32_to_bf16_u16(vals_f32))
        self._keepalive.append(u16)
        self.bf16_bufs[name] = Buffer.from_dlpack(u16).view(DType.bfloat16)
        self.f32_vals[name] = _bf16_u16_to_f32(u16)


def make_aux(name: str, cols: int) -> Aux:
    """Random bf16 gamma/beta/weight for the norms; empty otherwise."""
    aux = Aux()
    rng = np.random.default_rng(1)
    if name in ("layer_norm", "group_norm"):
        aux.add(
            "gamma", (rng.standard_normal(cols) * 0.1 + 1.0).astype(np.float32)
        )
        aux.add("beta", (rng.standard_normal(cols) * 0.1).astype(np.float32))
        if name == "group_norm":
            aux.num_groups = _group_norm_num_groups(cols)
    elif name == "rms_norm":
        aux.add(
            "weight", (rng.standard_normal(cols) * 0.1 + 1.0).astype(np.float32)
        )
    elif name == "rms_norm_rope":
        # f32 sandwich weight (the fusion keeps weight/epsilon in f32); cos/sin
        # are baked bf16 [1, cols] tables broadcast over rows inside the graph
        # (a full [rows, cols] table would be a multi-GB constant). Their values
        # are treated elementwise by both the kernel and the numpy reference, so
        # arbitrary cos/sin of random angles (magnitudes in [-1, 1]) suffice.
        aux.add(
            "weight",
            (rng.standard_normal(cols) * 0.1 + 1.0).astype(np.float32),
            dtype=DType.float32,
        )
        theta = rng.standard_normal(cols).astype(np.float32)
        aux.add("cos", np.cos(theta)[None, :])
        aux.add("sin", np.sin(theta)[None, :])
    elif name == "rms_norm_fused_quantize_fp8":
        # bf16 weight (rms_norm weight must match the bf16 input dtype); the
        # fusion requires multiply_before_cast=True on the rms_norm.
        aux.add(
            "weight", (rng.standard_normal(cols) * 0.1 + 1.0).astype(np.float32)
        )
    return aux


def build_graph(
    name: str,
    shape: tuple[int, ...],
    axis: int,
    device_ref: DeviceRef,
    aux: Aux,
) -> Graph:
    """Single-reduction graph over `axis` of a bf16 tensor of `shape`."""
    input_type = TensorType(DType.bfloat16, list(shape), device=device_ref)
    with Graph(f"{name}_axis{axis}", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        if name == "reduce_sum":
            out: TensorValue = ops.sum(x, axis=axis)
        elif name == "reduce_max":
            out = ops.max(x, axis=axis)
        elif name == "reduce_min":
            out = ops.min(x, axis=axis)
        elif name == "reduce_mean":
            out = ops.mean(x, axis=axis)
        elif name == "reduce_product":
            out = ops.prod(x, axis=axis)
        elif name == "argmax":
            out = ops.argmax(x, axis=axis)
        elif name == "argmin":
            out = ops.argmin(x, axis=axis)
        elif name == "reduce_min_and_max":
            norm_axis = axis + x.rank if axis < 0 else axis
            out_shape = list(shape)
            out_shape[norm_axis] = 2
            out = ops.custom(
                "mo.reduce.reduce_min_and_max",
                device=device_ref,
                values=[x],
                out_types=[
                    TensorType(DType.bfloat16, out_shape, device=device_ref)
                ],
                parameters={"axis": axis},
            )[0].tensor
        elif name == "softmax":
            out = ops.softmax(x, axis=axis)
        elif name == "logsoftmax":
            out = ops.logsoftmax(x, axis=axis)
        elif name == "layer_norm":
            gamma = ops.constant(
                aux.bf16_bufs["gamma"], DType.bfloat16, device_ref
            )
            beta = ops.constant(
                aux.bf16_bufs["beta"], DType.bfloat16, device_ref
            )
            out = ops.layer_norm(x, gamma, beta, epsilon=_LAYER_NORM_EPS)
        elif name == "rms_norm":
            weight = ops.constant(
                aux.bf16_bufs["weight"], DType.bfloat16, device_ref
            )
            out = ops.rms_norm(x, weight, epsilon=_RMS_NORM_EPS)
        elif name == "group_norm":
            # group_norm requires rank 3/4: input is [rows, channels=cols, 1]
            # (spatial=1); it groups the channel axis into num_groups.
            gamma = ops.constant(
                aux.bf16_bufs["gamma"], DType.bfloat16, device_ref
            )
            beta = ops.constant(
                aux.bf16_bufs["beta"], DType.bfloat16, device_ref
            )
            out = ops.group_norm(
                x,
                gamma,
                beta,
                num_groups=aux.num_groups,
                epsilon=_GROUP_NORM_EPS,
            )
        elif name == "row_mean_of_squares":
            out = ops.custom(
                "mo.reduce.row_mean_of_squares",
                device=device_ref,
                values=[x],
                out_types=[
                    TensorType(DType.float32, [shape[0], 1], device=device_ref)
                ],
            )[0].tensor
        elif name == "rms_norm_rope":
            # Unfused f32-sandwich RMSNorm + rotate-half RoPE. The GraphCompiler
            # ReduceRMSNormRoPEPattern fuses this exact shape into the single
            # mo.composite.rms_norm_rope kernel (input upcast absorbed by
            # prologue fusion, bf16 output via the kernel's decoupled dtype).
            rows, cols = shape[0], shape[-1]
            half = cols // 2
            weight = ops.constant(
                aux.bf16_bufs["weight"], DType.float32, device_ref
            )
            normed = ops.rms_norm(
                x.cast(DType.float32), weight, epsilon=_RMS_NORM_EPS
            ).cast(DType.bfloat16)
            x1, x2 = ops.split(normed, [half, half], axis=-1)
            rotated = ops.concat([-x2, x1], axis=-1)
            cos_b = ops.constant(
                aux.bf16_bufs["cos"], DType.bfloat16, device_ref
            ).broadcast_to([rows, cols])
            sin_b = ops.constant(
                aux.bf16_bufs["sin"], DType.bfloat16, device_ref
            ).broadcast_to([rows, cols])
            out = normed * cos_b + rotated * sin_b
        elif name == "rms_norm_fused_quantize_fp8":
            # Unfused RMSNorm(multiply_before_cast=True) + per-token dynamic-
            # scaled fp8 quantize. The RMSNormFusedQuantizeDynamicScaledFP8
            # pattern fuses these into mo.composite.rms_norm_fused_quantize_
            # dynamic_scaled_fp8 (two outputs: fp8 tensor + per-row scale). The
            # scale is written at [row, 0] (reduced axis pinned to 0), so the
            # scale out_type is [rows, 1].
            rows, cols = shape[0], shape[-1]
            weight = ops.constant(
                aux.bf16_bufs["weight"], DType.bfloat16, device_ref
            )
            normed = ops.rms_norm(
                x, weight, epsilon=_RMS_NORM_EPS, multiply_before_cast=True
            )
            scale_ub = ops.constant(
                _FP8_SCALE_UB, DType.float32, device=DeviceRef.CPU()
            )
            fp8_out, scale = ops.custom(
                "mo.quantize_dynamic_scaled_float8",
                device=device_ref,
                values=[normed, scale_ub],
                out_types=[
                    TensorType(
                        DType.float8_e4m3fn, [rows, cols], device=device_ref
                    ),
                    TensorType(DType.float32, [rows, 1], device=device_ref),
                ],
                parameters={"group_size_or_per_token": -1},
            )
            graph.output(fp8_out.tensor, scale.tensor)
            return graph
        else:
            raise ValueError(f"unknown reduction {name!r}")
        graph.output(out)
    return graph


def make_input(
    name: str, shape: tuple[int, ...], device: Device
) -> tuple[Buffer, np.ndarray]:
    """Random bf16 input on `device`; return (device buffer, exact f32 host view).

    reduce_product uses near-1 values (see module docstring); all other ops use
    standard-normal values.
    """
    rng = np.random.default_rng(0)
    if name == "reduce_product":
        f32 = (1.0 + 0.02 * rng.standard_normal(shape)).astype(np.float32)
    else:
        f32 = rng.standard_normal(shape).astype(np.float32)
    u16 = np.ascontiguousarray(_f32_to_bf16_u16(f32))
    buf = Buffer.from_dlpack(u16).view(DType.bfloat16).to(device)
    host_f32 = _bf16_u16_to_f32(u16)
    return buf, host_f32


def _out_to_f32(out: Buffer) -> np.ndarray:
    """(possibly bf16) device buffer -> f32 host array, same shape."""
    cpu = out if out.device.is_host else out.to(CPU())
    if cpu.dtype == DType.bfloat16:
        return _bf16_u16_to_f32(cpu.view(DType.uint16).to_numpy())
    return cpu.to_numpy().astype(np.float32)


def _reduce_len(shape: tuple[int, ...], axis: int) -> int:
    return shape[axis]


def _check_composite(
    name: str,
    model: Model,
    outs: list[Buffer],
    host_f32: np.ndarray,
    aux: Aux,
) -> tuple[bool, str]:
    """Correctness for the fused composites (last-axis, [rows, cols] input).

    Also asserts the composite kernel actually FIRED (via
    ``model.kernel_summaries``) -- otherwise the graph compiler left the unfused
    pattern in place and the timing would be for the wrong kernel.
    """
    n = int(np.minimum(_N_CHECK_ROWS, host_f32.shape[0]))
    xin = host_f32[:n].astype(np.float32)  # [n, cols]
    cols = xin.shape[-1]
    weight = aux.f32_vals["weight"]  # [cols]
    inv_rms = 1.0 / np.sqrt((xin**2).mean(-1, keepdims=True) + _RMS_NORM_EPS)
    summ = " ".join(model.kernel_summaries)

    if name == "rms_norm_rope":
        fused = "rms_norm_rope" in summ
        # f32-sandwich RMSNorm (multiply in f32) then downcast to bf16 before
        # the rotate-half RoPE, matching the unfused graph the kernel replaces.
        normed = _round_bf16(xin * inv_rms * weight)  # [n, cols]
        half = cols // 2
        x1, x2 = normed[..., :half], normed[..., half:]
        rotated = np.concatenate([-x2, x1], axis=-1)
        cos = aux.f32_vals["cos"]  # [1, cols], broadcast over rows
        sin = aux.f32_vals["sin"]
        ref = normed * cos + rotated * sin
        got = _out_to_f32(outs[0])[:n].reshape(n, cols)
        abs_err = np.abs(got - ref)
        tol = 5e-2 + 8e-2 * np.abs(ref)
        n_bad = int(np.sum(abs_err > tol))
        frac_bad = n_bad / got.size
        cos_sim = float(
            np.dot(got.ravel(), ref.ravel())
            / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12)
        )
        ok = fused and cos_sim > 0.999 and frac_bad <= 1e-2
        return (
            ok,
            (
                f"fused={fused} cos_sim={cos_sim:.5f} "
                f"max_abs_err={abs_err.max():.4g} over_tol={n_bad}/{got.size} "
                f"({frac_bad:.2%})"
            ),
        )

    # rms_norm_fused_quantize_fp8: multiply_before_cast=True, fused -> the
    # normalized row stays in f32 through the quantize (no bf16 rounding).
    fused = "fused_quantize_dynamic_scaled_fp8" in summ or (
        "rms_norm" in summ and "fp8" in summ
    )
    normed = (xin * inv_rms) * weight  # [n, cols] f32
    row_max = np.abs(normed).max(-1)  # [n]
    scale_factor = np.minimum(row_max, _FP8_SCALE_UB) / 448.0  # [n]
    scale_recip = np.where(scale_factor > 0, 1.0 / scale_factor, 0.0)
    # fp8 output: raw uint8 -> decoded scaled value (== normed * scale_recip).
    fp8_cpu = outs[0] if outs[0].device.is_host else outs[0].to(CPU())
    fp8_u8 = fp8_cpu.view(DType.uint8).to_numpy()
    got_q = _e4m3fn_u8_to_f32(fp8_u8)[:n].reshape(n, cols)
    got_scale = _out_to_f32(outs[1]).reshape(-1)[:n]  # [n]
    # scale check: reduction-order fp32 diffs in inv_rms/row_max -> small rel.
    scale_med_rel = float(
        np.median(
            np.abs(got_scale - scale_factor) / np.maximum(scale_factor, 1e-8)
        )
    )
    scale_ok = scale_med_rel < 5e-2
    # fp8 element check in scaled units: |decoded - clamp(normed*recip)| within
    # one e4m3 half-ULP (~2^-4 rel + 2^-9 abs). A few near-max elements per row
    # flip on reduction-order diffs (see agent memory), so allow a small frac.
    ref_scaled = np.clip(normed * scale_recip[:, None], -448.0, 448.0)
    half_ulp = np.abs(ref_scaled) * (2.0**-4) + (2.0**-9)
    finite = bool(np.isfinite(got_q).all())
    n_bad = int(np.sum(np.abs(got_q - ref_scaled) > half_ulp + 1e-6))
    frac_bad = n_bad / got_q.size
    ok = fused and finite and scale_ok and frac_bad <= 2e-2
    return (
        ok,
        (
            f"fused={fused} finite={finite} scale_med_rel={scale_med_rel:.4f} "
            f"fp8_over_ulp={n_bad}/{got_q.size} ({frac_bad:.2%})"
        ),
    )


def check_correctness(
    name: str,
    model: Model,
    buf: Buffer,
    host_f32: np.ndarray,
    axis: int,
    aux: Aux,
    device: Device,
) -> tuple[bool, str]:
    """Compare `model.execute` output to a numpy reference with a loose bf16 tol.

    Returns (ok, detail).
    """
    outs = model.execute(buf)
    out = outs[0]
    device.synchronize()
    rlen = _reduce_len(host_f32.shape, axis)

    if name in _COMPOSITE:
        return _check_composite(name, model, outs, host_f32, aux)

    if name in ("argmax", "argmin"):
        idx = (out if out.device.is_host else out.to(CPU())).to_numpy()
        idx = idx.astype(np.int64)  # keepdim shape, size-1 on `axis`
        got_val = np.take_along_axis(host_f32, idx, axis=axis)
        ref = (
            host_f32.max(axis=axis, keepdims=True)
            if name == "argmax"
            else host_f32.min(axis=axis, keepdims=True)
        )
        n_bad = int(np.sum(got_val != ref))
        ok = n_bad == 0
        return ok, f"value-match(tie-safe) n_mismatch={n_bad}/{idx.size}"

    if name == "reduce_min_and_max":
        got = _out_to_f32(out)
        got_min = np.take(got, 0, axis=axis)
        got_max = np.take(got, 1, axis=axis)
        ref_min = host_f32.min(axis=axis)
        ref_max = host_f32.max(axis=axis)
        n_bad = int(
            np.sum(np.abs(got_min - ref_min) > 1e-3)
            + np.sum(np.abs(got_max - ref_max) > 1e-3)
        )
        ok = n_bad == 0
        return (
            ok,
            f"min|max exact n_over_tol={n_bad}/{got_min.size + got_max.size}",
        )

    if name == "reduce_product":
        got = _out_to_f32(out).reshape(-1)
        ref = _bf16_accum_prod(host_f32, axis).reshape(-1)
        finite = bool(np.isfinite(got).all())
        denom = np.maximum(np.abs(ref), 1e-3)
        med_rel = float(np.median(np.abs(got - ref) / denom))
        # bf16 product accumulates in-dtype; the GPU tree order vs this sequential
        # emulation is order-sensitive, so this is a DIRECTIONAL check (op runs,
        # finite, right order of magnitude) rather than a tight numeric gate.
        ok = finite and med_rel < 0.5
        return (
            ok,
            (
                "bf16-directional"
                f" finite={finite} median_rel_err={med_rel:.3f} "
                f"max_abs_err={float(np.abs(got - ref).max()):.3g}"
            ),
        )

    got = _out_to_f32(out)

    if name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean"):
        got = got.reshape(-1)
        if name == "reduce_sum":
            ref = host_f32.sum(axis=axis).reshape(-1)
            # bf16 tree-sum vs f32 reference: error grows with the largest partial
            # sum, so a small fraction of large-|sum| rows exceed a per-term tol
            # (esp. non-inner over 512-1024 terms). Tolerate those; a broken reduce
            # fails ~all rows, not 2%.
            atol, rtol, frac_ok = 0.1 * float(np.sqrt(rlen)), 5e-2, 2e-2
        elif name == "reduce_mean":
            ref = host_f32.mean(axis=axis).reshape(-1)
            # mean = sum / N: sum-error/N ~ 0.1/sqrt(N); + bf16 output quant floor.
            atol, rtol, frac_ok = 0.1 / float(np.sqrt(rlen)) + 1e-2, 3e-2, 0.0
        else:  # max / min: representable exactly in bf16
            ref = (
                host_f32.max(axis=axis)
                if name == "reduce_max"
                else host_f32.min(axis=axis)
            ).reshape(-1)
            atol, rtol, frac_ok = 1e-3, 0.0, 0.0
    elif name == "row_mean_of_squares":
        got = got.reshape(-1)
        ref = (host_f32.astype(np.float32) ** 2).mean(axis=-1).reshape(-1)
        atol, rtol, frac_ok = 1e-2, 1e-2, 0.0
    else:  # softmax / logsoftmax / layer_norm / rms_norm / group_norm
        # group_norm input is [rows, cols, 1]; flatten trailing dims to [n, cols].
        n = int(np.minimum(_N_CHECK_ROWS, host_f32.shape[0]))
        xin = host_f32[:n].reshape(n, -1)
        got = got[:n].reshape(n, -1)
        if name == "softmax":
            m = xin.max(axis=-1, keepdims=True)
            e = np.exp(xin - m)
            ref = e / e.sum(axis=-1, keepdims=True)
            atol, rtol, frac_ok = 3e-3, 6e-2, 5e-3
        elif name == "logsoftmax":
            m = xin.max(axis=-1, keepdims=True)
            shifted = xin - m
            ref = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            atol, rtol, frac_ok = 5e-2, 6e-2, 5e-3
        elif name == "layer_norm":
            mean = xin.mean(axis=-1, keepdims=True)
            var = xin.var(axis=-1, keepdims=True)
            norm = (xin - mean) / np.sqrt(var + _LAYER_NORM_EPS)
            ref = aux.f32_vals["gamma"] * norm + aux.f32_vals["beta"]
            atol, rtol, frac_ok = 4e-2, 6e-2, 5e-3
        elif name == "group_norm":
            g = aux.num_groups
            cols = xin.shape[-1]
            xg = xin.reshape(n, g, cols // g)
            mean = xg.mean(axis=-1, keepdims=True)
            var = xg.var(axis=-1, keepdims=True)
            norm = ((xg - mean) / np.sqrt(var + _GROUP_NORM_EPS)).reshape(
                n, cols
            )
            ref = aux.f32_vals["gamma"] * norm + aux.f32_vals["beta"]
            atol, rtol, frac_ok = 5e-2, 6e-2, 5e-3
        else:  # rms_norm (Llama-style)
            ms = (xin**2).mean(axis=-1, keepdims=True)
            ref = (xin / np.sqrt(ms + _RMS_NORM_EPS)) * aux.f32_vals["weight"]
            atol, rtol, frac_ok = 4e-2, 6e-2, 5e-3

    abs_err = np.abs(got - ref)
    tol = atol + rtol * np.abs(ref)
    n_bad = int(np.sum(abs_err > tol))
    frac_bad = n_bad / got.size
    ok = frac_bad <= frac_ok
    return (
        ok,
        (
            f"max_abs_err={abs_err.max():.4g} over_tol={n_bad}/{got.size} "
            f"({frac_bad:.2%})"
        ),
    )


def time_execute(
    model: Model,
    buf: Buffer,
    device: Device,
    warmup: int,
    samples: int,
    iters_per_sample: int,
) -> list[float]:
    """ms-per-execute per sample; only the execute loop + trailing sync timed."""
    for _ in range(warmup):
        model.execute(buf)
    device.synchronize()

    ms_per_run: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(iters_per_sample):
            model.execute(buf)
        device.synchronize()
        t1 = time.perf_counter()
        ms_per_run.append((t1 - t0) * 1e3 / iters_per_sample)
    return ms_per_run


def _shape_label(shape: tuple[int, ...], axis: int, name: str) -> str:
    extra = ""
    if name == "group_norm":
        extra = f" g={_group_norm_num_groups(shape[-1])}"
    return f"{shape}@ax{axis}{extra}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iters-per-sample", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--reductions",
        type=str,
        default="",
        help="comma list to restrict reductions (default all)",
    )
    parser.add_argument(
        "--grid",
        choices=["inner", "noninner", "both"],
        default="both",
    )
    parser.add_argument(
        "--cols",
        type=str,
        default="",
        help="comma list to restrict inner cols (default all)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="skip timing; only build + correctness (fast validation pass)",
    )
    args = parser.parse_args()

    wanted = (
        [r.strip() for r in args.reductions.split(",") if r.strip()]
        if args.reductions
        else _ALL
    )
    cols = (
        [int(c) for c in args.cols.split(",") if c.strip()]
        if args.cols
        else _INNER_COLS
    )

    if args.device == "gpu":
        device: Device = Accelerator(0)
        device_ref = DeviceRef.GPU()
        session = InferenceSession(
            devices=[device], num_threads=args.num_threads
        )
    else:
        device_ref = DeviceRef.CPU()
        # Session BEFORE CPU(): the runtime keeps one global CPUDevice whose
        # thread options are fixed at first creation, so the engine's strict
        # creation must win before a default CPU() materializes it.
        session = InferenceSession(num_threads=args.num_threads)
        device = CPU()

    print(
        f"device={args.device}  warmup={args.warmup}  samples={args.samples} "
        f" iters/sample={args.iters_per_sample} "
        f" num_threads={args.num_threads}  check_only={args.check_only} "
        " branch-tip only (Stage A, no A/B)\n"
    )

    # (name, shape, axis) -> (ok, detail, med_or_min_ms, span_or_stab)
    results: dict[
        tuple[str, tuple[int, ...], int], tuple[bool, str, float, float]
    ] = {}

    # Build the work list: inner grid for all wanted; non-inner grid for pure.
    inner_shapes: list[tuple[int, ...]] = [
        (round(_TARGET_ELEMS / c), c) for c in cols
    ]
    work: list[tuple[str, tuple[int, ...], int]] = []
    if args.grid in ("inner", "both"):
        for name in wanted:
            for ishape in inner_shapes:
                # rms_norm_rope fuses only on an even last dim (rotate-half).
                if name == "rms_norm_rope" and ishape[-1] % 2 != 0:
                    continue
                work.append((name, ishape, -1))
    if args.grid in ("noninner", "both"):
        for name in wanted:
            if name in _PURE:
                for ax in _NONINNER_AXES:
                    work.append((name, _NONINNER_SHAPE, ax))

    for name, shape, axis in work:
        label = _shape_label(shape, axis, name)
        try:
            aux = make_aux(name, shape[-1])
            # group_norm needs rank>=3: normalize [rows, channels=cols, spatial=1].
            build_shape = (*shape, 1) if name == "group_norm" else shape
            graph = build_graph(name, build_shape, axis, device_ref, aux)
            model = session.load(graph)
            buf, host_f32 = make_input(name, build_shape, device)

            ok, detail = check_correctness(
                name, model, buf, host_f32, axis, aux, device
            )
            if args.check_only:
                results[(name, shape, axis)] = (ok, detail, float("nan"), 0.0)
                print(
                    f"  {name:<20} {label:<26} "
                    f"{'PASS' if ok else 'FAIL':<4} {detail}"
                )
                del host_f32, buf, model, graph
                continue

            ms = time_execute(
                model,
                buf,
                device,
                warmup=args.warmup,
                samples=args.samples,
                iters_per_sample=args.iters_per_sample,
            )
            med = statistics.median(ms)
            mn = min(ms)
            if args.device == "gpu":
                span = (max(ms) - mn) / med if med else 0.0
                perf, spread = med, span
                perf_tag = f"median={med:.4f}ms span={span:.1%}"
            else:
                perf, spread = mn, 0.0
                perf_tag = (
                    f"min={mn:.4f}ms median={med:.4f}ms max={max(ms):.4f}ms"
                )
            results[(name, shape, axis)] = (ok, detail, perf, spread)
            elems = int(np.prod(shape))
            traffic = elems * 2 * (2 if name in _FULL_OUTPUT else 1)
            gbps = traffic / (perf * 1e-3) / 1e9
            print(
                f"  {name:<20} {label:<26} "
                f"{perf_tag}  (~{gbps:.0f} GB/s)  "
                f"{'OK' if ok else 'CORRECTNESS-FAIL'}  [{detail}]"
            )
            del host_f32, buf, model, graph
        except Exception as exc:  # one bad combo must not kill the sweep
            msg = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
            results[(name, shape, axis)] = (
                False,
                f"BUILD/RUN-ERROR {msg}",
                float("nan"),
                0.0,
            )
            print(f"  {name:<20} {label:<26} ERROR  {msg}")
            traceback.print_exc(limit=1)

    # --- compact summary ---
    print(f"\n=== summary (branch tip, {args.device}) ===")
    for name in wanted:
        rows = [(s, ax) for (n, s, ax) in results if n == name]
        if not rows:
            continue
        print(f"\n{name}:")
        for shape, axis in sorted(
            rows,
            key=lambda sa: (sa[1], sa[0][-1] if sa[1] == -1 else sa[0][sa[1]]),
        ):
            ok, detail, perf, spread = results[(name, shape, axis)]
            tag = "PASS" if ok else "FAIL"
            if np.isnan(perf):
                perf_str = detail if not ok else "check-only"
            elif args.device == "gpu":
                perf_str = f"{perf:8.4f}ms span={spread:5.1%}"
            else:
                perf_str = f"min={perf:8.4f}ms"
            print(
                f"  {_shape_label(shape, axis, name):<26} {tag:<4} {perf_str}"
            )


if __name__ == "__main__":
    main()
