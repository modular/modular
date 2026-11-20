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
"""Fused RMSNorm + group-scaled dynamic FP8 quantize against an f64 reference.

`rms_norm_quantize_dynamic_scaled_fp8` must produce a blockwise-FP8
activation with `group_size_or_per_token=128` scales laid out
`[cols // 128, rows_padded]`. Each group scale must be the group's amax over
the FP8 max (clamped to the upper bound) within 1%, every dequantized element
must be within FP8 rounding of the f64 rms_norm, and the whole activation must
track it (cosine >= 0.999). Padded scale columns must stay untouched. A canary
run against a reference with a different weight offset checks the comparison
can fail. The graph test `test_rms_norm_quantize_fp8_grouped.py` compares the
fused op against `ops.rms_norm` followed by the quantize.
"""

from std.math import sqrt
from std.random import rand, seed
from std.utils.numerics import max_finite

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import rms_norm_quantize_dynamic_scaled_fp8

comptime IN_DTYPE = DType.bfloat16
comptime OUT_DTYPE = DType.float8_e4m3fn
comptime SCALE_DTYPE = DType.float32
comptime GROUP = 128
comptime SCALE_UB = Float32(1200.0)
comptime SENTINEL = Float32(-7.0)


struct Diff(ImplicitlyCopyable, Movable):
    var bad: Int
    var scale_rel_max: Float64
    var cos: Float64

    def __init__(out self):
        self.bad = 0
        self.scale_rel_max = 0.0
        self.cos = 0.0


def launch_fused[
    in_dtype: DType,
    out_dtype: DType,
    scale_dtype: DType,
    cols: Int,
    multiply_before_cast: Bool,
](
    ctx: DeviceContext,
    x_tt: TileTensor[in_dtype, ...],
    q_tt: TileTensor[mut=True, out_dtype, ...],
    s_tt: TileTensor[mut=True, scale_dtype, ...],
    gamma_tt: TileTensor[in_dtype, ...],
    rows: Int,
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
) raises:
    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var x_tt} -> SIMD[in_dtype, width]:
        return x_tt.load[width=width, alignment=alignment]((row, col))

    rms_norm_quantize_dynamic_scaled_fp8[
        group_size=GROUP,
        num_cols=cols,
        multiply_before_cast=multiply_before_cast,
    ](
        input_fn,
        q_tt,
        s_tt,
        gamma_tt,
        epsilon,
        weight_offset,
        SCALE_UB,
        ctx,
        rows,
    )


def run_case[
    in_dtype: DType,
    out_dtype: DType,
    scale_dtype: DType,
    cols: Int,
    multiply_before_cast: Bool,
](
    ctx: DeviceContext,
    rows: Int,
    fused_weight_offset: Float32,
    ref_weight_offset: Float32,
) raises -> Diff:
    comptime groups = cols // GROUP
    # f32 scales pad rows to a 16-byte multiple, as the graph op does.
    var rows_pad = (rows + 3) // 4 * 4
    var n = rows * cols

    var x_host = ctx.enqueue_create_host_buffer[in_dtype](n)
    var gamma_host = ctx.enqueue_create_host_buffer[in_dtype](cols)
    var x32 = ctx.enqueue_create_host_buffer[.float32](n)
    rand[.float32](x32.unsafe_ptr(), n, min=-3.0, max=3.0)
    for r in range(rows):
        # Per-row gain so every row's scales differ; a per-group outlier so
        # group scales differ within a row.
        var gain = 1.0 + 0.37 * Float32(r % 7)
        for c in range(cols):
            var v = x32[r * cols + c] * gain
            if c % GROUP == (r % GROUP):
                v *= 9.0
            x_host[r * cols + c] = v.cast[in_dtype]()
    for c in range(cols):
        var g = 0.6 + 0.9 * Float32(c % 13) / 13.0
        if c % 5 == 0:
            g = -g
        gamma_host[c] = g.cast[in_dtype]()

    comptime eps_f32 = Float32(1e-6)
    var eps = eps_f32.cast[in_dtype]()

    var x_buf = ctx.enqueue_create_buffer[in_dtype](n)
    var gamma_buf = ctx.enqueue_create_buffer[in_dtype](cols)
    var q_buf = ctx.enqueue_create_buffer[out_dtype](n)
    var s_buf = ctx.enqueue_create_buffer[scale_dtype](groups * rows_pad)
    ctx.enqueue_copy(x_buf, x_host)
    ctx.enqueue_copy(gamma_buf, gamma_host)
    var sentinel = SENTINEL.cast[scale_dtype]()
    s_buf.enqueue_fill(sentinel)

    var shape = Coord(rows, cols)
    launch_fused[in_dtype, out_dtype, scale_dtype, cols, multiply_before_cast](
        ctx,
        TileTensor(x_buf, row_major(shape)),
        TileTensor(q_buf, row_major(shape)),
        TileTensor(s_buf, row_major(Coord(groups, rows_pad))),
        TileTensor(gamma_buf, row_major(Coord(cols))),
        rows,
        eps,
        fused_weight_offset.cast[in_dtype](),
    )

    var q_host = ctx.enqueue_create_host_buffer[out_dtype](n)
    var s_host = ctx.enqueue_create_host_buffer[scale_dtype](groups * rows_pad)
    ctx.enqueue_copy(q_host, q_buf)
    ctx.enqueue_copy(s_host, s_buf)
    ctx.synchronize()

    var d = Diff()
    var dot: Float64 = 0.0
    var ref_sq: Float64 = 0.0
    var deq_sq: Float64 = 0.0
    var eps64 = Float64(eps.cast[.float32]())
    var off64 = Float64(ref_weight_offset)
    var fp8_max = Float64(max_finite[out_dtype]().cast[.float32]())
    var refs = List[Float64](length=GROUP, fill=0.0)
    for r in range(rows):
        var ssq: Float64 = 0.0
        for c in range(cols):
            var xv = Float64(x_host[r * cols + c].cast[.float32]())
            ssq += xv * xv
        var inv_rms = 1.0 / sqrt(ssq / Float64(cols) + eps64)
        for g in range(groups):
            var amax: Float64 = 0.0
            for k in range(GROUP):
                var c = g * GROUP + k
                var xv = Float64(x_host[r * cols + c].cast[.float32]())
                var gv = Float64(gamma_host[c].cast[.float32]()) + off64
                refs[k] = xv * inv_rms * gv
                amax = max(amax, abs(refs[k]))
            var s = Float64(s_host[g * rows_pad + r])
            if s == Float64(SENTINEL):
                raise Error("a scale for a real row was never written")
            var s_ref = min(amax, Float64(SCALE_UB)) / fp8_max
            d.scale_rel_max = max(
                d.scale_rel_max, abs(s - s_ref) / max(s_ref, 1e-30)
            )
            for k in range(GROUP):
                var deq = (
                    Float64(q_host[r * cols + g * GROUP + k].cast[.float32]())
                    * s
                )
                dot += refs[k] * deq
                ref_sq += refs[k] * refs[k]
                deq_sq += deq * deq
                # FP8 keeps 3 mantissa bits (2^-4 relative rounding), plus
                # the bf16 and scale rounding; the absolute term covers
                # values that land on FP8 subnormals.
                if abs(deq - refs[k]) > 0.1 * abs(refs[k]) + s / 256.0:
                    d.bad += 1
    # Padded scale columns are never written.
    for g in range(groups):
        for p in range(rows, rows_pad):
            if s_host[g * rows_pad + p] != sentinel:
                raise Error("a padded scale column was written")
    d.cos = dot / max(sqrt(ref_sq * deq_sq), 1e-300)

    # Keep the buffers alive past the launch that reads them by pointer.
    _ = x_buf
    _ = gamma_buf
    _ = q_buf
    _ = s_buf
    return d


def check_case[
    cols: Int, multiply_before_cast: Bool
](ctx: DeviceContext, rows: Int) raises:
    var d = run_case[
        IN_DTYPE, OUT_DTYPE, SCALE_DTYPE, cols, multiply_before_cast
    ](ctx, rows, 0.0, 0.0)
    var n = rows * cols
    print(
        t"rows={rows} cols={cols} mbc={multiply_before_cast} | bad={d.bad}/{n}"
        t" scale_rel_max={d.scale_rel_max} cos={d.cos}"
    )
    if d.cos < 0.999:
        raise Error("fused kernel is off vs the f64 reference")
    if d.bad > 0:
        raise Error("fused FP8 values are off the f64 reference")
    # bf16 rounding of one normalized value can move a group max by an ulp.
    if d.scale_rel_max > 1e-2:
        raise Error("fused group scales diverge from the f64 reference")


def check_canary(ctx: DeviceContext) raises:
    # A reference with a different weight offset must fail the checks, or
    # they prove nothing.
    var d = run_case[IN_DTYPE, OUT_DTYPE, SCALE_DTYPE, 2048, False](
        ctx, 5, 1.0, 0.0
    )
    print(t"canary (weight_offset 1.0 vs 0.0): bad={d.bad}/{5 * 2048}")
    if d.bad * 2 < 5 * 2048 or d.cos >= 0.999:
        raise Error("canary did not trip: the comparison is insensitive")


def main() raises:
    seed(42)
    var ctx = DeviceContext()
    comptime for mbc in [False, True]:
        check_case[2048, mbc](ctx, 1)
        check_case[2048, mbc](ctx, 5)
        check_case[2048, mbc](ctx, 16)
        check_case[2048, mbc](ctx, 64)
        check_case[6144, mbc](ctx, 7)
        check_case[6144, mbc](ctx, 64)
        # 256 columns at group 128: 32 threads at lane width 8 is one NVIDIA
        # warp but half a CDNA wavefront, so the lane width must drop there.
        check_case[256, mbc](ctx, 5)
        check_case[256, mbc](ctx, 64)
    check_canary(ctx)
    print("PASS")
