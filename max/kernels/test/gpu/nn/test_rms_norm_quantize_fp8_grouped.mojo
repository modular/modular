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
"""Fused RMSNorm + group-scaled dynamic FP8 quantize vs the unfused pair.

`rms_norm_quantize_dynamic_scaled_fp8` must reproduce what the graph runs
today for a blockwise-FP8 activation: `rms_norm` (bf16 out) followed by
`quantize_dynamic_scaled_fp8` with `group_size_or_per_token=128`, scales
laid out `[cols // 128, rows_padded]`. The two paths only differ in the
sum-of-squares reduction order, so every FP8 byte must match to within one
FP8 ulp with at most a 0.1% mismatch rate, the group scales must agree to
within a bf16 ulp, and the dequantized activation of BOTH paths must track
an f64 reference (cosine >= 0.999). Padded scale columns must stay
untouched. A canary run with a different weight offset checks the
comparison can fail.
"""

from std.math import sqrt
from std.memory import bitcast
from std.random import rand, seed
from std.utils.coord import ComptimeInt

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from linalg.fp8_quantization import quantize_dynamic_scaled_fp8
from nn.normalization import rms_norm, rms_norm_quantize_dynamic_scaled_fp8

comptime IN_DTYPE = DType.bfloat16
comptime OUT_DTYPE = DType.float8_e4m3fn
comptime SCALE_DTYPE = DType.float32
comptime GROUP = 128
comptime SCALE_UB = Float32(1200.0)
comptime SENTINEL = Float32(-7.0)


struct Diff(ImplicitlyCopyable, Movable):
    var mismatches: Int
    var max_ulp: Int
    var scale_rel_max: Float64
    var cos_fused: Float64
    var cos_sep: Float64

    def __init__(out self):
        self.mismatches = 0
        self.max_ulp = 0
        self.scale_rel_max = 0.0
        self.cos_fused = 0.0
        self.cos_sep = 0.0


# One launch per helper: two value closures over one buffer cannot share a
# scope; dtypes stay parameters so a symbolic `SIMD[dtype, w]` closure fits.


def launch_rms_norm[
    in_dtype: DType, cols: Int, multiply_before_cast: Bool
](
    ctx: DeviceContext,
    x_tt: TileTensor[in_dtype, ...],
    normed_tt: TileTensor[mut=True, in_dtype, ...],
    gamma_tt: TileTensor[in_dtype, ...],
    shape: Coord,
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
) raises:
    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var x_tt} -> SIMD[in_dtype, width]:
        var idx = x_tt.layout(coords)
        return x_tt.raw_load[width=width, alignment=alignment](idx)

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[in_dtype, width]) {var normed_tt}:
        var idx = normed_tt.layout(coords)
        normed_tt.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[in_dtype, width]](val)
        )

    rms_norm[
        in_dtype, 2, target="gpu", multiply_before_cast=multiply_before_cast
    ](
        input_fn,
        output_fn,
        shape,
        ComptimeInt[cols](),
        gamma_tt,
        epsilon,
        weight_offset,
        ctx,
    )


def launch_quantize[
    in_dtype: DType, out_dtype: DType, scale_dtype: DType, cols: Int
](
    ctx: DeviceContext,
    normed_tt: TileTensor[in_dtype, ...],
    q_tt: TileTensor[mut=True, out_dtype, ...],
    s_tt: TileTensor[mut=True, scale_dtype, ...],
    rows: Int,
) raises:
    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var normed_tt} -> SIMD[in_dtype, width]:
        var idx = normed_tt.layout(Coord(row, col))
        return normed_tt.raw_load[width=width, alignment=alignment](idx)

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype, group_size_or_per_token=GROUP, num_cols=cols
    ](input_fn, q_tt, s_tt, SCALE_UB, ctx, rows)


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
        var idx = x_tt.layout(Coord(row, col))
        return x_tt.raw_load[width=width, alignment=alignment](idx)

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
    sep_weight_offset: Float32,
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

    var eps = Float32(1e-6).cast[in_dtype]()
    var fused_off = fused_weight_offset.cast[in_dtype]()
    var sep_off = sep_weight_offset.cast[in_dtype]()

    var x_buf = ctx.enqueue_create_buffer[in_dtype](n)
    var gamma_buf = ctx.enqueue_create_buffer[in_dtype](cols)
    var normed_buf = ctx.enqueue_create_buffer[in_dtype](n)
    var q_sep_buf = ctx.enqueue_create_buffer[out_dtype](n)
    var q_fus_buf = ctx.enqueue_create_buffer[out_dtype](n)
    var s_sep_buf = ctx.enqueue_create_buffer[scale_dtype](groups * rows_pad)
    var s_fus_buf = ctx.enqueue_create_buffer[scale_dtype](groups * rows_pad)
    ctx.enqueue_copy(x_buf, x_host)
    ctx.enqueue_copy(gamma_buf, gamma_host)
    var sentinel = SENTINEL.cast[scale_dtype]()
    s_sep_buf.enqueue_fill(sentinel)
    s_fus_buf.enqueue_fill(sentinel)

    var shape = Coord(rows, cols)
    var scale_shape = Coord(groups, rows_pad)

    # Unfused pair, exactly as `mo.reduce.rms_norm` + the group quantize run.
    launch_rms_norm[in_dtype, cols, multiply_before_cast](
        ctx,
        TileTensor(x_buf, row_major(shape)),
        TileTensor(normed_buf, row_major(shape)),
        TileTensor(gamma_buf, row_major(Coord(cols))),
        shape,
        eps,
        sep_off,
    )
    launch_quantize[in_dtype, out_dtype, scale_dtype, cols](
        ctx,
        TileTensor(normed_buf, row_major(shape)),
        TileTensor(q_sep_buf, row_major(shape)),
        TileTensor(s_sep_buf, row_major(scale_shape)),
        rows,
    )

    # Fused kernel under test.
    launch_fused[in_dtype, out_dtype, scale_dtype, cols, multiply_before_cast](
        ctx,
        TileTensor(x_buf, row_major(shape)),
        TileTensor(q_fus_buf, row_major(shape)),
        TileTensor(s_fus_buf, row_major(scale_shape)),
        TileTensor(gamma_buf, row_major(Coord(cols))),
        rows,
        eps,
        fused_off,
    )

    var q_sep_host = ctx.enqueue_create_host_buffer[out_dtype](n)
    var q_fus_host = ctx.enqueue_create_host_buffer[out_dtype](n)
    var s_sep_host = ctx.enqueue_create_host_buffer[scale_dtype](
        groups * rows_pad
    )
    var s_fus_host = ctx.enqueue_create_host_buffer[scale_dtype](
        groups * rows_pad
    )
    ctx.enqueue_copy(q_sep_host, q_sep_buf)
    ctx.enqueue_copy(q_fus_host, q_fus_buf)
    ctx.enqueue_copy(s_sep_host, s_sep_buf)
    ctx.enqueue_copy(s_fus_host, s_fus_buf)
    ctx.synchronize()

    var d = Diff()
    var dot_fus: Float64 = 0.0
    var dot_sep: Float64 = 0.0
    var ref_sq: Float64 = 0.0
    var fus_sq: Float64 = 0.0
    var sep_sq: Float64 = 0.0
    var eps64 = Float64(eps.cast[.float32]())
    var off64 = Float64(fused_off.cast[.float32]())
    for r in range(rows):
        var ssq: Float64 = 0.0
        for c in range(cols):
            var xv = Float64(x_host[r * cols + c].cast[.float32]())
            ssq += xv * xv
        var inv_rms = 1.0 / sqrt(ssq / Float64(cols) + eps64)
        for g in range(groups):
            var s_sep = Float64(s_sep_host[g * rows_pad + r])
            var s_fus = Float64(s_fus_host[g * rows_pad + r])
            if s_sep == Float64(SENTINEL) or s_fus == Float64(SENTINEL):
                raise Error("a scale for a real row was never written")
            d.scale_rel_max = max(
                d.scale_rel_max, abs(s_sep - s_fus) / max(abs(s_sep), 1e-30)
            )
            for c in range(g * GROUP, (g + 1) * GROUP):
                var i = r * cols + c
                var xv = Float64(x_host[i].cast[.float32]())
                var gv = Float64(gamma_host[c].cast[.float32]()) + off64
                var ref_val = xv * inv_rms * gv
                var q_sep = q_sep_host[i]
                var q_fus = q_fus_host[i]
                var deq_sep = Float64(q_sep.cast[.float32]()) * s_sep
                var deq_fus = Float64(q_fus.cast[.float32]()) * s_fus
                dot_fus += ref_val * deq_fus
                dot_sep += ref_val * deq_sep
                ref_sq += ref_val * ref_val
                fus_sq += deq_fus * deq_fus
                sep_sq += deq_sep * deq_sep
                var b_sep = Int(bitcast[.uint8](q_sep))
                var b_fus = Int(bitcast[.uint8](q_fus))
                if b_sep != b_fus:
                    d.mismatches += 1
                    # Same sign: the e4m3 byte order is monotonic, so the
                    # byte distance is the ulp distance.
                    var ulp = (
                        abs(b_sep - b_fus) if (b_sep >> 7)
                        == (b_fus >> 7) else 999
                    )
                    d.max_ulp = max(d.max_ulp, ulp)
        # Padded scale columns are never written by either path.
        for g in range(groups):
            for p in range(rows, rows_pad):
                if (
                    s_sep_host[g * rows_pad + p] != sentinel
                    or s_fus_host[g * rows_pad + p] != sentinel
                ):
                    raise Error("a padded scale column was written")
    d.cos_fused = dot_fus / max(sqrt(ref_sq * fus_sq), 1e-300)
    d.cos_sep = dot_sep / max(sqrt(ref_sq * sep_sq), 1e-300)

    # Keep the buffers alive past the launches that read them by pointer.
    _ = x_buf
    _ = gamma_buf
    _ = normed_buf
    _ = q_sep_buf
    _ = q_fus_buf
    _ = s_sep_buf
    _ = s_fus_buf
    return d


def check_case[
    cols: Int, multiply_before_cast: Bool
](ctx: DeviceContext, rows: Int) raises:
    var d = run_case[
        IN_DTYPE, OUT_DTYPE, SCALE_DTYPE, cols, multiply_before_cast
    ](ctx, rows, 0.0, 0.0)
    var n = rows * cols
    print(
        "rows=",
        rows,
        " cols=",
        cols,
        " mbc=",
        multiply_before_cast,
        " | fp8 mismatches=",
        d.mismatches,
        "/",
        n,
        " max_ulp=",
        d.max_ulp,
        " scale_rel_max=",
        d.scale_rel_max,
        " cos_fused=",
        d.cos_fused,
        " cos_sep=",
        d.cos_sep,
        sep="",
    )
    if d.cos_sep < 0.999:
        raise Error("unfused reference pair is off vs the f64 reference")
    if d.cos_fused < 0.999:
        raise Error("fused kernel is off vs the f64 reference")
    if d.mismatches * 1000 > n:
        raise Error("fused FP8 bytes mismatch the unfused pair on > 0.1%")
    if d.max_ulp > 1:
        raise Error("fused FP8 byte differs from the unfused pair by > 1 ulp")
    # bf16 rounding of one normalized value can move a group max by an ulp.
    if d.scale_rel_max > 1e-2:
        raise Error("fused group scales diverge from the unfused pair")


def check_canary(ctx: DeviceContext) raises:
    # Different weight offsets must show up as FP8 mismatches, or the byte
    # comparison above proves nothing (only the pairwise compare is expected).
    var d = run_case[IN_DTYPE, OUT_DTYPE, SCALE_DTYPE, 2048, False](
        ctx, 5, 1.0, 0.0
    )
    print(
        "canary (weight_offset 1.0 vs 0.0): mismatches=",
        d.mismatches,
        "/",
        5 * 2048,
        sep="",
    )
    if d.mismatches * 2 < 5 * 2048:
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
