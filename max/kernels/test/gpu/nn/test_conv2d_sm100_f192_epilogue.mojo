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
"""The fused epilogues of the SM100 192 -> 192 channel conv2d route.

Each case runs the general SM100 kernel (`conv2d_fprop`, KRSC filter) and the
192 -> 192 route on the same operands and requires the two outputs to be equal
bit for bit. Operands are small integers drawn from a hash, so every fp32
partial sum is exact in any accumulation order and the only way for the two
kernels to differ is the epilogue itself: the dtype the accumulator is cast
to, the order of the residual add, the coordinates handed to the closures.

Cases (campaign case 8 geometry, and a two-image shape with boxes that cross
images):
- a compute epilogue, `relu(acc + bias[channel])`;
- a storing epilogue with the same math;
- a residual, `acc + beta * source`, with beta 1 and 0.5;
- a residual followed by a storing `relu` epilogue;
- the route as `dispatch_sm100_conv2d` reaches it, with an RSCF filter;
- fp16 through `conv_gpu`, with NHWC-coordinate closures, against cuDNN's
  convolution followed by the same arithmetic in fp16 on the host.
"""

from std.sys import has_accelerator
from std.testing import assert_equal

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.conv.gpu.nvidia.sm100.conv2d import (
    conv2d_fprop,
    conv2d_fprop_with_residual,
)
from nn.conv.gpu.nvidia.sm100.conv2d_f192_fprop import conv2d_f192_fprop
from nn.conv.gpu.nvidia.sm100.conv_config import (
    Conv2dConfig,
    Conv2dProblemShape,
)
from nn.conv.gpu.nvidia.sm100.dispatch import dispatch_sm100_conv2d
from nn.conv.conv import conv_cudnn, conv_gpu

comptime dt = DType.bfloat16
comptime CIN = 192
comptime COUT = 192

comptime COMPUTE = 0
comptime STORE = 1
comptime RESIDUAL = 2
comptime RESIDUAL_STORE = 3


def _hash_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 9) % 5 - 2)


def _fill(buf: HostBuffer[dt], n: Int, salt: Int):
    for i in range(n):
        buf[i] = _hash_value(i, salt).cast[dt]()


def _count_mismatches(got: HostBuffer[dt], want: HostBuffer[dt], n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if got[i].cast[.float32]() != want[i].cast[.float32]():
            bad += 1
    return bad


def run_case[
    mode: Int, rscf: Bool, via_dispatch: Bool
](
    ctx: DeviceContext,
    batch: Int,
    h: Int,
    w: Int,
    beta: Float32,
    name: StringLiteral,
) raises:
    """Runs one fused-epilogue problem on the general kernel and the route.

    Parameters:
        mode: Which epilogue (`COMPUTE`, `STORE`, `RESIDUAL`,
            `RESIDUAL_STORE`).
        rscf: Hand the route an RSCF filter instead of KRSC.
        via_dispatch: Enter through `dispatch_sm100_conv2d`; the filter is
            RSCF.

    Args:
        ctx: Device context.
        batch: Batch size.
        h: Input height (output height with pad 1).
        w: Input width.
        beta: Residual scale factor.
        name: Label printed with the result.
    """
    var m_total = batch * h * w
    var n_act = m_total * CIN
    var n_filt = COUT * 9 * CIN
    var n_out = m_total * COUT

    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=h,
        in_width=w,
        in_channels=CIN,
        out_channels=COUT,
        filter_h=3,
        filter_w=3,
        pad_h=1,
        pad_w=1,
    )

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var krsc_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var rscf_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](COUT)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    _fill(act_host, n_act, batch * 31 + h)
    _fill(krsc_host, n_filt, batch * 31 + h + 1)
    _fill(bias_host, COUT, 5)
    _fill(src_host, n_out, 9)
    var krsc_view = TileTensor(krsc_host, row_major(COUT, 3, 3, CIN))
    var rscf_view = TileTensor(rscf_host, row_major(3, 3, CIN, COUT))
    for f in range(COUT):
        for r in range(3):
            for s in range(3):
                for c in range(CIN):
                    rscf_view[r, s, c, f] = krsc_view[f, r, s, c]

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var krsc_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var route_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](COUT)
    var src_dev = ctx.enqueue_create_buffer[dt](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var ref_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(krsc_dev, krsc_host)
    comptime if rscf or via_dispatch:
        ctx.enqueue_copy(route_dev, rscf_host)
    else:
        ctx.enqueue_copy(route_dev, krsc_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_copy(src_dev, src_host)
    # Poison both outputs: a storing epilogue that never ran must not match.
    out_dev.enqueue_fill(Scalar[dt](-7))
    ref_dev.enqueue_fill(Scalar[dt](-9))

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, CIN))
    var krsc_tt = TileTensor(krsc_dev, row_major(COUT, 3, 3, CIN))
    # The filter the route reads: RSCF through dispatch or `rscf`, else KRSC.
    var route_tt = TileTensor(
        route_dev,
        row_major(3, 3, CIN, COUT) if (rscf or via_dispatch) else row_major(
            COUT, 3, 3, CIN
        ),
    )
    var out_tt = TileTensor(out_dev, row_major(batch, h, w, COUT))
    var ref_tt = TileTensor(ref_dev, row_major(batch, h, w, COUT))
    var src_tt = TileTensor(src_dev, row_major(batch, h, w, COUT))
    var bias_tt = TileTensor(bias_dev, row_major(COUT))
    # The storing closures write through views of their own: a view that
    # borrowed the output would alias the tensor they are handed alongside.
    var out_p = out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var ref_p = ref_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var out_2d = TileTensor(out_p, row_major(m_total, COUT))
    var ref_2d = TileTensor(ref_p, row_major(m_total, COUT))

    comptime config = Conv2dConfig[dt, dt, dt].default_bf16_1sm[
        swizzle=TensorMapSwizzle.SWIZZLE_128B
    ]()

    @always_inline
    @__allow_legacy_any_origin_fields
    def bias_relu[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt} -> SIMD[
        _dtype, width
    ]:
        comptime assert idx.rank == 2, "the epilogue contract is rank 2"
        var b = bias_tt.load[width=width]((idx[1],)).cast[_dtype]()
        return max(val + b, SIMD[_dtype, width](0))

    @always_inline
    @__allow_legacy_any_origin_fields
    def store_relu_ref[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var ref_2d}:
        comptime assert idx.rank == 2, "the epilogue contract is rank 2"
        var b = bias_tt.load[width=width]((idx[1],)).cast[_dtype]()
        ref_2d.store[width=Int(width), alignment=alignment](
            idx, max(val + b, SIMD[_dtype, width](0)).cast[dt]()
        )

    @always_inline
    @__allow_legacy_any_origin_fields
    def store_relu_out[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var out_2d}:
        comptime assert idx.rank == 2, "the epilogue contract is rank 2"
        var b = bias_tt.load[width=width]((idx[1],)).cast[_dtype]()
        out_2d.store[width=Int(width), alignment=alignment](
            idx, max(val + b, SIMD[_dtype, width](0)).cast[dt]()
        )

    @always_inline
    def relu_ref[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var ref_2d}:
        comptime assert idx.rank == 2, "the epilogue contract is rank 2"
        ref_2d.store[width=Int(width), alignment=alignment](
            idx, max(val, SIMD[_dtype, width](0)).cast[dt]()
        )

    @always_inline
    def relu_out[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_2d}:
        comptime assert idx.rank == 2, "the epilogue contract is rank 2"
        out_2d.store[width=Int(width), alignment=alignment](
            idx, max(val, SIMD[_dtype, width](0)).cast[dt]()
        )

    # `conv2d_f192_fprop` and `conv_gpu` take the residual source as an
    # any-origin pointer.
    var src_ptr = src_dev.unsafe_ptr().as_unsafe_any_origin()

    # ----- the general kernel -----
    comptime match mode:
        case == COMPUTE:
            conv2d_fprop[config=config](
                ref_tt, act_tt, krsc_tt, problem, bias_relu, ctx
            )
        case == STORE:
            conv2d_fprop[config=config](
                ref_tt, act_tt, krsc_tt, problem, store_relu_ref, ctx
            )
        case == RESIDUAL:
            conv2d_fprop_with_residual[config=config, has_residual=True](
                ref_tt, act_tt, krsc_tt, src_tt, beta, problem, ctx
            )
        case _:
            conv2d_fprop_with_residual[
                config=config, has_residual=True, has_epilogue_fn=True
            ](ref_tt, act_tt, krsc_tt, src_tt, beta, problem, ctx, relu_ref)

    # ----- the route -----
    comptime if via_dispatch:
        comptime match mode:
            case == COMPUTE:
                dispatch_sm100_conv2d[
                    filter_is_fcrs=False, has_compute_fn=True
                ](
                    act_tt,
                    route_tt,
                    out_tt,
                    Coord(1, 1),
                    ctx,
                    compute_fn=bias_relu,
                )
            case == STORE:
                dispatch_sm100_conv2d[
                    filter_is_fcrs=False, has_epilogue_fn=True
                ](
                    act_tt,
                    route_tt,
                    out_tt,
                    Coord(1, 1),
                    ctx,
                    epilogue_fn=store_relu_out,
                )
            case == RESIDUAL:
                dispatch_sm100_conv2d[filter_is_fcrs=False, has_residual=True](
                    act_tt,
                    route_tt,
                    out_tt,
                    Coord(1, 1),
                    ctx,
                    src_ptr,
                    beta,
                )
            case _:
                dispatch_sm100_conv2d[
                    filter_is_fcrs=False,
                    has_residual=True,
                    has_epilogue_fn=True,
                ](
                    act_tt,
                    route_tt,
                    out_tt,
                    Coord(1, 1),
                    ctx,
                    src_ptr,
                    beta,
                    epilogue_fn=relu_out,
                )
    else:
        comptime match mode:
            case == COMPUTE:
                conv2d_f192_fprop[dt, rscf_filter=rscf, has_compute_fn=True](
                    out_tt,
                    act_tt,
                    route_tt,
                    problem,
                    ctx,
                    compute_fn=bias_relu,
                )
            case == STORE:
                conv2d_f192_fprop[dt, rscf_filter=rscf, has_epilogue_fn=True](
                    out_tt,
                    act_tt,
                    route_tt,
                    problem,
                    ctx,
                    epilogue_fn=store_relu_out,
                )
            case == RESIDUAL:
                conv2d_f192_fprop[dt, rscf_filter=rscf, has_residual=True](
                    out_tt,
                    act_tt,
                    route_tt,
                    problem,
                    ctx,
                    src_ptr,
                    beta,
                )
            case _:
                conv2d_f192_fprop[
                    dt,
                    rscf_filter=rscf,
                    has_residual=True,
                    has_epilogue_fn=True,
                ](
                    out_tt,
                    act_tt,
                    route_tt,
                    problem,
                    ctx,
                    src_ptr,
                    beta,
                    epilogue_fn=relu_out,
                )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()
    var bad = _count_mismatches(out_host, ref_host, n_out)
    comptime filter_layout = "rscf" if rscf else "krsc"
    comptime entry = "dispatch" if via_dispatch else "kernel"
    print(
        t"   {name} mode {mode} {filter_layout} {entry} mismatches: {bad} of"
        t" {n_out}"
    )
    assert_equal(bad, 0, String(t"epilogue mismatch for {name}"))

    _ = act_dev^
    _ = krsc_dev^
    _ = route_dev^
    _ = bias_dev^
    _ = src_dev^
    _ = out_dev^
    _ = ref_dev^


def run_fp16_case[
    mode: Int
](
    ctx: DeviceContext,
    batch: Int,
    h: Int,
    w: Int,
    beta: Float32,
    name: StringLiteral,
) raises:
    """Runs fp16 through `conv_gpu` (RSCF filter, NHWC closures) against cuDNN.

    The reference is cuDNN's convolution into fp16 followed by the epilogue's
    arithmetic in fp16 on the host: round the accumulator to fp16, then every
    add, multiply and `max` rounds to fp16, which is what the production
    cuDNN path does and what the route's `round_first` mode reproduces.

    Parameters:
        mode: `COMPUTE`, `STORE`, `RESIDUAL` or `RESIDUAL_STORE`.

    Args:
        ctx: Device context.
        batch: Batch size.
        h: Input height (output height with pad 1).
        w: Input width.
        beta: Residual scale factor (1 or 0.5, exact in fp16).
        name: Label printed with the result.
    """
    comptime f16 = DType.float16
    var m_total = batch * h * w
    var n_act = m_total * CIN
    var n_filt = COUT * 9 * CIN
    var n_out = m_total * COUT

    var act_host = ctx.enqueue_create_host_buffer[f16](n_act)
    var fcrs_host = ctx.enqueue_create_host_buffer[f16](n_filt)
    var rscf_host = ctx.enqueue_create_host_buffer[f16](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[f16](COUT)
    var src_host = ctx.enqueue_create_host_buffer[f16](n_out)
    var out_host = ctx.enqueue_create_host_buffer[f16](n_out)
    var plain_host = ctx.enqueue_create_host_buffer[f16](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _hash_value(i, batch * 31 + h).cast[f16]()
    for i in range(n_filt):
        fcrs_host[i] = _hash_value(i, batch * 31 + h + 1).cast[f16]()
    for i in range(COUT):
        bias_host[i] = _hash_value(i, 5).cast[f16]()
    for i in range(n_out):
        src_host[i] = _hash_value(i, 9).cast[f16]()
    var fcrs_view = TileTensor(fcrs_host, row_major(COUT, CIN, 3, 3))
    var rscf_view = TileTensor(rscf_host, row_major(3, 3, CIN, COUT))
    for f in range(COUT):
        for c in range(CIN):
            for r in range(3):
                for s in range(3):
                    rscf_view[r, s, c, f] = fcrs_view[f, c, r, s]

    var act_dev = ctx.enqueue_create_buffer[f16](n_act)
    var fcrs_dev = ctx.enqueue_create_buffer[f16](n_filt)
    var rscf_dev = ctx.enqueue_create_buffer[f16](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[f16](COUT)
    var src_dev = ctx.enqueue_create_buffer[f16](n_out)
    var out_dev = ctx.enqueue_create_buffer[f16](n_out)
    var plain_dev = ctx.enqueue_create_buffer[f16](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(fcrs_dev, fcrs_host)
    ctx.enqueue_copy(rscf_dev, rscf_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_copy(src_dev, src_host)
    out_dev.enqueue_fill(Scalar[f16](-7))

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, CIN))
    var fcrs_tt = TileTensor(fcrs_dev, row_major(COUT, CIN, 3, 3))
    var rscf_tt = TileTensor(rscf_dev, row_major(3, 3, CIN, COUT))
    var out_tt = TileTensor(out_dev, row_major(batch, h, w, COUT))
    var plain_tt = TileTensor(plain_dev, row_major(batch, h, w, COUT))
    var bias_tt = TileTensor(bias_dev, row_major(COUT))
    var out_4d = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        row_major(batch, h, w, COUT),
    )
    # `conv_gpu` takes its residual source as an any-origin pointer.
    var src_ptr = src_dev.unsafe_ptr().as_unsafe_any_origin()

    # The closures take the NHWC coordinate `conv_gpu`'s callers write.
    @always_inline
    @__allow_legacy_any_origin_fields
    def bias_relu[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt} -> SIMD[
        _dtype, width
    ]:
        comptime assert idx.rank == 4, "conv_gpu closures take NHWC"
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        return max(val + b, SIMD[_dtype, width](0))

    @always_inline
    @__allow_legacy_any_origin_fields
    def store_relu[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var out_4d}:
        comptime assert idx.rank == 4, "conv_gpu closures take NHWC"
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        out_4d.store[width=Int(width), alignment=alignment](
            idx, max(val + b, SIMD[_dtype, width](0)).cast[f16]()
        )

    @always_inline
    @__allow_legacy_any_origin_fields
    def relu_store[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_4d}:
        comptime assert idx.rank == 4, "conv_gpu closures take NHWC"
        out_4d.store[width=Int(width), alignment=alignment](
            idx, max(val, SIMD[_dtype, width](0)).cast[f16]()
        )

    comptime match mode:
        case == COMPUTE:
            conv_gpu[f16, f16, f16, has_compute_fn=True](
                act_tt,
                rscf_tt,
                out_tt,
                Coord(1, 1),
                Coord(1, 1),
                Coord(1, 1, 1, 1),
                1,
                ctx,
                compute_fn=bias_relu,
            )
        case == STORE:
            conv_gpu[f16, f16, f16, has_epilogue_fn=True](
                act_tt,
                rscf_tt,
                out_tt,
                Coord(1, 1),
                Coord(1, 1),
                Coord(1, 1, 1, 1),
                1,
                ctx,
                epilogue_fn=store_relu,
            )
        case == RESIDUAL:
            conv_gpu[f16, f16, f16, has_residual=True](
                act_tt,
                rscf_tt,
                out_tt,
                Coord(1, 1),
                Coord(1, 1),
                Coord(1, 1, 1, 1),
                1,
                ctx,
                src_ptr,
                beta,
            )
        case _:
            conv_gpu[f16, f16, f16, has_residual=True, has_epilogue_fn=True](
                act_tt,
                rscf_tt,
                out_tt,
                Coord(1, 1),
                Coord(1, 1),
                Coord(1, 1, 1, 1),
                1,
                ctx,
                src_ptr,
                beta,
                epilogue_fn=relu_store,
            )

    # cuDNN's plain convolution is the reference's starting point.
    conv_cudnn[f16, f16, f16](
        act_tt.as_imm(),
        fcrs_tt.as_imm(),
        plain_tt,
        Coord(1, 1),
        Coord(1, 1),
        Coord(1, 1),
        1,
        ctx,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(plain_host, plain_dev)
    ctx.synchronize()

    var bad = 0
    var zero = Float32(0)
    var plain_2d = TileTensor(plain_host, row_major(m_total, COUT))
    var src_2d = TileTensor(src_host, row_major(m_total, COUT))
    var out_2d = TileTensor(out_host, row_major(m_total, COUT))
    for pixel in range(m_total):
        for c in range(COUT):
            var x = plain_2d[pixel, c].cast[.float32]()
            var want: Float32
            comptime if mode == COMPUTE or mode == STORE:
                var t = (x + bias_host[c].cast[.float32]()).cast[f16]()
                want = max(t.cast[.float32](), zero)
            else:
                var sc = (beta * src_2d[pixel, c].cast[.float32]()).cast[f16]()
                var t = (x + sc.cast[.float32]()).cast[f16]()
                comptime if mode == RESIDUAL:
                    want = t.cast[.float32]()
                else:
                    want = max(t.cast[.float32](), zero)
            if out_2d[pixel, c].cast[.float32]() != want:
                bad += 1
    print(t"   {name} fp16 mode {mode} mismatches: {bad} of {n_out}")
    assert_equal(bad, 0, String(t"fp16 epilogue mismatch for {name}"))

    _ = act_dev^
    _ = fcrs_dev^
    _ = rscf_dev^
    _ = bias_dev^
    _ = src_dev^
    _ = out_dev^
    _ = plain_dev^


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    with DeviceContext() as ctx:
        print("--- campaign case 8, 120 x 208 ---")
        run_case[COMPUTE, False, False](ctx, 1, 120, 208, 0.0, "case8")
        run_case[COMPUTE, True, False](ctx, 1, 120, 208, 0.0, "case8")
        run_case[STORE, True, False](ctx, 1, 120, 208, 0.0, "case8")
        run_case[RESIDUAL, True, False](ctx, 1, 120, 208, 1.0, "case8")
        run_case[RESIDUAL, False, False](ctx, 1, 120, 208, 0.5, "case8")
        run_case[RESIDUAL_STORE, True, False](ctx, 1, 120, 208, 0.5, "case8")

        print("--- two images, boxes that cross them ---")
        run_case[COMPUTE, True, False](ctx, 2, 12, 64, 0.0, "batch2")
        run_case[RESIDUAL_STORE, False, False](ctx, 2, 12, 64, 1.0, "batch2")

        print("--- dispatch_sm100_conv2d, RSCF filter ---")
        run_case[COMPUTE, True, True](ctx, 1, 120, 208, 0.0, "case8")
        run_case[STORE, True, True](ctx, 1, 120, 208, 0.0, "case8")
        run_case[RESIDUAL, True, True](ctx, 1, 120, 208, 1.0, "case8")
        run_case[RESIDUAL_STORE, True, True](ctx, 1, 120, 208, 0.5, "case8")

        print("--- fp16 through conv_gpu, against cuDNN ---")
        run_fp16_case[COMPUTE](ctx, 1, 120, 208, 0.0, "case6")
        run_fp16_case[STORE](ctx, 1, 120, 208, 0.0, "case6")
        run_fp16_case[RESIDUAL](ctx, 1, 120, 208, 1.0, "case6")
        run_fp16_case[RESIDUAL_STORE](ctx, 1, 120, 208, 0.5, "case6")
        run_fp16_case[COMPUTE](ctx, 2, 12, 64, 0.0, "batch2")
        run_fp16_case[RESIDUAL_STORE](ctx, 2, 12, 64, 1.0, "batch2")
    print("ALL F192 EPILOGUE TESTS PASSED")
