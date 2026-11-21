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
"""MI355X: `conv_gpu` routes the two measured geometries to the wave-specialized
kernels, and sends everything next to them down its old path.

The reference is MIOpen's convolution followed by the epilogue arithmetic on
the host, rounding to bf16 after the convolution and after each operation (the
order the kernels use): the bare conv, a storing epilogue that adds a
per-channel bias with and without a residual, a coordinate-dependent
`compute_fn` alone, and a relu `compute_fn` after a residual.
The tolerance is two bf16 ulps, the arms sum in different orders.

`ring_conv2d_arm` is the admission rule for the geometry; each case states the
arm it expects and asserts it at compile time. The neighbors of the admitted
geometries (fp16, another size, a batch of two, 64 channels, an RSCF filter)
are known to be declined by it; stride 2 and pad 0 keep the admitted geometry
and are declined at run time. Every case's output is checked either way.
"""

from std.sys import has_accelerator
from std.testing import assert_equal
from std.random import rand, seed

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.conv.conv import conv_gpu, conv_miopen
from nn.conv.gpu.amd.dispatch import (
    RING_FCRS,
    RING_NONE,
    RING_STAGED,
    ring_conv2d_arm,
)

comptime PLAIN = 0
comptime STORE_BIAS = 1
comptime RELU = 2
comptime COORD = 3
comptime STORE_NO_RESIDUAL = 4

comptime _BETA = Float32(0.75)


def coord_shift(h: Int, w: Int, c: Int) -> Int:
    """A small integer that depends on all three output coordinates, so a
    `compute_fn` fed the wrong coordinate shifts the wrong value."""
    return (h * 5 + w * 3 + c) % 7


def _exec[
    dt: DType,
    //,
    *,
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    F: Int,
    mode: Int,
    fcrs: Bool,
    stride: Int,
    pad: Int,
](
    ctx: DeviceContext,
    x: TileTensor[mut=True, dt, address_space=.GENERIC, ...],
    w: TileTensor[dt, address_space=.GENERIC, ...],
    ref_tt: TileTensor[mut=True, dt, address_space=.GENERIC, ...],
    got: TileTensor[mut=True, dt, address_space=.GENERIC, ...],
    bias_tt: TileTensor[dt, address_space=.GENERIC, ...],
    src: TileTensor[mut=True, dt, address_space=.GENERIC, ...],
    flags: TileTensor[mut=True, .int32, address_space=.GENERIC, ...],
) raises:
    """MIOpen's convolution into `ref_tt`, and `conv_gpu`'s into `got` with the
    epilogue `mode` names."""
    comptime H_out = (H + 2 * pad - 3) // stride + 1
    comptime W_out = (W + 2 * pad - 3) // stride + 1
    var src_p: MutPointer[Scalar[dt], MutAnyOrigin] = src.unsafe_ptr()
    # `got` is also `conv_gpu`'s output, so the closure writes through an
    # untracked view of it.
    var got_view = TileTensor(
        got.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), got.layout
    )
    var strides = Coord(stride, stride)
    var dilation = Coord(1, 1)
    var pads = Coord(pad, pad, pad, pad)

    conv_miopen[filter_is_fcrs=fcrs](
        x, w, ref_tt, strides, dilation, Coord(pad, pad), 1, ctx
    )

    @always_inline
    @__allow_legacy_any_origin_fields
    def store_bias[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {
        var bias_tt, var got_view, var flags
    }:
        comptime assert idx.rank == 4, "conv_gpu closures take NHWC"
        flags[0] = Int32(width)
        if (
            Int(idx[0]) >= N
            or Int(idx[1]) >= H_out
            or Int(idx[2]) >= W_out
            or Int(idx[3]) >= F
        ):
            flags[1] = 1
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        got_view.store[width=width, alignment=alignment](
            idx, (val + b).cast[dt]()
        )

    @always_inline
    def relu[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var flags} -> SIMD[_dtype, width]:
        flags[0] = Int32(width)
        return max(val, SIMD[_dtype, width](0))

    @always_inline
    def coord_bias[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var flags} -> SIMD[_dtype, width]:
        comptime assert idx.rank == 4, "conv_gpu closures take NHWC"
        flags[0] = Int32(width)
        if (
            Int(idx[0]) >= N
            or Int(idx[1]) >= H_out
            or Int(idx[2]) >= W_out
            or Int(idx[3]) >= F
        ):
            flags[1] = 1
        var shift = SIMD[_dtype, width](0)
        comptime for j in range(width):
            shift[j] = Scalar[_dtype](
                Float32(coord_shift(Int(idx[1]), Int(idx[2]), Int(idx[3]) + j))
            )
        return val + shift

    comptime if mode == PLAIN:
        conv_gpu[dt, dt, dt, filter_is_fcrs=fcrs](
            x, w, got, strides, dilation, pads, 1, ctx
        )
    elif mode == STORE_BIAS or mode == STORE_NO_RESIDUAL:
        conv_gpu[
            dt,
            dt,
            dt,
            has_epilogue_fn=True,
            filter_is_fcrs=fcrs,
            has_residual=mode == STORE_BIAS,
        ](
            x,
            w,
            got,
            strides,
            dilation,
            pads,
            1,
            ctx,
            src_p,
            _BETA,
            epilogue_fn=store_bias,
        )
    elif mode == COORD:
        conv_gpu[dt, dt, dt, filter_is_fcrs=fcrs, has_compute_fn=True](
            x, w, got, strides, dilation, pads, 1, ctx, compute_fn=coord_bias
        )
    else:
        conv_gpu[
            dt,
            dt,
            dt,
            filter_is_fcrs=fcrs,
            has_residual=True,
            has_compute_fn=True,
        ](
            x,
            w,
            got,
            strides,
            dilation,
            pads,
            1,
            ctx,
            src_p,
            _BETA,
            compute_fn=relu,
        )


def run_case[
    dt: DType,
    *,
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    F: Int,
    expect: Int,
    mode: Int = PLAIN,
    fcrs: Bool = True,
    stride: Int = 1,
    pad: Int = 1,
    expect_width: Int = 0,
](ctx: DeviceContext, label: StaticString) raises:
    comptime assert (
        ring_conv2d_arm(dt, fcrs, N, H, W, C, F, 3, 3) == expect
    ), "the case's expected arm is not the admission rule's"
    comptime H_out = (H + 2 * pad - 3) // stride + 1
    comptime W_out = (W + 2 * pad - 3) // stride + 1
    comptime n_in = N * H * W * C
    comptime n_w = F * C * 9
    comptime n_out = N * H_out * W_out * F
    print(t"== {label}: {dt} N={N} {H}x{W} C={C} F={F} mode={mode}")

    seed(11)
    var x_host = ctx.enqueue_create_host_buffer[dt](n_in)
    var w_host = ctx.enqueue_create_host_buffer[dt](n_w)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var bias_host = ctx.enqueue_create_host_buffer[dt](F)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var got_host = ctx.enqueue_create_host_buffer[dt](n_out)
    rand(x_host.unsafe_ptr(), n_in)
    rand(src_host.unsafe_ptr(), n_out)
    rand(bias_host.unsafe_ptr(), F)
    # the filter in FCRS order, or the same values in RSCF
    comptime if fcrs:
        rand(w_host.unsafe_ptr(), n_w)
    else:
        var w_values = ctx.enqueue_create_host_buffer[dt](n_w)
        rand(w_values.unsafe_ptr(), n_w)
        var w_fcrs = TileTensor(w_values, row_major[F, C, 3, 3]())
        var w_rscf = TileTensor(w_host, row_major[3, 3, C, F]())
        for f in range(F):
            for c in range(C):
                for r in range(3):
                    for s in range(3):
                        w_rscf[r, s, c, f] = w_fcrs[f, c, r, s]

    var x_buf = ctx.enqueue_create_buffer[dt](n_in)
    var w_buf = ctx.enqueue_create_buffer[dt](n_w)
    var src_buf = ctx.enqueue_create_buffer[dt](n_out)
    var bias_buf = ctx.enqueue_create_buffer[dt](F)
    var ref_buf = ctx.enqueue_create_buffer[dt](n_out)
    var got_buf = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(x_buf, x_host)
    ctx.enqueue_copy(w_buf, w_host)
    ctx.enqueue_copy(src_buf, src_host)
    ctx.enqueue_copy(bias_buf, bias_host)
    # a poisoned output: a path that never stores must not match
    got_buf.enqueue_fill(Scalar[dt].MAX_FINITE)

    var x = TileTensor(x_buf, row_major[N, H, W, C]())
    var ref_tt = TileTensor(ref_buf, row_major[N, H_out, W_out, F]())
    var got = TileTensor(got_buf, row_major[N, H_out, W_out, F]())
    var bias_tt = TileTensor(bias_buf, row_major[F]())
    var src = TileTensor(src_buf, row_major[N, H_out, W_out, F]())
    # [0]: the SIMD width the epilogue closure was last called with, [1]: set
    # when it was called with a coordinate outside the output
    var flags_buf = ctx.enqueue_create_buffer[.int32](2)
    flags_buf.enqueue_fill(Int32(0))
    var flags = TileTensor(flags_buf, row_major[2]())
    var flags_host = ctx.enqueue_create_host_buffer[.int32](2)
    comptime if fcrs:
        _exec[
            N=N,
            H=H,
            W=W,
            C=C,
            F=F,
            mode=mode,
            fcrs=True,
            stride=stride,
            pad=pad,
        ](
            ctx,
            x,
            TileTensor(w_buf, row_major[F, C, 3, 3]()),
            ref_tt,
            got,
            bias_tt,
            src,
            flags,
        )
    else:
        _exec[
            N=N,
            H=H,
            W=W,
            C=C,
            F=F,
            mode=mode,
            fcrs=False,
            stride=stride,
            pad=pad,
        ](
            ctx,
            x,
            TileTensor(w_buf, row_major[3, 3, C, F]()),
            ref_tt,
            got,
            bias_tt,
            src,
            flags,
        )
    ctx.enqueue_copy(ref_host, ref_buf)
    ctx.enqueue_copy(flags_host, flags_buf)
    ctx.enqueue_copy(got_host, got_buf)
    ctx.synchronize()

    comptime rel_tol = Float32(0.0157) if dt == .bfloat16 else Float32(0.002)
    var bad = 0
    var worst = Float32(0)
    for i in range(n_out):
        var conv = ref_host[i].cast[.float32]()
        var want = conv
        comptime if mode == STORE_BIAS or mode == RELU:
            var skip = src_host[i].cast[.float32]()
            want = (conv + _BETA * skip).cast[dt]().cast[.float32]()
            comptime if mode == STORE_BIAS:
                var b = bias_host[i % F].cast[.float32]()
                want = (want + b).cast[dt]().cast[.float32]()
            else:
                want = max(want, Float32(0))
        elif mode == STORE_NO_RESIDUAL:
            var b = bias_host[i % F].cast[.float32]()
            want = (conv + b).cast[dt]().cast[.float32]()
        elif mode == COORD:
            var pixel = i // F
            var shift = coord_shift(
                (pixel // W_out) % H_out, pixel % W_out, i % F
            )
            want = (conv + Float32(shift)).cast[dt]().cast[.float32]()
        var d = abs(got_host[i].cast[.float32]() - want)
        worst = max(worst, d)
        if not (d <= rel_tol * abs(want) + Float32(0.05)):
            if bad < 3:
                print(
                    t"   mismatch at {i}: got {got_host[i]} want {want} conv"
                    t" {conv}"
                )
            bad += 1
    print(t"   max_abs={worst} mismatches={bad}/{n_out}")
    assert_equal(bad, 0)
    comptime if mode != PLAIN:
        print(t"   closure width={flags_host[0]} out_of_range={flags_host[1]}")
        comptime if expect_width > 0:
            # a closure call for a pixel the output does not have is a read or
            # a write past it. The ring kernels make none; the 4-wave conv,
            # which the neighbors take, does not hold to that.
            assert_equal(flags_host[1], 0)
            # the 4-wave conv calls with 4 elements, the ring kernels with 1
            # (FCRS in place) or 8 (staged): the closure saw the ring kernel
            assert_equal(Int(flags_host[0]), expect_width)
    _ = x_buf^
    _ = w_buf^
    _ = src_buf^
    _ = bias_buf^
    _ = ref_buf^
    _ = got_buf^
    _ = flags_buf^


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    comptime bf = DType.bfloat16
    with DeviceContext() as ctx:
        # the admitted geometries, plain and with each epilogue. A compute_fn
        # with a residual runs after the conv (conv_gpu applies it in place),
        # so only the other closures show which kernel ran.
        comptime for mode in range(5):
            run_case[
                bf,
                N=1,
                H=30,
                W=30,
                C=96,
                F=96,
                expect=RING_FCRS,
                mode=mode,
                expect_width=1 if mode
                in (STORE_BIAS, COORD, STORE_NO_RESIDUAL) else 0,
            ](ctx, "30x30")
            run_case[
                bf,
                N=1,
                H=240,
                W=416,
                C=96,
                F=96,
                expect=RING_STAGED,
                mode=mode,
                expect_width=8 if mode
                in (STORE_BIAS, COORD, STORE_NO_RESIDUAL) else 0,
            ](ctx, "240x416")
        # their neighbors take the old path and stay correct
        run_case[bf, N=1, H=31, W=31, C=96, F=96, expect=RING_NONE, mode=COORD](
            ctx, "31x31 compute"
        )
        run_case[
            bf,
            N=1,
            H=240,
            W=416,
            C=96,
            F=96,
            expect=RING_NONE,
            fcrs=False,
            mode=COORD,
        ](ctx, "240x416 RSCF filter compute")
        run_case[.float16, N=1, H=30, W=30, C=96, F=96, expect=RING_NONE](
            ctx, "fp16"
        )
        run_case[bf, N=1, H=31, W=31, C=96, F=96, expect=RING_NONE](
            ctx, "31x31"
        )
        run_case[bf, N=2, H=30, W=30, C=96, F=96, expect=RING_NONE](
            ctx, "batch of two"
        )
        run_case[bf, N=1, H=30, W=30, C=64, F=64, expect=RING_NONE](
            ctx, "64 channels"
        )
        run_case[bf, N=1, H=30, W=30, C=96, F=96, expect=RING_FCRS, stride=2](
            ctx, "stride 2"
        )
        run_case[bf, N=1, H=30, W=30, C=96, F=96, expect=RING_FCRS, pad=0](
            ctx, "pad 0"
        )
        run_case[bf, N=1, H=30, W=30, C=96, F=96, expect=RING_NONE, fcrs=False](
            ctx, "RSCF filter"
        )
        run_case[
            bf,
            N=1,
            H=240,
            W=416,
            C=96,
            F=96,
            expect=RING_NONE,
            fcrs=False,
            mode=STORE_BIAS,
        ](ctx, "240x416 RSCF filter")
        run_case[bf, N=1, H=240, W=416, C=96, F=96, expect=RING_STAGED, pad=0](
            ctx, "240x416 pad 0"
        )
