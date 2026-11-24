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
"""The column-tile 3x3 conv2d route of `conv_gpu` (SM100, B200).

The oracle is a naive device convolution that accumulates in fp32 and rounds
once to the output dtype. Operands are small integers drawn from a hash, so
every partial sum is exact in any accumulation order (the sums stay within
the 2048 an fp16 output holds) and a mismatch can only be an indexing,
padding, swizzle or epilogue error; each output element is compared bit for
bit.

Each served problem (fp16, bf16 and fp32 96 -> 96 channels and fp32
192 -> 96, on the 240 x 416 image) runs through `conv_gpu`, plain and with
every epilogue form the entry point offers (a storing epilogue, a compute
epilogue, a residual alone, a residual followed by the storing epilogue), and
through `conv2d_column_dispatch` directly, so the kernel stays pinned even if
a dispatch guard changes.

One fp32 problem uses random operands with ten explicit mantissa bits, which
the tf32 tensor core holds exactly, and a tight tolerance: operand rounding to
bf16 or fp16 would fail it.

The old paths must keep serving what the kernel does not: other channel
counts (the 16-bit 192 -> 96 included), a smaller image, a batch of two, a stride of two, and a 3 x 3
over a group.
"""

from std.atomic import Atomic, Ordering
from std.math import ceildiv, isclose
from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true

from layout import (
    Coord,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    row_major,
)
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from nn.conv.conv import conv_gpu
from nn.conv.gpu.nvidia.sm100.conv2d_column_fprop import (
    conv2d_column_dispatch,
    conv2d_column_supported,
)
from nn.conv.gpu.nvidia.sm100.conv_config import Conv2dProblemShape

comptime BETA = 2
comptime PLAIN = 0
comptime BIAS = 1
comptime COMPUTE = 2
comptime RESIDUAL = 3
comptime RESIDUAL_BIAS = 4
comptime BLOCK = 256


def _small(i: Int, salt: Int, span: Int) -> Int:
    """A deterministic pseudo-random integer in [-span, span]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return (h >> 9) % (2 * span + 1) - span


def _fill[
    dt: DType, tf32_random: Bool
](buf: HostBuffer[dt], n: Int, salt: Int, span: Int):
    for i in range(n):
        comptime if tf32_random:
            # k / 1024 for |k| < 1024: eleven significant bits, exact in tf32.
            buf[i] = Scalar[dt](Float32(_small(i, salt, 1023)) / 1024.0)
        else:
            buf[i] = Scalar[dt](Float32(_small(i, salt, span)))


def _ref_conv[
    dt: DType,
    ALayout: TensorLayout,
    FLayout: TensorLayout,
    OLayout: TensorLayout,
](
    act: ImmTileTensor[dt, ALayout, ImmUntrackedOrigin],
    filt: ImmTileTensor[dt, FLayout, ImmUntrackedOrigin],
    dst: MutTileTensor[dt, OLayout, MutUntrackedOrigin],
    sh: Int32,
    sw: Int32,
    pad_h: Int32,
    pad_w: Int32,
    groups: Int32,
):
    """One thread per output element: FCRS filter, NHWC tensors, 3x3."""
    comptime assert act.flat_rank == 4
    comptime assert filt.flat_rank == 4
    comptime assert dst.flat_rank == 4
    var h = Int(act.dim[1]())
    var w = Int(act.dim[2]())
    var cin_g = Int(filt.dim[1]())
    var out_h = Int(dst.dim[1]())
    var out_w = Int(dst.dim[2]())
    var cout = Int(dst.dim[3]())
    var i = global_idx.x
    if i >= dst.num_elements():
        return
    var pixel, f = divmod(i, cout)
    var row, wo = divmod(pixel, out_w)
    var n, ho = divmod(row, out_h)
    var c0 = (f // (cout // Int(groups))) * cin_g
    var acc = Float32(0)
    for r in range(3):
        for s in range(3):
            var hi = ho * Int(sh) - Int(pad_h) + r
            var wi = wo * Int(sw) - Int(pad_w) + s
            if hi < 0 or hi >= h or wi < 0 or wi >= w:
                continue
            for c in range(cin_g):
                acc += Float32(act[n, hi, wi, c0 + c]) * Float32(
                    filt[f, c, r, s]
                )
    dst[n, ho, wo, f] = acc.cast[dt]()


def _count_bad[
    dt: DType,
    exact: Bool,
    MLayout: TensorLayout,
    BiasLayout: TensorLayout,
    BadLayout: TensorLayout,
](
    got: ImmTileTensor[dt, MLayout, ImmUntrackedOrigin],
    expect: ImmTileTensor[dt, MLayout, ImmUntrackedOrigin],
    bias: ImmTileTensor[dt, BiasLayout, ImmUntrackedOrigin],
    src: ImmTileTensor[dt, MLayout, ImmUntrackedOrigin],
    bad: MutTileTensor[.uint32, BadLayout, MutUntrackedOrigin],
    mode: Int32,
):
    """Counts the outputs, viewed `[pixels, channels]`, that differ from the
    reference with the epilogue `mode` applied to it, in the output dtype."""
    var pixel, c = divmod(global_idx.x, Int(got.dim[1]()))
    if pixel >= Int(got.dim[0]()):
        return
    var want = Scalar[dt](expect[pixel, c])
    var m = Int(mode)
    if m == RESIDUAL or m == RESIDUAL_BIAS:
        want += Scalar[dt](BETA) * src[pixel, c]
    if m == BIAS or m == RESIDUAL_BIAS:
        want = (want + bias[c]) + 1
    elif m == COMPUTE:
        want = want * 2 + bias[c]
    var wrong: Bool
    comptime if exact:
        wrong = got[pixel, c] != want
    else:
        var g = Float32(got[pixel, c])
        var e = Float32(want)
        wrong = not isclose(g, e, atol=1e-4, rtol=1e-4)
    if wrong:
        _ = Atomic[UInt32].fetch_add[ordering=Ordering.relaxed](
            bad.unsafe_ptr(), UInt32(1)
        )


def _check[
    dt: DType, exact: Bool
](
    ctx: DeviceContext,
    out_dev: DeviceBuffer[dt],
    ref_dev: DeviceBuffer[dt],
    bias_dev: DeviceBuffer[dt],
    src_dev: DeviceBuffer[dt],
    cout: Int,
    mode: Int,
) raises:
    var n_out = len(out_dev)
    var bad_dev = ctx.enqueue_create_buffer[.uint32](1)
    bad_dev.enqueue_fill(UInt32(0))
    var shape = row_major(n_out // cout, cout)
    var got = TileTensor(out_dev, shape)
    var expect = TileTensor(ref_dev, shape)
    var src = TileTensor(src_dev, shape)
    var bias = TileTensor(bias_dev, row_major(cout))
    var bad_tt = TileTensor(bad_dev, row_major[1]())
    comptime kernel = _count_bad[
        dt,
        exact,
        type_of(got).LayoutType,
        type_of(bias).LayoutType,
        type_of(bad_tt).LayoutType,
    ]
    ctx.enqueue_function[kernel](
        got.as_imm(),
        expect.as_imm(),
        bias.as_imm(),
        src.as_imm(),
        bad_tt,
        Int32(mode),
        grid_dim=ceildiv(n_out, BLOCK),
        block_dim=BLOCK,
    )
    var bad_host = ctx.enqueue_create_host_buffer[.uint32](1)
    ctx.enqueue_copy(bad_host, bad_dev)
    ctx.synchronize()
    var bad = Int(bad_host[0])
    if bad != 0:
        print(t"mode {mode} mismatches {bad} of {n_out}")
    assert_equal(bad, 0)


def run_problem[
    dt: DType,
    cin: Int,
    cout: Int,
    exact: Bool = True,
    served: Bool = True,
    direct: Bool = True,
](
    ctx: DeviceContext,
    batch: Int,
    h: Int,
    w: Int,
    stride: Int = 1,
    groups: Int = 1,
) raises:
    """Runs one problem through every launch form and checks each.

    Parameters:
        dt: Element type of every tensor.
        cin: Input channels.
        cout: Output channels.
        exact: Small-integer operands compared bit for bit; otherwise random
            tf32-exact operands compared with a tight tolerance.
        served: The problem is one the column kernel serves, so
            `conv2d_column_supported` accepts it and the direct launch runs.
        direct: Also launch `conv2d_column_dispatch` itself.

    Args:
        ctx: Device context.
        batch: Batch size.
        h: Input height.
        w: Input width.
        stride: Stride along both axes.
        groups: Convolution groups.
    """
    var out_h = (h + 2 - 3) // stride + 1
    var out_w = (w + 2 - 3) // stride + 1
    var n_act = batch * h * w * cin
    var n_filt = cout * (cin // groups) * 9
    var n_out = batch * out_h * out_w * cout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](cout)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    # fp16 holds integers to 2048: a 1728-term sum of {-1, 0, 1} products does.
    comptime span = 1 if dt == .float16 else 2
    _fill[dt, not exact](act_host, n_act, h * 7 + w, span)
    _fill[dt, not exact](filt_host, n_filt, 3, span)
    _fill[dt, not exact](bias_host, cout, 5, span)
    _fill[dt, not exact](src_host, n_out, 9, span)

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](cout)
    var src_dev = ctx.enqueue_create_buffer[dt](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var ref_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_copy(src_dev, src_host)

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, cin))
    var filt_tt = TileTensor(filt_dev, row_major(cout, cin // groups, 3, 3))
    var out_tt = TileTensor(out_dev, row_major(batch, out_h, out_w, cout))
    var ref_tt = TileTensor(ref_dev, row_major(batch, out_h, out_w, cout))
    comptime ref_kernel = _ref_conv[
        dt,
        type_of(act_tt).LayoutType,
        type_of(filt_tt).LayoutType,
        type_of(ref_tt).LayoutType,
    ]
    ctx.enqueue_function[ref_kernel](
        act_tt.as_imm(),
        filt_tt.as_imm(),
        ref_tt,
        Int32(stride),
        Int32(stride),
        Int32(1),
        Int32(1),
        Int32(groups),
        grid_dim=ceildiv(n_out, BLOCK),
        block_dim=BLOCK,
    )
    var bias_tt = TileTensor(bias_dev, row_major(cout))
    # The storing closure writes through a view of its own: it may not borrow
    # the tensor `conv_gpu` receives alongside it.
    var out_view = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        row_major(batch, out_h, out_w, cout),
    )
    # `conv_gpu` takes its residual source as an any-origin pointer.
    var src_ptr = src_dev.unsafe_ptr().as_unsafe_any_origin()
    var strides = Coord(stride, stride)
    var dilation = Coord(1, 1)
    var padding = Coord(1, 1, 1, 1)

    @always_inline
    @__allow_legacy_any_origin_fields
    def bias_plus_one[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var out_view}:
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        out_view.store[width=Int(width)](idx, (val + b + 1).cast[dt]())

    @always_inline
    @__allow_legacy_any_origin_fields
    def double_plus_bias[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt} -> SIMD[
        _dtype, width
    ]:
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        return val * 2 + b

    ctx.synchronize()

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, filter_is_fcrs=True](
        act_tt, filt_tt, out_tt, strides, dilation, padding, groups, ctx
    )
    _check[dt, exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, PLAIN)

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, has_epilogue_fn=True, filter_is_fcrs=True](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        dilation,
        padding,
        groups,
        ctx,
        epilogue_fn=bias_plus_one,
    )
    _check[dt, exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, BIAS)

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, has_compute_fn=True, filter_is_fcrs=True](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        dilation,
        padding,
        groups,
        ctx,
        compute_fn=double_plus_bias,
    )
    _check[dt, exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, COMPUTE)

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, filter_is_fcrs=True, has_residual=True](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        dilation,
        padding,
        groups,
        ctx,
        src_ptr,
        Float32(BETA),
    )
    _check[dt, exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, RESIDUAL)

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[
        dt,
        dt,
        dt,
        has_epilogue_fn=True,
        filter_is_fcrs=True,
        has_residual=True,
    ](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        dilation,
        padding,
        groups,
        ctx,
        src_ptr,
        Float32(BETA),
        bias_plus_one,
    )
    _check[dt, exact](
        ctx, out_dev, ref_dev, bias_dev, src_dev, cout, RESIDUAL_BIAS
    )

    comptime if direct:
        var problem = Conv2dProblemShape(
            batch=batch,
            in_height=h,
            in_width=w,
            in_channels=cin,
            out_channels=cout,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )
        assert_equal(conv2d_column_supported[dt](problem), served)
        if served:
            out_dev.enqueue_fill(Scalar[dt](-1000))
            assert_true(
                conv2d_column_dispatch[filter_is_fcrs=True](
                    act_tt, filt_tt, out_tt, 1, 1, ctx
                )
            )
            _check[dt, exact](
                ctx, out_dev, ref_dev, bias_dev, src_dev, cout, PLAIN
            )

            out_dev.enqueue_fill(Scalar[dt](-1000))
            assert_true(
                conv2d_column_dispatch[
                    filter_is_fcrs=True, has_epilogue_fn=True
                ](
                    act_tt,
                    filt_tt,
                    out_tt,
                    1,
                    1,
                    ctx,
                    epilogue_fn=bias_plus_one,
                )
            )
            _check[dt, exact](
                ctx, out_dev, ref_dev, bias_dev, src_dev, cout, BIAS
            )
    var kind = "exact" if exact else "random"
    print(
        t"PASS {dt} {cin} -> {cout} batch {batch} {h} x {w} stride {stride}"
        t" groups {groups} {kind}"
    )


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator")
    else:
        with DeviceContext() as ctx:
            # The served problems, on the production image.
            run_problem[.float16, 96, 96](ctx, 1, 240, 416)
            run_problem[.bfloat16, 96, 96](ctx, 1, 240, 416)
            run_problem[.float32, 96, 96](ctx, 1, 240, 416)
            run_problem[.float32, 192, 96](ctx, 1, 240, 416)

            # Full-mantissa operands that a tf32 tensor core holds exactly.
            run_problem[.float32, 96, 96, exact=False](ctx, 1, 240, 416)

            # What the kernel does not serve keeps its old path.
            run_problem[.float16, 192, 96, served=False](ctx, 1, 240, 416)
            run_problem[.bfloat16, 192, 96, served=False](ctx, 1, 240, 416)
            run_problem[.float16, 192, 192, served=False](ctx, 1, 120, 208)
            run_problem[.float16, 96, 96, served=False](ctx, 1, 130, 150)
            run_problem[.bfloat16, 96, 96, served=False](ctx, 2, 70, 90)
            run_problem[.float32, 96, 96, served=False](
                ctx, 1, 130, 150, stride=2
            )
            run_problem[.float32, 192, 192, served=False](ctx, 1, 120, 208)
            run_problem[.float32, 96, 96, served=False](
                ctx, 1, 130, 150, groups=2
            )
