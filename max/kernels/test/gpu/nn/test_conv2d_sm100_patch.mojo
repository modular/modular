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
"""The fp32 patch-embed conv2d route of `conv_gpu` (SM100, B200): a 14 x 14
filter at stride 14 over a 896 x 896 image of three channels, 1152 output
channels.

The oracle is a naive device convolution with fp32 accumulation. Operands are
small integers drawn from a hash, so every partial sum is exact in any
accumulation order and each output element is compared bit for bit; a
mismatch can only be an indexing, packing or epilogue error. The even and odd
output pixels read differently aligned boxes, so a shifted filter or a pixel
swapped between the parities fails here.

The problem runs through `conv_gpu`, plain and with every epilogue form the
entry point offers, and through `conv2d_patch_dispatch` directly, so the kernel
stays pinned even if a dispatch guard changes. A second problem uses random
operands with ten explicit mantissa bits, which the tf32 tensor core holds
exactly, and a tight tolerance: operand rounding to bf16 or fp16 would fail it.

The old path must keep serving what the kernel does not: a 448 x 448 image,
a batch of two, and a filter that overlaps (stride 7).
"""

from std.atomic import Atomic, Ordering
from std.math import ceildiv, isclose, isnan, nan
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
from nn.conv.gpu.nvidia.sm100.conv2d_patch_fprop import (
    conv2d_patch_dispatch,
    conv2d_patch_supported,
)
from linalg.utils import no_compute_fn, no_epilogue_fn

comptime CIN = 3
comptime FILTER = 14
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


def _fill[tf32_random: Bool](buf: HostBuffer[.float32], n: Int, salt: Int):
    for i in range(n):
        comptime if tf32_random:
            # k / 1024 for |k| < 1024: eleven significant bits, exact in tf32.
            buf[i] = Float32(_small(i, salt, 1023)) / 1024.0
        else:
            buf[i] = Float32(_small(i, salt, 2))


def _ref_conv[
    ALayout: TensorLayout, FLayout: TensorLayout, OLayout: TensorLayout
](
    act: ImmTileTensor[.float32, ALayout, ImmUntrackedOrigin],
    filt: ImmTileTensor[.float32, FLayout, ImmUntrackedOrigin],
    dst: MutTileTensor[.float32, OLayout, MutUntrackedOrigin],
    stride: Int32,
):
    """One thread per output element: FCRS filter, NHWC tensors, a 14 x 14
    filter without padding."""
    comptime assert act.flat_rank == 4
    comptime assert filt.flat_rank == 4
    comptime assert dst.flat_rank == 4
    var out_h = Int(dst.dim[1]())
    var out_w = Int(dst.dim[2]())
    var cout = Int(dst.dim[3]())
    var i = global_idx.x
    if i >= dst.num_elements():
        return
    var pixel, f = divmod(i, cout)
    var row, wo = divmod(pixel, out_w)
    var n, ho = divmod(row, out_h)
    var acc = Float32(0)
    for r in range(FILTER):
        for s in range(FILTER):
            var hi = ho * Int(stride) + r
            var wi = wo * Int(stride) + s
            for c in range(CIN):
                acc += act[n, hi, wi, c] * filt[f, c, r, s]
    dst[n, ho, wo, f] = acc


def _count_bad[
    exact: Bool,
    MLayout: TensorLayout,
    BiasLayout: TensorLayout,
    BadLayout: TensorLayout,
](
    got: ImmTileTensor[.float32, MLayout, ImmUntrackedOrigin],
    expect: ImmTileTensor[.float32, MLayout, ImmUntrackedOrigin],
    bias: ImmTileTensor[.float32, BiasLayout, ImmUntrackedOrigin],
    src: ImmTileTensor[.float32, MLayout, ImmUntrackedOrigin],
    bad: MutTileTensor[.uint32, BadLayout, MutUntrackedOrigin],
    mode: Int32,
):
    """Counts the outputs, viewed `[pixels, channels]`, that differ from the
    reference with the epilogue `mode` applied to it."""
    var pixel, c = divmod(global_idx.x, Int(got.dim[1]()))
    if pixel >= Int(got.dim[0]()):
        return
    var want = expect[pixel, c]
    var m = Int(mode)
    if m == RESIDUAL or m == RESIDUAL_BIAS:
        want += Float32(BETA) * src[pixel, c]
    if m == BIAS or m == RESIDUAL_BIAS:
        want = (want + bias[c]) + 1
    elif m == COMPUTE:
        want = want * 2 + bias[c]
    var have = got[pixel, c]
    var wrong: Bool
    if isnan(want) or isnan(have):
        # A non-finite input pixel must reach its own outputs and nothing else.
        wrong = isnan(want) != isnan(have)
    elif exact:
        wrong = have != want
    else:
        wrong = not isclose(have, want, atol=1e-4, rtol=1e-4)
    if wrong:
        _ = Atomic[UInt32].fetch_add[ordering=Ordering.relaxed](
            bad.unsafe_ptr(), UInt32(1)
        )


def _check[
    exact: Bool
](
    ctx: DeviceContext,
    out_dev: DeviceBuffer[.float32],
    ref_dev: DeviceBuffer[.float32],
    bias_dev: DeviceBuffer[.float32],
    src_dev: DeviceBuffer[.float32],
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
    exact: Bool = True, served: Bool = True, nonfinite: Bool = False
](
    ctx: DeviceContext,
    batch: Int,
    size: Int,
    cout: Int,
    stride: Int = FILTER,
) raises:
    """Runs one problem through every launch form and checks each.

    Parameters:
        exact: Small-integer operands compared bit for bit; otherwise random
            tf32-exact operands compared with a tight tolerance.
        served: The problem is one the patch kernel serves, so
            `conv2d_patch_supported` accepts it and the direct launch runs.
        nonfinite: Put a NaN in one odd-column patch and one even-column patch.
            The reference carries it to those patches' outputs only, so a
            neighbour that is not exactly the reference's fails.

    Args:
        ctx: Device context.
        batch: Batch size.
        size: Input height and width.
        cout: Output channels.
        stride: Stride along both axes.
    """
    comptime dt = DType.float32
    var out_size = (size - FILTER) // stride + 1
    var n_act = batch * size * size * CIN
    var n_filt = cout * CIN * FILTER * FILTER
    var n_out = batch * out_size * out_size * cout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](cout)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    _fill[not exact](act_host, n_act, size + 11)
    _fill[not exact](filt_host, n_filt, 3)
    _fill[not exact](bias_host, cout, 5)
    _fill[not exact](src_host, n_out, 9)
    comptime if nonfinite:
        # Elements a neighbour's box also covers: the first pixel of an odd
        # patch (read by the even patch before it), the first pixel of the
        # even patch after an odd one, and the last pixel's last channels of
        # an even patch (read by the odd patch after it).
        var nans = List[Tuple[Int, Int, Int, Int]]()
        nans.append((5, 9, 0, 0))
        nans.append((20, 30, 0, 0))
        nans.append((20, 30, FILTER - 1, CIN - 1))
        var act_host_tt = TileTensor(
            act_host, row_major(batch, size, size, CIN)
        )
        for nan_at in nans:
            var h = nan_at[0] * FILTER
            var w = nan_at[1] * FILTER + nan_at[2]
            act_host_tt[0, h, w, nan_at[3]] = nan[.float32]()
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

    var act_tt = TileTensor(act_dev, row_major(batch, size, size, CIN))
    var filt_tt = TileTensor(filt_dev, row_major(cout, CIN, FILTER, FILTER))
    var out_tt = TileTensor(out_dev, row_major(batch, out_size, out_size, cout))
    var ref_tt = TileTensor(ref_dev, row_major(batch, out_size, out_size, cout))
    comptime ref_kernel = _ref_conv[
        type_of(act_tt).LayoutType,
        type_of(filt_tt).LayoutType,
        type_of(ref_tt).LayoutType,
    ]
    ctx.enqueue_function[ref_kernel](
        act_tt.as_imm(),
        filt_tt.as_imm(),
        ref_tt,
        Int32(stride),
        grid_dim=ceildiv(n_out, BLOCK),
        block_dim=BLOCK,
    )
    var bias_tt = TileTensor(bias_dev, row_major(cout))
    # The storing closure writes through a view of its own: it may not borrow
    # the tensor `conv_gpu` receives alongside it.
    var out_view = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        row_major(batch, out_size, out_size, cout),
    )
    # `conv_gpu` takes its residual source as an any-origin pointer.
    var src_ptr = src_dev.unsafe_ptr().as_unsafe_any_origin()
    var strides = Coord(stride, stride)
    var dilation = Coord(1, 1)
    var padding = Coord(0, 0, 0, 0)

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
        act_tt, filt_tt, out_tt, strides, dilation, padding, 1, ctx
    )
    _check[exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, PLAIN)

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, has_epilogue_fn=True, filter_is_fcrs=True](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        dilation,
        padding,
        1,
        ctx,
        epilogue_fn=bias_plus_one,
    )
    _check[exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, BIAS)

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, has_compute_fn=True, filter_is_fcrs=True](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        dilation,
        padding,
        1,
        ctx,
        compute_fn=double_plus_bias,
    )
    _check[exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, COMPUTE)

    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, filter_is_fcrs=True, has_residual=True](
        act_tt,
        filt_tt,
        out_tt,
        strides,
        dilation,
        padding,
        1,
        ctx,
        src_ptr,
        Float32(BETA),
    )
    _check[exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, RESIDUAL)

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
        1,
        ctx,
        src_ptr,
        Float32(BETA),
        bias_plus_one,
    )
    _check[exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, RESIDUAL_BIAS)

    var supported = conv2d_patch_supported(
        Coord(batch, size, size, CIN),
        Coord(cout, CIN, FILTER, FILTER),
        strides,
        dilation,
        Coord(0, 0),
        1,
    )
    assert_equal(supported, served)
    if served:
        out_dev.enqueue_fill(Scalar[dt](-1000))
        assert_true(
            conv2d_patch_dispatch[False, False](
                act_tt,
                filt_tt,
                out_tt,
                strides,
                dilation,
                Coord(0, 0),
                1,
                ctx,
                no_epilogue_fn,
                no_compute_fn,
            )
        )
        _check[exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, PLAIN)

        out_dev.enqueue_fill(Scalar[dt](-1000))
        assert_true(
            conv2d_patch_dispatch[True, False](
                act_tt,
                filt_tt,
                out_tt,
                strides,
                dilation,
                Coord(0, 0),
                1,
                ctx,
                bias_plus_one,
                no_compute_fn,
            )
        )
        _check[exact](ctx, out_dev, ref_dev, bias_dev, src_dev, cout, BIAS)
    var kind = "exact" if exact else "random"
    print(
        t"PASS batch {batch} {size} x {size} cout {cout} stride {stride} {kind}"
    )


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator")
    else:
        with DeviceContext() as ctx:
            # The served patch embed.
            run_problem(ctx, 1, 896, 1152)
            # Full-mantissa operands that a tf32 tensor core holds exactly.
            run_problem[exact=False](ctx, 1, 896, 1152)
            # A NaN pixel stays in its own patch's outputs.
            run_problem[nonfinite=True](ctx, 1, 896, 1152)

            # What the kernel does not serve keeps its old path.
            run_problem[served=False](ctx, 1, 448, 1152)
            run_problem[served=False](ctx, 2, 224, 768)
            run_problem[served=False](ctx, 1, 133, 128, stride=7)
