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
"""The fp32 3 -> 96 channel 3x3 conv2d route of `conv_gpu` (SM100, B200).

The oracle is a host loop, not cuDNN. Operands are small integers drawn from a
hash, so every fp32 partial sum is exact in any accumulation order and a
mismatch can only be an indexing, padding or epilogue error; each output
element is compared bit for bit.

Every geometry runs six ways: through `conv_gpu` and through
`conv2d_fp32_small_c` directly (so the kernel is pinned even if a dispatch
guard changes), each plain and with the storing epilogue
`acc + bias[channel] + 1`; and, through `conv_gpu` only, a residual
`acc + 2 * source` alone and followed by the same storing epilogue (the
residual reaches the kernel as a store epilogue).

One geometry pair uses random fp32 operands (full mantissas) compared with a
1e-5 tolerance against the same host loop, so a future tf32 or bf16 operand
rounding would fail there.

Geometries: the production image (240 x 416); a batch of three 140 x 200
images, where each of the grid's warps walks 35 pixels, so a warp's run spans
a second pixel chunk and crosses an image boundary; odd extents; and
non-square stride, dilation and padding, which a swapped height/width value
would break.

The old path must keep serving what the kernel does not: 64 output channels,
three groups and a compute epilogue run through `conv_gpu` and are compared
with the host loop as well.
"""

from std.math import isclose
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
from max.gpu.host import DeviceContext, HostBuffer
from nn.conv.conv import conv_gpu
from nn.conv.conv_utils import no_conv_simd_epilogue_fn
from nn.conv.gpu.nvidia.sm100.conv2d_fp32_small_c import (
    conv2d_fp32_small_c,
    conv2d_fp32_small_c_supported,
)

comptime CIN = 3
comptime COUT = 96
comptime BETA = Float32(2)
comptime WARPS_PER_SM = 16
"""Two blocks of eight warps: the kernel's persistent grid."""

comptime PLAIN = 0
comptime BIAS = 1
comptime RESIDUAL = 2
comptime RESIDUAL_BIAS = 3


def _hash_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 9) % 5 - 2)


def _float_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random fp32 in [-1, 1] with a full mantissa."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 8) % 2000001) / 1000000.0 - 1.0


def _fill[exact: Bool](buf: HostBuffer[.float32], n: Int, salt: Int):
    for i in range(n):
        comptime if exact:
            buf[i] = _hash_value(i, salt)
        else:
            buf[i] = _float_value(i, salt)


def _out_extent(n: Int, pad: Int, dilation: Int, stride: Int) -> Int:
    return (n + 2 * pad - dilation * 2 - 1) // stride + 1


def host_conv[
    ALayout: TensorLayout, FLayout: TensorLayout, OLayout: TensorLayout
](
    act: ImmTileTensor[.float32, ALayout, _],
    filt: ImmTileTensor[.float32, FLayout, _],
    dst: MutTileTensor[.float32, OLayout, _],
    groups: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
):
    """The convolution as a loop: NHWC activation, FCRS filter."""
    comptime assert act.flat_rank == 4
    comptime assert filt.flat_rank == 4
    comptime assert dst.flat_rank == 4
    var h = Int(act.dim[1]())
    var w = Int(act.dim[2]())
    var out_h = Int(dst.dim[1]())
    var out_w = Int(dst.dim[2]())
    var cin_g = CIN // groups
    var cout_g = Int(dst.dim[3]()) // groups
    for n in range(Int(dst.dim[0]())):
        for ho in range(out_h):
            for wo in range(out_w):
                for f in range(Int(dst.dim[3]())):
                    var c0 = (f // cout_g) * cin_g
                    var acc = Float32(0)
                    for r in range(3):
                        for s in range(3):
                            var hi = ho * sh - ph + r * dh
                            var wi = wo * sw - pw + s * dw
                            if hi < 0 or hi >= h or wi < 0 or wi >= w:
                                continue
                            for c in range(cin_g):
                                acc += act[n, hi, wi, c0 + c] * filt[f, c, r, s]
                    dst[n, ho, wo, f] = acc


def _count_bad[
    exact: Bool
](
    got_buf: HostBuffer[.float32],
    expect_buf: HostBuffer[.float32],
    bias_buf: HostBuffer[.float32],
    src_buf: HostBuffer[.float32],
    cout: Int,
    mode: Int,
) -> Int:
    """Counts the outputs, viewed `[pixels, channels]`, that differ from the
    reference with the epilogue `mode` applied to it."""
    var shape = row_major(len(got_buf) // cout, cout)
    var got = TileTensor(got_buf, shape)
    var expect = TileTensor(expect_buf, shape)
    var src = TileTensor(src_buf, shape)
    var bias = TileTensor(bias_buf, row_major(cout))
    var bad = 0
    for pixel in range(Int(got.dim[0]())):
        for c in range(cout):
            var want = expect[pixel, c]
            if mode == RESIDUAL or mode == RESIDUAL_BIAS:
                want += BETA * src[pixel, c]
            if mode == BIAS or mode == RESIDUAL_BIAS:
                want += bias[c] + 1
            comptime if exact:
                if got[pixel, c] != want:
                    bad += 1
            else:
                # Tight enough that tf32 or bf16 operand rounding (about 1e-3)
                # fails, loose enough for the 27-term fp32 sum in another order
                # and the epilogue's association.
                if not isclose[symmetrical=False](
                    got[pixel, c], want, atol=1e-5, rtol=1e-5
                ):
                    bad += 1
    return bad


def run_geometry[
    exact: Bool = True
](
    ctx: DeviceContext,
    batch: Int,
    h: Int,
    w: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
) raises:
    """Runs one geometry through all six launch forms and checks each.

    Parameters:
        exact: Small-integer operands compared bit for bit; otherwise random
            fp32 operands compared with a tight tolerance.

    Args:
        ctx: Device context.
        batch: Batch size.
        h: Input height.
        w: Input width.
        sh: Stride along the height.
        sw: Stride along the width.
        dh: Dilation along the height.
        dw: Dilation along the width.
        ph: Padding along the height.
        pw: Padding along the width.
    """
    comptime dt = DType.float32
    var out_h = _out_extent(h, ph, dh, sh)
    var out_w = _out_extent(w, pw, dw, sw)
    var n_act = batch * h * w * CIN
    var n_filt = COUT * CIN * 9
    var n_out = batch * out_h * out_w * COUT

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](COUT)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    _fill[exact](act_host, n_act, h * 7 + w)
    _fill[exact](filt_host, n_filt, 3)
    _fill[exact](bias_host, COUT, 5)
    _fill[exact](src_host, n_out, 9)
    host_conv(
        TileTensor(act_host, row_major(batch, h, w, CIN)),
        TileTensor(filt_host, row_major(COUT, CIN, 3, 3)),
        TileTensor(ref_host, row_major(batch, out_h, out_w, COUT)),
        1,
        sh,
        sw,
        dh,
        dw,
        ph,
        pw,
    )

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](COUT)
    var src_dev = ctx.enqueue_create_buffer[dt](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_copy(src_dev, src_host)

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, CIN))
    var filt_tt = TileTensor(filt_dev, row_major(COUT, CIN, 3, 3))
    var out_tt = TileTensor(out_dev, row_major(batch, out_h, out_w, COUT))
    var bias_tt = TileTensor(bias_dev, row_major(COUT))
    # The storing closure writes through a view of its own: it may not borrow
    # the tensor `conv_gpu` receives alongside it.
    var out_view = TileTensor(
        out_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        row_major(batch, out_h, out_w, COUT),
    )
    # `conv_gpu` takes its residual source as an any-origin pointer.
    var src_ptr = src_dev.unsafe_ptr().as_unsafe_any_origin()
    var stride = Coord(sh, sw)
    var dilation = Coord(dh, dw)
    var padding = Coord(ph, ph, pw, pw)

    @always_inline
    @__allow_legacy_any_origin_fields
    def bias_plus_one[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var out_view}:
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        out_view.store[width=Int(width)](idx, (val + b + 1).cast[dt]())

    # Each launch form: 0 conv_gpu, 1 direct.
    comptime for direct in range(2):
        comptime for mode in range(2):
            out_dev.enqueue_fill(Scalar[dt](-1000))
            comptime if direct == 1:
                comptime if mode == BIAS:
                    conv2d_fp32_small_c[True](
                        act_tt,
                        filt_tt,
                        out_tt,
                        sh,
                        sw,
                        dh,
                        dw,
                        ph,
                        pw,
                        ctx,
                        bias_plus_one,
                    )
                else:
                    conv2d_fp32_small_c[False](
                        act_tt,
                        filt_tt,
                        out_tt,
                        sh,
                        sw,
                        dh,
                        dw,
                        ph,
                        pw,
                        ctx,
                        no_conv_simd_epilogue_fn,
                    )
            else:
                comptime if mode == BIAS:
                    conv_gpu[
                        dt,
                        dt,
                        dt,
                        has_epilogue_fn=True,
                        filter_is_fcrs=True,
                    ](
                        act_tt,
                        filt_tt,
                        out_tt,
                        stride,
                        dilation,
                        padding,
                        1,
                        ctx,
                        epilogue_fn=bias_plus_one,
                    )
                else:
                    conv_gpu[dt, dt, dt, filter_is_fcrs=True](
                        act_tt,
                        filt_tt,
                        out_tt,
                        stride,
                        dilation,
                        padding,
                        1,
                        ctx,
                    )
            ctx.enqueue_copy(out_host, out_dev)
            ctx.synchronize()
            assert_equal(
                _count_bad[exact](
                    out_host, ref_host, bias_host, src_host, COUT, mode
                ),
                0,
            )

    # The residual, alone and followed by the storing epilogue.
    out_dev.enqueue_fill(Scalar[dt](-1000))
    conv_gpu[dt, dt, dt, filter_is_fcrs=True, has_residual=True](
        act_tt,
        filt_tt,
        out_tt,
        stride,
        dilation,
        padding,
        1,
        ctx,
        src_ptr,
        BETA,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    assert_equal(
        _count_bad[exact](
            out_host, ref_host, bias_host, src_host, COUT, RESIDUAL
        ),
        0,
    )

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
        stride,
        dilation,
        padding,
        1,
        ctx,
        src_ptr,
        BETA,
        bias_plus_one,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    assert_equal(
        _count_bad[exact](
            out_host, ref_host, bias_host, src_host, COUT, RESIDUAL_BIAS
        ),
        0,
    )
    print(
        t"PASS batch {batch} in {h} x {w} stride {sh} {sw} dilation {dh} {dw}"
        t" pad {ph} {pw}"
    )


def run_old_path[
    cout: Int, groups: Int, compute: Bool
](ctx: DeviceContext) raises:
    """A convolution the kernel does not serve, through `conv_gpu`.

    Parameters:
        cout: Output channels.
        groups: Convolution groups.
        compute: Apply a compute epilogue `acc + bias[channel]`.

    Args:
        ctx: Device context.
    """
    comptime dt = DType.float32
    var batch = 2
    var h = 20
    var w = 21
    var out_h = _out_extent(h, 1, 1, 1)
    var out_w = _out_extent(w, 1, 1, 1)
    var n_act = batch * h * w * CIN
    var n_filt = cout * (CIN // groups) * 9
    var n_out = batch * out_h * out_w * cout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](cout)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    _fill[True](act_host, n_act, 21)
    _fill[True](filt_host, n_filt, 4)
    _fill[True](bias_host, cout, 6)
    host_conv(
        TileTensor(act_host, row_major(batch, h, w, CIN)),
        TileTensor(filt_host, row_major(cout, CIN // groups, 3, 3)),
        TileTensor(ref_host, row_major(batch, out_h, out_w, cout)),
        groups,
        1,
        1,
        1,
        1,
        1,
        1,
    )

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](cout)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    out_dev.enqueue_fill(Scalar[dt](-1000))

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, CIN))
    var filt_tt = TileTensor(filt_dev, row_major(cout, CIN // groups, 3, 3))
    var out_tt = TileTensor(out_dev, row_major(batch, out_h, out_w, cout))
    var bias_tt = TileTensor(bias_dev, row_major(cout))

    @always_inline
    @__allow_legacy_any_origin_fields
    def add_bias[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt} -> SIMD[
        _dtype, width
    ]:
        return val + bias_tt.load[width=width]((idx[3],)).cast[_dtype]()

    comptime if compute:
        conv_gpu[dt, dt, dt, has_compute_fn=True, filter_is_fcrs=True](
            act_tt,
            filt_tt,
            out_tt,
            Coord(1, 1),
            Coord(1, 1),
            Coord(1, 1, 1, 1),
            groups,
            ctx,
            compute_fn=add_bias,
        )
    else:
        conv_gpu[dt, dt, dt, filter_is_fcrs=True](
            act_tt,
            filt_tt,
            out_tt,
            Coord(1, 1),
            Coord(1, 1),
            Coord(1, 1, 1, 1),
            groups,
            ctx,
        )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var ref_2d = TileTensor(ref_host, row_major(n_out // cout, cout))
    var out_2d = TileTensor(out_host, row_major(n_out // cout, cout))
    var bad = 0
    for pixel in range(n_out // cout):
        for c in range(cout):
            var want = ref_2d[pixel, c]
            comptime if compute:
                want += bias_host[c]
            if out_2d[pixel, c] != want:
                bad += 1
    assert_equal(bad, 0)
    print(t"PASS old path cout {cout} groups {groups} compute {compute}")


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator")
    else:
        with DeviceContext() as ctx:
            # Servable and not servable envelopes.
            assert_true(conv2d_fp32_small_c_supported(3, 96, 3, 3))
            assert_false(conv2d_fp32_small_c_supported(3, 64, 3, 3))
            assert_false(conv2d_fp32_small_c_supported(4, 96, 3, 3))
            assert_false(conv2d_fp32_small_c_supported(3, 96, 5, 3))

            # The production image.
            run_geometry(ctx, 1, 240, 416, 1, 1, 1, 1, 1, 1)

            # Each warp walks 35 pixels of an 84000-pixel batch of three.
            var warps = ctx.default_device_info.sm_count * WARPS_PER_SM
            assert_true(3 * 140 * 200 > 33 * warps)
            run_geometry(ctx, 3, 140, 200, 1, 1, 1, 1, 1, 1)

            run_geometry(ctx, 2, 17, 29, 1, 1, 1, 1, 1, 1)
            run_geometry(ctx, 2, 33, 40, 2, 1, 1, 2, 1, 2)
            run_geometry(ctx, 1, 41, 37, 1, 3, 2, 1, 2, 0)
            run_geometry(ctx, 3, 5, 7, 1, 1, 1, 1, 0, 0)
            run_geometry(ctx, 1, 1, 3, 1, 1, 1, 1, 1, 1)

            # Random fp32 operands: operand rounding below fp32 would show.
            run_geometry[False](ctx, 2, 61, 77, 1, 1, 1, 1, 1, 1)
            run_geometry[False](ctx, 1, 40, 53, 2, 1, 1, 2, 1, 2)

            run_old_path[64, 1, False](ctx)
            run_old_path[96, 3, False](ctx)
            run_old_path[96, 1, True](ctx)
