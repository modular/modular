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
"""Whole-output correctness of the SM100 96-output-channel conv2d route.

The kernel under test is `conv2d_f96_fprop`; `conv_gpu` reaches it for bf16
and fp16 when `conv2d_f96_supported` accepts the problem; otherwise bf16 takes
the general SM100 kernel and fp16 takes cuDNN. Operands are small
integers drawn from a hash, so every fp32 partial sum is exact in any
accumulation order and the output must equal the reference bit for bit; a
wrong tap, channel block, pixel, or batch mapping changes the integers.

Covers:
- 240 x 416 images with 192 and 96 input channels, in fp16 and bf16, whole
  output, through the kernel and through `conv_gpu`;
- batch > 1 and an output row of exactly one im2col box, where a CTA's boxes
  straddle rows and images;
- a Q-slice 3-D convolution whose slices take the route;
- shapes the gates must decline, run through `conv_gpu` so the general
  path's result is checked against the same reference.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true

from layout import (
    Coord,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    row_major,
)
from max.gpu import global_idx
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.conv.conv import conv_gpu
from nn.conv.gpu.nvidia.sm100.qslice_conv3d import (
    dispatch_qslice_conv3d_sm100,
)
from nn.conv.gpu.nvidia.sm100.conv2d_f96_fprop import (
    conv2d_f96_fprop,
    conv2d_f96_supported,
)
from nn.conv.gpu.nvidia.sm100.conv2d import conv2d_fprop
from nn.conv.gpu.nvidia.sm100.conv_config import (
    Conv2dConfig,
    Conv2dProblemShape,
)

comptime COUT = 96
comptime REF_BLOCK = 128


def _hash_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 9) % 5 - 2)


def ref_conv_krsc[
    dt: DType,
    ALayout: TensorLayout,
    FLayout: TensorLayout,
    OLayout: TensorLayout,
](
    act: ImmTileTensor[dt, ALayout, ImmUntrackedOrigin],
    filt: ImmTileTensor[dt, FLayout, ImmUntrackedOrigin],
    dst: MutTileTensor[dt, OLayout, MutUntrackedOrigin],
    pad: Int32,
):
    """One thread per (output pixel, channel): fp32 FMA over the 3x3xC window
    of the NHWC activation against the KRSC filter."""
    comptime assert act.flat_rank == 4
    comptime assert filt.flat_rank == 4
    comptime assert dst.flat_rank == 4
    var batch = Int(act.dim[0]())
    var in_h = Int(act.dim[1]())
    var in_w = Int(act.dim[2]())
    var c_in = Int(act.dim[3]())
    var out_h = Int(dst.dim[1]())
    var out_w = Int(dst.dim[2]())
    var f_dim = Int(filt.dim[0]())
    var m = global_idx.x
    var f = global_idx.y
    if m >= batch * out_h * out_w or f >= f_dim:
        return
    var n, rem = divmod(m, out_h * out_w)
    var h, w = divmod(rem, out_w)
    var acc = Float32(0)
    for r in range(Int(filt.dim[1]())):
        var hi = h + r - Int(pad)
        if hi < 0 or hi >= in_h:
            continue
        for s in range(Int(filt.dim[2]())):
            var wi = w + s - Int(pad)
            if wi < 0 or wi >= in_w:
                continue
            for c in range(c_in):
                acc += (
                    act[n, hi, wi, c].cast[.float32]()
                    * filt[f, r, s, c].cast[.float32]()
                )
    dst[n, h, w, f] = acc.cast[dt]()


def _fill_inputs[
    dt: DType
](
    n_act: Int,
    n_filt: Int,
    salt: Int,
    act_host: HostBuffer[dt],
    filt_host: HostBuffer[dt],
):
    for i in range(n_act):
        act_host[i] = _hash_value(i, salt).cast[dt]()
    for i in range(n_filt):
        filt_host[i] = _hash_value(i, salt + 1).cast[dt]()


def _krsc_to_rscf[
    dt: DType
](
    krsc: HostBuffer[dt],
    mut rscf: HostBuffer[dt],
    f_dim: Int,
    r_dim: Int,
    s_dim: Int,
    c_dim: Int,
):
    """The RSCF filter `conv_gpu` takes, from the KRSC one the kernel takes."""
    var krsc_view = TileTensor(krsc, row_major(f_dim, r_dim, s_dim, c_dim))
    var rscf_view = TileTensor(rscf, row_major(r_dim, s_dim, c_dim, f_dim))
    for f in range(f_dim):
        for r in range(r_dim):
            for s in range(s_dim):
                for c in range(c_dim):
                    rscf_view[r, s, c, f] = krsc_view[f, r, s, c]


def _count_mismatches[
    dt: DType
](got: HostBuffer[dt], want: HostBuffer[dt], n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if got[i].cast[.float32]() != want[i].cast[.float32]():
            bad += 1
    return bad


def run_case[
    dt: DType, via_conv_gpu: Bool, expect_route: Bool, CIN: Int = 192
](ctx: DeviceContext, batch: Int, h: Int, w: Int, name: StringLiteral) raises:
    """Runs one 3x3 pad-1 `CIN` -> 96 problem and compares the whole output.

    Parameters:
        dt: Element type of activation, filter and output.
        via_conv_gpu: Enter through `conv_gpu` (RSCF filter) instead of the
            kernel's own entry point (KRSC filter).
        expect_route: The value `conv2d_f96_supported` must return; `conv_gpu`
            takes the f96 kernel exactly then.
        CIN: Input channels.

    Args:
        ctx: Device context.
        batch: Batch size.
        h: Input height (output height with pad 1).
        w: Input width.
        name: Label printed with the result.
    """
    var n_act = batch * h * w * CIN
    var n_filt = COUT * 9 * CIN
    var n_out = batch * h * w * COUT

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
    assert_equal(
        conv2d_f96_supported[dt](problem, ctx),
        expect_route,
        String(t"kernel selection disagrees for {name}"),
    )

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var krsc_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var rscf_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    _fill_inputs[dt](n_act, n_filt, batch * 31 + h, act_host, krsc_host)
    _krsc_to_rscf(krsc_host, rscf_host, COUT, 3, 3, CIN)

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var krsc_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var rscf_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var ref_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(krsc_dev, krsc_host)
    ctx.enqueue_copy(rscf_dev, rscf_host)
    # Poison the output so an unwritten tile cannot match the reference.
    out_dev.enqueue_fill(Scalar[dt](-7))
    ref_dev.enqueue_fill(Scalar[dt](-9))

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, CIN))
    var krsc_tt = TileTensor(krsc_dev, row_major(COUT, 3, 3, CIN))
    var rscf_tt = TileTensor(rscf_dev, row_major(3, 3, CIN, COUT))
    var out_tt = TileTensor(out_dev, row_major(batch, h, w, COUT))
    var ref_tt = TileTensor(ref_dev, row_major(batch, h, w, COUT))

    comptime if via_conv_gpu:
        conv_gpu[dt, dt, dt](
            act_tt,
            rscf_tt,
            out_tt,
            Coord(1, 1),
            Coord(1, 1),
            Coord(1, 1, 1, 1),
            1,
            ctx,
        )
    else:
        conv2d_f96_fprop[dt](out_tt, act_tt, krsc_tt, problem, ctx)

    comptime ref_kernel = ref_conv_krsc[
        dt,
        type_of(act_tt).LayoutType,
        type_of(krsc_tt).LayoutType,
        type_of(ref_tt).LayoutType,
    ]
    ctx.enqueue_function[ref_kernel](
        act_tt.as_imm(),
        krsc_tt.as_imm(),
        ref_tt,
        Int32(1),
        grid_dim=(ceildiv(batch * h * w, REF_BLOCK), COUT),
        block_dim=(REF_BLOCK, 1),
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()

    var bad = _count_mismatches(out_host, ref_host, n_out)
    comptime route = "conv_gpu" if via_conv_gpu else "kernel"
    print(t"   {name} {dt} {route} mismatches: {bad} of {n_out}")
    assert_equal(bad, 0, String(t"whole-output mismatch for {name}"))

    _ = act_dev^
    _ = krsc_dev^
    _ = rscf_dev^
    _ = out_dev^
    _ = ref_dev^


def run_qslice_case(ctx: DeviceContext) raises:
    """A Q-slice 3-D convolution: each of its `Q` 2-D slices (`D_out` images
    of 48 x 416, 192 -> 96 channels) takes the f96 route. The reference is the
    same slice-by-slice sum: a 2-D reference per slice rounded to bf16, then
    added in fp32 and rounded once, exactly as the dispatcher accumulates."""
    comptime dt = DType.bfloat16
    comptime D = 4
    comptime Q = 3
    comptime D_OUT = D - Q + 1
    comptime H = 48
    comptime W = 416
    comptime C = 192
    comptime n_act = D * H * W * C
    comptime n_slab = 9 * C * COUT
    comptime n_out = D_OUT * H * W * COUT

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var qrscf_host = ctx.enqueue_create_host_buffer[dt](Q * n_slab)
    var krsc_host = ctx.enqueue_create_host_buffer[dt](n_slab)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var slice_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _hash_value(i, 5).cast[dt]()
    # Filter value at (q, f, r, s, c), stored as QRSCF.
    var qrscf_fill = TileTensor(qrscf_host, row_major[Q, 3, 3, C, COUT]())
    for q in range(Q):
        for f in range(COUT):
            for r in range(3):
                for s in range(3):
                    for c in range(C):
                        qrscf_fill[q, r, s, c, f] = _hash_value(
                            (((f * 3 + r) * 3 + s) * C + c), 11 + q
                        ).cast[dt]()

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var qrscf_dev = ctx.enqueue_create_buffer[dt](Q * n_slab)
    var krsc_dev = ctx.enqueue_create_buffer[dt](n_slab)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var slice_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(qrscf_dev, qrscf_host)
    out_dev.enqueue_fill(Scalar[dt](-7))

    var handled = dispatch_qslice_conv3d_sm100(
        TileTensor(act_dev, row_major[1, D, H, W, C]()),
        TileTensor(qrscf_dev, row_major[Q, 3, 3, C, COUT]()),
        TileTensor(out_dev, row_major[1, D_OUT, H, W, COUT]()),
        Coord(1, 1, 1),
        Coord(1, 1, 1),
        Coord(0, 1, 1),
        1,
        ctx,
    )
    assert_true(handled, "the Q-slice dispatcher declined the shape")

    var accum = List[Float32](length=n_out, fill=0)
    for q in range(Q):
        var qrscf_view = TileTensor(qrscf_host, row_major[Q, 3, 3, C, COUT]())
        var krsc_view = TileTensor(krsc_host, row_major[COUT, 3, 3, C]())
        for f in range(COUT):
            for r in range(3):
                for s in range(3):
                    for c in range(C):
                        krsc_view[f, r, s, c] = qrscf_view[q, r, s, c, f]
        ctx.enqueue_copy(krsc_dev, krsc_host)
        # A window that slides by one depth slab per `q`, which a tile of
        # `D_OUT` slabs cannot express.
        var slice_act = TileTensor(
            act_dev.unsafe_ptr().unsafe_offset(q * H * W * C),
            row_major[D_OUT, H, W, C](),
        )
        var krsc_tt = TileTensor(krsc_dev, row_major[COUT, 3, 3, C]())
        var slice_tt = TileTensor(slice_dev, row_major[D_OUT, H, W, COUT]())
        comptime ref_kernel = ref_conv_krsc[
            dt,
            type_of(slice_act).LayoutType,
            type_of(krsc_tt).LayoutType,
            type_of(slice_tt).LayoutType,
        ]
        ctx.enqueue_function[ref_kernel](
            slice_act.as_imm(),
            krsc_tt.as_imm(),
            slice_tt,
            Int32(1),
            grid_dim=(ceildiv(D_OUT * H * W, REF_BLOCK), COUT),
            block_dim=(REF_BLOCK, 1),
        )
        ctx.enqueue_copy(slice_host, slice_dev)
        ctx.synchronize()
        for i in range(n_out):
            accum[i] += slice_host[i].cast[.float32]()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        if out_host[i].cast[.float32]() != accum[i].cast[dt]().cast[.float32]():
            bad += 1
    print(t"   qslice 3x3x3 192->96 mismatches: {bad} of {n_out}")
    assert_equal(bad, 0, "whole-output mismatch for the Q-slice 3-D case")


def run_route_probe[dt: DType, C: Int](ctx: DeviceContext) raises:
    """Fails if `conv_gpu` stops taking the f96 kernel on a supported shape.

    The two kernels add the K tiles in different orders (f96: channel block
    outer, filter tap inner; general: tap outer), and a fp32 sum of 2^24, 1,
    1, -2^24 depends on the order. One output pixel is built from those four
    products with an all-ones filter: the f96 order gives 0 and the general
    order gives 1. The `conv_gpu` result must equal the f96 kernel's and
    differ from the general kernel's.

    The operands are 2^24, 1 and -2^24 as products: bf16 holds them as
    they are, fp16 (largest value 65504) as 2^15 * 2^9, 2^-9 * 2^9 and
    -2^15 * 2^9.

    Parameters:
        dt: Element type, fp16 or bf16.
        C: Input channels, 96 or 192.

    Args:
        ctx: Device context.
    """
    comptime big = 16777216.0 if dt == .bfloat16 else 32768.0
    comptime one = 1.0 if dt == .bfloat16 else 0.001953125
    comptime filt = 1.0 if dt == .bfloat16 else 512.0
    comptime H = 12
    comptime W = 64
    comptime BK = 64 if C == 192 else 32
    comptime n_act = H * W * C
    comptime n_filt = COUT * 9 * C
    comptime n_out = H * W * COUT
    comptime h0 = 5
    comptime w0 = 10

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var ones_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = Scalar[dt](0)
    for i in range(n_filt):
        ones_host[i] = Scalar[dt](filt)
    # Input pixel (h0 - 1, w0 - 1) is filter tap 0 of output (h0, w0), and
    # (h0 - 1, w0) is tap 1.
    var act_pixels = TileTensor(act_host, row_major[H, W, C]())
    act_pixels[h0 - 1, w0 - 1, 0] = Scalar[dt](big)
    act_pixels[h0 - 1, w0 - 1, BK] = Scalar[dt](one)
    act_pixels[h0 - 1, w0 - 1, 2 * BK] = Scalar[dt](-big)
    act_pixels[h0 - 1, w0, 0] = Scalar[dt](one)

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, ones_host)

    var problem = Conv2dProblemShape(
        batch=1,
        in_height=H,
        in_width=W,
        in_channels=C,
        out_channels=COUT,
        filter_h=3,
        filter_w=3,
        pad_h=1,
        pad_w=1,
    )
    var act_tt = TileTensor(act_dev, row_major(1, H, W, C))
    var krsc_tt = TileTensor(filt_dev, row_major(COUT, 3, 3, C))
    var rscf_tt = TileTensor(filt_dev, row_major(3, 3, C, COUT))
    var out_tt = TileTensor(out_dev, row_major(1, H, W, COUT))
    comptime out_shape = row_major[H, W, COUT]()

    conv_gpu[dt, dt, dt](
        act_tt,
        rscf_tt,
        out_tt,
        Coord(1, 1),
        Coord(1, 1),
        Coord(1, 1, 1, 1),
        1,
        ctx,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var via_conv_gpu = TileTensor(out_host, out_shape)[h0, w0, 0].cast[
        .float32
    ]()

    conv2d_f96_fprop[dt](out_tt, act_tt, krsc_tt, problem, ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var f96 = TileTensor(out_host, out_shape)[h0, w0, 0].cast[.float32]()

    comptime general_config = Conv2dConfig[dt, dt, dt].default_bf16_1sm[
        swizzle=TensorMapSwizzle.SWIZZLE_128B if C
        == 192 else TensorMapSwizzle.SWIZZLE_64B,
        num_pipeline_stages_override=0 if C == 192 else 6,
    ]()
    conv2d_fprop[config=general_config](out_tt, act_tt, krsc_tt, problem, ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var general = TileTensor(out_host, out_shape)[h0, w0, 0].cast[.float32]()

    print(
        t"   route probe C = {C}: conv_gpu {via_conv_gpu} f96 {f96} general"
        t" {general}"
    )
    assert_true(f96 != general, "the probe cannot tell the kernels apart")
    assert_equal(via_conv_gpu, f96, "conv_gpu did not take the f96 kernel")

    _ = act_dev^
    _ = filt_dev^
    _ = out_dev^


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    with DeviceContext() as ctx:
        print("--- kernel entry point, whole output ---")
        # One 240 x 416 image, in both element types.
        run_case[.float16, False, True](ctx, 1, 240, 416, "fp16_240x416")
        run_case[.bfloat16, False, True](ctx, 1, 240, 416, "bf16_240x416")
        # Two images: a CTA's three tiles cross the image boundary.
        run_case[.float16, False, True](ctx, 2, 12, 64, "batch2_12x64")
        run_case[.bfloat16, False, True](ctx, 3, 8, 32, "batch3_8x32")
        # One box per output row.
        run_case[.bfloat16, False, True](ctx, 1, 24, 32, "one_box_row")

        print("--- conv_gpu dispatch, bf16 ---")
        run_case[.bfloat16, True, True](ctx, 1, 240, 416, "bf16_240x416")
        # 52 CTAs: just past one wave of the general kernel's tiles.
        run_case[.bfloat16, True, True](ctx, 1, 48, 416, "52_ctas")

        # An output row that is not a multiple of a tile or of 32.
        run_case[.bfloat16, True, True](ctx, 1, 120, 208, "ow208")
        # Small problems take the route too: it is faster than the general
        # kernel at every size.
        run_case[.bfloat16, True, True](ctx, 2, 12, 64, "small_m")

        print("--- shapes the gates must decline (fall through) ---")
        # OW % 32 == 0 but M = 11 * 32 = 352 is not a multiple of 384.
        run_case[.bfloat16, True, False](ctx, 1, 11, 32, "ragged_m")
        # 16 x 16 spatial.
        run_case[.bfloat16, True, False](ctx, 1, 16, 16, "small_16x16")

        print("--- C_in = 96 ---")
        run_case[.bfloat16, False, True, 96](ctx, 1, 240, 416, "bf16_240x416")
        run_case[.float16, False, True, 96](ctx, 1, 240, 416, "fp16_240x416")
        run_case[.bfloat16, False, True, 96](ctx, 2, 12, 64, "batch2_12x64")
        run_case[.bfloat16, False, True, 96](ctx, 1, 24, 32, "one_box_row")
        run_case[.bfloat16, True, True, 96](ctx, 1, 240, 416, "bf16_240x416")
        # Small problems take the route too.
        run_case[.bfloat16, True, True, 96](ctx, 2, 12, 64, "small_m")
        run_case[.bfloat16, True, True, 96](ctx, 1, 120, 208, "ow208")
        run_case[.bfloat16, True, True, 96](ctx, 1, 96, 100, "ow100")
        # 273 work units: two waves but an odd count, so not persistent.
        run_case[.bfloat16, False, True, 96](ctx, 1, 252, 416, "odd_units")
        # Four waves of work units.
        run_case[.bfloat16, False, True, 96](ctx, 2, 240, 416, "batch2")
        run_case[.bfloat16, False, True, 96](ctx, 3, 32, 72, "ow72_b3")
        run_case[.bfloat16, True, False, 96](ctx, 1, 11, 32, "ragged_m")

        print("--- Q-slice 3-D convolution ---")
        run_qslice_case(ctx)

        print("--- conv_gpu dispatch, fp16 ---")
        run_case[.float16, True, True](ctx, 1, 240, 416, "fp16_240x416")
        run_case[.float16, True, True](ctx, 1, 120, 208, "fp16_ow208")
        run_case[.float16, True, True, 96](ctx, 1, 240, 416, "fp16_240x416")
        run_case[.float16, True, True, 96](ctx, 2, 12, 64, "fp16_small_m")
        run_case[.float16, True, False, 96](ctx, 1, 11, 32, "fp16_ragged_m")

        print("--- routing ---")
        run_route_probe[.bfloat16, 192](ctx)
        run_route_probe[.bfloat16, 96](ctx)
        run_route_probe[.float16, 192](ctx)
        run_route_probe[.float16, 96](ctx)
    print("ALL F96 CONV2D TESTS PASSED")
