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
"""Whole-output correctness of the SM100 192 -> 192 channel conv2d route.

The kernel under test is `conv2d_f192_fprop`; `conv_gpu` reaches it for bf16
when `conv2d_f192_supported` accepts the problem and falls through to the
general SM100 kernel otherwise. Operands are small integers drawn from a hash,
so every fp32 partial sum is exact in any accumulation order and the output
must equal the reference bit for bit; a wrong tap, channel block, pixel,
output-channel half, or batch mapping changes the integers.

Covers:
- campaign case 8 (bf16, 120 x 208) and its fp16 twin, whole output, through
  the kernel and (bf16) through `conv_gpu`;
- batch > 1 and an output row of exactly one im2col box, where a CTA's boxes
  cross rows and images;
- shapes `conv2d_f192_supported` must refuse, run through `conv_gpu` so the
  general path's result is checked against the same reference.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import assert_equal

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
from nn.conv.conv import conv_gpu
from nn.conv.gpu.nvidia.sm100.conv2d_f192_fprop import (
    conv2d_f192_fprop,
    conv2d_f192_supported,
)
from nn.conv.gpu.nvidia.sm100.conv_config import Conv2dProblemShape

comptime CIN = 192
comptime KERNEL_KRSC = 0
"""The kernel's own entry point, KRSC filter."""
comptime KERNEL_RSCF = 1
"""The kernel's own entry point, RSCF filter read in place."""
comptime CONV_RSCF = 2
"""`conv_gpu` with the RSCF filter the graph compiler hands over."""
comptime CONV_FCRS = 3
"""`conv_gpu` with an FCRS filter, which the route packs first."""
comptime COUT = 192
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


def _krsc_to_fcrs[
    dt: DType
](
    krsc: HostBuffer[dt],
    mut fcrs: HostBuffer[dt],
    f_dim: Int,
    r_dim: Int,
    s_dim: Int,
    c_dim: Int,
):
    """The FCRS filter `conv_gpu` takes with `filter_is_fcrs`, from KRSC."""
    var krsc_view = TileTensor(krsc, row_major(f_dim, r_dim, s_dim, c_dim))
    var fcrs_view = TileTensor(fcrs, row_major(f_dim, c_dim, r_dim, s_dim))
    for f in range(f_dim):
        for r in range(r_dim):
            for s in range(s_dim):
                for c in range(c_dim):
                    fcrs_view[f, c, r, s] = krsc_view[f, r, s, c]


def _count_mismatches[
    dt: DType
](got: HostBuffer[dt], want: HostBuffer[dt], n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if got[i].cast[.float32]() != want[i].cast[.float32]():
            bad += 1
    return bad


def run_case[
    dt: DType, entry: Int, expect_route: Bool
](ctx: DeviceContext, batch: Int, h: Int, w: Int, name: StringLiteral) raises:
    """Runs one 3x3 pad-1 192 -> 192 problem and compares the whole output.

    Parameters:
        dt: Element type of activation, filter and output.
        entry: Which entry point and filter layout to run.
        expect_route: The value `conv2d_f192_supported` must return.

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
        conv2d_f192_supported[dt](problem, ctx),
        expect_route,
        String(t"conv2d_f192_supported disagrees for {name}"),
    )

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var krsc_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var rscf_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var fcrs_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    _fill_inputs[dt](n_act, n_filt, batch * 31 + h, act_host, krsc_host)
    _krsc_to_rscf(krsc_host, rscf_host, COUT, 3, 3, CIN)
    _krsc_to_fcrs(krsc_host, fcrs_host, COUT, 3, 3, CIN)

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var krsc_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var rscf_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var fcrs_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var ref_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(krsc_dev, krsc_host)
    ctx.enqueue_copy(rscf_dev, rscf_host)
    ctx.enqueue_copy(fcrs_dev, fcrs_host)
    # Poison the output so an unwritten tile cannot match the reference.
    out_dev.enqueue_fill(Scalar[dt](-7))
    ref_dev.enqueue_fill(Scalar[dt](-9))

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, CIN))
    var krsc_tt = TileTensor(krsc_dev, row_major(COUT, 3, 3, CIN))
    var rscf_tt = TileTensor(rscf_dev, row_major(3, 3, CIN, COUT))
    var fcrs_tt = TileTensor(fcrs_dev, row_major(COUT, CIN, 3, 3))
    var out_tt = TileTensor(out_dev, row_major(batch, h, w, COUT))
    var ref_tt = TileTensor(ref_dev, row_major(batch, h, w, COUT))

    comptime match entry:
        case == KERNEL_KRSC:
            conv2d_f192_fprop[dt](out_tt, act_tt, krsc_tt, problem, ctx)
        case == KERNEL_RSCF:
            conv2d_f192_fprop[dt, rscf_filter=True](
                out_tt, act_tt, rscf_tt, problem, ctx
            )
        case == CONV_RSCF:
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
        case _:
            conv_gpu[dt, dt, dt, filter_is_fcrs=True](
                act_tt,
                fcrs_tt,
                out_tt,
                Coord(1, 1),
                Coord(1, 1),
                Coord(1, 1, 1, 1),
                1,
                ctx,
            )

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
    print(t"   {name} {dt} {entry} mismatches: {bad} of {n_out}")
    assert_equal(bad, 0, String(t"whole-output mismatch for {name}"))

    _ = act_dev^
    _ = krsc_dev^
    _ = rscf_dev^
    _ = fcrs_dev^
    _ = out_dev^
    _ = ref_dev^


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    comptime bf16 = DType.bfloat16
    comptime fp16 = DType.float16
    with DeviceContext() as ctx:
        print("--- campaign case 8 (bf16) and case 6 (fp16), 120 x 208 ---")
        run_case[bf16, KERNEL_KRSC, True](ctx, 1, 120, 208, "case8")
        run_case[bf16, KERNEL_RSCF, True](ctx, 1, 120, 208, "case8")
        run_case[bf16, CONV_RSCF, True](ctx, 1, 120, 208, "case8")
        run_case[bf16, CONV_FCRS, True](ctx, 1, 120, 208, "case8")
        run_case[fp16, KERNEL_KRSC, True](ctx, 1, 120, 208, "case6")
        run_case[fp16, KERNEL_RSCF, True](ctx, 1, 120, 208, "case6")

        print(
            "--- fp16 through conv_gpu: RSCF takes the route, FCRS is cuDNN ---"
        )
        run_case[fp16, CONV_RSCF, True](ctx, 1, 120, 208, "case6")
        run_case[fp16, CONV_RSCF, True](ctx, 2, 12, 64, "batch2_12x64")
        run_case[fp16, CONV_FCRS, True](ctx, 1, 120, 208, "case6")
        run_case[fp16, CONV_RSCF, False](ctx, 1, 11, 16, "ragged_m")

        print("--- boxes that cross output rows and images ---")
        run_case[bf16, KERNEL_RSCF, True](ctx, 2, 12, 64, "batch2_12x64")
        run_case[bf16, CONV_RSCF, True](ctx, 2, 12, 64, "batch2_12x64")
        run_case[bf16, CONV_FCRS, True](ctx, 2, 12, 64, "batch2_12x64")
        run_case[fp16, KERNEL_KRSC, True](ctx, 3, 8, 32, "batch3_8x32")
        run_case[fp16, KERNEL_RSCF, True](ctx, 3, 8, 32, "batch3_8x32")
        # One 16-pixel output row per box row.
        run_case[bf16, KERNEL_RSCF, True](ctx, 1, 24, 16, "one_box_row")
        run_case[bf16, CONV_RSCF, True](ctx, 1, 16, 24, "ow24")
        run_case[bf16, KERNEL_RSCF, True](ctx, 1, 48, 40, "ow40")
        run_case[bf16, KERNEL_KRSC, True](ctx, 1, 96, 100, "ow100")
        run_case[bf16, KERNEL_RSCF, True](ctx, 1, 96, 100, "ow100")
        run_case[fp16, KERNEL_RSCF, True](ctx, 2, 8, 48, "batch2_8x48")

        print("--- shapes the route must refuse (fall through) ---")
        # M = 11 * 16 = 176 is not a multiple of 384.
        run_case[bf16, CONV_RSCF, False](ctx, 1, 11, 16, "ragged_m")
        run_case[bf16, CONV_FCRS, False](ctx, 1, 11, 16, "ragged_m")
        # 16 x 16 spatial, M = 256.
        run_case[bf16, CONV_RSCF, False](ctx, 1, 16, 16, "small_16x16")
    print("ALL F192 CONV2D TESTS PASSED")
