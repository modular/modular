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
"""Whole-output correctness of the SM100 1x5 filter (Conv1D) conv2d route.

The kernel under test is `conv2d_k1x5_fprop`; `conv_gpu` reaches it for bf16
and fp16 when `conv2d_k1x5_supported` accepts the problem and falls through to
the general SM100 kernel (bf16) or cuDNN (fp16) otherwise. Operands are small integers drawn from a hash,
so every fp32 partial sum is exact in any accumulation order and the output
must equal the reference bit for bit; a wrong tap, channel block, pixel, or
batch mapping changes the integers.

Covers:
- the campaign shape, 1 x 550 with 512 -> 1024 channels, fp16 (case 15) and
  bf16 (case 17), whole output, through the kernel and through `conv_gpu`
  with either filter layout;
- a tile that holds one output pixel, a width that is not a multiple of the
  im2col box, and batch and height above one, where a CTA's boxes straddle
  rows and images;
- shapes `conv2d_k1x5_supported` must refuse, run through `conv_gpu` so the
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
from nn.conv.gpu.nvidia.sm100.conv2d_k1x5_fprop import (
    conv2d_k1x5_fprop,
    conv2d_k1x5_supported,
)
from nn.conv.gpu.nvidia.sm100.conv_config import Conv2dProblemShape

comptime S = 5
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
):
    """One thread per (output pixel, channel): fp32 FMA over the 1x5xC window
    of the NHWC activation against the KRSC filter."""
    comptime assert act.flat_rank == 4
    comptime assert filt.flat_rank == 4
    comptime assert dst.flat_rank == 4
    var batch = Int(act.dim[0]())
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
    for s in range(S):
        for c in range(c_in):
            acc += (
                act[n, h, w + s, c].cast[.float32]()
                * filt[f, 0, s, c].cast[.float32]()
            )
    dst[n, h, w, f] = acc.cast[dt]()


def _krsc_to_rscf[
    dt: DType
](krsc: HostBuffer[dt], mut rscf: HostBuffer[dt], f_dim: Int, c_dim: Int):
    """The RSCF filter `conv_gpu` takes, from the KRSC one the kernel takes."""
    var krsc_view = TileTensor(krsc, row_major(f_dim, S, c_dim))
    var rscf_view = TileTensor(rscf, row_major(S, c_dim, f_dim))
    for f in range(f_dim):
        for s in range(S):
            for c in range(c_dim):
                rscf_view[s, c, f] = krsc_view[f, s, c]


def _krsc_to_fcrs[
    dt: DType
](krsc: HostBuffer[dt], mut fcrs: HostBuffer[dt], f_dim: Int, c_dim: Int):
    """The FCRS filter `conv_gpu` takes with `filter_is_fcrs`, from KRSC."""
    var krsc_view = TileTensor(krsc, row_major(f_dim, S, c_dim))
    var fcrs_view = TileTensor(fcrs, row_major(f_dim, c_dim, S))
    for f in range(f_dim):
        for c in range(c_dim):
            for s in range(S):
                fcrs_view[f, c, s] = krsc_view[f, s, c]


def _count_mismatches[
    dt: DType
](got: HostBuffer[dt], want: HostBuffer[dt], n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if got[i].cast[.float32]() != want[i].cast[.float32]():
            bad += 1
    return bad


def run_case[
    dt: DType, via_conv_gpu: Bool, expect_route: Bool, fcrs: Bool = False
](
    ctx: DeviceContext,
    batch: Int,
    h: Int,
    w: Int,
    cin: Int,
    cout: Int,
    name: StringLiteral,
) raises:
    """Runs one 1x5 pad-0 `cin` -> `cout` problem and compares the whole
    output.

    Parameters:
        dt: Element type of activation, filter and output.
        via_conv_gpu: Enter through `conv_gpu` (RSCF filter) instead of the
            kernel's own entry point (KRSC filter).
        expect_route: The value `conv2d_k1x5_supported` must return.
        fcrs: With `via_conv_gpu`, hand `conv_gpu` an FCRS filter.

    Args:
        ctx: Device context.
        batch: Batch size.
        h: Input height (output height).
        w: Input width (output width plus four).
        cin: Input channels.
        cout: Output channels.
        name: Label printed with the result.
    """
    var ow = w - (S - 1)
    var n_act = batch * h * w * cin
    var n_filt = cout * S * cin
    var n_out = batch * h * ow * cout

    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=h,
        in_width=w,
        in_channels=cin,
        out_channels=cout,
        filter_h=1,
        filter_w=S,
        pad_h=0,
        pad_w=0,
    )
    assert_equal(
        conv2d_k1x5_supported[dt](problem, ctx),
        expect_route,
        String(t"conv2d_k1x5_supported disagrees for {name}"),
    )

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var krsc_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var rscf_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var fcrs_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    var salt = batch * 31 + w
    for i in range(n_act):
        act_host[i] = _hash_value(i, salt).cast[dt]()
    for i in range(n_filt):
        krsc_host[i] = _hash_value(i, salt + 1).cast[dt]()
    _krsc_to_rscf(krsc_host, rscf_host, cout, cin)
    _krsc_to_fcrs(krsc_host, fcrs_host, cout, cin)

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

    var act_tt = TileTensor(act_dev, row_major(batch, h, w, cin))
    var krsc_tt = TileTensor(krsc_dev, row_major(cout, 1, S, cin))
    var rscf_tt = TileTensor(rscf_dev, row_major(1, S, cin, cout))
    var fcrs_tt = TileTensor(fcrs_dev, row_major(cout, cin, 1, S))
    var out_tt = TileTensor(out_dev, row_major(batch, h, ow, cout))
    var ref_tt = TileTensor(ref_dev, row_major(batch, h, ow, cout))

    comptime if via_conv_gpu and fcrs:
        conv_gpu[dt, dt, dt, filter_is_fcrs=True](
            act_tt,
            fcrs_tt,
            out_tt,
            Coord(1, 1),
            Coord(1, 1),
            Coord(0, 0, 0, 0),
            1,
            ctx,
        )
    elif via_conv_gpu:
        conv_gpu[dt, dt, dt](
            act_tt,
            rscf_tt,
            out_tt,
            Coord(1, 1),
            Coord(1, 1),
            Coord(0, 0, 0, 0),
            1,
            ctx,
        )
    else:
        conv2d_k1x5_fprop[dt](out_tt, act_tt, krsc_tt, problem, ctx)

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
        grid_dim=(ceildiv(batch * h * ow, REF_BLOCK), cout),
        block_dim=(REF_BLOCK, 1),
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()

    var bad = _count_mismatches(out_host, ref_host, n_out)
    comptime route = (
        ("conv_gpu_fcrs" if fcrs else "conv_gpu") if via_conv_gpu else "kernel"
    )
    print(t"   {name} {dt} {route} mismatches: {bad} of {n_out}")
    assert_equal(bad, 0, String(t"whole-output mismatch for {name}"))

    _ = act_dev^
    _ = krsc_dev^
    _ = rscf_dev^
    _ = fcrs_dev^
    _ = out_dev^
    _ = ref_dev^


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    with DeviceContext() as ctx:
        print("--- kernel entry point, whole output ---")
        # Campaign cases 15 (fp16) and 17 (bf16): 1 x 550, 546 output pixels,
        # so the last of nine tiles holds 34.
        run_case[.float16, False, True](ctx, 1, 1, 550, 512, 1024, "case15")
        run_case[.bfloat16, False, True](ctx, 1, 1, 550, 512, 1024, "case17")
        # One output pixel: the other 63 rows of the tile are dropped.
        run_case[.bfloat16, False, True](ctx, 1, 1, 5, 512, 1024, "one_px")
        # Output width 34 is not a multiple of the 16-pixel box, so boxes
        # straddle rows; three rows, two images.
        run_case[.bfloat16, False, True](ctx, 2, 3, 38, 512, 1024, "rows")
        run_case[.float16, False, True](ctx, 1, 5, 21, 512, 1024, "rows16")

        print("--- conv_gpu dispatch, bf16 ---")
        run_case[.bfloat16, True, True](ctx, 1, 1, 550, 512, 1024, "case17")
        run_case[.bfloat16, True, True](ctx, 2, 3, 38, 512, 1024, "rows")

        print("--- conv_gpu dispatch, fp16 ---")
        run_case[.float16, True, True](ctx, 1, 1, 550, 512, 1024, "case15")
        run_case[.float16, True, True, True](
            ctx, 1, 1, 550, 512, 1024, "case15"
        )

        print("--- conv_gpu dispatch, bf16, FCRS filter ---")
        run_case[.bfloat16, True, True, True](
            ctx, 1, 1, 550, 512, 1024, "case17"
        )
        run_case[.bfloat16, True, True, True](ctx, 2, 3, 38, 512, 1024, "rows")

        print("--- shapes the route must refuse (fall through) ---")
        # 1096 output pixels: past the one-wave grid the route is shaped for.
        run_case[.bfloat16, True, False](ctx, 1, 1, 1100, 512, 1024, "wide")
        # Other channel counts run the general kernel.
        run_case[.bfloat16, True, False](ctx, 1, 1, 134, 256, 1024, "c256")
        run_case[.bfloat16, True, False](ctx, 1, 1, 134, 512, 512, "f512")
        run_case[.bfloat16, True, False, True](
            ctx, 1, 1, 134, 256, 1024, "c256_fcrs"
        )
        run_case[.float16, True, False](ctx, 1, 1, 134, 256, 1024, "c256")
    print("ALL K1X5 CONV2D TESTS PASSED")
