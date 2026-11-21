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
"""Convolutions in bf16 whose output row is not 16 bytes, through `conv_gpu`
(B200).

The general SM100 conv2d kernel stores its output through a TMA descriptor
over the `[M, out_c]` GEMM view, and a TMA global stride must be a multiple of
16 bytes. An output of an even channel count that is not a multiple of eight
used to pass the dispatch gate and fail descriptor creation with
`CUDA_ERROR_INVALID_VALUE`, in conv2d and, through the Q-slice route's per-tap
2-D convs, in conv3d. Those convolutions now take the routes after the gate.

The cases cover each of those routes: a few outputs (the small-F kernel where
it exists, the naive kernel otherwise), 16 outputs and more that are not a
multiple of eight (the im2col matmul), an FCRS filter (cuDNN or the im2col
matmul), a fused residual, and conv3d with a QRSCF and an FCQRS filter, each
next to a case on the boundary that keeps the SM100 kernels. The oracle is a
naive GPU kernel that accumulates in fp32 over small integers, so the
convolution is exact; outputs are compared bit for bit with the oracle rounded
to bf16, allowing a fused residual or epilogue add to round before or after
the add.
"""

from std.math import ceildiv
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
from max.gpu.host import DeviceContext
from nn.conv.conv import conv_gpu
from nn.conv.gpu.nvidia.sm100.dispatch import test_alignment_sm100_conv2d

comptime PLAIN = 0
comptime STORE = 1
"""A storing epilogue that adds one."""
comptime RESIDUAL = 2
"""A fused residual add with `beta = 1`."""


def _exact_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = ((i * 2654435761 + salt * 40503 + 7) % 4294967296) >> 9
    return Float32(h % 5 - 2)


def naive_conv3d_kernel[
    dt: DType,
    taps_d: Int,
    taps_hw: Int,
    ActLayout: TensorLayout,
    FiltLayout: TensorLayout,
    DstLayout: TensorLayout,
    act_origin: ImmOrigin,
    filt_origin: ImmOrigin,
    dst_origin: MutOrigin,
](
    act: ImmTileTensor[dt, ActLayout, act_origin],
    filt: ImmTileTensor[dt, FiltLayout, filt_origin],
    dst: MutTileTensor[.float32, DstLayout, dst_origin],
    pad_hw32: Int32,
):
    """One output element per thread: NDHWC activation, QRSCF filter, stride
    1, padding `pad_hw` along H and W and none along D, fp32 accumulation. A
    2-D conv is one frame and one depth tap."""
    comptime assert act.flat_rank == 5 and filt.flat_rank == 5
    comptime assert dst.flat_rank == 1
    var batch = Int(act.dim[0]())
    var in_d = Int(act.dim[1]())
    var in_h = Int(act.dim[2]())
    var in_w = Int(act.dim[3]())
    var cin = Int(act.dim[4]())
    var fout = Int(filt.dim[4]())
    var pad = Int(pad_hw32)
    var out_d = in_d - taps_d + 1
    var out_h = in_h + 2 * pad - taps_hw + 1
    var out_w = in_w + 2 * pad - taps_hw + 1
    var tid = global_idx.x
    if tid >= batch * out_d * out_h * out_w * fout:
        return
    var vox, f = divmod(tid, fout)
    var nd, rem = divmod(vox, out_h * out_w)
    var n, d = divmod(nd, out_d)
    var h, w = divmod(rem, out_w)
    var acc = Float32(0)
    for q in range(taps_d):
        for r in range(taps_hw):
            var hi = h + r - pad
            if hi < 0 or hi >= in_h:
                continue
            for s in range(taps_hw):
                var wi = w + s - pad
                if wi < 0 or wi >= in_w:
                    continue
                for c in range(cin):
                    acc += (
                        act[n, d + q, hi, wi, c].cast[.float32]()
                        * filt[q, r, s, c, f].cast[.float32]()
                    )
    dst[tid] = acc


def _rounds_to(got: Float32, acc: Float32, add: Float32) -> Bool:
    """Whether `got` is `acc + add` in bf16, rounded once or with the
    convolution rounded first."""
    comptime bf = DType.bfloat16
    var once = (acc + add).cast[bf]().cast[.float32]()
    var twice = (
        (acc.cast[bf]().cast[.float32]() + add).cast[bf]().cast[.float32]()
    )
    return got == once or got == twice


def check[
    rank: Int,
    cin: Int,
    fout: Int,
    fcrs: Bool,
    mode: Int,
](ctx: DeviceContext, height: Int, width: Int) raises:
    """One 3x3 (or 3x3x3) convolution through `conv_gpu` against the oracle.

    Parameters:
        rank: 2 (NHWC, padding 0) or 3 (NDHWC, four frames, padding
            `(0, 1, 1)`).
        cin: Input channels.
        fout: Output channels.
        fcrs: The filter is FCRS (FCQRS) rather than RSCF (QRSCF).
        mode: PLAIN, STORE or RESIDUAL.

    Args:
        ctx: Device context.
        height: Input height.
        width: Input width.
    """
    comptime dt = DType.bfloat16
    comptime taps_d = 3 if rank == 3 else 1
    comptime pad = 1 if rank == 3 else 0
    comptime depth = 4 if rank == 3 else 1
    comptime taps = taps_d * 9
    var out_d = depth - taps_d + 1
    var out_h = height + 2 * pad - 2
    var out_w = width + 2 * pad - 2
    var n_act = depth * height * width * cin
    var n_filt = taps * cin * fout
    var n_out = out_d * out_h * out_w * fout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var filt_dev_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var ref_host = ctx.enqueue_create_host_buffer[.float32](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _exact_value(i, 11).cast[dt]()
    for i in range(n_filt):
        filt_host[i] = _exact_value(i, 3).cast[dt]()
    for i in range(n_out):
        src_host[i] = _exact_value(i, 29).cast[dt]()
        out_host[i] = Scalar[dt](-1000)
    # (Q)RSCF -> FC(Q)RS for the other layout.
    comptime if fcrs:
        for k in range(taps):
            for c in range(cin):
                for f in range(fout):
                    filt_dev_host[(f * cin + c) * taps + k] = filt_host[
                        (k * cin + c) * fout + f
                    ]
    else:
        for i in range(n_filt):
            filt_dev_host[i] = filt_host[i]

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_ref_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var src_dev = ctx.enqueue_create_buffer[dt](n_out)
    var ref_dev = ctx.enqueue_create_buffer[.float32](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_ref_dev, filt_host)
    ctx.enqueue_copy(filt_dev, filt_dev_host)
    ctx.enqueue_copy(src_dev, src_host)
    ctx.enqueue_copy(out_dev, out_host)

    var act_ref = TileTensor(
        act_dev.unsafe_ptr(), row_major(1, depth, height, width, cin)
    ).as_imm()
    var filt_ref = TileTensor(
        filt_ref_dev.unsafe_ptr(), row_major[taps_d, 3, 3, cin, fout]()
    ).as_imm()
    var ref_tt = TileTensor(ref_dev.unsafe_ptr(), row_major(n_out))
    comptime naive = naive_conv3d_kernel[
        dt,
        taps_d,
        3,
        act_ref.LayoutType,
        filt_ref.LayoutType,
        ref_tt.LayoutType,
        act_ref.origin,
        filt_ref.origin,
        ref_tt.origin,
    ]
    ctx.enqueue_function[naive](
        act_ref,
        filt_ref,
        ref_tt,
        Int32(pad),
        grid_dim=ceildiv(n_out, 256),
        block_dim=256,
    )
    ctx.enqueue_copy(ref_host, ref_dev)

    comptime if rank == 2:
        var act_tt = TileTensor(
            act_dev.unsafe_ptr(), row_major(1, height, width, cin)
        )
        var out_tt = TileTensor(out_dev, row_major(1, out_h, out_w, fout))
        var out_view = TileTensor(
            out_dev, row_major(1, out_h, out_w, fout)
        ).as_unsafe_any_origin()
        var stride = Coord(1, 1)
        var dilation = Coord(1, 1)
        var padding = Coord(pad, pad, pad, pad)

        @always_inline
        @__allow_legacy_any_origin_fields
        def plus_one_store[
            _dtype: DType, width: SIMDLength, alignment: Int = 1
        ](idx: Coord, val: SIMD[_dtype, width]) {var out_view}:
            out_view.store[width=Int(width)](idx, (val + 1).cast[dt]())

        comptime if fcrs:
            var filt_tt = TileTensor(filt_dev, row_major[fout, cin, 3, 3]())
            comptime if mode == STORE:
                conv_gpu[dt, dt, dt, has_epilogue_fn=True, filter_is_fcrs=True](
                    act_tt,
                    filt_tt,
                    out_tt,
                    stride,
                    dilation,
                    padding,
                    1,
                    ctx,
                    epilogue_fn=plus_one_store,
                )
            elif mode == RESIDUAL:
                conv_gpu[dt, dt, dt, filter_is_fcrs=True, has_residual=True](
                    act_tt,
                    filt_tt,
                    out_tt,
                    stride,
                    dilation,
                    padding,
                    1,
                    ctx,
                    src_dev.unsafe_ptr().as_unsafe_any_origin(),
                    Float32(1.0),
                )
            else:
                conv_gpu[dt, dt, dt, filter_is_fcrs=True](
                    act_tt, filt_tt, out_tt, stride, dilation, padding, 1, ctx
                )
        else:
            var filt_tt = TileTensor(filt_dev, row_major[3, 3, cin, fout]())
            comptime if mode == STORE:
                conv_gpu[dt, dt, dt, has_epilogue_fn=True](
                    act_tt,
                    filt_tt,
                    out_tt,
                    stride,
                    dilation,
                    padding,
                    1,
                    ctx,
                    epilogue_fn=plus_one_store,
                )
            elif mode == RESIDUAL:
                conv_gpu[dt, dt, dt, has_residual=True](
                    act_tt,
                    filt_tt,
                    out_tt,
                    stride,
                    dilation,
                    padding,
                    1,
                    ctx,
                    src_dev.unsafe_ptr().as_unsafe_any_origin(),
                    Float32(1.0),
                )
            else:
                conv_gpu[dt, dt, dt](
                    act_tt, filt_tt, out_tt, stride, dilation, padding, 1, ctx
                )
    else:
        comptime assert mode != RESIDUAL, "a 3-D conv fuses no residual"
        var act_tt = TileTensor(
            act_dev.unsafe_ptr(), row_major(1, depth, height, width, cin)
        )
        var out_tt = TileTensor(
            out_dev, row_major(1, out_d, out_h, out_w, fout)
        )
        var out_view = TileTensor(
            out_dev, row_major(1, out_d, out_h, out_w, fout)
        ).as_unsafe_any_origin()
        var stride = Coord(1, 1, 1)
        var dilation = Coord(1, 1, 1)
        var padding = Coord(0, 0, pad, pad, pad, pad)

        @always_inline
        @__allow_legacy_any_origin_fields
        def plus_one_store3[
            _dtype: DType, width: SIMDLength, alignment: Int = 1
        ](idx: Coord, val: SIMD[_dtype, width]) {var out_view}:
            out_view.store[width=Int(width)](idx, (val + 1).cast[dt]())

        comptime if fcrs:
            comptime assert mode == PLAIN, "an FCQRS filter fuses no epilogue"
            var filt_tt = TileTensor(filt_dev, row_major[fout, cin, 3, 3, 3]())
            conv_gpu[dt, dt, dt, filter_is_fcrs=True](
                act_tt, filt_tt, out_tt, stride, dilation, padding, 1, ctx
            )
        else:
            var filt_tt = TileTensor(filt_dev, row_major[3, 3, 3, cin, fout]())
            comptime if mode == STORE:
                conv_gpu[dt, dt, dt, has_epilogue_fn=True](
                    act_tt,
                    filt_tt,
                    out_tt,
                    stride,
                    dilation,
                    padding,
                    1,
                    ctx,
                    epilogue_fn=plus_one_store3,
                )
            else:
                conv_gpu[dt, dt, dt](
                    act_tt, filt_tt, out_tt, stride, dilation, padding, 1, ctx
                )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        var got = out_host[i].cast[.float32]()
        var add = Float32(0)
        comptime if mode == STORE:
            add = 1
        elif mode == RESIDUAL:
            add = src_host[i].cast[.float32]()
        if not _rounds_to(got, ref_host[i], add):
            bad += 1
    assert_equal(bad, 0)
    var layout = "fcrs" if fcrs else "rscf"
    var epilogue = "plain"
    comptime if mode == STORE:
        epilogue = "store"
    elif mode == RESIDUAL:
        epilogue = "residual"
    print(
        t"PASS bf16 conv{rank}d {cin} -> {fout} {height} x {width}"
        t" {layout} {epilogue}"
    )
    _ = act_dev^
    _ = filt_ref_dev^
    _ = filt_dev^
    _ = src_dev^
    _ = ref_dev^
    _ = out_dev^


def main() raises:
    comptime bf = DType.bfloat16
    # A TMA global stride is a multiple of 16 bytes: eight bf16 channels.
    assert_false(test_alignment_sm100_conv2d[bf, bf](64, 2))
    assert_false(test_alignment_sm100_conv2d[bf, bf](64, 4))
    assert_false(test_alignment_sm100_conv2d[bf, bf](64, 6))
    assert_false(test_alignment_sm100_conv2d[bf, bf](64, 12))
    assert_false(test_alignment_sm100_conv2d[bf, bf](64, 18))
    assert_true(test_alignment_sm100_conv2d[bf, bf](64, 8))
    assert_true(test_alignment_sm100_conv2d[bf, bf](64, 16))
    assert_false(test_alignment_sm100_conv2d[bf, bf](16, 8))

    comptime if not has_accelerator():
        print("no accelerator")
    else:
        with DeviceContext() as ctx:
            # A few outputs: the small-F kernel where it exists, else naive.
            check[2, 64, 2, False, PLAIN](ctx, 31, 29)
            check[2, 64, 4, False, PLAIN](ctx, 31, 29)
            check[2, 64, 4, False, STORE](ctx, 31, 29)
            # Under 16 outputs and more than four: the naive kernel.
            check[2, 64, 6, False, PLAIN](ctx, 31, 29)
            check[2, 64, 10, False, PLAIN](ctx, 31, 29)
            check[2, 64, 12, False, STORE](ctx, 31, 29)
            # 16 outputs and more: the im2col matmul, 36- and 40-byte rows.
            check[2, 64, 18, False, PLAIN](ctx, 31, 29)
            check[2, 64, 18, False, STORE](ctx, 31, 29)
            check[2, 64, 20, False, PLAIN](ctx, 31, 29)
            check[2, 64, 20, False, STORE](ctx, 31, 29)
            # An FCRS filter, as the graph passes it.
            check[2, 64, 4, True, PLAIN](ctx, 31, 29)
            check[2, 64, 4, True, STORE](ctx, 31, 29)
            check[2, 64, 6, True, PLAIN](ctx, 31, 29)
            check[2, 64, 18, True, PLAIN](ctx, 31, 29)
            check[2, 64, 18, True, STORE](ctx, 31, 29)
            # A fused residual.
            check[2, 64, 4, False, RESIDUAL](ctx, 31, 29)
            check[2, 64, 18, False, RESIDUAL](ctx, 31, 29)
            check[2, 64, 4, True, RESIDUAL](ctx, 31, 29)
            # conv3d: the Q-slice route's 2-D convs share the gate.
            check[3, 64, 6, False, PLAIN](ctx, 12, 10)
            check[3, 64, 12, False, STORE](ctx, 12, 10)
            check[3, 64, 18, False, PLAIN](ctx, 12, 10)
            check[3, 64, 6, True, PLAIN](ctx, 12, 10)
            # On the boundary: these keep the SM100 kernels.
            check[2, 64, 8, False, PLAIN](ctx, 31, 29)
            check[2, 64, 16, False, STORE](ctx, 31, 29)
            check[2, 64, 16, False, RESIDUAL](ctx, 31, 29)
            check[2, 64, 16, True, PLAIN](ctx, 31, 29)
            check[3, 64, 16, False, PLAIN](ctx, 12, 10)
