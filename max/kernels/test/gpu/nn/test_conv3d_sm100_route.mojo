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
"""The SM100 conv3d route of `conv_gpu` (B200): the dispatcher's guard, the
1x1x1 pointwise kernel, and the convolutions the guard must leave alone.

The oracle is a naive GPU kernel that accumulates in fp32. Operands are small
integers drawn from a hash, so every fp32 partial sum is exact in any
accumulation order and each output is compared bit for bit with the oracle's
value rounded to the output dtype, whichever kernel produced it.

Each case builds the tensors as a compiled graph does (the filter's shape
static, the activation's dynamic), asks `dispatch_conv3d_sm100` whether it
takes the convolution and checks the answer against what the guard promises,
then runs `conv_gpu` and compares the output with the oracle. A declined
convolution is therefore checked to still produce the right values on the
path it keeps, to a tolerance where that path rounds partial sums (the
Q-slice route does) and bit for bit otherwise. The 1x1x1 kernel also runs plain, with a storing epilogue and
with a compute epilogue, in both filter layouts.
"""

from std.math import ceildiv, isclose
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
from max.gpu.host import DeviceContext
from nn.conv.conv import conv_gpu
from nn.conv.conv_utils import (
    no_conv_simd_compute_fn,
    no_conv_simd_epilogue_fn,
)
from nn.conv.gpu.nvidia.sm100.conv3d_dispatch import dispatch_conv3d_sm100

comptime PLAIN = 0
comptime STORE = 1
comptime COMPUTE = 2
comptime BIAS = 3
"""A storing epilogue that adds a per-channel bias, as a graph's fused
`out = conv(x) + bias` does."""


def _hash(i: Int, salt: Int) -> Int:
    return ((i * 2654435761 + salt * 40503 + 7) % 4294967296) >> 9


def _exact_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    return Float32(_hash(i, salt) % 5 - 2)


def _scale(c: Coord) -> Float32:
    """A power of two chosen from the coordinate: exact at every dtype."""
    var s = (
        Int(c[1]) + 3 * Int(c[2]) + 5 * Int(c[3]) + 7 * (Int(c[4]) // 8)
    ) % 4
    return Float32(1 << s)


def _lane_scales[
    dtype: DType, width: SIMDLength
](c: Coord) -> SIMD[dtype, width]:
    """`_scale` of each channel a closure call covers, whatever its width."""
    var scales = SIMD[dtype, width]()
    comptime for e in range(Int(width)):
        scales[e] = _scale(
            Coord(Int(c[0]), Int(c[1]), Int(c[2]), Int(c[3]), Int(c[4]) + e)
        ).cast[dtype]()
    return scales


def naive_conv3d_kernel[
    dt: DType,
    taps_d: Int,
    taps_h: Int,
    taps_w: Int,
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
    pad_d32: Int32,
    pad_hw32: Int32,
):
    """One output element per thread: NDHWC activation, QRSCF filter, stride
    1, padding `pad_d` along D and `pad_hw` along H and W, fp32 accumulation."""
    var in_d = Int(act.dim[0]())
    var in_h = Int(act.dim[1]())
    var in_w = Int(act.dim[2]())
    var cin = Int(act.dim[3]())
    var fout = Int(filt.dim[4]())
    var pad_d = Int(pad_d32)
    var pad_hw = Int(pad_hw32)
    var out_d = in_d + 2 * pad_d - taps_d + 1
    var out_h = in_h + 2 * pad_hw - taps_h + 1
    var out_w = in_w + 2 * pad_hw - taps_w + 1
    var tid = global_idx.x
    var total = out_d * out_h * out_w * fout
    if tid >= total:
        return
    var vox, f = divmod(tid, fout)
    var dd, rem = divmod(vox, out_h * out_w)
    var h, w = divmod(rem, out_w)
    var acc = Float32(0)
    for q in range(taps_d):
        var di = dd + q - pad_d
        if di < 0 or di >= in_d:
            continue
        for r in range(taps_h):
            var hi = h + r - pad_hw
            if hi < 0 or hi >= in_h:
                continue
            for s in range(taps_w):
                var wi = w + s - pad_hw
                if wi < 0 or wi >= in_w:
                    continue
                for c in range(cin):
                    acc += (
                        act[di, hi, wi, c].cast[.float32]()
                        * filt[q, r, s, c, f].cast[.float32]()
                    )
    dst[tid] = acc


def check_route[
    dt: DType,
    cin: Int,
    fout: Int,
    taps_d: Int,
    taps_h: Int,
    taps_w: Int,
    pad_d: Int,
    pad_hw: Int,
    fcrs: Bool,
    mode: Int,
](
    ctx: DeviceContext, depth: Int, height: Int, width: Int, expect_taken: Bool
) raises:
    """One convolution through the dispatcher and `conv_gpu`.

    Parameters:
        dt: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        taps_d: Filter depth.
        taps_h: Filter height.
        taps_w: Filter width.
        pad_d: Padding along the depth.
        pad_hw: Padding along the height and the width.
        fcrs: The filter is FCQRS rather than QRSCF.
        mode: PLAIN, STORE (storing epilogue) or COMPUTE (compute epilogue).

    Args:
        ctx: Device context.
        depth: Input depth.
        height: Input height.
        width: Input width.
        expect_taken: Whether the guard must take the convolution.
    """
    comptime taps = taps_d * taps_h * taps_w
    var out_d = depth + 2 * pad_d - taps_d + 1
    var out_h = height + 2 * pad_hw - taps_h + 1
    var out_w = width + 2 * pad_hw - taps_w + 1
    var n_act = depth * height * width * cin
    var n_filt = taps * cin * fout
    var n_out = out_d * out_h * out_w * fout

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var filt_dev_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var ref_host = ctx.enqueue_create_host_buffer[.float32](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    for i in range(n_act):
        act_host[i] = _exact_value(i, 11).cast[dt]()
    for i in range(n_filt):
        filt_host[i] = _exact_value(i, 3).cast[dt]()
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
    var ref_dev = ctx.enqueue_create_buffer[.float32](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_ref_dev, filt_host)
    ctx.enqueue_copy(filt_dev, filt_dev_host)

    var act_ref = TileTensor(
        act_dev.unsafe_ptr(), row_major(depth, height, width, cin)
    ).as_imm()
    var filt_ref = TileTensor(
        filt_ref_dev.unsafe_ptr(),
        row_major[taps_d, taps_h, taps_w, cin, fout](),
    ).as_imm()
    var ref_tt = TileTensor(ref_dev.unsafe_ptr(), row_major(n_out))
    comptime naive = naive_conv3d_kernel[
        dt,
        taps_d,
        taps_h,
        taps_w,
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
        Int32(pad_d),
        Int32(pad_hw),
        grid_dim=ceildiv(n_out, 256),
        block_dim=256,
    )
    ctx.enqueue_copy(ref_host, ref_dev)

    # `conv_gpu` takes its input as mutable but only reads it.
    var act_tt = TileTensor(
        act_dev.unsafe_ptr(), row_major(1, depth, height, width, cin)
    )
    var out_tt = TileTensor(out_dev, row_major(1, out_d, out_h, out_w, fout))
    var out_view = TileTensor(
        out_dev, row_major(1, out_d, out_h, out_w, fout)
    ).as_unsafe_any_origin()
    var stride = Coord(1, 1, 1)
    var dilation = Coord(1, 1, 1)
    var padding = Coord(pad_d, pad_d, pad_hw, pad_hw, pad_hw, pad_hw)
    var sym_padding = Coord(pad_d, pad_hw, pad_hw)

    @always_inline
    @__allow_legacy_any_origin_fields
    def scale_store[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_view}:
        out_view.store[width=Int(width)](
            idx, (val * _lane_scales[_dtype, width](idx)).cast[dt]()
        )

    @always_inline
    def scale_compute[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
        return val * _lane_scales[_dtype, width](idx)

    var bias_host = ctx.enqueue_create_host_buffer[dt](fout)
    ctx.synchronize()
    for f in range(fout):
        bias_host[f] = _exact_value(f, 29).cast[dt]()
    var bias_dev = ctx.enqueue_create_buffer[dt](fout)
    ctx.enqueue_copy(bias_dev, bias_host)
    var bias_tt = TileTensor(bias_dev, row_major(fout))

    @always_inline
    @__allow_legacy_any_origin_fields
    def bias_store[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_view, var bias_tt}:
        var b = bias_tt.load[width=Int(width)]((idx[4],)).cast[_dtype]()
        out_view.store[width=Int(width)](idx, (val + b).cast[dt]())

    out_dev.enqueue_fill(Scalar[dt](-1000))

    comptime if fcrs:
        var filt_tt = TileTensor(
            filt_dev, row_major[fout, cin, taps_d, taps_h, taps_w]()
        )
        # A 3-D conv with an FCQRS filter cannot fuse an epilogue.
        comptime assert mode == PLAIN
        var taken = dispatch_conv3d_sm100[True](
            act_tt, filt_tt, out_tt, stride, dilation, sym_padding, 1, ctx
        )
        assert_equal(taken, expect_taken)
        out_dev.enqueue_fill(Scalar[dt](-1000))
        conv_gpu[dt, dt, dt, filter_is_fcrs=True](
            act_tt, filt_tt, out_tt, stride, dilation, padding, 1, ctx
        )
    else:
        var filt_tt = TileTensor(
            filt_dev, row_major[taps_d, taps_h, taps_w, cin, fout]()
        )
        comptime if mode == STORE:
            var taken = dispatch_conv3d_sm100[False, has_epilogue_fn=True](
                act_tt,
                filt_tt,
                out_tt,
                stride,
                dilation,
                sym_padding,
                1,
                ctx,
                scale_store,
            )
            assert_equal(taken, expect_taken)
            out_dev.enqueue_fill(Scalar[dt](-1000))
            conv_gpu[dt, dt, dt, has_epilogue_fn=True](
                act_tt,
                filt_tt,
                out_tt,
                stride,
                dilation,
                padding,
                1,
                ctx,
                epilogue_fn=scale_store,
            )
        elif mode == BIAS:
            var taken = dispatch_conv3d_sm100[False, has_epilogue_fn=True](
                act_tt,
                filt_tt,
                out_tt,
                stride,
                dilation,
                sym_padding,
                1,
                ctx,
                bias_store,
            )
            assert_equal(taken, expect_taken)
            out_dev.enqueue_fill(Scalar[dt](-1000))
            conv_gpu[dt, dt, dt, has_epilogue_fn=True](
                act_tt,
                filt_tt,
                out_tt,
                stride,
                dilation,
                padding,
                1,
                ctx,
                epilogue_fn=bias_store,
            )
        elif mode == COMPUTE:
            var taken = dispatch_conv3d_sm100[False, has_compute_fn=True](
                act_tt,
                filt_tt,
                out_tt,
                stride,
                dilation,
                sym_padding,
                1,
                ctx,
                compute_fn=scale_compute,
            )
            assert_equal(taken, expect_taken)
            out_dev.enqueue_fill(Scalar[dt](-1000))
            conv_gpu[dt, dt, dt, has_compute_fn=True](
                act_tt,
                filt_tt,
                out_tt,
                stride,
                dilation,
                padding,
                1,
                ctx,
                compute_fn=scale_compute,
            )
        else:
            var taken = dispatch_conv3d_sm100[False](
                act_tt, filt_tt, out_tt, stride, dilation, sym_padding, 1, ctx
            )
            assert_equal(taken, expect_taken)
            out_dev.enqueue_fill(Scalar[dt](-1000))
            conv_gpu[dt, dt, dt](
                act_tt, filt_tt, out_tt, stride, dilation, padding, 1, ctx
            )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var bad = 0
    for i in range(n_out):
        var want = ref_host[i]
        comptime if mode == BIAS:
            var f = i % fout
            want = (
                want.cast[dt]().cast[.float32]() + bias_host[f].cast[.float32]()
            )
        elif mode != PLAIN:
            var vox, f = divmod(i, fout)
            var dd, rem = divmod(vox, out_h * out_w)
            var h, w = divmod(rem, out_w)
            want = want.cast[dt]().cast[.float32]() * _scale(
                Coord(0, dd, h, w, f)
            )
        want = want.cast[dt]().cast[.float32]()
        var got = out_host[i].cast[.float32]()
        # A taken conv is bit-exact. A declined one keeps its path's
        # rounding (the Q-slice route rounds every slab to bf16), exact only
        # while the partial sums fit the output dtype's mantissa, which is
        # the case for the 1x1x1 conv; a deeper reduction gets a tolerance.
        if expect_taken or taps * cin * 4 <= 256:
            if got != want:
                bad += 1
        elif not isclose[symmetrical=False](
            got, want, atol=Float64(taps * cin) * 0.04, rtol=0.02
        ):
            bad += 1
    assert_equal(bad, 0)
    var layout = "fcrs" if fcrs else "qrscf"
    var epilogue = "plain"
    comptime if mode == STORE:
        epilogue = "store"
    elif mode == COMPUTE:
        epilogue = "compute"
    elif mode == BIAS:
        epilogue = "bias"
    var routed = "taken" if expect_taken else "declined"
    print(
        t"PASS {dt} {taps_d} x {taps_h} x {taps_w} {cin} -> {fout} {height} x"
        t" {width} depth {depth} {layout} {epilogue} {routed}"
    )
    _ = act_dev^
    _ = filt_dev^
    _ = filt_ref_dev^
    _ = ref_dev^
    _ = out_dev^
    _ = bias_dev^


def run_dtype[dt: DType](ctx: DeviceContext) raises:
    # The tensor-core arms round fp32 operands to tf32, so an fp32 QRSCF filter
    # stays on the exact fp32 naive kernel (and an FCQRS one runs on them, as
    # it did on cuDNN with tf32 allowed).
    comptime tc = dt != .float32
    # The 1x1x1 pointwise kernel. A bf16 QRSCF filter stays on the matmul
    # route it ties with.
    comptime pw_qrscf = dt != .bfloat16
    check_route[dt, 16, 16, 1, 1, 1, 0, 0, False, PLAIN](
        ctx, 3, 30, 52, pw_qrscf
    )
    check_route[dt, 16, 16, 1, 1, 1, 0, 0, True, PLAIN](ctx, 3, 30, 52, True)
    check_route[dt, 16, 16, 1, 1, 1, 0, 0, False, STORE](
        ctx, 3, 30, 52, pw_qrscf
    )
    check_route[dt, 16, 16, 1, 1, 1, 0, 0, False, COMPUTE](
        ctx, 3, 30, 52, pw_qrscf
    )
    check_route[dt, 16, 16, 1, 1, 1, 0, 0, False, BIAS](
        ctx, 3, 30, 52, pw_qrscf
    )
    # A plane the kernel was not measured on stays on its path.
    check_route[dt, 16, 16, 1, 1, 1, 0, 0, False, PLAIN](ctx, 3, 31, 52, False)
    # A residual-block shape on a plane it was not measured on, a padding the
    # kernel does not take, and channels it does not serve.
    check_route[dt, 96, 96, 3, 3, 3, 0, 1, False, PLAIN](ctx, 4, 64, 64, False)
    check_route[dt, 96, 96, 3, 3, 3, 1, 1, False, PLAIN](
        ctx, 4, 240, 416, False
    )
    check_route[dt, 32, 96, 3, 3, 3, 0, 1, False, PLAIN](
        ctx, 4, 240, 416, False
    )
    # The measured shapes through `conv_gpu`, in either layout and with the
    # fused epilogues. Each is as deep as its kernel needs to beat the old
    # path (see `_deep_enough`).
    check_route[dt, 384, 384, 3, 3, 3, 0, 1, False, PLAIN](ctx, 5, 30, 52, tc)
    # An FCQRS filter replaces cuDNN, which is faster at few frames, so it
    # needs more of them: 21 at 384 channels (the oracle is slow, one dtype).
    check_route[dt, 384, 384, 3, 3, 3, 0, 1, True, PLAIN](ctx, 5, 30, 52, False)
    comptime if dt == .bfloat16:
        check_route[dt, 384, 384, 3, 3, 3, 0, 1, True, PLAIN](
            ctx, 23, 30, 52, True
        )
    check_route[dt, 384, 384, 3, 3, 3, 0, 1, False, STORE](ctx, 8, 30, 52, tc)
    check_route[dt, 192, 384, 3, 1, 1, 0, 0, False, COMPUTE](
        ctx, 6, 60, 104, tc
    )
    check_route[dt, 192, 384, 3, 1, 1, 0, 0, True, PLAIN](
        ctx, 6, 60, 104, False
    )
    check_route[dt, 192, 384, 3, 1, 1, 0, 0, True, PLAIN](
        ctx, 23, 60, 104, True
    )
    check_route[dt, 16, 384, 3, 3, 3, 1, 1, False, STORE](ctx, 6, 30, 52, tc)
    # A bias, as a graph's fused `conv + bias` stores it.
    check_route[dt, 384, 384, 3, 3, 3, 0, 1, False, BIAS](ctx, 6, 30, 52, tc)
    check_route[dt, 192, 384, 3, 1, 1, 0, 0, False, BIAS](ctx, 6, 60, 104, tc)
    check_route[dt, 16, 384, 3, 3, 3, 1, 1, False, BIAS](ctx, 6, 30, 52, tc)
    comptime if tc:
        check_route[dt, 96, 96, 3, 3, 3, 0, 1, False, BIAS](
            ctx, 4, 240, 416, True
        )
        check_route[dt, 192, 192, 3, 3, 3, 0, 1, False, BIAS](
            ctx, 6, 120, 208, True
        )
    check_route[dt, 16, 384, 3, 3, 3, 1, 1, True, PLAIN](ctx, 6, 30, 52, True)
    # Too few output frames for the kernel to beat the old path: a fused
    # epilogue needs more of them than a plain store does.
    check_route[dt, 384, 384, 3, 3, 3, 0, 1, False, STORE](
        ctx, 5, 30, 52, False
    )
    check_route[dt, 192, 384, 3, 1, 1, 0, 0, False, PLAIN](
        ctx, 3, 60, 104, False
    )
    check_route[dt, 192, 384, 3, 1, 1, 0, 0, False, BIAS](
        ctx, 5, 60, 104, False
    )
    check_route[dt, 16, 384, 3, 3, 3, 1, 1, False, BIAS](ctx, 5, 30, 52, False)
    check_route[dt, 16, 384, 3, 3, 3, 1, 1, True, PLAIN](ctx, 3, 30, 52, False)
    comptime if tc:
        check_route[dt, 192, 192, 3, 3, 3, 0, 1, False, BIAS](
            ctx, 4, 120, 208, False
        )
    # A time conv deep enough to need a second wave of CTAs stays on its path.
    check_route[dt, 192, 384, 3, 1, 1, 0, 0, False, PLAIN](
        ctx, 40, 60, 104, False
    )


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator")
    else:
        with DeviceContext() as ctx:
            run_dtype[.bfloat16](ctx)
            run_dtype[.float16](ctx)
            run_dtype[.float32](ctx)
