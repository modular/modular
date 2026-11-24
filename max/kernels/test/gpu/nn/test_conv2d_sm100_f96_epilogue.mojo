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
"""The fused epilogues of the SM100 96 output channel conv2d route.

Each case runs the general SM100 kernel (`conv2d_fprop`, KRSC filter) and the
96 output channel route on the same operands and requires the two outputs to
be equal bit for bit. Operands are small integers drawn from a hash, so every
fp32 partial sum is exact in any accumulation order and the only way for the
two kernels to differ is the epilogue itself: the dtype the accumulator is
cast to, the order of the residual add, the coordinates handed to the closures.

For fp16 through `conv_gpu` the arithmetic runs in fp16 on the rounded
convolution output, as cuDNN followed by an elementwise kernel does, so that
reference is the plain route's output run through a separate elementwise
kernel; the activations are scaled so the convolution output exceeds 2048 and
a bias added before or after rounding gives different bits.

Epilogues: a compute epilogue `relu(acc + bias[channel])`; a storing epilogue
with the same math; a residual `acc + beta * source`; a residual followed by a
storing `relu`. Each runs through the kernel entry point (KRSC filter, 2-D
coordinates) and through `conv_gpu` (RSCF filter, NHWC coordinates), in bf16
and fp16 at 192 and 96 input channels.
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
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.conv.conv import conv_gpu
from nn.conv.gpu.nvidia.sm100.conv2d import (
    conv2d_fprop,
    conv2d_fprop_with_residual,
)
from nn.conv.gpu.nvidia.sm100.conv2d_f96_fprop import conv2d_f96_fprop
from nn.conv.gpu.nvidia.sm100.conv_config import (
    Conv2dConfig,
    Conv2dProblemShape,
)

comptime COUT = 96

comptime COMPUTE = 0
comptime STORE = 1
comptime RESIDUAL = 2
comptime RESIDUAL_STORE = 3

comptime KERNEL = 0
comptime CONV_GPU = 1


def _hash_value(i: Int, salt: Int) -> Float32:
    """A deterministic pseudo-random integer in [-2, 2]."""
    var h = (i * 2654435761 + salt * 40503 + 7) % 4294967296
    return Float32((h >> 9) % 5 - 2)


def _fill[dt: DType](buf: HostBuffer[dt], n: Int, salt: Int, scale: Int = 1):
    for i in range(n):
        buf[i] = (_hash_value(i, salt) * Float32(scale)).cast[dt]()


def ref_epilogue_kernel[
    dt: DType, mode: Int, MLayout: TensorLayout, BiasLayout: TensorLayout
](
    conv: ImmTileTensor[dt, MLayout, ImmUntrackedOrigin],
    bias: ImmTileTensor[dt, BiasLayout, ImmUntrackedOrigin],
    src: ImmTileTensor[dt, MLayout, ImmUntrackedOrigin],
    dst: MutTileTensor[dt, MLayout, MutUntrackedOrigin],
    beta: Float32,
):
    """The epilogue as a separate elementwise kernel on the rounded
    convolution output, in `dt` arithmetic. The tensors are `[pixels, COUT]`
    views."""
    var pixel, ch = divmod(global_idx.x, COUT)
    if pixel >= Int(conv.dim[0]()):
        return
    var c = conv[pixel, ch]
    var v: Scalar[dt]
    comptime if mode == COMPUTE or mode == STORE:
        v = max(c + bias[ch], Scalar[dt](0))
    elif mode == RESIDUAL:
        v = c + Scalar[dt](beta) * src[pixel, ch]
    else:
        v = max(c + Scalar[dt](beta) * src[pixel, ch], Scalar[dt](0))
    dst[pixel, ch] = v


def _count_mismatches[
    dt: DType
](got: HostBuffer[dt], want: HostBuffer[dt], n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if got[i].cast[.float32]() != want[i].cast[.float32]():
            bad += 1
    return bad


def run_case[
    dt: DType, CIN: Int, mode: Int, entry: Int
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
        dt: Element type, fp16 or bf16.
        CIN: Input channels, 96 or 192.
        mode: Which epilogue (`COMPUTE`, `STORE`, `RESIDUAL`,
            `RESIDUAL_STORE`).
        entry: `KERNEL` for `conv2d_f96_fprop` with a KRSC filter, `CONV_GPU`
            for `conv_gpu` with an RSCF filter.

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

    # fp16 through `conv_gpu` rounds the convolution first; scale the data so
    # that matters.
    comptime two_stage = dt == .float16 and entry == CONV_GPU
    comptime act_scale = 9 if two_stage else 1
    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var krsc_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var rscf_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var bias_host = ctx.enqueue_create_host_buffer[dt](COUT)
    var src_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var out_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var ref_host = ctx.enqueue_create_host_buffer[dt](n_out)
    ctx.synchronize()
    _fill(act_host, n_act, batch * 31 + h, act_scale)
    _fill(krsc_host, n_filt, batch * 31 + h + 1)
    _fill(bias_host, COUT, 5)
    _fill(src_host, n_out, 9)
    for f in range(COUT):
        for r in range(3):
            for s in range(3):
                for c in range(CIN):
                    rscf_host[((r * 3 + s) * CIN + c) * COUT + f] = krsc_host[
                        ((f * 3 + r) * 3 + s) * CIN + c
                    ]

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var krsc_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var route_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var bias_dev = ctx.enqueue_create_buffer[dt](COUT)
    var src_dev = ctx.enqueue_create_buffer[dt](n_out)
    var out_dev = ctx.enqueue_create_buffer[dt](n_out)
    var ref_dev = ctx.enqueue_create_buffer[dt](n_out)
    var plain_dev = ctx.enqueue_create_buffer[dt](n_out)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(krsc_dev, krsc_host)
    comptime if entry == CONV_GPU:
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
    var route_tt = TileTensor(
        route_dev,
        row_major(3, 3, CIN, COUT) if entry
        == CONV_GPU else row_major(COUT, 3, 3, CIN),
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
    var out_4d = TileTensor(out_p, row_major(batch, h, w, COUT))
    # `conv_gpu` takes its residual source as an any-origin pointer.
    var src_ptr = src_dev.unsafe_ptr().as_unsafe_any_origin()

    comptime config = Conv2dConfig[dt, dt, dt].default_bf16_1sm[
        swizzle=TensorMapSwizzle.SWIZZLE_128B if CIN
        == 192 else TensorMapSwizzle.SWIZZLE_64B,
        num_pipeline_stages_override=0 if CIN == 192 else 6,
    ]()

    # ----- closures with (pixel, channel) coordinates -----

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
    @__allow_legacy_any_origin_fields
    def relu_ref[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var ref_2d}:
        comptime assert idx.rank == 2, "the epilogue contract is rank 2"
        ref_2d.store[width=Int(width), alignment=alignment](
            idx, max(val, SIMD[_dtype, width](0)).cast[dt]()
        )

    @always_inline
    @__allow_legacy_any_origin_fields
    def relu_out[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_2d}:
        comptime assert idx.rank == 2, "the epilogue contract is rank 2"
        out_2d.store[width=Int(width), alignment=alignment](
            idx, max(val, SIMD[_dtype, width](0)).cast[dt]()
        )

    # ----- closures with NHWC coordinates (what `conv_gpu` takes) -----

    @always_inline
    @__allow_legacy_any_origin_fields
    def bias_relu_nhwc[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt} -> SIMD[
        _dtype, width
    ]:
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        return max(val + b, SIMD[_dtype, width](0))

    @always_inline
    @__allow_legacy_any_origin_fields
    def store_relu_nhwc[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var bias_tt, var out_4d}:
        var b = bias_tt.load[width=width]((idx[3],)).cast[_dtype]()
        out_4d.store[width=Int(width), alignment=alignment](
            idx, max(val + b, SIMD[_dtype, width](0)).cast[dt]()
        )

    @always_inline
    @__allow_legacy_any_origin_fields
    def relu_nhwc[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[_dtype, width]) {var out_4d}:
        out_4d.store[width=Int(width), alignment=alignment](
            idx, max(val, SIMD[_dtype, width](0)).cast[dt]()
        )

    # ----- the reference -----
    comptime if two_stage:
        var plain_tt = TileTensor(plain_dev, row_major(batch, h, w, COUT))
        conv_gpu[dt, dt, dt](
            act_tt,
            route_tt,
            plain_tt,
            Coord(1, 1),
            Coord(1, 1),
            Coord(1, 1, 1, 1),
            1,
            ctx,
        )
        var plain_2d = TileTensor(plain_dev, row_major(m_total, COUT))
        var src_2d = TileTensor(src_dev, row_major(m_total, COUT))
        comptime ref_kernel = ref_epilogue_kernel[
            dt,
            mode,
            type_of(plain_2d).LayoutType,
            type_of(bias_tt).LayoutType,
        ]
        ctx.enqueue_function[ref_kernel](
            plain_2d.as_imm(),
            bias_tt.as_imm(),
            src_2d.as_imm(),
            ref_2d,
            beta,
            grid_dim=ceildiv(n_out, 256),
            block_dim=256,
        )
    elif mode == COMPUTE:
        conv2d_fprop[config=config](
            ref_tt, act_tt, krsc_tt, problem, bias_relu, ctx
        )
    elif mode == STORE:
        conv2d_fprop[config=config](
            ref_tt, act_tt, krsc_tt, problem, store_relu_ref, ctx
        )
    elif mode == RESIDUAL:
        conv2d_fprop_with_residual[config=config, has_residual=True](
            ref_tt, act_tt, krsc_tt, src_tt, beta, problem, ctx
        )
    else:
        conv2d_fprop_with_residual[
            config=config, has_residual=True, has_epilogue_fn=True
        ](ref_tt, act_tt, krsc_tt, src_tt, beta, problem, ctx, relu_ref)

    # ----- the route -----
    comptime if entry == KERNEL:
        comptime match mode:
            case == COMPUTE:
                conv2d_f96_fprop[dt, has_compute_fn=True](
                    out_tt, act_tt, route_tt, problem, ctx, compute_fn=bias_relu
                )
            case == STORE:
                conv2d_f96_fprop[dt, has_epilogue_fn=True](
                    out_tt,
                    act_tt,
                    route_tt,
                    problem,
                    ctx,
                    epilogue_fn=store_relu_out,
                )
            case == RESIDUAL:
                conv2d_f96_fprop[dt, has_residual=True](
                    out_tt, act_tt, route_tt, problem, ctx, src_ptr, beta
                )
            case _:
                conv2d_f96_fprop[dt, has_epilogue_fn=True, has_residual=True](
                    out_tt,
                    act_tt,
                    route_tt,
                    problem,
                    ctx,
                    src_ptr,
                    beta,
                    epilogue_fn=relu_out,
                )
    else:
        comptime match mode:
            case == COMPUTE:
                conv_gpu[dt, dt, dt, has_compute_fn=True](
                    act_tt,
                    route_tt,
                    out_tt,
                    Coord(1, 1),
                    Coord(1, 1),
                    Coord(1, 1, 1, 1),
                    1,
                    ctx,
                    compute_fn=bias_relu_nhwc,
                )
            case == STORE:
                conv_gpu[dt, dt, dt, has_epilogue_fn=True](
                    act_tt,
                    route_tt,
                    out_tt,
                    Coord(1, 1),
                    Coord(1, 1),
                    Coord(1, 1, 1, 1),
                    1,
                    ctx,
                    epilogue_fn=store_relu_nhwc,
                )
            case == RESIDUAL:
                conv_gpu[dt, dt, dt, has_residual=True](
                    act_tt,
                    route_tt,
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
                conv_gpu[dt, dt, dt, has_epilogue_fn=True, has_residual=True](
                    act_tt,
                    route_tt,
                    out_tt,
                    Coord(1, 1),
                    Coord(1, 1),
                    Coord(1, 1, 1, 1),
                    1,
                    ctx,
                    src_ptr,
                    beta,
                    epilogue_fn=relu_nhwc,
                )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()
    var bad = _count_mismatches(out_host, ref_host, n_out)
    comptime route = "conv_gpu" if entry == CONV_GPU else "kernel"
    print(
        t"   {name} {dt} C = {CIN} mode {mode} {route} mismatches: {bad} of"
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
    _ = plain_dev^


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    comptime bf16 = DType.bfloat16
    comptime fp16 = DType.float16
    with DeviceContext() as ctx:
        print("--- kernel entry point, bf16 192 -> 96 ---")
        run_case[bf16, 192, COMPUTE, KERNEL](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[bf16, 192, STORE, KERNEL](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[bf16, 192, RESIDUAL, KERNEL](ctx, 1, 120, 208, 1.0, "ow208")
        run_case[bf16, 192, RESIDUAL_STORE, KERNEL](
            ctx, 2, 12, 64, 0.5, "batch2"
        )
        # 260 work units: persistent CTAs on a 148 SM part.
        run_case[bf16, 192, COMPUTE, KERNEL](ctx, 1, 240, 416, 0.0, "persist")
        run_case[bf16, 192, RESIDUAL_STORE, KERNEL](
            ctx, 1, 240, 416, 1.0, "persist"
        )

        print("--- kernel entry point, fp16 96 -> 96 ---")
        run_case[fp16, 96, COMPUTE, KERNEL](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[fp16, 96, STORE, KERNEL](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[fp16, 96, RESIDUAL, KERNEL](ctx, 1, 120, 208, 0.5, "ow208")
        run_case[fp16, 96, RESIDUAL_STORE, KERNEL](
            ctx, 2, 12, 64, 1.0, "batch2"
        )

        print("--- conv_gpu, RSCF filter ---")
        run_case[bf16, 96, COMPUTE, CONV_GPU](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[bf16, 96, STORE, CONV_GPU](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[bf16, 96, RESIDUAL, CONV_GPU](ctx, 1, 120, 208, 1.0, "ow208")
        run_case[bf16, 192, RESIDUAL_STORE, CONV_GPU](
            ctx, 2, 12, 64, 0.5, "batch2"
        )
        run_case[fp16, 192, COMPUTE, CONV_GPU](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[fp16, 192, STORE, CONV_GPU](ctx, 1, 120, 208, 0.0, "ow208")
        run_case[fp16, 192, RESIDUAL, CONV_GPU](ctx, 1, 120, 208, 1.0, "ow208")
        run_case[fp16, 96, RESIDUAL_STORE, CONV_GPU](
            ctx, 2, 12, 64, 0.5, "batch2"
        )
    print("ALL F96 EPILOGUE TESTS PASSED")
