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
"""Runtime-HW correctness for amd_4wave_conv.

The original coverage builds TileTensors with dynamic NHWC layouts, invokes
`amd_4wave_conv` with `use_runtime_hw=True`, and cross-checks the result against
the comptime-HW path.

The C384/F384 regression coverage enters through `conv_gpu` and checks the
complete post-epilogue output against MIOpen so the BM64 tail path is not
compared with another invocation of the same kernel. Runtime and static
controls exercise the automatic BM128 path.

FP16 C192/F96 RSCF coverage checks static and dynamic shapes against an exact
sparse-filter reference, including the padding boundary and output guards.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.random import rand

from layout import Coord, Idx, TileTensor, row_major

from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv
from nn.conv.conv import _conv_miopen, conv_gpu
from std.math import align_up


def _run[
    a_type: DType,
    c_type: DType,
    *,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int = 1,
    stride_w: Int = 1,
](ctx: DeviceContext) raises:
    comptime N = 1
    comptime H_out = (H + 2 * pad_h - R) // stride_h + 1
    comptime W_out = (W + 2 * pad_w - S) // stride_w + 1
    comptime M_total = N * H_out * W_out
    comptime K_real = R * S * C_in
    comptime K_padded = align_up(K_real, 256)

    print(
        t"== rt-hw conv: dtype= {String(a_type)}  H= {H}  W= {W}  C_in= {C_in} "
        t" C_out= {C_out}  RxS= {R} x {S}  stride= {stride_h} x {stride_w} "
        t" pad= {pad_h} x {pad_w}"
    )

    var input_dev = ctx.enqueue_create_buffer[a_type](N * H * W * C_in)
    var filter_dev = ctx.enqueue_create_buffer[a_type](C_out * K_padded)
    var output_static_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)
    var output_rt_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)

    var input_host = ctx.enqueue_create_host_buffer[a_type](N * H * W * C_in)
    var filter_host = ctx.enqueue_create_host_buffer[a_type](C_out * K_padded)
    rand(input_host.unsafe_ptr(), N * H * W * C_in)
    rand(filter_host.unsafe_ptr(), C_out * K_padded)
    for f in range(C_out):
        for k in range(K_real, K_padded):
            filter_host[f * K_padded + k] = 0
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    # ---- Static path (oracle) ----
    comptime static_in_layout = row_major[N, H, W, C_in]()
    comptime filter_layout = row_major[C_out, K_padded]()
    comptime static_out_layout = row_major[M_total, C_out]()
    var input_static = TileTensor(input_dev, static_in_layout)
    var filter = TileTensor(filter_dev, filter_layout)
    var output_static = TileTensor(output_static_dev, static_out_layout)
    amd_4wave_conv[
        H=H,
        W=W,
        H_out=H_out,
        W_out=W_out,
        R=R,
        S=S,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        C_in=C_in,
    ](input_static, filter, output_static, ctx)

    # ---- Runtime-HW path: build TileTensors with DYNAMIC NHWC dims ----
    comptime dyn_in_layout = row_major(Coord(N, H, W, C_in))
    comptime dyn_out_layout = row_major(Coord(M_total, C_out))
    var input_dyn = TileTensor(input_dev, dyn_in_layout)
    var output_rt = TileTensor(output_rt_dev, dyn_out_layout)
    # Note: H/W/H_out/W_out kwargs are *ignored* when use_runtime_hw=True
    # — the kernel reads input.dim() at runtime. R/S/stride/pad/C_in
    # still need to be comptime.
    amd_4wave_conv[
        R=R,
        S=S,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        C_in=C_in,
        use_runtime_hw=True,
    ](input_dyn, filter, output_rt, ctx)

    # ---- Compare byte-identical ----
    var static_host = ctx.enqueue_create_host_buffer[c_type](M_total * C_out)
    var rt_host = ctx.enqueue_create_host_buffer[c_type](M_total * C_out)
    ctx.enqueue_copy(static_host, output_static_dev)
    ctx.enqueue_copy(rt_host, output_rt_dev)
    ctx.synchronize()

    var mismatches = 0
    for i in range(M_total * C_out):
        if static_host[i] != rt_host[i]:
            mismatches += 1
    if mismatches > 0:
        print(t"  FAIL: {mismatches} / {M_total * C_out} differ")
        raise Error("runtime-HW conv != static-HW conv")
    print("  PASS:", M_total * C_out, "elements byte-identical")


def _run_c384_fcrs_parity[
    dtype: DType,
    *,
    H: Int,
    W: Int,
    dynamic_hw: Bool,
    expected_bm: Int,
](ctx: DeviceContext, label: StaticString) raises:
    """Compares an auto-tiled C384 FCRS dispatch with MIOpen."""
    comptime N = 1
    comptime C = 384
    comptime R = 3
    comptime S = 3
    comptime M = N * H * W
    comptime n_input = N * H * W * C
    comptime n_filter = C * C * R * S
    comptime n_output = M * C
    comptime assert expected_bm == (64 if M <= 4096 else 128)

    var input = ctx.enqueue_create_buffer[dtype](n_input)
    var filter = ctx.enqueue_create_buffer[dtype](n_filter)
    var candidate_output = ctx.enqueue_create_buffer[dtype](n_output)
    var reference_output = ctx.enqueue_create_buffer[dtype](n_output)
    var candidate_shape = ctx.enqueue_create_buffer[dtype](n_output)
    var reference_shape = ctx.enqueue_create_buffer[dtype](n_output)
    var input_host = ctx.enqueue_create_host_buffer[dtype](n_input)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](n_filter)
    rand(input_host.unsafe_ptr(), n_input)
    rand(filter_host.unsafe_ptr(), n_filter)
    ctx.enqueue_copy(input, input_host)
    ctx.enqueue_copy(filter, filter_host)

    var filter_tt = TileTensor(filter, row_major[C, C, R, S]())
    var stride = Coord(1, 1)
    var dilation = Coord(1, 1)
    var padding = Coord(1, 1, 1, 1)
    var symmetric_padding = Coord(1, 1)

    @inline(.always)
    def _run(
        input_tt: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
        candidate_shape_tt: TileTensor[
            mut=True, dtype, address_space=.GENERIC, ...
        ],
        candidate_store_tt: TileTensor[
            mut=True, dtype, address_space=.GENERIC, ...
        ],
        reference_shape_tt: TileTensor[
            mut=True, dtype, address_space=.GENERIC, ...
        ],
        reference_store_tt: TileTensor[
            mut=True, dtype, address_space=.GENERIC, ...
        ],
    ) raises {imm}:
        comptime static_n = input_tt.static_shape[0]
        comptime static_h = input_tt.static_shape[1]
        comptime static_w = input_tt.static_shape[2]
        comptime static_c = input_tt.static_shape[3]
        comptime assert static_n == N and static_c == C
        comptime if dynamic_hw:
            comptime assert static_h < 0 and static_w < 0
        else:
            comptime assert static_h == H and static_w == W

        @inline(.always)
        def candidate_store[
            store_dtype: DType,
            width: SIMDLength,
            alignment: Int = 1,
        ](coords: Coord, value: SIMD[store_dtype, width]) {
            var candidate_store_tt
        }:
            candidate_store_tt.store[width=width, alignment=alignment](
                coords, rebind[SIMD[dtype, width]](value)
            )

        @inline(.always)
        def reference_store[
            store_dtype: DType,
            width: SIMDLength,
            alignment: Int = 1,
        ](coords: Coord, value: SIMD[store_dtype, width]) {
            var reference_store_tt
        }:
            reference_store_tt.store[width=width, alignment=alignment](
                coords, rebind[SIMD[dtype, width]](value)
            )

        conv_gpu[
            input_type=dtype,
            filter_type=dtype,
            output_type=dtype,
            has_epilogue_fn=True,
            filter_is_fcrs=True,
        ](
            input_tt,
            filter_tt,
            candidate_shape_tt,
            stride,
            dilation,
            padding,
            1,
            ctx,
            epilogue_fn=candidate_store,
        )
        _conv_miopen[has_epilogue_fn=True, filter_is_fcrs=True](
            input_tt,
            filter_tt,
            reference_shape_tt,
            stride,
            dilation,
            symmetric_padding,
            1,
            ctx,
            reference_store,
        )

        var candidate_host = ctx.enqueue_create_host_buffer[dtype](n_output)
        var reference_host = ctx.enqueue_create_host_buffer[dtype](n_output)
        ctx.enqueue_copy(candidate_host, candidate_store_tt.unsafe_ptr())
        ctx.enqueue_copy(reference_host, reference_store_tt.unsafe_ptr())
        ctx.synchronize()

        var max_abs: Float32 = 0
        var l1_diff: Float32 = 0
        var l1_reference: Float32 = 0
        for i in range(n_output):
            var actual = Float32(candidate_host[i])
            var expected = Float32(reference_host[i])
            var difference = abs(actual - expected)
            max_abs = max(max_abs, difference)
            l1_diff += difference
            l1_reference += abs(expected)
        var l1_relative = (
            l1_diff / l1_reference if l1_reference > 0 else l1_diff
        )
        comptime path = "runtime_hw" if dynamic_hw else "static_hw"
        print(
            t"C384_FCRS_PARITY label={label} path={path} oracle=miopen"
            t" epilogue=store expected_bm={expected_bm} M={M}"
            t" elements={n_output} max_abs={max_abs} l1_relative={l1_relative}"
        )
        if max_abs >= 0.5 and l1_relative >= 0.01:
            raise Error(
                "C384 FCRS post-epilogue whole-output comparison failed"
            )

    comptime if dynamic_hw:
        var runtime_h = H
        var runtime_w = W
        var layout = row_major(Coord(Idx[N], runtime_h, runtime_w, Idx[C]))
        _run(
            TileTensor(input, layout),
            TileTensor(candidate_shape, layout),
            TileTensor(candidate_output, layout),
            TileTensor(reference_shape, layout),
            TileTensor(reference_output, layout),
        )
    else:
        comptime layout = row_major[N, H, W, C]()
        _run(
            TileTensor(input, layout),
            TileTensor(candidate_shape, layout),
            TileTensor(candidate_output, layout),
            TileTensor(reference_shape, layout),
            TileTensor(reference_output, layout),
        )

    _ = input^
    _ = filter^
    _ = candidate_output^
    _ = reference_output^
    _ = candidate_shape^
    _ = reference_shape^


def _sparse_activation(y: Int, x: Int, c: Int) -> Int:
    return (3 * y + 5 * x + 7 * c) % 9 - 4


def _run_sparse_reference[
    static_h: Int,
    static_w: Int,
    filters: Int,
    dynamic: Bool,
    filter_is_fcrs: Bool = False,
    has_epilogue_fn: Bool = False,
    channels: Int = 192,
](ctx: DeviceContext, h: Int, w: Int) raises:
    comptime dtype = DType.float16
    comptime guard = 64
    comptime filter_size = 3 * 3 * channels * filters
    comptime channel_offset = 96 if channels == 2 * filters else 97
    var n = 1
    var input_size = n * h * w * channels
    var output_size = n * h * w * filters
    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_host = ctx.enqueue_create_host_buffer[dtype](
        output_size + 2 * guard
    )
    var input_view = TileTensor(input_host, row_major(h, w, Idx[channels]))
    for y in range(h):
        for x in range(w):
            for c in range(channels):
                input_view[y, x, c] = Scalar[dtype](_sparse_activation(y, x, c))
    comptime filter_layout = row_major(
        Idx[filters if filter_is_fcrs else 3],
        Idx[channels if filter_is_fcrs else 3],
        Idx[3 if filter_is_fcrs else channels],
        Idx[3 if filter_is_fcrs else filters],
    )
    for i in range(filter_size):
        filter_host[i] = 0
    var filter_view = TileTensor(filter_host, filter_layout)
    for f in range(filters):
        var other = (f + channel_offset) % channels
        comptime if filter_is_fcrs:
            filter_view[f, f, 1, 1] = 1
            filter_view[f, other, 0, 0] = 0.5
        else:
            filter_view[1, 1, f, f] = 1
            filter_view[0, 0, other, f] = 0.5
    for i in range(output_size + 2 * guard):
        output_host[i] = bitcast[dtype](UInt16(0x7E01))
    var input = ctx.enqueue_create_buffer[dtype](input_size)
    var filter = ctx.enqueue_create_buffer[dtype](filter_size)
    var output = ctx.enqueue_create_buffer[dtype](output_size + 2 * guard)
    ctx.enqueue_copy(input, input_host)
    ctx.enqueue_copy(filter, filter_host)
    ctx.enqueue_copy(output, output_host)
    var filter_tt = TileTensor(filter, filter_layout)
    var stride = Coord(1, 1)
    var dilation = Coord(1, 1)
    var padding = Coord(1, 1, 1, 1)

    def invoke(
        input_tt: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
        output_tt: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    ) raises {imm}:
        comptime if dynamic:
            comptime assert (
                input_tt.static_shape[0] < 0
                and input_tt.static_shape[1] < 0
                and input_tt.static_shape[2] < 0
            )
        else:
            comptime assert (
                input_tt.static_shape[0] == 1
                and input_tt.static_shape[1] == static_h
                and input_tt.static_shape[2] == static_w
            )
        comptime if has_epilogue_fn:
            var shape_host = ctx.enqueue_create_host_buffer[dtype](
                output_size + 2 * guard
            )
            for i in range(output_size + 2 * guard):
                shape_host[i] = bitcast[dtype](UInt16(0x7E01))
            var shape = ctx.enqueue_create_buffer[dtype](
                output_size + 2 * guard
            )
            ctx.enqueue_copy(shape, shape_host)
            var shape_tt = TileTensor(
                shape.unsafe_ptr().unsafe_offset(guard), output_tt.layout
            )

            def store[
                store_dtype: DType, width: SIMDLength, alignment: Int = 1
            ](coords: Coord, value: SIMD[store_dtype, width]) {var output_tt}:
                output_tt.store[width=width, alignment=alignment](
                    coords,
                    rebind[SIMD[dtype, width]](value) + SIMD[dtype, width](1),
                )

            conv_gpu[
                dtype,
                dtype,
                dtype,
                filter_is_fcrs=filter_is_fcrs,
                has_epilogue_fn=True,
            ](
                input_tt,
                filter_tt,
                shape_tt,
                stride,
                dilation,
                padding,
                1,
                ctx,
                epilogue_fn=store,
            )
            ctx.enqueue_copy(shape_host, shape)
            ctx.synchronize()
            for i in range(output_size + 2 * guard):
                if bitcast[.uint16](shape_host[i]) != UInt16(0x7E01):
                    raise Error("store epilogue wrote shape-only output")
        else:
            conv_gpu[dtype, dtype, dtype, filter_is_fcrs=filter_is_fcrs](
                input_tt,
                filter_tt,
                output_tt,
                stride,
                dilation,
                padding,
                1,
                ctx,
            )

    comptime if dynamic:
        var input_tt = TileTensor(input, row_major(n, h, w, Idx[channels]))
        var output_tt = TileTensor(
            output.unsafe_ptr().unsafe_offset(guard),
            row_major(n, h, w, Idx[filters]),
        )
        invoke(input_tt, output_tt)
    else:
        var input_tt = TileTensor(
            input,
            row_major(Idx[1], Idx[static_h], Idx[static_w], Idx[channels]),
        )
        var output_tt = TileTensor(
            output.unsafe_ptr().unsafe_offset(guard),
            row_major(Idx[1], Idx[static_h], Idx[static_w], Idx[filters]),
        )
        invoke(input_tt, output_tt)
    ctx.enqueue_copy(output_host, output)
    ctx.synchronize()
    for i in range(guard):
        if bitcast[.uint16](output_host[i]) != UInt16(0x7E01) or bitcast[
            .uint16
        ](output_host[guard + output_size + i]) != UInt16(0x7E01):
            raise Error("sparse convolution output guard corrupted")
    for y in range(h):
        for x in range(w):
            for f in range(filters):
                var expected = Float32(_sparse_activation(y, x, f))
                if y > 0 and x > 0:
                    expected += (
                        Float32(
                            _sparse_activation(
                                y - 1, x - 1, (f + channel_offset) % channels
                            )
                        )
                        * 0.5
                    )
                comptime if has_epilogue_fn:
                    expected += 1.0
                var actual = Float32(
                    output_host[guard + (y * w + x) * filters + f]
                )
                if actual != expected:
                    print(
                        t"sparse convolution mismatch: y={y} x={x} f={f}"
                        t" actual={actual} expected={expected}"
                    )
                    raise Error("sparse convolution exact output mismatch")


def main() raises:
    with DeviceContext() as ctx:
        # 1x1 pointwise (BF16)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=256,
            C_out=128,
            R=1,
            S=1,
            pad_h=0,
            pad_w=0,
        ](ctx)
        # 3x3 halo (BF16)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 stride=2 (BF16)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 halo (FP8)
        _run[
            DType.float8_e4m3fn,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # FLUX VAE-style shapes: dynamic H/W (via the runtime-HW path),
        # static C_in/C_out/R/S. Mirrors the conv shapes the production
        # dispatcher routes through `use_runtime_hw=True` after FLUX2's
        # graph compiler lowers symbolic image_height/image_width.
        # 3x3 same-pad, BF16 — 128ch
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 same-pad, BF16 — 256ch
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=256,
            C_out=256,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 same-pad, BF16 — 512ch (FLUX VAE main block size)
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=512,
            C_out=512,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 BF16 — 256→128 transition
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=256,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 stride=2 BF16 — downsample
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=256,
            C_out=512,
            R=3,
            S=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 1x1 pointwise BF16 — 512→128
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=32,
            W=32,
            C_in=512,
            C_out=128,
            R=1,
            S=1,
            pad_h=0,
            pad_w=0,
        ](ctx)
        # 3x3 pad=2 BF16 — non-1-pad
        _run[
            DType.bfloat16,
            DType.bfloat16,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=2,
            pad_w=2,
        ](ctx)
        # Scoped C384/F384 rule: dynamic H/W and M=1560 select BM64.
        _run_c384_fcrs_parity[
            .bfloat16,
            H=30,
            W=52,
            dynamic_hw=True,
            expected_bm=64,
        ](ctx, "runtime_tail_bf16_bm64")
        _run_c384_fcrs_parity[
            .float16,
            H=30,
            W=52,
            dynamic_hw=True,
            expected_bm=64,
        ](ctx, "runtime_tail_f16_bm64")
        _run_c384_fcrs_parity[
            .bfloat16,
            H=30,
            W=52,
            dynamic_hw=False,
            expected_bm=64,
        ](ctx, "static_tail_bf16_bm64")

        # M=5120 is outside the scoped BM64 rule and stays on auto BM128.
        _run_c384_fcrs_parity[
            .bfloat16,
            H=64,
            W=80,
            dynamic_hw=True,
            expected_bm=128,
        ](ctx, "runtime_large_bf16_auto_bm128")
        _run_c384_fcrs_parity[
            .bfloat16,
            H=64,
            W=80,
            dynamic_hw=False,
            expected_bm=128,
        ](ctx, "static_large_bf16_auto_bm128")
        _run_sparse_reference[240, 416, 96, False](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True](ctx, 238, 416)
        _run_sparse_reference[240, 416, 96, False, True](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True, True](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True, True](ctx, 238, 416)
        _run_sparse_reference[240, 416, 96, False, False, True](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True, False, True](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, False, True, True](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True, True, True](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True, True, True](ctx, 238, 416)
        _run_sparse_reference[120, 208, 192, False](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, True](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, True](ctx, 116, 208)
        _run_sparse_reference[120, 208, 192, False, True](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, True, True](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, True, True](ctx, 116, 208)
        _run_sparse_reference[120, 208, 192, False, False, True](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, True, False, True](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, False, True, True](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, True, True, True](ctx, 120, 208)
        _run_sparse_reference[120, 208, 192, True, True, True](ctx, 116, 208)
        # C96/F96 selects BM64/BN128 up to 131072 output rows (M = 99840 here,
        # 98944 on the ragged row) and BM128/BN128 above it (M = 159744).
        _run_sparse_reference[240, 416, 96, False, channels=96](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True, channels=96](ctx, 240, 416)
        _run_sparse_reference[240, 416, 96, True, channels=96](ctx, 238, 416)
        _run_sparse_reference[240, 416, 96, False, True, channels=96](
            ctx, 240, 416
        )
        _run_sparse_reference[240, 416, 96, True, True, channels=96](
            ctx, 240, 416
        )
        _run_sparse_reference[240, 416, 96, False, False, True, channels=96](
            ctx, 240, 416
        )
        _run_sparse_reference[240, 416, 96, True, True, True, channels=96](
            ctx, 238, 416
        )
        _run_sparse_reference[384, 416, 96, False, channels=96](ctx, 384, 416)
        _run_sparse_reference[384, 416, 96, True, channels=96](ctx, 384, 416)
