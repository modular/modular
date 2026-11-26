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
"""Exact check of the AMD 4-wave conv2d reading an RSCF filter in place.

With a 16-bit dtype and `block_k_override=64`, the dispatcher hands the RSCF
filter to the kernel as stored: K-major LDS tiles, transposing LDS reads, no
FRSC repack launch. Inputs and filters are small integers, so every partial sum
is exact in fp32 and the output must equal a host reference bit for bit, for
whatever order the MFMAs accumulate in. Rows cover a filter column count that
leaves part of the last N tile empty (`F = 72`, `104`), a K that is not a
multiple of two BK tiles (`C = 40`), both BN geometries (128 and 64), and the
static and runtime-HW launch paths.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast

from layout import Idx, TileTensor, row_major

from nn.conv.gpu.amd.dispatch import dispatch_amd_4wave_conv2d

comptime GUARD = 64


def _lcg(mut state: Int) -> Int:
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    return state >> 8


def _run[
    dtype: DType,
    H: Int,
    W: Int,
    C: Int,
    F: Int,
    R: Int,
    pad: Int,
    dynamic: Bool,
](ctx: DeviceContext, label: StaticString) raises:
    comptime OH = H + 2 * pad - R + 1
    comptime OW = W + 2 * pad - R + 1
    comptime NI = H * W * C
    comptime NF = R * R * C * F
    comptime NO = OH * OW * F
    comptime poison = bitcast[dtype](UInt16(0x7E01))

    var input_host = ctx.enqueue_create_host_buffer[dtype](NI)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](NF)
    var output_host = ctx.enqueue_create_host_buffer[dtype](NO + 2 * GUARD)
    var state = 12345 + H + 7 * C + 13 * F
    for i in range(NI):
        input_host[i] = Scalar[dtype](Float32(_lcg(state) % 3) - 1)
    for i in range(NF):
        var sparse = _lcg(state) % 4 == 0
        filter_host[i] = Scalar[dtype](
            Float32(_lcg(state) % 3) - 1 if sparse else Float32(0)
        )
    for i in range(NO + 2 * GUARD):
        output_host[i] = poison

    var input = ctx.enqueue_create_buffer[dtype](NI)
    var filter = ctx.enqueue_create_buffer[dtype](NF)
    var output = ctx.enqueue_create_buffer[dtype](NO + 2 * GUARD)
    ctx.enqueue_copy(input, input_host)
    ctx.enqueue_copy(filter, filter_host)
    ctx.enqueue_copy(output, output_host)

    var filter_tt = TileTensor(
        filter, row_major(Idx[R], Idx[R], Idx[C], Idx[F])
    )
    var strides = (1, 1)
    var dilation = (1, 1)
    var pads = (pad, pad)

    def launch(
        x: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
        y: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    ) raises {imm}:
        var hit = dispatch_amd_4wave_conv2d[
            input_type=dtype,
            filter_type=dtype,
            output_type=dtype,
            filter_is_fcrs=False,
            block_k_override=64,
        ](x, filter_tt, y, strides, dilation, pads, 1, ctx)
        if not hit:
            raise Error(t"dispatch declined {label}")

    comptime if dynamic:
        var n = 1
        var h = H
        var w = W
        var oh = OH
        var ow = OW
        var x = TileTensor(input, row_major(n, h, w, Idx[C]))
        var y = TileTensor(
            output.unsafe_ptr().unsafe_offset(GUARD),
            row_major(n, oh, ow, Idx[F]),
        )
        launch(x, y)
    else:
        var x = TileTensor(input, row_major(Idx[1], Idx[H], Idx[W], Idx[C]))
        var y = TileTensor(
            output.unsafe_ptr().unsafe_offset(GUARD),
            row_major(Idx[1], Idx[OH], Idx[OW], Idx[F]),
        )
        launch(x, y)
    ctx.enqueue_copy(output_host, output)
    ctx.synchronize()

    for i in range(GUARD):
        if bitcast[.uint16](output_host[i]) != UInt16(0x7E01) or bitcast[
            .uint16
        ](output_host[GUARD + NO + i]) != UInt16(0x7E01):
            raise Error(t"output guard corrupted: {label}")

    var filter_f32 = List[Float32](capacity=NF)
    for i in range(NF):
        filter_f32.append(Float32(filter_host[i]))
    var input_f32 = List[Float32](capacity=NI)
    for i in range(NI):
        input_f32.append(Float32(input_host[i]))

    for y in range(OH):
        for x in range(OW):
            for f in range(F):
                var acc = Float32(0)
                for r in range(R):
                    var iy = y - pad + r
                    if iy < 0 or iy >= H:
                        continue
                    for s in range(R):
                        var ix = x - pad + s
                        if ix < 0 or ix >= W:
                            continue
                        var in_base = (iy * W + ix) * C
                        var f_base = ((r * R + s) * C) * F + f
                        for c in range(C):
                            acc += (
                                input_f32[in_base + c]
                                * filter_f32[f_base + c * F]
                            )
                var expected = Float32(Scalar[dtype](acc))
                var actual = Float32(output_host[GUARD + (y * OW + x) * F + f])
                if actual != expected:
                    print(
                        t"mismatch {label} y={y} x={x} f={f} {actual} !="
                        t" {expected}"
                    )
                    raise Error(t"RSCF direct conv mismatch: {label}")
    print(t"PASS {label}")


def main() raises:
    with DeviceContext() as ctx:
        # BN = 128 tile (BM64 for C96 -> 96), static and runtime-HW.
        _run[.bfloat16, 24, 24, 96, 96, 3, 1, False](ctx, "bf16_c96_f96")
        _run[.float16, 24, 24, 96, 96, 3, 1, True](ctx, "f16_c96_f96_dyn")
        # Output channels that leave the last N tile partly empty.
        _run[.bfloat16, 16, 16, 96, 104, 3, 1, False](ctx, "bf16_c96_f104")
        _run[.float16, 16, 16, 192, 72, 3, 1, True](ctx, "f16_c192_f72_dyn")
        # K = 360 pads to 384; rows 360..383 of the filter read as zero.
        _run[.bfloat16, 16, 16, 40, 128, 3, 1, False](ctx, "bf16_c40_f128")
        # C = 160 is a multiple of the 32-column A sub-tile but not of BK, so
        # a K tile can straddle two pixels (the warp-uniform A loader path).
        _run[.bfloat16, 16, 16, 160, 64, 3, 1, False](ctx, "bf16_c160_f64")
        # Two N tiles of 128.
        _run[.float16, 16, 16, 64, 256, 3, 1, True](ctx, "f16_c64_f256_dyn")
        # BN = 64 tile (the C384 rule), half-tiles of 32 columns.
        _run[.bfloat16, 8, 8, 384, 384, 3, 1, False](ctx, "bf16_c384_f384")
        # 1x1: K = C = two BK tiles exactly.
        _run[.bfloat16, 16, 16, 128, 96, 1, 0, False](ctx, "bf16_1x1_c128")
