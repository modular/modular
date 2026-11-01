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
"""Smoke test for the MXFP4 / MXFP8 dequantization kernel.

Validates `dequant_mx` on the GPU against a CPU reference, for both formats,
both scale layouts (rank-2 `[N, K // 32]` and the rank-5 SM100 interleaved
layout `[ceildiv(N, 128), ceildiv(K, 128), 32, 4, 4]`), both output dtypes, and
shapes that are not multiples of the 128-row / 128-column scale tile.

The reference builds the E8M0 scale from its definition (the code is the
float32 exponent field), decodes E4M3 itself, and lays out the interleaved
scales from their definition rather than through the library helpers. The
comparison is exact: scaling by a power of two is exact in float32.
"""

from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.mx_dequant import dequant_mx
from linalg.fp4_utils import E2M1_TO_FLOAT32


def _pack_fp4_pair(low: UInt8, high: UInt8) -> UInt8:
    """Packs two 4-bit FP4 values into one uint8 byte (low nibble first)."""
    return (high & UInt8(0x0F)) << UInt8(4) | (low & UInt8(0x0F))


def _e4m3_value(code: UInt8) -> Float32:
    """Decodes a finite E4M3 code from its definition (bias 7, 3 mantissa).

    Codes 0x7F and 0xFF are NaN and are never generated.
    """
    var exp = Int((code >> UInt8(3)) & UInt8(0xF))
    var mant = Int(code & UInt8(0x7))
    var magnitude: Float32
    if exp == 0:
        # Subnormal: mant * 2^(1 - 7 - 3) = mant * 2^-9.
        magnitude = Float32(mant) * Float32(0.001953125)
    else:
        magnitude = (Float32(1.0) + Float32(mant) / Float32(8.0)) * (
            Float32(1 << exp) / Float32(128.0)
        )
    if (code & UInt8(0x80)) != 0:
        return -magnitude
    return magnitude


def _e8m0_value(code: UInt8) -> Float32:
    """2^(code - 127): the E8M0 code is the float32 exponent field."""
    return bitcast[.float32](UInt32(code) << UInt32(23))


def _scale_code(row: Int, block: Int) -> UInt8:
    """A valid E8M0 code in 100..150, so no product over/underflows."""
    return UInt8(100 + (row * 7 + block * 3) % 51)


def _fp8_code(row: Int, col: Int) -> UInt8:
    """A finite E4M3 code: every magnitude 0x00..0x7E, some negative."""
    var code = UInt8((row * 13 + col * 5) % 127)
    if (row + col) % 3 == 0:
        code |= UInt8(0x80)
    return code


def test_mx_dequant[
    num_rows: Int,
    num_cols: Int,
    out_dtype: DType,
    fp8: Bool,
    interleaved: Bool,
](ctx: DeviceContext) raises:
    """Runs the kernel for a compile-time shape."""
    comptime in_dtype = DType.float8_e4m3fn if fp8 else DType.uint8
    comptime in_cols = num_cols if fp8 else num_cols // 2
    comptime scale_cols = num_cols // 32
    comptime row_tiles = ceildiv(num_rows, 128)
    comptime col_tiles = ceildiv(num_cols, 128)
    comptime scales_size = (
        row_tiles
        * col_tiles
        * 32
        * 4
        * 4 if interleaved else num_rows
        * scale_cols
    )

    print(
        t"  fp8= {fp8}  rows= {num_rows}  cols= {num_cols}  dtype= {out_dtype} "
        t" interleaved= {interleaved}"
    )

    comptime in_size = num_rows * in_cols
    comptime out_size = num_rows * num_cols

    var in_host = ctx.enqueue_create_host_buffer[in_dtype](in_size)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        scales_size
    )
    var expected_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)

    # Zero-fill the scales first: the interleaved layout has padding.
    for i in range(scales_size):
        scales_host[i] = bitcast[.float8_e8m0fnu](UInt8(0))

    # Weights.
    for row in range(num_rows):
        for col in range(in_cols):
            comptime if fp8:
                in_host[row * in_cols + col] = bitcast[in_dtype](
                    _fp8_code(row, col)
                )
            else:
                # Every 4-bit code appears, with a row-dependent shift.
                var low = UInt8((col * 2 + row) % 16)
                var high = UInt8((col * 2 + 1 + row * 3) % 16)
                in_host[row * in_cols + col] = bitcast[in_dtype](
                    _pack_fp4_pair(low, high)
                )

    # Scales, in the layout under test.
    for row in range(num_rows):
        for block in range(scale_cols):
            var code = bitcast[.float8_e8m0fnu](_scale_code(row, block))
            comptime if interleaved:
                # [row // 128, block // 4, row % 32, (row % 128) // 32,
                #  block % 4] in a [row_tiles, col_tiles, 32, 4, 4] tensor.
                var idx = (
                    (
                        ((row // 128) * col_tiles + (block // 4)) * 32
                        + (row % 32)
                    )
                    * 4
                    + ((row % 128) // 32)
                ) * 4 + (block % 4)
                scales_host[idx] = code
            else:
                scales_host[row * scale_cols + block] = code

    # CPU reference: element * 2^(code - 127), then the output cast.
    for row in range(num_rows):
        for col in range(num_cols):
            var element: Float32
            comptime if fp8:
                element = _e4m3_value(_fp8_code(row, col))
            else:
                var byte = bitcast[.uint8](in_host[row * in_cols + col // 2])
                var shift = UInt8((col % 2) * 4)
                var nibble = Int((byte >> shift) & UInt8(0x0F))
                element = E2M1_TO_FLOAT32[nibble]
            var scale = _e8m0_value(_scale_code(row, col // 32))
            expected_host[row * num_cols + col] = (element * scale).cast[
                out_dtype
            ]()

    var in_device = ctx.enqueue_create_buffer[in_dtype](in_size)
    var scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](scales_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](out_size)
    ctx.enqueue_copy(in_device, in_host)
    ctx.enqueue_copy(scales_device, scales_host)
    ctx.synchronize()

    var in_tt = TileTensor(in_device, row_major[num_rows, in_cols]())
    var out_tt = TileTensor(out_device, row_major[num_rows, num_cols]())

    comptime if interleaved:
        var scales_tt = TileTensor(
            scales_device,
            row_major(Idx[row_tiles], Idx[col_tiles], Idx[32], Idx[4], Idx[4]),
        )
        dequant_mx["gpu"](
            ctx,
            out_tt,
            in_tt,
            scales_tt,
            num_rows=num_rows,
            num_cols=num_cols,
        )
    else:
        var scales_tt = TileTensor(
            scales_device, row_major[num_rows, scale_cols]()
        )
        dequant_mx["gpu"](
            ctx,
            out_tt,
            in_tt,
            scales_tt,
            num_rows=num_rows,
            num_cols=num_cols,
        )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var num_mismatches = 0
    for i in range(out_size):
        var got = out_host[i].cast[.float32]()
        var exp = expected_host[i].cast[.float32]()
        if got != exp:
            if num_mismatches < 5:
                var row, col = divmod(i, num_cols)
                print(
                    t"    MISMATCH [ {row} , {col} ]: got= {got}  expected="
                    t" {exp}"
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print("    FAIL: ", num_mismatches, " mismatches of ", out_size)
        raise Error("MX dequant test failed")
    print("    PASS")


def _run_all[
    num_rows: Int, num_cols: Int, out_dtype: DType
](ctx: DeviceContext) raises:
    test_mx_dequant[num_rows, num_cols, out_dtype, False, False](ctx)
    test_mx_dequant[num_rows, num_cols, out_dtype, False, True](ctx)
    test_mx_dequant[num_rows, num_cols, out_dtype, True, False](ctx)
    test_mx_dequant[num_rows, num_cols, out_dtype, True, True](ctx)


def main() raises:
    with DeviceContext() as ctx:
        print("MX Dequant Smoke Tests")
        print("======================")

        print("-- float32 output --")
        _run_all[1, 32, .float32](ctx)
        _run_all[64, 64, .float32](ctx)
        _run_all[128, 512, .float32](ctx)
        # Rows and columns that are not multiples of the 128 x 128 scale tile.
        _run_all[37, 96, .float32](ctx)
        _run_all[130, 160, .float32](ctx)
        _run_all[200, 320, .float32](ctx)

        print("-- bfloat16 output --")
        _run_all[64, 64, .bfloat16](ctx)
        _run_all[128, 512, .bfloat16](ctx)
        _run_all[130, 160, .bfloat16](ctx)

        print("-- large shape --")
        _run_all[2880, 2880, .bfloat16](ctx)

        print("======================")
        print("ALL TESTS PASSED")
