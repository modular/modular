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
"""Smoke test for the NVFP4 dequantization kernel.

Validates `dequant_nvfp4` on the GPU against a CPU reference, for both scale
layouts (rank-2 `[N, K // 16]` and the rank-5 SM100 interleaved layout), both
output dtypes, and shapes that are not multiples of the 128-row / 64-column
scale tile. The reference decodes E4M3 and lays out the interleaved scales from
their definitions rather than through the library helpers, and the comparison
is exact: `(e2m1 * |scale|) * tensor_sf` is the same sequence of IEEE
operations on both sides.
"""

from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.nvfp4_dequant import dequant_nvfp4
from linalg.fp4_utils import E2M1_TO_FLOAT32


def _pack_fp4_pair(low: UInt8, high: UInt8) -> UInt8:
    """Packs two 4-bit FP4 values into one uint8 byte (low nibble first)."""
    return (high & UInt8(0x0F)) << UInt8(4) | (low & UInt8(0x0F))


def _e4m3_magnitude(code: UInt8) -> Float32:
    """Decodes the magnitude of a finite E4M3 code from its definition.

    Bias 7, three mantissa bits. The sign bit is ignored: the kernel takes
    `abs` of the scale. Codes 0x7F and 0xFF are NaN and are never generated.
    """
    var exp = Int((code >> UInt8(3)) & UInt8(0xF))
    var mant = Int(code & UInt8(0x7))
    if exp == 0:
        # Subnormal: mant * 2^(1 - 7 - 3) = mant * 2^-9.
        return Float32(mant) * Float32(0.001953125)
    return (Float32(1.0) + Float32(mant) / Float32(8.0)) * (
        Float32(1 << exp) / Float32(128.0)
    )


def _scale_code(row: Int, block: Int) -> UInt8:
    """A valid E4M3 code for a block: every finite magnitude, some negative."""
    var code = UInt8((row * 7 + block * 3) % 127)  # 0x00..0x7E, never NaN
    if (row + block) % 5 == 0:
        code |= UInt8(0x80)
    return code


def test_nvfp4_dequant[
    num_rows: Int,
    num_cols: Int,
    out_dtype: DType,
    interleaved: Bool,
](ctx: DeviceContext, tensor_sf: Float32) raises:
    """Runs the kernel for a compile-time shape and a runtime tensor scale."""
    comptime packed_cols = num_cols // 2
    comptime scale_cols = num_cols // 16
    comptime row_tiles = ceildiv(num_rows, 128)
    comptime col_tiles = ceildiv(num_cols, 64)
    comptime scales_size = (
        row_tiles
        * col_tiles
        * 32
        * 4
        * 4 if interleaved else num_rows
        * scale_cols
    )

    print(
        t"  rows= {num_rows}  cols= {num_cols}  dtype= {out_dtype} "
        t" interleaved= {interleaved}  tensor_sf= {tensor_sf}"
    )

    comptime in_size = num_rows * packed_cols
    comptime out_size = num_rows * num_cols

    var in_host = ctx.enqueue_create_host_buffer[.uint8](in_size)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        scales_size
    )
    var expected_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)

    # Zero-fill the scales first: the interleaved layout has padding.
    for i in range(scales_size):
        scales_host[i] = bitcast[.float8_e4m3fn](UInt8(0))

    # Packed weights: every 4-bit code appears, with a row-dependent shift so
    # rows differ.
    for row in range(num_rows):
        for col in range(packed_cols):
            var low = UInt8((col * 2 + row) % 16)
            var high = UInt8((col * 2 + 1 + row * 3) % 16)
            in_host[row * packed_cols + col] = _pack_fp4_pair(low, high)

    # Scales, in the layout under test.
    for row in range(num_rows):
        for block in range(scale_cols):
            var code = bitcast[.float8_e4m3fn](_scale_code(row, block))
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

    # CPU reference: (e2m1 * |scale|) * tensor_sf, then the output cast.
    for row in range(num_rows):
        for col in range(num_cols):
            var byte = in_host[row * packed_cols + col // 2]
            var shift = UInt8((col % 2) * 4)
            var nibble = Int((byte >> shift) & UInt8(0x0F))
            var scale = _e4m3_magnitude(_scale_code(row, col // 16))
            var value = (E2M1_TO_FLOAT32[nibble] * scale) * tensor_sf
            expected_host[row * num_cols + col] = value.cast[out_dtype]()

    var in_device = ctx.enqueue_create_buffer[.uint8](in_size)
    var scales_device = ctx.enqueue_create_buffer[.float8_e4m3fn](scales_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](out_size)
    ctx.enqueue_copy(in_device, in_host)
    ctx.enqueue_copy(scales_device, scales_host)
    ctx.synchronize()

    var in_tt = TileTensor(in_device, row_major[num_rows, packed_cols]())
    var out_tt = TileTensor(out_device, row_major[num_rows, num_cols]())

    comptime if interleaved:
        var scales_tt = TileTensor(
            scales_device,
            row_major(Idx[row_tiles], Idx[col_tiles], Idx[32], Idx[4], Idx[4]),
        )
        dequant_nvfp4["gpu"](
            ctx,
            out_tt,
            in_tt,
            scales_tt,
            tensor_sf,
            num_rows=num_rows,
            num_cols=num_cols,
        )
    else:
        var scales_tt = TileTensor(
            scales_device, row_major[num_rows, scale_cols]()
        )
        dequant_nvfp4["gpu"](
            ctx,
            out_tt,
            in_tt,
            scales_tt,
            tensor_sf,
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
        raise Error("NVFP4 dequant test failed")
    print("    PASS")


def _run_layouts[
    num_rows: Int, num_cols: Int, out_dtype: DType
](ctx: DeviceContext, tensor_sf: Float32) raises:
    test_nvfp4_dequant[num_rows, num_cols, out_dtype, False](ctx, tensor_sf)
    test_nvfp4_dequant[num_rows, num_cols, out_dtype, True](ctx, tensor_sf)


def main() raises:
    with DeviceContext() as ctx:
        print("NVFP4 Dequant Smoke Tests")
        print("=========================")

        print("-- float32 output --")
        _run_layouts[1, 16, .float32](ctx, Float32(1.0))
        _run_layouts[64, 64, .float32](ctx, Float32(1.0))
        _run_layouts[128, 512, .float32](ctx, Float32(0.0123))
        # Rows and columns that are not multiples of the 128 x 64 scale tile.
        _run_layouts[37, 80, .float32](ctx, Float32(0.5))
        _run_layouts[130, 192, .float32](ctx, Float32(0.0123))
        _run_layouts[200, 320, .float32](ctx, Float32(2.0))

        print("-- bfloat16 output --")
        _run_layouts[64, 64, .bfloat16](ctx, Float32(1.0))
        _run_layouts[128, 512, .bfloat16](ctx, Float32(0.0123))
        _run_layouts[130, 192, .bfloat16](ctx, Float32(0.0123))

        print("-- large shape --")
        _run_layouts[2880, 2880, .bfloat16](ctx, Float32(0.0123))

        print("=========================")
        print("ALL TESTS PASSED")
