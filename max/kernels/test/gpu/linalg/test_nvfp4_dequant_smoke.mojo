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

Validates dequant_nvfp4 by comparing GPU output against a CPU reference for
several shapes, scale dtypes (float8_e4m3fn and pre-multiplied float32), and
scale values. The kernel uses a software E2M1 LUT, so this test runs on any
GPU — it is the fallback path for architectures without native FP4 matmul
support (pre-Blackwell NVIDIA).
"""

from std.math import ceildiv
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.mxfp4_dequant import dequant_nvfp4
from linalg.fp4_utils import E2M1_TO_FLOAT32, NVFP4_SF_VECTOR_SIZE


def _pack_fp4_pair(low: UInt8, high: UInt8) -> UInt8:
    """Packs two 4-bit FP4 values into one uint8 byte."""
    return (high & UInt8(0x0F)) << UInt8(4) | (low & UInt8(0x0F))


def _cpu_dequant_nvfp4[
    scales_dtype: DType,
    out_dtype: DType = DType.bfloat16,
](
    expected: UnsafePointer[mut=True, Scalar[out_dtype], _],
    input_data: UnsafePointer[mut=False, Scalar[DType.uint8], _],
    scales_data: UnsafePointer[mut=False, Scalar[scales_dtype], _],
    num_rows: Int,
    num_cols: Int,
):
    """CPU reference: dequant NVFP4 packed uint8 with E4M3 or f32 scales."""
    var packed_cols = num_cols // 2
    var scale_cols = ceildiv(num_cols, NVFP4_SF_VECTOR_SIZE)

    for row in range(num_rows):
        for col in range(num_cols):
            var packed_col = col // 2
            var packed_byte = input_data[row * packed_cols + packed_col]
            var nibble_shift = UInt8((col % 2) * 4)
            var fp4_bits = Int((packed_byte >> nibble_shift) & UInt8(0x0F))
            var fp32_val = E2M1_TO_FLOAT32[fp4_bits]

            var scale_col = col // NVFP4_SF_VECTOR_SIZE
            var scale_f32 = scales_data[row * scale_cols + scale_col].cast[
                DType.float32
            ]()

            var result = (fp32_val * scale_f32).cast[out_dtype]()
            expected[row * num_cols + col] = result


def test_nvfp4_dequant[
    num_rows: Int,
    num_cols: Int,
    scales_dtype: DType = DType.float8_e4m3fn,
    out_dtype: DType = DType.bfloat16,
](ctx: DeviceContext, scale_value: Float32) raises:
    """Tests NVFP4 dequant for compile-time shape and runtime scale."""
    comptime packed_cols = num_cols // 2
    comptime scale_cols = ceildiv(num_cols, NVFP4_SF_VECTOR_SIZE)

    # FP8 output has lower precision; use a wider tolerance.
    comptime tol = Float32(
        0.1
    ) if out_dtype == DType.float8_e4m3fn else Float32(0.01)

    # Round the requested scale through the scales dtype so the CPU
    # reference and the kernel see the identical stored value.
    var scale_stored = scale_value.cast[scales_dtype]()
    print(
        "  rows=",
        num_rows,
        " cols=",
        num_cols,
        " scales=",
        scales_dtype,
        " out=",
        out_dtype,
        " scale=",
        scale_stored.cast[DType.float32](),
    )

    comptime in_size = num_rows * packed_cols
    comptime scales_size = num_rows * scale_cols
    comptime out_size = num_rows * num_cols

    # Allocate and fill host input
    var in_host = ctx.enqueue_create_host_buffer[DType.uint8](in_size)
    var scales_host = ctx.enqueue_create_host_buffer[scales_dtype](scales_size)
    var expected_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    # Vary scales per block so wrong scale indexing cannot pass unnoticed.
    for i in range(scales_size):
        scales_host[i] = (
            scale_value * Float32(1 + i % 3) / Float32(1 + (i // 7) % 2)
        ).cast[scales_dtype]()

    for row in range(num_rows):
        for col in range(packed_cols):
            var low = UInt8((row + col * 2) % 16)
            var high = UInt8((row + col * 2 + 1) % 16)
            in_host[row * packed_cols + col] = _pack_fp4_pair(low, high)

    # CPU reference
    _cpu_dequant_nvfp4[scales_dtype, out_dtype](
        expected_host.unsafe_ptr(),
        in_host.unsafe_ptr(),
        scales_host.unsafe_ptr(),
        num_rows,
        num_cols,
    )

    # Device buffers
    var in_device = ctx.enqueue_create_buffer[DType.uint8](in_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](out_size)

    ctx.enqueue_copy(in_device, in_host)
    ctx.enqueue_copy(scales_device, scales_host)
    ctx.synchronize()

    # Create TileTensors with compile-time row-major layouts
    var in_tt = TileTensor(in_device, row_major[num_rows, packed_cols]())
    var scales_tt = TileTensor(scales_device, row_major[num_rows, scale_cols]())
    var out_tt = TileTensor(out_device, row_major[num_rows, num_cols]())

    # Run GPU kernel
    dequant_nvfp4(
        ctx,
        out_tt,
        in_tt,
        scales_tt,
        num_rows=num_rows,
        num_cols=num_cols,
    )
    ctx.synchronize()

    # Copy output back
    var out_host_buf = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    ctx.enqueue_copy(out_host_buf, out_device)
    ctx.synchronize()

    # Compare
    var max_err = Float32(0.0)
    var num_mismatches = 0
    for i in range(out_size):
        var got = out_host_buf[i].cast[DType.float32]()
        var exp = expected_host[i].cast[DType.float32]()
        var err = abs(got - exp)
        max_err = max(max_err, err)
        if err > tol:
            if num_mismatches < 5:
                var row = i // num_cols
                var col = i % num_cols
                print(
                    "    MISMATCH [",
                    row,
                    ",",
                    col,
                    "]: got=",
                    got,
                    " expected=",
                    exp,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL: ",
            num_mismatches,
            " mismatches, max_err=",
            max_err,
        )
        raise Error("NVFP4 dequant test failed")

    print("    PASS max_err=", max_err)


def main() raises:
    with DeviceContext() as ctx:
        print("NVFP4 Dequant Smoke Tests (any GPU)")
        print("======================================")

        # E4M3 block scales (raw checkpoint scales, no per-tensor scale)
        print("-- E4M3 scales --")
        test_nvfp4_dequant[64, 64](ctx, 1.0)
        test_nvfp4_dequant[128, 512](ctx, 2.0)
        test_nvfp4_dequant[256, 2880](ctx, 0.5)
        test_nvfp4_dequant[100, 192](ctx, 1.5)

        # Pre-multiplied float32 scales (the path the Python fallback uses:
        # block_scale * weight_scale_2)
        print("-- float32 scales --")
        test_nvfp4_dequant[64, 64, scales_dtype=DType.float32](ctx, 1.0)
        test_nvfp4_dequant[128, 512, scales_dtype=DType.float32](ctx, 0.125)
        test_nvfp4_dequant[256, 2880, scales_dtype=DType.float32](ctx, 3.0)

        # Large shape (gemma-4-26B-A4B expert dimensions)
        print("-- Large shape --")
        test_nvfp4_dequant[2880, 2880](ctx, 1.0)

        # FP8 output
        print("-- FP8 output --")
        test_nvfp4_dequant[64, 64, out_dtype=DType.float8_e4m3fn](ctx, 1.0)

        print("======================================")
        print("ALL TESTS PASSED")
