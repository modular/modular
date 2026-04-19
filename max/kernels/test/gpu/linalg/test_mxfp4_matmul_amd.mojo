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
"""Tests for the native MXFP4 block-scaled matmul kernel on AMD CDNA4.

Validates mxfp4_block_scaled_matmul_amd against a per-element GPU reference
that uses the llvm.amdgcn.cvt.scalef32.pk.f32.fp4 intrinsic for FP4→FP32
dequantization and scalar accumulation.

Usage:
  mojo test_mxfp4_matmul_amd.mojo
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64
from std.sys.intrinsics import llvm_intrinsic

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import mxfp4_block_scaled_matmul_amd


# ===----------------------------------------------------------------------=== #
# Reference kernel: scalar FP4 dequant matmul on GPU
# ===----------------------------------------------------------------------=== #


def block_scaled_matmul_ref(
    a_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    a_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    b_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """Per-element GPU reference for MXFP4 block-scaled matmul.

    Each thread computes one (m, n) output element by dequantizing
    packed FP4 data via the CDNA4 cvt.scalef32.pk.f32.fp4 intrinsic,
    multiplying with E8M0 scales, and accumulating in FP32.
    """

    @always_inline
    def cast_fp4x2_to_fp32x2[
        byte_select: Int
    ](packed: Int32, scale: Float32) -> SIMD[DType.float32, 2]:
        return llvm_intrinsic[
            "llvm.amdgcn.cvt.scalef32.pk.f32.fp4",
            SIMD[DType.float32, 2],
        ](packed, scale, Int32(byte_select))

    var m = global_idx.x
    var n = global_idx.y

    if m >= M or n >= N:
        return

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    var am_scales_ptr = a_scales_ptr + m * k_groups
    var bn_scales_ptr = b_scales_ptr + n * k_groups

    var am_ptr = a_ptr + m * (K // 2)
    var bn_ptr = b_ptr + n * (K // 2)

    var accum = SIMD[DType.float32, 2](0)

    for ko in range(k_groups):
        var a_scale = am_scales_ptr[ko].cast[DType.float32]()
        var b_scale = bn_scales_ptr[ko].cast[DType.float32]()

        for ki in range(0, MXFP4_SF_VECTOR_SIZE // 2, 4):
            var a_data = bitcast[DType.int32, 1](am_ptr.load[width=4](ki))
            var b_data = bitcast[DType.int32, 1](bn_ptr.load[width=4](ki))

            comptime for byte_select in range(4):
                accum += cast_fp4x2_to_fp32x2[byte_select](
                    a_data, a_scale
                ) * cast_fp4x2_to_fp32x2[byte_select](b_data, b_scale)

        am_ptr += MXFP4_SF_VECTOR_SIZE // 2
        bn_ptr += MXFP4_SF_VECTOR_SIZE // 2

    c_ptr[m * N + n] = accum.reduce_add()


# ===----------------------------------------------------------------------=== #
# Test harness
# ===----------------------------------------------------------------------=== #


def test_mxfp4_matmul[
    M_static: Int, N_static: Int, K_static: Int
](ctx: DeviceContext) raises:
    """Test mxfp4_block_scaled_matmul_amd against the GPU reference.

    All dimensions are compile-time static (required by the kernel).
    K is the logical FP4 element count; packed data has K//2 bytes.

    Parameters:
        M_static: Number of rows in A / C.
        N_static: Number of rows in B (transposed) / cols in C.
        K_static: Logical K dimension (FP4 elements, must be multiple of 128).
    """
    comptime assert (
        K_static % 128 == 0
    ), "K must be a multiple of 128 (MFMA K dimension)"
    comptime assert (
        K_static % MXFP4_SF_VECTOR_SIZE == 0
    ), "K must be a multiple of MXFP4_SF_VECTOR_SIZE (32)"

    print(M_static, "x", N_static, "x", K_static)

    comptime input_dtype = DType.uint8
    comptime scales_dtype = DType.float8_e8m0fnu
    comptime output_dtype = DType.float32

    comptime K_PACKED = K_static // 2
    comptime K_SCALES = K_static // MXFP4_SF_VECTOR_SIZE

    # Sizes
    comptime a_size = M_static * K_PACKED
    comptime b_size = N_static * K_PACKED
    comptime c_size = M_static * N_static
    comptime a_scales_size = M_static * K_SCALES
    comptime b_scales_size = N_static * K_SCALES

    # Layouts
    comptime a_shape = row_major[M_static, K_PACKED]()
    comptime b_shape = row_major[N_static, K_PACKED]()
    comptime c_shape = row_major[M_static, N_static]()
    comptime a_scales_shape = row_major[M_static, K_SCALES]()
    comptime b_scales_shape = row_major[N_static, K_SCALES]()

    # Host allocations
    var a_host = alloc[Scalar[input_dtype]](a_size)
    var b_host = alloc[Scalar[input_dtype]](b_size)
    var a_scales_host = alloc[Scalar[scales_dtype]](a_scales_size)
    var b_scales_host = alloc[Scalar[scales_dtype]](b_scales_size)
    var c_host = alloc[Scalar[output_dtype]](c_size)
    var c_host_ref = alloc[Scalar[output_dtype]](c_size)

    # Random packed FP4 data (same pattern as test_mxfp4_dequant_matmul_amd).
    for i in range(a_size):
        a_host[i] = UInt8(random_ui64(0, 255))
    for i in range(b_size):
        b_host[i] = UInt8(random_ui64(0, 255))

    # Scales: each lane holds 32 FP4 elements and its own E8M0 scale,
    # matching MX per-32 granularity. Each scale can be independent.
    # Use exponent range [125..129] (scales 0.25..4.0) to keep
    # accumulated values in FP32-representable range.
    for i in range(a_scales_size):
        a_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))
    for i in range(b_scales_size):
        b_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))

    # Device allocations
    var a_dev = ctx.enqueue_create_buffer[input_dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[input_dtype](b_size)
    var a_scales_dev = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var c_dev = ctx.enqueue_create_buffer[output_dtype](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[output_dtype](c_size)

    # Copy to device
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    # TileTensor views
    var a_tt = TileTensor[mut=False](a_dev, a_shape)
    var b_tt = TileTensor[mut=False](b_dev, b_shape)
    var c_tt = TileTensor[mut=True](c_dev, c_shape)
    var a_scales_tt = TileTensor[mut=False](a_scales_dev, a_scales_shape)
    var b_scales_tt = TileTensor[mut=False](b_scales_dev, b_scales_shape)

    # --- Run kernel under test ---
    mxfp4_block_scaled_matmul_amd(
        c_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx
    )

    # --- Run reference kernel ---
    comptime BLOCK_DIM = 32
    ctx.enqueue_function_experimental[block_scaled_matmul_ref](
        a_dev,
        b_dev,
        a_scales_dev,
        b_scales_dev,
        c_ref_dev,
        M_static,
        N_static,
        K_static,
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    # Copy results back
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    # var m = M_static
    # var N = N_static
    # print("\nFirst 32x32 block of Actual Output Matrix:")
    # for row in range(min(32, m)):
    #     for col in range(min(32, N)):
    #         var actual = c_host[row * N + col].cast[DType.float32]()
    #         print(actual, end=", " if col < min(32, N) - 1 else "")
    #     print()

    # print("\nFirst 32x32 block of Reference Output Matrix:")
    # for row in range(min(32, m)):
    #     for col in range(min(32, N)):
    #         var expected = c_host_ref[row * N + col].cast[DType.float32]()
    #         print(expected, end=", " if col < min(32, N) - 1 else "")
    #     print()

    # Validate
    assert_almost_equal(
        c_host,
        c_host_ref,
        c_size,
        atol=0.05,
        rtol=0.05,
    )

    print("  PASSED")

    # Cleanup
    a_host.free()
    b_host.free()
    a_scales_host.free()
    b_scales_host.free()
    c_host.free()
    c_host_ref.free()


def main() raises:
    with DeviceContext() as ctx:
        print("===> MXFP4 block-scaled matmul (native CDNA4 MFMA)")

        # Small: basic correctness
        test_mxfp4_matmul[128, 128, 128](ctx)

        # K > BK: exercises the K-loop
        test_mxfp4_matmul[128, 128, 256](ctx)
        test_mxfp4_matmul[128, 128, 512](ctx)

        # Non-square tiles
        test_mxfp4_matmul[256, 128, 256](ctx)
        test_mxfp4_matmul[128, 256, 256](ctx)

        # Larger: multiple blocks in M and N
        test_mxfp4_matmul[256, 256, 256](ctx)
        test_mxfp4_matmul[256, 256, 512](ctx)

        # Stress: deeper K
        test_mxfp4_matmul[128, 128, 1024](ctx)

        print("==== All MXFP4 block-scaled matmul tests passed ====")
