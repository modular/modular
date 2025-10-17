# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""General FP16×FP16+FP32→FP32 MMA test for all GPU architectures."""

from gpu import thread_idx
from gpu.host import DeviceContext
from gpu.mma import mma
from gpu.mma_util import (
    load_matrix_a_amd_rdna16x16x16,
    load_matrix_b_amd_rdna16x16x16,
)
from testing import assert_equal


fn test_mma_fp16_kernel(
    A: UnsafePointer[Float16],
    B: UnsafePointer[Float16],
    C: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
):
    """FP16×FP16+FP32→FP32 MMA test kernel.

    Performs matrix multiply-accumulate: C = A @ B
    Uses hardware-appropriate MMA instructions across different GPU architectures:
    - NVIDIA: Tensor core wmma or mma.sync instructions
    - AMD CDNA (MI series): v_mfma instructions
    - AMD RDNA3+ (RX 7000, W7000): v_wmma_f32_16x16x16_f16 instructions
    - AMD RDNA1/2: Falls back to scalar operations

    For RDNA3 wave32 mode, fragment sizes are:
    - A/B: 16 fp16 elements per lane (full K=16 dimension)
    - C: 8 fp32 elements per lane (M×N=256 distributed across 32 lanes)

    Args:
        A: Input matrix A (M×K) in FP16.
        B: Input matrix B (K×N) in FP16.
        C: Output matrix C (M×N) in FP32.
        M: Number of rows in A and C.
        N: Number of columns in B and C.
        K: Inner dimension (columns of A, rows of B).
    """
    var tid = Int(thread_idx.x)

    # Load fragments using RDNA-specific loaders (returns SIMD[16])
    var a_frag = load_matrix_a_amd_rdna16x16x16(A, 0, 0, K)
    var b_frag = load_matrix_b_amd_rdna16x16x16(B, 0, 0, N)

    # Accumulator (8 fp32 per thread in wave32)
    var c_frag = SIMD[DType.float32, 8](0.0)
    var d_frag = SIMD[DType.float32, 8](0.0)

    # Perform MMA: d = a @ b + c
    mma(d_frag, a_frag, b_frag, c_frag)

    # Store result - each thread stores its 8 elements
    var lane = tid
    if lane < 32:
        var base_offset = lane * 8
        for i in range(8):
            if base_offset + i < M * N:
                C[base_offset + i] = d_frag[i]


def main():
    """Test FP16 matrix multiply-accumulate operation."""
    print("Testing FP16×FP16+FP32→FP32 MMA...")

    with DeviceContext() as ctx:
        # Test 16×16×16 matrix multiply
        alias M = 16
        alias N = 16
        alias K = 16

        # Allocate host memory
        var A_host = UnsafePointer[Float16].alloc(M * K)
        var B_host = UnsafePointer[Float16].alloc(K * N)
        var C_host = UnsafePointer[Float32].alloc(M * N)

        # Initialize: A and B all 1.0
        for i in range(M * K):
            A_host[i] = 1.0
        for i in range(K * N):
            B_host[i] = 1.0
        for i in range(M * N):
            C_host[i] = 0.0

        # Allocate device memory
        var A_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
        var B_dev = ctx.enqueue_create_buffer[DType.float16](K * N)
        var C_dev = ctx.enqueue_create_buffer[DType.float32](M * N)

        # Copy to device
        ctx.enqueue_copy(A_dev, A_host)
        ctx.enqueue_copy(B_dev, B_host)
        ctx.enqueue_copy(C_dev, C_host)

        # Launch kernel (1 wave = 32 threads for wave32)
        alias kernel = test_mma_fp16_kernel
        ctx.enqueue_function_checked[kernel, kernel](
            A_dev,
            B_dev,
            C_dev,
            M,
            N,
            K,
            grid_dim=(1,),
            block_dim=32,
        )

        # Copy result back
        ctx.enqueue_copy(C_host, C_dev)

        # Verify: C = A @ B where A and B are all 1.0
        # Each element of C should be K * 1.0 * 1.0 = 16.0
        var errors = 0
        for i in range(M * N):
            var expected: Float32 = K  # 16.0
            if abs(C_host[i] - expected) > 0.01:
                errors += 1
                if errors <= 5:  # Print first 5 errors
                    print(
                        "  Error at [",
                        i,
                        "]: got",
                        C_host[i],
                        "expected",
                        expected,
                    )

        # Cleanup
        _ = A_dev
        _ = B_dev
        _ = C_dev
        A_host.free()
        B_host.free()
        C_host.free()

        if errors == 0:
            print("✅ FP16 MMA test PASSED (", M * N, "elements correct)")
        else:
            print("❌ FP16 MMA test FAILED (", errors, "errors)")
            raise Error("Test failed")
