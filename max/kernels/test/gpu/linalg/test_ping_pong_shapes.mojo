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
"""Test ping-pong kernel across different shapes.

Documents known limitations:
- FP8: Only works when M % 256 == 0 (BM tile alignment)
- BF16: Works with all M values
"""

from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from testing import assert_equal
from linalg.matmul.gpu.amd.pingpong_kernel import ping_pong_matmul
from testing import assert_true


fn test_shape[
    in_dtype: DType, M: Int, N: Int, K: Int, enable_swizzle: Bool = True
](ctx: DeviceContext) raises:
    """Test a single shape."""
    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    # Initialize with simple pattern
    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            host_a[i] = Scalar[in_dtype](Float32(1 + (i % 2)))
        for i in range(K * N):
            host_b[i] = Scalar[in_dtype](Float32(1 + (i % 2)))

    var a_tensor = LayoutTensor[in_dtype, Layout.row_major(M, K)](device_a)
    var b_tensor = LayoutTensor[in_dtype, Layout.row_major(N, K)](device_b)
    var c_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N)](device_c)

    ctx.enqueue_memset(device_c, 0)
    ctx.enqueue_memset(device_c_ref, 0)

    # Run kernel
    ping_pong_matmul[enable_swizzle=enable_swizzle](
        a_tensor, b_tensor, c_tensor, ctx
    )

    # Compute reference
    var c_ref_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N)](
        device_c_ref
    )
    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    # Validate
    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        var tol = Float32(1.0) if in_dtype == DType.float8_e4m3fn else Float32(
            0.0
        )

        for i in range(M * N):
            var diff = abs(host_c[i] - host_c_ref[i])
            if diff > tol:
                errors += 1
                if errors <= 5:
                    var row = i // N
                    var col = i % N
                    print(
                        "  Mismatch at row",
                        row,
                        "col",
                        col,
                        ":",
                        host_c[i],
                        "vs",
                        host_c_ref[i],
                    )

        assert_true(errors == 0, msg=String("Test failed:", errors, "errors"))


def main():
    with DeviceContext() as ctx:
        print("Testing Ping-Pong Kernel Shape Compatibility")
        print("=" * 60)

        # BF16: Works with all M values
        print("\nBF16 - Testing various M values:")
        print("  M=4096 (aligned)...", end="")
        test_shape[DType.bfloat16, 4096, 4096, 4096, enable_swizzle=True](ctx)
        print(" PASSED")

        print("  M=1000 (unaligned)...", end="")
        test_shape[DType.bfloat16, 1000, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=300...", end="")
        test_shape[DType.bfloat16, 300, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=100 (small)...", end="")
        test_shape[DType.bfloat16, 100, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=16 (very small)...", end="")
        test_shape[DType.bfloat16, 16, 4096, 4096](ctx)
        print(" PASSED")

        # FP8 - Testing various M values
        print("\nFP8 - Testing aligned M values (M % 256 == 0):")
        print("  M=4096...", end="")
        test_shape[DType.float8_e4m3fn, 4096, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=2048...", end="")
        test_shape[DType.float8_e4m3fn, 2048, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=256...", end="")
        test_shape[DType.float8_e4m3fn, 256, 4096, 4096](ctx)
        print(" PASSED")

        # Test FP8 without swizzle to isolate the issue
        print("\nFP8 - Testing unaligned M without swizzle:")
        print("  M=1000...", end="")
        test_shape[DType.float8_e4m3fn, 1024, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=300...", end="")
        test_shape[DType.float8_e4m3fn, 300, 4096, 4096](ctx)
        print(" PASSED")

        # FP8 partial blocks: test 32×32×64 MMA (M % 32 == 0, M % 256 != 0)
        # Runtime dispatch:
        # - M % 256 == 0: 16×16×128 (full blocks, highest throughput)
        # - M % 32 == 0: 32×32×64 (2× K throughput vs 16×16×32)
        # - M % 32 != 0: 16×16×32 (baseline, handles unaligned M)

        print("\nFP8 - Testing 32×32×64 MMA (M % 32 == 0, M % 256 != 0):")
        print("  M=992 (partial block, 32-aligned)...", end="")
        test_shape[DType.float8_e4m3fn, 992, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=960 (partial block, 32-aligned)...", end="")
        test_shape[DType.float8_e4m3fn, 960, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=1088 (partial block, 32-aligned)...", end="")
        test_shape[DType.float8_e4m3fn, 1088, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=288 (partial block, 32-aligned, edge case)...", end="")
        test_shape[DType.float8_e4m3fn, 288, 4096, 4096](ctx)
        print(" PASSED")

        print("\nFP8 - Testing 16×16×128 MMA (M % 256 == 0):")
        print("  M=1024 (full blocks)...", end="")
        test_shape[DType.float8_e4m3fn, 1024, 4096, 4096](ctx)
        print(" PASSED")

        print("\nFP8 - Testing 16×16×128 MMA (unaligned M, fallback):")
        print("  M=1000 (partial block, unaligned)...", end="")
        test_shape[DType.float8_e4m3fn, 1000, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=1001 (partial block, unaligned)...", end="")
        test_shape[DType.float8_e4m3fn, 1001, 4096, 4096](ctx)
        print(" PASSED")

        print("\n" + "=" * 60)
        print("All tests passed!")
