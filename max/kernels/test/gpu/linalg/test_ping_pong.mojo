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
"""AMD Ping-Pong Matmul Tests.

Supports both BF16 and FP8 via compile-time flag:
  mojo -D FP8=false test_ping_pong.mojo  # BF16 (default)
  mojo -D FP8=true test_ping_pong.mojo   # FP8
"""

from sys import env_get_bool

from gpu import WARP_SIZE
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from layout._fillers import random
import linalg.matmul.vendor.blas as vendor_blas
from testing import assert_equal
from random import random_si64
from linalg.matmul.gpu.amd.pingpong_kernel import ping_pong_matmul

# Compile-time dtype selection: -D FP8=true for FP8, otherwise BF16
comptime USE_FP8 = env_get_bool["FP8", False]()
comptime input_dtype = DType.float8_e4m3fn if USE_FP8 else DType.bfloat16

comptime test_size = 4 * 1024


def test_ping_pong_kernel_amd[
    in_dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    enable_swizzle: Bool = False,
](ctx: DeviceContext):
    """Test ping-pong kernel with parameterized input dtype."""
    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        # Use deterministic pattern for FP8, random for BF16
        comptime if in_dtype == DType.float8_e4m3fn:
            for i in range(M * K):
                # Alternate between 1 and 2
                host_a[i] = Scalar[in_dtype](Float32(1 + (i % 2)))
            for i in range(K * N):
                # Alternate between 1 and 2 (avoid negative for simpler FP8 math)
                host_b[i] = Scalar[in_dtype](Float32(1 + (i % 2)))
        else:
            for i in range(M * K):
                var val = random_si64(0, 20)
                host_a[i] = val.cast[in_dtype]()
            for i in range(K * N):
                var val = random_si64(0, 20)
                host_b[i] = val.cast[in_dtype]()

    var a_device_tensor = LayoutTensor[in_dtype, Layout.row_major(M, K)](
        device_a
    )
    var b_device_tensor = LayoutTensor[in_dtype, Layout.row_major(N, K)](
        device_b
    )
    var c_device_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N)](
        device_c
    )

    ctx.enqueue_memset(device_c, 0)
    ctx.enqueue_memset(device_c_ref, 0)

    # Run kernel under test
    ping_pong_matmul[enable_swizzle=enable_swizzle](
        a_device_tensor, b_device_tensor, c_device_tensor, ctx
    )

    # Compute reference
    var c_device_ref_tensor = LayoutTensor[
        DType.float32, Layout.row_major(M, N)
    ](device_c_ref)
    vendor_blas.matmul(
        ctx,
        c_device_ref_tensor,
        a_device_tensor,
        b_device_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    # Validate results
    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        var printed = 0
        for i in range(M * N):
            var diff = abs(host_c[i] - host_c_ref[i])
            # Use tolerance for FP8 (limited precision)
            var tol = Float32(
                1.0
            ) if in_dtype == DType.float8_e4m3fn else Float32(0.0)
            if diff > tol and host_c[i] != host_c_ref[i]:
                if printed < 10:
                    print(
                        "Mismatch at (",
                        i // N,
                        ",",
                        i % N,
                        ") ",
                        host_c[i],
                        " vs ",
                        host_c_ref[i],
                    )
                    printed += 1
                errors += 1

        if errors != 0:
            print("First row actual vs ref:")
            for j in range(min(16, N)):
                print(host_c[j], host_c_ref[j])

        assert_equal(errors, 0)


def main():
    with DeviceContext() as ctx:
        print("Running AMD Ping-Pong Kernel Tests")
        print("  Input dtype:", input_dtype)
        print("  Matrix size:", test_size)

        # Test without swizzle
        print("  Testing without swizzle...")
        for i in range(1):
            print("Test ", i, " without swizzle")
            test_ping_pong_kernel_amd[
                input_dtype,
                test_size,
                test_size,
                test_size,
                enable_swizzle=False,
            ](ctx)
        print("  PASSED: No swizzle")

        # Test with swizzle (16x32 subtile-aligned layout)
        print("  Testing with swizzle...")
        for i in range(1):
            print("Test ", i, " with swizzle")
            test_ping_pong_kernel_amd[
                input_dtype,
                test_size,
                test_size,
                test_size,
                enable_swizzle=True,
            ](ctx)
        print("  PASSED: With swizzle")

        print("==== AMD Ping-Pong Kernel Tests passed ====")
