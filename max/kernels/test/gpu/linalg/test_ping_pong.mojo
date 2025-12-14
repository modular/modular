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

from gpu import WARP_SIZE
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from layout._fillers import random
import linalg.matmul.vendor.blas as vendor_blas
from testing import assert_equal
from random import random_si64
from linalg.matmul.gpu.amd.pingpong_kernel import ping_pong_matmul


def test_ping_pong_kernel_amd[
    M: Int,
    N: Int,
    K: Int,
    enable_swizzle: Bool = False,
    use_transpose_load: Bool = False,
](ctx: DeviceContext):
    var device_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var device_b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var device_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            var val = random_si64(0, 20)
            host_a[i] = val.cast[DType.bfloat16]()

        for i in range(K * N):
            var val = random_si64(0, 20)
            host_b[i] = val.cast[DType.bfloat16]()

    var a_device_tensor = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(M, K),
    ](device_a)

    var b_device_tensor = LayoutTensor[DType.bfloat16, Layout.row_major(N, K)](
        device_b
    )

    var c_device_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N)](
        device_c
    )

    var c_device_ref_tensor = LayoutTensor[
        DType.float32, Layout.row_major(M, N)
    ](device_c_ref)

    ctx.enqueue_memset(device_c, 0)

    ping_pong_matmul[
        enable_swizzle=enable_swizzle, use_transpose_load=use_transpose_load
    ](a_device_tensor, b_device_tensor, c_device_tensor, ctx)

    vendor_blas.matmul(
        ctx,
        c_device_ref_tensor,
        a_device_tensor,
        b_device_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        var printed = 0
        for i in range(M * N):
            if host_c[i] != host_c_ref[i]:
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

        comptime size = 4 * 1024

        # Test without swizzle
        print("  Testing without swizzle...")
        for i in range(1):
            print("Test ", i, " without swizzle")
            test_ping_pong_kernel_amd[
                size,
                size,
                size,
                enable_swizzle=False,
            ](ctx)
        print("  PASSED: No swizzle")

        # Test with swizzle (16x32 subtile-aligned layout)
        print("  Testing with swizzle...")
        for i in range(1):
            print("Test ", i, " with swizzle")
            test_ping_pong_kernel_amd[
                size,  # M
                size,  # N
                size,  # K
                enable_swizzle=True,
            ](ctx)
        print("  PASSED: With swizzle")

        # Test with transpose load without swizzle
        print("  Testing with transpose load (no swizzle)...")
        for i in range(1):
            print("Test ", i, " with transpose load (no swizzle)")
            test_ping_pong_kernel_amd[
                size,  # M
                size,  # N
                size,  # K
                enable_swizzle=False,
                use_transpose_load=True,
            ](ctx)
        print("  PASSED: Transpose load (no swizzle)")

        # Test with transpose load WITH swizzle
        print("  Testing with transpose load + swizzle...")
        for i in range(1):
            print("Test ", i, " with transpose load + swizzle")
            test_ping_pong_kernel_amd[
                size,  # M
                size,  # N
                size,  # K
                enable_swizzle=True,
                use_transpose_load=True,
            ](ctx)
        print("  PASSED: Transpose load + swizzle")

        print("==== AMD Ping-Pong Kernel Tests passed ====")
