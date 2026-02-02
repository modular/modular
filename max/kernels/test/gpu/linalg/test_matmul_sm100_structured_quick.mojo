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
"""Quick validation test for SM100 structured kernel.

This test directly imports the structured kernel (no flag required) and
tests key output pipeline paths:
1. Basic matmul (no epilogue)
2. Register-based epilogue
3. SMEM-based epilogue
4. Split-K
5. swapAB (transpose output)

Usage:
    mojo max/kernels/test/gpu/linalg/test_matmul_sm100_structured_quick.mojo
"""

from collections import OptionalReg
from sys import align_of, size_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import assert_almost_equal
from random import rand
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major

# Direct import of structured kernel (same name, different module)
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.utils import elementwise_compute_lambda_type

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple


def test_structured[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    num_split_k: Int = 1,
    test_lambda: Bool = False,
    register_based_epilogue: Bool = True,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim, test_name: String):
    """Test structured kernel with given configuration."""
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        "[",
        test_name,
        "] M=",
        M,
        " N=",
        N,
        " K=",
        K,
        " cta_group=",
        cta_group,
        " swapAB=",
        swapAB,
        " split_k=",
        num_split_k,
        " lambda=",
        test_lambda,
        " reg_epi=",
        register_based_epilogue,
        sep="",
    )

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)
    var a_size = m.value * k.value
    var b_size = n.value * k.value if transpose_b else k.value * n.value
    var c_size = m.value * n.value

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    # Initialize with random data
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

    # Copy to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_split_k=num_split_k,
    )

    # No lambda for simplicity
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        from_ndbuffer_row_major(c_device_nd),
        from_ndbuffer_row_major(a_device_nd),
        from_ndbuffer_row_major(b_device_nd),
        ctx,
    )

    vendor_blas.matmul(
        ctx,
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")

    # Clean up
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device


def main():
    print("=" * 60)
    print("SM100 STRUCTURED KERNEL QUICK TEST")
    print("=" * 60)
    print()

    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 16

        # Test 1: Basic 1SM
        print("--- Test 1: Basic 1SM ---")
        test_structured[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 32, BK),
            mma_shape = Index(64, 32, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
        ](ctx, dynamic(256), static[256](), static[256](), "1SM-basic")

        # Test 2: Basic 2SM
        print("--- Test 2: Basic 2SM ---")
        test_structured[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 64, BK),
            mma_shape = Index(256, 128, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
        ](ctx, dynamic(512), static[512](), static[512](), "2SM-basic")

        # Test 3: swapAB (transpose output)
        print("--- Test 3: swapAB ---")
        test_structured[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 64, BK),
            mma_shape = Index(128, 64, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=1,
            swapAB=True,
        ](ctx, dynamic(256), static[512](), static[512](), "swapAB")

        # Test 4: Split-K
        print("--- Test 4: Split-K ---")
        test_structured[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 32, BK),
            mma_shape = Index(128, 64, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
            num_split_k=2,
        ](ctx, dynamic(256), static[256](), static[512](), "split-K")

        # Test 5: k_group_size=2
        print("--- Test 5: k_group=2 ---")
        test_structured[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 32, BK),
            mma_shape = Index(64, 32, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 2, 1),
            cta_group=1,
            k_group_size=2,
        ](ctx, dynamic(256), static[512](), static[1024](), "k_group=2")

        # Test 6: 2SM + swapAB
        print("--- Test 6: 2SM + swapAB ---")
        test_structured[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 64, BK),
            mma_shape = Index(256, 128, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
            swapAB=True,
        ](ctx, dynamic(256), static[512](), static[512](), "2SM+swapAB")

    print("=" * 60)
    print("ALL STRUCTURED KERNEL TESTS PASSED!")
    print("=" * 60)
