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
"""SM100 Matmul Smoke Test - Quick validation for development iteration.

This test file provides a minimal but comprehensive set of tests covering
the key code paths in the SM100 matmul kernel. It is designed to run quickly
(target: < 60 seconds) while catching most regressions.

Coverage:
- cta_group=1 (1SM) and cta_group=2 (2SM)
- Single CTA (1x1x1) and multi-CTA clusters (4x4x1)
- swapAB=True and swapAB=False
- k_group_size=1 and k_group_size=2
- Split-K kernel
- Different MMA shapes (small: 64x32, large: 128x128)
- Dynamic M dimension (common inference pattern)

Usage:
    bazel test //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke --config=b200

For full validation, run the comprehensive tests in CI.
"""

from sys import size_of

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
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple


def test_blackwell_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    num_split_k: Int = 1,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    """Generic test function for SM100 matmul kernel variants."""
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        "[SMOKE] dtypes=(",
        a_type,
        ", ",
        b_type,
        ", ",
        c_type,
        ") shape=(",
        M,
        ", ",
        N,
        ", ",
        K,
        ") mma=",
        mma_shape,
        " cta_group=",
        cta_group,
        " cluster=(",
        cluster_shape[0],
        ",",
        cluster_shape[1],
        ") swapAB=",
        swapAB,
        " k_group=",
        k_group_size,
        " split_k=",
        num_split_k,
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
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_split_k=num_split_k,
    )

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

    # Clean up host allocations
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
    print("SM100 MATMUL SMOKE TEST")
    print("=" * 60)
    print()

    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 16

        # ============================================================
        # Test 1: Basic 1SM kernel (cta_group=1), single CTA cluster
        # ============================================================
        print("--- Test 1: 1SM kernel, single CTA (1x1x1) ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 32, BK),
            mma_shape = Index(64, 32, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
        ](ctx, dynamic(256), static[256](), static[256]())

        # ============================================================
        # Test 2: 1SM kernel with larger cluster (4x4x1)
        # ============================================================
        print("--- Test 2: 1SM kernel, multi-CTA cluster (4x4x1) ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 64, BK),
            mma_shape = Index(128, 64, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=1,
            block_swizzle_size=4,
        ](ctx, dynamic(512), static[512](), static[512]())

        # ============================================================
        # Test 3: 2SM kernel (cta_group=2)
        # ============================================================
        print("--- Test 3: 2SM kernel (cta_group=2) ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 64, BK),
            mma_shape = Index(256, 128, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
            block_swizzle_size=4,
        ](ctx, dynamic(512), static[512](), static[512]())

        # ============================================================
        # Test 4: swapAB=True (different memory access pattern)
        # ============================================================
        print("--- Test 4: swapAB=True ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 64, BK),
            mma_shape = Index(128, 64, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=1,
            swapAB=True,
        ](ctx, dynamic(256), static[512](), static[512]())

        # ============================================================
        # Test 5: k_group_size=2 (K-dimension tiling)
        # ============================================================
        print("--- Test 5: k_group_size=2 ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 32, BK),
            mma_shape = Index(64, 32, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 2, 1),
            cta_group=1,
            k_group_size=2,
        ](ctx, dynamic(256), static[512](), static[1024]())

        # ============================================================
        # Test 6: Split-K kernel
        # ============================================================
        print("--- Test 6: Split-K kernel (num_split_k=2) ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 32, BK),
            mma_shape = Index(128, 64, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
            num_split_k=2,
        ](ctx, dynamic(256), static[256](), static[512]())

        # ============================================================
        # Test 7: Large MMA shape (stress test)
        # ============================================================
        print("--- Test 7: Large MMA shape (128x128) ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 128, BK),
            mma_shape = Index(128, 128, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
        ](ctx, dynamic(512), static[512](), static[512]())

        # ============================================================
        # Test 8: Dynamic M with misaligned size (common inference)
        # ============================================================
        print("--- Test 8: Misaligned M dimension (inference pattern) ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 64, BK),
            mma_shape = Index(64, 64, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
        ](ctx, dynamic(317), static[512](), static[256]())

        # ============================================================
        # Test 9: Small block, large cluster (multicast stress)
        # ============================================================
        print("--- Test 9: Small block, large cluster (8x2x1) ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(64, 32, BK),
            mma_shape = Index(64, 32, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](8, 2, 1),
            cta_group=1,
            block_swizzle_size=2,
        ](ctx, dynamic(256), static[256](), static[128]())

        # ============================================================
        # Test 10: 2SM with swapAB (combined features)
        # ============================================================
        print("--- Test 10: 2SM + swapAB combined ---")
        test_blackwell_matmul[
            dtype,
            dtype,
            DType.bfloat16,
            block_tile_shape = Index(128, 64, BK),
            mma_shape = Index(256, 128, MMA_K),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
            swapAB=True,
        ](ctx, dynamic(256), static[512](), static[512]())

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 60)
