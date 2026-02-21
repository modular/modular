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
"""Minimal smoke test for blockwise FP8 - representative configs for fast iteration.

Covers:
- 1SM path (cta_group=1): block_tile_shape == umma_shape
- 2SM path (cta_group=2): umma_shape = 2x block_tile_shape

Target: < 1 minute compile + run for debugging purposes.
"""

from math import align_up, ceildiv
from sys import size_of
from linalg.matmul.gpu.sm100.config import MatmulConfig
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import (
    assert_almost_equal,
    assert_with_measure,
)
from random import rand
from internal_utils._measure import relative_difference
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    lt_to_tt,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul import (
    blockwise_fp8_matmul,
)

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


fn test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    scales_type: DType = DType.float32,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    var M = m.value
    var N = n.value
    var K = k.value

    comptime BLOCK_SCALE_K = 128
    comptime accum_type = get_accum_type[c_type]()

    if M * size_of[DType.float32]() % 16 != 0:
        raise Error("TMA expects M to be divisible by 16 bytes")

    print(
        "in/out dtypes=(",
        a_type,
        ", ",
        b_type,
        ", ",
        c_type,
        ") ",
        " problem shape=(",
        M,
        ", ",
        N,
        ", ",
        K,
        ") ",
        "mma_shape=",
        mma_shape,
        " block_tile_shape=",
        block_tile_shape,
        " cta_group=",
        cta_group,
        " cluster_shape=(",
        cluster_shape[0],
        ", ",
        cluster_shape[1],
        ", ",
        cluster_shape[2],
        ")",
    )

    # Allocate host memory
    a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(M * K)
    b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(N * K)
    c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(M * N)
    c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(M * N)

    a_host = NDBuffer[a_type, 2](a_host_ptr, DimList(M, K))
    b_host = NDBuffer[b_type, 2](b_host_ptr, DimList(N, K))
    c_host = NDBuffer[c_type, 2](c_host_ptr, DimList(M, N))
    c_host_ref = NDBuffer[c_type, 2](c_host_ref_ptr, DimList(M, N))

    # Calculate scales dimensions
    var a_scales_shape_k = ceildiv(K, BLOCK_SCALE_K)
    var b_scales_shape_n = ceildiv(N, BLOCK_SCALE_K)
    var b_scales_shape_k = ceildiv(K, BLOCK_SCALE_K)

    a_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        a_scales_shape_k * M
    )
    b_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        b_scales_shape_n * b_scales_shape_k
    )

    a_scales_host = NDBuffer[scales_type, 2](
        a_scales_host_ptr, DimList(a_scales_shape_k, M)
    )
    b_scales_host = NDBuffer[scales_type, 2](
        b_scales_host_ptr, DimList(b_scales_shape_n, b_scales_shape_k)
    )

    # Allocate device memory
    a_device = ctx.enqueue_create_buffer[a_type](M * K)
    b_device = ctx.enqueue_create_buffer[b_type](N * K)
    c_device = ctx.enqueue_create_buffer[c_type](M * N)
    c_device_ref = ctx.enqueue_create_buffer[c_type](M * N)
    a_scales_device = ctx.enqueue_create_buffer[scales_type](
        a_scales_shape_k * M
    )
    b_scales_device = ctx.enqueue_create_buffer[scales_type](
        b_scales_shape_n * b_scales_shape_k
    )

    dynamic_a_shape = DimList(M, K)
    dynamic_b_shape = DimList(N, K)
    dynamic_c_shape = DimList(M, N)
    dynamic_a_scales_shape = DimList(a_scales_shape_k, M)
    dynamic_b_scales_shape = DimList(b_scales_shape_n, b_scales_shape_k)

    a_device_nd = NDBuffer[a_type, 2](a_device.unsafe_ptr(), dynamic_a_shape)
    b_device_nd = NDBuffer[b_type, 2](b_device.unsafe_ptr(), dynamic_b_shape)
    c_device_nd = NDBuffer[c_type, 2](c_device.unsafe_ptr(), dynamic_c_shape)
    c_device_ref_nd = NDBuffer[c_type, 2](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )
    a_scales_device_nd = NDBuffer[scales_type, 2](
        a_scales_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    b_scales_device_nd = NDBuffer[scales_type, 2](
        b_scales_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    c_host.zero()
    c_host_ref.zero()

    # Initialize with random data
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())
    rand(a_scales_host.data, a_scales_host.num_elements())
    rand(b_scales_host.data, b_scales_host.num_elements())

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a_lt = from_ndbuffer_row_major(a_device_nd)
    var b_lt = from_ndbuffer_row_major(b_device_nd)
    var c_lt = from_ndbuffer_row_major(c_device_nd)
    var a_scales_lt = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales_lt = from_ndbuffer_row_major(b_scales_device_nd)
    var a = lt_to_tt(a_lt)
    var b = lt_to_tt(b_lt)
    var c = lt_to_tt(c_lt)
    var a_scales = lt_to_tt(a_scales_lt)
    var b_scales = lt_to_tt(b_scales_lt)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
    )

    blockwise_fp8_matmul[
        transpose_b=transpose_b,
        a_scales_type=scales_type,
        b_scales_type=scales_type,
        config=matmul_config,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        ctx,
    )

    # Reference implementation
    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk = Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
    ](
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        a_scales_device_nd,
        b_scales_device_nd,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_with_measure[relative_difference](
        c_host.data, c_host_ref.data, c_host.num_elements(), threshold=0.001
    )

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^

    print("PASSED")


def main():
    with DeviceContext() as ctx:
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime in_dtype = DType.float8_e4m3fn
        comptime BK = (swizzle.bytes() // size_of[in_dtype]())
        comptime MMA_K = 32
        comptime out_dtype = DType.bfloat16

        print("=== SMOKE TEST: 1SM and 2SM Configurations ===")

        # ============================================================
        # 1SM PATH (cta_group=1): block_tile_shape == umma_shape
        # From test_matmul_sm100_1sm_blockwise_fp8.mojo
        # ============================================================
        print("\n--- 1SM Tests (cta_group=1) ---")

        # Config: mma_m_scale=1, mma_n_scale=2
        # block_tile_shape = (64, 16, 128), umma_shape = (64, 16, 32)
        comptime block_tile_1sm = Index(64, 16, BK)
        comptime umma_shape_1sm = Index(64, 16, MMA_K)

        print("block_tile_shape", block_tile_1sm, "umma_shape", umma_shape_1sm)

        # Shape from 1sm test: (1000, 576, 7168)
        _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
            in_dtype,
            in_dtype,
            out_dtype,
            block_tile_1sm,
            umma_shape_1sm,
            cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=1,
        ](
            ctx,
            dynamic(512),
            static[576](),
            static[512](),
        )

        # ============================================================
        # 2SM PATH (cta_group=2): umma_shape = 2x block_tile_shape
        # From test_matmul_sm100_2sm_blockwise_fp8.mojo
        # ============================================================
        print("\n--- 2SM Tests (cta_group=2) ---")

        # Config: mma_m_scale=1, mma_n_scale=2
        # block_tile_shape = (64, 16, 128), umma_shape = (128, 32, 32)
        comptime block_tile_2sm = Index(64, 16, BK)
        comptime umma_shape_2sm = Index(128, 32, MMA_K)

        print("block_tile_shape", block_tile_2sm, "umma_shape", umma_shape_2sm)

        # Shape from 2sm test: (1000, 576, 7168) -> smaller for smoke
        _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
            in_dtype,
            in_dtype,
            out_dtype,
            block_tile_2sm,
            umma_shape_2sm,
            cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=2,
        ](
            ctx,
            dynamic(512),
            static[576](),
            static[512](),
        )

        # Additional 2SM test with larger cluster (4,4,1) from original tests
        _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
            in_dtype,
            in_dtype,
            out_dtype,
            block_tile_2sm,
            umma_shape_2sm,
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=2,
        ](
            ctx,
            dynamic(512),
            static[4096](),
            static[1024](),
        )

        print("\n=== ALL SMOKE TESTS PASSED ===")
