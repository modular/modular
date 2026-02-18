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
"""Blockwise FP8 2SM tests - Part A (mma_m_scale=1, 128xN MMA shapes)."""

from math import ceildiv
from sys import argv, size_of
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
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
from utils.static_tuple import StaticTuple


fn simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


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

    comptime static_a_scales_shape = DimList(
        ceildiv(Int(k.dim), BLOCK_SCALE_K), m.dim
    )
    comptime static_b_scales_shape = DimList(
        ceildiv(Int(n.dim), BLOCK_SCALE_K), ceildiv(Int(k.dim), BLOCK_SCALE_K)
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(k.value, BLOCK_SCALE_K), m.value
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, BLOCK_SCALE_K), ceildiv(k.value, BLOCK_SCALE_K)
    )

    var a_size = m.value * k.value
    var b_size = n.value * k.value if transpose_b else k.value * n.value
    var c_size = m.value * n.value
    var a_scales_size = ceildiv(k.value, BLOCK_SCALE_K) * m.value
    var b_scales_size = ceildiv(n.value, BLOCK_SCALE_K) * ceildiv(
        k.value, BLOCK_SCALE_K
    )

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

    var a_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        a_scales_size
    )
    var a_scales_host = NDBuffer[scales_type, 2, _, static_a_scales_shape](
        a_scales_host_ptr, dynamic_a_scales_shape
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        b_scales_size
    )
    var b_scales_host = NDBuffer[scales_type, 2, _, static_b_scales_shape](
        b_scales_host_ptr, dynamic_b_scales_shape
    )

    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var a_scales_device_nd = NDBuffer[scales_type, 2, _, static_a_scales_shape](
        a_scales_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_size)
    var b_scales_device_nd = NDBuffer[scales_type, 2, _, static_b_scales_shape](
        b_scales_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    c_host.zero()
    c_host_ref.zero()

    # Initialize matmul operands
    if simple_init():
        for m in range(M):
            for k in range(K):
                a_host[m, k] = Scalar[a_type](1.0)
        for n in range(N):
            for k in range(K):
                b_host[n, k] = Scalar[b_type](1.0)

        for m in range(M):
            for k in range(K):
                a_scales_host[k // BLOCK_SCALE_K, m] = Scalar[scales_type](0.5)
        for n in range(N):
            for k in range(K):
                b_scales_host[n // BLOCK_SCALE_K, k // BLOCK_SCALE_K] = Scalar[
                    scales_type
                ](0.5)

    else:
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


def main():
    with DeviceContext() as ctx:
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime in_dtype = DType.float8_e4m3fn
        comptime BK = (swizzle.bytes() // size_of[in_dtype]())
        comptime MMA_K = 32
        comptime out_dtype = DType.bfloat16

        # Part A: mma_m_scale = 1 only (128x16N MMA shapes for 2SM)
        @parameter
        for mma_n_scale in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
            comptime block_tile_shape = Index(64, 8 * mma_n_scale, BK)
            comptime umma_shape = Index(128, 16 * mma_n_scale, MMA_K)

            print(
                "block_tile_shape",
                block_tile_shape,
                "umma_shape",
                umma_shape,
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=2,
            ](
                ctx,
                dynamic(1000),
                static[576](),
                static[7168](),
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=2,
            ](
                ctx,
                dynamic(1000),
                static[576](),
                static[256 + 64](),
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                scales_type = DType.bfloat16,
                cta_group=2,
            ](
                ctx,
                dynamic(1000),
                static[32768](),
                static[512](),
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
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

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=2,
            ](
                ctx,
                dynamic(500),
                static[24576](),
                static[1536](),
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](8, 2, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=2,
            ](
                ctx,
                dynamic(1024),
                static[1536](),
                static[7168](),
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](2, 2, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                scales_type = DType.bfloat16,
                cta_group=2,
            ](
                ctx,
                static[1024](),
                static[1024](),
                static[2048](),
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=2,
            ](
                ctx,
                dynamic(8192),
                static[2560](),
                static[8192](),
            )
