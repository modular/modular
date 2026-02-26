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

from sys import argv, size_of

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
from layout.tile_tensor import TileTensor
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_batched_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


fn simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_batched_matmul_tma_umma_warp_specialized[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
](ctx: DeviceContext, batch: ValOrDim, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var B = batch.value
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        String(
            "in/out dtypes=(",
            a_type,
            ", ",
            b_type,
            ", ",
            c_type,
            ") ",
            " problem shape=(",
            B,
            ", ",
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
            " swapAB=",
            swapAB,
            " k_group_size=",
            k_group_size,
        )
    )

    comptime static_a_shape = DimList(batch.dim, m.dim, k.dim)
    comptime static_b_shape = DimList(
        batch.dim, n.dim, k.dim
    ) if transpose_b else DimList(batch.dim, k.dim, n.dim)
    comptime static_c_shape = DimList(batch.dim, m.dim, n.dim)
    var dynamic_a_shape = DimList(batch.value, m.value, k.value)
    var dynamic_b_shape = DimList(
        batch.value, n.value, k.value
    ) if transpose_b else DimList(batch.value, k.value, n.value)
    var dynamic_c_shape = DimList(batch.value, m.value, n.value)

    var a_size = batch.value * m.value * k.value
    var b_size = batch.value * n.value * k.value
    var c_size = batch.value * m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var a_host = NDBuffer[a_type, 3, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var b_host = NDBuffer[b_type, 3, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 3, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 3, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[a_type, 3, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[b_type, 3, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 3, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 3, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    # Initialize matmul operands
    if simple_init():
        for b in range(B):
            for mi in range(M):
                for ki in range(K):
                    a_host[b, mi, ki] = Float32(ki).cast[a_type]()
        for b in range(B):
            for ni in range(N):
                for ki in range(K):
                    b_host[b, ni, ki] = Float32(1 if ni == ki else 0).cast[
                        b_type
                    ]()
    else:
        rand(a_host.data, a_host.num_elements())
        rand(b_host.data, b_host.num_elements())

    # Move operands to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    # TileTensors for the kernel under test
    var a_tensor = TileTensor(a_device_nd)
    var b_tensor = TileTensor(b_device_nd)
    var c_tensor = TileTensor(c_device_nd)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
    )

    blackwell_batched_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](c_tensor, a_tensor, b_tensor, ctx)

    # Reference: per-batch vendor_blas.matmul
    var a_lt = from_ndbuffer_row_major(a_device_nd)
    var b_lt = from_ndbuffer_row_major(b_device_nd)
    var c_ref_lt = from_ndbuffer_row_major(c_device_ref_nd)

    from layout import Layout, LayoutTensor, RuntimeLayout

    @parameter
    fn _reshape_to_2d[layout: Layout]() -> Layout:
        return Layout.row_major(
            layout.shape[1].value(),
            layout.shape[2].value(),
        )

    for b in range(B):
        comptime a_2d_layout = _reshape_to_2d[a_lt.layout]()
        comptime b_2d_layout = _reshape_to_2d[b_lt.layout]()
        comptime c_2d_layout = _reshape_to_2d[c_ref_lt.layout]()

        var a_2d = LayoutTensor[
            a_type, a_2d_layout, address_space = a_lt.address_space
        ](
            a_lt.ptr + b * M * K,
            RuntimeLayout[a_2d_layout].row_major(IndexList[2](M, K)),
        )
        var b_2d = LayoutTensor[
            b_type, b_2d_layout, address_space = b_lt.address_space
        ](
            b_lt.ptr + b * N * K,
            RuntimeLayout[b_2d_layout].row_major(IndexList[2](N, K)),
        )
        var c_ref_2d = LayoutTensor[
            c_type, c_2d_layout, address_space = c_ref_lt.address_space
        ](
            c_ref_lt.ptr + b * M * N,
            RuntimeLayout[c_2d_layout].row_major(IndexList[2](M, N)),
        )

        vendor_blas.matmul(
            ctx,
            c_ref_2d,
            a_2d,
            b_2d,
            c_row_major=True,
            transpose_b=transpose_b,
        )
        ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.0001,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main():
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 16
        comptime cta_group = 2

        # 256x256x16 MMA, 2x1x1 cluster
        comptime for bm in [128]:
            comptime for bn in [64, 128]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(
                    cta_group * bm, cta_group * bn, MMA_K
                )

                # Basic: small batch, small shape
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                ](
                    ctx,
                    dynamic(2),
                    dynamic(128),
                    static[128](),
                    static[128](),
                )

                # Medium
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                ](
                    ctx,
                    dynamic(4),
                    dynamic(256),
                    static[512](),
                    static[256](),
                )

                # Large, non-aligned M
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                    block_swizzle_size=8,
                ](
                    ctx,
                    dynamic(2),
                    dynamic(1000),
                    static[1024](),
                    static[1040](),
                )

                # Large batch
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 2, 1),
                    cta_group=cta_group,
                    block_swizzle_size=4,
                ](
                    ctx,
                    dynamic(16),
                    dynamic(256),
                    static[128](),
                    static[512](),
                )

                # Multi-cluster
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=cta_group,
                    block_swizzle_size=1,
                ](
                    ctx,
                    dynamic(3),
                    dynamic(500),
                    static[2048](),
                    static[4096](),
                )

        # swapAB tests (2SM)
        comptime for bn in [64, 128]:
            comptime block_tile_shape = Index(128, bn, BK)
            comptime umma_shape = Index(cta_group * 128, cta_group * bn, MMA_K)

            test_blackwell_batched_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                cta_group=cta_group,
                swapAB=True,
            ](
                ctx,
                dynamic(2),
                dynamic(128),
                static[128](),
                static[128](),
            )

            test_blackwell_batched_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                cta_group=cta_group,
                swapAB=True,
                block_swizzle_size=4,
            ](
                ctx,
                dynamic(4),
                dynamic(256),
                static[512](),
                static[256](),
            )
