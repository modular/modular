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

import linalg.matmul.vendor.blas as vendor_blas
from buffer import DimList, NDBuffer
from gpu import barrier, warp_id, lane_id
from gpu.host import DeviceContext

# from testing import assert_almost_equal
from gpu import thread_idx
from gpu.compute.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from internal_utils import assert_equal
from random import rand
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.tensor_core_async import (
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from utils import StaticTuple


fn wgmma_kernel_ss[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    WMMA_M: Int,
    WMMA_N: Int,
    WMMA_K: Int,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    transpose_b: Bool = False,
](
    a_gmem: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b_gmem: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    c_gmem: LayoutTensor[c_type, c_layout, MutAnyOrigin],
):
    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    comptime num_output_regs = WMMA_M * WMMA_N // 128
    var c_reg = StaticTuple[Float32, num_output_regs](0)

    comptime M = a_layout.shape[0].value()
    comptime K = a_layout.shape[1].value()
    comptime N = c_layout.shape[1].value()

    comptime b_tile_dim0 = N if transpose_b else WMMA_K
    comptime b_tile_dim1 = WMMA_K if transpose_b else N

    for k_i in range(K // WMMA_K):
        var a_gmem_tile = a_gmem.tile[M, WMMA_K](0, k_i)

        var b_tile_coord0 = 0 if transpose_b else k_i
        var b_tile_coord1 = k_i if transpose_b else 0
        var b_gmem_tile = b_gmem.tile[b_tile_dim0, b_tile_dim1](
            b_tile_coord0, b_tile_coord1
        )

        if thread_idx.x == 0:
            a_smem_tile.copy_from(a_gmem_tile)
            b_smem_tile.copy_from(b_gmem_tile)

        barrier()

        var mat_a_desc = _lhs_descriptor(a_smem_tile)
        var mat_b_desc = _rhs_descriptor[transpose_b](b_smem_tile)

        wgmma_fence_aligned()

        c_reg = wgmma_async[
            WMMA_M,
            WMMA_N,
            WMMA_K,
            a_type=a_type,
            b_type=b_type,
        ](mat_a_desc, mat_b_desc, c_reg)
        wgmma_commit_group_sync()
        wgmma_wait_group_sync()

    var th_local_res = (
        c_gmem.tile[16, WMMA_N](Int(warp_id()), 0)
        .vectorize[1, 2]()
        .distribute[Layout.row_major(8, 4)](lane_id())
    )

    for i in range(num_output_regs):
        th_local_res[(i // 2) % 2, i // 4][i % 2] = c_reg[i].cast[
            c_gmem.dtype
        ]()


fn wgmma_e4m3_e4m3_f32[
    M: Int,
    N: Int,
    K: Int,
    c_type: DType,
    transpose_b: Bool = False,
    a_reg: Bool = False,
](ctx: DeviceContext) raises:
    print(
        "== wgmma_e4m3_e4m3_f32_64xNx16(N, r/s) => ",
        N,
        ", r" if a_reg else ", s",
        sep="",
    )

    comptime static_a_shape = DimList(M, K)
    comptime static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    comptime static_c_shape = DimList(M, N)

    var a_host_ptr = UnsafePointer[Scalar[DType.float8_e4m3fn]].alloc(M * K)
    var a_host = NDBuffer[DType.float8_e4m3fn, 2, _, static_a_shape](a_host_ptr)
    var b_size = N * K if transpose_b else K * N
    var b_host_ptr = UnsafePointer[Scalar[DType.float8_e4m3fn]].alloc(b_size)
    var b_host = NDBuffer[DType.float8_e4m3fn, 2, _, static_b_shape](b_host_ptr)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(M * N)
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](c_host_ptr)
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(M * N)
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](c_host_ref_ptr)

    var a_device = ctx.enqueue_create_buffer[DType.float8_e4m3fn](M * K)
    var a_device_nd = NDBuffer[DType.float8_e4m3fn, 2, _, static_a_shape](
        a_device.unsafe_ptr()
    )
    var b_device = ctx.enqueue_create_buffer[DType.float8_e4m3fn](b_size)
    var b_device_nd = NDBuffer[DType.float8_e4m3fn, 2, _, static_b_shape](
        b_device.unsafe_ptr()
    )
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr()
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr()
    )

    # Initialize matmul operands
    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())
    c_host.zero()
    c_host_ref.zero()

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    var c_tensor = from_ndbuffer_row_major(c_device_nd)
    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var b_tensor = from_ndbuffer_row_major(b_device_nd)

    comptime a_smem_layout = tile_layout_k_major[
        DType.float8_e4m3fn, BM=M, BK=32
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        DType.float8_e4m3fn, BM=N, BK=32
    ]()

    comptime kernel = wgmma_kernel_ss[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        c_type,
        a_tensor.layout,
        b_tensor.layout,
        c_tensor.layout,
        M,
        N,
        K,
        a_smem_layout,
        b_smem_layout,
        transpose_b=transpose_b,
    ]

    ctx.enqueue_function_experimental[kernel](
        a_tensor,
        b_tensor,
        c_tensor,
        grid_dim=(1, 1),
        block_dim=(128),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    if transpose_b:
        vendor_blas.matmul(
            ctx,
            c_device_ref_nd,
            a_device_nd,
            b_device_nd,
            c_row_major=True,
            transpose_b=True,
        )

    else:
        # TODO: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major_ptr = UnsafePointer[
            Scalar[DType.float8_e4m3fn]
        ].alloc(N * K)
        var b_host_col_major = NDBuffer[
            DType.float8_e4m3fn, 2, _, DimList(N, K)
        ](b_host_col_major_ptr)

        for i in range(N):
            for j in range(K):
                b_host_col_major[i, j] = b_host[j, i]

        var b_device_col_major = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
            N * K
        )
        var b_device_col_major_nd = NDBuffer[
            DType.float8_e4m3fn, 2, _, DimList(N, K)
        ](b_device_col_major.unsafe_ptr())
        ctx.enqueue_copy(b_device_col_major, b_host_col_major_ptr)

        vendor_blas.matmul(
            ctx,
            c_device_ref_nd,
            a_device_nd,
            b_device_col_major_nd,
            c_row_major=True,
            transpose_b=True,
        )

        b_host_col_major_ptr.free()
        _ = b_device_col_major^

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_equal(c_host.data, c_host_ref.data, c_host.num_elements())

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
        comptime for n in range(8, 32, 8):
            wgmma_e4m3_e4m3_f32[
                64,
                n,
                32,
                DType.bfloat16,
                True,
            ](ctx)
