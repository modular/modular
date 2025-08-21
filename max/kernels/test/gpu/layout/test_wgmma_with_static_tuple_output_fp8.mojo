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

import linalg.vendor_blas
from buffer import DimList
from gpu import barrier
from gpu.host import DeviceContext

# from testing import assert_almost_equal
from gpu.id import thread_idx
from gpu.memory import AddressSpace
from gpu.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_equal,
    random,
    zero,
)
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.tensor_core_async import (
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)

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
    a_gmem: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b_gmem: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    c_gmem: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
):
    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    alias num_output_regs = WMMA_M * WMMA_N // 128
    var c_reg = StaticTuple[Scalar[DType.float32], num_output_regs](0)

    alias M = a_layout.shape[0].value()
    alias K = a_layout.shape[1].value()
    alias N = c_layout.shape[1].value()

    alias b_tile_dim0 = N if transpose_b else WMMA_K
    alias b_tile_dim1 = WMMA_K if transpose_b else N

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

    var warp_id = thread_idx.x // 32
    var lane_id = thread_idx.x % 32

    var th_local_res = (
        c_gmem.tile[16, WMMA_N](warp_id, 0)
        .vectorize[1, 2]()
        .distribute[Layout.row_major(8, 4)](lane_id)
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

    alias static_a_shape = DimList(M, K)
    alias static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    alias static_c_shape = DimList(M, N)

    var a_host = HostNDBuffer[DType.float8_e4m3fn, 2, static_a_shape]()
    var b_host = HostNDBuffer[DType.float8_e4m3fn, 2, static_b_shape]()
    var c_host = HostNDBuffer[c_type, 2, static_c_shape]()
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape]()

    var a_device = DeviceNDBuffer[DType.float8_e4m3fn, 2, static_a_shape](
        ctx=ctx
    )
    var b_device = DeviceNDBuffer[DType.float8_e4m3fn, 2, static_b_shape](
        ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](ctx=ctx)
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](ctx=ctx)

    # Initialize matmul operands
    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    var c_tensor = from_ndbuffer_row_major(c_device.tensor)
    var a_tensor = from_ndbuffer_row_major(a_device.tensor)
    var b_tensor = from_ndbuffer_row_major(b_device.tensor)

    alias a_smem_layout = tile_layout_k_major[
        DType.float8_e4m3fn, BM=M, BK=32
    ]()
    alias b_smem_layout = tile_layout_k_major[
        DType.float8_e4m3fn, BM=N, BK=32
    ]()

    alias kernel = wgmma_kernel_ss[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        c_type,
        Layout.row_major(M, K),
        Layout.row_major(N, K) if transpose_b else Layout.row_major(K, N),
        Layout.row_major(M, N),
        M,
        N,
        K,
        a_smem_layout,
        b_smem_layout,
        transpose_b=transpose_b,
    ]

    ctx.enqueue_function[kernel](
        a_tensor,
        b_tensor,
        c_tensor,
        grid_dim=(1, 1),
        block_dim=(128),
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    if transpose_b:
        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=True,
        )

    else:
        # TODO: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = HostNDBuffer[
            DType.float8_e4m3fn, 2, DimList(N, K)
        ]()

        for i in range(N):
            for j in range(K):
                b_host_col_major.tensor[i, j] = b_host.tensor[j, i]

        var b_device_col_major = DeviceNDBuffer[
            DType.float8_e4m3fn, 2, DimList(N, K)
        ](ctx=ctx)
        ctx.enqueue_copy(
            b_device_col_major.buffer, b_host_col_major.tensor.data
        )

        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device_col_major.tensor,
            c_row_major=True,
            transpose_b=True,
        )

    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    assert_equal(c_host.tensor, c_host_ref.tensor)

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor


fn main() raises:
    with DeviceContext() as ctx:

        @parameter
        for n in range(8, 32, 8):
            wgmma_e4m3_e4m3_f32[
                64,
                n,
                32,
                DType.bfloat16,
                True,
            ](ctx)
