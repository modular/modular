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
from max.gpu import warp_id, lane_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext

from max.gpu import thread_idx
from max.gpu.compute.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from internal_utils import assert_equal
from std.random import rand
from layout import TensorLayout, Coord, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import stack_allocation
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
)
from wgmma_test_utils import _lhs_descriptor, _rhs_descriptor
from std.utils import StaticTuple


def wgmma_kernel_ss[
    ASmemLayout: TensorLayout,
    BSmemLayout: TensorLayout,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: TileLayout,
    b_layout: TileLayout,
    c_layout: TileLayout,
    WMMA_M: Int,
    WMMA_N: Int,
    WMMA_K: Int,
    a_smem_layout: ASmemLayout,
    b_smem_layout: BSmemLayout,
    transpose_b: Bool = False,
](
    a_gmem: TileTensor[a_type, type_of(a_layout), ImmutAnyOrigin],
    b_gmem: TileTensor[b_type, type_of(b_layout), ImmutAnyOrigin],
    c_gmem: TileTensor[c_type, type_of(c_layout), MutAnyOrigin],
):
    comptime assert ASmemLayout.all_dims_known
    comptime assert BSmemLayout.all_dims_known
    comptime assert a_gmem.rank == a_gmem.flat_rank == 2
    comptime assert b_gmem.rank == b_gmem.flat_rank == 2
    comptime assert type_of(a_gmem).LayoutType.shape_known
    comptime assert type_of(b_gmem).LayoutType.shape_known
    comptime assert c_gmem.rank == c_gmem.flat_rank == 2
    var a_smem_tile = stack_allocation[
        dtype=a_type, address_space=.SHARED, alignment=128
    ](a_smem_layout)

    var b_smem_tile = stack_allocation[
        dtype=b_type, address_space=.SHARED, alignment=128
    ](b_smem_layout)

    comptime num_output_regs = WMMA_M * WMMA_N // 128
    var c_reg = StaticTuple[Float32, num_output_regs](0)

    comptime M = Int(a_layout.shape[0]().value())
    comptime K = Int(a_layout.shape[1]().value())
    comptime N = Int(c_layout.shape[1]().value())

    comptime b_tile_dim0 = N if transpose_b else WMMA_K
    comptime b_tile_dim1 = WMMA_K if transpose_b else N

    for k_i in range(K // WMMA_K):
        var a_gmem_tile = a_gmem.tile[M, WMMA_K](Coord(0, k_i))

        var b_tile_coord0 = 0 if transpose_b else k_i
        var b_tile_coord1 = k_i if transpose_b else 0
        var b_gmem_tile = b_gmem.tile[b_tile_dim0, b_tile_dim1](
            Coord(b_tile_coord0, b_tile_coord1)
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
        c_gmem.tile[16, WMMA_N](Coord(warp_id(), 0))
        .vectorize[1, 2]()
        .distribute[row_major[8, 4]()](lane_id())
    )

    for i in range(num_output_regs):
        th_local_res[(i // 2) % 2, i // 4][i % 2] = c_reg[i].cast[
            c_gmem.dtype
        ]()


def wgmma_e4m3_e4m3_f32[
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

    comptime a_shape = row_major[M, K]()
    comptime b_shape = row_major[
        N if transpose_b else K, K if transpose_b else N
    ]()
    comptime c_shape = row_major[M, N]()

    var a_host_ptr = ctx.enqueue_create_host_buffer[.float8_e4m3fn](M * K)
    var b_size = N * K if transpose_b else K * N
    var b_host_ptr = ctx.enqueue_create_host_buffer[.float8_e4m3fn](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[.float8_e4m3fn](M * K)
    var a_device_tt = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[.float8_e4m3fn](b_size)
    var b_device_tt = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_tt = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_ref_tt = TileTensor(c_device_ref, c_shape)

    # Initialize matmul operands
    rand(a_host_ptr.unsafe_ptr(), M * K)
    rand(b_host_ptr.unsafe_ptr(), b_size)
    # c buffers don't need init — overwritten by copy from device.

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime a_smem_layout = tile_layout_k_major_typed[
        DType.float8_e4m3fn, BM=M, BK=32
    ]
    comptime b_smem_layout = tile_layout_k_major_typed[
        DType.float8_e4m3fn, BM=N, BK=32
    ]

    comptime kernel = wgmma_kernel_ss[
        type_of(a_smem_layout),
        type_of(b_smem_layout),
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        c_type,
        a_shape,
        b_shape,
        c_shape,
        M,
        N,
        K,
        a_smem_layout,
        b_smem_layout,
        transpose_b=transpose_b,
    ]

    ctx.enqueue_function[kernel](
        a_device_tt.as_imm().as_unsafe_any_origin(),
        b_device_tt.as_imm().as_unsafe_any_origin(),
        c_device_tt.as_unsafe_any_origin(),
        grid_dim=(1, 1),
        block_dim=(128),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    if transpose_b:
        vendor_blas.matmul(
            ctx,
            c_device_ref_tt,
            a_device_tt,
            b_device_tt,
            c_row_major=True,
            transpose_b=True,
        )

    else:
        # TODO: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major_ptr = ctx.enqueue_create_host_buffer[
            DType.float8_e4m3fn
        ](N * K)

        for i in range(N):
            for j in range(K):
                b_host_col_major_ptr[i * K + j] = b_host_ptr[j * N + i]

        var b_device_col_major = ctx.enqueue_create_buffer[.float8_e4m3fn](
            N * K
        )
        var b_device_col_major_tt = TileTensor(
            b_device_col_major, row_major[N, K]()
        )
        ctx.enqueue_copy(b_device_col_major, b_host_col_major_ptr)

        vendor_blas.matmul(
            ctx,
            c_device_ref_tt,
            a_device_tt,
            b_device_col_major_tt,
            c_row_major=True,
            transpose_b=True,
        )

        _ = b_device_col_major^

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_equal(c_host._storage, c_host_ref._storage, c_host.num_elements())


def main() raises:
    with DeviceContext() as ctx:
        comptime for n in range(8, 32, 8):
            wgmma_e4m3_e4m3_f32[
                64,
                n,
                32,
                DType.bfloat16,
                True,
            ](ctx)
