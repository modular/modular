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
from layout import TensorLayout, Coord, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import stack_allocation
from layout._fillers import arange
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
)
from wgmma_test_utils import _lhs_descriptor, _rhs_descriptor
from std.testing import assert_almost_equal

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
    a_gmem: TileTensor[a_type, type_of(a_layout), MutAnyOrigin],
    b_gmem: TileTensor[b_type, type_of(b_layout), MutAnyOrigin],
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
        dtype=.bfloat16, address_space=.SHARED, alignment=128
    ](a_smem_layout)

    var b_smem_tile = stack_allocation[
        dtype=.bfloat16, address_space=.SHARED, alignment=128
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
            a_type=DType.bfloat16,
            b_type=DType.bfloat16,
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


def wgmma_bf16_bf16_f32[
    M: Int, N: Int, K: Int, transpose_b: Bool = False, a_reg: Bool = False
](ctx: DeviceContext) raises:
    print(
        "== wgmma_bf16_bf16_f32_64xNx16(N, r/s) => ",
        N,
        ", r" if a_reg else ", s",
        sep="",
    )

    var a = HostDeviceTileTensor[.bfloat16, type_of(row_major[M, K]())](
        row_major[M, K](), ctx
    )
    arange(a.host_tensor())

    var b = HostDeviceTileTensor[.bfloat16, type_of(row_major[N, K]())](
        row_major[N, K](), ctx
    )
    arange(b.host_tensor())

    var c = HostDeviceTileTensor[.bfloat16, type_of(row_major[M, N]())](
        row_major[M, N](), ctx
    )
    var c_ref = HostDeviceTileTensor[.bfloat16, type_of(row_major[M, N]())](
        row_major[M, N](), ctx
    )

    comptime a_smem_layout = tile_layout_k_major_typed[.bfloat16, BM=M, BK=16]

    comptime b_smem_layout = tile_layout_k_major_typed[.bfloat16, BM=N, BK=16]

    comptime kernel = wgmma_kernel_ss[
        type_of(a_smem_layout),
        type_of(b_smem_layout),
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        row_major[M, K](),
        row_major[N, K](),
        row_major[M, N](),
        M,
        N,
        K,
        a_smem_layout,
        b_smem_layout,
        transpose_b=transpose_b,
    ]

    a.to_device()
    b.to_device()
    ctx.enqueue_function[kernel](
        a.device_tensor().as_unsafe_any_origin(),
        b.device_tensor().as_unsafe_any_origin(),
        c.device_tensor().as_unsafe_any_origin(),
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()

    var a_buf = a.device_tensor()
    var b_buf = b.device_tensor()
    var c_ref_buf = c_ref.device_tensor()

    vendor_blas.matmul(
        ctx,
        c_ref_buf,
        a_buf,
        b_buf,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    c.to_host()
    c_ref.to_host()

    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                c_ref.host_tensor()[m, n],
                c.host_tensor()[m, n],
                atol=1e-3,
                rtol=1e-3,
            )

    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^


def main() raises:
    with DeviceContext() as ctx:
        comptime for n in range(8, 264, 8):
            wgmma_bf16_bf16_f32[64, n, 16, True](ctx)
