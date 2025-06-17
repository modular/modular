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

from sys import sizeof

from gpu import barrier
from gpu.cluster import block_rank_in_cluster, cluster_sync
from gpu.host import DeviceContext, Dim
from gpu.host._compile import get_gpu_target
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, cluster_dim, cluster_idx, thread_idx
from gpu.memory import fence_mbarrier_init, tma_store_fence
from gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from layout import Layout, LayoutTensor
from layout._fillers import arange, random
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_swizzle
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from testing import assert_equal

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn tma_swizzle_multicast_load_kernel[
    dtype: DType,
    layout: Layout,
    cluster_tile_layout: Layout,
    subcluster_tile_layout: Layout,
    desc_layout: Layout,
    CLUSTER_M: UInt,
    CLUSTER_N: UInt,
](
    dst: LayoutTensor[dtype, layout, MutableAnyOrigin],
    tma_tile: TMATensorTile[dtype, subcluster_tile_layout, desc_layout],
):
    alias cluster_tileM = cluster_tile_layout.shape[0].value()
    alias cluster_tileN = cluster_tile_layout.shape[1].value()
    alias expected_bytes = cluster_tile_layout.size() * sizeof[dtype]()

    alias subcluster_tileM = subcluster_tile_layout.shape[0].value()
    alias subcluster_tileN = subcluster_tile_layout.shape[1].value()

    var block_rank = block_rank_in_cluster()
    var rank_m = Int(block_rank // CLUSTER_N)
    var rank_n = Int(block_rank % CLUSTER_N)

    alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N
    var tma_multicast_mask = (1 << CLUSTER_SIZE) - 1

    tile = LayoutTensor[
        dtype,
        cluster_tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    barrier()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    barrier()

    # we use cluster_sync() together with a mbarrier init fence to ensure cluster-wide visibility of the mbarrier initialization
    cluster_sync()
    fence_mbarrier_init()

    if thread_idx.x == 0:
        mbar[0].expect_bytes(expected_bytes)
        var slice_cord_y = (
            cluster_idx.y * cluster_tileM + rank_m * subcluster_tileM
        )
        var slice_cord_x = (
            cluster_idx.x * cluster_tileN + rank_n * subcluster_tileN
        )
        var copy_offset = subcluster_tileM * subcluster_tileN * block_rank

        tma_tile.async_multicast_load(
            __type_of(tile)(tile.ptr + copy_offset),
            mbar[0],
            (slice_cord_x, slice_cord_y),
            tma_multicast_mask,
        )

    barrier()

    mbar[0].wait()

    # we use another cluster_sync() to ensure that none of CTAs in the cluster doesn’t exit prematurely while the other is still waiting for the multicast load to complete.
    cluster_sync()
    fence_mbarrier_init()

    if block_rank == 0 and thread_idx.x == 0:
        dst_tile = dst.tile[cluster_tileM, cluster_tileN](
            cluster_idx.y, cluster_idx.x
        )
        dst_tile.copy_from(tile)


def test_tma_multicast_swizzle[
    dtype: DType,
    shape: IndexList[2],
    cluster_tile_shape: IndexList[2],
    CLUSTER_M: UInt,
    CLUSTER_N: UInt,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext):
    alias tileM = cluster_tile_shape[0]
    alias tileN = cluster_tile_shape[1]
    alias subcluster_tile_shape = Index(tileM // CLUSTER_M, tileN // CLUSTER_N)

    alias layout = Layout.row_major(shape[0], shape[1])
    var src = ManagedLayoutTensor[dtype, layout](ctx)
    var dst = ManagedLayoutTensor[dtype, layout](ctx)  # FIX THIS

    @parameter
    if dtype is DType.float8_e4m3fn:
        random(src.tensor())
        random(dst.tensor())
    else:
        arange(src.tensor(), 0)
        arange(dst.tensor(), 0)

    var tma_tensor = create_tma_tile[
        dtype, 2, subcluster_tile_shape, swizzle_mode=swizzle_mode
    ](ctx, src.device_tensor())

    # print test info
    alias use_multiple_loads = (
        tma_tensor.layout.size() > tma_tensor.desc_layout.size()
    )
    alias test_name = "test " + String(dtype) + (
        " multiple " if use_multiple_loads else " single "
    ) + "tma w/ " + String(swizzle_mode) + " multicast"
    print(test_name)

    alias kernel = tma_swizzle_multicast_load_kernel[
        dtype = __type_of(tma_tensor).dtype,
        layout=layout,
        cluster_tile_layout = Layout.row_major(tileM, tileN),
        subcluster_tile_layout = __type_of(tma_tensor).layout,
        desc_layout = __type_of(tma_tensor).desc_layout,
        CLUSTER_M=CLUSTER_M,
        CLUSTER_N=CLUSTER_N,
    ]
    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(
            (shape[1] // cluster_tile_shape[1]) * CLUSTER_N,
            (shape[0] // cluster_tile_shape[0]) * CLUSTER_M,
        ),
        block_dim=(1),
        cluster_dim=Dim(CLUSTER_N, CLUSTER_M, 1),
    )

    ctx.synchronize()
    # Descriptor tile is the copy per tma instruction. One load could have multiple tma copies.
    alias descM = __type_of(tma_tensor).desc_layout.shape[0].value()
    alias descN = __type_of(tma_tensor).desc_layout.shape[1].value()
    alias desc_tile_size = descM * descN

    desc_tile = LayoutTensor[
        dtype, __type_of(tma_tensor).desc_layout, MutableAnyOrigin
    ].stack_allocation()

    src_host = src.tensor()
    dst_host = dst.tensor()

    alias swizzle = make_swizzle[dtype, swizzle_mode]()

    dest_tile = LayoutTensor[
        dtype, Layout.row_major(tileM, tileN), MutableAnyOrigin
    ].stack_allocation()
    src_tile = LayoutTensor[
        dtype, Layout.row_major(tileM, tileN), MutableAnyOrigin
    ].stack_allocation()

    for dest_tile_m in range(shape[0] // tileM):
        for dest_tile_n in range(shape[1] // tileN):
            dest_tile.copy_from(
                dst_host.tile[tileM, tileN](dest_tile_m, dest_tile_n)
            )
            src_tile.copy_from(
                src_host.tile[tileM, tileN](dest_tile_m, dest_tile_n)
            )

            dst_tile_ptr = dest_tile.ptr
            for desc_tile_m in range(tileM // descM):
                for desc_tile_n in range(tileN // descN):
                    desc_tile.copy_from(
                        src_tile.tile[descM, descN](desc_tile_m, desc_tile_n)
                    )
                    for i in range(desc_tile_size):
                        desc_idx = swizzle(i)
                        assert_equal(
                            desc_tile.ptr[desc_idx].cast[DType.float64](),
                            dst_tile_ptr[i].cast[DType.float64](),
                        )
                    dst_tile_ptr += desc_tile_size

    _ = src^
    _ = dst^


def main():
    with DeviceContext() as ctx:
        print("bfloat16 single tma w/ no swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 8),
            cluster_tile_shape = Index(16, 8),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 32),
            cluster_tile_shape = Index(16, 16),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 32),
            cluster_tile_shape = Index(8, 16),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        print("bfloat16 multi tma w/ no swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 16),
            cluster_tile_shape = Index(16, 16),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 64),
            cluster_tile_shape = Index(16, 32),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 64),
            cluster_tile_shape = Index(8, 32),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        print("bfloat16 single tma w/ 32B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 16),
            cluster_tile_shape = Index(16, 16),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 64),
            cluster_tile_shape = Index(16, 32),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 64),
            cluster_tile_shape = Index(8, 32),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        print("bfloat16 multi tma w/ 32B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 32),
            cluster_tile_shape = Index(16, 32),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 128),
            cluster_tile_shape = Index(16, 64),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 128),
            cluster_tile_shape = Index(8, 64),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        print("bfloat16 single tma w/ 64B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 32),
            cluster_tile_shape = Index(16, 32),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 128),
            cluster_tile_shape = Index(16, 64),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 128),
            cluster_tile_shape = Index(8, 64),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        print("bfloat16 multi tma w/ 64B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 64),
            cluster_tile_shape = Index(16, 64),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 256),
            cluster_tile_shape = Index(16, 128),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 256),
            cluster_tile_shape = Index(8, 128),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        print("bfloat16 single tma w/ 128B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 64),
            cluster_tile_shape = Index(16, 64),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 256),
            cluster_tile_shape = Index(16, 128),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 256),
            cluster_tile_shape = Index(8, 128),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print("bfloat16 multi tma w/ 128B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 128),
            cluster_tile_shape = Index(16, 128),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(32, 512),
            cluster_tile_shape = Index(16, 256),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.bfloat16,
            shape = Index(8, 512),
            cluster_tile_shape = Index(8, 256),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print("float8_e4m3fn single tma w/ no swizzle multicast")
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 16),
            cluster_tile_shape = Index(16, 16),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 64),
            cluster_tile_shape = Index(16, 32),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 64),
            cluster_tile_shape = Index(8, 32),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        print("float8_e4m3fn single tma w/ 32B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 32),
            cluster_tile_shape = Index(16, 32),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 128),
            cluster_tile_shape = Index(16, 64),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 128),
            cluster_tile_shape = Index(8, 64),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        print("float8_e4m3fn single tma w/ 64B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 64),
            cluster_tile_shape = Index(16, 64),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 256),
            cluster_tile_shape = Index(16, 128),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 256),
            cluster_tile_shape = Index(8, 128),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        print("float8_e4m3fn single tma w/ 128B swizzle multicast")
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 128),
            cluster_tile_shape = Index(16, 128),
            CLUSTER_M=2,
            CLUSTER_N=1,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(32, 512),
            cluster_tile_shape = Index(16, 256),
            CLUSTER_M=2,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
        test_tma_multicast_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 512),
            cluster_tile_shape = Index(8, 256),
            CLUSTER_M=1,
            CLUSTER_N=2,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
