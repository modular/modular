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

from math import align_up
from sys import sizeof

from gpu import WARP_SIZE, barrier
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, lane_id, thread_idx, block_id_in_cluster
from gpu.memory import AddressSpace
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from gpu.cluster import (
    elect_one_sync_with_mask,
    block_rank_in_cluster,
    cluster_sync,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg import vendor_blas
from testing import assert_almost_equal

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn tma_umma_kernel_pair_cta[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias num_m_mmas = BM // mma_shape[0]
    alias num_n_mmas = BN // mma_shape[1]

    alias CLUSTER_M = Int(cluster_shape[0])
    alias CLUSTER_N = Int(cluster_shape[1])

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var smem = external_memory[
        UInt8, address_space = AddressSpace.SHARED, alignment=8
    ]()

    alias a_smem_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_smem_bytes = b_smem_layout.size() * sizeof[b_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()

    var smem_poll = (smem + a_smem_bytes + b_smem_bytes).bitcast[Int64]()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](a_smem.static_alignment_cast[128]())

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](b_smem.static_alignment_cast[128]())

    alias accum_type = get_accum_type[a_type]()

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (smem_poll.bitcast[Int64]() + 4).bitcast[UInt32]()

    alias c_frag_size = MMA_M * MMA_N // 128 // cta_group
    var c_frag = SIMD[accum_type, c_frag_size]()

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    alias expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    var tma_mbar_ptr = smem_poll.bitcast[Int64]()
    var mma_mbar_ptr = smem_poll.bitcast[Int64]() + 2

    tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    alias max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    if elect_one_warp and elect_one_thread:
        tma_mbar[0].init()
        mma_mbar[0].init(cluster_shape[0] // cta_group + cluster_shape[1] - 1)

    cluster_sync()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    tmem_addr = ptr_tmem_addr[0]

    alias a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    alias b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    alias aSBO = a_canonical_layout[0].stride[1].value() * sizeof[a_type]()
    alias aLBO = a_canonical_layout[1].stride[1].value() * sizeof[a_type]()
    alias b_stride01 = b_canonical_layout[0].stride[1].value()
    alias b_stride11 = b_canonical_layout[1].stride[1].value()
    alias b_k_stride = b_stride11 * 2 * sizeof[b_type]()
    alias bSBO = (b_stride01 if transpose_b else b_stride11) * sizeof[b_type]()
    alias bLBO = (b_stride11 if transpose_b else b_stride01) * sizeof[b_type]()

    adesc_base = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](
        a_smem_tile.ptr
    )
    bdesc_base = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](
        b_smem_tile.ptr
    )

    idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (rank_m % cta_group, rank_m // cta_group, rank_n)

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    @parameter
    for i in range(CLUSTER_N):
        a_multicast_mask |= 1 << (i * CLUSTER_M)

    @parameter
    for i in range(CLUSTER_M // cta_group):
        b_multicast_mask |= 1 << (i * cta_group)

    a_multicast_mask <<= rank_m
    b_multicast_mask <<= peer_cta_coord[0]
    b_multicast_mask <<= rank_n * CLUSTER_M

    var a_mma_mask = a_multicast_mask >> peer_cta_coord[0]
    var b_mma_mask = b_multicast_mask >> peer_cta_coord[0]
    var c_mma_mask: UInt16 = (a_mma_mask | a_mma_mask << 1) | (
        b_mma_mask | b_mma_mask << 1
    )

    for i in range(num_iters):
        if elect_one_warp and elect_one_thread:
            if elect_one_cta:
                tma_mbar[0].expect_bytes(expected_bytes)

            var a_gmem_slice_coord = (
                peer_cta_coord[2] * a_tma_rows + block_idx.x * BM
            )
            var b_gmem_slice_coord = (
                peer_cta_coord[1] * b_tma_rows
                + peer_cta_coord[0] * BN
                + block_idx.y * MMA_N
            )

            var a_smem_reshape = a_smem_tile.reshape[Layout.row_major(BM, BK)]()
            var b_smem_reshape = b_smem_tile.reshape[Layout.row_major(BN, BK)]()

            a_tma_op.async_multicast_load[cta_group](
                a_smem_reshape.split[CLUSTER_N]()[peer_cta_coord[2]],
                tma_mbar[0],
                (UInt(i) * BK, a_gmem_slice_coord),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load[cta_group](
                b_smem_reshape.split[CLUSTER_M // cta_group]()[
                    peer_cta_coord[1]
                ],
                tma_mbar[0],
                (UInt(i) * BK, b_gmem_slice_coord),
                b_multicast_mask,
            )

        if elect_one_cta:
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            if elect_one_warp:
                adesc = adesc_base
                bdesc = bdesc_base

                if i == 0:
                    if elect_one_thread:
                        mma[cta_group, c_scale=0](
                            adesc, bdesc, tmem_addr, idesc
                        )

                    @parameter
                    for j in range(1, BK // mma_shape[2]):
                        adesc += mma_shape[2] * sizeof[a_type]()
                        bdesc += b_k_stride
                        if elect_one_thread:
                            mma[cta_group, c_scale=1](
                                adesc, bdesc, tmem_addr, idesc
                            )
                else:

                    @parameter
                    for j in range(BK // mma_shape[2]):
                        if elect_one_thread:
                            mma[cta_group, c_scale=1](
                                adesc, bdesc, tmem_addr, idesc
                            )
                        adesc += mma_shape[2] * sizeof[a_type]()
                        bdesc += b_k_stride

                if elect_one_thread:
                    mma_arrive_multicast[cta_group](mma_mbar, c_mma_mask)
        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=BN,
        type=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)
    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[cta_group]()
        tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)

    warp_id = thread_idx.x // WARP_SIZE

    var c_gmem_block = c.tile[MMA_M, MMA_N](
        peer_cta_coord[1], peer_cta_coord[2]
    )
    var c_gmem_slice = c_gmem_block.tile[BM, MMA_N](peer_cta_coord[0], 0)
    var c_gmem_frag = c_gmem_slice.tile[BM // 2, BN](
        warp_id % 2, warp_id // 2
    ).vectorize[1, 2]()

    @parameter
    for i in range(c_frag_size // 2):
        c_gmem_frag[lane_id(), i] = rebind[c_gmem_frag.element_type](
            SIMD[accum_type, 2](c_frag[2 * i], c_frag[2 * i + 1]).cast[c_type]()
        )


def test_tma_umma_pair_cta[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    cta_group: Int = 1,
](ctx: DeviceContext):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    print(
        "mma_"
        + "s"
        + "s_bf16_bf16_f32 block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(mma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
    )

    alias M = prob_shape[0]
    alias N = prob_shape[1]
    alias K = prob_shape[2]

    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(a.tensor[update=False]())

    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    arange(b.tensor[update=False]())

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a.device_tensor())
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(
            BN // (cluster_shape[0] // cta_group), BK
        ) if transpose_b else Index(BK, BN // (cluster_shape[0] // cta_group)),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    alias smem_size = BM * BK * sizeof[a_type]() + BN * BK * sizeof[
        b_type
    ]() + 16 + 16 + 16

    alias kernel = tma_umma_kernel_pair_cta[
        a_type,
        b_type,
        c_type,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        Layout.row_major(M, N),
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        block_tile_shape,
        mma_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        cta_group=cta_group,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c.device_tensor(),
        K // BK,
        grid_dim=(
            align_up(M // BM, Int(cluster_shape[0])),
            align_up(N // BN // cta_group, Int(cluster_shape[1])),
            1,
        ),
        block_dim=(128),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )

    vendor_blas.matmul(
        ctx,
        c_ref.device_buffer(),
        a.device_buffer[update=False](),
        b.device_buffer[update=False](),
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    c_host = c.tensor()
    c_host_ref = c_ref.tensor()
    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                c_host[m, n],
                c_host_ref[m, n],
                atol=1e-3,
                rtol=1e-4,
                msg=String(m) + ", " + String(n),
            )
            # print(m, n, c_host[m, n], c_host_ref[m, n])

    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^


def main():
    with DeviceContext() as ctx:
        test_tma_umma_pair_cta[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(256, 512, 128),
            Index(64, 64, 64),
            Index(128, 128, 16),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            cta_group=2,
        ](ctx)

        test_tma_umma_pair_cta[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(256, 1024, 128),
            Index(64, 128, 64),
            Index(128, 256, 16),
            cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            cta_group=2,
        ](ctx)

        test_tma_umma_pair_cta[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(128, 512, 128),
            Index(64, 128, 64),
            Index(128, 256, 16),
            cluster_shape = StaticTuple[Int32, 3](2, 2, 1),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            cta_group=2,
        ](ctx)

        test_tma_umma_pair_cta[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(128, 256, 128),
            Index(64, 128, 64),
            Index(128, 256, 16),
            cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            cta_group=2,
        ](ctx)
