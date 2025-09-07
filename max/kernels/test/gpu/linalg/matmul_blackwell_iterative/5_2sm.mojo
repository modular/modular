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
from sys import size_of, argv
from hashlib import default_comp_time_hasher
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu import WARP_SIZE, barrier
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, lane_id, thread_idx, block_id_in_cluster
from gpu.memory import AddressSpace, fence_async_view_proxy
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from gpu.mma import st_matrix
from layout import (
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    LayoutTensor,
    IntTuple,
    UNKNOWN_VALUE,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    st_matrix_n_layout,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.cluster import (
    elect_one_sync,
    block_rank_in_cluster,
    cluster_sync,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg import vendor_blas
from linalg.mmaop_sm100 import MmaOpSM100_SS

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn kernel_5[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,  # must pass mma_m by mma_n as this layout, since that's how much each output has to be
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    num_iters: UInt,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)
    alias num_k_mmas = BK // mma_shape[2]

    alias CLUSTER_M = Int(cluster_shape[0])
    alias CLUSTER_N = Int(cluster_shape[1])

    alias TMA_BN = c_layout.shape[1].value()
    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()
    alias c_smem_layout = Layout.row_major(BM, MMA_N)

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    alias sub_a_smem_layout = tile_layout_k_major[
        a_type, BM, 64, swizzle_mode=a_swizzle
    ]()
    alias sub_b_smem_layout = tile_layout_k_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]()

    alias sub_a_smem_tile_t = LayoutTensor[
        a_type,
        sub_a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias sub_b_smem_tile_t = LayoutTensor[
        b_type,
        sub_b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias c_smem_tile_t = LayoutTensor[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    var smem = external_memory[
        UInt8, address_space = AddressSpace.SHARED, alignment=8
    ]()

    alias a_smem_bytes = a_smem_layout.size() * size_of[a_type]()
    alias b_smem_bytes = b_smem_layout.size() * size_of[b_type]()
    alias c_smem_bytes = c_smem_layout.size() * size_of[c_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var c_smem = (smem + a_smem_bytes + b_smem_bytes).bitcast[Scalar[c_type]]()

    var c_smem_tile = c_smem_tile_t(c_smem)

    var smem_pool = (smem + a_smem_bytes + b_smem_bytes + c_smem_bytes).bitcast[
        Int64
    ]()

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

    alias c_frag_size = MMA_M * MMA_N // 128 // cta_group
    var c_frag = SIMD[accum_type, c_frag_size]()

    alias a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    alias expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    var tma_mbar_ptr = smem_pool.bitcast[Int64]()
    var mma_mbar_ptr = tma_mbar_ptr + 2
    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (mma_mbar_ptr + 2).bitcast[UInt32]()

    tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = elect_one_sync()
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
    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape = Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

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

            @parameter
            for j in range(BK // 64):
                alias k = 64 * j
                alias a_offset = a_smem_layout(IntTuple(0, k))
                alias b_offset = b_smem_layout(IntTuple(0, k))
                constrained[((a_offset * size_of[a_type]()) % 128) == 0]()
                constrained[((b_offset * size_of[b_type]()) % 128) == 0]()
                sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)

                var a_smem_slice = __type_of(sub_a_smem_tile)(
                    sub_a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size
                )
                var b_smem_slice = __type_of(sub_b_smem_tile)(
                    sub_b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size
                )
                a_tma_op.async_multicast_load[cta_group](
                    a_smem_slice,
                    tma_mbar[0],
                    (UInt(i * BK + k), UInt(a_gmem_slice_coord)),
                    a_multicast_mask,
                )

                b_tma_op.async_multicast_load[cta_group](
                    b_smem_slice,
                    tma_mbar[0],
                    (UInt(i * BK + k), UInt(b_gmem_slice_coord)),
                    b_multicast_mask,
                )

        if elect_one_cta:
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            if elect_one_warp and elect_one_thread:
                mma_op.mma(
                    a_smem_tile,
                    b_smem_tile,
                    tmem_addr,
                    init_c=(i == 0),  # Initialize C on first iteration
                )

                mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    warp_id = thread_idx.x // WARP_SIZE

    # For tcgen05.ld 16x256, we need to split the register to deal with
    # loading 32 lanes for each warp.
    c_frag_upper, c_frag_lower = c_frag.split()

    # warp_id 0 -> 0, 16
    # warp_id 1 -> 32, 48
    # warp_id 2 -> 64, 80
    # warp_id 3 -> 96, 112
    c_frag_upper = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8 if MMA_M == 128 else MMA_N // 8,
        dtype=accum_type,
        pack=False,
        width = c_frag_upper.size,
    ](tmem_addr | ((warp_id * 32) << 16))

    c_frag_lower = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8 if MMA_M == 128 else MMA_N // 8,
        dtype=accum_type,
        pack=False,
        width = c_frag_lower.size,
    ](tmem_addr | ((warp_id * 32 + 16) << 16))
    tcgen05_load_wait()

    alias C_WBM = BM // 2 if MMA_M == 128 else BM // 4
    alias C_WBN = BN if MMA_M == 128 else MMA_N
    var c_coord_x = warp_id % 2 if MMA_M == 128 else warp_id
    var c_coord_y = warp_id // 2 if MMA_M == 128 else 0

    # 32 x BN
    c_warp_tile = c_smem_tile.tile[C_WBM, C_WBN](c_coord_x, c_coord_y)

    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    alias st_matrix_swizzle = make_swizzle[c_type, c_swizzle]()
    alias NUM_TMA_TILES = MMA_N // TMA_BN
    alias NUM_ST_MATRIX = BN // TMA_BN if MMA_M == 128 else MMA_N // TMA_BN
    alias C_SPLIT_ROWS = BM * NUM_TMA_TILES // 2 if MMA_M == 128 else BM * NUM_TMA_TILES

    var c_smem_tile_reshaped = c_smem_tile.reshape[
        Layout.row_major(BM * NUM_TMA_TILES, TMA_BN)
    ]()

    var split_coord_x = warp_id // 2 if MMA_M == 128 else 0
    var c_smem_split = c_smem_tile_reshaped.tile[C_SPLIT_ROWS, TMA_BN](
        split_coord_x, 0
    )

    @parameter
    for tma_n in range(NUM_ST_MATRIX):
        var c_smem_iter = c_smem_split.tile[BM, TMA_BN](tma_n, 0)
        var c_smem_warp_tile = c_smem_iter.tile[32, TMA_BN](
            warp_id % 2 if MMA_M == 128 else warp_id, 0
        )
        var upper = c_smem_warp_tile.tile[16, TMA_BN](0, 0)
        var lower = c_smem_warp_tile.tile[16, TMA_BN](1, 0)

        @parameter
        for i in range(TMA_BN // 16):
            var d_reg_upper = c_frag_upper.slice[
                8, offset = (i + tma_n * (TMA_BN // 16)) * 8
            ]().cast[DType.bfloat16]()
            var d_reg_lower = c_frag_lower.slice[
                8, offset = (i + tma_n * (TMA_BN // 16)) * 8
            ]().cast[DType.bfloat16]()

            var st_matrix_args = RuntimeTuple[
                IntTuple(
                    UNKNOWN_VALUE,
                    IntTuple(
                        i,
                        0,
                        UNKNOWN_VALUE,
                    ),
                )
            ](lane_id(), i, 0, 0)

            var d_reg_upper_packed = bitcast[DType.float32, 4](d_reg_upper)
            var d_reg_lower_packed = bitcast[DType.float32, 4](d_reg_lower)

            st_matrix[simd_width=4](
                upper.ptr.offset(
                    st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                ),
                d_reg_upper_packed,
            )
            st_matrix[simd_width=4](
                lower.ptr.offset(
                    st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                ),
                d_reg_lower_packed,
            )

    barrier()

    # SMEM -> GMEM: Direct TMA store
    # UMMA (tensor memory) → registers → shared memory → global memory
    #           c_frag                   c_smem_tile      c_tma_op
    if elect_one_warp and thread_idx.x < UInt(NUM_TMA_TILES):
        var row_start = block_idx.x * BM

        var col_start = block_idx.y * MMA_N + thread_idx.x * TMA_BN

        fence_async_view_proxy()
        var c_smem_offset = c_smem_tile.ptr.offset(BM * TMA_BN * thread_idx.x)

        var c_tma_tile = LayoutTensor[
            c_type,
            c_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ](c_smem_offset)

        c_tma_op.async_store(c_tma_tile, (UInt(col_start), UInt(row_start)))
        c_tma_op.commit_group()
        c_tma_op.wait_group[0]()

    if elect_one_warp:
        tcgen05_release_allocation_lock[cta_group]()
        tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)

    cluster_sync()


fn blackwell_kernel_5[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)
    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias MMA_M = umma_shape[0]
    alias MMA_N = umma_shape[1]
    alias MMA_K = umma_shape[2]

    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM // cluster_shape[1], 64), swizzle_mode=a_swizzle
    ](ctx, a)

    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(
            BN // (cluster_shape[0] // cta_group), 64
        ) if transpose_b else Index(64, BN // (cluster_shape[0] // cta_group)),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    # TODO: 64 satisfies 128B swizzle, we need set TMA_BN according to swizzle mode
    c_tma_op = create_tma_tile[BM, 64, swizzle_mode=c_swizzle](ctx, c)

    alias smem_size = (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + BM * MMA_N * size_of[c_type]()
    ) + 16 + 16 + 16 + 16

    alias kernel = kernel_5[
        a_type,
        b_type,
        c_type,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c_tma_op).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(c_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=cta_group,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
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


def test_blackwell_kernel_5[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var M = m.value
    var N = n.value
    var K = k.value

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
        + "\nMNK="
        + String(M)
        + "x"
        + String(N)
        + "x"
        + String(K)
        + " cluster_shape=("
        + String(cluster_shape[0])
        + ", "
        + String(cluster_shape[1])
        + ", "
        + String(cluster_shape[2])
        + ")"
    )

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    var at = a_host.tensor
    var bt = b_host.tensor
    for m in range(M):
        for k in range(K):
            at[m, k] = Float32(k).cast[a_type]()
    for n in range(N):
        for k in range(K):
            bt[n, k] = Float32(1 if n == k else 0).cast[b_type]()
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    blackwell_kernel_5[
        transpose_b=transpose_b,
        umma_shape=mma_shape,
        block_tile_shape=block_tile_shape,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=2,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        ctx,
    )

    if benchmark:
        alias num_runs = 50
        alias num_warmup = 20

        @always_inline
        @parameter
        fn run_kernel(ctx: DeviceContext) raises:
            blackwell_kernel_5[
                transpose_b=transpose_b,
                umma_shape=mma_shape,
                block_tile_shape=block_tile_shape,
                cluster_shape=cluster_shape,
                a_swizzle=a_swizzle,
                b_swizzle=b_swizzle,
                c_swizzle=c_swizzle,
                cta_group=2,
            ](
                c_device.tensor,
                a_device.tensor,
                b_device.tensor,
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        print("finished warmup")

        var nstime = ctx.execution_time[run_kernel](num_runs) / num_runs
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12

        print("  Average time: ", sectime * 1000, " ms")
        print("  Performance: ", TFlop / sectime, " TFLOPS")
        print()
    else:
        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
        ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
        ctx.synchronize()

        alias rtol = 1e-2

        assert_almost_equal(
            c_host.tensor,
            c_host_ref.tensor,
            atol=0.0001,
            rtol=rtol,
        )

    print("\n=== TEST PASSED ===")
    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


fn get_dic_of_shapes(
    index: Int, dic_bro: Dict[Int, Tuple[Int, Int, Int], *_, **_]
) -> Tuple[Int, Int, Int]:
    try:
        return dic_bro[index]
    except error:
        print("error")
        return (128, 128, 128)


fn make_dic_of_shapes() -> (
    Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]
):
    var dic = Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]()
    dic[0] = (4096, 4096, 4096)
    return dic^


fn benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    alias a_type = DType.bfloat16
    alias b_type = DType.bfloat16
    alias c_type = DType.bfloat16
    alias transpose_b = True

    alias dic_of_shapes = make_dic_of_shapes()

    print("Shapes: [M, N, K]")

    alias block_tile_shape = Index(128, 128, 64)
    alias umma_shape = Index(256, 256, 16)

    @parameter
    for i in range(len(dic_of_shapes)):
        alias shape = get_dic_of_shapes(i, dic_of_shapes)
        print(
            "Benchmarking shape: [",
            shape[0],
            ",",
            shape[1],
            ",",
            shape[2],
            "]",
        )
        test_blackwell_kernel_5[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            block_tile_shape,
            umma_shape,
            cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            benchmark=True,
        ](ctx, dynamic(4096), static[2560](), static[8192]())


def main():
    with DeviceContext() as ctx:
        if is_benchmark():
            # Run the benchmark
            print("\n\n========== Running Benchmarks ==========\n")
            benchmark_blackwell_matmul(ctx)
            return

        alias block_tile_shape = Index(128, 128, 64)
        alias umma_shape = Index(256, 256, 16)

        test_blackwell_kernel_5[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            block_tile_shape,
            umma_shape,
            cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx, dynamic(4096), static[4096](), static[4096]())
