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

from sys import size_of
from utils.numerics import min_finite, max_finite
from gpu import WARP_SIZE, barrier
from gpu import lane_id as get_lane_id
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, lane_id, thread_idx
from gpu.memory import AddressSpace, external_memory
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from gpu.cluster import block_rank_in_cluster
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg import vendor_blas
from testing import assert_almost_equal
from layout._fillers import random
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn tma_umma_kernel_ss[
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
    num_threads: UInt = 128,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    constrained[num_threads == 128 or num_threads == 256]()
    constrained[
        a_type == b_type and a_type in (DType.float8_e4m3fn, DType.bfloat16),
        (
            "a_type and b_type must be the same and either float8_e4m3fn or"
            " bfloat16"
        ),
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // MMA_K

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    a_smem = rebind[
        UnsafePointer[
            Scalar[a_type], address_space = AddressSpace.SHARED, alignment=128
        ]
    ](
        external_memory[
            Scalar[a_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )
    alias a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    alias a_size = a_smem_layout.size()
    alias b_size = b_smem_layout.size()

    constrained[
        ((a_size * size_of[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[
        ((b_size * size_of[b_type]()) % 16) == 0, "preserve alignment"
    ]()
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (
        (b_smem + b_size)
        .bitcast[UInt32]()
        .static_alignment_cast[alignment=16]()
    )

    alias accum_type = get_accum_type[a_type]()

    alias c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag = SIMD[accum_type, c_frag_size]()

    alias a_expected_bytes = a_size * size_of[a_type]()
    alias b_expected_bytes = b_size * size_of[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (
        (ptr_tmem_addr + 2)
        .bitcast[SharedMemBarrier]()
        .static_alignment_cast[alignment=8]()
    )
    mma_mbar = (tma_mbar + 1).static_alignment_cast[alignment=8]()

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    alias max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = ptr_tmem_addr[0]

    @parameter
    if num_threads > 128:
        if thread_idx.x >= 128:
            tmem_addr += 1 << 20  # offset for lane 16

    alias a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    alias b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    alias aSBO = a_canonical_layout[0].stride[1].value() * size_of[a_type]()
    alias aLBO = a_canonical_layout[1].stride[1].value() * size_of[a_type]()
    alias b_stride01 = b_canonical_layout[0].stride[1].value()
    alias b_stride11 = b_canonical_layout[1].stride[1].value()
    alias bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[b_type]()
    alias bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[b_type]()

    adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    alias mma_kind = UMMAKind.KIND_F8F6F4 if a_type == DType.float8_e4m3fn else UMMAKind.KIND_F16
    idesc = UMMAInsDescriptor[mma_kind].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    for i in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)

            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (UInt(i) * BK, block_idx.y * BM),
            )
            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (UInt(i) * BK, block_idx.x * BN) if transpose_b else (
                    block_idx.x * BN,
                    UInt(i) * BK,
                ),
            )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            if i == 0:
                mma[c_scale=0](adesc, bdesc, tmem_addr, idesc)

                @parameter
                for j in range(1, num_k_mmas):
                    alias idx = IntTuple(0, MMA_K * j)
                    alias a_offset = a_smem_layout(idx) * size_of[a_type]()
                    alias b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma[c_scale=1](
                        adesc + a_offset, bdesc + b_offset, tmem_addr, idesc
                    )
            else:

                @parameter
                for j in range(num_k_mmas):
                    alias idx = IntTuple(0, MMA_K * j)
                    alias a_offset = a_smem_layout(idx) * size_of[a_type]()
                    alias b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma[c_scale=1](
                        adesc + a_offset, bdesc + b_offset, tmem_addr, idesc
                    )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    alias num_warps = num_threads // WARP_SIZE
    warp_id = thread_idx.x // WARP_SIZE

    @parameter
    if num_threads > 128:
        warp_id = 2 * (warp_id % 4) + warp_id // 4

    ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id, n_mma
            )

            c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            alias num_vecs_m = c_gmem_frag.layout.shape[0].value()
            alias num_vecs_n = c_gmem_frag.layout.shape[1].value()

            @parameter
            for n_vec in range(num_vecs_n):

                @parameter
                for m_vec in range(num_vecs_m):
                    alias i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn tma_umma_kernel_ts[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: UInt = 128,
](
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    constrained[num_threads == 128 or num_threads == 256]()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N

    constrained[
        num_m_mmas == 1 and num_n_mmas == 1,
        "num_m_mmas and num_n_mmas must be 1",
    ]()
    constrained[
        a_type == b_type and a_type == DType.bfloat16,
        "a_type and b_type must be the same and bfloat16 type",
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    b_smem = rebind[
        UnsafePointer[
            Scalar[b_type], address_space = AddressSpace.SHARED, alignment=128
        ]
    ](
        external_memory[
            Scalar[b_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )
    alias b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    var b_smem_tile = b_smem_tile_t(b_smem)
    alias b_size = b_smem_tile_t.layout.size()

    alias accum_type = get_accum_type[a_type]()

    constrained[
        ((b_size * size_of[b_type]()) % 16) == 0, "preserve alignment"
    ]()
    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (
        (b_smem + b_size)
        .bitcast[UInt32]()
        .static_alignment_cast[alignment=16]()
    )

    alias c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag = SIMD[accum_type, c_frag_size]()

    alias b_expected_bytes = b_size * size_of[b_type]()
    alias expected_bytes = b_expected_bytes

    tma_mbar = (
        (ptr_tmem_addr + 2)
        .bitcast[SharedMemBarrier]()
        .static_alignment_cast[alignment=8]()
    )
    mma_mbar = (tma_mbar + 1).static_alignment_cast[alignment=8]()

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = thread_idx.x == 0
    alias max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    @parameter
    if num_threads > 128:
        if thread_idx.x >= 128:
            tmem_addr += 1 << 20  # offset for lane 16
    var c_tmem: UInt32 = tmem_addr
    var a_tmem: UInt32 = tmem_addr + MMA_N

    alias b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    alias b_stride01 = b_canonical_layout[0].stride[1].value()
    alias b_stride11 = b_canonical_layout[1].stride[1].value()
    alias bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[b_type]()
    alias bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[b_type]()

    bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    alias num_warps = num_threads // WARP_SIZE
    warp_id = thread_idx.x // WARP_SIZE

    @parameter
    if num_threads > 128:
        warp_id = 2 * (warp_id % 4) + warp_id // 4

    alias a_frag_size = BM * BK * size_of[a_type]() // 4 // num_threads
    var a_frag = SIMD[DType.uint32, a_frag_size]()

    for i in range(num_iters):
        # Load A from global memory to registers.
        # Each thread loads 32 values
        a_gmem_tile = a.tile[BM, BK](block_idx.y, i)
        a_gmem_warp_tile = a_gmem_tile.tile[BM // num_warps, BK](warp_id, 0)
        # Vectorize by 4 for 16x256 load, each thread loads multiple vector
        # of size 2x4B=4xBF16
        a_gmem_frag = a_gmem_warp_tile.vectorize[1, 4]().distribute[
            Layout.row_major(8, 4)
        ](get_lane_id())
        alias num_vecs_m = a_gmem_frag.layout.shape[0].value()
        alias num_vecs_k = a_gmem_frag.layout.shape[1].value()

        @parameter
        for k in range(num_vecs_k):

            @parameter
            for j in range(num_vecs_m):
                vec = a_gmem_frag[j, k]
                alias idx = k * num_vecs_m + j
                a_frag[2 * idx] = bitcast[DType.uint32, 1](vec.split()[0])
                a_frag[2 * idx + 1] = bitcast[DType.uint32, 1](vec.split()[1])

        tcgen05_st[
            datapaths=16,
            bits=256,
            repeat = BK * size_of[a_type]() // 4 // 8,
            pack=False,
        ](a_tmem, a_frag)

        # store_wait synchronizes within a warp. One warp could go ahead
        # while other warps are still storing to tmem.
        tcgen05_store_wait()
        barrier()

        # Load B by TMA
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)

            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (UInt(i) * BK, block_idx.x * BN) if transpose_b else (
                    block_idx.x * BN,
                    UInt(i) * BK,
                ),
            )

        # Sync TMA and tcgen05_st because the latter can sync across warps.
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            alias atmem_kstride = mma_shape[2] // 2  # * size_of[a_type]()
            if i == 0:
                mma[c_scale=0](a_tmem, bdesc, c_tmem, idesc)

                @parameter
                for j in range(1, BK // mma_shape[2]):
                    alias b_idx = IntTuple(MMA_N * 0, MMA_K * j)
                    alias b_offset = b_smem_layout(b_idx) * size_of[b_type]()
                    mma[c_scale=1](
                        a_tmem + j * atmem_kstride,
                        bdesc + b_offset,
                        c_tmem,
                        idesc,
                    )
            else:

                @parameter
                for j in range(BK // mma_shape[2]):
                    alias b_idx = IntTuple(MMA_N * 0, MMA_K * j)
                    alias b_offset = b_smem_layout(b_idx) * size_of[b_type]()
                    mma[c_scale=1](
                        a_tmem + j * atmem_kstride,
                        bdesc + b_offset,
                        c_tmem,
                        idesc,
                    )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # Each thread owns a row in c tile. This is inefficient but to
    # test the instruction shape.
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](c_tmem)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id, n_mma
            )

            c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            alias num_vecs_m = c_gmem_frag.layout.shape[0].value()
            alias num_vecs_n = c_gmem_frag.layout.shape[1].value()

            @parameter
            for n_vec in range(num_vecs_n):

                @parameter
                for m_vec in range(num_vecs_m):
                    alias i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


def test_tma_umma[
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
    a_smem: Bool = True,
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
        + ("s" if a_smem else "t")
        + "s_"
        + String(a_type)
        + "_"
        + String(b_type)
        + "_"
        + String(c_type)
        + " problem shape "
        + String(prob_shape)
        + " block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(mma_shape)
        + " A "
        + (String(a_swizzle) if a_smem else "tmem")
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

    random(
        a.tensor[update=False](),
        min=min_finite[a_type](),
        max=max_finite[a_type](),
    )

    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    var b_col_major = ManagedLayoutTensor[b_type, Layout.row_major(N, K)](ctx)

    random(
        b.tensor[update=False](),
        min=min_finite[b_type](),
        max=max_finite[b_type](),
    )

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, BK), swizzle_mode=a_swizzle
    ](ctx, a.device_tensor())
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    alias block_dim = 2 * MMA_M

    @parameter
    if a_smem:
        alias smem_use = (BM + BN) * size_of[a_type]() * BK + 24
        alias kernel = tma_umma_kernel_ss[
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
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            a_tma_op,
            b_tma_op,
            c.device_tensor(),
            K // BK,
            grid_dim=(N // BN, M // BM),
            block_dim=(block_dim),
            shared_mem_bytes=Int(smem_use),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_use
            ),
        )

    else:
        alias smem_use = BN * size_of[b_type]() * BK + 24
        alias kernel = tma_umma_kernel_ts[
            a_type,
            b_type,
            c_type,
            Layout.row_major(M, K),
            __type_of(b_tma_op).layout,
            Layout.row_major(M, N),
            __type_of(b_tma_op).desc_layout,
            block_tile_shape,
            mma_shape,
            transpose_b=transpose_b,
            b_swizzle=b_swizzle,
            num_threads=block_dim,
        ]

        ctx.enqueue_function[kernel](
            a.device_tensor(),
            b_tma_op,
            c.device_tensor(),
            K // BK,
            grid_dim=(N // BN, M // BM),
            block_dim=(block_dim),
            shared_mem_bytes=Int(smem_use),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_use
            ),
        )

    @parameter
    if a_type == DType.float8_e4m3fn and (not transpose_b):
        # NOTE: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = b_col_major.tensor()
        var b_tensor = b.tensor()
        for i in range(N):
            for j in range(K):
                b_host_col_major[i, j] = b_tensor[j, i]

        vendor_blas.matmul(
            ctx,
            c_ref.device_buffer(),
            a.device_buffer[update=False](),
            b_col_major.device_buffer[update=True](),
            c_row_major=True,
            transpose_b=True,
        )

    else:
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
    _ = b_col_major^
    _ = c^
    _ = c_ref^


def main():
    with DeviceContext() as ctx:

        @parameter
        for dtype in [DType.bfloat16, DType.float8_e4m3fn]:

            @parameter
            for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:

                @parameter
                for BK_scale in range(0, 2):
                    alias BK = (swizzle.bytes() // size_of[dtype]()) * (
                        1 + BK_scale
                    )

                    @parameter
                    for mma_size_scale in range(0, 2):
                        alias MMA_M = 64 * (1 + mma_size_scale)
                        alias MMA_K = 32 if dtype == DType.float8_e4m3fn else 16

                        @parameter
                        for size_scale in range(1, 3):

                            @parameter
                            for transpose_b in range(0, 2):
                                test_tma_umma[
                                    dtype,
                                    dtype,
                                    DType.bfloat16,
                                    Index(
                                        MMA_M * size_scale,
                                        128 * size_scale,
                                        BK * size_scale,
                                    ),
                                    Index(MMA_M, 128, BK),
                                    Index(MMA_M, 128, MMA_K),
                                    a_swizzle=swizzle,
                                    b_swizzle=swizzle,
                                    transpose_b=transpose_b,
                                ](ctx)

                                @parameter
                                if dtype == DType.bfloat16:
                                    test_tma_umma[
                                        dtype,
                                        dtype,
                                        DType.bfloat16,
                                        Index(
                                            MMA_M * size_scale,
                                            128 * size_scale,
                                            BK * size_scale,
                                        ),
                                        Index(MMA_M, 128, BK),
                                        Index(MMA_M, 128, MMA_K),
                                        b_swizzle=swizzle,
                                        transpose_b=transpose_b,
                                        a_smem=False,
                                    ](ctx)
