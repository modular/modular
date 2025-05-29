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

from math import ceildiv
from sys import sizeof, simdwidthof

from gpu import WARP_SIZE, barrier
from gpu import warp_id as get_warp_id, lane_id as get_lane_id
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, lane_id, thread_idx, block_id_in_cluster
from gpu.memory import AddressSpace
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import product
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg import vendor_blas
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from testing import assert_almost_equal

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
    a_smem: Bool = True,
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

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    alias accum_type = get_accum_type[a_type]()

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = stack_allocation[
        1, UInt32, address_space = AddressSpace.SHARED, alignment=16
    ]()

    alias c_frag_size = MMA_M * MMA_N // 128 // cta_group
    var c_frag = SIMD[accum_type, c_frag_size]()

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    mma_mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

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
        tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    @parameter
    if cta_group == 1:
        barrier()
    else:
        cluster_sync()

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

    for i in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)
            a_tma_op.async_copy(
                a_smem_tile, tma_mbar[0], (UInt(i) * BK, block_idx.y * BM)
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

        barrier()

        @parameter
        if cta_group == 1:
            if elect_one_thread:
                adesc = adesc_base
                bdesc = bdesc_base
                if i == 0:
                    mma[c_scale=0](adesc, bdesc, tmem_addr, idesc)

                    @parameter
                    for j in range(1, BK // mma_shape[2]):
                        adesc += mma_shape[2] * sizeof[a_type]()
                        bdesc += b_k_stride
                        mma[c_scale=1](adesc, bdesc, tmem_addr, idesc)
                else:

                    @parameter
                    for j in range(BK // mma_shape[2]):
                        mma[c_scale=1](adesc, bdesc, tmem_addr, idesc)
                        adesc += mma_shape[2] * sizeof[a_type]()
                        bdesc += b_k_stride

                mma_arrive(mma_mbar)

            mma_mbar[0].wait(mma_phase)
            mma_phase ^= 1
        else:
            # even cta issue mma
            if elect_one_cta:
                if elect_one_thread:
                    adesc = adesc_base
                    bdesc = bdesc_base

                    if i == 0:
                        mma[cta_group, c_scale=0](
                            adesc, bdesc, tmem_addr, idesc
                        )

                        @parameter
                        for j in range(1, BK // mma_shape[2]):
                            adesc += mma_shape[2] * sizeof[a_type]()
                            bdesc += b_k_stride
                            mma[cta_group, c_scale=1](
                                adesc, bdesc, tmem_addr, idesc
                            )
                    else:

                        @parameter
                        for j in range(BK // mma_shape[2]):
                            mma[cta_group, c_scale=1](
                                adesc, bdesc, tmem_addr, idesc
                            )
                            adesc += mma_shape[2] * sizeof[a_type]()
                            bdesc += b_k_stride

                    mma_arrive_multicast[cta_group](mma_mbar, 0x000F)
            mma_mbar[0].wait(mma_phase)
            mma_phase ^= 1

    @parameter
    if cta_group == 1:
        c_frag = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat = BN // 8,
            type=accum_type,
            pack=False,
            width=c_frag_size,
        ](tmem_addr)
    else:
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

    @parameter
    if cta_group == 1:
        ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma

                c_gmem_warp_tile = ctile.tile[mma_shape[0] // 4, mma_shape[1]](
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
    else:
        if elect_one_cta:
            var c_gmem_block = c.tile[BM, MMA_N](block_id_in_cluster.y, 0)
            var c_gmem_slice = c_gmem_block.tile[BM // 2, BN](
                warp_id % 2, warp_id // 2
            ).vectorize[1, 2]()

            @parameter
            for i in range(c_frag_size // 2):
                c_gmem_slice[lane_id(), i] = rebind[c_gmem_slice.element_type](
                    SIMD[accum_type, 2](c_frag[2 * i], c_frag[2 * i + 1]).cast[
                        c_type
                    ]()
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
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias num_m_mmas = BM // mma_shape[0]
    alias num_n_mmas = BN // mma_shape[1]

    constrained[
        num_m_mmas == 1 and num_n_mmas == 1,
        "num_m_mmas and num_n_mmas must be 1",
    ]()

    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    alias accum_type = get_accum_type[a_type]()

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = stack_allocation[
        1, UInt32, address_space = AddressSpace.SHARED, alignment=16
    ]()

    alias c_frag_size = mma_shape[0] * mma_shape[1] // 128
    var c_frag = SIMD[accum_type, c_frag_size]()

    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = b_expected_bytes

    tma_mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    mma_mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

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
    var c_tmem: UInt32 = tmem_addr
    var a_tmem: UInt32 = tmem_addr + mma_shape[1]  # * 4

    alias b_canonical_layout = tile_to_descriptor[b_type, b_smem_layout]()
    alias bSBO = b_canonical_layout[0].stride[1].value() * sizeof[b_type]()
    alias bLBO = b_canonical_layout[1].stride[1].value() * sizeof[b_type]()

    bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
    ]()

    warp_id = thread_idx.x // WARP_SIZE

    alias a_frag_size = BM * BK * sizeof[a_type]() // 4 // num_threads
    var a_frag = SIMD[DType.uint32, a_frag_size]()

    for i in range(num_iters):
        # Load A from global memory to registers.
        # Each thread loads 32 values
        a_gmem_tile = a.tile[BM, BK](block_idx.y, i)
        a_gmem_warp_tile = a_gmem_tile.tile[BM // 4, BK](warp_id, 0)
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
            repeat = BK * sizeof[a_type]() // 4 // 8,
            pack=False,
        ](a_tmem, a_frag)

        # store_wait synchronizes within a warp. One warp could go ahead
        # while other warps are still storing to tmem.
        tcgen05_store_wait()

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
            alias atmem_kstride = mma_shape[2] // 2  # * sizeof[a_type]()
            if i == 0:
                mma[c_scale=0](a_tmem, bdesc, c_tmem, idesc)

                @parameter
                for j in range(1, BK // mma_shape[2]):
                    bdesc += mma_shape[2] * sizeof[b_type]()
                    mma[c_scale=1](
                        a_tmem + j * atmem_kstride, bdesc, c_tmem, idesc
                    )
            else:

                @parameter
                for j in range(BK // mma_shape[2]):
                    mma[c_scale=1](
                        a_tmem + j * atmem_kstride, bdesc, c_tmem, idesc
                    )
                    bdesc += mma_shape[2] * sizeof[b_type]()

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # Each thread owns a row in c tile. This is inefficient but to
    # test the instruction shape.
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = mma_shape[1] // 8,
        type=accum_type,
        pack=False,
        width=c_frag_size,
    ](c_tmem)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    c_gmem_warp_tile = c.tile[mma_shape[0] // 4, mma_shape[1]](warp_id, 0)
    c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
        Layout.row_major(8, 4)
    ](get_lane_id())
    alias num_vecs_m = c_gmem_frag.layout.shape[0].value()
    alias num_vecs_n = c_gmem_frag.layout.shape[1].value()

    @parameter
    for n_vec in range(num_vecs_n):

        @parameter
        for m_vec in range(num_vecs_m):
            alias i_vec = n_vec * num_vecs_m + m_vec

            c_gmem_frag[m_vec, n_vec] = rebind[c_gmem_frag.element_type](
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
        + "s_bf16_bf16_f32 block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(mma_shape)
        + " A "
        + (String(a_swizzle) if a_smem else "tmem")
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
        a_type, 2, Index(BM, BK), swizzle_mode=a_swizzle
    ](ctx, a.device_tensor())
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    @parameter
    if a_smem:
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
            a_smem=a_smem,
            cta_group=cta_group,
        ]
        ctx.enqueue_function[kernel](
            a_tma_op,
            b_tma_op,
            c.device_tensor(),
            K // BK,
            grid_dim=(N // BN, M // BM),
            block_dim=(128),
        )

    else:
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
            num_threads=128,
        ]

        ctx.enqueue_function[kernel](
            a.device_tensor(),
            b_tma_op,
            c.device_tensor(),
            K // BK,
            grid_dim=(1, 1),
            block_dim=(128),
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

        @parameter
        for size_scale in range(1, 3):

            @parameter
            for transpose_b in range(0, 2):
                test_tma_umma[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(64 * size_scale, 128 * size_scale, 64 * size_scale),
                    Index(64, 128, 64),
                    Index(64, 128, 16),
                    a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    transpose_b=transpose_b,
                ](ctx)

        test_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 128, 64),
            Index(64, 128, 64),
            Index(64, 128, 16),
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            a_smem=False,
        ](ctx)

        @parameter
        for transpose_b in range(0, 2):
            test_tma_umma[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(128, 128, 64),
                Index(64, 64, 64),
                Index(128, 128, 16),
                cluster_shape = StaticTuple[Int32, 3](2, 2, 1),
                a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                cta_group=2,
                transpose_b=transpose_b,
            ](ctx)

        @parameter
        for transpose_b in range(0, 2):
            test_tma_umma[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(128, 256, 64),
                Index(64, 128, 64),
                Index(128, 256, 16),
                cluster_shape = StaticTuple[Int32, 3](2, 2, 1),
                a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                cta_group=2,
                transpose_b=transpose_b,
            ](ctx)
