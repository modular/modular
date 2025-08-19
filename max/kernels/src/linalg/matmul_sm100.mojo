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

from collections import OptionalReg
from sys import sizeof, alignof
from math import ceildiv
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu import WARP_SIZE, barrier
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, lane_id, thread_idx
from gpu.memory import AddressSpace, external_memory
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from layout import Layout, LayoutTensor
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.cluster import block_rank_in_cluster
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg.mmaop_sm100 import MmaOpSM100_SS

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from layout.runtime_layout import RuntimeTuple
from .utils import elementwise_epilogue_type


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn matmul_sm100_fallback_kernel[
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
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    constrained[num_threads == 128 or num_threads == 256]()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]  # BM
    alias MMA_N = mma_shape[1]  # BN
    alias MMA_K = mma_shape[2]  # 16
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // MMA_K

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
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

    # a_smem_layout is a description of how tile is arranged in memory, and LayoutTensor is a pointer to memory + a layout, taking in a_smem as its pointer
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
        ((a_size * sizeof[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[((b_size * sizeof[b_type]()) % 16) == 0, "preserve alignment"]()
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
    var ptr_tmem_addr = (
        (b_smem + b_size)
        .bitcast[UInt32]()
        .static_alignment_cast[alignment=16]()
    )

    alias accum_type = get_accum_type[a_type]()

    alias c_frag_size = MMA_M * MMA_N // num_threads  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag = SIMD[
        accum_type, c_frag_size
    ]()  # array of accumulator elements

    alias a_expected_bytes = a_size * sizeof[a_type]()
    alias b_expected_bytes = b_size * sizeof[b_type]()
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

    # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = ptr_tmem_addr[0]

    # Create MmaOpSM100_SS instance to handle MMA operations
    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=1,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    for i in range(num_iters):
        # so only one thread per CTA does the copy
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

        # wait for the copy to finish
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        # now we do the mma, again only one thread issues the instruction
        if elect_one_thread:
            # Use MmaOpSM100_SS to perform the MMA operation
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(i == 0),  # Initialize C on first iteration
            )

            mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # eventually all of c has been accumulated, so we load it from tmem_addr into c_frag registers using tcgen05_ld
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()  # wait for the load to finish

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    alias num_warps = num_threads // WARP_SIZE
    warp_id = thread_idx.x // WARP_SIZE

    ctile, ctile_coords, _ = c.tile_with_offset[BM, BN](
        block_idx.y, block_idx.x
    )
    alias c_coord_type = __type_of(ctile_coords)

    var M = c.dim[0]()
    alias N = c.layout.shape[1].value()

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // num_warps, MMA_N](
                    4 * m_mma + warp_id, n_mma
                )
            )
            c_gmem_warp_tile_coords = ctile_coords + rebind[c_coord_type](
                _c_gmem_warp_tile_coords
            )

            c_gmem_frag, _c_gmem_frag_coords, _ = c_gmem_warp_tile.vectorize[
                1, 2
            ]().distribute_with_offset[Layout.row_major(8, 4)](lane_id())
            new_c_gmem_frag_coords = rebind[c_coord_type](_c_gmem_frag_coords)
            new_c_gmem_frag_coords[1] *= 2
            c_gmem_frag_coords = (
                c_gmem_warp_tile_coords + new_c_gmem_frag_coords
            )

            alias num_vecs_m = c_gmem_frag.layout.shape[0].value()
            alias num_vecs_n = c_gmem_frag.layout.shape[1].value()

            @parameter
            for n_vec in range(num_vecs_n):

                @parameter
                for m_vec in range(num_vecs_m):
                    alias i_vec = n_vec * num_vecs_m + m_vec
                    alias dst_idx = __type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    alias dst_m_offset = dst_idx // N
                    alias dst_n_offset = dst_idx % N
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < N:
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        @parameter
                        if elementwise_lambda_fn:
                            alias alignment = alignof[SIMD[c_type, 2]]()
                            alias epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (Int(m), Int(n)), c_mn
                            )
                        else:
                            c_gmem_frag[m_vec, n_vec] = rebind[
                                c_gmem_frag.element_type
                            ](c_mn)


fn matmul_sm100_fallback[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, *_, **_],
    a: LayoutTensor[a_type, a_layout, *_, **_],
    b: LayoutTensor[b_type, b_layout, *_, **_],
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    constrained[
        a_type == b_type and a_type in (DType.bfloat16, DType.float8_e4m3fn),
        "Only support bfloat16 and float8_e4m3fn",
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, BK), swizzle_mode=a_swizzle
    ](ctx, a)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    alias smem_use = (BM * sizeof[a_type]() + BN * sizeof[b_type]()) * BK + 24

    alias block_dim = 128

    alias kernel = matmul_sm100_fallback_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        ceildiv(K, BK),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
