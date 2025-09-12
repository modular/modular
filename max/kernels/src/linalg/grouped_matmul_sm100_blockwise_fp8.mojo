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
from logger import Logger
from collections import OptionalReg
from sys import size_of, align_of
from math import ceildiv, gcd
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.id import warp_id as get_warp_id
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
from layout.runtime_layout import RuntimeTuple, RuntimeLayout, UNKNOWN_VALUE
from .utils import elementwise_epilogue_type
from buffer.buffer import NDBuffer
from linalg.matmul_sm100_blockwise_fp8 import (
    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel,
)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
fn matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    accum_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_offsets_layout: Layout,
    expert_ids_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    a_scales_tile_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, MutableAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutableAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        accum_type, a_scales_tile_layout, a_scales_desc_layout
    ],
    b_scales: LayoutTensor[accum_type, b_scales_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    constrained[transpose_b, "Only support transposed B"]()
    constrained[num_threads == 128]()
    constrained[
        accum_type == DType.float32, "Only support float32 for accumulator"
    ]()

    var expert_idx = Int(block_idx.z)
    M = rebind[UInt32](a_offsets[expert_idx + 1]) - rebind[UInt32](
        a_offsets[expert_idx]
    )
    alias N = c.layout.shape[1].value()
    alias K = a_layout.shape[1].value()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // MMA_K

    constrained[N % BN == 0, "N must be divisible by BN"]()
    constrained[
        BN <= BK or gcd(BN, BK) == BN - BK,
        "BN <= BK or gcd(BN, BK) == BN - BK",
    ]()

    a_start_row = a_offsets[expert_idx]
    expert = expert_ids[expert_idx]
    b_start_row = expert * N

    m_start = block_idx.y * BM
    n_start = block_idx.x * BN
    a_m_start = UInt(a_start_row) + m_start
    b_n_start = UInt(b_start_row) + n_start
    if m_start >= UInt(M) or n_start >= UInt(N):
        # print("m_start: ", m_start, "n_start: ", n_start, "M: ", M, "N: ", N)
        return

    # make sure A and B scales are compatible
    alias b_scales_expert = b_scales_layout.shape[0].value()
    alias b_scales_n = b_scales_layout.shape[1].value()
    alias b_scales_k = b_scales_layout.shape[2].value()
    alias a_scales_k = a_scales_layout.shape[0].value()

    b_scales_2d = LayoutTensor[
        accum_type,
        Layout.row_major(b_scales_expert * b_scales_n, b_scales_k),
        b_scales.origin,
        address_space = b_scales.address_space,
    ](b_scales.ptr)

    constrained[
        N % b_scales_n == 0 and K % b_scales_k == 0 and K % a_scales_k == 0,
        "N and K must be divisible by b_scales.shape[1] and b_scales.shape[2]",
    ]()

    alias B_SCALING_BLOCK_N = N // b_scales_n
    alias B_SCALING_BLOCK_K = K // b_scales_k
    alias A_SCALING_BLOCK = K // a_scales_k
    constrained[
        BK == B_SCALING_BLOCK_K == B_SCALING_BLOCK_N == A_SCALING_BLOCK,
        "Only support SCALING SIZE of 128! got:"
        + String(BK)
        + " "
        + String(B_SCALING_BLOCK_K)
        + " "
        + String(B_SCALING_BLOCK_N)
        + " "
        + String(A_SCALING_BLOCK),
    ]()

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    alias a_scales_smem_layout = Layout.row_major(1, BM)

    a_smem = rebind[
        UnsafePointer[
            Scalar[a_type], address_space = AddressSpace.SHARED, alignment2=128
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
    alias a_scales_smem_tile_t = LayoutTensor[
        accum_type,
        a_scales_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    alias a_size = a_smem_layout.size()
    alias b_size = b_smem_layout.size()
    alias a_scales_size = a_scales_smem_layout.size()

    constrained[
        ((a_size * size_of[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[
        ((b_size * size_of[b_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[
        ((a_scales_size * size_of[accum_type]()) % 16) == 0,
        "preserve alignment",
    ]()

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()
    var a_scales_smem = (b_smem + b_size).bitcast[Scalar[accum_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)
    var a_scales_smem_tile = a_scales_smem_tile_t(a_scales_smem)

    var ptr_tmem_addr = (
        (a_scales_smem + a_scales_size)
        .bitcast[UInt32]()
        .static_alignment_cast[alignment=16]()
    )

    alias a_expected_bytes = a_size * size_of[a_type]()
    alias b_expected_bytes = b_size * size_of[b_type]()
    alias a_scales_expected_bytes = a_scales_size * size_of[accum_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes + a_scales_expected_bytes

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

    var warp_id = get_warp_id()
    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    alias max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    barrier()

    tmem_addr = ptr_tmem_addr[0]

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

    # final results accumulator regs for C
    alias c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag = SIMD[accum_type, c_frag_size]()

    # temporary accumulators for TMEM loads
    alias total_repeat = BN // 8
    alias repeat = 1  # a higher repeat will probably get us better performance, but it will increase register pressure
    alias temp_cfrags_size = 4 * repeat

    constrained[
        total_repeat % repeat == 0, "total_repeat must be divisible by repeat"
    ]()
    var c_frag_temp = SIMD[accum_type, temp_cfrags_size]()

    for k_iter in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)

            var k_start = UInt(k_iter) * BK
            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (UInt(k_start), UInt(a_m_start)),
            )

            a_scales_tma_op.async_copy(
                a_scales_smem_tile,
                tma_mbar[0],
                (UInt(a_m_start), UInt(k_iter)),
            )

            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (UInt(k_start), UInt(b_n_start)) if transpose_b else (
                    UInt(b_n_start),
                    UInt(k_start),
                ),
            )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(True),  # Initialize C on first iteration
            )

            mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

        @parameter
        for ld_iter in range(total_repeat // repeat):
            c_frag_temp = tcgen05_ld[
                datapaths=16,
                bits=256,
                repeat=repeat,
                dtype=accum_type,
                pack=False,
                width=temp_cfrags_size,
            ](tmem_addr + ld_iter * 8 * repeat)
            tcgen05_load_wait()  # wait for the load to finish

            var b_scale: Scalar[accum_type]
            b_scale_m_offset = UInt(expert * b_scales_n)

            @parameter
            if BN != BK:
                var global_n = block_idx.x * BN

                var begin_n = min(BN, BK - global_n % BK)
                alias end_n = BN  # if N % BN !=0 then it should be  min(BN, N - block_idx.x * BN)

                var idx0 = global_n // BK
                var next_n = begin_n if begin_n < end_n else BN

                if ld_iter < (next_n // 8):
                    b_scale = rebind[Scalar[accum_type]](
                        b_scales_2d[b_scale_m_offset + idx0, k_iter]
                    )
                else:
                    b_scale = rebind[Scalar[accum_type]](
                        b_scales_2d[b_scale_m_offset + idx0 + 1, k_iter]
                    )

            else:
                b_scale = rebind[Scalar[accum_type]](
                    b_scales_2d[b_scale_m_offset + block_idx.x, k_iter]
                )

            var m_offset = (warp_id * 16) + (lane_id() // 4)

            # TODO: this is an ugly way to calculate the m offset, need to rethink how we can make this more efficient
            @parameter
            for j in range(temp_cfrags_size // 2):
                var local_m = m_offset + (j % 2) * 8
                var a_scale = a_scales_smem_tile[0, local_m]

                var scale = a_scale * b_scale

                c_frag[ld_iter * temp_cfrags_size + 2 * j] += c_frag_temp[
                    2 * j
                ] * rebind[Scalar[accum_type]](scale)
                c_frag[ld_iter * temp_cfrags_size + 2 * j + 1] += c_frag_temp[
                    2 * j + 1
                ] * rebind[Scalar[accum_type]](scale)

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    alias num_warps = num_threads // WARP_SIZE
    warp_id = UInt(thread_idx.x // WARP_SIZE)

    alias c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
    alias c_gmem_type = LayoutTensor[
        c_type,
        c_gmem_layout,
        MutableAnyOrigin,
        layout_int_type = DType.int32,
        address_space = AddressSpace.GENERIC,
    ]

    # FIXME: A list literal initializer should be enough here, but somehow Mojo fails to infer that.
    var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
        Index(M, N), Index(N, 1)
    )

    var c_by_expert = c_gmem_type(
        c.ptr + a_start_row * N, c_gmem_runtime_layout
    )

    ctile, ctile_coords, _ = c_by_expert.tile_with_offset[BM, BN](
        block_idx.y, block_idx.x
    )
    alias c_coord_type = __type_of(ctile_coords)

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
                            alias alignment = align_of[SIMD[c_type, 2]]()
                            alias epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (Int(a_start_row + m), Int(n)), c_mn
                            )
                        else:
                            c_gmem_frag[m_vec, n_vec] = rebind[
                                c_gmem_frag.element_type
                            ](c_mn)


fn grouped_matmul_sm100_blockwise_scaled_fp8[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_offsets_layout: Layout,
    expert_ids_layout: Layout,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    accum_type: DType,
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
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, *_, **_],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, *_, **_],
    a_scales: LayoutTensor[accum_type, a_scales_layout, *_, **_],
    b_scales: LayoutTensor[accum_type, b_scales_layout, *_, **_],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
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

    alias num_experts = b_layout.shape[0].value()
    alias N = c_layout.shape[1].value()
    alias K = a_layout.shape[1].value()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    constrained[BK == 128, "blockwise scaled fp8 only works with BK = 128"]()

    var a_scales_1 = a_scales.dim(1)
    debug_assert(a_scales_1 == c.dim(0), "a_scales.dim(1) must be equal to M")

    var a_scales_0 = a_scales.dim(0)
    debug_assert(
        K % a_scales_0 == 0 and (K // a_scales_0) == BK,
        (
            "K must be divisible by a_scales.dim(0) and BK must be equal to K"
            " // a_scales.dim(0)"
        ),
    )

    var b_scales_0 = b_scales.dim(1)
    var b_scales_1 = b_scales.dim(2)
    debug_assert(
        (N % b_scales_0 == 0 and (N // b_scales_0) == BK)
        and (K % b_scales_1 == 0 and (K // b_scales_1) == BK),
        (
            "N must be divisible by b_scales.dim(0) and BK must be equal to N"
            " // b_scales.dim(0) and K must be divisible by b_scales.dim(1) and"
            " BK must be equal to K // b_scales.dim(1)"
        ),
    )

    var logger = Logger()
    logger.info(
        "Executing Grouped 1D2D Blockwise Scaled FP8 GEMM (BLOCK_SCALE_SIZE ="
        " 128)"
    )
    logger.info("Max tokens per expert: ", max_num_tokens_per_expert)
    logger.info("Number of active experts: ", num_active_experts)
    logger.info(
        "A Scales Shape: [", a_scales.dim(0), ", ", a_scales.dim(1), "]"
    )
    logger.info(
        "B Scales Shape: [",
        b_scales.dim(0),
        ", ",
        b_scales.dim(1),
        ", ",
        b_scales.dim(2),
        "]",
    )

    # LayoutTensors are already in the right format for TMA operations
    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, BK), swizzle_mode=a_swizzle
    ](ctx, a)

    b_2d = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        b.origin,
        address_space = b.address_space,
    ](b.ptr)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b_2d)

    a_scales_tma_op = create_tma_tile[1, BM](ctx, a_scales)

    alias smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24 + size_of[accum_type]() * BM

    alias block_dim = 128

    alias kernel = matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_kernel[
        a_type,
        b_type,
        c_type,
        accum_type,
        __type_of(a).layout,
        __type_of(b).layout,
        __type_of(c).layout,
        __type_of(a_offsets).layout,
        __type_of(expert_ids).layout,
        __type_of(a_scales).layout,
        __type_of(b_scales).layout,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(a_scales_tma_op).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(a_scales_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        a_offsets,
        expert_ids,
        c,
        a_scales_tma_op,
        b_scales,
        ceildiv(K, BK),
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
