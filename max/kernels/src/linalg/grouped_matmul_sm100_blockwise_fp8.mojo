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
from collections import Optional
from math import align_up, ceildiv, gcd
from sys import align_of, size_of, simd_width_of
from gpu.host.info import B200, H100
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier
from gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_id_in_cluster, block_idx, lane_id, thread_idx
from gpu import warp_id as get_warp_id
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.compute.arch.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.int_tuple import IntTuple
from layout.layout_tensor import LayoutTensorIter, zipped_divide, upcast
from layout.swizzle import make_swizzle
from layout.runtime_tuple import idx2crd, crd2idx
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile,
)
from logger import Logger
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .arch.sm100 import MmaOpSM100_SS
from .matmul.gpu.sm100.config import MatmulConfig
from .matmul.gpu.sm100.blockwise_fp8 import (
    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel,
)
from .matmul.gpu.sm100.matmul import WarpRole, consumer_main_loop, stsm_helper
from .matmul.gpu.sm100.pipeline import ProducerConsumerPipeline
from .grouped_matmul_tile_scheduler import TileScheduler, WorkInfo
from .utils import elementwise_epilogue_type

comptime logger = Logger()


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
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
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, MutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, MutAnyOrigin],
    num_iters: UInt,
):
    comptime assert transpose_b, "Only support transposed B"
    comptime assert num_threads == 128
    comptime assert (
        accum_type == DType.float32
    ), "Only support float32 for accumulator"

    var expert_idx = Int(block_idx.z)
    M = rebind[UInt32](a_offsets[expert_idx + 1]) - rebind[UInt32](
        a_offsets[expert_idx]
    )
    comptime N = c.layout.shape[1].value()
    comptime K = a_layout.shape[1].value()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    comptime assert N % BN == 0, "N must be divisible by BN"
    comptime assert (
        BN <= BK or gcd(BN, BK) == BN - BK
    ), "BN <= BK or gcd(BN, BK) == BN - BK"

    a_start_row_vec = a_offsets[expert_idx]
    comptime assert a_start_row_vec.size == 1
    var a_start_row = a_start_row_vec[0]

    expert_vec = expert_ids[expert_idx]
    comptime assert expert_vec.size == 1
    var expert = expert_vec[0]

    b_start_row = expert * Int32(N)

    m_start = block_idx.y * UInt(BM)
    n_start = block_idx.x * UInt(BN)
    a_m_start = UInt(a_start_row) + m_start
    b_n_start = UInt(b_start_row) + n_start
    if m_start >= UInt(M) or n_start >= UInt(N):
        # print("m_start: ", m_start, "n_start: ", n_start, "M: ", M, "N: ", N)
        return

    # make sure A and B scales are compatible
    comptime b_scales_expert = b_scales_layout.shape[0].value()
    comptime b_scales_n = b_scales_layout.shape[1].value()
    comptime b_scales_k = b_scales_layout.shape[2].value()
    comptime a_scales_k = a_scales_layout.shape[0].value()

    b_scales_2d = LayoutTensor[
        b_scales_type,
        Layout.row_major(b_scales_expert * b_scales_n, b_scales_k),
        b_scales.origin,
        address_space = b_scales.address_space,
    ](b_scales.ptr)

    comptime assert (
        N % b_scales_n == 0 and K % b_scales_k == 0 and K % a_scales_k == 0
    ), "N and K must be divisible by b_scales.shape[1] and b_scales.shape[2]"

    comptime B_SCALING_BLOCK_N = N // b_scales_n
    comptime B_SCALING_BLOCK_K = K // b_scales_k
    comptime A_SCALING_BLOCK = K // a_scales_k
    comptime assert (
        BK == B_SCALING_BLOCK_K == B_SCALING_BLOCK_N == A_SCALING_BLOCK
    ), (
        "Only support SCALING SIZE of 128! got:"
        + String(BK)
        + " "
        + String(B_SCALING_BLOCK_K)
        + " "
        + String(B_SCALING_BLOCK_N)
        + " "
        + String(A_SCALING_BLOCK)
    )

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    comptime a_scales_smem_layout = Layout.row_major(1, BM)

    a_smem = rebind[
        UnsafePointer[Scalar[a_type], address_space = AddressSpace.SHARED]
    ](
        external_memory[
            Scalar[a_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )

    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime a_scales_smem_tile_t = LayoutTensor[
        accum_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()
    comptime a_scales_size = a_scales_smem_layout.size()

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (a_scales_size * size_of[accum_type]()) % 16
    ) == 0, "preserve alignment"

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var warp_id = get_warp_id()
    var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

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
    comptime c_frag_size = MMA_M * MMA_N // Int(num_threads)
    var c_frag = SIMD[accum_type, c_frag_size]()

    # temporary accumulators for TMEM loads
    comptime total_repeat = BN // 8
    comptime repeat = 1  # a higher repeat will probably get us better performance, but it will increase register pressure
    comptime temp_cfrags_size = 4 * repeat

    comptime assert (
        total_repeat % repeat == 0
    ), "total_repeat must be divisible by repeat"
    var c_frag_temp = SIMD[accum_type, temp_cfrags_size]()

    for k_iter in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            var k_start = Int(k_iter) * BK
            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (k_start, Int(a_m_start)),
            )

            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (k_start, Int(b_n_start)) if transpose_b else (
                    Int(b_n_start),
                    k_start,
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

        comptime for ld_iter in range(total_repeat // repeat):
            c_frag_temp = tcgen05_ld[
                datapaths=16,
                bits=256,
                repeat=repeat,
                dtype=accum_type,
                pack=False,
                width=temp_cfrags_size,
            ](tmem_addr + UInt32(ld_iter * 8 * repeat))
            tcgen05_load_wait()  # wait for the load to finish

            var b_scale: Scalar[accum_type]
            b_scale_m_offset = UInt(expert * Int32(b_scales_n))

            comptime if BN != BK:
                var global_n = block_idx.x * UInt(BN)

                var begin_n = min(BN, BK - Int(global_n % UInt(BK)))
                comptime end_n = BN  # if N % BN !=0 then it should be  min(BN, N - block_idx.x * BN)

                var idx0 = global_n // UInt(BK)
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
            comptime for j in range(temp_cfrags_size // 2):
                var local_m = m_offset + UInt((j % 2) * 8)
                var a_scale = a_scales[k_iter, a_m_start + local_m].cast[
                    accum_type
                ]()

                var scale = a_scale * b_scale

                c_frag[ld_iter * temp_cfrags_size + 2 * j] += c_frag_temp[
                    2 * j
                ] * rebind[Scalar[accum_type]](scale)
                c_frag[ld_iter * temp_cfrags_size + 2 * j + 1] += c_frag_temp[
                    2 * j + 1
                ] * rebind[Scalar[accum_type]](scale)

        barrier()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // UInt(WARP_SIZE)
    warp_id = thread_idx.x // UInt(WARP_SIZE)

    comptime c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
    comptime c_gmem_type = LayoutTensor[
        c_type,
        c_gmem_layout,
        MutAnyOrigin,
        layout_int_type = DType.int32,
        address_space = AddressSpace.GENERIC,
    ]

    # FIXME: A list literal initializer should be enough here, but somehow Mojo fails to infer that.
    var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
        Index(M, N), Index(N, 1)
    )

    var c_by_expert = c_gmem_type(
        c.ptr + a_start_row * UInt32(N), c_gmem_runtime_layout
    )

    ctile, ctile_coords, _ = c_by_expert.tile_with_offset[BM, BN](
        Int(block_idx.y), Int(block_idx.x)
    )
    comptime c_coord_type = type_of(ctile_coords)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // Int(num_warps), MMA_N](
                    4 * m_mma + Int(warp_id), n_mma
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

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    comptime dst_idx = type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    comptime dst_m_offset = dst_idx // N
                    comptime dst_n_offset = dst_idx % N
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < UInt32(N):
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        comptime if elementwise_lambda_fn:
                            comptime alignment = align_of[SIMD[c_type, 2]]()
                            comptime epilogue = elementwise_lambda_fn.value()
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
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    transpose_b: Bool,
    //,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, ...],
    a: LayoutTensor[a_type, a_layout, ...],
    b: LayoutTensor[b_type, b_layout, ...],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, ...],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, ...],
    a_offsets: LayoutTensor[a_offsets_type, a_offsets_layout, ...],
    expert_ids: LayoutTensor[expert_ids_type, expert_ids_layout, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime assert config.transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type and a_type == DType.float8_e4m3fn
    ), "Only support float8_e4m3fn for A and B"

    comptime accum_type = get_accum_type[a_type]()

    comptime num_experts = b_layout.shape[0].value()
    comptime N = c_layout.shape[1].value()
    comptime K = a_layout.shape[1].value()

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]

    comptime assert BK == 128, "blockwise scaled fp8 only works with BK = 128"

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

    logger.info(
        "Executing SM100 Basic Grouped 1D2D Blockwise Scaled FP8 GEMM"
        " (BLOCK_SCALE_SIZE = 128)"
    )
    logger.info("Max tokens per expert: ", max_num_tokens_per_expert)
    logger.info("Number of active experts: ", num_active_experts)
    logger.info(
        "A Scales Shape: [", a_scales.dim(0), ", ", a_scales.dim(1), "]", sep=""
    )
    logger.info(
        "B Scales Shape: [",
        b_scales.dim(0),
        ", ",
        b_scales.dim(1),
        ", ",
        b_scales.dim(2),
        "]",
        sep="",
    )

    # LayoutTensors are already in the right format for TMA operations
    a_tma_op = create_tensor_tile[
        Index(BM, BK), swizzle_mode = config.a_swizzle
    ](ctx, a)

    b_2d = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        b.origin,
        address_space = b.address_space,
    ](b.ptr)
    b_tma_op = create_tensor_tile[
        Index(BN, BK) if config.transpose_b else Index(BK, BN),
        swizzle_mode = config.b_swizzle,
    ](ctx, b_2d)

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24 + size_of[accum_type]() * BM

    comptime block_dim = 128

    comptime kernel = matmul_sm100_grouped_blockwise_scaled_fp8_1d2d_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        type_of(a).layout,
        type_of(b).layout,
        type_of(c).layout,
        type_of(a_offsets).layout,
        type_of(expert_ids).layout,
        type_of(a_scales).layout,
        type_of(b_scales).layout,
        type_of(a_tma_op).layout,
        type_of(b_tma_op).layout,
        type_of(a_tma_op).desc_layout,
        type_of(b_tma_op).desc_layout,
        config.block_tile_shape,
        config.mma_shape,
        transpose_b = config.transpose_b,
        a_swizzle = config.a_swizzle,
        b_swizzle = config.b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        a_offsets,
        expert_ids,
        c,
        a_scales,
        b_scales,
        UInt(ceildiv(K, BK)),
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


@always_inline
fn _get_accumulator_size[
    *,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
]() -> IndexList[2]:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime stageN = c_smem_layout.shape[1].value()
    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages
    comptime data_paths = 16
    comptime bits = 256
    comptime repeats = stageN // (bits // 32)

    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime num_elements = repeats * fragment_size

    return Index(num_stages, num_elements)


@always_inline
fn load_AB[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_scales_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_scales_smem_layout: Layout,
    num_pipeline_stages: UInt,
    expert_ids_layout: Layout,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_layout, a_scales_desc_layout
    ],
    a_smem: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    a_scales_smem: LayoutTensorIter[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    load_mma_pipeline: ProducerConsumerPipeline[Int(num_pipeline_stages)],
    peer_cta_coord: Tuple[UInt, UInt, UInt],
    work_tile_coord: Tuple[UInt, UInt],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: UInt,
    elect_one_cta: Bool,
    scheduler: TileScheduler,
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, ImmutAnyOrigin],
):
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime a_scales_expected_bytes = a_scales_smem_layout.size() * size_of[
        a_scales_type
    ]()
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (
        a_expected_bytes + b_expected_bytes + a_scales_expected_bytes
    )

    comptime a_tma_load_size = a_desc_layout.size()
    comptime b_tma_load_size = b_desc_layout.size()
    comptime a_scales_tma_load_size = a_scales_desc_layout.size()
    comptime a_tma_rows = a_desc_layout.shape[0].value()
    comptime b_tma_rows = b_desc_layout.shape[0].value()

    var stage = load_mma_pipeline.producer_stage()

    # Wait until MMA (consumer) has used the buffer.
    load_mma_pipeline.wait_consumer()

    var a_gmem_slice_coord = (
        peer_cta_coord[2] * UInt(a_tma_rows) + work_tile_coord[0]
    )
    var expert_id = expert_ids[Int(scheduler.current_group_idx)]
    var b_gmem_slice_coord_vec = type_of(expert_id)(
        peer_cta_coord[1] * UInt(b_tma_rows)
        + peer_cta_coord[0] * UInt(BN)
        + work_tile_coord[1]
    ) + expert_id * type_of(expert_id)(scheduler.static_MN)
    comptime assert b_gmem_slice_coord_vec.size == 1
    var b_gmem_slice_coord = b_gmem_slice_coord_vec[0]

    var a_smem_tile = a_smem.next(stage)[]
    var b_smem_tile = b_smem.next(stage)[]
    var a_scales_smem_tile = a_scales_smem.next(stage)[]

    var a_smem_slice = type_of(a_smem_tile)(
        a_smem_tile.ptr + peer_cta_coord[2] * UInt(a_tma_load_size)
    )
    var b_smem_slice = type_of(b_smem_tile)(
        b_smem_tile.ptr + peer_cta_coord[1] * UInt(b_tma_load_size)
    )
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

        a_tma_op.async_multicast_load[cta_group](
            a_smem_slice,
            tma_mbar[0],
            (iter_idx * UInt(BK), a_gmem_slice_coord),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[0],
            (iter_idx * UInt(BK), UInt(b_gmem_slice_coord)),
            b_multicast_mask,
        )

        a_scales_tma_op.async_copy[cta_group](
            a_scales_smem_tile,
            tma_mbar[0],
            (Int(work_tile_coord[0]), Int(iter_idx)),
        )


@always_inline
fn multi_stage_reg_epilogue[
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    accum_type: DType,
    accum_layout: Layout,
    c_tensor_layout: Layout,
    /,
    *,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    is_lower_frag_required: Bool,
    cta_group: Int,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle,
](
    c_upper_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
        ...,
    ],
    c_lower_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
        ...,
    ],
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_tensor_layout, MutAnyOrigin],
    c_coord: Tuple[UInt, UInt],
    elect_one_warp: Bool,
    group_end_idx: UInt32,
):
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime num_stages = accum_layout.shape[0].value()
    comptime num_elements = accum_layout.shape[1].value()

    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime repeats = num_elements // fragment_size
    comptime stageN = repeats * (bits // 32)
    comptime fragments_per_stage = fragment_size * repeats

    comptime swizzle = make_swizzle[c_type, c_swizzle]()

    var warp_id = get_warp_id()

    comptime for stage in range(num_stages):
        var upper_frag = c_upper_main_tile.load[fragments_per_stage](stage, 0)
        var lower_frag = c_lower_main_tile.load[fragments_per_stage](stage, 0)

        # Assume double-buffer for shared memory packing
        var c_smem_tile = c_iter.next(stage % 2)[]
        comptime c_smem_tile_m = 32 if cta_group == 2 else BM // num_output_warps
        var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, stageN](
            Int(warp_id), 0
        )

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[data_paths, stageN](
            0, 0
        )
        stsm_helper[swizzle, UInt(stageN), swizzle_mode=c_swizzle](
            upper_frag, c_smem_warp_tile_upper
        )

        var c_smem_warp_tile_lower = c_smem_warp_tile.tile[data_paths, stageN](
            1, 0
        )

        comptime if is_lower_frag_required:
            stsm_helper[swizzle, UInt(stageN), swizzle_mode=c_swizzle](
                lower_frag, c_smem_warp_tile_lower
            )

        # Guard the write to shared memory is done.
        named_barrier[Int32(num_output_warps * WARP_SIZE)]()

        var lane = lane_id()

        comptime CG2_TMA_BM = c_smem_tile.layout.shape[
            0
        ].value() if MMA_M == 256 else BM
        comptime CG1_TMA_BM = c_smem_tile.layout.shape[0].value()
        comptime TMA_BM = CG2_TMA_BM if cta_group == 2 else CG1_TMA_BM

        var cg2_elect_one_warp = (
            warp_id == 0 if MMA_M == 256 else warp_id % 2 == 0
        )
        var cg1_elect_one_warp = warp_id == 0
        var elect_one_warp = (
            cg2_elect_one_warp if cta_group == 2 else cg1_elect_one_warp
        )

        var coord_n_mma_m256 = c_coord[1] + UInt(stage * stageN)
        var coord_n_mma_m128 = (
            c_coord[1] + UInt(stage * stageN) + UInt(BN * Int(warp_id // 2))
        )

        var cg2_coord_n = coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        var cg1_coord_n = coord_n_mma_m256
        var coord_n = cg2_coord_n if cta_group == 2 else cg1_coord_n
        var coord_m = c_coord[0]
        var cg2_c_smem_coord_m = 0 if MMA_M == 256 else (warp_id // 2)
        var cg1_c_smem_coord_m = UInt(0)
        var c_smem_coord_m = (
            cg2_c_smem_coord_m if cta_group == 2 else cg1_c_smem_coord_m
        )

        if (
            size_of[c_type]() != 2
            or UInt32(coord_m) + UInt32(TMA_BM) >= group_end_idx
        ):
            comptime output_threads = num_output_warps * WARP_SIZE
            comptime c_smem_M = c_smem_tile.layout.shape[0].value()
            comptime RLayout32Bits[layout: Layout] = RuntimeLayout[
                layout,
                element_type = DType.uint32,
                linear_idx_type = DType.uint32,
            ]
            comptime simd_size = simd_width_of[c_type]()
            comptime alignment = align_of[SIMD[c_type, simd_size]]()
            comptime thread_n = stageN // simd_size
            comptime assert (
                stageN % simd_size == 0
            ), "stageN must be divisible by simd_size"
            comptime assert (
                output_threads % thread_n == 0
            ), "output_threads must be divisible by thread_n"
            comptime thread_layout = Layout.row_major(
                output_threads // thread_n, thread_n
            )

            comptime for i in range(c_smem_M // TMA_BM):
                var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](i, 0)
                comptime split_layout = c_smem_split.layout
                var split_rt = RLayout32Bits[split_layout]()
                comptime zipped = zipped_divide(
                    upcast(split_layout, simd_size), thread_layout
                )
                var zipped_rt = RLayout32Bits[zipped]()

                # zipped.shape[1][1] == 1 by construction
                comptime for j in range(zipped.shape[1][0].value()):
                    var input_crd = RuntimeTuple[
                        IntTuple(UNKNOWN_VALUE, j), element_type = DType.uint32
                    ](Int(thread_idx.x), j)
                    var linear_idx = zipped_rt(input_crd) * UInt32(simd_size)
                    var linear_tup = RuntimeTuple[
                        IntTuple(UNKNOWN_VALUE), element_type = DType.uint32
                    ](Int(linear_idx))
                    var cmem_crd = idx2crd(
                        linear_tup, split_rt.shape, split_rt.stride
                    )
                    var local_i = cmem_crd[0].get_int()
                    var local_j = cmem_crd[1].get_int()
                    var global_i = coord_m + UInt(local_i)
                    var global_j = coord_n + UInt(local_j)
                    if global_i < UInt(group_end_idx):
                        # src_ptr = c_smem_split.ptr + swizzle(linear_idx)
                        src_ptr = c_smem_split.ptr + (
                            linear_idx if size_of[c_type]()
                            != 2 else swizzle(linear_idx)
                        )
                        src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        dst_crd = RuntimeTuple[
                            IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)
                        ](Int(global_i), Int(global_j))
                        dst_ptr = c.ptr + c.runtime_layout(dst_crd)
                        dst_ptr.store[width=simd_size, alignment=alignment](src)
        else:
            var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](
                Int(c_smem_coord_m), 0
            )

            if elect_one_warp and lane == 0:
                fence_async_view_proxy()
                c_tma_op.async_store(
                    c_smem_split,
                    (
                        coord_n,
                        coord_m,
                    ),
                )
                c_tma_op.commit_group()

            # Keep one tma store in fly
            comptime if stage < num_stages - 1:
                c_tma_op.wait_group[1]()
            # Last stage guard all tma store to finish
            else:
                c_tma_op.wait_group[0]()

        comptime if stage > 0 and stage < num_stages - 1:
            # Guard the tma read from shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()


@always_inline
fn promote_accumulators[
    pipeline_stages: UInt,
    num_accum_pipeline_stages: Int,
    accum_type: DType,
    accum_layout: Layout,
    a_scales_type: DType,
    b_scales_type: DType,
    b_scales_layout: Layout,
    a_scales_smem_layout: Layout,
    expert_ids_layout: Layout,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    CLUSTER_SIZE: Int32,
    is_lower_frag_required: Bool,
    num_output_warps: Int,
](
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, ImmutAnyOrigin],
    b_scales_n: Int,
    a_scales_smem_iter: LayoutTensorIter[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_upper_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
        ...,
    ],
    c_lower_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
        ...,
    ],
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_pipeline_stages],
    tmem_addr: UInt32,
    load_mma_pipeline: ProducerConsumerPipeline[Int(pipeline_stages)],
    work_tile_coord: Tuple[UInt, UInt],
    elect_one_warp: Bool,
    stage_stride_cols: UInt,
    k_iter: UInt,
    problem_shape: StaticTuple[Int32, 3],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, ImmutAnyOrigin],
    scheduler: TileScheduler,
):
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime assert (
        a_scales_type == b_scales_type and accum_type == DType.float32
    ), "Only support float32 for a_scales, b_scales, and accum_type"
    # Rows each warp is responsible for:
    # warp_id 0 -> 0-15 upper, 16-31 lower
    # warp_id 1 -> 32-47 upper, 48-63 lower
    # warp_id 2 -> 64-79 upper, 80-95 lower
    # warp_id 3 -> 96-111 upper, 112-127 lower

    var M = problem_shape[0]
    var N = problem_shape[1]
    var K = problem_shape[2]

    comptime num_stages = accum_layout.shape[0].value()
    comptime num_elements = accum_layout.shape[1].value()
    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime assert fragment_size == 4, "fragment_size must be 4"
    comptime repeats = num_elements // fragment_size
    comptime stageN = repeats * (bits // 32)
    comptime load_width = 2

    var bn = work_tile_coord[1]

    var tma_load_stage_index = load_mma_pipeline.consumer_stage()

    # scale_b index calculation when MMA_N != BK(128)
    var b_scale_idx0 = 0
    var b_scale_next_n = 0
    var b_scale_0: Scalar[accum_type]
    var b_scale_1: Scalar[accum_type]
    var expert_id = expert_ids[Int(scheduler.current_group_idx)]
    var b_scale_m_offset = expert_id * type_of(expert_id)(b_scales_n)

    comptime if MMA_N != BK:
        comptime assert stageN <= gcd(MMA_N, BK) and (
            gcd(MMA_N, BK) % stageN == 0
        ), (
            "gcd(MMA_N, BK) must be divisible by stageN. If not then this"
            " step should be updated to support non-divisible case"
            " accordingly"
        )

        var global_bn_start = bn
        var begin_n = min(BK - Int(global_bn_start % UInt(BK)), MMA_N)
        var end_n = min(N - Int32(global_bn_start), Int32(MMA_N))

        # find the first b_scale index just by dividing by block size (128)
        # we use `b_scale_next_n` to find the second b_scale index later
        b_scale_idx0 = Int(global_bn_start // UInt(BK))
        # If MMA_N > BK (128) then we should use two scales_b in each block. `next_n` determines the border between the two scales_b.
        # Example: N = 960, MMA_N = 192, num_of_b_scales: ceildiv(960, BK) = 8
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[idx0]--------------->|<--block_scales[idx0+1]-->
        #                                                           next_n(128)

        # this condition determines the border between the two scale_b and whether we have two scale_b in this block or one
        b_scale_next_n = begin_n if begin_n < Int(end_n) else MMA_N
        # Example 1: N = 896, MMA_N = 192, num_of_b_scales: ceildiv(896, BK) = 7
        # This will be the last block on the horizontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <------------------------------------------------------------------------------------->|<
        # <-----------------------------------block_scales[6]----------------------------------->|<
        #                                                                                     next_n (192)

        # Example 2: N = 904, MMA_N = 192, num_of_b_scales: ceildiv(N, BK) = 8
        # This will be the last block on the horizontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[6]------------------>|<-----block_scales[7]---->
        #                                                           next_n(128)

        # prefetch b scales
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[
                b_scale_m_offset + type_of(b_scale_m_offset)(b_scale_idx0),
                k_iter,
            ].cast[accum_type]()
        )
        # this mean in this block we have two scale_b
        if b_scale_next_n < MMA_N:
            b_scale_1 = rebind[Scalar[accum_type]](
                b_scales[
                    b_scale_m_offset
                    + type_of(b_scale_m_offset)(b_scale_idx0)
                    + 1,
                    k_iter,
                ].cast[accum_type]()
            )
        else:
            b_scale_1 = 0.0

    else:
        # when MMA_N == BK == 128 we only have one scale_b per block
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[
                b_scale_m_offset + type_of(b_scale_m_offset)(bn // UInt(MMA_N)),
                k_iter,
            ].cast[accum_type]()
        )
        b_scale_1 = 0.0

    var warp_id = get_warp_id()

    # we update the column offset to include the current stage
    var staged_c_row: UInt
    var staged_c_col: UInt

    comptime if MMA_M == 256 or (MMA_M == 128 and cta_group == 1):
        # based on layout A/D (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-a)
        staged_c_row = warp_id * UInt(WARP_SIZE)
        staged_c_col = UInt(0)
    elif MMA_M == 64 and cta_group == 1:
        # based on layout F (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-f)
        staged_c_row = warp_id * UInt(WARP_SIZE // 2)
        staged_c_col = UInt(0)
    else:
        # based on layout B (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-b)
        staged_c_row = (warp_id % 2) * UInt(WARP_SIZE)
        staged_c_col = UInt(BN) * (warp_id // 2)

    # this is the tensor memory layout
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-fragments-shape-16256b
    # we use it to figure out the starting coordinate
    comptime threads_per_row = UInt(
        stageN // repeats // load_width
    )  # 4 threads per row
    var top_frag_upper_coord = StaticTuple[UInt32, 2](
        UInt32(lane_id() // threads_per_row),
        UInt32(lane_id() % threads_per_row * load_width),
    )

    # getting the other 3 coordinates is straightforward. Each fragment is spaced out by 16 rows
    # and within each fragment the elements are spaced out by 8 rows(this can be seen by the tv layout).
    var bottom_frag_upper_coord = StaticTuple[UInt32, 2](
        top_frag_upper_coord[0] + 8, top_frag_upper_coord[1]
    )

    var top_frag_lower_coord = StaticTuple[UInt32, 2](
        top_frag_upper_coord[0] + 16, top_frag_upper_coord[1]
    )

    var bottom_frag_lower_coord = StaticTuple[UInt32, 2](
        top_frag_lower_coord[0] + 8, top_frag_lower_coord[1]
    )

    var mma_output_stage = mma_output_pipeline.consumer_stage()
    var tmem_offset = mma_output_stage * UInt32(stage_stride_cols) + tmem_addr
    mma_output_pipeline.wait_producer()

    var a_scales_smem = a_scales_smem_iter.next(tma_load_stage_index)[]
    # load a_scales from SMEM
    var upper_sfa0_smem = a_scales_smem[
        0, UInt32(staged_c_row) + top_frag_upper_coord[0]
    ].cast[accum_type]()
    var upper_sfa1_smem = a_scales_smem[
        0, UInt32(staged_c_row) + bottom_frag_upper_coord[0]
    ].cast[accum_type]()

    var lower_sfa0_smem = Scalar[accum_type]()
    var lower_sfa1_smem = Scalar[accum_type]()

    comptime if is_lower_frag_required:
        lower_sfa0_smem = rebind[Scalar[accum_type]](
            a_scales_smem[
                0, UInt32(staged_c_row) + top_frag_lower_coord[0]
            ].cast[accum_type]()
        )
        lower_sfa1_smem = rebind[Scalar[accum_type]](
            a_scales_smem[
                0, UInt32(staged_c_row) + bottom_frag_lower_coord[0]
            ].cast[accum_type]()
        )

    syncwarp()
    if lane_id() < UInt(CLUSTER_SIZE):
        _ = load_mma_pipeline.consumer_mbar(tma_load_stage_index)[0].arrive()
    syncwarp()

    comptime rep_frag_size = repeats * fragment_size
    var upper_frag: SIMD[accum_type, rep_frag_size]
    var lower_frag = SIMD[accum_type, rep_frag_size]()

    comptime for stage in range(num_stages):
        var stage_tmem_addr = tmem_offset + UInt32(stage * stageN)
        upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeats,
            dtype=accum_type,
            pack=False,
            width=rep_frag_size,
        ](stage_tmem_addr)

        comptime if is_lower_frag_required:
            lower_frag = tcgen05_ld[
                datapaths=data_paths,
                bits=bits,
                repeat=repeats,
                dtype=accum_type,
                pack=False,
                width=rep_frag_size,
            ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        comptime if stage == num_stages - 1:
            comptime if cta_group == 1:
                _ = mbarrier_arrive(
                    mma_output_pipeline.consumer_mbar(mma_output_stage)
                )
            else:
                umma_arrive_leader_cta(
                    mma_output_pipeline.consumer_mbar(mma_output_stage)
                )

        var b_scale: Scalar[accum_type]

        comptime if MMA_N != BK:
            # check if we cross the border between the two scale_b
            b_scale = (
                b_scale_0 if (stage * stageN + Int(staged_c_col))
                < b_scale_next_n else b_scale_1
            )
        else:
            b_scale = b_scale_0

        comptime for ld_iter in range(repeats):
            comptime for j in range(fragment_size // 2):
                comptime offset = ld_iter * fragment_size + j * 2

                var upper_elems = upper_frag.slice[2, offset=offset]()
                var lower_elems = lower_frag.slice[2, offset=offset]()

                var upper_a_scale = (
                    upper_sfa0_smem if j == 0 else upper_sfa1_smem
                )
                var lower_a_scale = (
                    lower_sfa0_smem if j == 0 else lower_sfa1_smem
                )

                var upper_scale = upper_a_scale * b_scale
                var lower_scale = lower_a_scale * b_scale

                c_upper_main_tile[stage, offset] += rebind[Scalar[accum_type]](
                    upper_elems[0]
                ) * rebind[Scalar[accum_type]](upper_scale)
                c_upper_main_tile[stage, offset + 1] += rebind[
                    Scalar[accum_type]
                ](upper_elems[1]) * rebind[Scalar[accum_type]](upper_scale)

                comptime if is_lower_frag_required:
                    c_lower_main_tile[stage, offset] += rebind[
                        Scalar[accum_type]
                    ](lower_elems[0]) * rebind[Scalar[accum_type]](lower_scale)
                    c_lower_main_tile[stage, offset + 1] += rebind[
                        Scalar[accum_type]
                    ](lower_elems[1]) * rebind[Scalar[accum_type]](lower_scale)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
fn blackwell_gmm_tma_umma_warp_specialized_blockwise_fp8_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,  # must pass mma_m by mma_n as this layout, since that's how much each output has to be
    c_tensor_layout: Layout,
    a_scales_tile_layout: Layout,
    a_scales_type: DType,
    a_offsets_layout: Layout,
    b_scales_type: DType,
    b_scales_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    num_pipeline_stages: UInt,
    cluster_shape: StaticTuple[Int32, 3],
    expert_n: Int,
    expert_ids_layout: Layout,
    b_scales_n: Int,
](
    num_active_experts: Int,
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_tensor_layout, MutAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_tile_layout, a_scales_desc_layout
    ],
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, ImmutAnyOrigin],
    num_iters: UInt,
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, ImmutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, ImmutAnyOrigin],
    problem_shape: StaticTuple[Int32, 3],
):
    comptime num_output_warps = 4

    comptime accum_type = get_accum_type[a_type]()

    comptime assert (
        b_scales_type == a_scales_type and accum_type == DType.float32
    ), "Only support float32 for a_scales and b_scales"
    comptime assert transpose_b, "only support k-major B"

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = config.cluster_shape[0] * config.cluster_shape[1]
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = NUM_TMEM_COLS // config.num_accum_pipeline_stages

    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = config.cta_group * EPILOGUE_THREADS

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime assert BK == 128, "Only support BK = 128"
    comptime assert MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK, (
        "MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK. MMA_N="
        + String(MMA_N)
        + ", GCD="
        + String(gcd(MMA_N, BK))
    )

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M: Int = config.cluster_shape[0]
    comptime CLUSTER_N: Int = config.cluster_shape[1]

    comptime a_tma_load_size = a_desc_layout.size()
    comptime b_tma_load_size = b_desc_layout.size()
    comptime a_tma_rows = a_desc_layout.shape[0].value()
    comptime b_tma_rows = b_desc_layout.shape[0].value()
    comptime c_smem_layout = Layout.row_major(BM, MMA_N)

    # keep the physical SMEM buffer BM x MMA_N
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode = config.a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]()

    comptime a_scales_smem_layout = Layout.row_major(1, BM)

    base_ptr_smem = external_memory[
        Scalar[a_type],
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]()

    comptime a_smem_size = a_smem_layout.size() * Int(num_pipeline_stages)
    comptime b_smem_size = b_smem_layout.size() * Int(num_pipeline_stages)
    comptime c_smem_size = config.output_tile_shape[
        0
    ] * config.output_tile_shape[1] * config.num_output_stages

    comptime a_scales_smem_size = a_scales_smem_layout.size() * Int(
        num_pipeline_stages
    )

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()
    var a_scales_smem_base = (c_smem_base + c_smem_size).bitcast[
        Scalar[a_scales_type]
    ]()

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        a_smem_base,
        a_smem_size,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        b_smem_base,
        b_smem_size,
    )

    var c_smem_iter = LayoutTensorIter[
        c_type,
        Layout.row_major(
            config.output_tile_shape[0], config.output_tile_shape[1]
        ),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](c_smem_base, c_smem_size)

    var a_scales_smem = LayoutTensorIter[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        a_scales_smem_base,
        a_scales_smem_size,
    )
    var load_mma_mbar_ptr = (a_scales_smem_base + a_scales_smem_size).bitcast[
        SharedMemBarrier
    ]()

    # Load warp as producer and mma warp as consumer
    var load_mma_pipeline = ProducerConsumerPipeline[Int(num_pipeline_stages)](
        load_mma_mbar_ptr
    )

    var mma_output_mbar_ptr = load_mma_mbar_ptr + 2 * Int(num_pipeline_stages)
    var mma_output_pipeline = ProducerConsumerPipeline[
        config.num_accum_pipeline_stages
    ](mma_output_mbar_ptr)

    var clc_full_mbar_ptr = (
        mma_output_mbar_ptr + 2 * config.num_accum_pipeline_stages
    )
    var clc_empty_mbar_ptr = clc_full_mbar_ptr + config.num_clc_pipeline_stages

    # Load warp as producer and scheduler warp as consumer.
    # No data dependence. Introduce dependence to prevent CLC goes too ahead.
    # In the extreme case, all ctas keep querying next work simultaneously,
    # there will be no guarantee they get balanced number of tiles.
    var load_clc_pipeline = ProducerConsumerPipeline[
        config.num_clc_pipeline_stages
    ](clc_empty_mbar_ptr + config.num_clc_pipeline_stages)

    var clc_response_ptr = (
        clc_empty_mbar_ptr + 3 * config.num_clc_pipeline_stages
    ).bitcast[Int128]()

    var tmem_dealloc_mbar_ptr = (
        clc_response_ptr + config.num_clc_pipeline_stages
    ).bitcast[Int64]()

    var ptr_tmem_addr = (tmem_dealloc_mbar_ptr + 1).bitcast[UInt32]()

    clc_response = clc_response_ptr.bitcast[UInt128]()
    clc_full_mbar = clc_full_mbar_ptr.bitcast[SharedMemBarrier]()
    clc_empty_mbar = clc_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    tmem_dealloc_mbar = tmem_dealloc_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = (
        block_rank_in_cluster() % 2 == 0 if config.cta_group == 2 else True
    )
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    var warp_id = get_warp_id()
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()
        a_scales_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            Int32(
                config.cluster_shape[0] // config.cta_group
                + config.cluster_shape[1]
                - 1
                + CLUSTER_SIZE * (EPILOGUE_THREADS // 32)
            ),
        )

        mma_output_pipeline.init_mbars(
            accum_pipeline_producer_arv_count,
            Int32(accum_pipeline_consumer_arv_count),
        )
        load_clc_pipeline.init_mbars(
            Int32(clc_throttle_producer_arv_count),
            Int32(clc_throttle_consumer_arv_count),
        )

        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * config.cta_group))

    comptime for i in range(config.num_clc_pipeline_stages):
        clc_full_mbar[i].init(clc_producer_arv_count)
        clc_empty_mbar[i].init(Int32(clc_consumer_arv_count))

    fence_mbarrier_init()
    cluster_sync()

    var clc_pipe_producer_state = PipelineState[config.num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_pipe_consumer_state = PipelineState[
        config.num_clc_pipeline_stages
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        config.block_tile_shape,
        config.mma_shape,
        accum_type=accum_type,
        cta_group = config.cta_group,
        cluster_shape = config.cluster_shape,
        a_swizzle = config.a_swizzle,
        b_swizzle = config.b_swizzle,
        transpose_b=transpose_b,
    ]()

    # var scheduler = TileScheduler[
    #     num_stages = config.num_clc_pipeline_stages,
    #     cluster_shape = Index[dtype = DType.uint32](
    #         config.cluster_shape[0],
    #         config.cluster_shape[1],
    #         config.cluster_shape[2],
    #     ),
    #     block_swizzle_size = config.block_swizzle_size,
    #     rasterize_order = config.raster_order,
    # ](cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)

    var scheduler = TileScheduler[
        static_MN=expert_n,
        cluster = Index(
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        cta_group = config.cta_group,
        tile_shape = Index(
            config.block_tile_shape[0],
            config.block_tile_shape[1],
            config.block_tile_shape[2],
        ),
        swapAB=False,
    ](num_active_experts, a_offsets)

    var work_info = scheduler.fetch_next_work()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        rank_m % UInt(config.cta_group),
        rank_m // UInt(config.cta_group),
        rank_n,
    )  # v,m,n

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))
    # they all have the same v and m, but different n,

    comptime for i in range(CLUSTER_M // config.cta_group):
        b_multicast_mask |= UInt16(1 << (i * config.cta_group))

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * UInt(CLUSTER_M))

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    if WarpRole.is_main_load():
        # var required_clc_query = True

        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue
            # DO TMA LOAD

            for i in range(num_iters):
                load_AB[
                    block_tile_shape = config.block_tile_shape,
                    mma_shape = config.mma_shape,
                    cta_group = config.cta_group,
                ](
                    a_tma_op,
                    b_tma_op,
                    a_scales_tma_op,
                    a_smem,
                    b_smem,
                    a_scales_smem,
                    load_mma_pipeline,
                    peer_cta_coord,
                    (UInt(work_info.m), UInt(work_info.n)),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                    scheduler,
                    expert_ids,
                )
                load_mma_pipeline.producer_step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        comptime for i in range(num_pipeline_stages):
            load_mma_pipeline.wait_consumer()
            load_mma_pipeline.producer_step()

    if WarpRole.is_mma():
        tcgen05_alloc[Int32(config.cta_group)](ptr_tmem_addr, max_tmem_cols)
        syncwarp()
        # non blocking, arrives and proceeds
        named_barrier_arrive[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)

        tmem_addr = ptr_tmem_addr[0]

        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue
            # scheduler fetch next work
            next_work_info = scheduler.fetch_next_work()
            # DO MMA
            if elect_one_cta:
                for i in range(num_iters):
                    var mma_output_mma_stage = (
                        mma_output_pipeline.producer_stage()
                    )
                    mma_output_pipeline.wait_consumer()
                    var tmem_offset = tmem_addr + (
                        mma_output_mma_stage * UInt32(stage_stride_cols)
                    )

                    consumer_main_loop[
                        block_tile_shape = config.block_tile_shape,
                        mma_shape = config.mma_shape,
                        cta_group = config.cta_group,
                        cluster_shape = config.cluster_shape,
                    ](
                        tmem_offset,
                        a_smem,
                        b_smem,
                        load_mma_pipeline,
                        mma_op,
                        elect_one_warp,
                        0,
                        0,
                    )
                    load_mma_pipeline.consumer_step()

                    # mma arrive multicast will track completion of all mma prior to this barrier.
                    if elect_one_sync():
                        comptime if config.cta_group == 1:
                            mma_arrive[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                )
                            )
                        else:
                            mma_arrive_multicast[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                ),
                                UInt16(mma_complete_mask),
                            )
                    mma_output_pipeline.producer_step()

            work_info = next_work_info

        tcgen05_release_allocation_lock[Int32(config.cta_group)]()

        # wait for epilogue to finish
        tmem_dealloc_mbar[].wait()

        tcgen05_dealloc[Int32(config.cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        tmem_addr = ptr_tmem_addr[0]

        while not work_info.is_done():
            if not work_info.is_valid():
                work_info = scheduler.fetch_next_work()
                continue

            # TODO: zero output

            comptime reg_info = _get_accumulator_size[
                c_smem_layout = c_smem_iter.layout,
                block_tile_shape = config.block_tile_shape,
                mma_shape = config.mma_shape,
                cta_group = config.cta_group,
            ]()

            comptime is_lower_frag_required = not (
                config.cta_group == 1 and config.block_tile_shape[0] == 64
            )
            # final results accumulator regs for C
            var c_upper_main_tile = LayoutTensor[
                accum_type,
                Layout.row_major(reg_info[0], reg_info[1]),
                MutAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ].stack_allocation()

            var c_lower_main_tile = LayoutTensor[
                accum_type,
                Layout.row_major(reg_info[0], reg_info[1]),
                MutAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ].stack_allocation()

            _ = c_upper_main_tile.fill(0.0)

            comptime if is_lower_frag_required:
                _ = c_lower_main_tile.fill(0.0)

            for k_iter in range(num_iters):
                promote_accumulators[
                    block_tile_shape = config.block_tile_shape,
                    mma_shape = config.mma_shape,
                    cta_group = config.cta_group,
                    CLUSTER_SIZE = Int32(CLUSTER_SIZE),
                    is_lower_frag_required=is_lower_frag_required,
                    num_output_warps=num_output_warps,
                ](
                    b_scales,
                    b_scales_n,
                    a_scales_smem,
                    c_upper_main_tile,
                    c_lower_main_tile,
                    # accum_pipeline_consumer_state,
                    mma_output_pipeline,
                    tmem_addr,
                    load_mma_pipeline,
                    work_tile_coord=(UInt(work_info.m), UInt(work_info.n)),
                    elect_one_warp=elect_one_warp,
                    stage_stride_cols=UInt(stage_stride_cols),
                    k_iter=k_iter,
                    problem_shape=problem_shape,
                    expert_ids=expert_ids,
                    scheduler=scheduler,
                )
                load_mma_pipeline.consumer_step()
                mma_output_pipeline.consumer_step()

            # TODO (KERN-2081): investigate why this barrier is needed and if we can move/remove it
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()

            # wait for CUDA core promotion to finish and store result
            # scheduler fetch next work
            multi_stage_reg_epilogue[
                block_tile_shape = config.block_tile_shape,
                mma_shape = config.mma_shape,
                is_lower_frag_required=is_lower_frag_required,
                cta_group = config.cta_group,
                num_output_warps=num_output_warps,
                c_swizzle = config.c_swizzle,
            ](
                c_upper_main_tile,
                c_lower_main_tile,
                c_smem_iter,
                c_tma_op,
                c,
                c_coord=(UInt(work_info.m), UInt(work_info.n)),
                elect_one_warp=elect_one_warp,
                group_end_idx=rebind[Scalar[DType.uint32]](
                    scheduler.group_offsets[
                        Int(scheduler.current_group_idx + 1)
                    ]
                ),
            )

            next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        comptime if config.cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


fn grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
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
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    transpose_b: Bool,
    //,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, ...],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin, ...],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin, ...],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, ImmutAnyOrigin, ...],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, ImmutAnyOrigin, ...],
    a_offsets: LayoutTensor[
        a_offsets_type, a_offsets_layout, ImmutAnyOrigin, ...
    ],
    expert_ids: LayoutTensor[
        expert_ids_type, expert_ids_layout, ImmutAnyOrigin, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime assert config.cta_group == 1, "Only support cta_group == 1"
    comptime assert (
        config.cluster_shape[0] == 1
        and config.cluster_shape[1] == 1
        and config.cluster_shape[2] == 1
    ), "Only support cluster_shape == (1, 1, 1). Got " + String(
        config.cluster_shape
    )
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type and a_type == DType.float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime assert (
        a_scales_type == b_scales_type
    ), "Only support float32 for scales"

    if (a_scales.dim(1) * size_of[a_scales_type]()) % 16 != 0:
        raise Error(
            "a_scales should be a multiple of 16 bytes on the M dimension"
        )

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (1, 2), "Only support cta_group == 2"
    comptime assert not config.AB_swapped, "Swapped AB is not supported"

    # var M = c.dim(0)
    # var N = c.dim(1)
    # var K = a.dim(1)
    comptime num_experts = b_layout.shape[0].value()
    comptime N = c_layout.shape[1].value()
    comptime K = a_layout.shape[1].value()

    a_tma_op = create_tensor_tile[
        Index(BM // config.cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
    ](ctx, a)

    comptime expert_n = N
    b_2d = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        b.origin,
        address_space = b.address_space,
    ](b.ptr)
    b_tma_op = create_tensor_tile[
        Index(
            BN // (config.cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (config.cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode = config.b_swizzle,
    ](ctx, b_2d)

    a_scales_tma_op = create_tma_tile[1, BM](ctx, a_scales)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(64, config.output_tile_shape[1])
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    var c_tma_op = create_tensor_tile[
        c_tma_tile_shape,
        swizzle_mode = config.c_swizzle,
    ](ctx, c)

    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    comptime b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    comptime a_scales_smem_bytes_per_stage = BM * size_of[a_scales_type]()
    comptime AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage

    comptime c_smem_bytes = config.output_tile_shape[
        0
    ] * config.output_tile_shape[1] * config.num_output_stages * size_of[
        c_type
    ]()

    comptime MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
    comptime CLC_RESPONSE_BYTES = size_of[Int128]()  # 16 bytes per response
    comptime TMEM_ADDR_BYTES = size_of[
        Int32
    ]()  # 4 bytes or 32 bits for tensor memory address

    comptime accum_full_mbar_bytes = MBAR_BYTES * config.num_accum_pipeline_stages
    comptime accum_empty_mbar_bytes = MBAR_BYTES * config.num_accum_pipeline_stages

    comptime clc_response_bytes = CLC_RESPONSE_BYTES * config.num_clc_pipeline_stages
    comptime clc_full_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_empty_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_throttle_full_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_throttle_empty_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages

    comptime tmem_addr_bytes = TMEM_ADDR_BYTES
    comptime tmem_dealloc_mbar_bytes = MBAR_BYTES

    comptime tmem_writeout_smem = c_smem_bytes + tmem_addr_bytes + tmem_dealloc_mbar_bytes
    comptime accum_smem = accum_full_mbar_bytes + accum_empty_mbar_bytes
    comptime clc_smem = (
        clc_response_bytes
        + clc_full_mbar_bytes
        + clc_empty_mbar_bytes
        + clc_throttle_full_mbar_bytes
        + clc_throttle_empty_mbar_bytes
    )
    comptime smem_leftover = (b200_smem) - (
        clc_smem + accum_smem + tmem_writeout_smem
    )

    comptime tma_mbar_bytes_per_stage = MBAR_BYTES
    comptime mma_mbar_bytes_per_stage = MBAR_BYTES

    comptime producer_consumer_smem_per_stage = (
        AB_smem_per_stage
        + a_scales_smem_bytes_per_stage
        + tma_mbar_bytes_per_stage
        + mma_mbar_bytes_per_stage
    )

    comptime max_pipeline_stages = UInt(
        smem_leftover // producer_consumer_smem_per_stage
    )

    comptime assert (
        max_pipeline_stages >= 1
    ), "not enough smem even for one pipeline stage!"

    comptime producer_consumer_smem = producer_consumer_smem_per_stage * Int(
        max_pipeline_stages
    )

    comptime smem_size = (
        clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
    )

    comptime b_scales_expert = b_scales_layout.shape[0].value()
    comptime b_scales_n = b_scales_layout.shape[1].value()
    comptime b_scales_k = b_scales_layout.shape[2].value()
    comptime a_scales_k = a_scales_layout.shape[0].value()
    var b_scales_2d = LayoutTensor[
        b_scales_type,
        Layout.row_major(b_scales_expert * b_scales_n, b_scales_k),
        b_scales.origin,
        address_space = b_scales.address_space,
    ](b_scales.ptr)

    comptime kernel = blackwell_gmm_tma_umma_warp_specialized_blockwise_fp8_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        c_layout,
        a_scales_tma_op.layout,
        a_scales_type,
        a_offsets.layout,
        b_scales_type,
        b_scales_2d.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        a_scales_tma_op.desc_layout,
        transpose_b=transpose_b,
        config=config,
        num_pipeline_stages=max_pipeline_stages,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        expert_n=expert_n,
        expert_ids_layout=expert_ids_layout,
        b_scales_n=b_scales_n,
    ]

    # TODO
    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    comptime cluster_shape = config.cluster_shape

    # TODO
    var problem_shape = StaticTuple[Int32, 3](
        Int32(max_num_tokens_per_expert), Int32(N), Int32(K)
    )

    ctx.enqueue_function[kernel, kernel, dump_asm=False](
        num_active_experts,
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c,
        a_scales_tma_op,
        a_offsets,
        UInt(ceildiv(K, BK)),
        b_scales_2d,
        expert_ids,
        problem_shape,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


fn grouped_matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    tokens_padded_per_expert: Bool = False,
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin, _],
    a: NDBuffer[a_type, 2, ImmutAnyOrigin, _],
    b: NDBuffer[b_type, 3, ImmutAnyOrigin, _],
    a_scales: NDBuffer[a_scales_type, 2, ImmutAnyOrigin, _],
    b_scales: NDBuffer[b_scales_type, 3, ImmutAnyOrigin, _],
    a_offsets: NDBuffer[a_offsets_type, 1, ImmutAnyOrigin, _],
    expert_ids: NDBuffer[expert_ids_type, 1, ImmutAnyOrigin, _],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (
        ctx.default_device_info == B200 or ctx.default_device_info == H100
    ), "Only support SM100 or SM90"
    comptime assert (
        m_scale_granularity == 1
        and n_scale_granularity == k_scale_granularity == 128
    ), "Only support (1,128,128) scale granularity"
    comptime assert transpose_b, "Only support transpose_b = True"
    comptime assert (
        a_type == b_type == DType.float8_e4m3fn
    ), "input A and B dtype should be float8_e4m3fn"
    comptime assert (
        a_scales_type == b_scales_type == DType.float32
    ), "input A and B scales dtype should be float32"
    comptime assert (
        input_scale_granularity == "block"
        and weight_scale_granularity == "block"
    ), "Only support block-wise scale granularity"
    comptime assert a_offsets_type == DType.uint32, (
        "Only uint32 is supported for a_offsets in grouped blockwise scaled"
        " fp8 matmul"
    )
    comptime assert expert_ids_type == DType.int32, (
        "Only int32 is supported for expert_ids in grouped blockwise scaled"
        " fp8 matmul"
    )

    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = from_ndbuffer_row_major(b)
    var c_tensor = from_ndbuffer_row_major(c)
    var a_scales_tensor = from_ndbuffer_row_major(a_scales)
    var b_scales_tensor = from_ndbuffer_row_major(b_scales)
    var a_offsets_tensor = from_ndbuffer_row_major(a_offsets)
    var expert_ids_tensor = from_ndbuffer_row_major(expert_ids)

    comptime num_experts = b.shape.get[0]()
    comptime N = b.shape.get[1]()
    comptime K = b.shape.get[2]()
    var seq_len = a.dim[0]()

    if num_active_experts == 0 or max_num_tokens_per_expert == 0:
        return

    comptime if ctx.default_device_info == B200 and tokens_padded_per_expert:
        comptime umma_shape: IndexList[3] = Index(64, 64, 32)

        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            cluster_shape=Index(1, 1, 1),
            mma_shape=umma_shape,
            cta_group=1,
            AB_swapped=False,
            k_group_size=1,
        )
        grouped_matmul_sm100_blockwise_scaled_fp8_persistent[config=config,](
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales_tensor,
            b_scales_tensor,
            a_offsets_tensor,
            expert_ids_tensor,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
        return

    else:
        naive_blockwise_scaled_fp8_grouped_matmul[
            BLOCK_DIM_M=16,
            BLOCK_DIM_N=16,
            transpose_b=transpose_b,
            scales_granularity_mnk = Index(
                m_scale_granularity, n_scale_granularity, k_scale_granularity
            ),
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales_tensor,
            b_scales_tensor,
            a_offsets_tensor,
            expert_ids_tensor,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
