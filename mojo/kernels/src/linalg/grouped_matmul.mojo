# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import ceildiv
from sys import sizeof

import linalg.vendor_blas
from buffer.dimlist import Dim, DimList, _make_tuple
from collections import OptionalReg
from gpu import WARP_SIZE, barrier, MAX_THREADS_PER_BLOCK_METADATA
from gpu.grid_controls import (
    pdl_launch_attributes,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.host.info import H100
from gpu.id import (
    block_dim,
    block_idx,
    thread_idx,
    global_idx,
    grid_dim,
    lane_id,
    block_id_in_cluster,
)
from gpu.memory import AddressSpace
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_ldmatrix_swizzle, make_swizzle
from layout.layout_tensor import (
    copy_local_to_dram,
    LayoutTensorIter,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple, UNKNOWN_VALUE
from layout._ndbuffer_stub import from_ndbuffer_row_major
from utils.numerics import get_accum_type
from layout.tensor_core_async import (
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
    st_matrix_n_layout,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from buffer.buffer import NDBuffer
from sys._assembly import inlined_assembly
from sys import alignof, simdwidthof
from gpu.cluster import (
    elect_one_sync,
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
)
from gpu.intrinsics import warpgroup_reg_dealloc, warpgroup_reg_alloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from gpu.sync import cp_async_bulk_wait_group, named_barrier
from pathlib import Path

from .utils import elementwise_epilogue_type
from linalg.matmul_tile_scheduler import TileScheduler, MatmulSchedule
from linalg.matmul_sm90 import (
    warp_specialized_gemm_output,
    _get_c_smem_layout,
    cluster_size,
)
from .utils_gpu import block_swizzle, MatmulConfig
from gpu.warp import broadcast
from gpu.mma import st_matrix
from memory import bitcast
from stdlib.bit import log2_floor

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4

# ===----------------------------------------------------------------------=== #
# Naive grouped matmul
# ===----------------------------------------------------------------------=== #


fn naive_grouped_matmul[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool = True,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    constrained[transpose_b, "Only support transposed B in grouped matmul."]()

    ctx.enqueue_function[
        naive_grouped_matmul_kernel[
            c_type,
            c_shape,
            a_type,
            a_shape,
            b_type,
            b_shape,
        ]
    ](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        grid_dim=(
            ceildiv(c.dim[1](), 32),
            ceildiv(max_num_tokens_per_expert, 16),
            num_active_experts,
        ),
        block_dim=(32, 16, 1),
    )


fn naive_grouped_matmul_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
):
    # There has to be a better way :(
    var M: UInt = UInt(Int(a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]))
    N = b.dim[1]()
    K = b.dim[2]()

    a_start_row = a_offsets[block_idx.z]
    a_by_expert = a.data + a_start_row * K

    expert = expert_ids[block_idx.z]
    b_by_expert = b.data + expert * N * K

    # indices in current matmul
    n = global_idx.x
    m = global_idx.y

    if n >= N or m >= M:
        return

    alias accum_type = get_accum_type[a_type]()

    var accum = Scalar[accum_type](0.0)

    for k in range(K):
        accum += (
            a_by_expert[m * K + k].cast[accum_type]()
            * b_by_expert[n * K + k].cast[accum_type]()
        )

    c_by_expert = c.data + a_start_row * N
    c_by_expert[m * N + n] = accum.cast[c_type]()


# ===----------------------------------------------------------------------=== #
# H100 grouped matmul
# ===----------------------------------------------------------------------=== #


fn grouped_matmul[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
    wgmma_shape: IndexList[3],
    config: MatmulConfig[a_type, b_type, c_type, transpose_b, wgmma_shape],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    alias num_experts = b.shape.get[0]()
    alias N = b.shape.get[1]()
    alias K = b.shape.get[2]()

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )

    alias CLUSTER_N = UInt(Int(cluster_shape[0]))
    alias CLUSTER_M = UInt(Int(cluster_shape[1]))

    alias c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        config.num_pipeline_stages,
        False,
    ]()
    alias c_smem_tile = Index(
        c_smem_layout.shape[0].value(), c_smem_layout.shape[1].value()
    )

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    # Create TMA op for the entire A tensor including all tokens.
    a_tensor = from_ndbuffer_row_major(a)
    a_tma_op = create_tma_tile[
        a_type,
        2,
        Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a_tensor)

    # Flattne B tensor into a 2D tensor for easier TMA support.
    b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK),
        swizzle_mode=b_swizzle,
    ](ctx, b_tensor)

    # Create a dummy TMA op for C, we don't support TMA store for output.
    c_tensor = from_ndbuffer_row_major(c)
    c_tma_op = create_tma_tile[
        c_type,
        2,
        Index(BM, BK),
        swizzle_mode=c_swizzle,
    ](ctx, c_tensor)

    alias num_threads = WARP_GROUP_SIZE * config.num_consumer + WARP_GROUP_SIZE
    alias smem_size = Int(config.num_pipeline_stages) * (
        BM * BK * sizeof[a_type]()
        + BN * BK * sizeof[b_type]()
        + (sizeof[Int64]() * 2)
    ) + c_smem_layout.size() * sizeof[c_type]()

    alias kernel = grouped_matmul_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a_tensor).layout,
        __type_of(b_tensor).layout,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c_tensor).layout,
        config.block_tile_shape,
        wgmma_shape,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(c_tma_op).desc_layout,
        c_smem_layout,
        c_swizzle=c_swizzle,
        cluster_shape=cluster_shape,
        transpose_b=True,
        num_threads=num_threads,
        pipeline_stages = config.num_pipeline_stages,
        use_tma_store=False,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_offsets,
        expert_ids,
        c_tensor,
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
    `nvvm.cluster_dim`=cluster_shape,
)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn grouped_matmul_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    c_smem_layout: Layout,
    cluster_shape: StaticTuple[Int32, 3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = True,
    num_threads: Int = 128,
    pipeline_stages: Int = 7,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_smem_layout, c_desc_layout],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    alias num_consumer = (num_threads // 128) - 1
    alias num_consumer_threads = num_consumer * 128
    alias CLUSTER_N = UInt(Int(cluster_shape[0]))
    alias CLUSTER_M = UInt(Int(cluster_shape[1]))
    alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    alias K = b_layout.shape[1].value()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK, a_swizzle]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK, b_swizzle]()

    alias simd_size = simdwidthof[c_type]()

    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]

    alias accum_type = get_accum_type[a_type]()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    alias use_cluster = cluster_size[cluster_shape]() > 1

    var block_idx_swizzle = block_swizzle(
        Index[element_bitwidth=32, unsigned=True](block_idx.x, block_idx.y),
        Index[element_bitwidth=32, unsigned=True](grid_dim.x, grid_dim.y),
    ) if not use_cluster else Index[element_bitwidth=32, unsigned=True](
        block_idx.x, block_idx.y
    )

    # The block may be OOB because we create blocks based the maximum
    # number of tokens per expert.
    M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
    if UInt32(block_idx_swizzle[1] * BM) >= M:
        return

    a_start_row = a_offsets[block_idx.z]

    alias N = c_layout.shape[1].value()
    expert = expert_ids[block_idx.z]
    b_start_row = expert * N

    wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    var smem = external_memory[
        UInt8, address_space = AddressSpace.SHARED, alignment=8
    ]()

    alias a_smem_size = a_smem_layout.size() * pipeline_stages
    alias b_smem_size = b_smem_layout.size() * pipeline_stages

    alias a_smem_bytes = a_smem_size * sizeof[a_type]()
    alias b_smem_bytes = b_smem_size * sizeof[b_type]()

    alias c_smem_size = c_smem_layout.size()
    alias c_smem_bytes = c_smem_size * sizeof[c_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var c_smem = (smem + a_smem_bytes + b_smem_bytes).bitcast[Scalar[c_type]]()
    var smem_poll = (smem + a_smem_bytes + b_smem_bytes + c_smem_bytes).bitcast[
        Int64
    ]()

    var a_smem_iter = LayoutTensorIter[
        a_type,
        a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](a_smem.static_alignment_cast[128](), a_smem_size)

    var b_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](b_smem.static_alignment_cast[128](), b_smem_size)

    var c_smem_tile = LayoutTensor[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](c_smem.static_alignment_cast[128]())

    var a_mbars_ptr = smem_poll.bitcast[Int64]()
    var b_mbars_ptr = smem_poll.bitcast[Int64]() + pipeline_stages

    full = a_mbars_ptr.bitcast[SharedMemBarrier]()
    empty = b_mbars_ptr.bitcast[SharedMemBarrier]()

    var warp_group_idx = thread_idx.x // WARP_GROUP_SIZE
    var warp_group_thread_idx = thread_idx.x % WARP_GROUP_SIZE
    var num_k_iters = K // BK

    var rank_m = block_id_in_cluster.y
    var rank_n = block_id_in_cluster.x

    var lane_predicate = elect_one_sync()
    if thread_idx.x == 0:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()

        @parameter
        for i in range(pipeline_stages):
            full[i].init(1)
            empty[i].init(num_consumer * CLUSTER_SIZE)

    # We need this to guarantee that the Pipeline init is visible
    # To all producers and consumer blocks in the cluster
    @parameter
    if cluster_size[cluster_shape]() > 1:
        fence_mbarrier_init()
        cluster_sync_relaxed()
    else:
        barrier()

    if warp_group_idx == 0:
        alias num_regs = 24 if num_consumer <= 2 else 32
        warpgroup_reg_dealloc[num_regs]()
        if warp_group_thread_idx == 0 and lane_predicate:
            var multicast_column_mask = 0

            @parameter
            for i in range(Int(CLUSTER_M)):
                multicast_column_mask |= 1 << (i * CLUSTER_N)

            var multicast_row_mask = ((1 << CLUSTER_N) - 1) << (
                rank_m * CLUSTER_N
            )

            var write_pipeline_states = PipelineState[pipeline_stages]()
            for i in range(num_k_iters):
                var write_idx = write_pipeline_states.index()

                empty[write_idx].wait(write_pipeline_states.phase())

                var a_smem_tile = a_smem_iter.next(write_idx)[]
                var b_smem_tile = b_smem_iter.next(write_idx)[]

                full[write_idx].expect_bytes(expected_bytes)

                @parameter
                if CLUSTER_N > 1:
                    if rank_n == 0:
                        a_tma_op.async_multicast_load(
                            a_smem_tile,
                            full[write_idx],
                            (UInt(i) * BK, block_idx.y * BM),
                            UInt16(multicast_row_mask),
                        )

                else:
                    a_tma_op.async_copy(
                        a_smem_tile,
                        full[write_idx],
                        (
                            UInt(i) * BK,
                            UInt(Int(a_start_row))
                            + UInt(block_idx_swizzle[1]) * BM,
                        ),
                    )

                @parameter
                if CLUSTER_M > 1:
                    if rank_m == 0:
                        b_tma_op.async_multicast_load(
                            b_smem_tile,
                            full[write_idx],
                            (UInt(i) * BK, block_idx.x * BN),
                            UInt16(multicast_column_mask << rank_n),
                        )

                else:
                    b_tma_op.async_copy(
                        b_smem_tile,
                        full[write_idx],
                        (
                            UInt(i) * BK,
                            UInt(Int(b_start_row))
                            + UInt(block_idx_swizzle[0]) * BN,
                        ),
                    )

                write_pipeline_states.step()

    else:

        @parameter
        if num_consumer == 1 or num_consumer == 2:
            alias num_regs = 256 if num_consumer == 1 else 240
            warpgroup_reg_alloc[num_regs]()
        else:
            warpgroup_reg_alloc[160]()

        var local_warp_group_idx = warp_group_idx - 1

        var c_reg_tile = LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        _ = c_reg_tile.fill(0.0)

        @parameter
        for i in range(pipeline_stages):

            @parameter
            if cluster_size[cluster_shape]() > 1:
                if warp_group_thread_idx < CLUSTER_SIZE:
                    _ = empty[i].arrive_cluster(warp_group_thread_idx)
            else:
                if warp_group_thread_idx == 0:
                    _ = empty[i].arrive()

        var read_pipeline_states = PipelineState[pipeline_stages]()

        for _ in range(num_k_iters):
            var read_idx = read_pipeline_states.index()

            full[read_idx].wait(read_pipeline_states.phase())

            var a_smem_tile = a_smem_iter.next(read_idx)[]
            var b_smem_tile = b_smem_iter.next(read_idx)[]

            wgmma_op.arrive()
            wgmma_op.wgmma[num_consumer](
                a_smem_tile,
                b_smem_tile,
                c_reg_tile,
                local_warp_group_idx,
            )
            wgmma_op.commit_group()
            wgmma_op.wait_group()

            @parameter
            if cluster_size[cluster_shape]() > 1:
                if warp_group_thread_idx < CLUSTER_SIZE:
                    _ = empty[read_idx].arrive_cluster(warp_group_thread_idx)
            else:
                if warp_group_thread_idx == 0:
                    _ = empty[read_idx].arrive()

            read_pipeline_states.step()

        # C layout for current expert
        alias c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
        c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout, bitwidth=32](
            Index(M, N), Index(N, 1)
        )

        c_by_expert = LayoutTensor[
            c_type,
            c_gmem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.GENERIC,
            layout_bitwidth=32,
        ](c.ptr + a_start_row * N, c_gmem_runtime_layout)

        warp_specialized_gemm_output[
            BM=BM,
            BN=BN,
            c_swizzle=c_swizzle,
            wgmma_shape=wgmma_shape,
            num_consumer=num_consumer,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c_tma_op,
            c_by_expert,
            c_smem_tile,
            c_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            thread_idx.x - WARP_GROUP_SIZE,
            block_idx_swizzle[1],
            block_idx_swizzle[0],
        )

    # TO ensure SEMEM destruction doesn't happen
    @parameter
    if cluster_size[cluster_shape]() > 1:
        cluster_sync()
