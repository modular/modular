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
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.host.info import H100
from gpu.id import (
    block_dim,
    block_idx,
    thread_idx,
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
from layout.layout_tensor import copy_local_to_dram, LayoutTensorIter
from layout.nd_buffer_stub import from_ndbuffer_row_major
from utils.numerics import get_accum_type
from layout.tensor_core_async import (
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)
from layout.tma_async import (
    PipelineState,
    TMABarrier,
    TMATensorTile,
    create_tma_tile,
    create_mbarrier_array,
)

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from buffer.buffer import NDBuffer
from sys._assembly import inlined_assembly
from sys import alignof
from gpu.cluster import (
    elect_one_sync,
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
)
from gpu.intrinsics import warpgroup_reg_dealloc, warpgroup_reg_alloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from pathlib import Path

from .utils import elementwise_epilogue_type
from linalg.matmul_tile_scheduler import TileScheduler, MatmulSchedule
from .utils_gpu import block_swizzle, MatmulConfig

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4


@always_inline
fn cluster_size[cluster_shape: StaticTuple[Int32, 3]]() -> Int32:
    var size: Int32 = 1

    @parameter
    for i in range(3):
        size *= cluster_shape[i]
    return size


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
    `nvvm.grid_constant`=StaticTuple[Int, 2](0, 1),
)
fn tma_wgmma_warp_specialized_gemm_kernel_persistant[
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
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    num_threads: Int,
    grid_shape: IndexList[2] = Index(16, 8),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    pipeline_stages: Int = 7,
    schedule: MatmulSchedule = MatmulSchedule.TILE2D,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout],
    problem_shape: IndexList[3],
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    alias num_consumer = (num_threads // 128) - 1

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    alias K = b_layout.shape[1].value()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]

    alias accum_type = get_accum_type[a_type]()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    var scheduler = TileScheduler[block_tile_shape, grid_shape](problem_shape)

    var smem = external_memory[
        UInt8, address_space = AddressSpace.SHARED, alignment=8
    ]()

    alias a_smem_size = a_smem_layout.size() * pipeline_stages
    alias b_smem_size = b_smem_layout.size() * pipeline_stages

    alias a_smem_bytes = a_smem_size * sizeof[a_type]()
    alias b_smem_bytes = b_smem_size * sizeof[b_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var smem_poll = (smem + a_smem_bytes + b_smem_bytes).bitcast[Int64]()

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

    var a_mbars_ptr = smem_poll.bitcast[Int64]()
    var b_mbars_ptr = smem_poll.bitcast[Int64]() + pipeline_stages
    full = create_mbarrier_array[pipeline_stages](a_mbars_ptr)
    empty = create_mbarrier_array[pipeline_stages](b_mbars_ptr)

    var warp_group_idx = thread_idx.x // WARP_GROUP_SIZE
    var warp_group_thread_idx = thread_idx.x % WARP_GROUP_SIZE

    var lane_predicate = elect_one_sync()
    if thread_idx.x == 0:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()

        @parameter
        for i in range(pipeline_stages):
            full[i].init(1)
            empty[i].init(num_consumer)

    barrier()

    if warp_group_idx == 0:
        alias num_regs = 24 if num_consumer <= 2 else 32
        warpgroup_reg_dealloc[num_regs]()
        if warp_group_thread_idx == 0 and lane_predicate:
            var write_pipeline_states = PipelineState[pipeline_stages]()

            while scheduler.work_info():
                var k_start = UInt(Int(scheduler.work_info().k_start))
                var num_k_iters = UInt(Int(scheduler.work_info().num_k_tiles))
                var m_coord = UInt(Int(scheduler.work_info().m))
                var n_coord = UInt(Int(scheduler.work_info().n))

                for i in range(k_start, num_k_iters):
                    var write_idx = write_pipeline_states.index()

                    TMABarrier(b_mbars_ptr + write_idx).wait(
                        write_pipeline_states.phase()
                    )
                    var a_smem_tile = a_smem_iter.next(write_idx)[]
                    var b_smem_tile = b_smem_iter.next(write_idx)[]

                    TMABarrier(a_mbars_ptr + write_idx).expect_bytes(
                        expected_bytes
                    )

                    a_tma_op.async_copy(
                        a_smem_tile,
                        TMABarrier(a_mbars_ptr + write_idx),
                        (UInt(i) * BK, m_coord),
                    )

                    b_tma_op.async_copy(
                        b_smem_tile,
                        TMABarrier(a_mbars_ptr + write_idx),
                        (UInt(i) * BK, n_coord),
                    )

                    write_pipeline_states.step()
                scheduler.advance()

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
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        @parameter
        for i in range(pipeline_stages):
            if warp_group_thread_idx == 0:
                _ = empty[i].arrive()

        var read_pipeline_states = PipelineState[pipeline_stages]()

        while scheduler.work_info():
            _ = c_reg_tile.fill(0.0)

            var k_start = UInt(Int(scheduler.work_info().k_start))
            var num_k_iters = UInt(Int(scheduler.work_info().num_k_tiles))
            var block_y = UInt(Int(ceildiv(scheduler.work_info().m, BM)))
            var block_x = UInt(Int(ceildiv(scheduler.work_info().n, BN)))

            for i in range(k_start, num_k_iters):
                var read_idx = read_pipeline_states.index()

                TMABarrier(a_mbars_ptr + read_idx).wait(
                    read_pipeline_states.phase()
                )
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
                wgmma_op.wait_for_all()

                if warp_group_thread_idx == 0:
                    _ = TMABarrier(b_mbars_ptr + read_idx).arrive()
                read_pipeline_states.step()

            var c_gmem_tile = c.tile[BM, BN](block_y, block_x)
            var c_gmem_split = c_gmem_tile.tile[BM // num_consumer, BN](
                local_warp_group_idx, 0
            )
            var warp_id = warp_group_thread_idx // WARP_SIZE

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # (m_mma, n_mma) is coordinates for a warp group's tile.
                    # A warp group is 4x1 warps.
                    warp_tile = c_gmem_split.tile[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](m_mma * 4 + warp_id, n_mma)

                    # Tile at (mma_id, 0) is a long vector containing all fragments
                    # for this warp.
                    c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

                    # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
                    # elementwise. This pattern repeates to fill the warp tile.
                    copy_local_to_dram[Layout.row_major(8, 4)](
                        warp_tile.vectorize[1, 2](),
                        c_frag.vectorize[1, 2](),
                    )
            scheduler.advance()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
    `nvvm.grid_constant`=StaticTuple[Int, 2](0, 1),
    `nvvm.cluster_dim`=cluster_shape,
)
fn tma_wgmma_warp_specialized_gemm_kernel[
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
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    cluster_shape: StaticTuple[Int32, 3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    num_threads: Int = 128,
    pipeline_stages: Int = 7,
    partitioned_multicast: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout],
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    alias num_consumer = (num_threads // 128) - 1

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

    constrained[
        not partitioned_multicast
        or a_swizzle.bytes() // sizeof[a_type]() == BK,
        (
            "Currently partitioned multi-casting is only supported when BK =="
            " (a_swizzle.bytes // sizeof[a_type]"
        ),
    ]()
    constrained[
        not partitioned_multicast
        or b_swizzle.bytes() // sizeof[b_type]() == BK,
        (
            "Currently partitioned multi-casting is only supported when BK =="
            " (b_swizzle.bytes // sizeof[b_type]"
        ),
    ]()

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

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var smem_poll = (smem + a_smem_bytes + b_smem_bytes).bitcast[Int64]()

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

    var a_mbars_ptr = smem_poll.bitcast[Int64]()
    var b_mbars_ptr = smem_poll.bitcast[Int64]() + pipeline_stages
    full = create_mbarrier_array[pipeline_stages](a_mbars_ptr)
    empty = create_mbarrier_array[pipeline_stages](b_mbars_ptr)

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

                # empty[write_idx].wait(write_pipeline_states.phase())
                TMABarrier(b_mbars_ptr + write_idx).wait(
                    write_pipeline_states.phase()
                )
                var a_smem_tile = a_smem_iter.next(write_idx)[]
                var b_smem_tile = b_smem_iter.next(write_idx)[]

                TMABarrier(a_mbars_ptr + write_idx).expect_bytes(expected_bytes)

                @parameter
                if CLUSTER_N > 1:

                    @parameter
                    if partitioned_multicast:
                        var a_gmem_slice_coord = block_idx.y * BM + Int(
                            rank_n
                        ) * a_tma_rows
                        var a_smem_slice = __type_of(a_smem_tile)(
                            a_smem_tile.ptr + rank_n * a_tma_load_size
                        )

                        a_tma_op.async_multicast_load(
                            a_smem_slice,
                            TMABarrier(a_mbars_ptr + write_idx),
                            (UInt(i) * BK, a_gmem_slice_coord),
                            UInt16(multicast_row_mask),
                        )

                    else:
                        if rank_n == 0:
                            a_tma_op.async_multicast_load(
                                a_smem_tile,
                                TMABarrier(a_mbars_ptr + write_idx),
                                (UInt(i) * BK, block_idx.y * BM),
                                UInt16(multicast_row_mask),
                            )

                else:
                    a_tma_op.async_copy(
                        a_smem_tile,
                        TMABarrier(a_mbars_ptr + write_idx),
                        (UInt(i) * BK, UInt(block_idx_swizzle[1]) * BM),
                    )

                @parameter
                if CLUSTER_M > 1:

                    @parameter
                    if partitioned_multicast:
                        var b_gmem_slice_coord = block_idx.x * BN + Int(
                            rank_m
                        ) * b_tma_rows
                        var b_smem_slice = __type_of(b_smem_tile)(
                            b_smem_tile.ptr + rank_m * b_tma_load_size
                        )

                        b_tma_op.async_multicast_load(
                            b_smem_slice,
                            TMABarrier(a_mbars_ptr + write_idx),
                            (UInt(i) * BK, b_gmem_slice_coord),
                            UInt16(multicast_column_mask << rank_n),
                        )

                    else:
                        if rank_m == 0:
                            b_tma_op.async_multicast_load(
                                b_smem_tile,
                                TMABarrier(a_mbars_ptr + write_idx),
                                (UInt(i) * BK, block_idx.x * BN),
                                UInt16(multicast_column_mask << rank_n),
                            )

                else:
                    b_tma_op.async_copy(
                        b_smem_tile,
                        TMABarrier(a_mbars_ptr + write_idx),
                        # (UInt(i) * BK, block_idx.x * BN),
                        (UInt(i) * BK, UInt(block_idx_swizzle[0]) * BN),
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

        for i in range(num_k_iters):
            var read_idx = read_pipeline_states.index()

            # full[read_idx].wait(read_pipeline_states.phase())
            TMABarrier(a_mbars_ptr + read_idx).wait(
                read_pipeline_states.phase()
            )
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
            wgmma_op.wait_for_all()

            @parameter
            if cluster_size[cluster_shape]() > 1:
                if warp_group_thread_idx < CLUSTER_SIZE:
                    _ = TMABarrier(b_mbars_ptr + read_idx).arrive_cluster(
                        warp_group_thread_idx
                    )
            else:
                if warp_group_thread_idx == 0:
                    _ = TMABarrier(b_mbars_ptr + read_idx).arrive()

            read_pipeline_states.step()

        # var c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
        var c_gmem_tile = c.tile[BM, BN](
            block_idx_swizzle[1], block_idx_swizzle[0]
        )
        var c_gmem_split = c_gmem_tile.tile[BM // num_consumer, BN](
            local_warp_group_idx, 0
        )
        var warp_id = warp_group_thread_idx // WARP_SIZE

        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()

            # Output dimensions in global memory.
            alias N = c_layout.shape[1].value()
            var M: UInt = c.dim(0)

            lane = lane_id()

            c_frag_vec2 = c_reg_tile.vectorize[1, 2]()

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    warp_tile = c_gmem_split.tile[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](m_mma * 4 + warp_id, n_mma)

                    gmem_frag = warp_tile.vectorize[1, 2]().distribute[
                        Layout.row_major(8, 4)
                    ](lane)
                    thread_offset = gmem_frag.distance(c.ptr)

                    alias num_vecs = __type_of(gmem_frag).layout.size()

                    @parameter
                    for i in range(num_vecs):
                        alias dst_idx = __type_of(gmem_frag).layout(i)

                        m = Int((thread_offset + dst_idx) // N)
                        n = Int((thread_offset + dst_idx) % N)

                        alias alignment = alignof[SIMD[c_type, 2]]()
                        if m < M and n < N:
                            epilogue[alignment=alignment](
                                (m, n), c_frag_vec2[mma_id, i].cast[c_type]()
                            )
        else:

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # (m_mma, n_mma) is coordinates for a warp group's tile.
                    # A warp group is 4x1 warps.
                    warp_tile = c_gmem_split.tile[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](m_mma * 4 + warp_id, n_mma)

                    # Tile at (mma_id, 0) is a long vector containing all fragments
                    # for this warp.
                    c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

                    # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
                    # elementwise. This pattern repeates to fill the warp tile.
                    copy_local_to_dram[Layout.row_major(8, 4)](
                        warp_tile.vectorize[1, 2](), c_frag.vectorize[1, 2]()
                    )

    # TO ensure SEMEM destruction doesn't happen
    @parameter
    if cluster_size[cluster_shape]() > 1:
        cluster_sync()


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 2](0, 1))
fn hopper_matmul_tma_wgmma_kernel[
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
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    transpose_b: Bool = True,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout],
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    alias K = b_layout.shape[1].value()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias num_m_mmas = BM // wgmma_shape[0]
    alias num_n_mmas = BN // wgmma_shape[1]

    alias accum_type = get_accum_type[a_type]()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias num_k_iters = K // BK

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    wgmma_op = TensorCoreAsync[
        accum_type, a_type, b_type, wgmma_shape, transpose_b=transpose_b
    ]()

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    var mbar = TMABarrier()
    var phase = PipelineState[1]()

    if thread_idx.x == 0:
        mbar.init()

    barrier()

    for i in range(num_k_iters):
        if thread_idx.x == 0:
            mbar.expect_bytes(expected_bytes)
            a_tma_op.async_copy(
                a_smem_tile,
                mbar,
                (UInt(i) * BK, block_idx.y * BM),
            )

            b_tma_op.async_copy(
                b_smem_tile,
                mbar,
                (UInt(i) * BK, block_idx.x * BN),
            )
        barrier()

        mbar.wait(phase.phase())
        phase.step()

        wgmma_op.arrive()
        wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        wgmma_op.wait_for_all()

        barrier()

    c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    warp_id = thread_idx.x // WARP_SIZE

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            # (m_mma, n_mma) is coordinates for a warp group's tile.
            # A warp group is 4x1 warps.
            warp_tile = c_gmem_tile.tile[wgmma_shape[0] // 4, wgmma_shape[1]](
                m_mma * 4 + warp_id, n_mma
            )

            # Tile at (mma_id, 0) is a long vector containing all fragments
            # for this warp.
            c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

            # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
            # elementwise. This pattern repeates to fill the warp tile.
            copy_local_to_dram[Layout.row_major(8, 4)](
                warp_tile.vectorize[1, 2](), c_frag.vectorize[1, 2]()
            )


fn hopper_matmul_tma_wgmma[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
    wgmma_n: Int = 128,
](
    c_device: NDBuffer[c_type, 2, c_shape],
    a_device: NDBuffer[a_type, 2, a_shape],
    b_device: NDBuffer[b_type, 2, b_shape],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    alias block_tile_shape = Index(64, wgmma_n, 32)
    alias wgmma_shape = Index(64, wgmma_n, 16)

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK]()

    a_tma_op = create_tma_tile[a_type, 2, Index(BM, BK)](ctx, a)
    b_tma_op = create_tma_tile[b_type, 2, Index(BN, BK)](ctx, b)

    alias kernel = hopper_matmul_tma_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a).layout,
        __type_of(b).layout,
        Layout.row_major(BM, BK),
        Layout.row_major(BN, BK),
        __type_of(c).layout,
        block_tile_shape,
        wgmma_shape,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        a_smem_layout,
        b_smem_layout,
        transpose_b=True,
    ]
    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(128),
    )


fn _get_grid_shape(num_tiles_n: Int) -> IndexList[2]:
    # A Naive heristic to select grid shape based on number of tile in N.
    if num_tiles_n % 8 == 0:
        return Index(8, 16)

    # Hardcode values on purpose until we move this inside tile scheduler
    # in a more robust way.
    alias h100_num_SMs = H100.sm_count
    num_blocks_n = min(num_tiles_n, h100_num_SMs)

    return Index(
        num_blocks_n,
        h100_num_SMs // num_blocks_n,
    )


fn warp_specialize_gemm_with_multicasting[
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
    grid_shape: OptionalReg[IndexList[2]] = None,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
](
    c_device: NDBuffer[c_type, 2, c_shape],
    a_device: NDBuffer[a_type, 2, a_shape],
    b_device: NDBuffer[b_type, 2, b_shape],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    alias N_static = c_shape.get[1]()

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    alias grid_shape_adjusted = grid_shape.value() if grid_shape else _get_grid_shape(
        ceildiv(N_static, BN)
    )

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )
    alias CLUSTER_N = UInt(Int(cluster_shape[0]))
    alias CLUSTER_M = UInt(Int(cluster_shape[1]))

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B

    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK, a_swizzle]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK, b_swizzle]()

    a_tma_op = create_tma_tile[
        a_type,
        2,
        Index(BM // CLUSTER_N, BK) if config.partitioned_multicast else Index(
            BM, BK
        ),
        swizzle_mode=a_swizzle,
    ](ctx, a)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN // CLUSTER_M, BK) if config.partitioned_multicast else Index(
            BN, BK
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    alias num_threads = WARP_GROUP_SIZE * config.num_consumer + WARP_GROUP_SIZE
    alias smem_size = Int(config.num_pipeline_stages) * (
        a_smem_layout.size() * sizeof[a_type]()
        + b_smem_layout.size() * sizeof[b_type]()
        + (sizeof[Int64]() * 2)
    )

    @parameter
    if schedule != MatmulSchedule.NONE:
        alias kernel = tma_wgmma_warp_specialized_gemm_kernel_persistant[
            a_type,
            b_type,
            c_type,
            __type_of(a).layout,
            __type_of(b).layout,
            __type_of(a_tma_op).layout,
            __type_of(b_tma_op).layout,
            __type_of(c).layout,
            config.block_tile_shape,
            wgmma_shape,
            __type_of(a_tma_op).desc_layout,
            __type_of(b_tma_op).desc_layout,
            a_smem_layout,
            b_smem_layout,
            grid_shape = grid_shape_adjusted if schedule
            == MatmulSchedule.TILE2D else Index(132, 1),
            num_threads=num_threads,
            transpose_b=True,
            schedule=schedule,
            pipeline_stages = config.num_pipeline_stages,
        ]

        ctx.enqueue_function[kernel](
            a_tma_op,
            b_tma_op,
            c,
            Index(M, N, K),
            grid_dim=(grid_shape_adjusted[0], grid_shape_adjusted[1]),
            block_dim=(num_threads),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_size
            ),
        )
    else:
        alias kernel = tma_wgmma_warp_specialized_gemm_kernel[
            a_type,
            b_type,
            c_type,
            __type_of(a).layout,
            __type_of(b).layout,
            __type_of(a_tma_op).layout,
            __type_of(b_tma_op).layout,
            __type_of(c).layout,
            config.block_tile_shape,
            wgmma_shape,
            __type_of(a_tma_op).desc_layout,
            __type_of(b_tma_op).desc_layout,
            a_smem_layout,
            b_smem_layout,
            cluster_shape=cluster_shape,
            transpose_b=True,
            num_threads=num_threads,
            pipeline_stages = config.num_pipeline_stages,
            partitioned_multicast = config.partitioned_multicast,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]

        ctx.enqueue_function[kernel](
            a_tma_op,
            b_tma_op,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(num_threads),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_size
            ),
        )
