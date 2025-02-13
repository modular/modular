# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from gpu import barrier, WARP_SIZE
from gpu.host import DeviceContext, FuncAttribute
from gpu.host import Dim as ClusterDim
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.id import block_idx, thread_idx, block_dim
from gpu.intrinsics import warpgroup_reg_dealloc, warpgroup_reg_alloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from memory import stack_allocation
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.nd_buffer_stub import from_ndbuffer_row_major
from layout.layout_tensor import LayoutTensorIter, copy_local_to_dram
from layout.tensor_core import get_accum_type
from layout.tensor_core_async import (
    tile_layout_k_major,
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
)
from layout.tma_async import (
    TMATensorTile,
    create_tma_tile,
    TMABarrier,
    PipelineState,
    create_mbarrier_array,
)
from buffer.dimlist import Dim, DimList, _make_tuple
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    fill,
    linspace,
    random,
    zero,
)
import linalg.vendor_blas
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from math import ceildiv
from sys import sizeof, simdwidthof
from sys._assembly import inlined_assembly
from sys import alignof
from gpu.cluster import (
    elect_one_sync,
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
)

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
    transpose_b: Bool = True,
    num_threads: Int = 128,
    pipeline_stages: Int = 7,
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
        accum_type, a_type, b_type, wgmma_shape, transpose_b=transpose_b
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

    var block_rank = block_rank_in_cluster()

    var rank_m = block_rank / CLUSTER_N
    var rank_n = block_rank % CLUSTER_N

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
                        (UInt(i) * BK, block_idx.y * BM),
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
                        (UInt(i) * BK, block_idx.x * BN),
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
            if warp_group_thread_idx < CLUSTER_SIZE:
                _ = empty[i].arrive_cluster(warp_group_thread_idx)

        var read_pipeline_states = PipelineState[pipeline_stages]()

        for i in range(num_k_iters):
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
            wgmma_op.wait_for_all()

            if warp_group_thread_idx < CLUSTER_SIZE:
                _ = empty[read_idx].arrive_cluster(warp_group_thread_idx)
            read_pipeline_states.step()

        var c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
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
                    warp_tile.vectorize[1, 2](), c_frag.vectorize[1, 2]()
                )

    # TO ensure SEMEM destruction doesn't happen
    @parameter
    if cluster_size[cluster_shape]() > 1:
        cluster_sync()


def test_warp_specialize_gemm_with_multicasting[
    wgmma_n: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    cluster_shape: StaticTuple[Int32, 3],
    num_consumer: Int = 1,
    transpose_b: Bool = True,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,):
    var M = m.value
    var N = n.value
    var K = k.value

    alias CLUSTER_N = cluster_shape[0]
    alias CLUSTER_M = cluster_shape[1]

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
    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy_to_device(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy_to_device(c_device_ref.buffer, c_host_ref.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)

    alias block_tile_shape = Index(128, wgmma_n, 64)
    alias wgmma_shape = Index(64, wgmma_n, 16)

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    print(
        "wgmma_n",
        wgmma_n,
        " : PROBLEM SHAPE (M,N,K): (",
        M,
        "x",
        N,
        "x",
        K,
        ") - ",
        "BLOCKS SHAPE (BM,BN,BK): (",
        BM,
        "x",
        BN,
        "x",
        BK,
        ") - ",
        "CLUSTER DIMS (M,N): (",
        CLUSTER_M,
        "x",
        CLUSTER_N,
        ") NUM CONSUMERS: ",
        num_consumer,
    )

    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK]()

    a_tma_op = create_tma_tile[a_type, 2, Index(BM, BK)](ctx, a)
    b_tma_op = create_tma_tile[b_type, 2, Index(BN, BK)](ctx, b)

    debug_assert(
        ((M // BM) % (CLUSTER_M)) == 0,
        String(
            "Number of blocks on M axis should be multiple of cluster dim. M",
            "(M // BM=",
            String(M // BM),
            ") CLUSTER SIZE:",
            String(CLUSTER_M),
        ),
    )

    debug_assert(
        ((N // BN) % (CLUSTER_N)) == 0,
        String(
            "Number of blocks on M axis should be multiple of cluster dim. N",
            "N // BN=(",
            String(N // BN),
            ") CLUSTER SIZE:",
            String(CLUSTER_N),
        ),
    )

    alias num_threads = WARP_GROUP_SIZE * num_consumer + WARP_GROUP_SIZE

    alias pipeline_stages = 3
    alias smem_size = pipeline_stages * (
        a_smem_layout.size() * sizeof[a_type]()
        + b_smem_layout.size() * sizeof[b_type]()
        + (sizeof[Int64]() * 2)
    )

    alias kernel = tma_wgmma_warp_specialized_gemm_kernel[
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
        cluster_shape=cluster_shape,
        transpose_b=True,
        num_threads=num_threads,
        pipeline_stages=pipeline_stages,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )

    ctx.synchronize()

    with vendor_blas.Handle() as handle:
        vendor_blas.matmul(
            ctx,
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )

    ctx.synchronize()

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy_from_device(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    alias rtol = 1e-2
    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=rtol,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:
        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            StaticTuple[Int32, 3](1, 2, 1),
            num_consumer=2,
        ](ctx, static[256](), static[256](), static[128]())

        test_warp_specialize_gemm_with_multicasting[
            64,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            StaticTuple[Int32, 3](1, 2, 1),
        ](ctx, static[256](), static[64](), static[128]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            StaticTuple[Int32, 3](2, 1, 1),
            num_consumer=2,
        ](ctx, static[128](), static[512](), static[128]())

        test_warp_specialize_gemm_with_multicasting[
            64,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            StaticTuple[Int32, 3](2, 1, 1),
        ](ctx, static[128](), static[128](), static[128]())

        test_warp_specialize_gemm_with_multicasting[
            256,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            StaticTuple[Int32, 3](2, 2, 1),
        ](ctx, static[256](), static[512](), static[128]())

        test_warp_specialize_gemm_with_multicasting[
            64,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            StaticTuple[Int32, 3](2, 2, 1),
            num_consumer=2,
        ](ctx, static[256](), static[128](), static[128]())

        alias wgmma_n = List[Int](8, 32, 64, 128, 256)
        alias num_ins = 5

        print("# 2x1 warp specialized gemm with multicasting tests")

        @parameter
        for i in range(num_ins):
            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](1, 2, 1),
            ](ctx, static[1024](), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](1, 2, 1),
                num_consumer=2,
            ](ctx, dynamic(1024), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](1, 2, 1),
            ](ctx, dynamic(199), static[1024](), static[1024]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](1, 2, 1),
            ](ctx, dynamic(200), static[512](), static[256]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](1, 2, 1),
                num_consumer=2,
            ](ctx, dynamic(201), static[2048](), static[256]())

        print("# 1x2 warp specialized gemm with multicasting tests")

        @parameter
        for i in range(num_ins):
            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 1, 1),
                num_consumer=2,
            ](ctx, static[1024](), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 1, 1),
            ](ctx, dynamic(1024), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 1, 1),
                num_consumer=2,
            ](ctx, dynamic(99), static[1024](), static[1024]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 1, 1),
            ](ctx, dynamic(100), static[512](), static[256]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 1, 1),
                num_consumer=2,
            ](ctx, dynamic(201), static[2048](), static[256]())

        print("# 2x2 warp specialized gemm with multicasting tests")

        @parameter
        for i in range(num_ins):
            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 2, 1),
                num_consumer=2,
            ](ctx, static[1024](), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 2, 1),
            ](ctx, dynamic(1024), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 2, 1),
                num_consumer=2,
            ](ctx, dynamic(199), static[1024](), static[1024]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 2, 1),
            ](ctx, dynamic(200), static[512](), static[256]())

            test_warp_specialize_gemm_with_multicasting[
                wgmma_n[i],
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 2, 1),
                num_consumer=2,
            ](ctx, dynamic(201), static[2048](), static[256]())
