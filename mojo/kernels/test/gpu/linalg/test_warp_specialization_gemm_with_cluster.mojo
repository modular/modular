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
from utils.numerics import get_accum_type
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
from linalg.matmul_sm90 import warp_specialize_gemm_with_multicasting

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4


def test_warp_specialize_gemm_with_multicasting[
    wgmma_n: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    cluster_shape: StaticTuple[Int32, 3],
    num_consumer: Int = 1,
    transpose_b: Bool = True,
    partitioned_multicast: Bool = False,
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
        " MULTICAST MODE: ",
        "PARTITIONED" if partitioned_multicast else "BROADCAST",
    )

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

    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        block_tile_shape = Index(128, wgmma_n, 64),
        cluster_shape=cluster_shape,
        wgmma_n=wgmma_n,
        num_consumer=num_consumer,
        partitioned_multicast=partitioned_multicast,
        use_persistant_kernel=False,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
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


def main():
    with DeviceContext() as ctx:

        @parameter
        for multicast_mode in range(2):
            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](1, 2, 1),
                num_consumer=2,
                partitioned_multicast = Bool(multicast_mode),
            ](ctx, static[256](), static[256](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                64,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](1, 2, 1),
                partitioned_multicast = Bool(multicast_mode),
            ](ctx, static[256](), static[64](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 1, 1),
                num_consumer=2,
                partitioned_multicast = Bool(multicast_mode),
            ](ctx, static[128](), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                64,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 1, 1),
                partitioned_multicast = Bool(multicast_mode),
            ](ctx, static[128](), static[128](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                256,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 2, 1),
                partitioned_multicast = Bool(multicast_mode),
            ](ctx, static[256](), static[512](), static[128]())

            test_warp_specialize_gemm_with_multicasting[
                64,
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                StaticTuple[Int32, 3](2, 2, 1),
                num_consumer=2,
                partitioned_multicast = Bool(multicast_mode),
            ](ctx, static[256](), static[128](), static[128]())

        alias wgmma_n = List[Int](8, 32, 64, 128, 256)

        print("# 2x1 warp specialized gemm with multicasting tests")

        @parameter
        for multicast_mode in range(2):

            @parameter
            for i in range(len(wgmma_n)):
                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](1, 2, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, static[1024](), static[512](), static[128]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](1, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(1024), static[512](), static[128]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](1, 2, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(199), static[1024](), static[1024]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](1, 2, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(200), static[512](), static[256]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](1, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(201), static[2048](), static[256]())

        print("# 1x2 warp specialized gemm with multicasting tests")

        @parameter
        for multicast_mode in range(2):

            @parameter
            for i in range(len(wgmma_n)):
                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 1, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, static[1024](), static[512](), static[128]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 1, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(1024), static[512](), static[128]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 1, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(99), static[1024](), static[1024]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 1, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(100), static[512](), static[256]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 1, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(201), static[2048](), static[256]())

        print("# 2x2 warp specialized gemm with multicasting tests")

        @parameter
        for multicast_mode in range(2):

            @parameter
            for i in range(len(wgmma_n)):
                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, static[1024](), static[512](), static[128]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 2, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(1024), static[512](), static[128]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(199), static[1024](), static[1024]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 2, 1),
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(200), static[512](), static[256]())

                test_warp_specialize_gemm_with_multicasting[
                    wgmma_n[i],
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    StaticTuple[Int32, 3](2, 2, 1),
                    num_consumer=2,
                    partitioned_multicast = Bool(multicast_mode),
                ](ctx, dynamic(201), static[2048](), static[256]())
