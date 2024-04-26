# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s

from math import div_ceil, isclose, isnan
from buffer import NDBuffer
from buffer.list import DimList
from collections.optional import OptionalReg
from memory.unsafe import DTypePointer
from memory.reference import _GPUAddressSpace as AddressSpace
from LinAlg.MatmulGPU import matmul_kernel_naive
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
)
from gpu.host import Context, Function, synchronize, Stream, CacheConfig
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.mma import mma, ld_matrix
from gpu.memory import (
    async_copy,
    async_copy_wait_all,
    async_copy_commit_group,
    async_copy_wait_group,
)
from testing import assert_almost_equal
from sys import argv
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.int_tuple import IntTuple, fill_like
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    copy_dram_to_sram_async,
    copy_sram_to_local,
    copy_local_to_dram,
    _swizzle_signature,
)
from layout.swizzle import Swizzle
from gpu.device_print import _printf
from pathlib import Path


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# Shape for tf32 mma.
alias MMA_M = 16
alias MMA_N = 8
alias MMA_K = 8


# Mask ^ tid's 2 least significant and every 8 threads share one mask.
# This reproduces the thread map in Cutlass when BK=16.
@always_inline
fn xor_2bits_per8T[type: DType](tid: Scalar[type]) -> Scalar[type]:
    return Swizzle[2, 0, 3]()(tid)


@always_inline
fn ld_mma[
    num_matrices: Int,
    # Refactor the three parameters with ComposedLayout
    thread_layout: Layout,
    swizzle: OptionalReg[_swizzle_signature] = None,
    *,
    # work around parameter deduction
    __layout: Layout,
    __element_layout: Layout,
    __index_type: DType,
    __masked: Bool,
](
    mat: LayoutTensor[
        _,
        __layout,
        address_space = AddressSpace.SHARED,
        element_layout=__element_layout,
        index_type=__index_type,
        masked=__masked,
    ],
    offset: Int,
) -> SIMD[mat.dtype, num_matrices]:
    constrained[
        num_matrices == 2 or num_matrices == 4,
        "Only support loading 2 or 4 matrices.",
    ]()

    # TODO: Either optimize signed int division or restrict this to uint32.
    var lane_id = UInt32(ThreadIdx.x()) % WARP_SIZE

    alias stride = thread_layout.stride[0].value()
    alias simd_size = simdwidthof[mat.dtype]()

    # TODO: the index calculation can be refactored when layout(i) works on GPU.
    # var row_offset = thread_layout(lane_id)
    var row_offset = lane_id % 16 * stride + lane_id // 16 if num_matrices == 4 else lane_id % 8 * stride + lane_id // 8
    row_offset += offset

    @parameter
    if swizzle:
        alias swizzle_fn = swizzle.value()
        row_offset = swizzle_fn(row_offset)

    row_offset = row_offset * simd_size

    return ld_matrix[mat.dtype, num_matrices](mat.ptr + row_offset)


fn multistage_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
](
    c: LayoutTensor[c_type, c_layout],
    a: LayoutTensor[a_type, a_layout],
    b: LayoutTensor[b_type, b_layout],
):
    constrained[
        c_type == DType.float32
        and a_type == DType.float32
        and b_type == DType.float32,
        "Only support tf32 mma",
    ]()

    constrained[BK == 16, "Only support BK = 16."]()

    alias simd_size = simdwidthof[c_type]()

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == num_threads // WARP_SIZE,
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = ThreadIdx.x()
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE

    # Coordinates of the current warp.
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    var a_smem_tiles = LayoutTensor[
        a_type,
        Layout.row_major(num_pipeline_stages * BM, BK),
        address_space = AddressSpace.SHARED,
    ].stack_allocation().split[num_pipeline_stages]()

    var b_smem_tiles = LayoutTensor[
        b_type,
        Layout.row_major(num_pipeline_stages * BN, BK),
        address_space = AddressSpace.SHARED,
    ].stack_allocation().split[num_pipeline_stages]()

    alias thread_async_copy_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    # Prefetch (num_pipeline_stages - 1) stages.
    @unroll
    for stage in range(num_pipeline_stages - 1):
        copy_dram_to_sram_async[
            src_thread_layout=thread_async_copy_layout,
            dst_thread_layout=thread_async_copy_layout,
            swizzle=xor_2bits_per8T,
        ](
            a_smem_tiles[stage].vectorize[1, simd_size](),
            a.tile[BM, BK](BlockIdx.y(), stage).vectorize[1, simd_size](),
        )

        copy_dram_to_sram_async[
            src_thread_layout=thread_async_copy_layout,
            dst_thread_layout=thread_async_copy_layout,
            swizzle=xor_2bits_per8T,
        ](
            b_smem_tiles[stage].vectorize[1, simd_size](),
            b.tile[BN, BK](BlockIdx.x(), stage).vectorize[1, simd_size](),
        )

        async_copy_commit_group()

    # Guard stage 0.
    async_copy_wait_group(num_pipeline_stages - 2)
    barrier()

    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    # Register tiles.
    # TODO: parameterize fragment size based on data type.
    var a_reg_tiles = LayoutTensor[
        a_type, Layout.row_major(2 * num_m_mmas, 4)
    ].stack_allocation().vectorize[1, 4]().split[2]()
    var b_reg_tiles = LayoutTensor[
        b_type, Layout.row_major(2 * num_n_mmas, 2)
    ].stack_allocation().vectorize[1, 2]().split[2]()
    var c_reg_tile = LayoutTensor[
        c_type, Layout.row_major(num_m_mmas * num_n_mmas, 4)
    ].stack_allocation().vectorize[1, 4]()

    c_reg_tile.fill(0)

    alias a_frag_type = a_reg_tiles[0].element_type
    alias b_frag_type = b_reg_tiles[0].element_type

    # Load shared -> registers for stage 0's mma.
    # TODO: remove the cast.
    var a_warp_tile = a_smem_tiles[0].tile[WM, BK](int(warp_y), 0)
    var b_warp_tile = b_smem_tiles[0].tile[WN, BK](int(warp_x), 0)

    # TODO: possbile to not rebind?
    @unroll
    for m_mma in range(num_m_mmas):
        var a_mma_tile = a_warp_tile.tile[MMA_M, BK](m_mma, 0)
        a_reg_tiles[0][m_mma, 0] = rebind[a_frag_type](
            ld_mma[
                4,
                Layout(IntTuple(16, 2), IntTuple(BK // simd_size, 1)),
                swizzle=xor_2bits_per8T,
            ](a_mma_tile, 0)
        )

    @unroll
    for n_mma in range(num_n_mmas):
        var b_mma_tile = b_warp_tile.tile[MMA_N, BK](n_mma, 0)
        b_reg_tiles[0][n_mma, 0] = rebind[b_frag_type](
            ld_mma[
                2,
                Layout(IntTuple(8, 2), IntTuple(BK // simd_size, 1)),
                swizzle=xor_2bits_per8T,
            ](b_mma_tile, 0)
        )

    var num_k_tiles = div_ceil(K, BK)

    for k_tile_id in range(num_k_tiles):
        var stage = k_tile_id % num_pipeline_stages

        # TODO: remove the cast.
        var a_warp_tile = a_smem_tiles[stage].tile[WM, BK](int(warp_y), 0)
        var b_warp_tile = b_smem_tiles[stage].tile[WN, BK](int(warp_x), 0)

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @unroll
        for k_mma in range(num_k_mmas):
            var current = k_mma % 2
            var next = (k_mma + 1) % 2
            var next_stage = (k_tile_id + 1) % num_pipeline_stages

            if k_mma == num_k_mmas - 1:
                a_warp_tile = a_smem_tiles[next_stage].tile[WM, BK](
                    int(warp_y), 0
                )
                b_warp_tile = b_smem_tiles[next_stage].tile[WN, BK](
                    int(warp_x), 0
                )

            @unroll
            for m_mma in range(num_m_mmas):
                var a_mma_tile = a_warp_tile.tile[MMA_M, BK](m_mma, 0)
                a_reg_tiles[next][m_mma, 0] = rebind[a_frag_type](
                    ld_mma[
                        4,
                        Layout(IntTuple(16, 2), IntTuple(BK // simd_size, 1)),
                        swizzle=xor_2bits_per8T,
                    ](a_mma_tile, (k_mma + 1) % num_k_mmas * MMA_K // simd_size)
                )

            @unroll
            for n_mma in range(num_n_mmas):
                var b_mma_tile = b_warp_tile.tile[MMA_N, BK](n_mma, 0)
                b_reg_tiles[next][n_mma, 0] = rebind[b_frag_type](
                    ld_mma[
                        2,
                        Layout(IntTuple(8, 2), IntTuple(BK // simd_size, 1)),
                        swizzle=xor_2bits_per8T,
                    ](b_mma_tile, (k_mma + 1) % num_k_mmas * MMA_K // simd_size)
                )

            @unroll
            for m_mma in range(num_m_mmas):

                @unroll
                for n_mma in range(num_n_mmas):
                    mma(
                        c_reg_tile[m_mma * num_n_mmas + n_mma, 0],
                        a_reg_tiles[current][m_mma, 0],
                        b_reg_tiles[current][n_mma, 0],
                        c_reg_tile[m_mma * num_n_mmas + n_mma, 0],
                    )

            if k_mma + 2 == num_k_mmas:
                # Prefetch next k tile from global memory to current shared memory buffer.
                var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1
                var prefetch_stage = prefetch_tile_id % num_pipeline_stages

                # TODO: Extend the async copy instrinsic to creat dummy copies. The
                # prefetch for the three two iterations should be dummy.
                copy_dram_to_sram_async[
                    src_thread_layout=thread_async_copy_layout,
                    dst_thread_layout=thread_async_copy_layout,
                    swizzle=xor_2bits_per8T,
                ](
                    a_smem_tiles[prefetch_stage].vectorize[1, simd_size](),
                    a.tile[BM, BK](
                        BlockIdx.y(), prefetch_tile_id % num_k_tiles
                    ).vectorize[1, simd_size](),
                )

                copy_dram_to_sram_async[
                    src_thread_layout=thread_async_copy_layout,
                    dst_thread_layout=thread_async_copy_layout,
                    swizzle=xor_2bits_per8T,
                ](
                    b_smem_tiles[prefetch_stage].vectorize[1, simd_size](),
                    b.tile[BN, BK](
                        BlockIdx.x(), prefetch_tile_id % num_k_tiles
                    ).vectorize[1, simd_size](),
                )

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(num_pipeline_stages - 2)
                barrier()

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN](BlockIdx.y(), BlockIdx.x())
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](int(warp_y), int(warp_x))

    @unroll
    for m_mma in range(num_m_mmas):

        @unroll
        for n_mma in range(num_n_mmas):
            var c_gmem_mma_tile = c_gmem_warp_tile.tile[MMA_M, MMA_N](
                m_mma, n_mma
            )
            var c_frag = c_gmem_mma_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](int(lane_id))
            var c_reg = c_reg_tile[m_mma * num_n_mmas + n_mma, 0]
            c_frag.aligned_store[2](0, 0, SIMD[c_type, 2](c_reg[0], c_reg[1]))
            c_frag.aligned_store[2](1, 0, SIMD[c_type, 2](c_reg[2], c_reg[3]))


fn test() raises:
    alias num_threads = 128
    alias num_pipeline_stages = 4
    alias M = 1024
    alias N = 1024
    alias K = 128
    alias BM = 64
    alias BN = 128
    alias BK = 16
    alias WM = 32
    alias WN = 64

    var stream = Stream()

    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(N, K)
    alias c_layout = Layout.row_major(M, N)

    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(K * N)
    var b_trans_host = DTypePointer[DType.float32].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var c_host_ref = DTypePointer[DType.float32].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = m * K + k

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = k * N + n
            b_trans_host[n * K + k] = k * N + n

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)
    var c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_trans_host, K * N)

    var c_buffer = NDBuffer[DType.float32, 2, DimList(M, N)](c_device)
    var a_buffer = NDBuffer[DType.float32, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[DType.float32, 2, DimList(K, N)](b_device)

    var c_tensor = LayoutTensor[DType.float32, c_layout](c_device)
    var a_tensor = LayoutTensor[DType.float32, a_layout](a_device)
    var b_tensor = LayoutTensor[DType.float32, b_layout](b_device)

    alias gemm = multistage_gemm[
        DType.float32,
        c_layout,
        DType.float32,
        a_layout,
        DType.float32,
        b_layout,
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
    ]
    # TODO: The cache config doesn't really help here, see #38391.
    var func = Function[__type_of(gemm), gemm](
        threads_per_block=num_threads,
        cache_config=CacheConfig.PREFER_SHARED,
        # dump_ptx=Path("./pipelined-gemm.ptx"),
    )

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(stream: Stream) raises:
            for i in range(nrun):
                func(
                    c_tensor,
                    a_tensor,
                    b_tensor,
                    grid_dim=(div_ceil(N, BN), div_ceil(M, BM), 1),
                    block_dim=(num_threads, 1, 1),
                    stream=stream,
                )

        # Warmup
        for i in range(nwarmup):
            run_func(stream)

        var nstime = time_function[run_func](stream) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlops/s", TFlop / sectime)

    func(
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(div_ceil(N, BN), div_ceil(M, BM), 1),
        block_dim=(num_threads, 1, 1),
        stream=stream,
    )

    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)
    _copy_host_to_device(b_device, b_host, K * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        DType.float32, DType.float32, DType.float32, BLOCK_DIM
    ]
    var func_naive = Function[__type_of(gemm_naive), gemm_naive](
        threads_per_block=256
    )
    var c_buffer_ref = NDBuffer[DType.float32, 2, DimList(M, N)](c_device_ref)
    func_naive(
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=0.01):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=0.01)

    _free(c_device)
    _free(c_device_ref)
    _free(a_device)
    _free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()
    b_trans_host.free()

    _ = func^
    _ = func_naive^
    _ = stream^


def main():
    try:
        with Context() as ctx:
            test()
    except e:
        print("ERROR:", e)
