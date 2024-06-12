# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s

from collections.optional import OptionalReg
from math import ceildiv, isclose
from pathlib import Path
from sys import argv

from buffer import NDBuffer
from buffer.list import DimList
from gpu import WARP_SIZE, BlockIdx, ThreadIdx, barrier, lane_id
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from gpu.host import Context, FuncAttribute, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
    dynamic_shared_memory,
)
from gpu.mma import ld_matrix, mma
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy_dram_to_sram_async,
    copy_local_to_sram,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.swizzle import Swizzle
from layout.tensor_core import (
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
    TensorCore,
)
from LinAlg.MatmulCublas import cublas_matmul
from memory.reference import _GPUAddressSpace as AddressSpace
from memory.unsafe import DTypePointer
from testing import assert_almost_equal

from utils.index import Index, StaticIntTuple


# fmt: off
@always_inline
fn block_swizzle_by_scale[scale0: Int](
    block_idx: StaticIntTuple[2], grid_dim: StaticIntTuple[2]
) -> StaticIntTuple[2]:
    var scale = scale0
    var num_partitions = (1 << scale)
    while (grid_dim[0] & (num_partitions - 1)) != 0 and scale > 1:
        scale -= 1
        num_partitions = (1 << scale)

    var bx = block_idx[0] >> scale
    var by = (block_idx[1] << scale)  + ((block_idx[0]) & ((1 << scale) - 1))
    bx = bx + by // grid_dim[1] * (grid_dim[0] >> scale)
    by = by % grid_dim[1]

    return Index(bx, by)
# fmt: on


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# Mask ^ tid's 2 least significant and every 8 threads share one mask.
# This reproduces the thread map in Cutlass when BK=16.
@always_inline
fn xor_2bits_per8T[type: DType](tid: Scalar[type]) -> Scalar[type]:
    return Swizzle[2, 0, 3]()(tid)


# Figure out the math using BN, BK, dtype to define swizzle parameters.
@always_inline
fn xor_3bits_per16T[type: DType](tid: Scalar[type]) -> Scalar[type]:
    return Swizzle[3, 0, 4]()(tid)


@always_inline
fn identity[type: DType](tid: Scalar[type]) -> Scalar[type]:
    return tid


@always_inline
fn args_to_tuple[swap: Bool](arg_0: Int, arg_1: Int) -> Tuple[Int, Int]:
    @parameter
    if swap:
        return Tuple(arg_1, arg_0)
    else:
        return Tuple(arg_0, arg_1)


fn multistage_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
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
    # Hold on adding fp16 because it counld have differnet precisions than bf16.
    constrained[
        (a_type == DType.float32 or a_type == DType.bfloat16)
        and a_type == b_type == c_type,
        "Pipeline gemm only supports tf32 or BF16 mma",
    ]()

    constrained[
        (BK == 16 and a_type == DType.float32)
        or (BK == 32 and a_type == DType.bfloat16),
        "Pipeline gemm only supports BK = 16 w/ FP32 and BK = 32 w/ BF16.",
    ]()

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

    # Only apply block swizzling for half precision types.
    alias swizzle_block = a_type.is_half_float() and b_type.is_half_float()

    var block_idx = block_swizzle_by_scale[3](
        Index(BlockIdx.x(), BlockIdx.y()), Index(N // BN, M // BM)
    ) if swizzle_block else Index(BlockIdx.x(), BlockIdx.y())

    # Coordinates of the current warp.
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    # Prepare circular shared memory buffer for A and B.
    # Each pipeline stage has its own buffer.
    var a_smem = dynamic_shared_memory[
        Scalar[a_type], alignment = alignof[SIMD[a_type, simd_size]]()
    ]()
    alias a_smem_size = num_pipeline_stages * BM * BK
    var a_smem_iter = LayoutTensorIter[
        a_type, Layout.row_major(BM, BK), AddressSpace.SHARED, circular=True
    ](a_smem, a_smem_size)

    # There is one pre-allocated shared buffer. Explicitly offset B after at A's end.
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[b_type]]()
    alias b_smem_size = num_pipeline_stages * BK * BN
    alias BD_0 = BN if transpose_b else BK
    alias BD_1 = BK if transpose_b else BN
    alias b_smem_layout = Layout.row_major(BD_0, BD_1)
    var b_smem_iter = LayoutTensorIter[
        b_type, b_smem_layout, AddressSpace.SHARED, circular=True
    ](b_smem, b_smem_size)

    # global memory iterator
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], 0)
    var b_tile_coords = args_to_tuple[transpose_b](0, block_idx[0])
    alias b_tile_axis = 1 if transpose_b else 0
    var b_gmem_iter = b.tiled_iterator[BD_0, BD_1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    alias async_copy_a_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    alias async_copy_b_layout = Layout.row_major(
        num_threads * simd_size // BD_1, BD_1 // simd_size
    )
    alias async_copy_b_swizzle = None if transpose_b else (
        OptionalReg[_swizzle_signature](
            xor_2bits_per8T if a_type == DType.float32 else xor_3bits_per16T
        )
    )

    # Prefetch (num_pipeline_stages - 1) stages.
    @parameter
    for stage in range(num_pipeline_stages - 1):
        var a_smem_tile = a_smem_iter.next(stage).get()
        var b_smem_tile = b_smem_iter.next(stage).get()

        copy_dram_to_sram_async[
            thread_layout=async_copy_a_layout,
            swizzle=xor_2bits_per8T,
        ](
            a_smem_tile.vectorize[1, simd_size](),
            a_gmem_iter.get().vectorize[1, simd_size](),
        )

        copy_dram_to_sram_async[
            thread_layout=async_copy_b_layout, swizzle=async_copy_b_swizzle
        ](
            b_smem_tile.vectorize[1, simd_size](),
            b_gmem_iter.get().vectorize[1, simd_size](),
        )

        async_copy_commit_group()

        a_gmem_iter += 1
        b_gmem_iter += 1

    # Guard stage 0.
    async_copy_wait_group(num_pipeline_stages - 2)
    barrier()

    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_k_mmas = BK // MMA_K
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[a_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    # Register tiles.
    # TODO: parameterize fragment size based on data type.
    var a_reg_tiles = LayoutTensor[
        a_type, Layout.row_major(2 * num_m_mmas, a_frag_size)
    ].stack_allocation().vectorize[1, a_frag_size]().split[2]()
    var b_reg_tiles = LayoutTensor[
        b_type, Layout.row_major(2 * num_n_mmas, b_frag_size)
    ].stack_allocation().vectorize[1, b_frag_size]().split[2]()
    var c_reg_tile = LayoutTensor[
        accum_type, Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size)
    ].stack_allocation()

    c_reg_tile.fill(0)

    var a_warp_tile = a_smem_iter.get().tile[WM, BK](int(warp_y), 0)

    # TODO: warp the following in the tile method, maybe tile[shape: IntTuple].
    # I can't use b_warp_tile = ... if transpose_b else ... because the operands
    # are deduced as different types since their layout are different.
    alias b_wtile_dim0 = WN if transpose_b else BK
    alias b_wtile_dim1 = BK if transpose_b else WN
    var b_wtile_coord0 = int(warp_x) if transpose_b else 0
    var b_wtile_coord1 = 0 if transpose_b else int(warp_x)
    var b_warp_tile = b_smem_iter.get().tile[b_wtile_dim0, b_wtile_dim1](
        b_wtile_coord0, b_wtile_coord1
    )

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b]()

    mma_op.load_a(a_warp_tile, a_reg_tiles[0])
    mma_op.load_b(b_warp_tile, b_reg_tiles[0])

    var num_k_tiles = ceildiv(K, BK)

    for k_tile_id in range(num_k_tiles):
        var a_iter = a_smem_iter.next(k_tile_id)
        var b_iter = b_smem_iter.next(k_tile_id)

        var a_warp_tile = a_iter.get().tile[WM, BK](int(warp_y), 0)
        var b_warp_tile = b_iter.get().tile[b_wtile_dim0, b_wtile_dim1](
            b_wtile_coord0,
            b_wtile_coord1,
        )

        # Perform prefetch registers and mma until current shared memory tile's
        # data has all been loaded to registers.
        @parameter
        for k_mma in range(num_k_mmas):
            var current = k_mma % 2
            var next = (k_mma + 1) % 2

            if k_mma == num_k_mmas - 1:
                var a_smem_next_tile = a_iter.next().get()
                var b_smem_next_tile = b_iter.next().get()

                a_warp_tile = a_smem_next_tile.tile[WM, BK](int(warp_y), 0)
                b_warp_tile = b_smem_next_tile.tile[b_wtile_dim0, b_wtile_dim1](
                    b_wtile_coord0, b_wtile_coord1
                )

            mma_op.load_a(
                a_warp_tile, a_reg_tiles[next], (k_mma + 1) % num_k_mmas
            )
            mma_op.load_b(
                b_warp_tile, b_reg_tiles[next], (k_mma + 1) % num_k_mmas
            )

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    # fmt: off
                    mma(
                        c_reg_tile.vectorize[1, c_frag_size]()[n_mma * num_m_mmas + m_mma, 0],
                        a_reg_tiles[current][m_mma, 0],
                        b_reg_tiles[current][n_mma, 0],
                        c_reg_tile.vectorize[1, c_frag_size]()[n_mma * num_m_mmas + m_mma, 0],
                    )
                    # fmt: on

            if k_mma + 2 == num_k_mmas:
                var prefetch_tile_id = k_tile_id + num_pipeline_stages - 1

                # Prefetch one k tile (if valid) from global memory to current
                # shared memory buffer.
                if prefetch_tile_id < num_k_tiles:
                    # fmt: off
                    var a_smem_prefetch_tile = a_iter.next(num_pipeline_stages - 1).get()
                    var b_smem_prefetch_tile = b_iter.next(num_pipeline_stages - 1).get()
                    # fmt: on

                    # TODO: Extend the async copy instrinsic to creat dummy copies. The
                    # prefetch for the three two iterations should be dummy.
                    copy_dram_to_sram_async[
                        thread_layout=async_copy_a_layout,
                        swizzle=xor_2bits_per8T,
                    ](
                        a_smem_prefetch_tile.vectorize[1, simd_size](),
                        a_gmem_iter.get().vectorize[1, simd_size](),
                    )

                    copy_dram_to_sram_async[
                        thread_layout=async_copy_b_layout,
                        swizzle=async_copy_b_swizzle,
                    ](
                        b_smem_prefetch_tile.vectorize[1, simd_size](),
                        b_gmem_iter.get().vectorize[1, simd_size](),
                    )

                    a_gmem_iter += 1
                    b_gmem_iter += 1

                async_copy_commit_group()

                # Guard the next k tile's shared memory buffer.
                async_copy_wait_group(num_pipeline_stages - 2)
                barrier()

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN](block_idx[1], block_idx[0])
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](int(warp_y), int(warp_x))

    # Store FP32 mma results to half precision buffer in global memory.
    # Each thread's fragment has 2x2 fp32 values. Casting to half float and
    # directly storing to global memory results in 2 4B writes. Following cutlass,
    # we stage the fragments in shared memory so that each thread can store 16B.
    @parameter
    if c_type.is_half_float():
        # Stage fragments in shared memory. Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            accum_type,
            Layout.row_major(BM, BN),
            address_space = AddressSpace.SHARED,
        ](a_smem.bitcast[Scalar[accum_type]]())
        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            int(warp_y), int(warp_x)
        )
        copy_local_to_sram[thread_layout = Layout.row_major(8, 4)](
            accum_smem_warp_tile.vectorize[1, 2](),
            c_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                num_threads * simd_size // BN, BN // simd_size
            )
        ](
            c_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )

    # Store FP32 results to FP32 buffer in global memory.
    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            c_gmem_warp_tile.vectorize[1, 2](),
            c_reg_tile.bitcast[c_type]().vectorize[1, 2]().transpose(),
        )


fn test[type: DType, transpose_b: Bool]() raises:
    alias num_pipeline_stages = 4
    alias M = 8192
    alias N = 8192
    alias K = 128
    alias BM = 128
    alias BN = 128
    alias BK = 32 if type == DType.bfloat16 else 16
    alias WM = 64
    alias WN = 64
    alias shared_mem_bytes = 80 * 1024

    alias num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

    var stream = Stream()

    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    alias c_layout = Layout.row_major(M, N)

    var a_host = DTypePointer[type].alloc(M * K)
    var b_host = DTypePointer[type].alloc(K * N)
    var b_trans_host = DTypePointer[type].alloc(K * N)
    var c_host = DTypePointer[type].alloc(M * N)
    var c_host_ref = DTypePointer[type].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = m * K + k

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = k * N + n

            @parameter
            if transpose_b:
                b_trans_host[n * K + k] = k * N + n
            else:
                b_trans_host[k * N + n] = k * N + n

    var a_device = _malloc[type](M * K)
    var b_device = _malloc[type](K * N)
    var c_device = _malloc[type](M * N)
    var c_device_ref = _malloc[type](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_trans_host, K * N)

    var a_buffer = NDBuffer[type, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[type, 2, DimList(K, N)](b_device)

    var c_tensor = LayoutTensor[type, c_layout](c_device)
    var a_tensor = LayoutTensor[type, a_layout](a_device)
    var b_tensor = LayoutTensor[type, b_layout](b_device)

    alias gemm = multistage_gemm[
        type,  # c_type
        c_layout,
        type,  # a_type
        a_layout,
        type,  # b_type
        b_layout,
        transpose_b,
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_pipeline_stages,
    ]
    # TODO: The cache config doesn't really help here, see #38391.
    var func = Function[gemm](
        threads_per_block=num_threads,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            shared_mem_bytes
        ),
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_ptx=Path("./pipeline-gemm.ptx"),
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
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                    block_dim=(num_threads, 1, 1),
                    shared_mem_bytes=shared_mem_bytes,
                    stream=stream,
                )

        # Warmup
        for i in range(nwarmup):
            func(
                c_tensor,
                a_tensor,
                b_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(num_threads, 1, 1),
                shared_mem_bytes=shared_mem_bytes,
                stream=stream,
            )

        var nstime = time_function[run_func](stream) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(
            "Tranpose B ",
            transpose_b,
            nrun,
            " runs avg(s)",
            sectime,
            "TFlops/s",
            TFlop / sectime,
        )

    func(
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=shared_mem_bytes,
        stream=stream,
    )

    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)

    var c_buffer_ref = NDBuffer[type, 2, DimList(M, N)](c_device_ref)

    var handle = Pointer[cublasContext]()
    check_cublas_error(cublasCreate(Pointer.address_of(handle)))
    check_cublas_error(
        cublas_matmul(
            handle,
            c_buffer_ref,
            a_buffer,
            b_buffer,
            c_row_major=True,
            transpose_b=transpose_b,
        )
    )
    check_cublas_error(cublasDestroy(handle))

    synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    var rtol = 0.01 if transpose_b else 0.002
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=rtol):
            print(
                i,
                c_host[i],
                c_host_ref[i],
                abs((c_host[i] - c_host_ref[i]) / c_host_ref[i]),
            )
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=rtol)

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
    _ = stream^


def main():
    try:
        with Context() as ctx:
            test[DType.float32, False]()
            test[DType.float32, True]()
            test[DType.bfloat16, False]()
    except e:
        print("ERROR:", e)
