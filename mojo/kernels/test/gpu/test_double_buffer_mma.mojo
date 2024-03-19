# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s

from math import div_ceil, isclose, isnan
from pathlib import Path
from sys import argv

from buffer import NDBuffer
from buffer.list import DimList
from gpu import (
    WARP_SIZE,
    AddressSpace,
    BlockDim,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
)
from gpu.host import Context, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.memory import async_copy, async_copy_wait_all
from gpu.mma import mma
from Matmul import matmul_kernel_naive
from memory.unsafe import DTypePointer
from testing import assert_almost_equal


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


alias MMA_M = 16
alias MMA_N = 8
alias MMA_K = 8


fn sgemm_double_buffer[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    itype: DType,
    BM: Scalar[itype],
    BN: Scalar[itype],
    BK: Scalar[itype],
    WM: Scalar[itype],
    WN: Scalar[itype],
    TM: Scalar[itype],
    TN: Scalar[itype],
    NUM_THREADS: Scalar[itype],
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
):
    alias _uint = Scalar[itype]

    var M = Scalar[itype](c.dim[0]())
    var N = Scalar[itype](c.dim[1]())
    var K = Scalar[itype](a.dim[1]())

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    var tid = Scalar[itype](ThreadIdx.x())
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE

    # Coordinates of the current warp.
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    # Warp shape in 2D.
    alias warp_dim_x = WN // TN
    alias warp_dim_y = WM // TM
    constrained[
        warp_dim_x * warp_dim_y == WARP_SIZE,
        "Warp 2d shape doesn't match 32 threads",
    ]()

    # Pad BM to avoid back conflict
    alias pad_avoid_bank_conflict = Scalar[itype](4)
    alias BM_padded = BM + pad_avoid_bank_conflict

    # Double buffer in shared memory.
    var a_tile = stack_allocation[
        int(2 * BM_padded * BK), a_type, address_space = AddressSpace.SHARED
    ]()
    var b_tile = stack_allocation[
        int(2 * BK * BN), b_type, address_space = AddressSpace.SHARED
    ]()

    # Configure the load for A.
    # Each thread load one elements from A.
    alias num_iters_loada = (BM * BK) // NUM_THREADS
    alias num_rows_per_iter_loada = NUM_THREADS // BK
    # Current thread loads (loada_x, loada_y)
    var loada_x = tid % BK
    var loada_y = tid // BK

    # Configure the load for B.
    # Each threads load 4 elements from B using float4.
    alias simd_size_int = simdwidthof[c_type]()
    alias simd_size = Scalar[itype](simd_size_int)
    alias num_iters_loadb = (BK * BN) // NUM_THREADS // simd_size
    alias num_rows_per_iter_loadb = (NUM_THREADS * simd_size) // BN
    # Current thread loads vector at (loadb_x, loadb_y)
    var loadb_x = tid * simd_size % BN
    var loadb_y = tid * simd_size // BN

    # Cast pointers from generic to global.
    var a_gmem_ptr = a.data.address.address_space_cast[AddressSpace.GLOBAL]()
    var b_gmem_ptr = b.data.address.address_space_cast[AddressSpace.GLOBAL]()

    # Current block updates a [BM, BN] tile in C.
    # Find the this tile's coordinates in C and set the offsets in A, B.
    var c_row = BlockIdx.y() * BM
    var c_col = BlockIdx.x() * BN
    var a_gmem_tile = a_gmem_ptr + c_row * K
    var b_gmem_tile = b_gmem_ptr + c_col

    # K tile base in A and B
    var loada_gmem_ptr = a_gmem_tile + loada_y * K + loada_x
    var loadb_gmem_ptr = b_gmem_tile + loadb_y * N + loadb_x

    # Thread's loading position in A and B shared memory tile.
    var storea_smem_ptr = a_tile + (tid % BK) * BM_padded + tid // BK
    var storeb_smem_ptr = b_tile + tid * simd_size

    # Load A's first tile in K to shared memory. Transpose it while loading.
    @unroll
    for i in range(num_iters_loada):
        async_copy[4](
            loada_gmem_ptr + i * num_rows_per_iter_loada * K,
            storea_smem_ptr + i * num_rows_per_iter_loada,
        )

    # Load B's first tile in K to shared memory.
    @unroll
    for i in range(num_iters_loadb):
        async_copy[16](
            loadb_gmem_ptr + i * num_rows_per_iter_loadb * N,
            storeb_smem_ptr + i * num_rows_per_iter_loadb * BN,
        )

    async_copy_wait_all()
    barrier()

    # Shifts for switching buffer
    alias a_smem_shift = BM_padded * BK
    alias b_smem_shift = BN * BK
    alias a_gmem_shift = BK
    var b_gmem_shift = N * BK

    # Advance A and B to next k tile.
    loada_gmem_ptr += a_gmem_shift
    loadb_gmem_ptr += b_gmem_shift

    # Alternate share memory buffer for loading.
    storea_smem_ptr += a_smem_shift
    storeb_smem_ptr += b_smem_shift

    constrained[num_iters_loada <= MMA_K]()
    constrained[num_iters_loadb <= MMA_K]()

    alias num_mma_n = WN // MMA_N
    alias num_mma_m = WM // MMA_M
    alias num_mma = num_mma_m * num_mma_n

    # Thread id for thread swizzling
    var group_id = lane_id >> 2
    var thread_id_in_group = lane_id % 4
    # A smem tile is transposed
    var row_a0 = thread_id_in_group
    var row_a2 = thread_id_in_group + 4
    var row_a1 = thread_id_in_group
    var row_a3 = thread_id_in_group + 4
    var col_a0 = group_id
    var col_a1 = group_id + 8
    var col_a2 = group_id
    var col_a3 = group_id + 8
    var row_b0 = thread_id_in_group
    var row_b1 = thread_id_in_group + 4
    var col_b0_b1 = group_id
    var row_c0_c1 = group_id
    var row_c2_c3 = group_id + 8
    var col_c0 = (thread_id_in_group * 2) + (0 & 0x1)
    var col_c1 = (thread_id_in_group * 2) + (1 & 0x1)
    var col_c2 = (thread_id_in_group * 2) + (2 & 0x1)
    var col_c3 = (thread_id_in_group * 2) + (3 & 0x1)

    # Double buffer in registers (fragments in nvidia terms).
    var a_reg = NDBuffer[
        a_type, 2, DimList(2, num_mma_m * 4)
    ].stack_allocation()
    var b_reg = NDBuffer[
        b_type, 2, DimList(2, num_mma_n * 2)
    ].stack_allocation()
    var c_reg = NDBuffer[
        c_type, 3, DimList(num_mma_m, num_mma_n, 4)
    ].stack_allocation()
    c_reg.zero()

    # Load address in shared memory for fma.
    var loada_smem_ptr = a_tile + warp_y * WM  # + mma_y * simd_size
    var loadb_smem_ptr = b_tile + warp_x * WN  # + mma_x * simd_size

    alias alignment = alignof[SIMD[c_type, simd_size_int]]()

    # Load A fragments to the first buffer.
    @unroll
    for i in range(num_mma_m):
        var a0 = loada_smem_ptr.load(row_a0 * BM_padded + col_a0 + i * MMA_M)
        var a1 = loada_smem_ptr.load(row_a1 * BM_padded + col_a1 + i * MMA_M)
        var a2 = loada_smem_ptr.load(row_a2 * BM_padded + col_a2 + i * MMA_M)
        var a3 = loada_smem_ptr.load(row_a3 * BM_padded + col_a3 + i * MMA_M)
        a_reg.simd_store[4]((0, i * 4), SIMD[a_type, 4](a0, a1, a2, a3))

    # Load B fragments to the first buffer.
    @unroll
    for i in range(num_mma_n):
        var b0 = loadb_smem_ptr.load(row_b0 * BN + col_b0_b1 + i * MMA_N)
        var b1 = loadb_smem_ptr.load(row_b1 * BN + col_b0_b1 + i * MMA_N)
        b_reg.simd_store[2]((0, i * 2), SIMD[b_type, 2](b0, b1))

    var num_k_tiles = Scalar[itype](div_ceil(int(K), int(BK)))

    # Buffer id for the double buffers. They alternate.
    var buffer_id = 0
    var next_buffer_id = buffer_id ^ 0x1

    # Update (num_k_tile - 1) tiles while switching buffers.
    for k_tile_id in range(num_k_tiles - 1):

        @unroll
        for k in range(0, BK, MMA_K):
            var next_k = (k + MMA_K) % int(BK)

            if k == int(BK - MMA_K):
                async_copy_wait_all()
                barrier()

                # fmt: off
                # Switch shared memory buffer.
                loada_smem_ptr = loada_smem_ptr + a_smem_shift if (k_tile_id % 2 == 0) \
                    else loada_smem_ptr - a_smem_shift
                storea_smem_ptr = storea_smem_ptr - a_smem_shift if (k_tile_id % 2 == 0) \
                    else storea_smem_ptr + a_smem_shift
                loadb_smem_ptr = loadb_smem_ptr + b_smem_shift if (k_tile_id % 2 == 0) \
                    else loadb_smem_ptr - b_smem_shift
                storeb_smem_ptr = storeb_smem_ptr - b_smem_shift if (k_tile_id % 2 == 0) \
                    else storeb_smem_ptr + b_smem_shift
                # fmt: on

                # Advance to the next k tile.
                loada_gmem_ptr += BK
                loadb_gmem_ptr += BK * N

            # Fill the other A fragments buffer.
            @unroll
            for i in range(num_mma_m):
                var a0 = loada_smem_ptr.load(
                    (next_k + row_a0) * BM_padded + col_a0 + i * MMA_M
                )
                var a1 = loada_smem_ptr.load(
                    (next_k + row_a1) * BM_padded + col_a1 + i * MMA_M
                )
                var a2 = loada_smem_ptr.load(
                    (next_k + row_a2) * BM_padded + col_a2 + i * MMA_M
                )
                var a3 = loada_smem_ptr.load(
                    (next_k + row_a3) * BM_padded + col_a3 + i * MMA_M
                )
                a_reg.simd_store[4](
                    (next_buffer_id, i * 4), SIMD[a_type, 4](a0, a1, a2, a3)
                )

            # Fill the other B fragments buffer.
            @unroll
            for i in range(num_mma_n):
                var b0 = loadb_smem_ptr.load(
                    (next_k + row_b0) * BN + col_b0_b1 + i * MMA_N
                )
                var b1 = loadb_smem_ptr.load(
                    (next_k + row_b1) * BN + col_b0_b1 + i * MMA_N
                )
                b_reg.simd_store[2](
                    (next_buffer_id, i * 2), SIMD[b_type, 2](b0, b1)
                )

            # Load next k tile from global memory to shared memory.
            # if k < int(num_iters_loada):
            if k == 0:

                @unroll
                for kk in range(num_iters_loada):
                    async_copy[4](
                        loada_gmem_ptr + kk * num_rows_per_iter_loada * K,
                        storea_smem_ptr + kk * num_rows_per_iter_loada,
                    )
                # if k < int(num_iters_loadb):

                @unroll
                for kk in range(num_iters_loadb):
                    async_copy[16](
                        loadb_gmem_ptr + kk * num_rows_per_iter_loadb * N,
                        storeb_smem_ptr + kk * num_rows_per_iter_loadb * BN,
                    )

            @unroll
            for i in range(num_mma_m):

                @unroll
                for j in range(num_mma_n):
                    var d = c_reg.load[width=4]((i, j, 0))
                    mma(
                        d,
                        a_reg.load[width=4]((buffer_id, i * 4)),
                        b_reg.load[width=2]((buffer_id, j * 2)),
                        d,
                    )
                    c_reg.simd_store[width=4]((i, j, 0), d)

            # Alternate buffer
            buffer_id ^= 0x1
            next_buffer_id ^= 0x1

    # Last k tile.
    @unroll
    for k in range(0, BK, MMA_K):
        var next_k = (k + MMA_K) % int(BK)

        if k < int(BK - MMA_K):

            @unroll
            for i in range(num_mma_m):
                var a0 = loada_smem_ptr.load(
                    (next_k + row_a0) * BM_padded + col_a0 + i * MMA_M
                )
                var a1 = loada_smem_ptr.load(
                    (next_k + row_a1) * BM_padded + col_a1 + i * MMA_M
                )
                var a2 = loada_smem_ptr.load(
                    (next_k + row_a2) * BM_padded + col_a2 + i * MMA_M
                )
                var a3 = loada_smem_ptr.load(
                    (next_k + row_a3) * BM_padded + col_a3 + i * MMA_M
                )
                a_reg.simd_store[4](
                    (next_buffer_id, i * 4), SIMD[a_type, 4](a0, a1, a2, a3)
                )

            @unroll
            for i in range(num_mma_n):
                var b0 = loadb_smem_ptr.load(
                    (next_k + row_b0) * BN + col_b0_b1 + i * MMA_N
                )
                var b1 = loadb_smem_ptr.load(
                    (next_k + row_b1) * BN + col_b0_b1 + i * MMA_N
                )
                b_reg.simd_store[2](
                    (next_buffer_id, i * 2), SIMD[b_type, 2](b0, b1)
                )

        @unroll
        for i in range(num_mma_m):

            @unroll
            for j in range(num_mma_n):
                var d = c_reg.load[width=4]((i, j, 0))
                mma(
                    d,
                    a_reg.load[width=4]((buffer_id, i * 4)),
                    b_reg.load[width=2]((buffer_id, j * 2)),
                    d,
                )
                c_reg.simd_store[width=4]((i, j, 0), d)

        # Alternate buffer
        buffer_id ^= 0x1
        next_buffer_id ^= 0x1

    # Point to the start of the current warp.
    var c_gmem_ptr = c.data + (c_row + warp_y * WM) * N + (c_col + warp_x * WN)
    # c_gmem_ptr += mma_y * simd_size * N + mma_x * simd_size

    @unroll
    for i in range(num_mma_m):

        @unroll
        for j in range(num_mma_n):
            var c_mma_tile = c_gmem_ptr + (i * MMA_M) * N + j * MMA_N
            c_mma_tile.store[width=2](
                int(row_c0_c1 * N + col_c0), c_reg.load[width=2]((i, j, 0))
            )
            c_mma_tile.store[width=2](
                int(row_c2_c3 * N + col_c2), c_reg.load[width=2]((i, j, 2))
            )


fn test() raises:
    alias NUM_THREADS = 256
    alias M = 8192
    alias N = 8192
    alias K = 128
    alias BM = 128
    alias BN = 128
    alias BK = 16
    alias WM = 32
    alias WN = 64
    alias TM = 8
    alias TN = 8

    var stream = Stream()

    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var c_host_ref = DTypePointer[DType.float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i

    for i in range(K * N):
        b_host[i] = i + 1

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)
    var c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    var c_buffer = NDBuffer[DType.float32, 2, DimList(M, N)](c_device)
    var a_buffer = NDBuffer[DType.float32, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[DType.float32, 2, DimList(K, N)](b_device)

    alias gemm = sgemm_double_buffer[
        DType.float32,
        DimList(M, N),
        DType.float32,
        DimList(M, K),
        DType.float32,
        DimList(K, N),
        DType.uint32,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
        NUM_THREADS,
    ]
    var func = Function[__type_of(gemm), gemm](
        threads_per_block=NUM_THREADS, dump_ptx=Path("./mm.ptx")
    )

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(stream: Stream) raises:
            for i in range(nrun):
                func(
                    c_buffer,
                    a_buffer,
                    b_buffer,
                    grid_dim=(div_ceil(N, BN), div_ceil(M, BM), 1),
                    block_dim=(NUM_THREADS, 1, 1),
                    stream=stream,
                )

        # Warmup
        for i in range(nwarmup):
            run_func(stream)

        var nstime = time_function[run_func](stream) / nrun
        var sectime = nstime * 1e-9
        var TFlog = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlogs/s", TFlog / sectime)

    func(
        c_buffer,
        a_buffer,
        b_buffer,
        grid_dim=(div_ceil(N, BN), div_ceil(M, BM), 1),
        block_dim=(NUM_THREADS, 1, 1),
        stream=stream,
    )

    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        DType.float32, DType.float32, DType.float32, BLOCK_DIM
    ]
    var func_naive = Function[__type_of(gemm_naive), gemm_naive](
        threads_per_block=NUM_THREADS
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
            break
        # assert_almost_equal(c_host[i], c_host_ref[i])

    _free(c_device)
    _free(c_device_ref)
    _free(a_device)
    _free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()

    _ = func ^
    _ = func_naive ^
    _ = stream ^


def main():
    try:
        with Context() as ctx:
            test()
    except e:
        print("CUDA_ERROR:", e)
