# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose
from pathlib import Path
from sys import argv

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, AddressSpace, BlockIdx, ThreadIdx, barrier, lane_id
from gpu.host import (
    CacheConfig,
    Context,
    CudaInstance,
    Device,
    Function,
    Stream,
)
from gpu.host.event import time_function
from gpu.memory import async_copy, async_copy_wait_all
from gpu.mma import mma
from linalg.matmul_gpu import matmul_kernel_naive
from memory.unsafe import DTypePointer
from testing import assert_almost_equal


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


@always_inline
fn loada[
    itype: DType,
    BM: Scalar[itype],
    BN: Scalar[itype],
    BK: Scalar[itype],
    NUM_THREADS: Scalar[itype],
    atype: DType,
](
    M: Scalar[itype],
    N: Scalar[itype],
    K: Scalar[itype],
    warp_id: Scalar[itype],
    lane_id: Scalar[itype],
    gptr: DTypePointer[atype, address_space = AddressSpace.GLOBAL],
    sptr: DTypePointer[atype, address_space = AddressSpace.SHARED],
):
    alias MMA_M = Scalar[itype](16)
    alias MMA_N = Scalar[itype](8)
    alias MMA_K = Scalar[itype](8)
    alias num_warps = NUM_THREADS // 32

    # Configure loading A.
    alias num_loada_tiles_m = BM // MMA_M
    alias num_loada_tiles_k = BK // MMA_K
    # Each thread load 1 elment each time.
    alias num_loada_iters = BM * BK // NUM_THREADS
    # Each mma tile uses 4 warps to for loading.
    alias num_loada_tiles_per_iter = num_warps // 4
    var warp_id_in_tile = warp_id % 4
    # Point to the current buffer

    # Load A from global memory to shared memory.
    @parameter
    for i in range(num_loada_iters):
        var mma_tile_id = warp_id // 4 + i * num_loada_tiles_per_iter
        var mma_tile_x = mma_tile_id // num_loada_tiles_m
        var mma_tile_y = mma_tile_id % num_loada_tiles_m
        var gmem_ptr = gptr + mma_tile_y * MMA_M * K + mma_tile_x * MMA_K + warp_id_in_tile * 2 * K
        # t0  t1  t2  t3  t4  t5  t6  t7    <- e.g. 1st row in mma tile
        # t16 t17 t18 t19 t20 t21 t22 t23   <- e.g. 2nd row
        # ...
        # t8  t9  t10 t11 t12 t13 t14 t15   <- e.g. 8th row
        # t24 t25 t26 t27 t28 t29 t30 t31   <- e.g. 9th row
        gmem_ptr += (
            (lane_id % 16 // 8) * 8 * K + (lane_id // 16) * K + lane_id % 8
        )
        # t0  t4  t8  t12 | t1  t5  t9  t13 | t2  t6  t10 t14 | t3  t7  t11 t15
        # t16 t20 t24 t28 | t17 t21 t25 t29 | t18 t22 t26 t30 | t19 t23 t27 t31
        var smem_ptr = sptr + mma_tile_id * MMA_M * MMA_K + warp_id_in_tile * WARP_SIZE
        smem_ptr += lane_id // 16 * 16 + lane_id % 4 * 4 + lane_id // 4 % 4
        async_copy[4](gmem_ptr, smem_ptr)


@always_inline
fn loadb[
    itype: DType,
    BM: Scalar[itype],
    BN: Scalar[itype],
    BK: Scalar[itype],
    NUM_THREADS: Scalar[itype],
    btype: DType,
](
    M: Scalar[itype],
    N: Scalar[itype],
    K: Scalar[itype],
    warp_id: Scalar[itype],
    lane_id: Scalar[itype],
    gptr: DTypePointer[btype, address_space = AddressSpace.GLOBAL],
    sptr: DTypePointer[btype, address_space = AddressSpace.SHARED],
):
    alias MMA_M = Scalar[itype](16)
    alias MMA_N = Scalar[itype](8)
    alias MMA_K = Scalar[itype](8)
    alias num_warps = NUM_THREADS // 32

    # Configure loading B.
    alias num_loadb_tiles_k = BK // MMA_K
    alias num_loadb_tiles_n = BN // MMA_N
    # Each thread load 1 elment each time.
    alias num_loadb_iters = BK * BN // NUM_THREADS // 2
    # Each mma tile is loaded by 1 warp.
    alias num_loadb_tiles_per_iter = num_warps

    # Load B from global memory to shared memory.
    @parameter
    for i in range(num_loadb_iters):
        var mma_tile_id = warp_id + i * num_loadb_tiles_per_iter
        var mma_tile_x = mma_tile_id % num_loadb_tiles_n
        var mma_tile_y = mma_tile_id // num_loadb_tiles_n
        var gmem_ptr = gptr + mma_tile_y * MMA_K * N + mma_tile_x * MMA_N
        # t0  t1  t2  t3  t4  t5  t6  t7
        # t8  t9  t10 t11 t12 t13 t14 t15
        # t16 t17 t18 t19 t20 t21 t22 t23
        # t24 t25 t26 t27 t28 t29 t30 t31
        gmem_ptr += lane_id // 8 * N + lane_id % 8
        # t0, ,t8, ,t16, ,t24, ,t1, ,t9, ,t17, ,t25, ...
        var smem_ptr = sptr + mma_tile_id * MMA_K * MMA_N
        smem_ptr += lane_id % 8 * 8 + lane_id // 8 * 2
        async_copy[4](gmem_ptr, smem_ptr)
        # Load next 4 rows.
        # , ,t0, ,t8, ,t16, ,t24, ,t1, ,t9, ,t17, ,t25, ...
        gmem_ptr += 4 * N
        smem_ptr += 1
        async_copy[4](gmem_ptr, smem_ptr)


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
    NUM_THREADS: Scalar[itype],
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
):
    constrained[
        NUM_THREADS == 128 or NUM_THREADS == 256,
        "Only support 128 or 256 threads",
    ]()

    alias _uint = Scalar[itype]

    alias MMA_M = Scalar[itype](16)
    alias MMA_N = Scalar[itype](8)
    alias MMA_K = Scalar[itype](8)

    var M = Scalar[itype](c.dim[0]())
    var N = Scalar[itype](c.dim[1]())
    var K = Scalar[itype](a.dim[1]())

    alias num_warps = NUM_THREADS // WARP_SIZE
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    var tid = Scalar[itype](ThreadIdx.x())
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE

    # Coordinates of the current warp.
    var warp_x = warp_id % num_warps_n
    var warp_y = warp_id // num_warps_n

    # Double buffer in shared memory.
    alias a_smem_size = BM * BK
    alias b_smem_size = BN * BK
    var a_tile = stack_allocation[
        int(2 * a_smem_size), a_type, address_space = AddressSpace.SHARED
    ]()
    var b_tile = stack_allocation[
        int(2 * b_smem_size), b_type, address_space = AddressSpace.SHARED
    ]()

    # Cast pointers from generic to global.
    var a_gmem_ptr = a.data.address.bitcast[
        address_space = AddressSpace.GLOBAL
    ]()
    var b_gmem_ptr = b.data.address.bitcast[
        address_space = AddressSpace.GLOBAL
    ]()

    # Current block updates a [BM, BN] tile in C.
    # Find the this tile's coordinates in C and set the offsets in A, B.
    var c_row = BlockIdx.y() * BM
    var c_col = BlockIdx.x() * BN
    var a_gmem_tile = a_gmem_ptr + int(c_row * K)
    var b_gmem_tile = b_gmem_ptr + int(c_col)

    # Point to the current buffer
    var storea_smem_ptr = a_tile
    var storeb_smem_ptr = b_tile

    # Load A and B's first shared memory buffer.
    loada[itype, BM, BN, BK, NUM_THREADS, a_type](
        M, N, K, warp_id, lane_id, a_gmem_tile, storea_smem_ptr
    )
    loadb[itype, BM, BN, BK, NUM_THREADS, b_type](
        M, N, K, warp_id, lane_id, b_gmem_tile, storeb_smem_ptr
    )

    async_copy_wait_all()
    barrier()

    # Shifts for switching buffer
    alias a_gmem_shift = BK
    var b_gmem_shift = N * BK
    a_gmem_tile += int(a_gmem_shift)
    b_gmem_tile += int(b_gmem_shift)

    # Alternate share memory buffer for loading.
    storea_smem_ptr += int(a_smem_size)
    storeb_smem_ptr += int(b_smem_size)

    alias num_mma_n = WN // MMA_N
    alias num_mma_m = WM // MMA_M
    alias num_mma = num_mma_m * num_mma_n

    # Double buffer in registers (fragments in nvidia terms).
    # fmt: off
    var a_reg = NDBuffer[a_type, 2, DimList(2, num_mma_m * 4)].stack_allocation()
    var b_reg = NDBuffer[b_type, 2, DimList(2, num_mma_n * 2)].stack_allocation()
    var c_reg = NDBuffer[c_type, 3, DimList(num_mma_m, num_mma_n, 4)].stack_allocation()
    c_reg.zero()
    # fmt: on

    # Load address in shared memory for fma.
    var loada_smem_ptr = a_tile + int(warp_y * WM * MMA_K)
    var loadb_smem_ptr = b_tile + int(warp_x * WN * MMA_K)

    alias align4 = alignof[SIMD[c_type, 4]]()
    alias align2 = alignof[SIMD[c_type, 2]]()

    # Load A fragments to the first buffer.
    @parameter
    for i in range(num_mma_m):
        var vec = SIMD[size=4].load[alignment=align4](
            loada_smem_ptr, i * MMA_M * MMA_K + lane_id * 4
        )
        a_reg.store[width=4, alignment=align4](
            (0, i * 4), SIMD[a_type, 4](vec[0], vec[2], vec[1], vec[3])
        )

    # Load B fragments to the first buffer.
    @parameter
    for i in range(num_mma_n):
        var vec = SIMD[size=2].load[alignment=align2](
            loadb_smem_ptr, i * MMA_K * MMA_N + lane_id * 2
        )
        b_reg.store[width=2, alignment=align2]((0, i * 2), vec)

    var num_k_tiles = Scalar[itype](ceildiv(int(K), int(BK)))

    # Buffer id for the double buffers. They alternate.
    var buffer_id = 0
    var next_buffer_id = buffer_id ^ 0x1

    # Update (num_k_tile - 1) tiles while switching buffers.
    for k_tile_id in range(num_k_tiles - 1):

        @parameter
        for k in range(0, BK, MMA_K):
            var next_k = (k + MMA_K) % int(BK)

            if k == int(BK - MMA_K):
                async_copy_wait_all()
                barrier()

                # fmt: off
                # Switch shared memory buffer.
                loada_smem_ptr = loada_smem_ptr + int(a_smem_size) if (k_tile_id % 2 == 0) \
                    else loada_smem_ptr - int(a_smem_size)
                storea_smem_ptr = storea_smem_ptr - int(a_smem_size) if (k_tile_id % 2 == 0) \
                    else storea_smem_ptr + int(a_smem_size)
                loadb_smem_ptr = loadb_smem_ptr + int(b_smem_size) if (k_tile_id % 2 == 0) \
                    else loadb_smem_ptr - int(b_smem_size)
                storeb_smem_ptr = storeb_smem_ptr - int(b_smem_size) if (k_tile_id % 2 == 0) \
                    else storeb_smem_ptr + int(b_smem_size)
                # fmt: on

                # Advance to the next k tile.
                a_gmem_tile += int(BK)
                b_gmem_tile += int(BK * N)

            # Fill the other A fragments buffer.
            @parameter
            for i in range(num_mma_m):
                var vec = SIMD[size=4].load[alignment=align4](
                    loada_smem_ptr,
                    next_k * BM + i * MMA_M * MMA_K + lane_id * 4,
                )
                a_reg.store[width=4, alignment=align4](
                    (next_buffer_id, i * 4),
                    SIMD[a_type, 4](vec[0], vec[2], vec[1], vec[3]),
                )

            # Fill the other B fragments buffer.
            @parameter
            for i in range(num_mma_n):
                var vec = SIMD[size=2].load[alignment=align2](
                    loadb_smem_ptr,
                    next_k * BN + i * MMA_K * MMA_N + lane_id * 2,
                )
                b_reg.store[width=2, alignment=align2](
                    (next_buffer_id, i * 2), vec
                )

            # Load next k tile from global memory to shared memory.
            # if k < int(num_iters_loada):
            if k == 0:
                loada[itype, BM, BN, BK, NUM_THREADS, a_type](
                    M, N, K, warp_id, lane_id, a_gmem_tile, storea_smem_ptr
                )

                loadb[itype, BM, BN, BK, NUM_THREADS, b_type](
                    M, N, K, warp_id, lane_id, b_gmem_tile, storeb_smem_ptr
                )

            @parameter
            for i in range(num_mma_m):

                @parameter
                for j in range(num_mma_n):
                    var d = c_reg.load[width=4]((i, j, 0))
                    mma(
                        d,
                        a_reg.load[width=4]((buffer_id, i * 4)),
                        b_reg.load[width=2]((buffer_id, j * 2)),
                        d,
                    )
                    c_reg.store[width=4]((i, j, 0), d)

            # Alternate buffer
            buffer_id ^= 0x1
            next_buffer_id ^= 0x1

    # Last k tile.
    @parameter
    for k in range(0, BK, MMA_K):
        var next_k = (k + MMA_K) % int(BK)

        if k < int(BK - MMA_K):

            @parameter
            for i in range(num_mma_m):
                var vec = SIMD[size=4].load[alignment=align4](
                    loada_smem_ptr,
                    next_k * BM + i * MMA_M * MMA_K + lane_id * 4,
                )
                a_reg.store[width=4, alignment=align4](
                    (next_buffer_id, i * 4),
                    SIMD[a_type, 4](vec[0], vec[2], vec[1], vec[3]),
                )

            @parameter
            for i in range(num_mma_n):
                var vec = SIMD[size=2].load[alignment=align2](
                    loadb_smem_ptr,
                    next_k * BN + i * MMA_K * MMA_N + lane_id * 2,
                )
                b_reg.store[width=2, alignment=align2](
                    (next_buffer_id, i * 2), vec
                )

        @parameter
        for i in range(num_mma_m):

            @parameter
            for j in range(num_mma_n):
                var d = c_reg.load[width=4]((i, j, 0))
                mma(
                    d,
                    a_reg.load[width=4]((buffer_id, i * 4)),
                    b_reg.load[width=2]((buffer_id, j * 2)),
                    d,
                )
                c_reg.store[width=4]((i, j, 0), d)

        # Alternate buffer
        buffer_id ^= 0x1
        next_buffer_id ^= 0x1

    # Point to the start of the current warp.
    var c_gmem_ptr = c.data + (c_row + warp_y * WM) * N + (c_col + warp_x * WN)
    # c_gmem_ptr += mma_y * simd_size * N + mma_x * simd_size

    @parameter
    for i in range(num_mma_m):

        @parameter
        for j in range(num_mma_n):
            var c_mma_tile = c_gmem_ptr + (i * MMA_M) * N + j * MMA_N
            SIMD[size=2].store[alignment=align2](
                c_mma_tile,
                int((lane_id // 4) * N + lane_id % 4 * 2),
                c_reg.load[width=2, alignment=align2]((i, j, 0)),
            )
            SIMD[size=2].store[alignment=align2](
                c_mma_tile,
                int((lane_id // 4 + 8) * N + lane_id % 4 * 2),
                c_reg.load[width=2, alignment=align2]((i, j, 2)),
            )


fn test(ctx: Context) raises:
    alias NUM_THREADS = 256
    alias M = 8192
    alias N = 8192
    alias K = 128
    alias BM = 256
    alias BN = 128
    alias BK = 16
    alias WM = 64
    alias WN = 64

    var stream = Stream(ctx)

    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var c_host_ref = DTypePointer[DType.float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i

    for i in range(K * N):
        b_host[i] = i + 1

    var a_device = ctx.malloc[Float32](M * K)
    var b_device = ctx.malloc[Float32](K * N)
    var c_device = ctx.malloc[Float32](M * N)
    var c_device_ref = ctx.malloc[Float32](M * N)

    ctx.copy_host_to_device(a_device, a_host, M * K)
    ctx.copy_host_to_device(b_device, b_host, K * N)

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
        NUM_THREADS,
    ]
    var func = Function[gemm](
        ctx,
        threads_per_block=NUM_THREADS,  # dump_ptx=Path("./mm.ptx")
        cache_config=CacheConfig.PREFER_SHARED,
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
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
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
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(NUM_THREADS, 1, 1),
        stream=stream,
    )

    ctx.synchronize()

    ctx.copy_device_to_host(c_host, c_device, M * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        DType.float32, DType.float32, DType.float32, BLOCK_DIM
    ]
    var func_naive = Function[gemm_naive](ctx, threads_per_block=NUM_THREADS)
    var c_buffer_ref = NDBuffer[DType.float32, 2, DimList(M, N)](c_device_ref)
    func_naive(
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.synchronize()
    ctx.copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=0.01):
            print(i, c_host[i], c_host_ref[i])
            break
        # assert_almost_equal(c_host[i], c_host_ref[i])

    ctx.free(c_device)
    ctx.free(c_device_ref)
    ctx.free(a_device)
    ctx.free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()

    _ = func^
    _ = func_naive^
    _ = stream^


def main():
    try:
        with CudaInstance() as instance:
            with Context(Device(instance)) as ctx:
                test(ctx)
    except e:
        print("CUDA_ERROR:", e)
