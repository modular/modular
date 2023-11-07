# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil
from gpu.host import Context, Dim, Function, Stream, synchronize
import gpu.host.benchmark
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu import BlockIdx, BlockDim, ThreadIdx, barrier, WARP_SIZE
from gpu.memory import AddressSpace
from memory import memset_zero, stack_allocation
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, bitcast
from tensor import Tensor

from utils.index import Index
from utils.list import DimList

alias BLOCK_DIM = 8


@always_inline
fn __nvvm_ldg_f4[type: DType](x: DTypePointer[type]) -> SIMD[type, 4]:
    # Load a register variable from global state space via non-coherent cache.

    alias alignment = Int32(alignof[SIMD[type, 4]]())

    @parameter
    if type == DType.float32:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f32.p0v4f32", SIMD[DType.float32, 4]
            ](x.bitcast[DType.float32](), alignment)
        )
    else:
        constrained[False, "Unhandled DType"]()
        return 0


@always_inline
fn loadFromGmem[
    BM: Int, BN: Int, BK: Int, rowStrideA: Int, rowStrideB: Int
](
    N: Int,
    K: Int,
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    As: DTypePointer[DType.float32, AddressSpace.SHARED],
    Bs: DTypePointer[DType.float32, AddressSpace.SHARED],
    innerRowA: Int,
    innerColA: Int,
    innerRowB: Int,
    innerColB: Int,
):
    for offset in range(0, BM - rowStrideA + 1, rowStrideA):
        # Load 4 elements at a time and store to shared memory.
        let tmp = __nvvm_ldg_f4[DType.float32](
            a_ptr.offset((innerRowA + offset) * K + innerColA * 4)
        )

        @unroll
        for i in range(4):
            As.store((innerColA * 4 + i) * BM + innerRowA + offset, tmp[i])

    for offset in range(0, BK - rowStrideB + 1, rowStrideB):
        # Load 4 elements at a time and store to shared memory.
        let tmp = __nvvm_ldg_f4[DType.float32](
            b_ptr.offset((innerRowB + offset) * N + innerColB * 4)
        )
        Bs.aligned_simd_store[4, 16](
            (innerRowB + offset) * BN + innerColB * 4, tmp
        )


# BM: The threadblock size for M dimension SMEM caching.
# BN: The threadblock size for N dimension SMEM caching.
# BK: The threadblock size for K dimension SMEM caching.
# WM: M dim of continuous tile computed by each warp.
# WN: N dim of continuous tile computed by each warp.
# WMITER: The number of subwarp tiling steps in M dimension.
# WNITER: The number of subwarp tiling steps in N dimension.
# TM: The per-thread tile size for M dimension.
# TN: The per-thread tile size for N dimension.


@__llvm_metadata(`nvvm.maxntid`=[NUM_THREADS])
fn sgemmWarptiling[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WMITER: Int,
    WNITER: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    M: Int,
    N: Int,
    K: Int,
    alpha: Float32,
    beta: Float32,
):
    let cRow = BlockIdx.y()
    let cCol = BlockIdx.x()

    # Placement of the warp in the threadblock tile.
    let warpIdx = ThreadIdx.x() // WARP_SIZE  # the warp this thread is in
    let warpCol = warpIdx % (BN // WN)
    let warpRow = warpIdx // (BN // WN)

    # Size of the warp subtile.
    alias WSUBM = WM // WMITER  # 64/2=32
    alias WSUBN = WN // WNITER  # 32/2=16

    # Placement of the thread in the warp subtile.
    let threadIdxInWarp = ThreadIdx.x() % WARP_SIZE  # [0, 31]
    let threadColInWarp = threadIdxInWarp % (WSUBN // TN)  # i%(16/4)
    let threadRowInWarp = threadIdxInWarp // (WSUBN // TN)  # i/4

    # Allocate space for the current blocktile in SMEM.
    let As = stack_allocation[
        BM * BK, DType.float32, address_space = AddressSpace.SHARED
    ]()
    let Bs = stack_allocation[
        BK * BN, DType.float32, address_space = AddressSpace.SHARED
    ]()

    # Move blocktile to beginning of A's row and B's column.
    var aa_ptr = a_ptr.offset(cRow * BM * K)
    var bb_ptr = b_ptr.offset(cCol * BN)
    # Move C_ptr to warp's output tile
    let cc_ptr = c_ptr.offset(
        (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN
    )

    # Calculate the indices that this thread will load into SMEM.
    # We load 128bit / 32bit = 4 elements per thread at each step.
    let innerRowA = ThreadIdx.x() // (BK // 4)
    let innerColA = ThreadIdx.x() % (BK // 4)
    alias rowStrideA = (NUM_THREADS * 4) // BK
    let innerRowB = ThreadIdx.x() // (BN // 4)
    let innerColB = ThreadIdx.x() % (BN // 4)
    alias rowStrideB = NUM_THREADS // (BN // 4)

    # TODO: We want these to be register-allocated!
    # Allocate thread-local cache for results in register file.
    var threadResults = StaticTuple[WMITER * TM * WNITER * TN, Float32](0.0)

    # We cache into registers on the warptile level.
    var regM = StaticTuple[WMITER * TM, Float32](0.0)

    var regN = StaticTuple[WNITER * TN, Float32](0.0)

    # Outer-most loop over block tiles.
    for bkIdx in range(0, K, BK):
        loadFromGmem[BM, BN, BK, rowStrideA, rowStrideB](
            N,
            K,
            aa_ptr,
            bb_ptr,
            As,
            Bs,
            innerRowA,
            innerColA,
            innerRowB,
            innerColB,
        )
        barrier()

        for dotIdx in range(BK):
            # Populate registers for whole warptile.
            @unroll
            for wSubRowIdx in range(WMITER):

                @unroll
                for i in range(TM // 4):
                    let tmp = As.aligned_simd_load[4, 16](
                        (dotIdx * BM)
                        + warpRow * WM
                        + wSubRowIdx * WSUBM
                        + threadRowInWarp * TM
                        + i * 4
                    )
                    regM[wSubRowIdx * TM + 0 + i * 4] = tmp[0]
                    regM[wSubRowIdx * TM + 1 + i * 4] = tmp[1]
                    regM[wSubRowIdx * TM + 2 + i * 4] = tmp[2]
                    regM[wSubRowIdx * TM + 3 + i * 4] = tmp[3]

            @unroll
            for wSubColIdx in range(WNITER):

                @unroll
                for i in range(TN // 4):
                    let tmp = Bs.aligned_simd_load[4, 16](
                        (dotIdx * BN)
                        + warpCol * WN
                        + wSubColIdx * WSUBN
                        + threadColInWarp * TN
                    )
                    regN[wSubColIdx * TN + 0 + i * 4] = tmp[0]
                    regN[wSubColIdx * TN + 1 + i * 4] = tmp[1]
                    regN[wSubColIdx * TN + 2 + i * 4] = tmp[2]
                    regN[wSubColIdx * TN + 3 + i * 4] = tmp[3]

            # Execute warptile matmul.
            @unroll
            for wSubRowIdx in range(WMITER):

                @unroll
                for wSubColIdx in range(WNITER):
                    # Calculate per-thread results.
                    @unroll
                    for resIdxM in range(TM):

                        @unroll
                        for resIdxN in range(TN):
                            threadResults[
                                (wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                + (wSubColIdx * TN)
                                + resIdxN
                            ] += (
                                regM[wSubRowIdx * TM + resIdxM]
                                * regN[wSubColIdx * TN + resIdxN]
                            )
        aa_ptr = aa_ptr.offset(BK)  # move BK columns to right
        bb_ptr = bb_ptr.offset(BK * N)  # move BK rows down
        barrier()

    # Write out the results.
    @unroll
    for wSubRowIdx in range(WMITER):

        @unroll
        for wSubColIdx in range(WNITER):
            # Move C pointer to current warp subtile.
            let C_interim: DTypePointer[DType.float32] = cc_ptr.offset(
                (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN
            )

            @unroll
            for resIdxM in range(TM):

                @unroll
                for resIdxN in range(0, TN, 4):
                    # Load C vector into registers.
                    var tmp = __nvvm_ldg_f4[DType.float32](
                        C_interim.offset(
                            (threadRowInWarp * TM + resIdxM) * N
                            + threadColInWarp * TN
                            + resIdxN
                        )
                    )
                    # Perform GEMM update in reg.
                    let i = (wSubRowIdx * TM + resIdxM) * (
                        WNITER * TN
                    ) + wSubColIdx * TN + resIdxN

                    @unroll
                    for k in range(4):
                        tmp[k] = alpha * threadResults[i + k] + beta * tmp[k]
                    C_interim.aligned_simd_store[4, 16](
                        (threadRowInWarp * TM + resIdxM) * N
                        + threadColInWarp * TN
                        + resIdxN,
                        tmp,
                    )


fn matmul_naive(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    let x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    let y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= m or y >= n:
        return

    let a = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        a_ptr, Index(m, k)
    )
    let b = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        b_ptr, Index(k, n)
    )
    let c = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        c_ptr, Index(m, n)
    )

    var accum = Float32(0)
    for i in range(k):
        accum = a[x, i] * b[i, y] + accum
    c[Index(x, y)] = accum


# CHECK-LABEL: run_matmul_kernel_10
fn run_matmul_kernel_10() raises:
    print("== run_matmul_kernel_10")

    let M = 4096
    let N = 4096
    let K = 4096

    # GEMM input parameters, C=α*AB+β*C
    let alpha: Float32 = 1.0
    let beta: Float32 = 0.0

    # TODO: Find best for target GPU.
    #       For A100 see below (based on siboehm repo):
    # alias K10_NUM_THREADS = 128
    # alias K10_BN = 128
    # alias K10_BM = 64
    # alias K10_BK = 16
    # alias K10_WN = 64
    # alias K10_WM = 32
    # alias K10_WNITER = 1
    # alias K10_TN = 4
    # alias K10_TM = 4
    # Settings for A6000
    alias K10_NUM_THREADS = 128
    alias K10_BN = 128
    alias K10_BM = 128
    alias K10_BK = 16
    alias K10_WN = 64
    alias K10_WM = 64
    alias K10_WNITER = 4
    alias K10_TN = 4
    alias K10_TM = 8

    alias NUM_WARPS = K10_NUM_THREADS / 32
    alias K10_WMITER = (K10_WM * K10_WN) // (32 * K10_TM * K10_TN * K10_WNITER)

    # Warptile in threadblocktile.
    constrained[(K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0)]()
    constrained[(K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS]()

    # Threads in warpsubtile.
    constrained[
        (K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0
    ]()

    # Warpsubtile in warptile.
    constrained[(K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0)]()

    constrained[
        (K10_NUM_THREADS * 4) % K10_BK == 0,
        (
            "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
            "issues during GMEM->SMEM tiling (loading only parts of the "
            "final row of Bs during each iteraion)"
        ),
    ]()
    constrained[
        (K10_NUM_THREADS * 4) % K10_BN == 0,
        (
            "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
            "issues during GMEM->SMEM tiling (loading only parts of the "
            "final row of As during each iteration)"
        ),
    ]()

    constrained[
        K10_BN % (16 * K10_TN) == 0,
        "BN must be a multiple of 16*TN to avoid quantization effects",
    ]()
    constrained[
        K10_BM % (16 * K10_TM) == 0,
        "BM must be a multiple of 16*TM to avoid quantization effects",
    ]()

    constrained[
        (K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
        "BM*BK must be a multiple of 4*256 to vectorize loads",
    ]()
    constrained[
        (K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
        "BN*BK must be a multiple of 4*256 to vectorize loads",
    ]()

    constrained[
        K10_TM % 4 == 0,
        "TM must be a multiple of 4",
    ]()

    constrained[
        K10_TN % 4 == 0,
        "TN must be a multiple of 4",
    ]()

    let stream = Stream()

    var a_host = Pointer[Float32].alloc(M * K)
    var b_host = Pointer[Float32].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)
    var c_host_naive = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host.store(i, i)

    for i in range(K * N):
        b_host.store(i, i + 1)

    for i in range(M * N):
        c_host.store(i, 0)

    for i in range(M * N):
        c_host_naive.store(i, 0)

    let a_device = _malloc[Float32](M * K)
    let b_device = _malloc[Float32](K * N)
    let c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    let func = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
            Float32,
            Float32,
        ) -> None, sgemmWarptiling[
            BM=K10_BM,
            BN=K10_BN,
            BK=K10_BK,
            WM=K10_WM,
            WN=K10_WN,
            WMITER=K10_WMITER,
            WNITER=K10_WNITER,
            TM=K10_TM,
            TN=K10_TN,
            NUM_THREADS=K10_NUM_THREADS,
        ]
    ](threads_per_block=K10_NUM_THREADS)

    @always_inline
    @parameter
    fn run_func() raises:
        func(
            (div_ceil(N, K10_BN), div_ceil(M, K10_BM)),
            (K10_NUM_THREADS,),
            a_device,
            b_device,
            c_device,
            M,
            N,
            K,
            alpha,
            beta,
            stream=stream,
        )

    var nstime = time_function[run_func]()
    let flops = 2 * M * N * K
    let sectime = nstime / 1000000000
    print("WARP-TILING MATMUL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # Perform naive matmul to compare results & performance.

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    let func_naive = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) -> None, matmul_naive
    ]()

    @always_inline
    @parameter
    fn run_func_naive() raises:
        func_naive(
            (div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            (BLOCK_DIM, BLOCK_DIM),
            a_device,
            b_device,
            c_device,
            M,
            N,
            K,
            stream=stream,
        )

    nstime = time_function[run_func_naive]()
    let sectime2 = nstime / 1000000000
    print("NAIVE MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_naive, c_device, M * N)

    var failed = False
    for i in range(M * N):
        if c_host.load(i) != c_host_naive.load(i):
            failed = True

    # CHECK: Success
    if not failed:
        print("Success 🎉: results match")
        print("Performance warp-tiling vs. naive: ", sectime2 / sectime, "x")
    else:
        print("Failed ❌: results mismatch")

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul_kernel_10()
    except e:
        print("CUDA_ERROR:", e)
