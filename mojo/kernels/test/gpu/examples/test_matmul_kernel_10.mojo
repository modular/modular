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
from gpu import BlockIdx, ThreadIdx, barrier, WARP_SIZE
from gpu.memory import AddressSpace
from memory import memset_zero, stack_allocation
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, bitcast
from tensor import Tensor

from utils.index import Index
from utils.list import DimList


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
        # const float4 tmp = reinterpret_cast<const float4 *>(
        #    &a_ptr[(innerRowA + offset) * K + innerColA * 4])[0];
        # let tmp = a_ptr.aligned_simd_load[4, 16]((innerRowA + offset) * K + innerColA * 4)
        let tmp = __nvvm_ldg_f4[DType.float32](
            a_ptr.offset((innerRowA + offset) * K + innerColA * 4)
        )

        @unroll
        for i in range(4):
            As.store((innerColA * 4 + i) * BM + innerRowA + offset, tmp[i])

    for offset in range(0, BK - rowStrideB + 1, rowStrideB):
        # reinterpret_cast<float4 *>(
        #    &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        #    reinterpret_cast<const float4 *>(
        #        &b_ptr[(innerRowB + offset) * N + innerColB * 4])[0];
        # let tmp = b_ptr.aligned_simd_load[4, 16]((innerRowB + offset) * N + innerColB * 4)
        let tmp = __nvvm_ldg_f4[DType.float32](
            b_ptr.offset((innerRowB + offset) * N + innerColB * 4)
        )
        Bs.aligned_simd_store[4, 16](
            (innerRowB + offset) * BN + innerColB * 4, tmp
        )


@always_inline
fn processFromSmem[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WMITER: Int,
    WNITER: Int,
    WSUBM: Int,
    WSUBN: Int,
    TM: Int,
    TN: Int,
](
    regM: Buffer[WMITER * TM, DType.float32],
    regN: Buffer[WNITER * TN, DType.float32],
    threadResults: Buffer[WMITER * TM * WNITER * TN, DType.float32],
    As: DTypePointer[DType.float32, AddressSpace.SHARED],
    Bs: DTypePointer[DType.float32, AddressSpace.SHARED],
    warpRow: Int,
    warpCol: Int,
    threadRowInWarp: Int,
    threadColInWarp: Int,
):
    for dotIdx in range(BK):
        # populate registers for whole warptile
        for wSubRowIdx in range(WMITER):
            for i in range(TM):
                regM[wSubRowIdx * TM + i] = As.load(
                    (dotIdx * BM)
                    + warpRow * WM
                    + wSubRowIdx * WSUBM
                    + threadRowInWarp * TM
                    + i
                )

        for wSubColIdx in range(WNITER):
            for i in range(TN):
                regN[wSubColIdx * TN + i] = Bs.load(
                    (dotIdx * BN)
                    + warpCol * WN
                    + wSubColIdx * WSUBN
                    + threadColInWarp * TN
                    + i
                )

        # execute warptile matmul
        for wSubRowIdx in range(WMITER):
            for wSubColIdx in range(WNITER):
                # calculate per-thread results
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


# BM The threadblock size for M dimension SMEM caching.
# BN The threadblock size for N dimension SMEM caching.
# BK The threadblock size for K dimension SMEM caching.
# WM M dim of continuous tile computed by each warp
# WN N dim of continuous tile computed by each warp
# WMITER The number of subwarp tiling steps in M dimension.
# WNITER The number of subwarp tiling steps in N dimension.
# TM The per-thread tile size for M dimension.
# TN The per-thread tile size for N dimension.


@__llvm_metadata(`nvvm.maxntid`=[NUM_THREADS])
fn sgemmWarptiling[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
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

    # Placement of the warp in the threadblock tile
    let warpIdx = ThreadIdx.x() // WARP_SIZE  # the warp this thread is in
    let warpCol = warpIdx % (BN // WN)
    let warpRow = warpIdx // (BN // WN)

    # size of the warp subtile
    alias WMITER = (WM * WN) // (WARP_SIZE * TM * TN * WNITER)
    alias WSUBM = WM // WMITER  # 64/2=32
    alias WSUBN = WN // WNITER  # 32/2=16

    # Placement of the thread in the warp subtile
    let threadIdxInWarp = ThreadIdx.x() % WARP_SIZE  # [0, 31]
    let threadColInWarp = threadIdxInWarp % (WSUBN // TN)  # i%(16/4)
    let threadRowInWarp = threadIdxInWarp // (WSUBN // TN)  # i/4

    # allocate space for the current blocktile in SMEM
    let As = stack_allocation[
        BM * BK, DType.float32, address_space = AddressSpace.SHARED
    ]()
    let Bs = stack_allocation[
        BK * BN, DType.float32, address_space = AddressSpace.SHARED
    ]()

    # Move blocktile to beginning of A's row and B's column
    var aa_ptr = a_ptr.offset(cRow * BM * K)
    var bb_ptr = b_ptr.offset(cCol * BN)
    # Move C_ptr to warp's output tile
    let cc_ptr = c_ptr.offset(
        (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN
    )

    # calculating the indices that this thread will load into SMEM
    # we'll load 128bit / 32bit = 4 elements per thread at each step
    let innerRowA = ThreadIdx.x() // (BK // 4)
    let innerColA = ThreadIdx.x() % (BK // 4)
    alias rowStrideA = (NUM_THREADS * 4) // BK
    let innerRowB = ThreadIdx.x() // (BN // 4)
    let innerColB = ThreadIdx.x() % (BN // 4)
    alias rowStrideB = NUM_THREADS // (BN // 4)

    # TODO: Are these eventually register-allocated in Mojo?
    # allocate thread-local cache for results in registerfile
    let threadResults = Buffer[
        WMITER * TM * WNITER * TN, DType.float32
    ].stack_allocation()
    threadResults.zero()
    # we cache into registers on the warptile level
    let regM = Buffer[WMITER * TM, DType.float32].stack_allocation()
    regM.zero()
    let regN = Buffer[WNITER * TN, DType.float32].stack_allocation()
    regN.zero()

    # outer-most loop over block tiles
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
        processFromSmem[
            BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN
        ](
            regM,
            regN,
            threadResults,
            As,
            Bs,
            warpRow,
            warpCol,
            threadRowInWarp,
            threadColInWarp,
        )
        aa_ptr = aa_ptr.offset(BK)  # move BK columns to right
        bb_ptr = bb_ptr.offset(BK * N)  # move BK rows down
        barrier()

    # write out the results
    for wSubRowIdx in range(WMITER):
        for wSubColIdx in range(WNITER):
            # move C pointer to current warp subtile
            let C_interim: DTypePointer[DType.float32] = cc_ptr.offset(
                (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN
            )
            for resIdxM in range(TM):
                for resIdxN in range(0, TN, 4):
                    # load C vector into registers
                    # float4 tmp = reinterpret_cast<float4 *>(
                    #    &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                    #                threadColInWarp * TN + resIdxN])[0]
                    # var tmp = C_interim.aligned_simd_load[4, 16]((threadRowInWarp * TM + resIdxM) * N +
                    #                threadColInWarp * TN + resIdxN)
                    var tmp = __nvvm_ldg_f4[DType.float32](
                        C_interim.offset(
                            (threadRowInWarp * TM + resIdxM) * N
                            + threadColInWarp * TN
                            + resIdxN
                        )
                    )
                    # perform GEMM update in reg
                    let i = (wSubRowIdx * TM + resIdxM) * (
                        WNITER * TN
                    ) + wSubColIdx * TN + resIdxN

                    @unroll
                    for i in range(4):
                        tmp[i] = alpha * threadResults[i + i] + beta * tmp[i]

                    # write back
                    # reinterpret_cast<float4 *>(
                    #    &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                    #                threadColInWarp * TN + resIdxN])[0] = tmp
                    C_interim.aligned_simd_store[4, 16](
                        (threadRowInWarp * TM + resIdxM) * N
                        + threadColInWarp * TN
                        + resIdxN,
                        tmp,
                    )


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
    # TODO: Add static asserts from siboehm code.
    alias K10_WMITER = (K10_WM * K10_WN) // (32 * K10_TM * K10_TN * K10_WNITER)

    # warptile in threadblocktile
    constrained[(K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0)]()
    constrained[(K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS]()

    # threads in warpsubtile
    constrained[
        (K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0
    ]()

    # warpsubtile in warptile
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

    let stream = Stream()

    var a_host = Pointer[Float32].alloc(M * K)
    var b_host = Pointer[Float32].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host.store(i, 0.1)

    for i in range(K * N):
        b_host.store(i, 1)

    for i in range(M * N):
        c_host.store(i, 0)

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
            WNITER=K10_WNITER,
            TM=K10_TM,
            TN=K10_TN,
            NUM_THREADS=K10_NUM_THREADS,
        ]
    ]()
    # verbose=True, dump_ptx=True

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

    let nstime = time_function[run_func]()
    let flops = 2 * M * N * K
    let sectime = nstime / 1000000000
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")

    _copy_device_to_host(c_host, c_device, M * N)

    var failed = False
    # For 128x128x128 : 12.800012588500977
    # For 4096x4096x4096: 409.61578369140625
    for i in range(M * N):
        if c_host.load(i) != Float32(409.61578369140625):
            failed = True

    # CHECK: success
    if not failed:
        print("success")

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul_kernel_10()
    except e:
        print("CUDA_ERROR:", e)
