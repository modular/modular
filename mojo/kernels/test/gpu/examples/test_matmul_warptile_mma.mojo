# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, max, min
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.mma import mma
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
from utils._optional import Optional
from Matmul import matmul_kernel_naive
from MatmulUtils import elementwise_lambda_fn_sig_type


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


# MMA dimensions. FP32 = TF32*TF32 + FP32. We use the following (via library):
# llvm.nvvm.mma.m16n8k8.row.col.tf32
alias MMA_M = 16
alias MMA_N = 8
alias MMA_K = 8


# BM: The thread block size for M dimension.
# BN: The thread block size for N dimension.
# BK: The thread block size for K dimension.
# (Each thread block works on a tile of MxN dimension)
# WM: Warp tile size for M dimension.
# WN: Warp tile size for N dimension.
# (Each warp works on a tile of WMxWN dimension)
# WMITER: The number of subwarp tiling steps in M dimension.
# WNITER: The number of subwarp tiling steps in N dimension.
# (Each warp sequentially computes sub-tiles of WMITERxWNITER dimension)
@__llvm_metadata(`nvvm.maxntid`=[int(NUM_THREADS)])
fn sgemm_warp_tiling_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    indexing_integral_dtype: DType,
    BM: Scalar[indexing_integral_dtype],
    BN: Scalar[indexing_integral_dtype],
    BK: Scalar[indexing_integral_dtype],
    WM: Scalar[indexing_integral_dtype],
    WN: Scalar[indexing_integral_dtype],
    WMITER: Scalar[indexing_integral_dtype],
    WNITER: Scalar[indexing_integral_dtype],
    NUM_THREADS: Scalar[indexing_integral_dtype],
    elementwise_lambda_fn: Optional[elementwise_lambda_fn_sig_type] = None,
](
    mat_c: NDBuffer[2, c_shape, c_type],
    mat_a: NDBuffer[2, a_shape, a_type],
    mat_b: NDBuffer[2, b_shape, b_type],
):
    let M: Scalar[indexing_integral_dtype] = mat_c.dim(0)
    let K: Scalar[indexing_integral_dtype] = mat_a.dim(1)
    let N: Scalar[indexing_integral_dtype] = mat_c.dim(1)

    let c_row: Scalar[indexing_integral_dtype] = BlockIdx.y()
    let c_col: Scalar[indexing_integral_dtype] = BlockIdx.x()

    # Warp in which the thread is (in current thread block tile).
    # e.g., for 128 threads thread block, we have 4 warps, so warpIdx takes
    # values [0, 3]
    let warp_idx = Scalar[indexing_integral_dtype](
        ThreadIdx.x()
    ) // WARP_SIZE  # the warp this thread is in

    # warpCol and warpRow indicate the warp tile position within a thread block.
    # Each thread block is divided in warp tiles of shape WMxWN.
    # e.g., for WM = WN = 64, a thread block of BM = BN = 128 is divided in
    # 4 warp tiles laid out in a 2x2 grid (each of 64x64 size).
    # Warp with warpIdx = 0 would be (warpCol, warpRow) = (0, 0).
    let warp_col = warp_idx % (BN // WN)
    let warp_row = warp_idx // (BN // WN)

    # Each warp tile is divided in sub-warp tiles of size MMA_M x MMA_N = 16x8.
    # These are executed sequentially from a (single) warp.

    # ==========================================================================
    # Indexing for MMA load / store:
    # Uses indexing information from:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    # Chapter 9.7.13.4.7. Matrix Fragments for mma.m16n8k8
    # (with the difference noted below for A matrix indices/registers)

    let lane_id = ThreadIdx.x() & 31
    let group_id = lane_id >> 2
    let thread_id_in_group = lane_id % 4

    # Indices (row, col) for registers a[0], a[1], a[2], a[3].
    # Note: this differs from the generic case, because in our case shared
    #       memory for blocks of A matrix is *TRANSPOSED*.
    #       So, row and col indices are transposed here to account for this.
    let row_a0 = thread_id_in_group
    let row_a2 = thread_id_in_group + 4
    let row_a1 = thread_id_in_group
    let row_a3 = thread_id_in_group + 4
    let col_a0 = group_id
    let col_a1 = group_id + 8
    let col_a2 = group_id
    let col_a3 = group_id + 8

    # Indices (row, col) for registers b[0], b[1].
    let row_b0 = thread_id_in_group
    let row_b1 = thread_id_in_group + 4
    let col_b0_b1 = group_id

    # Indices (row, col) for registers d[0], d[1], d[2], d[3].
    # Same indices for registers c[0-3].
    let row_cd0_cd1 = group_id
    let row_cd2_cd3 = group_id + 8
    let col_cd0 = (thread_id_in_group * 2) + (0 & 0x1)
    let col_cd1 = (thread_id_in_group * 2) + (1 & 0x1)
    let col_cd2 = (thread_id_in_group * 2) + (2 & 0x1)
    let col_cd3 = (thread_id_in_group * 2) + (3 & 0x1)

    # ==========================================================================

    # Allocate space for the current block tile in SMEM.
    # Pad the A tile in share memory to avoid bank conflicts.
    # Use 4 to comply with f4 alignment used in accumulation.
    alias sram_bank_padding_size = 4
    alias BM_padded = BM + sram_bank_padding_size

    # Allocate space for the current blocktile in SMEM.
    let a_sram = NDBuffer[
        1,
        DimList(int(BK * BM_padded)),
        a_type,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    let b_sram = NDBuffer[
        1,
        DimList(int(BK * BN)),
        b_type,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Move blocktile to beginning of A's row and B's column.
    var aa_ptr = mat_a._offset(Index(c_row * BM, 0))
    var bb_ptr = mat_b._offset(Index(0, c_col * BN))
    # Move C_ptr to warp's output tile
    let cc_ptr = mat_c._offset(
        Index(c_row * BM + warp_row * WM, c_col * BN + warp_col * WN)
    )

    # Calculate the indices that this thread will load into SMEM.
    # We load 128bit / 32bit = 4 elements per thread at each step.
    let inner_row_a = Scalar[indexing_integral_dtype](ThreadIdx.x()) // (
        BK // 4
    )
    let inner_col_a = Scalar[indexing_integral_dtype](ThreadIdx.x()) % (BK // 4)
    alias row_stride_a = (NUM_THREADS * 4) // BK
    let inner_row_b = Scalar[indexing_integral_dtype](ThreadIdx.x()) // (
        BN // 4
    )
    let inner_co_ib = Scalar[indexing_integral_dtype](ThreadIdx.x()) % (BN // 4)
    alias row_stride_b = NUM_THREADS // (BN // 4)

    let c_reg = stack_allocation[int(WMITER * WNITER), SIMD[c_type, 4]]()
    memset_zero(c_reg, int(WMITER * WNITER))

    # Indicates chunks of size 8 (across A horizontally or across B vertically).
    # This is because MMA_K = 8.
    alias CHUNK_K = BK // MMA_K

    # Outer-most loop over block tiles.
    for bk_idx in range(0, int(K), int(BK)):
        for offset in range(0, int(BM - row_stride_a + 1), int(row_stride_a)):
            # Load 4 elements at a time and store to shared memory.
            let tmp = __nvvm_ldg_f4[a_type](
                aa_ptr.offset(int((inner_row_a + offset) * K + inner_col_a * 4))
            )

            @unroll
            for i in range(4):
                a_sram[
                    int(
                        (inner_col_a * 4 + i) * BM_padded + inner_row_a + offset
                    )
                ] = tmp[i]

        for offset in range(0, int(BK - row_stride_b + 1), int(row_stride_b)):
            # Load 4 elements at a time and store to shared memory.
            let tmp = __nvvm_ldg_f4[b_type](
                bb_ptr.offset(int((inner_row_b + offset) * N + inner_co_ib * 4))
            )
            b_sram.aligned_simd_store[4, 16](
                Index((inner_row_b + offset) * BN + inner_co_ib * 4),
                tmp,
            )

        barrier()

        # For each thread block across the K dimension, we move across CHUNK_K
        # tiles (across A horizontally and across B vertically).
        for dot_idx in range(int(CHUNK_K)):
            # We cache into registers on the warp tile level.
            # a_reg holds the 4 registers needed per A fragment. We have an
            # array that contains such 4 registers for WMITER chunks.
            # Each MMA_M x MMA_N = 16x8 output chunk uses:
            # a MMA_M x MMA_K = 16x8 chunk from A matrix.
            # a MMA_K x MMA_N = 8x8 chunk from B matrix.
            # There are WMITER x WNITER such chunks per warp.
            let a_reg = stack_allocation[int(WMITER), SIMD[a_type, 4]]()
            let b_reg = stack_allocation[int(WNITER), SIMD[b_type, 2]]()

            # Populate registers for the whole warp tile (hence the for loop)
            # for A (loading values from shared memory).
            # Note: indexing here takes into account that a_sram is transposed.
            @unroll
            for w_sub_row_idx in range(int(WMITER)):
                # Indicates the start 16x8 sub-warp tile row in the context of
                # block tile.
                let subwarp_tile_row_A = dot_idx * MMA_K
                # Indicates the start of 16x8 sub-warp tile column in the
                # context of block tile.
                let subwarp_tile_col_A = warp_row * WM + w_sub_row_idx * MMA_M

                let val0 = a_sram[
                    Index(
                        (subwarp_tile_row_A + row_a0) * BM_padded
                        + subwarp_tile_col_A
                        + col_a0
                    )
                ]

                let val1 = a_sram[
                    Index(
                        (subwarp_tile_row_A + row_a1) * BM_padded
                        + subwarp_tile_col_A
                        + col_a1
                    )
                ]

                let val2 = a_sram[
                    Index(
                        (subwarp_tile_row_A + row_a2) * BM_padded
                        + subwarp_tile_col_A
                        + col_a2
                    )
                ]

                let val3 = a_sram[
                    Index(
                        (subwarp_tile_row_A + row_a3) * BM_padded
                        + subwarp_tile_col_A
                        + col_a3
                    )
                ]

                a_reg.offset(w_sub_row_idx).store(
                    SIMD[a_type, 4](val0, val1, val2, val3)
                )

            # Populate registers for the whole warp tile (hence the for loop)
            # for A (loading values from shared memory).
            @unroll
            for w_sub_col_idx in range(int(WNITER)):
                # Indicates the start 16x8 sub-warp tile row in the context of
                # the block tile.
                let subwarp_tile_row_B = dot_idx * MMA_K
                # Indicates the start of 16x8 ub-warp tile column in the context
                # of the block tile.
                let subwarp_tile_col_B = warp_col * WN + w_sub_col_idx * MMA_N

                let val0 = b_sram[
                    Index(
                        (subwarp_tile_row_B + row_b0) * BN
                        + subwarp_tile_col_B
                        + col_b0_b1
                    )
                ]

                let val1 = b_sram[
                    Index(
                        (subwarp_tile_row_B + row_b1) * BN
                        + subwarp_tile_col_B
                        + col_b0_b1
                    )
                ]

                b_reg.offset(w_sub_col_idx).store(SIMD[b_type, 2](val0, val1))

            # Execute matmul at the warp tile level (sequentially loop over all
            # WMITER x WNITER sub-warp tiles of size MMA_M x MMA_N at the
            # output)
            @unroll
            for w_sub_row_idx in range(int(WMITER)):

                @unroll
                for w_sub_col_idx in range(int(WNITER)):
                    # MMA_M*MMA_N*MMA_K mma library function call.
                    mma(
                        c_reg[w_sub_row_idx * WNITER + w_sub_col_idx],
                        a_reg[w_sub_row_idx],
                        b_reg[w_sub_col_idx],
                        c_reg[w_sub_row_idx * WNITER + w_sub_col_idx],
                    )

        # Move BK columns to right.
        aa_ptr = aa_ptr.offset(int(BK))  # move BK columns to right
        # Move BK rows down.
        bb_ptr = bb_ptr.offset(int(BK * N))  # move BK rows down
        barrier()

    # Write out the results.
    @unroll
    for w_sub_row_idx in range(int(WMITER)):

        @unroll
        for w_sub_col_idx in range(int(WNITER)):
            # Move C pointer to current sub-warp tile.
            let C_interim: DTypePointer[c_type] = cc_ptr.offset(
                int((w_sub_row_idx * MMA_M) * N + w_sub_col_idx * MMA_N)
            )

            # Load from c_reg to vec register vec[0-3].
            let vec = c_reg[w_sub_row_idx * WNITER + w_sub_col_idx]

            @parameter
            if elementwise_lambda_fn:
                alias elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda[c_type, 1](
                    Index(row_cd0_cd1 * N, col_cd0), vec[0]
                )
                elementwise_lambda[c_type, 1](
                    Index(row_cd0_cd1 * N, col_cd1), vec[1]
                )
                elementwise_lambda[c_type, 1](
                    Index(row_cd2_cd3 * N, col_cd2), vec[2]
                )
                elementwise_lambda[c_type, 1](
                    Index(row_cd2_cd3 * N, col_cd3), vec[3]
                )
            else:
                # Store result.
                C_interim.store(
                    row_cd0_cd1 * N + col_cd0,
                    vec[0],
                )

                C_interim.store(
                    row_cd0_cd1 * N + col_cd1,
                    vec[1],
                )

                C_interim.store(
                    row_cd2_cd3 * N + col_cd2,
                    vec[2],
                )

                C_interim.store(
                    row_cd2_cd3 * N + col_cd3,
                    vec[3],
                )


# CHECK-LABEL: run_matmul_mma_warptiling
fn run_matmul_mma_warptiling() raises:
    print("== run_matmul_mma_warptiling")

    # Note: Has been tested for correctness for various sizes and M != N != K.
    alias M = 4096
    alias N = 4096
    alias K = 4096

    # Relative different threshold (to account for numerical accuracy loss
    # because of tensor core use with TF32).
    alias REL_DIFF_THRESHOLD = 0.01

    # Used for naive matmul.
    alias BLOCK_DIM = 8

    # TODO: Find best for target GPU.
    #       For A100 see below (based on siboehm repo):
    # Number of threads per block.
    alias K10_NUM_THREADS = 128
    # Block tile dimensions (across M, N, K dimensions).
    alias K10_BN = 128
    alias K10_BM = 128
    alias K10_BK = 16
    # Warp tile dimensions.
    alias K10_WN = 64
    alias K10_WM = 64
    # Sub-warp tile dimensions are MMA_M x MMA_N
    # WMITER rows x WNITER columns per sub-warp tile
    # (a warp sequentially loops over all sub-warp tiles).
    alias K10_WNITER = K10_WN // MMA_N
    alias K10_WMITER = K10_WM // MMA_M
    # Number of 32-thread warps per thread block.
    alias NUM_WARPS = K10_NUM_THREADS // WARP_SIZE

    print("BM =", K10_BM)
    print("BN =", K10_BN)
    print("BK =", K10_BK)
    print("WM =", K10_WM)
    print("WN =", K10_WN)
    print("WMITER =", K10_WMITER)
    print("WNITER =", K10_WNITER)
    print("NUM_WARPS =", NUM_WARPS)

    # Warptile in threadblocktile.
    constrained[(K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0)]()
    constrained[(K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS]()

    # Threads in warpsubtile.
    constrained[
        (K10_WM * K10_WN) % (WARP_SIZE * K10_WMITER * K10_WNITER) == 0
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

    let c_buffer = NDBuffer[2, DimList(M, N), DType.float32](c_device)
    let a_buffer = NDBuffer[2, DimList(M, K), DType.float32](a_device)
    let b_buffer = NDBuffer[2, DimList(K, N), DType.float32](b_device)

    let func = Function[
        fn (
            NDBuffer[2, DimList(M, N), DType.float32],
            NDBuffer[2, DimList(M, K), DType.float32],
            NDBuffer[2, DimList(K, N), DType.float32],
        ) capturing -> None, sgemm_warp_tiling_kernel[
            DType.float32,
            DimList(M, N),
            DType.float32,
            DimList(M, K),
            DType.float32,
            DimList(K, N),
            indexing_integral_dtype = DType.uint32,
            BM=K10_BM,
            BN=K10_BN,
            BK=K10_BK,
            WM=K10_WM,
            WN=K10_WN,
            WMITER=K10_WMITER,
            WNITER=K10_WNITER,
            NUM_THREADS=K10_NUM_THREADS,
        ]
    ](threads_per_block=K10_NUM_THREADS)

    @always_inline
    @parameter
    fn run_func(stream: Stream) raises:
        func(
            stream,
            (div_ceil(N, K10_BN), div_ceil(M, K10_BM)),
            (K10_NUM_THREADS,),
            c_buffer,
            a_buffer,
            b_buffer,
        )

    var nstime = time_function[run_func](stream)
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
        ) capturing -> None, matmul_kernel_naive[
            c_type = DType.float32,
            a_type = DType.float32,
            b_type = DType.float32,
            BLOCK_DIM=BLOCK_DIM,
        ]
    ]()

    @always_inline
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            stream,
            (div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            (BLOCK_DIM, BLOCK_DIM),
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
        )

    nstime = time_function[run_func_naive](stream)
    let sectime2 = nstime / 1000000000
    print("NAIVE MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_naive, c_device, M * N)

    var failed = False
    for i in range(M * N):
        let outVal = c_host.load(i)
        let outRef = c_host_naive.load(i)
        let relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (
            (relDiff > REL_DIFF_THRESHOLD)
            or math.isnan(outVal)
            or math.isnan(outRef)
        ):
            failed = True
            print(i, outVal, outRef)

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
            run_matmul_mma_warptiling()
    except e:
        print("CUDA_ERROR:", e)
