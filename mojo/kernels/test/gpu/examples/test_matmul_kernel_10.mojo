# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO (#33518): -t flag is required right now because the kernel assumes C is zeroed
# RUN: %mojo-no-debug-no-assert %s -t | FileCheck %s

from collections import OptionalReg
from math import ceildiv
from sys import has_amd_gpu_accelerator, llvm_intrinsic
from sys.info import alignof

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    thread_idx,
)
from gpu.host import DeviceContext
from gpu.intrinsics import ldg
from gpu.memory import AddressSpace
from linalg.utils import elementwise_epilogue_type
from memory import UnsafePointer, bitcast, memset_zero, stack_allocation

from utils import StaticTuple
from utils.index import Index
from utils.numerics import isnan

alias BLOCK_DIM = 8


# BM: The threadblock size for M dimension SMEM caching.
# BN: The threadblock size for N dimension SMEM caching.
# BK: The threadblock size for K dimension SMEM caching.
# WM: M dim of continuous tile computed by each warp.
# WN: N dim of continuous tile computed by each warp.
# WMITER: The number of subwarp tiling steps in M dimension.
# WNITER: The number of subwarp tiling steps in N dimension.
# TM: The per-thread tile size for M dimension.
# TN: The per-thread tile size for N dimension.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS)
)
fn sgemm_warp_tiling_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
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
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    mat_c: NDBuffer[c_type, 2, c_shape],
    mat_a: NDBuffer[a_type, 2, a_shape],
    mat_b: NDBuffer[b_type, 2, b_shape],
    alpha: Scalar[c_type],
    beta: Scalar[c_type],
):
    var K = mat_a.dim[1]()
    var N = mat_c.dim[1]()

    var c_row = block_idx.y
    var c_col = block_idx.x

    # Placement of the warp in the threadblock tile.
    var warp_idx = thread_idx.x // WARP_SIZE  # the warp this thread is in
    var warp_col = warp_idx % (BN // WN)
    var warp_row = warp_idx // (BN // WN)

    # Size of the warp subtile.
    alias w_sub_m = WM // WMITER  # 64/2=32
    alias w_sub_n = WN // WNITER  # 32/2=16

    # Placement of the thread in the warp subtile.
    var thread_Idx_In_warp = thread_idx.x % WARP_SIZE  # [0, 31]
    var thread_col_in_warp = thread_Idx_In_warp % (w_sub_n // TN)  # i%(16/4)
    var thread_row_in_warp = thread_Idx_In_warp // (w_sub_n // TN)  # i/4

    # Allocate space for the current blocktile in SMEM.
    # Pad the A tile in share memory to avoid bank conflicts.
    # Use 4 to comply with f4 alignment used in accumulation.
    alias sram_bank_padding_size = 4
    alias BM_padded = BM + sram_bank_padding_size
    var a_sram = NDBuffer[
        a_type,
        1,
        DimList(Int(BK * BM_padded)),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var b_sram = NDBuffer[
        b_type,
        1,
        DimList(Int(BK * BN)),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Move blocktile to beginning of A's row and B's column.
    var aa_ptr = mat_a._offset(Index(c_row * BM, 0))
    var bb_ptr = mat_b._offset(Index(0, c_col * BN))
    # Move C_ptr to warp's output tile
    var M_offset_warp = c_row * BM + warp_row * WM
    var N_offset_warp = c_col * BN + warp_col * WN
    var cc_ptr = mat_c._offset(Index(M_offset_warp, N_offset_warp))

    # Calculate the indices that this thread will load into SMEM.
    # We load 128bit / 32bit = 4 elements per thread at each step.
    var inner_row_a = thread_idx.x // (BK // 4)
    var inner_col_a = thread_idx.x % (BK // 4)
    alias row_stride_a = (NUM_THREADS * 4) // BK
    var inner_row_b = thread_idx.x // (BN // 4)
    var inner_co_ib = thread_idx.x % (BN // 4)
    alias row_stride_b = NUM_THREADS // (BN // 4)

    # TODO: We want these to be register-allocated!
    # Allocate thread-local cache for results in register file.
    var thread_results = NDBuffer[
        c_type,
        4,
        DimList(Int(WMITER), Int(WNITER), Int(TM), Int(TN)),
    ]().stack_allocation()
    thread_results.zero()

    # We cache into registers on the warptile level.
    var reg_m = NDBuffer[
        a_type, 2, DimList(Int(WMITER), Int(TM))
    ]().stack_allocation()
    reg_m.zero()

    var reg_n = NDBuffer[
        b_type, 2, DimList(Int(WNITER), Int(TN))
    ]().stack_allocation()
    reg_n.zero()

    # Outer-most loop over block tiles.
    for _ in range(0, K, BK):
        for offset in range(0, Int(BM - row_stride_a + 1), Int(row_stride_a)):
            # Load 4 elements at a time and store to shared memory.
            var tmp = ldg[width=4](
                aa_ptr.offset(Int((inner_row_a + offset) * K + inner_col_a * 4))
            )

            @parameter
            for i in range(4):
                a_sram[
                    Int(
                        (inner_col_a * 4 + i) * BM_padded + inner_row_a + offset
                    )
                ] = tmp[i]

        for offset in range(0, Int(BK - row_stride_b + 1), Int(row_stride_b)):
            # Load 4 elements at a time and store to shared memory.
            var tmp = ldg[width=4](
                bb_ptr.offset(Int((inner_row_b + offset) * N + inner_co_ib * 4))
            )
            b_sram.store[alignment=16](
                Index((inner_row_b + offset) * BN + inner_co_ib * 4),
                tmp,
            )

        barrier()

        for dot_idx in range(BK):
            # Populate registers for whole warptile.
            @parameter
            for w_sub_row_idx in range(WMITER):

                @parameter
                for i in range(0, Int(TM), 4):
                    var vec = a_sram.load[width=4, alignment=16](
                        Int(
                            (dot_idx * BM_padded)
                            + warp_row * WM
                            + w_sub_row_idx * w_sub_m
                            + thread_row_in_warp * TM
                            + i
                        )
                    )
                    reg_m.store(Index(w_sub_row_idx, i), vec)

            @parameter
            for w_sub_col_idx in range(WNITER):

                @parameter
                for i in range(0, Int(TN), 4):
                    var vec = b_sram.load[width=4, alignment=16](
                        Int(
                            (dot_idx * BN)
                            + warp_col * WN
                            + w_sub_col_idx * w_sub_n
                            + thread_col_in_warp * TN
                        )
                    )
                    reg_n.store(Index(w_sub_col_idx, i), vec)

            # Execute warptile matmul.
            @parameter
            for w_sub_row_idx in range(WMITER):

                @parameter
                for w_sub_col_idx in range(WNITER):
                    # Calculate per-thread results.
                    @parameter
                    for res_idx_m in range(TM):

                        @parameter
                        for res_idx_n in range(TN):
                            thread_results[
                                Index(
                                    w_sub_row_idx,
                                    w_sub_col_idx,
                                    res_idx_m,
                                    res_idx_n,
                                )
                            ] += (
                                reg_m[w_sub_row_idx, res_idx_m].cast[c_type]()
                                * reg_n[w_sub_col_idx, res_idx_n].cast[c_type]()
                            )
        aa_ptr = aa_ptr.offset(Int(BK))  # move BK columns to right
        bb_ptr = bb_ptr.offset(Int(BK * N))  # move BK rows down
        barrier()

    # Write out the results.
    @parameter
    for w_sub_row_idx in range(WMITER):

        @parameter
        for w_sub_col_idx in range(WNITER):
            # Move C pointer to current warp subtile.
            var M_offset_subtile = w_sub_row_idx * w_sub_m
            var N_offset_subtile = w_sub_col_idx * w_sub_n
            var C_interim = cc_ptr.offset(
                Int((M_offset_subtile) * N + N_offset_subtile)
            )

            @parameter
            for res_idx_m in range(TM):

                @parameter
                for res_idx_n in range(0, Int(TN), 4):
                    var M_offset_val = thread_row_in_warp * TM + res_idx_m
                    var N_offset_val = thread_col_in_warp * TN + res_idx_n
                    var c_idx = M_offset_val * N + N_offset_val
                    var result_vec = thread_results.load[width=4](
                        Index(
                            w_sub_row_idx,
                            w_sub_col_idx,
                            res_idx_m,
                            res_idx_n,
                        )
                    )

                    var vec = alpha * result_vec + beta * C_interim.load[
                        width=4, alignment=16
                    ](Int(c_idx))

                    @parameter
                    if elementwise_lambda_fn:
                        alias elementwise_lambda = elementwise_lambda_fn.value()
                        elementwise_lambda[c_type, 4](
                            Index(
                                M_offset_warp + M_offset_subtile + M_offset_val,
                                N_offset_warp + N_offset_subtile + N_offset_val,
                            ),
                            vec,
                        )
                    else:
                        C_interim.store[alignment=16](Int(c_idx), vec)


fn matmul_naive(
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    c_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    var x: UInt = global_idx.x
    var y: UInt = global_idx.y

    if x >= m or y >= n:
        return

    var a = NDBuffer[DType.float32, 2](a_ptr, Index(m, k))
    var b = NDBuffer[DType.float32, 2](b_ptr, Index(k, n))
    var c = NDBuffer[DType.float32, 2](c_ptr, Index(m, n))

    var accum = Float32(0)
    for i in range(k):
        accum = a[x, i] * b[i, y] + accum
    c[Index(x, y)] = accum


# CHECK-LABEL: run_matmul_kernel_10
fn bench_matmuls(mut m: Bench, ctx: DeviceContext) raises:
    print("== run_matmul_kernel_10")

    alias M = 4096
    alias N = 4096
    alias K = 4096

    # TODO: Find best for target GPU.
    #       For A100 see below (based on siboehm repo).
    #       For MI300X we need to further autotune (below is a working version).
    # alias K10_NUM_THREADS = 256 if has_amd_gpu_accelerator() else 128
    # alias K10_BN = 128
    # alias K10_BM = 64
    # alias K10_BK = 16
    # alias K10_WN = 64
    # alias K10_WM = 32
    # alias K10_WNITER = 1
    # alias K10_TN = 4
    # alias K10_TM = 4
    # Settings for A6000
    alias K10_NUM_THREADS = 256 if has_amd_gpu_accelerator() else 128
    alias K10_BN = 128
    alias K10_BM = 256 if has_amd_gpu_accelerator() else 128
    alias K10_BK = 16
    alias K10_WN = 64
    alias K10_WM = 128 if has_amd_gpu_accelerator() else 64
    alias K10_WNITER = 4
    alias K10_TN = 4
    alias K10_TM = 8

    alias NUM_WARPS = K10_NUM_THREADS / WARP_SIZE
    alias K10_WMITER = (K10_WM * K10_WN) // (
        WARP_SIZE * K10_TM * K10_TN * K10_WNITER
    )

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

    var a_host = UnsafePointer[Float32].alloc(M * K)
    var b_host = UnsafePointer[Float32].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var c_host_naive = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i

    for i in range(K * N):
        b_host[i] = i + 1

    for i in range(M * N):
        c_host[i] = 0

    for i in range(M * N):
        c_host_naive[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)

    var c_buffer = NDBuffer[DType.float32, 2, DimList(M, N)](
        c_device.unsafe_ptr()
    )
    var a_buffer = NDBuffer[DType.float32, 2, DimList(M, K)](
        a_device.unsafe_ptr()
    )
    var b_buffer = NDBuffer[DType.float32, 2, DimList(K, N)](
        b_device.unsafe_ptr()
    )

    alias sgemm_type = sgemm_warp_tiling_kernel[
        DType.float32,
        DimList(M, N),
        DType.float32,
        DimList(M, K),
        DType.float32,
        DimList(K, N),
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

    @parameter
    @always_inline
    fn bench_matmul_10(mut b: Bencher):
        @parameter
        @always_inline
        fn run_func(ctx: DeviceContext) raises:
            ctx.enqueue_function[sgemm_type](
                c_buffer,
                a_buffer,
                b_buffer,
                Float32(1),
                Float32(0),
                grid_dim=(ceildiv(N, K10_BN), ceildiv(M, K10_BM)),
                block_dim=(K10_NUM_THREADS,),
            )

        b.iter_custom[run_func](ctx)

    m.bench_function[bench_matmul_10](
        BenchId("matmul_sgemm_10"),
        ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
    )
    _ = a_buffer
    _ = b_buffer
    _ = c_buffer

    ctx.enqueue_copy_from_device(c_host, c_device)

    # Perform naive matmul to compare results & performance.

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)

    @parameter
    @always_inline
    fn bench_naive(mut b: Bencher):
        @parameter
        @always_inline
        fn run_func_naive(ctx: DeviceContext) raises:
            ctx.enqueue_function[matmul_naive](
                a_device,
                b_device,
                c_device,
                M,
                N,
                K,
                grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )

        b.iter_custom[run_func_naive](ctx)

    m.bench_function[bench_naive](
        BenchId("matmul_naive"),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
    )

    ctx.enqueue_copy_from_device(c_host_naive, c_device)
    ctx.synchronize()

    for i in range(M * N):
        if (
            c_host[i] != c_host_naive[i]
            or isnan(c_host_naive[i])
            or isnan(c_host[i])
        ):
            print(c_host[i])
            print(c_host_naive[i])
            raise "Failed ❌: results mismatch"

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive


def main():
    with DeviceContext() as ctx:
        var m = Bench()
        bench_matmuls(m, ctx)
        m.dump_report()
