# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from buffer import NDBuffer
from buffer.dimlist import DimList
from layout.math import outer_product_acc
from layout.tensor_core import TensorCore
from gpu import ThreadIdx, BlockIdx, BlockDim, barrier, WARP_SIZE
from gpu.host import DeviceContext, DeviceBuffer
import time
from math import ceildiv
from gpu.memory import async_copy_wait_all
from memory.reference import _GPUAddressSpace as AddressSpace
from layout.int_tuple import IntTuple
from layout.runtime_tuple import RuntimeTuple
from layout.runtime_layout import RuntimeLayout, UNKNOWN_VALUE
from utils import StaticIntTuple
from builtin.io import _printf
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from linalg.cublas import cublas_matmul
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure

from utils.index import Index


alias NWARMUP = 1
alias NRUN = 1


fn time_kernel[
    func: fn (DeviceContext) raises capturing -> None
](inout m: Bench, ctx: DeviceContext, size: Int, kernel_name: String) raises:
    @parameter
    @always_inline
    fn bench_func(inout m: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            func(ctx)

        m.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(kernel_name), ThroughputMeasure(BenchMetric.elements, 2 * size)
    )


fn run_cublas[
    dtype: DType
](
    inout m: Bench,
    ctx: DeviceContext,
    M: Int,
    N: Int,
    K: Int,
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
) raises:
    var a_device = NDBuffer[dtype, 2](a, DimList(M, K))
    var b_device = NDBuffer[dtype, 2](b, DimList(K, N))
    var c_device_ref = NDBuffer[dtype, 2](
        c,
        DimList(M, N),
    )

    var handle = UnsafePointer[cublasContext]()
    check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))

    @parameter
    fn bench_func(inout m: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _ = cublas_matmul(
                handle,
                c_device_ref,
                a_device,
                b_device,
                c_row_major=True,
                transpose_b=False,
            )

        m.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId("cublas"),
        ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
    )
    # Do one iteration for verification.
    ctx.memset(DeviceBuffer[dtype](ctx, c, M * N, owning=False), 0)
    check_cublas_error(
        cublas_matmul(
            handle,
            c_device_ref,
            a_device,
            b_device,
            c_row_major=True,
            transpose_b=False,
        )
    )

    check_cublas_error(cublasDestroy(handle))


fn gemm_kernel_1[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    a: LayoutTensor[dtype, a_layout],
    b: LayoutTensor[dtype, b_layout],
    c: LayoutTensor[dtype, c_layout],
):
    var col = ThreadIdx.y()
    var row = ThreadIdx.x()
    var bidx = BlockIdx.x()
    var bidy = BlockIdx.y()

    var dst = c.tile[BM, BN](bidy, bidx)
    var dst_reg: c.element_type = 0
    for k in range(b.dim(0)):
        var a_tile = a.tile[BM, 1](bidy, k)
        var b_tile = b.tile[1, BN](k, bidx)
        dst_reg += a_tile[row, 0] * b_tile[0, col]
    dst[row, col] += dst_reg


fn run_gemm_kernel_1[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    inout m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var K = a.shape[1]()

    var func = ctx.compile_function[
        gemm_kernel_1[dtype, a.layout, b.layout, c.layout, BM, BN]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN, BM),
        )

    time_kernel[run_func](m, ctx, M * N * K, "naive")

    # Do one iteration for verifciation
    ctx.memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function(
        func,
        a,
        b,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BN, BM),
    )

    _ = func^


fn gemm_kernel_2[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    a: LayoutTensor[dtype, a_layout],
    b: LayoutTensor[dtype, b_layout],
    c: LayoutTensor[dtype, c_layout],
):
    var col = ThreadIdx.x()
    var row = ThreadIdx.y()
    var bidx = BlockIdx.x()
    var bidy = BlockIdx.y()

    var dst = c.tile[BM, BN](bidy, bidx)

    var dst_reg: c.element_type = 0
    for k in range(b.dim(0)):
        var a_tile = a.tile[BM, 1](bidy, k)
        var b_tile = b.tile[1, BN](k, bidx)
        dst_reg += a_tile[row, 0] * b_tile[0, col]
    dst[row, col] += dst_reg


fn run_gemm_kernel_2[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    inout m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var K = a.shape[1]()

    var func = ctx.compile_function[
        gemm_kernel_2[dtype, a.layout, b.layout, c.layout, BM, BN]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN, BM),
        )

    time_kernel[run_func](m, ctx, M * N * K, "mem_coalesce")

    # Do one iteration for verifciation
    ctx.memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function(
        func,
        a,
        b,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BN, BM),
    )

    _ = func^


fn gemm_kernel_3[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    NUM_THREADS: Int,
](
    a: LayoutTensor[dtype, a_layout],
    b: LayoutTensor[dtype, b_layout],
    c: LayoutTensor[dtype, c_layout],
):
    var col = ThreadIdx.x() % BN
    var row = ThreadIdx.x() // BN

    var dst = c.tile[BM, BN](BlockIdx.y(), BlockIdx.x())

    alias smem_layout = Layout.row_major(BM, BN)
    var a_smem = LayoutTensor[
        dtype, smem_layout, address_space = AddressSpace.SHARED
    ].stack_allocation()
    var b_smem = LayoutTensor[
        dtype, smem_layout, address_space = AddressSpace.SHARED
    ].stack_allocation()

    var dst_reg: c.element_type = 0

    for block in range(b.dim(0) // BK):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](BlockIdx.y(), block)
        var b_tile = b.tile[BK, BN](block, BlockIdx.x())
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)
        async_copy_wait_all()
        barrier()

        @parameter
        for k in range(BK):
            dst_reg += a_smem[row, k] * b_smem[k, col]
        barrier()

    dst[row, col] += dst_reg


fn run_gemm_kernel_3[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    inout m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var K = a.shape[1]()

    var func = ctx.compile_function[
        gemm_kernel_3[dtype, a.layout, b.layout, c.layout, BM, BN, BK, BM * BN]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BM * BN),
        )

    time_kernel[run_func](m, ctx, M * N * K, "shared_mem")

    # Do one iteration for verifciation
    ctx.memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function(
        func,
        a,
        b,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BM * BN),
    )

    _ = func^


fn gemm_kernel_4[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    NUM_THREADS: Int,
](
    a: LayoutTensor[dtype, a_layout],
    b: LayoutTensor[dtype, b_layout],
    c: LayoutTensor[dtype, c_layout],
):
    var col = ThreadIdx.x() % BN
    var row = ThreadIdx.x() // BN
    var bidx = BlockIdx.x()
    var bidy = BlockIdx.y()

    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, 1](row, col)

    var a_smem = LayoutTensor[
        dtype, Layout.row_major(BM, BK), address_space = AddressSpace.SHARED
    ].stack_allocation()
    var b_smem = LayoutTensor[
        dtype, Layout.row_major(BK, BN), address_space = AddressSpace.SHARED
    ].stack_allocation()

    var dst_reg = LayoutTensor[dtype, Layout(TM)].stack_allocation()
    dst_reg.copy_from(dst)

    for block in range(b.dim(0) // BK):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](BlockIdx.y(), block)
        var b_tile = b.tile[BK, BN](block, BlockIdx.x())
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)
        async_copy_wait_all()
        barrier()

        @parameter
        for k in range(BK):
            var a_tile = a_smem.tile[TM, 1](row, k)
            var b_tile = b_smem.tile[1, BN](k, 0)
            var b_val = b_tile[0, col]

            @parameter
            for t in range(TM):
                dst_reg[t] += a_tile[t, 0] * b_val
        barrier()

    dst.copy_from(dst_reg)


fn run_gemm_kernel_4[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
](
    inout m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var K = a.shape[1]()

    alias NUM_THREADS = (BM * BN) // TM
    var func = ctx.compile_function[
        gemm_kernel_4[
            dtype, a.layout, b.layout, c.layout, BM, BN, BK, TM, NUM_THREADS
        ]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "1d_blocktiling")

    # Do one iteration for verifciation
    ctx.memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function(
        func,
        a,
        b,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )
    _ = func^


fn gemm_kernel_5[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    a: LayoutTensor[dtype, a_layout],
    b: LayoutTensor[dtype, b_layout],
    c: LayoutTensor[dtype, c_layout],
):
    var partition_col = ThreadIdx.x() % (BN // TN)
    var partition_row = ThreadIdx.x() // (BN // TN)
    var bidx = BlockIdx.x()
    var bidy = BlockIdx.y()

    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, TN](
        partition_row, partition_col
    )

    var a_smem = LayoutTensor[
        dtype, Layout.row_major(BM, BK), address_space = AddressSpace.SHARED
    ].stack_allocation()
    var b_smem = LayoutTensor[
        dtype, Layout.row_major(BK, BN), address_space = AddressSpace.SHARED
    ].stack_allocation()

    var dst_reg = LayoutTensor[
        dtype, Layout.row_major(TM, TN)
    ].stack_allocation()
    dst_reg.copy_from(dst)

    var a_reg = LayoutTensor[dtype, Layout(TM)].stack_allocation()
    var b_reg = LayoutTensor[dtype, Layout(TN)].stack_allocation()

    var ntiles = b.dim(0) // BK

    for block in range(ntiles):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](BlockIdx.y(), block)
        var b_tile = b.tile[BK, BN](block, BlockIdx.x())
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

        async_copy_wait_all()
        barrier()

        @parameter
        for k in range(BK):
            var a_tile = a_smem.tile[TM, 1](partition_row, k)
            var b_tile = b_smem.tile[1, TN](k, partition_col)
            a_reg.copy_from(a_tile)
            b_reg.copy_from(b_tile)
            outer_product_acc(dst_reg, a_reg, b_reg)
        barrier()

    dst.copy_from(dst_reg)


fn run_gemm_kernel_5[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
](
    inout m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var K = a.shape[1]()

    alias NUM_THREADS = (BM * BN) // (TM * TN)
    var func = ctx.compile_function[
        gemm_kernel_5[
            dtype, a.layout, b.layout, c.layout, BM, BN, BK, TM, TN, NUM_THREADS
        ]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "2d_blocktiling")
    # Do one iteration for verifciation
    ctx.memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function(
        func,
        a,
        b,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )
    _ = func^


fn matmul_kernel_tc[
    dtype: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
](
    A: LayoutTensor[dtype, layout_a],
    B: LayoutTensor[dtype, layout_b],
    C: LayoutTensor[dtype, layout_c],
):
    alias M = C.shape[0]()
    alias N = C.shape[1]()
    alias K = A.shape[1]()

    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    var warp_id = ThreadIdx.x() // 32

    warp_y = warp_id // (BN // WN)
    warp_x = warp_id % (BN // WN)

    C_warp_tile = C.tile[BM, BN](BlockIdx.y(), BlockIdx.x()).tile[WM, WN](
        warp_y, warp_x
    )

    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    A_sram_tile = LayoutTensor[
        A.dtype, Layout.row_major(BM, BK), address_space = AddressSpace.SHARED
    ].stack_allocation()

    B_sram_tile = LayoutTensor[
        B.dtype, Layout.row_major(BK, BN), address_space = AddressSpace.SHARED
    ].stack_allocation()

    c_reg = (
        LayoutTensor[C.dtype, Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N)]
        .stack_allocation()
        .vectorize[1, 4]()
    ).fill(0)

    for k_i in range(K // BK):
        barrier()
        A_dram_tile = A.tile[BM, BK](BlockIdx.y(), k_i)
        B_dram_tile = B.tile[BK, BN](k_i, BlockIdx.x())
        copy_dram_to_sram_async[thread_layout = Layout.row_major(4, 8)](
            A_sram_tile.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout = Layout.row_major(4, 8)](
            B_sram_tile.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]()
        )
        async_copy_wait_all()
        barrier()

        A_warp_tile = A_sram_tile.tile[WM, BK](warp_y, 0)
        B_warp_tile = B_sram_tile.tile[BK, WN](0, warp_x)

        @parameter
        for mma_k in range(BK // MMA_K):

            @parameter
            for mma_m in range(WM // MMA_M):

                @parameter
                for mma_n in range(WN // MMA_N):
                    c_reg_m_n = rebind[mma_op.c_reg_type](c_reg[mma_m, mma_n])
                    A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)
                    a_reg = mma_op.load_a(A_mma_tile)
                    b_reg = mma_op.load_b(B_mma_tile)
                    c_reg[mma_m, mma_n] = rebind[c_reg.element_type](
                        mma_op.mma(
                            a_reg,
                            b_reg,
                            c_reg_m_n,
                        )
                    )

    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            mma_op.store_d(
                C_mma_tile, rebind[mma_op.c_reg_type](c_reg[mma_m, mma_n])
            )


fn run_gemm_kernel_tc[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
](
    inout m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var K = a.shape[1]()

    alias NUM_WARPS = (BM // WM) * (BN // WN)
    var func = ctx.compile_function[
        matmul_kernel_tc[
            dtype, a.layout, b.layout, c.layout, BM, BN, BK, WM, WN
        ]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_WARPS * WARP_SIZE),
        )

    time_kernel[run_func](m, ctx, M * N * K, "tensor_core")
    # Do one iteration for verification
    ctx.memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function(
        func,
        a,
        b,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_WARPS * WARP_SIZE),
    )
    _ = func^
