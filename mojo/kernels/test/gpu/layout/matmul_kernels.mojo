# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import time
from math import ceildiv
from sys.info import simdwidthof

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from builtin.io import _printf
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.cublas.cublas import check_cublas_error, cublasContext
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import async_copy_wait_all
from layout.int_tuple import IntTuple
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from layout.math import outer_product_acc
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore
import linalg.gpu_blas
from memory import UnsafePointer
from memory.pointer import _GPUAddressSpace as AddressSpace

from utils import IndexList
from utils.index import Index

alias NWARMUP = 1
alias NRUN = 1


fn time_kernel[
    func: fn (DeviceContext) raises capturing -> None
](mut m: Bench, ctx: DeviceContext, size: Int, kernel_name: String) raises:
    @parameter
    @always_inline
    fn bench_func(mut m: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            func(ctx)

        m.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(kernel_name), ThroughputMeasure(BenchMetric.elements, 2 * size)
    )


fn run_cublas[
    dtype: DType, enable_tc: Bool = False
](
    mut m: Bench,
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

    with gpu_blas.Handle[gpu_blas.Backend.CUBLAS]() as handle:

        @parameter
        fn bench_func(mut m: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext) raises:
                gpu_blas.matmul[use_tf32=enable_tc](
                    ctx,
                    handle,
                    c_device_ref,
                    a_device,
                    b_device,
                    c_row_major=True,
                    transpose_b=False,
                )

            m.iter_custom[kernel_launch](ctx)

        @parameter
        fn get_bench_id() -> StringLiteral:
            @parameter
            if enable_tc:
                return "cublas_tensorcore"
            else:
                return "cublas"

        m.bench_function[bench_func](
            BenchId(get_bench_id()),
            ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
        )
        # Do one iteration for verification.
        ctx.memset(DeviceBuffer[dtype](ctx, c, M * N, owning=False), 0)
        gpu_blas.matmul(
            ctx,
            handle,
            c_device_ref,
            a_device,
            b_device,
            c_row_major=True,
            transpose_b=False,
        )


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
    var col = ThreadIdx.y
    var row = ThreadIdx.x
    var bidx = BlockIdx.x
    var bidy = BlockIdx.y

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
    mut m: Bench,
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
    var col = ThreadIdx.x
    var row = ThreadIdx.y
    var bidx = BlockIdx.x
    var bidy = BlockIdx.y

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
    mut m: Bench,
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
    var col = ThreadIdx.x % BN
    var row = ThreadIdx.x // BN

    var dst = c.tile[BM, BN](BlockIdx.y, BlockIdx.x)

    var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    var dst_reg: c.element_type = 0

    for block in range(b.dim(0) // BK):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](BlockIdx.y, block)
        var b_tile = b.tile[BK, BN](block, BlockIdx.x)
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
    mut m: Bench,
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
    var col = ThreadIdx.x % BN
    var row = ThreadIdx.x // BN
    var bidx = BlockIdx.x
    var bidy = BlockIdx.y

    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, 1](row, col)

    var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    var dst_reg = tb[dtype]().layout[TM]().local().alloc()
    dst_reg.copy_from(dst)

    for block in range(b.dim(0) // BK):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](BlockIdx.y, block)
        var b_tile = b.tile[BK, BN](block, BlockIdx.x)
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
    mut m: Bench,
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
    var partition_col = ThreadIdx.x % (BN // TN)
    var partition_row = ThreadIdx.x // (BN // TN)
    var bidx = BlockIdx.x
    var bidy = BlockIdx.y

    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, TN](
        partition_row, partition_col
    )

    var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    var dst_reg = tb[dtype]().row_major[TM, TN]().local().alloc()
    dst_reg.copy_from(dst)
    var a_reg = tb[dtype]().layout[TM]().local().alloc()
    var b_reg = tb[dtype]().layout[TN]().local().alloc()

    var ntiles = b.dim(0) // BK

    for block in range(ntiles):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](BlockIdx.y, block)
        var b_tile = b.tile[BK, BN](block, BlockIdx.x)
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
    mut m: Bench,
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


fn gemm_kernel_6[
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
    alias simd_width = simdwidthof[dtype]()
    var partition_col = ThreadIdx.x % (BN // TN)
    var partition_row = ThreadIdx.x // (BN // TN)
    var bidx = BlockIdx.x
    var bidy = BlockIdx.y

    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, TN](
        partition_row, partition_col
    )
    var dst_vec = dst.vectorize[1, simd_width]()

    # use column major for the local A storage to get the transpose
    var a_smem = tb[dtype]().col_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    var dst_reg = tb[dtype]().row_major[TM, TN]().local().alloc()
    var dst_reg_vec = dst_reg.vectorize[1, simd_width]()
    dst_reg_vec.copy_from(dst_vec)

    var a_reg = tb[dtype]().layout[TM]().local().alloc()
    var b_reg = tb[dtype]().layout[TN]().local().alloc()

    var ntiles = b.dim(0) // BK

    for block in range(ntiles):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](BlockIdx.y, block)
        var b_tile = b.tile[BK, BN](block, BlockIdx.x)
        copy_dram_to_sram_async[thread_layout=load_a_layout](
            a_smem.vectorize[simd_width, 1](), a_tile.vectorize[simd_width, 1]()
        )
        copy_dram_to_sram_async[thread_layout=load_b_layout](
            b_smem.vectorize[1, simd_width](), b_tile.vectorize[1, simd_width]()
        )

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

    dst_vec.copy_from(dst_reg_vec)


fn run_gemm_kernel_6[
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
    mut m: Bench,
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
        gemm_kernel_6[
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

    time_kernel[run_func](m, ctx, M * N * K, "vectorized_mem_access")
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

    var warp_id = ThreadIdx.x // 32

    warp_y = warp_id // (BN // WN)
    warp_x = warp_id % (BN // WN)

    C_warp_tile = C.tile[BM, BN](BlockIdx.y, BlockIdx.x).tile[WM, WN](
        warp_y, warp_x
    )

    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    A_sram_tile = tb[A.dtype]().row_major[BM, BK]().shared().alloc()
    B_sram_tile = tb[B.dtype]().row_major[BK, BN]().shared().alloc()

    c_reg = (
        tb[C.dtype]()
        .row_major[WM // MMA_M, (WN * 4) // MMA_N]()
        .local()
        .alloc()
        .fill(0)
    )

    for k_i in range(K // BK):
        barrier()
        A_dram_tile = A.tile[BM, BK](BlockIdx.y, k_i)
        B_dram_tile = B.tile[BK, BN](k_i, BlockIdx.x)
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
                    c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)
                    A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)
                    a_reg = mma_op.load_a(A_mma_tile)
                    b_reg = mma_op.load_b(B_mma_tile)
                    var d_reg_m_n = mma_op.mma_op(
                        a_reg,
                        b_reg,
                        c_reg_m_n,
                    )
                    c_reg_m_n.copy_from(d_reg_m_n)

    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)
            mma_op.store_d(C_mma_tile, c_reg_m_n)


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
    mut m: Bench,
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
