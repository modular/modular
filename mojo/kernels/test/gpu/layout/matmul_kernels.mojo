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
from gpu import ThreadIdx, BlockIdx, BlockDim, barrier
from gpu.host import DeviceContext
import time
from math import ceildiv
from gpu.memory import async_copy_wait_all
from memory.reference import _GPUAddressSpace as AddressSpace
from layout.int_tuple import IntTuple
from utils import StaticIntTuple
from builtin.io import _printf
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from linalg.cublas import cublas_matmul


alias NWARMUP = 1
alias NRUN = 1


fn time_kernel[
    func: fn (DeviceContext) raises capturing -> None
](ctx: DeviceContext) raises -> Int:
    for _ in range(NWARMUP):
        func(ctx)
    var dt = ctx.execution_time[func](NRUN) // NRUN
    return dt


fn run_cublas[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
](
    ctx: DeviceContext,
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
) raises -> Int:
    var M = a_layout.shape[0].value()
    var N = b_layout.shape[1].value()
    var K = b_layout.shape[0].value()

    var handle = UnsafePointer[cublasContext]()
    check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))
    var a_device = NDBuffer[dtype, 2](a, DimList(M, K))
    var b_device = NDBuffer[dtype, 2](b, DimList(K, N))
    var c_device_ref = NDBuffer[dtype, 2](
        c,
        DimList(M, N),
    )

    @__copy_capture(c_device_ref, a_device, b_device)
    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
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

    for _ in range(NWARMUP):
        run_func(ctx)
    var dt = ctx.execution_time[run_func](NRUN) // NRUN
    check_cublas_error(cublasDestroy(handle))
    return dt


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
    for k in range(b.shape[0]()):
        var a_tile = a.tile[BM, 1](bidy, k)
        var b_tile = b.tile[1, BN](k, bidx)
        dst_reg += a_tile[row, k] * b_tile[k, col]
    dst[row, col] = dst_reg


fn run_gemm_kernel_1[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises -> Int:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var func = ctx.compile_function[
        gemm_kernel_1[dtype, a_layout, b_layout, c_layout, BM, BN]
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

    var dt = time_kernel[run_func](ctx)
    _ = func^
    return dt


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
    for k in range(b.shape[0]()):
        var a_tile = a.tile[BM, 1](bidy, k)
        var b_tile = b.tile[1, BN](k, bidx)
        dst_reg += a_tile[row, k] * b_tile[k, col]
    dst[row, col] = dst_reg


fn run_gemm_kernel_2[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises -> Int:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var func = ctx.compile_function[
        gemm_kernel_2[dtype, a_layout, b_layout, c_layout, BM, BN]
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

    var dt = time_kernel[run_func](ctx)
    _ = func^
    return dt


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

    for block in range(b.shape[0]() // BK):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BN)
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
    dst[row, col] = dst_reg


fn run_gemm_kernel_3[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises -> Int:
    var M = a.shape[0]()
    var N = b.shape[1]()
    var func = ctx.compile_function[
        gemm_kernel_3[dtype, a_layout, b_layout, c_layout, BM, BN, BK, BM * BN]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(M, BM), ceildiv(N, BN)),
            block_dim=(BM * BN),
        )

    var dt = time_kernel[run_func](ctx)
    _ = func^
    return dt


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

    var dst_reg = LayoutTensor[dtype, Layout(TM)].stack_allocation().fill(0)

    for block in range(b.shape[0]() // BK):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BN)
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
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises -> Int:
    var M = a.shape[0]()
    var N = b.shape[1]()
    alias NUM_THREADS = (BM * BN) // TM
    var func = ctx.compile_function[
        gemm_kernel_4[
            dtype, a_layout, b_layout, c_layout, BM, BN, BK, TM, NUM_THREADS
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
            grid_dim=(ceildiv(M, BM), ceildiv(N, BN)),
            block_dim=(NUM_THREADS),
        )

    var dt = time_kernel[run_func](ctx)
    _ = func^
    return dt


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
    ].stack_allocation().fill(0)

    var a_reg = LayoutTensor[dtype, Layout(TM)].stack_allocation()
    var b_reg = LayoutTensor[dtype, Layout(TN)].stack_allocation()

    var ntiles = b.shape[0]() // BK

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
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises -> Int:
    var M = a.shape[0]()
    var N = b.shape[1]()
    alias NUM_THREADS = (BM * BN) // (TM * TN)
    var func = ctx.compile_function[
        gemm_kernel_5[
            dtype, a_layout, b_layout, c_layout, BM, BN, BK, TM, TN, NUM_THREADS
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
            grid_dim=(ceildiv(M, BM), ceildiv(N, BN)),
            block_dim=(NUM_THREADS),
        )

    var dt = time_kernel[run_func](ctx)
    _ = func^
    return dt
