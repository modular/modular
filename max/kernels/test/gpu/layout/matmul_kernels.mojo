# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
import time
from collections.string import StaticString
from math import ceildiv
from sys.info import simdwidthof

import linalg.vendor_blas
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from builtin.io import _printf
from gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from gpu import warp_id as get_warp_id
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

    with vendor_blas.Handle() as handle:

        @parameter
        fn bench_func(mut m: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext) raises:
                vendor_blas.matmul[use_tf32=enable_tc](
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
        fn get_bench_id() -> String:
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
        ctx.enqueue_memset(DeviceBuffer[dtype](ctx, c, M * N, owning=False), 0)
        vendor_blas.matmul(
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
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a simple nested loop structure to compute the matrix
    multiplication. Each thread computes a single element of the output matrix
    C by accumulating the dot product of the corresponding row of A and column
    of B.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    # Calculate the column and row indices for each thread.
    var col = thread_idx.y
    var row = thread_idx.x
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C that this thread is
    # responsible for computing.
    var dst = c.tile[BM, BN](bidy, bidx)

    # Initialize a register to accumulate the result for this thread.
    var dst_reg: c.element_type = 0

    # Iterate over the K dimension to compute the dot product.
    for k in range(b.dim[0]()):
        # Get the corresponding tiles from matrices A and B.
        var a_tile = a.tile[BM, 1](bidy, k)
        var b_tile = b.tile[1, BN](k, bidx)

        # Multiply the elements and accumulate the result.
        dst_reg += a_tile[row, 0] * b_tile[0, col]

    # Write the final accumulated result to the output matrix.
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

    var func = ctx.compile_function_unchecked[
        gemm_kernel_1[dtype, a.layout, b.layout, c.layout, BM, BN]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function_unchecked(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN, BM),
        )

    time_kernel[run_func](m, ctx, M * N * K, "naive")

    # Do one iteration for verifciation
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function_unchecked(
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
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    GEMM kernel that performs matrix multiplication C = A * B with
    memory coalescing optimizations.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel optimizes memory access patterns by ensuring that
    threads within a warp access contiguous memory locations. It
    tiles the input matrices A and B and computes the matrix
    multiplication using register tiling.

    Each thread computes a single element of the output matrix C by
    accumulating the partial results in a register. The final result
    is then stored back to the output matrix.
    """

    var col = thread_idx.x
    var row = thread_idx.y
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C
    var dst = c.tile[BM, BN](bidy, bidx)

    # Initialize the register to accumulate the result
    var dst_reg: c.element_type = 0

    # Iterate over the K dimension
    for k in range(b.dim[0]()):
        # Get the tiles of input matrices A and B
        var a_tile = a.tile[BM, 1](bidy, k)
        var b_tile = b.tile[1, BN](k, bidx)

        # Compute the partial result and accumulate it in the register
        dst_reg += a_tile[row, 0] * b_tile[0, col]

    # Store the final result back to the output matrix
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

    var func = ctx.compile_function_unchecked[
        gemm_kernel_2[dtype, a.layout, b.layout, c.layout, BM, BN]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function_unchecked(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN, BM),
        )

    time_kernel[run_func](m, ctx, M * N * K, "mem_coalesce")

    # Do one iteration for verifciation
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function_unchecked(
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
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    shared memory to improve performance.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        NUM_THREADS: The total number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a tiling strategy to compute the matrix multiplication.
    Each thread block computes a BM x BN tile of the output matrix C. The
    input matrices A and B are loaded into shared memory in tiles of size
    BM x BK and BK x BN, respectively.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the
    number of rows in B.
    """
    # Calculate the column and row indices for each thread
    var col = thread_idx.x % BN
    var row = thread_idx.x // BN

    # Get the tile of the output matrix C that this thread block is responsible for
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x)

    # Allocate shared memory for tiles of input matrices A and B
    var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    # Initialize the register to accumulate the result
    var dst_reg: c.element_type = 0

    # Iterate over tiles of input matrices A and B
    for block in range(b.dim[0]() // BK):
        # Define the layout for loading tiles of A and B into shared memory
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)

        # Get the tiles of A and B for the current iteration
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)

        # Asynchronously copy tiles of A and B from global memory to shared memory
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

        # Wait for all asynchronous copies to complete
        async_copy_wait_all()

        # Synchronize threads to ensure shared memory is populated
        barrier()

        # Perform matrix multiplication on the tiles in shared memory
        @parameter
        for k in range(BK):
            dst_reg += a_smem[row, k] * b_smem[k, col]

        # Synchronize threads before loading the next tiles
        barrier()

    # Write the result to the output matrix
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

    var func = ctx.compile_function_unchecked[
        gemm_kernel_3[dtype, a.layout, b.layout, c.layout, BM, BN, BK, BM * BN]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function_unchecked(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BM * BN),
        )

    time_kernel[run_func](m, ctx, M * N * K, "shared_mem")

    # Do one iteration for verifciation
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function_unchecked(
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
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    shared memory.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        TM: The tile size in the M dimension.
        NUM_THREADS: The number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a tiled approach to compute the matrix multiplication. It
    loads tiles of matrices A and B into shared memory, and then each thread
    computes a partial result using the tiles in shared memory. The partial
    results are accumulated in registers and finally stored back to the output
    matrix C.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    # Calculate the column and row indices for each thread.
    var col = thread_idx.x % BN
    var row = thread_idx.x // BN
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C that this thread is
    # responsible for computing.
    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, 1](row, col)

    # Allocate shared memory for tiles of A and B.
    var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    # Allocate a register tile to store the partial results.
    var dst_reg = tb[dtype]().layout[TM]().local().alloc()
    dst_reg.copy_from(dst)

    # Iterate over the tiles of A and B in the K dimension.
    for block in range(b.dim[0]() // BK):
        # Define the layout for loading tiles of A and B into shared
        # memory.
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)

        # Get the tiles of A and B for the current block.
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)

        # Load the tiles of A and B into shared memory asynchronously.
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

        # Wait for all asynchronous copies to complete.
        async_copy_wait_all()
        barrier()

        # Iterate over the elements in the K dimension within the tiles.
        @parameter
        for k in range(BK):
            # Get the corresponding tiles from shared memory.
            var a_tile = a_smem.tile[TM, 1](row, k)
            var b_tile = b_smem.tile[1, BN](k, 0)
            var b_val = b_tile[0, col]

            # Multiply the elements and accumulate the partial results.
            @parameter
            for t in range(TM):
                dst_reg[t] += a_tile[t, 0] * b_val

        # Synchronize all threads before loading the next tiles.
        barrier()

    # Write the final accumulated results to the output matrix.
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
    var func = ctx.compile_function_unchecked[
        gemm_kernel_4[
            dtype, a.layout, b.layout, c.layout, BM, BN, BK, TM, NUM_THREADS
        ]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function_unchecked(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "1d_blocktiling")

    # Do one iteration for verifciation
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function_unchecked(
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
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        TM: The tile size in the M dimension.
        TN: The tile size in the N dimension.
        NUM_THREADS: The total number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a 2D block tiling strategy to compute the matrix
    multiplication. Each thread block computes a BM x BN tile of the output
    matrix C. Within each thread block, threads are further divided into
    TM x TN tiles to enable thread-level parallelism.

    The kernel loads tiles of A and B into shared memory to reduce global
    memory accesses. It then performs the matrix multiplication using
    register-level tiling and accumulates the results in registers.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    var partition_col = thread_idx.x % (BN // TN)
    var partition_row = thread_idx.x // (BN // TN)
    var bidx = block_idx.x
    var bidy = block_idx.y

    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, TN](
        partition_row, partition_col
    )

    var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    var dst_reg = tb[dtype]().row_major[TM, TN]().local().alloc()
    dst_reg.copy_from(dst)
    var a_reg = tb[dtype]().layout[TM]().local().alloc()
    var b_reg = tb[dtype]().layout[TN]().local().alloc()

    var ntiles = b.dim[0]() // BK

    for block in range(ntiles):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)
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

    var func = ctx.compile_function_unchecked[
        gemm_kernel_5[
            dtype, a.layout, b.layout, c.layout, BM, BN, BK, TM, TN, NUM_THREADS
        ]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function_unchecked(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "2d_blocktiling")
    # Do one iteration for verifciation
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function_unchecked(
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
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B with
    vectorized memory access.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        TM: The tile size in the M dimension.
        TN: The tile size in the N dimension.
        NUM_THREADS: The total number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.

    This kernel uses a 2D block tiling strategy to compute the matrix
    multiplication. Each thread block computes a BM x BN tile of the output
    matrix C. Within each thread block, threads are further divided into TM x
    TN tiles to enable thread-level parallelism.

    The kernel loads tiles of A and B into shared memory using vectorized
    memory access to improve memory bandwidth utilization. It then performs the
    matrix multiplication using register-level tiling and accumulates the
    results in registers.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """

    alias simd_width = simdwidthof[dtype]()
    var partition_col = thread_idx.x % (BN // TN)
    var partition_row = thread_idx.x // (BN // TN)
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C that this thread is responsible
    # for computing.
    var dst = c.tile[BM, BN](bidy, bidx).tile[TM, TN](
        partition_row, partition_col
    )
    var dst_vec = dst.vectorize[1, simd_width]()

    # Allocate shared memory for tiles of A and B.
    # Use column-major layout for A to get the transpose.
    var a_smem = tb[dtype]().col_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    # Allocate register tiles to store the partial results and operands.
    var dst_reg = tb[dtype]().row_major[TM, TN]().local().alloc()
    var dst_reg_vec = dst_reg.vectorize[1, simd_width]()
    dst_reg_vec.copy_from(dst_vec)

    var a_reg = tb[dtype]().layout[TM]().local().alloc()
    var b_reg = tb[dtype]().layout[TN]().local().alloc()

    var ntiles = b.dim[0]() // BK

    # Iterate over the tiles of A and B in the K dimension.
    for block in range(ntiles):
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](block_idx.y, block)
        var b_tile = b.tile[BK, BN](block, block_idx.x)

        # Load the tiles of A and B into shared memory using vectorized
        # memory access.
        copy_dram_to_sram_async[thread_layout=load_a_layout](
            a_smem.vectorize[simd_width, 1](), a_tile.vectorize[simd_width, 1]()
        )
        copy_dram_to_sram_async[thread_layout=load_b_layout](
            b_smem.vectorize[1, simd_width](), b_tile.vectorize[1, simd_width]()
        )

        async_copy_wait_all()
        barrier()

        # Iterate over the elements in the K dimension within the tiles.
        @parameter
        for k in range(BK):
            # Load the corresponding tiles from shared memory into registers.
            var a_tile = a_smem.tile[TM, 1](partition_row, k)
            var b_tile = b_smem.tile[1, TN](k, partition_col)
            a_reg.copy_from(a_tile)
            b_reg.copy_from(b_tile)

            # Perform outer product and accumulate the partial results.
            outer_product_acc(dst_reg, a_reg, b_reg)

        barrier()

    # Write the final accumulated results to the output matrix.
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
    var func = ctx.compile_function_unchecked[
        gemm_kernel_6[
            dtype, a.layout, b.layout, c.layout, BM, BN, BK, TM, TN, NUM_THREADS
        ]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function_unchecked(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "vectorized_mem_access")
    # Do one iteration for verifciation
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function_unchecked(
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
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[dtype, layout_a, MutableAnyOrigin],
    B: LayoutTensor[dtype, layout_b, MutableAnyOrigin],
    C: LayoutTensor[dtype, layout_c, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    tensor cores.

    Parameters:
        dtype: The data type of the input and output tensors.
        layout_a: The layout of the input tensor A.
        layout_b: The layout of the input tensor B.
        layout_c: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        WM: The warp tile size in the M dimension.
        WN: The warp tile size in the N dimension.
        MMA_M: Tensor core instruction shape in M dimension.
        MMA_N: Tensor core instruction shape in N dimension.
        MMA_K: Tensor core instruction shape in K dimension.

    Args:
        A: The input tensor A.
        B: The input tensor B.
        C: The output tensor C.

    This kernel uses a tiled approach with tensor cores to compute the matrix
    multiplication. It loads tiles of matrices A and B into shared memory, and
    then each warp computes a partial result using tensor cores. The partial
    results are accumulated in registers and finally stored back to the output
    matrix C.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """
    alias M = C.shape[0]()  # Number of rows in matrix C
    alias N = C.shape[1]()  # Number of columns in matrix C
    alias K = A.shape[1]()  # Number of columns in matrix A

    var warp_id = get_warp_id()  # Warp ID within the block

    # Calculate warp tile coordinates within the block
    warp_y = warp_id // (BN // WN)
    warp_x = warp_id % (BN // WN)

    # Get the warp tile of the output matrix C
    C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    # Create tensor core operation object
    mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    # Allocate shared memory for tiles of A and B
    A_sram_tile = tb[A.dtype]().row_major[BM, BK]().shared().alloc()
    B_sram_tile = tb[B.dtype]().row_major[BK, BN]().shared().alloc()

    # Allocate register tile for accumulating partial results
    c_reg = (
        tb[C.dtype]()
        .row_major[WM // MMA_M, (WN * 4) // MMA_N]()
        .local()
        .alloc()
        .fill(0)
    )

    # Iterate over tiles of A and B in the K dimension
    for k_i in range(K // BK):
        barrier()  # Synchronize before loading new tiles

        # Get the tiles of A and B for the current iteration
        A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles of A and B into shared memory asynchronously
        copy_dram_to_sram_async[thread_layout = Layout.row_major(4, 8)](
            A_sram_tile.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout = Layout.row_major(4, 8)](
            B_sram_tile.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()  # Wait for async copies to complete
        barrier()  # Synchronize after loading tiles

        # Get the warp tiles of A and B from shared memory
        A_warp_tile = A_sram_tile.tile[WM, BK](warp_y, 0)
        B_warp_tile = B_sram_tile.tile[BK, WN](0, warp_x)

        # Iterate over the elements in the K dimension within the tiles
        @parameter
        for mma_k in range(BK // MMA_K):

            @parameter
            for mma_m in range(WM // MMA_M):

                @parameter
                for mma_n in range(WN // MMA_N):
                    # Get the register tile for the current MMA operation
                    c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)

                    # Get the MMA tiles of A and B
                    A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    # Load fragments of A and B into registers
                    a_reg = mma_op.load_a(A_mma_tile)
                    b_reg = mma_op.load_b(B_mma_tile)

                    # Perform MMA operation and accumulate the result
                    var d_reg_m_n = mma_op.mma_op(
                        a_reg,
                        b_reg,
                        c_reg_m_n,
                    )

                    # Store the accumulated result back to the register tile
                    c_reg_m_n.copy_from(d_reg_m_n)

    # Write the final accumulated results to the output matrix
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
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
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
    var func = ctx.compile_function_unchecked[
        matmul_kernel_tc[
            dtype,
            a.layout,
            b.layout,
            c.layout,
            BM,
            BN,
            BK,
            WM,
            WN,
            MMA_M,
            MMA_N,
            MMA_K,
        ]
    ]()

    @always_inline
    @parameter
    fn run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function_unchecked(
            func,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_WARPS * WARP_SIZE),
        )

    time_kernel[run_func](m, ctx, M * N * K, "tensor_core")
    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function_unchecked(
        func,
        a,
        b,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_WARPS * WARP_SIZE),
    )
    _ = func^
