# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys.info import simd_width_of
from std.sys import align_of

import linalg.matmul.vendor.blas as vendor_blas
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    thread_idx,
    warp_id as get_warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import async_copy_wait_all
from layout import Layout
from layout.layout_tensor import copy_dram_to_sram_async
from layout.math import outer_product_acc
from layout.tensor_core import TensorCore

from layout import TileTensor, Idx, Coord, TensorLayout, col_major, row_major
from layout.tile_tensor import stack_allocation
from layout.tile_io import copy_dram_to_sram_async as copy_tiles_async

from std.utils.index import Index

comptime NWARMUP = 1
comptime NRUN = 1


def time_kernel[
    func: def(DeviceContext) raises capturing -> None
](mut m: Bench, ctx: DeviceContext, size: Int, kernel_name: String) raises:
    @inline(.always)
    def bench_func(mut m: Bencher) {imm}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            func(ctx)

        bencher_iter_custom(m, kernel_launch, ctx)

    m.bench_function(
        bench_func,
        BenchId(kernel_name),
        [ThroughputMeasure(BenchMetric.elements, 2 * size)],
    )


def run_cublas[
    dtype: DType, enable_tc: Bool = False
](
    mut m: Bench,
    ctx: DeviceContext,
    M: Int,
    N: Int,
    K: Int,
    a: ImmPointer[Scalar[dtype], _],
    b: ImmPointer[Scalar[dtype], _],
    c: MutPointer[Scalar[dtype], _],
) raises:
    var a_device = TileTensor(a, row_major(M, K))
    var b_device = TileTensor(b, row_major(K, N))
    var c_device_ref = TileTensor(c, row_major(M, N))

    with vendor_blas.Handle() as _handle:

        def bench_func(mut m: Bencher) {imm}:
            @inline(.always)
            def kernel_launch(ctx: DeviceContext) raises {imm}:
                vendor_blas.matmul[use_tf32=enable_tc](
                    ctx,
                    c_device_ref,
                    a_device,
                    b_device,
                    c_row_major=True,
                    transpose_b=False,
                )

            bencher_iter_custom(m, kernel_launch, ctx)

        def get_bench_id() -> String:
            comptime if enable_tc:
                return "cublas_tensorcore"
            else:
                return "cublas"

        m.bench_function(
            bench_func,
            BenchId(get_bench_id()),
            [ThroughputMeasure(BenchMetric.elements, 2 * M * N * K)],
        )
        # Do one iteration for verification.
        ctx.enqueue_memset(DeviceBuffer[dtype](ctx, c, M * N, owning=False), 0)
        vendor_blas.matmul(
            ctx,
            c_device_ref,
            a_device,
            b_device,
            c_row_major=True,
            transpose_b=False,
        )


def gemm_kernel_1[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
](
    a: TileTensor[dtype, a_layout, ImmutAnyOrigin],
    b: TileTensor[dtype, b_layout, ImmutAnyOrigin],
    c: TileTensor[dtype, c_layout, MutAnyOrigin],
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
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    # Calculate the column and row indices for each thread.
    var col = thread_idx.y
    var row = thread_idx.x
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C that this thread is
    # responsible for computing.
    var dst = c.tile[BM, BN](Coord(bidy, bidx))

    # Initialize a register to accumulate the result for this thread.
    var dst_reg: type_of(c).ElementType = 0

    # Iterate over the K dimension to compute the dot product.
    for k in range(Int(b.dim[0]())):
        # Get the corresponding tiles from matrices A and B.
        var a_tile = a.tile[BM, 1](Coord(bidy, k))
        var b_tile = b.tile[1, BN](Coord(k, bidx))

        # Multiply the elements and accumulate the result.
        dst_reg += a_tile[row, 0] * b_tile[0, col]

    # Write the final accumulated result to the output matrix.
    dst[row, col] += dst_reg


def run_gemm_kernel_1[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    mut m: Bench,
    ctx: DeviceContext,
    a: TileTensor,
    b: TileTensor,
    c: TileTensor,
) raises:
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(a.dim[1]())

    comptime func = gemm_kernel_1[
        dtype, type_of(a.layout), type_of(b.layout), type_of(c.layout), BM, BN
    ]

    @inline(.always)
    @__parameter
    def run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function[func](
            a.as_imm().as_unsafe_any_origin(),
            b.as_imm().as_unsafe_any_origin(),
            c.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN, BM),
        )

    time_kernel[run_func](m, ctx, M * N * K, "naive")

    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function[func](
        a.as_imm().as_unsafe_any_origin(),
        b.as_imm().as_unsafe_any_origin(),
        c.as_unsafe_any_origin(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BN, BM),
    )


def gemm_kernel_2[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
](
    a: TileTensor[dtype, a_layout, ImmutAnyOrigin],
    b: TileTensor[dtype, b_layout, ImmutAnyOrigin],
    c: TileTensor[dtype, c_layout, MutAnyOrigin],
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
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2

    var col = thread_idx.x
    var row = thread_idx.y
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C
    var dst = c.tile[BM, BN](Coord(bidy, bidx))

    # Initialize the register to accumulate the result
    var dst_reg: type_of(c).ElementType = 0

    # Iterate over the K dimension
    for k in range(Int(b.dim[0]())):
        # Get the tiles of input matrices A and B
        var a_tile = a.tile[BM, 1](Coord(bidy, k))
        var b_tile = b.tile[1, BN](Coord(k, bidx))

        # Compute the partial result and accumulate it in the register
        dst_reg += a_tile[row, 0] * b_tile[0, col]

    # Store the final result back to the output matrix
    dst[row, col] += dst_reg


def run_gemm_kernel_2[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    mut m: Bench,
    ctx: DeviceContext,
    a: TileTensor,
    b: TileTensor,
    c: TileTensor,
) raises:
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(a.dim[1]())

    comptime kernel = gemm_kernel_2[
        dtype, type_of(a.layout), type_of(b.layout), type_of(c.layout), BM, BN
    ]

    @inline(.always)
    @__parameter
    def run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            a.as_imm().as_unsafe_any_origin(),
            b.as_imm().as_unsafe_any_origin(),
            c.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN, BM),
        )

    time_kernel[run_func](m, ctx, M * N * K, "mem_coalesce")

    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function[kernel](
        a.as_imm().as_unsafe_any_origin(),
        b.as_imm().as_unsafe_any_origin(),
        c.as_unsafe_any_origin(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BN, BM),
    )


def gemm_kernel_3[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, ImmutAnyOrigin],
    b: TileTensor[dtype, b_layout, ImmutAnyOrigin],
    c: TileTensor[dtype, c_layout, MutAnyOrigin],
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
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    # Calculate the column and row indices for each thread
    var row, col = udivmod(thread_idx.x, BN)

    # Get the tile of the output matrix C that this thread block is responsible for
    var dst = c.tile[BM, BN](Coord(block_idx.y, block_idx.x))

    # Allocate shared memory for tiles of input matrices A and B
    var a_smem = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=align_of[dtype]()
    ](row_major[BM, BK]())
    var b_smem = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=align_of[dtype]()
    ](row_major[BK, BN]())

    # Initialize the register to accumulate the result
    var dst_reg: type_of(c).ElementType = 0

    # Iterate over tiles of input matrices A and B
    for block in range(Int(b.dim[0]()) // BK):
        # Define the layout for loading tiles of A and B into shared memory
        comptime load_a_layout = row_major[NUM_THREADS // BK, BK]()
        comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()

        # Get the tiles of A and B for the current iteration
        var a_tile = a.tile[BM, BK](Coord(block_idx.y, block))
        var b_tile = b.tile[BK, BN](Coord(block, block_idx.x))

        # Asynchronously copy tiles of A and B from global memory to shared memory
        copy_tiles_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_tiles_async[thread_layout=load_b_layout](b_smem, b_tile)

        # Wait for all asynchronous copies to complete
        async_copy_wait_all()

        # Synchronize threads to ensure shared memory is populated
        barrier()

        # Perform matrix multiplication on the tiles in shared memory
        comptime for k in range(BK):
            dst_reg += a_smem[row, k] * b_smem[k, col]

        # Synchronize threads before loading the next tiles
        barrier()

    # Write the result to the output matrix
    dst[row, col] += dst_reg


def run_gemm_kernel_3[
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
    a: TileTensor,
    b: TileTensor,
    c: TileTensor,
) raises:
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(a.dim[1]())

    comptime kernel = gemm_kernel_3[
        dtype,
        type_of(a.layout),
        type_of(b.layout),
        type_of(c.layout),
        BM,
        BN,
        BK,
        BM * BN,
    ]

    @inline(.always)
    @__parameter
    def run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            a.as_imm().as_unsafe_any_origin(),
            b.as_imm().as_unsafe_any_origin(),
            c.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BM * BN),
        )

    time_kernel[run_func](m, ctx, M * N * K, "shared_mem")

    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function[kernel](
        a.as_imm().as_unsafe_any_origin(),
        b.as_imm().as_unsafe_any_origin(),
        c.as_unsafe_any_origin(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BM * BN),
    )


def gemm_kernel_4[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, ImmutAnyOrigin],
    b: TileTensor[dtype, b_layout, ImmutAnyOrigin],
    c: TileTensor[dtype, c_layout, MutAnyOrigin],
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
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    # Calculate the column and row indices for each thread.
    var row, col = udivmod(thread_idx.x, BN)
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C that this thread is
    # responsible for computing.
    var dst = c.tile[BM, BN](Coord(bidy, bidx)).tile[TM, 1](Coord(row, col))

    # Allocate shared memory for tiles of A and B.
    var a_smem = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=align_of[dtype]()
    ](row_major[BM, BK]())
    var b_smem = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=align_of[dtype]()
    ](row_major[BK, BN]())

    # Allocate a register tile to store the partial results.
    var dst_reg = stack_allocation[
        dtype=dtype, address_space=.LOCAL, alignment=align_of[dtype]()
    ](row_major[TM]())
    dst_reg.copy_from(dst)

    # Iterate over the tiles of A and B in the K dimension.
    for block in range(Int(b.dim[0]()) // BK):
        # Define the layout for loading tiles of A and B into shared
        # memory.
        comptime load_a_layout = row_major[NUM_THREADS // BK, BK]()
        comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()

        # Get the tiles of A and B for the current block.
        var a_tile = a.tile[BM, BK](Coord(block_idx.y, block))
        var b_tile = b.tile[BK, BN](Coord(block, block_idx.x))

        # Load the tiles of A and B into shared memory asynchronously.
        copy_tiles_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_tiles_async[thread_layout=load_b_layout](b_smem, b_tile)

        # Wait for all asynchronous copies to complete.
        async_copy_wait_all()
        barrier()

        # Iterate over the elements in the K dimension within the tiles.
        comptime for k in range(BK):
            # Get the corresponding tiles from shared memory.
            var a_tile = a_smem.tile[TM, 1](Coord(row, k))
            var b_tile = b_smem.tile[1, BN](Coord(k, 0))
            var b_val = b_tile[0, col]

            # Multiply the elements and accumulate the partial results.
            comptime for t in range(TM):
                dst_reg[t] += a_tile[t, 0] * b_val

        # Synchronize all threads before loading the next tiles.
        barrier()

    # Write the final accumulated results to the output matrix.
    dst.copy_from(dst_reg)


def run_gemm_kernel_4[
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
    a: TileTensor,
    b: TileTensor,
    c: TileTensor,
) raises:
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(a.dim[1]())

    comptime NUM_THREADS = (BM * BN) // TM
    comptime kernel = gemm_kernel_4[
        dtype,
        type_of(a.layout),
        type_of(b.layout),
        type_of(c.layout),
        BM,
        BN,
        BK,
        TM,
        NUM_THREADS,
    ]

    @inline(.always)
    @__parameter
    def run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            a.as_imm().as_unsafe_any_origin(),
            b.as_imm().as_unsafe_any_origin(),
            c.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "1d_blocktiling")

    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function[kernel](
        a.as_imm().as_unsafe_any_origin(),
        b.as_imm().as_unsafe_any_origin(),
        c.as_unsafe_any_origin(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )


def gemm_kernel_5[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, ImmutAnyOrigin],
    b: TileTensor[dtype, b_layout, ImmutAnyOrigin],
    c: TileTensor[dtype, c_layout, MutAnyOrigin],
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
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var partition_row, partition_col = udivmod(thread_idx.x, BN // TN)
    var bidx = block_idx.x
    var bidy = block_idx.y

    var dst = c.tile[BM, BN](Coord(bidy, bidx)).tile[TM, TN](
        Coord(partition_row, partition_col)
    )

    var a_smem = stack_allocation[
        dtype=dtype,
        address_space=.SHARED,
        alignment=align_of[SIMD[dtype, simd_width_of[dtype]()]](),
    ](row_major[BM, BK]())
    var b_smem = stack_allocation[
        dtype=dtype,
        address_space=.SHARED,
        alignment=align_of[SIMD[dtype, simd_width_of[dtype]()]](),
    ](row_major[BK, BN]())

    var dst_reg = stack_allocation[
        dtype=dtype,
        address_space=.LOCAL,
        alignment=align_of[SIMD[dtype, simd_width_of[dtype]()]](),
    ](row_major[TM, TN]())
    dst_reg.copy_from(dst)
    var a_reg = stack_allocation[
        dtype=dtype,
        address_space=.LOCAL,
        alignment=align_of[SIMD[dtype, simd_width_of[dtype]()]](),
    ](row_major[TM]())
    var b_reg = stack_allocation[
        dtype=dtype,
        address_space=.LOCAL,
        alignment=align_of[SIMD[dtype, simd_width_of[dtype]()]](),
    ](row_major[TN]())

    var ntiles = Int(b.dim[0]()) // BK

    for block in range(ntiles):
        comptime load_a_layout = row_major[NUM_THREADS // BK, BK]()
        comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()
        var a_tile = a.tile[BM, BK](Coord(block_idx.y, block))
        var b_tile = b.tile[BK, BN](Coord(block, block_idx.x))
        copy_tiles_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_tiles_async[thread_layout=load_b_layout](b_smem, b_tile)

        async_copy_wait_all()
        barrier()

        comptime for k in range(BK):
            var a_tile = a_smem.tile[TM, 1](Coord(partition_row, k))
            var b_tile = b_smem.tile[1, TN](Coord(k, partition_col))
            a_reg.copy_from(a_tile)
            b_reg.copy_from(b_tile)
            outer_product_acc(dst_reg, a_reg, b_reg)
        barrier()

    dst.copy_from(dst_reg)


def run_gemm_kernel_5[
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
    a: TileTensor,
    b: TileTensor,
    c: TileTensor,
) raises:
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(a.dim[1]())

    comptime NUM_THREADS = (BM * BN) // (TM * TN)

    comptime kernel = gemm_kernel_5[
        dtype,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).LayoutType,
        BM,
        BN,
        BK,
        TM,
        TN,
        NUM_THREADS,
    ]

    @inline(.always)
    @__parameter
    def run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            a.as_imm().as_unsafe_any_origin(),
            b.as_imm().as_unsafe_any_origin(),
            c.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "2d_blocktiling")
    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function[kernel](
        a.as_imm().as_unsafe_any_origin(),
        b.as_imm().as_unsafe_any_origin(),
        c.as_unsafe_any_origin(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )


def gemm_kernel_6[
    dtype: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    a: TileTensor[dtype, a_layout, ImmutAnyOrigin],
    b: TileTensor[dtype, b_layout, ImmutAnyOrigin],
    c: TileTensor[dtype, c_layout, MutAnyOrigin],
):
    """
    Accumulates `C += A * B` with vectorized memory access.

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

    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    comptime simd_width = simd_width_of[dtype]()
    comptime vector_alignment = align_of[SIMD[dtype, simd_width]]()
    var partition_row, partition_col = udivmod(thread_idx.x, BN // TN)
    var bidx = block_idx.x
    var bidy = block_idx.y

    # Get the tile of the output matrix C that this thread is responsible
    # for computing.
    var dst = c.tile[BM, BN](Coord(bidy, bidx)).tile[TM, TN](
        Coord(partition_row, partition_col)
    )
    var dst_vec = dst.vectorize[1, simd_width]()

    # Allocate shared memory for tiles of A and B.
    # Use column-major layout for A to get the transpose.
    var a_smem = stack_allocation[
        dtype, address_space=.SHARED, alignment=vector_alignment
    ](col_major[BM, BK]())
    var b_smem = stack_allocation[
        dtype, address_space=.SHARED, alignment=vector_alignment
    ](row_major[BK, BN]())

    # Allocate register tiles to store the partial results and operands.
    var dst_reg = stack_allocation[
        dtype, address_space=.LOCAL, alignment=vector_alignment
    ](row_major[TM, TN]())
    var dst_reg_vec = dst_reg.vectorize[1, simd_width]()
    dst_reg_vec.copy_from(dst_vec)

    var a_reg = stack_allocation[
        dtype, address_space=.LOCAL, alignment=vector_alignment
    ](row_major[TM]())
    var b_reg = stack_allocation[
        dtype, address_space=.LOCAL, alignment=vector_alignment
    ](row_major[TN]())

    var ntiles = Int(b.dim[0]()) // BK

    # Iterate over the tiles of A and B in the K dimension.
    for block in range(ntiles):
        comptime load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        comptime load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](Coord(block_idx.y, block))
        var b_tile = b.tile[BK, BN](Coord(block, block_idx.x))

        # Keep the strided A-vector geometry at the legacy copy boundary;
        # native vectorization represents only contiguous scalar lanes.
        copy_dram_to_sram_async[thread_layout=load_a_layout](
            a_smem.to_layout_tensor().vectorize[simd_width, 1](),
            a_tile.to_layout_tensor().vectorize[simd_width, 1](),
        )
        copy_dram_to_sram_async[thread_layout=load_b_layout](
            b_smem.to_layout_tensor().vectorize[1, simd_width](),
            b_tile.to_layout_tensor().vectorize[1, simd_width](),
        )

        async_copy_wait_all()
        barrier()

        # Iterate over the elements in the K dimension within the tiles.
        comptime for k in range(BK):
            # Load the corresponding tiles from shared memory into registers.
            var a_tile = a_smem.tile[TM, 1](Coord(partition_row, k))
            var b_tile = b_smem.tile[1, TN](Coord(k, partition_col))
            a_reg.copy_from(a_tile)
            b_reg.copy_from(b_tile)

            # Perform outer product and accumulate the partial results.
            outer_product_acc(dst_reg, a_reg, b_reg)

        barrier()

    # Write the final accumulated results to the output matrix.
    dst_vec.copy_from(dst_reg_vec)


def run_gemm_kernel_6[
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
    a: TileTensor,
    b: TileTensor,
    c: TileTensor,
) raises:
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(a.dim[1]())

    comptime NUM_THREADS = (BM * BN) // (TM * TN)
    comptime kernel = gemm_kernel_6[
        dtype,
        type_of(a.layout),
        type_of(b.layout),
        type_of(c.layout),
        BM,
        BN,
        BK,
        TM,
        TN,
        NUM_THREADS,
    ]

    @inline(.always)
    @__parameter
    def run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            a.as_imm().as_unsafe_any_origin(),
            b.as_imm().as_unsafe_any_origin(),
            c.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_THREADS),
        )

    time_kernel[run_func](m, ctx, M * N * K, "vectorized_mem_access")
    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function[kernel](
        a.as_imm().as_unsafe_any_origin(),
        b.as_imm().as_unsafe_any_origin(),
        c.as_unsafe_any_origin(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )


def matmul_kernel_tc[
    dtype: DType,
    layout_a: TensorLayout,
    layout_b: TensorLayout,
    layout_c: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: TileTensor[dtype, layout_a, ImmutAnyOrigin],
    B: TileTensor[dtype, layout_b, ImmutAnyOrigin],
    C: TileTensor[dtype, layout_c, MutAnyOrigin],
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
    comptime assert A.flat_rank == B.flat_rank == C.flat_rank == 2
    comptime K = A.static_shape[1]

    var warp_id = get_warp_id()  # Warp ID within the block

    # Calculate warp tile coordinates within the block
    var warp_y, warp_x = udivmod(warp_id, BN // WN)

    # Get the warp tile of the output matrix C
    var C_warp_tile = C.tile[BM, BN](Coord(block_idx.y, block_idx.x)).tile[
        WM, WN
    ](Coord(warp_y, warp_x))

    # Ensure warp tile dimensions are multiples of instruction shape
    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0
    ), "Warp tile should be an integer multiple of instruction shape"

    # Create tensor core operation object
    var mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    # Allocate shared memory for tiles of A and B
    var A_sram_tile = stack_allocation[
        A.dtype,
        address_space=.SHARED,
        alignment=align_of[SIMD[A.dtype, 4]](),
    ](row_major[BM, BK]())
    var B_sram_tile = stack_allocation[
        B.dtype,
        address_space=.SHARED,
        alignment=align_of[SIMD[B.dtype, 4]](),
    ](row_major[BK, BN]())

    # Allocate register tile for accumulating partial results
    var c_reg = stack_allocation[
        C.dtype,
        address_space=.LOCAL,
        alignment=align_of[SIMD[C.dtype, 4]](),
    ](row_major[WM // MMA_M, (WN * 4) // MMA_N]()).fill(0)

    # Iterate over tiles of A and B in the K dimension
    for k_i in range(K // BK):
        barrier()  # Synchronize before loading new tiles

        # Get the tiles of A and B for the current iteration
        var A_dram_tile = A.tile[BM, BK](Coord(block_idx.y, k_i))
        var B_dram_tile = B.tile[BK, BN](Coord(k_i, block_idx.x))

        # Load tiles of A and B into shared memory asynchronously
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_sram_tile.to_layout_tensor().vectorize[1, 4](),
            A_dram_tile.to_layout_tensor().vectorize[1, 4](),
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            B_sram_tile.to_layout_tensor().vectorize[1, 4](),
            B_dram_tile.to_layout_tensor().vectorize[1, 4](),
        )

        async_copy_wait_all()  # Wait for async copies to complete
        barrier()  # Synchronize after loading tiles

        # Get the warp tiles of A and B from shared memory
        var A_warp_tile = A_sram_tile.tile[WM, BK](Coord(warp_y, 0))
        var B_warp_tile = B_sram_tile.tile[BK, WN](Coord(0, warp_x))

        # Iterate over the elements in the K dimension within the tiles
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    # Get the register tile for the current MMA operation
                    var c_reg_m_n = c_reg.tile[1, 4](Coord(mma_m, mma_n))

                    # Get the MMA tiles of A and B
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](
                        Coord(mma_m, mma_k)
                    )
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](
                        Coord(mma_k, mma_n)
                    )

                    # Keep the existing hardware fragment representation at
                    # the MMA boundary while surrounding views stay native.
                    var a_reg = mma_op.load_a(A_mma_tile.to_layout_tensor())
                    var b_reg = mma_op.load_b(B_mma_tile.to_layout_tensor())

                    # Perform MMA operation and accumulate the result
                    var d_reg_m_n = mma_op.mma_op(
                        a_reg,
                        b_reg,
                        c_reg_m_n.to_layout_tensor(),
                    )

                    # Store the accumulated result back to the register tile
                    # Legacy MMA results promise only scalar stack alignment.
                    var d_native = TileTensor[
                        C.dtype, address_space=d_reg_m_n.address_space
                    ](d_reg_m_n.ptr, row_major[1, 4]())
                    var d_reg = d_native.load[
                        width=4, alignment=align_of[C.dtype]()
                    ]((0, 0))
                    c_reg_m_n.store[alignment=align_of[C.dtype]()](
                        (0, 0), d_reg
                    )

    # Write the final accumulated results to the output matrix
    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](Coord(mma_m, mma_n))
            var c_reg_m_n = c_reg.tile[1, 4](Coord(mma_m, mma_n))
            mma_op.store_d(C_mma_tile, c_reg_m_n)


def run_gemm_kernel_tc[
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
    a: TileTensor,
    b: TileTensor,
    c: TileTensor,
) raises:
    comptime assert a.rank == b.rank == c.rank == 2
    comptime assert a.flat_rank == b.flat_rank == c.flat_rank == 2
    var M = Int(a.dim[0]())
    var N = Int(b.dim[1]())
    var K = Int(a.dim[1]())

    comptime NUM_WARPS = (BM // WM) * (BN // WN)
    comptime kernel = matmul_kernel_tc[
        dtype,
        type_of(a.layout),
        type_of(b.layout),
        type_of(c.layout),
        BM,
        BN,
        BK,
        WM,
        WN,
        MMA_M,
        MMA_N,
        MMA_K,
    ]

    @inline(.always)
    @__parameter
    def run_func(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            a.as_imm().as_unsafe_any_origin(),
            b.as_imm().as_unsafe_any_origin(),
            c.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(NUM_WARPS * WARP_SIZE),
        )

    time_kernel[run_func](m, ctx, M * N * K, "tensor_core")
    # Do one iteration for verification
    ctx.enqueue_memset(
        DeviceBuffer[dtype](
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](c.ptr),
            M * N,
            owning=False,
        ),
        0,
    )
    ctx.enqueue_function[kernel](
        a.as_imm().as_unsafe_any_origin(),
        b.as_imm().as_unsafe_any_origin(),
        c.as_unsafe_any_origin(),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_WARPS * WARP_SIZE),
    )
