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

# DOC: max/tutorials/custom-ops-matmul.mdx

import compiler
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    thread_idx,
    warp_id,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from layout.math import outer_product_acc
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore
from math import ceildiv
from runtime.asyncrt import DeviceContextPtr
from sys.info import (
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    simdwidthof,
)
from tensor_internal import (
    InputTensor,
    ManagedTensorSlice,
    OutputTensor,
)
from utils import StaticTuple
from utils.index import Index

# The number of threads per block to use for the optimized kernels.
# Used only in llvm_metadata for MAX_THREADS_PER_BLOCK_METADATA.
# Not the most performant for all kernels, used sparingly on nvidia accelerators.
alias OPTIMIZED_NUM_THREADS = 256 if has_amd_gpu_accelerator() else 1024

# The block size to use for the optimized kernels.
alias OPTIMIZED_BLOCK_SIZE = 16 if has_amd_gpu_accelerator() else 32

# ===-----------------------------------------------------------------------===#
# Naive matrix multiplication (CPU)
# ===-----------------------------------------------------------------------===#


fn naive_matrix_multiplication_cpu(
    output: ManagedTensorSlice,
    a: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
    b: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
):
    """A naive matrix multiplication used as a fallback on CPU hardware."""
    var M = a.shape()[0]
    var N = b.shape()[1]
    var K = b.shape()[0]

    for row in range(M):
        for col in range(N):
            for k in range(K):
                output[row, col] = output[row, col] + a[row, k] * b[k, col]


# ===-----------------------------------------------------------------------===#
# Naive matrix multiplication (GPU)
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn naive_matrix_multiplication[
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
    GEMM kernel that performs matrix multiplication C = A * B.

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

    This kernel uses a simple for loop structure to compute the matrix
    multiplication. Each thread computes a single element of the output matrix
    C by accumulating the dot product of the corresponding row of A and column
    of B.

    The kernel assumes that the input matrices A and B are compatible for
    matrix multiplication, i.e., the number of columns in A equals the number
    of rows in B.
    """

    var M = a.dim[0]()
    var N = b.dim[1]()
    var K = b.dim[0]()

    # Calculate the column and row indices for each thread.
    var row = block_dim.x * block_idx.x + thread_idx.x
    var col = block_dim.y * block_idx.y + thread_idx.y

    # Initialize a register to accumulate the result for this thread.
    var dst_reg: c.element_type = 0

    # Iterate over the K dimension to compute the dot product.
    if row < M and col < N:
        for k_index in range(K):
            # Multiply the elements and accumulate the result.
            dst_reg = dst_reg + a[row, k_index] * b[k_index, col]

    # Write the final accumulated result to the output matrix.
    c[row, col] = dst_reg


# ===-----------------------------------------------------------------------===#
# Matrix multiplication with global memory coalescing
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](OPTIMIZED_NUM_THREADS)
)
fn coalescing_matrix_multiplication[
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
    threads within a warp access contiguous memory locations.

    Each thread computes a single element of the output matrix C by
    accumulating the partial results in a register. The final result
    is then stored back to the output matrix.
    """

    var M = a.dim[0]()
    var N = b.dim[1]()
    var K = b.dim[0]()

    # Calculate the column and row indices for each thread.
    # Have adjacent threads work on the same row to allow for memory coalescing
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    # Initialize a register to accumulate the result for this thread.
    var dst_reg: c.element_type = 0

    # Iterate over the K dimension to compute the dot product.
    if row < M and col < N:
        for k_index in range(K):
            # Multiply the elements and accumulate the result.
            dst_reg = dst_reg + a[row, k_index] * b[k_index, col]

    # Write the final accumulated result to the output matrix.
    c[row, col] = dst_reg


# ===-----------------------------------------------------------------------===#
# Matrix multiplication with shared memory tiling
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](OPTIMIZED_NUM_THREADS)
)
fn tiled_matrix_multiplication[
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


# ===-----------------------------------------------------------------------===#
# Matrix multiplication with shared memory tiling and register tiling
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn tiled_register_matrix_multiplication[
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
    shared memory and register tiling.

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

    # Get the tile of the output matrix C that this thread is
    # responsible for computing.
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, 1](row, col)

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


# ===-----------------------------------------------------------------------===#
# Matrix multiplication with block tiling
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn block_tiled_matrix_multiplication[
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

    var dst = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, TN](
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


# ===-----------------------------------------------------------------------===#
# Matrix multiplication with vectorized memory access
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn block_tiled_vectorized_matrix_multiplication[
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

    # Get the tile of the output matrix C that this thread is responsible
    # for computing.
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, TN](
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


# ===-----------------------------------------------------------------------===#
# Matrix multiplication using Tensor Cores
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn tensor_core_matrix_multiplication[
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

    # Calculate warp tile coordinates within the block
    warp_y = warp_id() // (BN // WN)
    warp_x = warp_id() % (BN // WN)

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


# ===-----------------------------------------------------------------------===#
# The matrix multiplication graph operation
# ===-----------------------------------------------------------------------===#


@compiler.register("matrix_multiplication")
struct MatrixMultiplication[algorithm: StaticString]:
    """
    The central custom operation that dispatches to multiple different
    matrix multiplication implementations, depending on target hardware and
    selected algorithm.
    """

    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        output: OutputTensor[rank=2],
        a: InputTensor[dtype = output.dtype, rank = output.rank],
        b: InputTensor[dtype = output.dtype, rank = output.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "gpu":
            a_layout = a.to_layout_tensor()
            b_layout = b.to_layout_tensor()
            out_layout = output.to_layout_tensor()

            M = a_layout.shape[0]()
            N = b_layout.shape[1]()

            gpu_ctx = ctx.get_device_context()

            # Zero out the memory in the outbound tensor.
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output.dtype]]](out_layout.ptr),
                    M * N,
                    owning=False,
                ),
                0,
            )

            # We support several compile-time variants for the matrix
            # multiplication calculation:
            # - "naive": A naive matrix multiplication using LayoutTensors.
            # - "coalescing": Matrix multiplication with memory coalescing
            #   optimizations.
            # - "tiled": Matrix multiplication using a tiling strategy.
            # - "tiled_register": Matrix multiplication using shared memory
            #   and register tiling .
            # - "block_tiled": Matrix multiplication using a 2D block tiling
            #   strategy.
            # - "block_tiled_vectorized": Matrix multiplication using a
            #   further-optimized 2D block tiling strategy.
            # - "tensor_core": Matrix multiplication using Tensor Cores.
            # In each case, the specific matrix multiplication function is
            # compiled and enqueued to run on the GPU.
            @parameter
            if algorithm == "naive":
                alias BM = 16
                alias BN = 16
                gpu_ctx.enqueue_function[
                    naive_matrix_multiplication[
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BN, BM),
                )
            elif algorithm == "coalescing":
                alias BM = OPTIMIZED_BLOCK_SIZE
                alias BN = OPTIMIZED_BLOCK_SIZE
                gpu_ctx.enqueue_function[
                    coalescing_matrix_multiplication[
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BN, BM),
                )
            elif algorithm == "tiled":
                alias BM = OPTIMIZED_BLOCK_SIZE
                alias BN = OPTIMIZED_BLOCK_SIZE
                alias BK = OPTIMIZED_BLOCK_SIZE
                alias NUM_THREADS = BM * BN
                gpu_ctx.enqueue_function[
                    tiled_matrix_multiplication[
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                        BK,
                        NUM_THREADS,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BM * BN),
                )
            elif algorithm == "tiled_register":
                alias BM = 64
                alias BN = 64
                alias BK = 8
                alias TM = 16
                alias NUM_THREADS = (BM * BN) // TM
                gpu_ctx.enqueue_function[
                    tiled_register_matrix_multiplication[
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                        BK,
                        TM,
                        NUM_THREADS,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS),
                )
            elif algorithm == "block_tiled":
                alias BM = 128
                alias BN = 128
                alias BK = 8
                alias TM = 8
                alias TN = 8
                alias NUM_THREADS = (BM * BN) // (TM * TN)
                gpu_ctx.enqueue_function[
                    block_tiled_matrix_multiplication[
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                        BK,
                        TM,
                        TN,
                        NUM_THREADS,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS),
                )
            elif algorithm == "block_tiled_vectorized":
                alias BM = 128
                alias BN = 128
                alias BK = 8
                alias TM = 8
                alias TN = 8
                alias NUM_THREADS = (BM * BN) // (TM * TN)
                gpu_ctx.enqueue_function[
                    block_tiled_vectorized_matrix_multiplication[
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                        BK,
                        TM,
                        TN,
                        NUM_THREADS,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS),
                )
            elif algorithm == "tensor_core":

                @parameter
                if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
                    alias BM = 64
                    alias BN = 64
                    alias BK = OPTIMIZED_BLOCK_SIZE
                    alias WM = 32
                    alias WN = WARP_SIZE
                    # different MMA shapes for AMD and NVIDIA, see:
                    # https://docs.modular.com/mojo/kernels/layout/tensor_core/TensorCore/
                    alias MMA_M = 16
                    alias MMA_N = 16 if has_amd_gpu_accelerator() else 8
                    alias MMA_K = 4
                    alias NUM_WARPS = (BM // WM) * (BN // WN)
                    gpu_ctx.enqueue_function[
                        tensor_core_matrix_multiplication[
                            output.dtype,
                            a_layout.layout,
                            b_layout.layout,
                            out_layout.layout,
                            BM,
                            BN,
                            BK,
                            WM,
                            WN,
                            MMA_M,
                            MMA_N,
                            MMA_K,
                        ]
                    ](
                        a_layout,
                        b_layout,
                        out_layout,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_WARPS * WARP_SIZE),
                    )
                else:
                    raise Error("Tensor Cores are not available on this device")
            else:
                raise Error("No known matmul algorithm:", algorithm)

        else:
            naive_matrix_multiplication_cpu(output, a, b)
