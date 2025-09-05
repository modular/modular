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

from compiler_internal import register
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    thread_idx,
    warp_id,
    lane_id,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import async_copy_wait_all, AddressSpace
from gpu.sync import AMDScheduleBarrierMask
from gpu.sync import schedule_barrier as amd_schedule_barrier
from gpu.sync import schedule_group_barrier
from layout import IntTuple
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    ThreadScope,
    UNKNOWN_VALUE,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_dram_to_local,
    copy_local_to_shared,
    copy_local_to_dram,
)
from layout.math import outer_product_acc

from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore
from layout.layout import blocked_product


from layout.swizzle import Swizzle
from layout._utils import TensorCoreKGroup

from math import ceildiv
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr
from sys.info import (
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    simd_width_of,
)
from tensor_internal import (
    InputTensor,
    ManagedTensorSlice,
    OutputTensor,
)
from utils import StaticTuple
from utils.index import Index, IndexList

# Import AMD helper functions and structs from the kernels subdirectory
from kernels.amd_helpers import (
    compare_equal,
    amd_scheduling_hints,
    AMD_MMA,
    MMATileBuffers,
    mma,
    copy_local_to_dram_32_32_8,
)


@compiler.register("tensor_core_mma")
struct TensorCoreMMA[algorithm: StaticString]:
    """
    The central custom operation that dispatches to multiple different
    matrix multiplication implementations, depending on target hardware and
    selected algorithm.
    """

    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
        M: Int,
        N: Int,
        K: Int,
    ](
        output: OutputTensor[dtype = DType.float32, rank=2],
        a: InputTensor[dtype = DType.float16, rank=2],
        b: InputTensor[dtype = DType.float16, rank=2],
        perform_validation: Bool,
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "gpu":
            a_layout = a.to_layout_tensor()
            b_layout = b.to_layout_tensor()

            gpu_ctx = ctx.get_device_context()

            var b_ptr_to_use: UnsafePointer[Scalar[DType.float16]]

            # Only transpose the B matrix if we are validating the results,
            # otherwise we can pretend the matrix is already transposed
            if algorithm == "mma_tile_buffers" and perform_validation:
                # Create transposed layout tensor using the transposed dimensions N×K
                # Allocate device memory for transposed matrix
                var b_transposed_buffer = gpu_ctx.enqueue_create_buffer[
                    DType.float16
                ](N * K)
                var b_transposed_ptr = b_transposed_buffer.unsafe_ptr()

                # Copy with transpose: element at (i,j) in original K×N goes to (j,i) in transposed N×K
                for i in range(K):  # rows of original K×N matrix
                    for j in range(N):  # cols of original K×N matrix
                        b_transposed_ptr[j * K + i] = b_layout.ptr[i * N + j]

                b_ptr_to_use = b_transposed_ptr
            else:
                b_ptr_to_use = b.unsafe_ptr()

            var b_layout_transposed = LayoutTensor[
                b.dtype, Layout.row_major(N, K), MutableAnyOrigin
            ](b_ptr_to_use)

            out_layout = output.to_layout_tensor()

            gpu_ctx.synchronize()

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
            gpu_ctx.synchronize()  # Ensure clearing is complete

            # We support several compile-time variants for the matrix multiplication calculation:
            # - "naive_tensor": A naive matrix multiplication using LayoutTensors and AMD Tensor Core instructions.
            # - "basic_shared_mem": A basic matrix multiplication using shared memory and AMD Tensor Core instructions.
            # - "multi_block_tiled": A tiled matrix multiplication using shared memory and AMD Tensor Core instructions.
            # - "scheduler_hints": A tiled matrix multiplication using scheduler hints and AMD Tensor Core instructions.
            # - "double_buffer": A tiled matrix multiplication using double buffering and AMD Tensor Core instructions.
            # - "mma_tile_buffers": A matrix multiplication using tile buffers and AMD Tensor Core instructions.

            @parameter
            if algorithm == "naive_tensor":

                @parameter
                if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
                    alias BM = 64
                    alias BN = 64
                    alias BK = 8
                    # # AMD supports 16x16x16 and 32x32x8 mma instructions for bf16
                    alias MMA_M = 32
                    alias MMA_N = 32
                    alias MMA_K = 8
                    alias NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
                    alias NUM_THREADS = NUM_WARPS * WARP_SIZE

                    gpu_ctx.enqueue_function[
                        naive_tensor[
                            a.dtype,
                            output.dtype,
                            a_layout.layout,
                            b_layout.layout,
                            out_layout.layout,
                            BM,
                            BN,
                            BK,
                            MMA_M,
                            MMA_N,
                            MMA_K,
                        ]
                    ](
                        a_layout,
                        b_layout,
                        out_layout,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            elif algorithm == "basic_shared_mem":

                @parameter
                if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
                    alias BM = 64
                    alias BN = 64
                    alias BK = 8
                    # # AMD supports 16x16x16 and 32x32x8 mma instructions for bf16
                    alias MMA_M = 32
                    alias MMA_N = 32
                    alias MMA_K = 8
                    alias NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
                    alias NUM_THREADS = NUM_WARPS * WARP_SIZE

                    gpu_ctx.enqueue_function[
                        basic_shared_mem[
                            a.dtype,
                            output.dtype,
                            a_layout.layout,
                            b_layout.layout,
                            out_layout.layout,
                            BM,
                            BN,
                            BK,
                            MMA_M,
                            MMA_N,
                            MMA_K,
                        ]
                    ](
                        a_layout,
                        b_layout,
                        out_layout,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            elif algorithm == "multi_block_tiled":

                @parameter
                if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
                    alias BM = 256
                    alias BN = 256
                    alias BK = 64
                    alias WM = BM // 2
                    alias WN = BN // 2
                    # # AMD supports 16x16x16 and 32x32x8 mma instructions for bf16
                    alias MMA_M = 32
                    alias MMA_N = 32
                    alias MMA_K = 8
                    alias NUM_WARPS = (BM // WM) * (BN // WN)
                    alias NUM_THREADS = NUM_WARPS * WARP_SIZE

                    gpu_ctx.enqueue_function[
                        multi_block_tiled[
                            a.dtype,
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
                        block_dim=(NUM_THREADS, 1),
                    )
            elif algorithm == "scheduler_hints":

                @parameter
                if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
                    alias BM = 256
                    alias BN = 256
                    alias BK = 64
                    alias WM = BM // 2
                    alias WN = BN // 2
                    # # AMD supports 16x16x16 and 32x32x8 mma instructions for bf16
                    alias MMA_M = 32
                    alias MMA_N = 32
                    alias MMA_K = 8
                    alias NUM_WARPS = (BM // WM) * (BN // WN)
                    alias NUM_THREADS = NUM_WARPS * WARP_SIZE

                    gpu_ctx.enqueue_function[
                        scheduler_hints[
                            a.dtype,
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
                        block_dim=(NUM_THREADS, 1),
                    )
            elif algorithm == "double_buffer":

                @parameter
                if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
                    alias BM = 128
                    alias BN = 128
                    alias BK = 32
                    alias WM = BM // 2
                    alias WN = BN // 2
                    # # AMD supports 16x16x16 and 32x32x8 mma instructions for bf16
                    alias MMA_M = 32
                    alias MMA_N = 32
                    alias MMA_K = 8
                    alias NUM_WARPS = (BM // WM) * (BN // WN)
                    alias NUM_THREADS = NUM_WARPS * WARP_SIZE

                    gpu_ctx.enqueue_function[
                        double_buffer[
                            a.dtype,
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
                        block_dim=(NUM_THREADS, 1),
                    )
            elif algorithm == "mma_tile_buffers":

                @parameter
                if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
                    alias BM = 256
                    alias BN = 256
                    alias BK = 64
                    alias WM = BM // 2
                    alias WN = BN // 2
                    alias WK = BK
                    # # AMD supports 16x16x16 and 32x32x8 mma instructions for bf16
                    alias MMA_M = 32
                    alias MMA_N = 32
                    alias MMA_K = 8

                    alias NUM_WARPS = (BM // WM) * (BN // WN)
                    alias NUM_THREADS = NUM_WARPS * WARP_SIZE

                    gpu_ctx.enqueue_function[
                        mma_tile_buffers[
                            a.dtype,
                            output.dtype,
                            a_layout.layout,
                            b_layout_transposed.layout,
                            out_layout.layout,
                            BM,
                            BN,
                            BK,
                            WM,
                            WN,
                            WK,
                            MMA_M,
                            MMA_N,
                            MMA_K,
                        ]
                    ](
                        a_layout,
                        b_layout_transposed,
                        out_layout,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            else:
                raise Error("No known matmul algorithm:", algorithm)

            if perform_validation:
                var reference_buf = gpu_ctx.enqueue_create_buffer[output.dtype](
                    M * N
                )
                var reference = LayoutTensor[
                    output.dtype, out_layout.layout, MutableAnyOrigin
                ](reference_buf.unsafe_ptr())

                gpu_ctx.synchronize()

                alias BM = 32
                alias BN = 32
                alias BK = 32
                alias WM = BM // 2
                alias WN = BN // 2
                alias MMA_M = 16
                alias MMA_N = 16
                alias MMA_K = 16
                alias NUM_WARPS = (BM // WM) * (BN // WN)
                alias NUM_THREADS = NUM_WARPS * WARP_SIZE

                gpu_ctx.enqueue_function[
                    naive_tensor[
                        a.dtype,
                        output.dtype,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                        BK,
                        MMA_M,
                        MMA_N,
                        MMA_K,
                    ]
                ](
                    a_layout,
                    b_layout,
                    reference,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS, 1),
                )

                gpu_ctx.synchronize()

                print_results = True
                compare_equal[output.dtype, out_layout.layout](
                    reference, out_layout, print_results
                )

                gpu_ctx.synchronize()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn naive_tensor[
    input_type: DType,
    output_type: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[input_type, layout_a, MutableAnyOrigin],
    B: LayoutTensor[input_type, layout_b, MutableAnyOrigin],
    C: LayoutTensor[output_type, layout_c, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    tensor cores.

    Parameters:
        input_type: The data type of the input tensors.
        output_type: The data type of the output tensor.
        layout_a: The layout of the input tensor A.
        layout_b: The layout of the input tensor B.
        layout_c: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        MMA_M: Tensor core instruction shape in M dimension.
        MMA_N: Tensor core instruction shape in N dimension.
        MMA_K: Tensor core instruction shape in K dimension.

    Args:
        A: The input tensor A.
        B: The input tensor B.
        C: The output tensor C.

    This kernel is the naive implementation of matrix multiplication using tensor cores.
    It loads tiles of matrices A and B directly from global memory, and then each warp computes
    a partial result using tensor cores. The partial results are accumulated in registers and
    finally stored back to the output matrix C.

    The kernel assumes that the input matrices A and B are compatible for matrix multiplication,
    i.e., the number of columns in A equals the number of rows in B.
    """
    alias M = C.shape[0]()  # Number of rows in matrix C
    alias N = C.shape[1]()  # Number of columns in matrix C
    alias K = A.shape[1]()  # Number of columns in matrix A

    # Calculate thread configuration from compile-time constants
    alias NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
    alias NUM_THREADS = NUM_WARPS * WARP_SIZE
    alias simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    warp_y, warp_x = divmod(Int(warp_id()), Int(BN // MMA_N))

    # Get the warp tile of the output matrix C
    C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[MMA_M, MMA_N](
        warp_y, warp_x
    )

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    mma_op = TensorCore[output_type, input_type, Index(MMA_M, MMA_N, MMA_K)]()

    # Calculate correct accumulator fragment size based on MMA configuration
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    alias frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate only small register tile for accumulating partial results
    c_reg = tb[output_type]().row_major[1, frag_size]().local().alloc().fill(0)

    # Naive approach: Load directly from global memory for each tensor core operation
    # No intermediate tile caching - simpler but less efficient
    for k_i in range(ceildiv(K, BK)):
        # Get the tiles of A and B for the current iteration
        A_block_tile = A.tile[BM, BK](block_idx.y, k_i)
        B_block_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Get the warp tiles directly from global memory (naive approach)
        A_warp_tile = A_block_tile.tile[MMA_M, MMA_K](warp_y, 0)
        B_warp_tile = B_block_tile.tile[MMA_K, MMA_N](0, warp_x)

        # Load fragments directly from global memory
        a_reg = mma_op.load_a(A_warp_tile)
        b_reg = mma_op.load_b(B_warp_tile)

        # Perform MMA operation using f32 accumulator
        d_reg = mma_op.mma_op(a_reg, b_reg, c_reg)

        # Manual accumulation: bypass TensorCore store_d
        # Copy result directly to register tile
        c_reg.copy_from(d_reg)

    # Write the final accumulated results to the output matrix (f32 -> f32)
    # Manual store: copy register values directly to global memory
    alias warp_layout = Layout.row_major(MMA_M // frag_size, MMA_N)

    var dst = C_warp_tile.vectorize[4, 1]().distribute[warp_layout](lane_id())
    dst.copy_from(c_reg.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn basic_shared_mem[
    input_type: DType,
    output_type: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[input_type, layout_a, MutableAnyOrigin],
    B: LayoutTensor[input_type, layout_b, MutableAnyOrigin],
    C: LayoutTensor[output_type, layout_c, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    tensor cores.

    Parameters:
        input_type: The data type of the input tensors.
        output_type: The data type of the output tensor.
        layout_a: The layout of the input tensor A.
        layout_b: The layout of the input tensor B.
        layout_c: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        MMA_M: Tensor core instruction shape in M dimension.
        MMA_N: Tensor core instruction shape in N dimension.
        MMA_K: Tensor core instruction shape in K dimension.

    Args:
        A: The input tensor A.
        B: The input tensor B.
        C: The output tensor C.

    This kernel uses a tiled approach with tensor cores to compute the matrix
    multiplication. Each warp loads a single tile of matrices A and B into shared memory, and
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

    # Calculate thread configuration from compile-time constants
    alias NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
    alias NUM_THREADS = NUM_WARPS * WARP_SIZE
    alias simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    warp_y, warp_x = divmod(Int(warp_id()), Int(BN // MMA_N))

    # Get the warp tile of the output matrix C
    C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[MMA_M, MMA_N](
        warp_y, warp_x
    )

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    mma_op = TensorCore[output_type, input_type, Index(MMA_M, MMA_N, MMA_K)]()

    # Allocate shared memory for tiles of A and B
    A_sram_tile = tb[input_type]().row_major[BM, BK]().shared().alloc()
    B_sram_tile = tb[input_type]().row_major[BK, BN]().shared().alloc()

    # Calculate correct accumulator fragment size based on MMA configuration
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    alias frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    c_reg = tb[output_type]().row_major[1, frag_size]().local().alloc().fill(0)

    # Iterate over tiles of A and B in the K dimension
    for k_i in range(ceildiv(K, BK)):
        # Use separate optimized thread layouts for A and B tiles
        # A_sram_tile: 64×8, so use 32×8 thread layout (256 threads total)
        # B_sram_tile: 8×64, so use 8×32 thread layout (256 threads total)
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)  # 32×8
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)  # 8×32

        # Get the tiles of A and B for the current iteration
        A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles using properly sized thread layouts to avoid out-of-bounds access
        copy_dram_to_sram[thread_layout=load_a_layout](A_sram_tile, A_dram_tile)
        copy_dram_to_sram[thread_layout=load_b_layout](B_sram_tile, B_dram_tile)
        barrier()  # Synchronize after loading tiles

        # Get the warp tiles of A and B from shared memory
        A_warp_tile = A_sram_tile.tile[MMA_M, MMA_K](warp_y, 0)
        B_warp_tile = B_sram_tile.tile[MMA_K, MMA_N](0, warp_x)

        # Load fragments
        a_reg = mma_op.load_a(A_warp_tile)
        b_reg = mma_op.load_b(B_warp_tile)

        # Perform MMA operation using f32 accumulator
        d_reg = mma_op.mma_op(a_reg, b_reg, c_reg)

        # Manual accumulation: bypass TensorCore store_d
        # Copy result directly to register tile
        c_reg.copy_from(d_reg)

    # Write the final accumulated results to the output matrix (f32 -> f32)
    # Manual store: copy register values directly to global memory
    alias warp_layout = Layout.row_major(MMA_M // frag_size, MMA_N)

    var dst = C_warp_tile.vectorize[4, 1]().distribute[warp_layout](lane_id())
    dst.copy_from(c_reg.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn multi_block_tiled[
    input_type: DType,
    output_type: DType,
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
    A: LayoutTensor[input_type, layout_a, MutableAnyOrigin],
    B: LayoutTensor[input_type, layout_b, MutableAnyOrigin],
    C: LayoutTensor[output_type, layout_c, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    tensor cores.

    Parameters:
        input_type: The data type of the input tensors.
        output_type: The data type of the output tensor.
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

    # Calculate thread configuration from compile-time constants
    alias NUM_WARPS = (BM // WM) * (BN // WN)
    alias NUM_THREADS = NUM_WARPS * WARP_SIZE
    alias simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    warp_y, warp_x = divmod(Int(warp_id()), Int(BN // MMA_N))

    # Get the warp tile of the output matrix C
    C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    mma_op = TensorCore[output_type, input_type, Index(MMA_M, MMA_N, MMA_K)]()

    # Allocate shared memory for tiles of A and B
    A_sram_tile = tb[input_type]().row_major[BM, BK]().shared().alloc()
    B_sram_tile = tb[input_type]().row_major[BK, BN]().shared().alloc()

    # Calculate correct accumulator fragment size based on MMA configuration
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    alias frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    c_reg = (
        tb[output_type]()
        .row_major[WM // MMA_M, (WN * frag_size) // MMA_N]()
        .local()
        .alloc()
        .fill(0)
    )

    # Thread layout for memory transfers
    alias load_layout = Layout.row_major(
        16, 16
    )  # 256 threads - full utilization

    # Iterate over tiles of A and B in the K dimension
    for k_i in range(ceildiv(K, BK)):
        # Get the tiles of A and B for the current iteration
        A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles using non-vectorized synchronous copy (working version)
        copy_dram_to_sram[thread_layout=load_layout](A_sram_tile, A_dram_tile)
        copy_dram_to_sram[thread_layout=load_layout](B_sram_tile, B_dram_tile)
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
                    # Get the MMA tiles directly from shared memory
                    A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    # Get the register tile for the current MMA operation
                    c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)

                    # Load fragments
                    a_reg = mma_op.load_a(A_mma_tile)
                    b_reg = mma_op.load_b(B_mma_tile)

                    # Perform MMA operation using f32 accumulator
                    d_reg = mma_op.mma_op(a_reg, b_reg, c_reg_m_n)

                    # Manual accumulation: bypass TensorCore store_d
                    # Copy result directly to register tile
                    c_reg_m_n.copy_from(d_reg)

    # Write the final accumulated results to the output matrix (f32 -> f32)
    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)

            # Manual store: copy register values directly to global memory
            alias warp_layout = Layout.row_major(MMA_M // frag_size, MMA_N)

            var dst = C_mma_tile.vectorize[4, 1]().distribute[warp_layout](
                lane_id()
            )
            dst.copy_from(c_reg_m_n.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn scheduler_hints[
    input_type: DType,
    output_type: DType,
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
    A: LayoutTensor[input_type, layout_a, MutableAnyOrigin],
    B: LayoutTensor[input_type, layout_b, MutableAnyOrigin],
    C: LayoutTensor[output_type, layout_c, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    tensor cores.

    Parameters:
        input_type: The data type of the input tensors.
        output_type: The data type of the output tensor.
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

    # Calculate thread configuration from compile-time constants
    alias NUM_WARPS = (BM // WM) * (BN // WN)
    alias NUM_THREADS = NUM_WARPS * WARP_SIZE
    alias simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    warp_y, warp_x = divmod(Int(warp_id()), Int(BN // MMA_N))

    # Get the warp tile of the output matrix C
    C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    mma_op = TensorCore[output_type, input_type, Index(MMA_M, MMA_N, MMA_K)]()

    # Allocate single set of shared memory buffers (single buffering to fit memory limit)
    A_sram_tile = tb[input_type]().row_major[BM, BK]().shared().alloc()
    B_sram_tile = tb[input_type]().row_major[BK, BN]().shared().alloc()

    # Calculate correct accumulator fragment size based on MMA configuration
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    alias frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    c_reg = (
        tb[output_type]()
        .row_major[WM // MMA_M, (WN * frag_size) // MMA_N]()
        .local()
        .alloc()
        .fill(0)
    )

    # Thread layout for memory transfers
    alias load_layout = Layout.row_major(
        16, 16
    )  # 256 threads - full utilization

    # Simplified single-buffer pipeline (similar to basic_shared_mem but with AMD scheduling)
    for k_i in range(ceildiv(K, BK)):
        # Get the tiles of A and B for the current iteration
        A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles using synchronous copy (single buffering)
        copy_dram_to_sram[thread_layout=load_layout](A_sram_tile, A_dram_tile)
        copy_dram_to_sram[thread_layout=load_layout](B_sram_tile, B_dram_tile)
        barrier()  # Synchronize after loading tiles

        # Schedule barrier after loading
        @parameter
        if has_amd_gpu_accelerator():
            amd_schedule_barrier()

        # Get the warp tiles from shared memory
        A_warp_tile = A_sram_tile.tile[WM, BK](warp_y, 0)
        B_warp_tile = B_sram_tile.tile[BK, WN](0, warp_x)

        # Perform MMA operations on current tile with AMD scheduling hints
        @parameter
        for mma_k in range(BK // MMA_K):

            @parameter
            for mma_m in range(WM // MMA_M):

                @parameter
                for mma_n in range(WN // MMA_N):
                    # Get the MMA tiles from shared memory
                    A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    # Get the register tile for the current MMA operation
                    c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)

                    # Load fragments and perform MMA
                    a_reg = mma_op.load_a(A_mma_tile)
                    b_reg = mma_op.load_b(B_mma_tile)
                    d_reg = mma_op.mma_op(a_reg, b_reg, c_reg_m_n)

                    # Manual accumulation for 32x32x8
                    c_reg_m_n.copy_from(d_reg)

        # Add AMD scheduling hints between tiles
        @parameter
        if has_amd_gpu_accelerator():
            amd_scheduling_hints[
                input_type,
                output_type,
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                MMA_K,
                IndexList[3](6, 3, 2),
            ]()

    # Final schedule barrier before output phase
    @parameter
    if has_amd_gpu_accelerator():
        amd_schedule_barrier()

    # === OUTPUT PHASE ===
    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_tile = c_reg.tile[1, frag_size](mma_m, mma_n)

            alias warp_layout = Layout.row_major(MMA_M // frag_size, MMA_N)

            var dst = C_mma_tile.vectorize[4, 1]().distribute[warp_layout](
                lane_id()
            )
            dst.copy_from(c_reg_tile.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn double_buffer[
    input_type: DType,
    output_type: DType,
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
    A: LayoutTensor[input_type, layout_a, MutableAnyOrigin],
    B: LayoutTensor[input_type, layout_b, MutableAnyOrigin],
    C: LayoutTensor[output_type, layout_c, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B using
    tensor cores.

    Parameters:
        input_type: The data type of the input tensors.
        output_type: The data type of the output tensor.
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

    # Calculate thread configuration from compile-time constants
    alias NUM_WARPS = (BM // WM) * (BN // WN)
    alias NUM_THREADS = NUM_WARPS * WARP_SIZE
    alias simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    warp_y, warp_x = divmod(Int(warp_id()), Int(BN // MMA_N))

    # Get the warp tile of the output matrix C
    C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    mma_op = TensorCore[output_type, input_type, Index(MMA_M, MMA_N, MMA_K)]()

    # Allocate two sets of shared memory buffers for double buffering
    A_sram_buffer_0 = tb[input_type]().row_major[BM, BK]().shared().alloc()
    A_sram_buffer_1 = tb[input_type]().row_major[BM, BK]().shared().alloc()
    B_sram_buffer_0 = tb[input_type]().row_major[BK, BN]().shared().alloc()
    B_sram_buffer_1 = tb[input_type]().row_major[BK, BN]().shared().alloc()

    # Calculate correct accumulator fragment size based on MMA configuration
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    alias frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    c_reg = (
        tb[output_type]()
        .row_major[WM // MMA_M, (WN * frag_size) // MMA_N]()
        .local()
        .alloc()
        .fill(0)
    )
    # Thread layout for memory transfers
    alias load_layout = Layout.row_major(
        32, 8
    )  # 256 threads - full utilization

    # Calculate total K iterations
    var k_iterations = ceildiv(K, BK)

    # Track which buffer set is currently being used for computation
    var compute_buffer_idx = 0

    # === PIPELINE STAGE 1: Initial Load ===
    # Load the first tile into buffer 0
    if k_iterations > 0:
        var A_dram_tile_0 = A.tile[BM, BK](block_idx.y, 0)
        var B_dram_tile_0 = B.tile[BK, BN](0, block_idx.x)

        copy_dram_to_sram[thread_layout=load_layout](
            A_sram_buffer_0, A_dram_tile_0
        )
        copy_dram_to_sram[thread_layout=load_layout](
            B_sram_buffer_0, B_dram_tile_0
        )
        barrier()  # Synchronize initial load

    # === PIPELINE STAGE 2: Main Double-Buffered Loop ===
    for k_i in range(k_iterations):
        var use_buffer_0 = (k_i % 2) == 0

        # Select current compute buffers
        var A_compute_buffer = (
            A_sram_buffer_0 if use_buffer_0 else A_sram_buffer_1
        )
        var B_compute_buffer = (
            B_sram_buffer_0 if use_buffer_0 else B_sram_buffer_1
        )

        # Select next load buffers (alternate set)
        var A_load_buffer = A_sram_buffer_1 if use_buffer_0 else A_sram_buffer_0
        var B_load_buffer = B_sram_buffer_1 if use_buffer_0 else B_sram_buffer_0

        # === ASYNC LOAD: Start loading NEXT iteration while computing current ===
        var next_k = k_i + 1
        if next_k < k_iterations:
            var A_dram_tile_next = A.tile[BM, BK](block_idx.y, next_k)
            var B_dram_tile_next = B.tile[BK, BN](next_k, block_idx.x)

            # Start loading next iteration's data into alternate buffers
            # This happens in parallel with computation below
            # Vectorize to create 4-byte elements (2 × f16 = 4 bytes)
            var A_dram_vectorized = A_dram_tile_next.vectorize[
                1, 2
            ]()  # 2 f16s per element
            var B_dram_vectorized = B_dram_tile_next.vectorize[
                1, 2
            ]()  # 2 f16s per element
            var A_load_vectorized = A_load_buffer.vectorize[1, 2]()
            var B_load_vectorized = B_load_buffer.vectorize[1, 2]()

            # Now each element is 4 bytes, so async copy works
            copy_dram_to_sram_async[thread_layout=load_layout](
                A_load_vectorized, A_dram_vectorized
            )
            copy_dram_to_sram_async[thread_layout=load_layout](
                B_load_vectorized, B_dram_vectorized
            )

        # === COMPUTE PHASE: Use current buffers for computation ===
        # Get the warp tiles from the current compute buffers
        A_warp_tile = A_compute_buffer.tile[WM, BK](warp_y, 0)
        B_warp_tile = B_compute_buffer.tile[BK, WN](0, warp_x)

        # Perform MMA operations on current tile
        @parameter
        for mma_k in range(BK // MMA_K):

            @parameter
            for mma_m in range(WM // MMA_M):

                @parameter
                for mma_n in range(WN // MMA_N):
                    # Get the MMA tiles from shared memory
                    A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    # Get the register tile for the current MMA operation
                    c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)

                    # Load fragments and perform MMA
                    a_reg = mma_op.load_a(A_mma_tile)
                    b_reg = mma_op.load_b(B_mma_tile)
                    d_reg = mma_op.mma_op(a_reg, b_reg, c_reg_m_n)

                    # Manual accumulation: bypass TensorCore store_d
                    # Copy result directly to register tile
                    c_reg_m_n.copy_from(d_reg)

        # === SYNC: Ensure next iteration's data is ready ===
        if next_k < k_iterations:
            async_copy_wait_all()  # Wait for async loads to complete
            barrier()  # Ensure all threads see the loaded data

    # === OUTPUT PHASE: Write results to global memory manually ===
    # Bypass TensorCore store_d and use direct register-to-memory copy
    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)

            # Manual store: copy register values directly to global memory
            alias warp_layout = Layout.row_major(MMA_M // frag_size, MMA_N)

            var dst = C_mma_tile.vectorize[4, 1]().distribute[warp_layout](
                lane_id()
            )
            dst.copy_from(c_reg_m_n.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn mma_tile_buffers[
    input_type: DType,
    output_type: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[input_type, layout_a, MutableAnyOrigin],
    B: LayoutTensor[input_type, layout_b, MutableAnyOrigin],
    C: LayoutTensor[output_type, layout_c, MutableAnyOrigin],
):
    """
    AMD-style tiled GEMM kernel with sophisticated scheduling hints.

    Parameters:
        input_type: The data type of the input tensors.
        output_type: The data type of the output tensor.
        layout_a: The layout of the input tensor A.
        layout_b: The layout of the input tensor B.
        layout_c: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        WM: The warp tile size in the M dimension.
        WN: The warp tile size in the N dimension.
        WK: The warp tile size in the K dimension.
        MMA_M: Tensor core instruction shape in M dimension.
        MMA_N: Tensor core instruction shape in N dimension.
        MMA_K: Tensor core instruction shape in K dimension.

    Args:
        A: The input tensor A.
        B: The input tensor B.
        C: The output tensor C.

    This implementation follows the multi-stage pipeline approach used in
    AMD's optimized GEMM kernel with strategic placement of scheduling hints
    and K-group processing.
    """
    # Validate input constraints
    alias transpose_b = True
    constrained[
        transpose_b, "Transpose b must be true for this implementation"
    ]()

    # Matrix dimensions from input tensors
    var M = A.dim[0]()
    var N = B.dim[0 if transpose_b else 1]()
    var K = B.dim[1 if transpose_b else 0]()
    alias stride = B.stride[0]()

    # Type alias for accumulator type
    alias accum_type = DType.float32

    # SIMD and vectorization parameters
    alias simd_width = simd_width_of[input_type]()

    # Warp organization
    alias num_warps_m = UInt(BM // WM)
    alias num_warps_n = UInt(BN // WN)
    alias num_warps_k = UInt(BK // WK)

    alias warps_per_block = num_warps_m * num_warps_n * num_warps_k

    # MMA instruction tiling
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    # K dimension tiling
    alias k_group_size = 16 // simd_width
    alias k_tile_size = MMA_K * k_group_size
    alias num_k_tiles = WK // k_tile_size

    # Thread and warp indices
    var warp_id = warp_id()
    var warp_km, warp_n = divmod(warp_id, num_warps_n)
    var warp_k, warp_m = divmod(warp_km, num_warps_m)

    # Helper function for thread layout
    @parameter
    fn get_thread_layout() -> Layout:
        # TODO: Document the logic behind this layout
        # Define a layout that corresponds to the below pattern:
        #
        # | T00 T01 T02 T03 | T16 T17 T18 T19 | ...
        # | T04 T05 T06 T07 | T20 T21 T22 T23 |
        # | T08 T09 T10 T11 | T24 T25 T26 T27 |
        # | T12 T13 T14 T15 | T28 T29 T30 T31 |
        # | T64 T65 T66 T67 | T80 T81 T82 T83 | ...
        # | T68 T69 T70 T71 | T84 T85 T86 T87 |
        # | T72 T73 T74 T75 | T88 T89 T90 T91 |
        # | T76 T77 T78 T79 | T92 T93 T94 T95 |
        alias inner_block_size = 16
        alias inner_block_cols = k_tile_size // simd_width  # 4/2
        alias inner_block_rows = inner_block_size // inner_block_cols  # 4/8

        alias base_layout = Layout.row_major(
            inner_block_rows, inner_block_cols
        )  # (4, 4) or (8, 2)

        alias num_repeats_col = BK // k_tile_size  # 2/4
        alias outer_block_size = num_repeats_col * inner_block_size  # 32/64
        alias num_repeats_row = 256 // outer_block_size  # 8/4

        alias tiler_layout = Layout.row_major(  # (8, 2) or (4, 4)
            num_repeats_row,
            num_repeats_col,
        )

        # (((4, 8), (4, 2)):((4, 32), (1, 16))) or (((8, 4), (2, 4)):((2, 64), (1, 16)))
        return blocked_product(base_layout, tiler_layout)

    # Helper function for shared memory layout
    @parameter
    fn get_smem_layout[block_rows: Int]() -> Layout:
        # Shared memory layout
        #
        # - base_layout: Layout.row_major(block_rows, k_tile_size) -> block_rows×k_tile_size tiles
        # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
        # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
        #
        # Resulting shape: block_rows×(k_tile_size × num_repeats) = block_rows×BK tensor
        # Where BK = k_tile_size × num_repeats, k_tile_size = MMA_K × k_group_size
        #
        # This creates num_repeats blocks of block_rows×k_tile_size arranged horizontally:
        # Within each k_tile_size-column block, elements are consecutive (stride 1)
        # Between blocks: stride = block_rows × k_tile_size
        #
        # ASCII diagram for block_rows=64, k_tile_size=32, BK=64 (showing first 2 of 2 blocks):
        # ┌─────────────────────────────────────────────────────────────────────────┐
        # │         Block 0 (64×32)             │         Block 1 (64×32)           │
        # ├─────────────────────────────────────┼───────────────────────────────────┤
        # │   0    1    2  ...   30   31        │ 2048 2049 2050 ... 2078 2079      │
        # │  32   33   34  ...   62   63        │ 2080 2081 2082 ... 2110 2111      │
        # │  64   65   66  ...   94   95        │ 2112 2113 2114 ... 2142 2143      │
        # │  96   97   98  ...  126  127        │ 2144 2145 2146 ... 2174 2175      │
        # │ ...                                 │  ...                              │
        # │2016 2017 2018  ... 2046 2047        │ 4064 4065 4066 ... 4094 4095      │
        # └─────────────────────────────────────────────────────────────────────────┘
        # stride between blocks = block_rows × k_tile_size = 64 × 32 = 2048

        alias base_layout = Layout.row_major(block_rows, k_tile_size)
        alias num_repeats = BK // k_tile_size
        alias tiler_layout = Layout.row_major(1, num_repeats)

        # return blocked_product(base_layout, tiler_layout, coalesce_output=True)
        return blocked_product(base_layout, tiler_layout)

    # AMD TensorCore operator for matrix multiplication
    alias mma_shape = IndexList[3](MMA_M, MMA_N, MMA_K)
    alias amd_mma = AMD_MMA[
        out_type=accum_type,
        in_type=input_type,
        shape=mma_shape,
        transpose_b=transpose_b,
        k_group_size=k_group_size,
        num_k_tiles=num_k_tiles,
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        simd_width=simd_width,
        swizzle = Swizzle(3, 0, 1),
        BK=BK,
        WK=WK,
    ]

    var a_tiles = MMATileBuffers[
        get_smem_layout[BM](),
        tensor_type = __type_of(A),
        thread_layout = get_thread_layout(),
        block_rows=BM,
        warp_rows=WM,
        stride=stride,
        num_mmas=num_m_mmas,
        mma_type=amd_mma,
    ](A, Int(warp_m), Int(warp_k), Int(block_idx.y))

    # B (weights matrix) memory
    var b_tiles = MMATileBuffers[
        get_smem_layout[BN](),
        tensor_type = __type_of(B),
        thread_layout = get_thread_layout(),
        block_rows=BN,
        warp_rows=WN,
        stride=stride,
        num_mmas=num_n_mmas,
        mma_type=amd_mma,
    ](B, Int(warp_n), Int(warp_k), Int(block_idx.x))

    # Calculate the correct number of accumulator registers based on MMA instruction shape
    # AMD 32x32x8 MFMA requires 16 f32 accumulator values per thread (with WARP_SIZE=64)
    alias frag_size = (MMA_M * MMA_N) // WARP_SIZE

    alias c_reg_tile_type = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var c_reg_tile = c_reg_tile_type.stack_allocation().fill(0)

    # Helper functions for matrix operations
    @always_inline
    @parameter
    fn load_tiles_from_dram():
        a_tiles.load_from_dram()
        b_tiles.load_from_dram()

    @always_inline
    @parameter
    fn copy_tiles_to_shared():
        a_tiles.copy_to_shared()
        b_tiles.copy_to_shared()

    @always_inline
    @parameter
    fn load_tiles_from_shared[k_tile_idx: Int]():
        a_tiles.load_tile_from_shared[k_tile_idx, is_a=True]()
        b_tiles.load_tile_from_shared[k_tile_idx, is_a=True]()

    # GEMM Computation Pipeline
    # This kernel implements a pipelined approach optimized for AMD GPUs:
    # 1. Load: Transfer first tiles from global to shared memory
    # 2. Prepare: Load shared memory data to registers, prefetch next tiles
    # 3. Main Loop: Process tiles with overlapped computation and data movement
    # 4. Finalize: Process remaining tiles and write results back

    # Stage 1: Initial data loading - Global→Local→Shared memory transfer
    load_tiles_from_dram()
    copy_tiles_to_shared()

    barrier()

    # Stage 2: First tile preparation - Register loading and prefetching
    load_tiles_from_dram()
    load_tiles_from_shared[0]()

    amd_schedule_barrier()

    # Stage 3: Main computation loop - Pipelined execution with double buffering
    var k_iterations = K // BK
    if k_iterations > 2:
        for _ in range(2, k_iterations):

            @parameter
            for k_tile_idx in range(1, num_k_tiles):
                load_tiles_from_shared[k_tile_idx]()

            mma[0, swap_a_b=transpose_b](a_tiles, b_tiles, c_reg_tile)

            barrier()

            copy_tiles_to_shared()
            load_tiles_from_dram()

            @parameter
            for k_tile_idx in range(1, num_k_tiles):
                mma[k_tile_idx, swap_a_b=transpose_b](
                    a_tiles, b_tiles, c_reg_tile
                )

            barrier()

            load_tiles_from_shared[0]()

            amd_scheduling_hints[
                input_type,
                output_type,
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                MMA_K,
                IndexList[3](6, 6, 2),
            ]()

    amd_schedule_barrier()

    @parameter
    for k_tile_idx in range(1, num_k_tiles):
        load_tiles_from_shared[k_tile_idx]()

    barrier()

    copy_tiles_to_shared()

    @parameter
    for k_tile_idx in range(num_k_tiles):
        mma[k_tile_idx, swap_a_b=transpose_b](a_tiles, b_tiles, c_reg_tile)

    amd_schedule_barrier()

    barrier()

    @parameter
    for k_tile_idx in range(num_k_tiles):
        load_tiles_from_shared[k_tile_idx]()

    @parameter
    for k_tile_idx in range(num_k_tiles):
        mma[k_tile_idx, swap_a_b=transpose_b](a_tiles, b_tiles, c_reg_tile)

    amd_schedule_barrier()

    # --- Write results to output tensor ---
    # Output stage: Transfer results from registers to global memory
    var c_block_tile = C.tile[BM, BN](block_idx.y, block_idx.x)
    var c_warp_tile = c_block_tile.tile[WM, WN](
        warp_m, warp_n
    )  # 128 x 128 -> 128 x (8 x 16)

    @parameter
    if MMA_M == 16:
        alias output_thread_layout = Layout.col_major(16, 4)
        copy_local_to_dram[
            output_thread_layout, thread_scope = ThreadScope.WARP
        ](c_reg_tile.vectorize[1, 4](), c_reg_tile.vectorize[1, 4](), C)

    else:
        alias output_thread_layout = Layout.col_major(32, 2)
        copy_local_to_dram_32_32_8[
            output_thread_layout, thread_scope = ThreadScope.WARP
        ](c_warp_tile.vectorize[1, 4](), c_reg_tile.vectorize[1, 4](), C)
