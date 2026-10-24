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

import extensibility

from std.math import abs, ceildiv, isfinite
from std.math.uutils import udivmod
from std.memory import bitcast
from std.sys.info import (
    current_accelerator,
    simd_width_of,
)

from extensibility import register
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
)
from max.gpu.sync import schedule_barrier as amd_schedule_barrier

# Import AMD helper functions and structs from the kernels subdirectory

from .amd_helpers import amd_scheduling_hints, compare_equal
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from layout.layout import blocked_product
from linalg.matmul.gpu.amd.matmul_mma import MmaOp
from structured_kernels.amd_tile_io import (
    RegTileLoader,
    RegTileWriter,
    RegTileWriterLDS,
)
from std.bit import log2_floor
from std.collections import Optional
from layout.tile_io import copy_dram_to_sram, copy_dram_to_sram_async
from layout.swizzle import Swizzle
from layout.tensor_core import TensorCore

from extensibility import InputTensor, OutputTensor

from std.utils import StaticTuple
from std.utils.index import Index, IndexList


@extensibility.register("tensor_core_mma")
struct TensorCoreMMA[algorithm: StaticString]:
    """
    The central custom operation that dispatches to multiple different
    matrix multiplication implementations, depending on target hardware and
    selected algorithm.
    """

    @staticmethod
    def execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
        M: Int,
        N: Int,
        K: Int,
    ](
        output: OutputTensor[dtype=.float32, rank=2, ...],
        a: InputTensor[dtype=.float16, rank=2, ...],
        b: InputTensor[dtype=.float16, rank=2, ...],
        perform_validation: Bool,
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        comptime if target == "gpu":
            var a_tt = a.to_tile_tensor().as_unsafe_any_origin()
            var b_tt = b.to_tile_tensor().as_unsafe_any_origin()

            var gpu_ctx = ctx
            comptime warp_size = gpu_ctx.default_device_info.warp_size
            comptime if gpu_ctx.target.is_nvidia_gpu() and Self.algorithm == "mma_tile_buffers":
                raise Error("mma_tile_buffers is AMD-only")

            var b_ptr_to_use: Pointer[Float16, MutAnyOrigin]

            # Only transpose the B matrix if we are validating the results,
            # otherwise we can pretend the matrix is already transposed
            if Self.algorithm == "mma_tile_buffers" and perform_validation:
                # Create transposed layout tensor using the transposed dimensions NxK
                # Allocate device memory for transposed matrix
                var b_transposed_buffer = gpu_ctx.enqueue_create_buffer[
                    DType.float16
                ](N * K)
                var b_transposed_ptr = b_transposed_buffer.unsafe_ptr()

                # Copy with transpose: element at (i,j) in original KxN goes to (j,i) in transposed NxK
                for i in range(K):  # rows of original KxN matrix
                    for j in range(N):  # cols of original KxN matrix
                        b_transposed_ptr[
                            unsafe_offset=j * K + i
                        ] = b_tt._storage[unsafe_offset=i * N + j]

                b_ptr_to_use = b_transposed_ptr.as_unsafe_any_origin()
            else:
                b_ptr_to_use = b.unsafe_ptr()

            # `mma_tile_buffers` consumes the (transposed) B matrix as a
            # `TileTensor` (static `row_major[N, K]` so the kernel's
            # `comptime stride` stays compile-time). The other algorithms use the
            # `TileTensor` views (`a_tt`/`b_tt`/`out_tt`) directly.
            var b_transposed_tt = TileTensor(
                Span(unsafe_ptr=b_ptr_to_use, length=N * K), row_major[N, K]()
            ).as_unsafe_any_origin()

            var out_tt = output.to_tile_tensor().as_unsafe_any_origin()
            comptime if gpu_ctx.target.is_nvidia_gpu():
                comptime assert out_tt.rank == out_tt.flat_rank == 2
                var output_strides = out_tt.layout.stride_coord()
                if Int(output_strides[0]) != N or Int(output_strides[1]) != 1:
                    raise Error(
                        "NVIDIA tensor-core examples require row-major output"
                    )

            gpu_ctx.synchronize()

            # Zero out the memory in the outbound tensor.
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    out_tt._storage,
                    M * N,
                    owning=False,
                ),
                0,
            )
            gpu_ctx.synchronize()  # Ensure clearing is complete

            # We support several compile-time variants for the matrix multiplication calculation:
            # - "naive_tensor": A naive matrix multiplication using TileTensors and tensor core instructions.
            # - "basic_shared_mem": A basic matrix multiplication using shared memory and tensor core instructions.
            # - "multi_block_tiled": A tiled matrix multiplication using shared memory and tensor core instructions.
            # - "scheduler_hints": A tiled matrix multiplication using scheduler hints and tensor core instructions.
            # - "double_buffer": A tiled matrix multiplication using double buffering and tensor core instructions.
            # - "mma_tile_buffers": A matrix multiplication using tile buffers and AMD Tensor Core instructions.

            comptime if Self.algorithm == "naive_tensor":
                comptime if gpu_ctx.target.is_nvidia_gpu() or gpu_ctx.target.is_amd_gpu():
                    comptime BM = 32 if gpu_ctx.target.is_nvidia_gpu() else 64
                    comptime BN = 32 if gpu_ctx.target.is_nvidia_gpu() else 64
                    comptime BK = 16 if gpu_ctx.target.is_nvidia_gpu() else 8
                    comptime MMA_M = 16 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_N = 8 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_K = 16 if gpu_ctx.target.is_nvidia_gpu() else 8
                    comptime NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
                    comptime NUM_THREADS = NUM_WARPS * warp_size
                    comptime if gpu_ctx.target.is_nvidia_gpu():
                        comptime assert M > 0 and N > 0 and K > 0
                        comptime assert (
                            M % BM == 0 and N % BN == 0 and K % BK == 0
                        ), (
                            "NVIDIA tensor-core examples require complete block"
                            " tiles"
                        )
                    comptime native_kernel = naive_tensor[
                        a.dtype,
                        output.dtype,
                        type_of(a_tt).LayoutType,
                        type_of(b_tt).LayoutType,
                        type_of(out_tt).LayoutType,
                        BM,
                        BN,
                        BK,
                        MMA_M,
                        MMA_N,
                        MMA_K,
                    ]

                    gpu_ctx.enqueue_function[native_kernel](
                        a_tt,
                        b_tt,
                        out_tt,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            elif Self.algorithm == "basic_shared_mem":
                comptime if gpu_ctx.target.is_nvidia_gpu() or gpu_ctx.target.is_amd_gpu():
                    comptime BM = 32 if gpu_ctx.target.is_nvidia_gpu() else 64
                    comptime BN = 32 if gpu_ctx.target.is_nvidia_gpu() else 64
                    comptime BK = 16 if gpu_ctx.target.is_nvidia_gpu() else 8
                    comptime MMA_M = 16 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_N = 8 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_K = 16 if gpu_ctx.target.is_nvidia_gpu() else 8
                    comptime NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
                    comptime NUM_THREADS = NUM_WARPS * warp_size
                    comptime if gpu_ctx.target.is_nvidia_gpu():
                        comptime assert M > 0 and N > 0 and K > 0
                        comptime assert (
                            M % BM == 0 and N % BN == 0 and K % BK == 0
                        ), (
                            "NVIDIA tensor-core examples require complete block"
                            " tiles"
                        )
                    comptime basic_shared_mem_kernel = basic_shared_mem[
                        a.dtype,
                        output.dtype,
                        type_of(a_tt).LayoutType,
                        type_of(b_tt).LayoutType,
                        type_of(out_tt).LayoutType,
                        BM,
                        BN,
                        BK,
                        MMA_M,
                        MMA_N,
                        MMA_K,
                    ]

                    gpu_ctx.enqueue_function[basic_shared_mem_kernel](
                        a_tt,
                        b_tt,
                        out_tt,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            elif Self.algorithm == "multi_block_tiled":
                comptime if gpu_ctx.target.is_nvidia_gpu() or gpu_ctx.target.is_amd_gpu():
                    comptime BM = 256
                    comptime BN = 256
                    comptime BK = 64
                    comptime WM = BM // 2
                    comptime WN = BN // 2
                    comptime MMA_M = 16 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_N = 8 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_K = 16 if gpu_ctx.target.is_nvidia_gpu() else 8
                    comptime NUM_WARPS = (BM // WM) * (BN // WN)
                    comptime NUM_THREADS = NUM_WARPS * warp_size
                    comptime if gpu_ctx.target.is_nvidia_gpu():
                        comptime assert M > 0 and N > 0 and K > 0
                        comptime assert (
                            M % BM == 0 and N % BN == 0 and K % BK == 0
                        ), (
                            "NVIDIA tensor-core examples require complete block"
                            " tiles"
                        )
                    comptime multi_block_tiled_kernel = multi_block_tiled[
                        a.dtype,
                        output.dtype,
                        type_of(a_tt).LayoutType,
                        type_of(b_tt).LayoutType,
                        type_of(out_tt).LayoutType,
                        BM,
                        BN,
                        BK,
                        WM,
                        WN,
                        MMA_M,
                        MMA_N,
                        MMA_K,
                    ]

                    gpu_ctx.enqueue_function[multi_block_tiled_kernel](
                        a_tt,
                        b_tt,
                        out_tt,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            elif Self.algorithm == "scheduler_hints":
                comptime if gpu_ctx.target.is_nvidia_gpu() or gpu_ctx.target.is_amd_gpu():
                    comptime BM = 256
                    comptime BN = 256
                    comptime BK = 64
                    comptime WM = BM // 2
                    comptime WN = BN // 2
                    comptime MMA_M = 16 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_N = 8 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_K = 16 if gpu_ctx.target.is_nvidia_gpu() else 8
                    comptime NUM_WARPS = (BM // WM) * (BN // WN)
                    comptime NUM_THREADS = NUM_WARPS * warp_size
                    comptime if gpu_ctx.target.is_nvidia_gpu():
                        comptime assert M > 0 and N > 0 and K > 0
                        comptime assert (
                            M % BM == 0 and N % BN == 0 and K % BK == 0
                        ), (
                            "NVIDIA tensor-core examples require complete block"
                            " tiles"
                        )
                    comptime scheduler_hints_kernel = scheduler_hints[
                        a.dtype,
                        output.dtype,
                        type_of(a_tt).LayoutType,
                        type_of(b_tt).LayoutType,
                        type_of(out_tt).LayoutType,
                        BM,
                        BN,
                        BK,
                        WM,
                        WN,
                        MMA_M,
                        MMA_N,
                        MMA_K,
                    ]

                    gpu_ctx.enqueue_function[scheduler_hints_kernel](
                        a_tt,
                        b_tt,
                        out_tt,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            elif Self.algorithm == "double_buffer":
                comptime if gpu_ctx.target.is_nvidia_gpu() or gpu_ctx.target.is_amd_gpu():
                    comptime BM = 128
                    comptime BN = 128
                    comptime BK = 32
                    comptime WM = BM // 2
                    comptime WN = BN // 2
                    comptime MMA_M = 16 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_N = 8 if gpu_ctx.target.is_nvidia_gpu() else 32
                    comptime MMA_K = 16 if gpu_ctx.target.is_nvidia_gpu() else 8
                    comptime NUM_WARPS = (BM // WM) * (BN // WN)
                    comptime NUM_THREADS = NUM_WARPS * warp_size
                    comptime if gpu_ctx.target.is_nvidia_gpu():
                        comptime assert M > 0 and N > 0 and K > 0
                        comptime assert (
                            M % BM == 0 and N % BN == 0 and K % BK == 0
                        ), (
                            "NVIDIA tensor-core examples require complete block"
                            " tiles"
                        )
                    comptime double_buffer_kernel = double_buffer[
                        a.dtype,
                        output.dtype,
                        type_of(a_tt).LayoutType,
                        type_of(b_tt).LayoutType,
                        type_of(out_tt).LayoutType,
                        BM,
                        BN,
                        BK,
                        WM,
                        WN,
                        MMA_M,
                        MMA_N,
                        MMA_K,
                    ]

                    gpu_ctx.enqueue_function[double_buffer_kernel](
                        a_tt,
                        b_tt,
                        out_tt,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            elif Self.algorithm == "mma_tile_buffers":
                comptime if gpu_ctx.target.is_amd_gpu():
                    comptime BM = 256
                    comptime BN = 256
                    comptime BK = 64
                    comptime WM = BM // 2
                    comptime WN = BN // 2
                    comptime WK = BK
                    # # AMD supports 16x16x16 and 32x32x8 mma instructions for bf16
                    comptime MMA_M = 32
                    comptime MMA_N = 32
                    comptime MMA_K = 8

                    comptime NUM_WARPS = (BM // WM) * (BN // WN)
                    comptime NUM_THREADS = NUM_WARPS * warp_size

                    # `mma_tile_buffers` takes `TileTensor` A/B/C. B is the
                    # (transposed) `row_major[N, K]` `TileTensor` built above.
                    comptime mma_tile_buffers_kernel = mma_tile_buffers[
                        a.dtype,
                        output.dtype,
                        type_of(a_tt).LayoutType,
                        type_of(b_transposed_tt).LayoutType,
                        type_of(out_tt).LayoutType,
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

                    gpu_ctx.enqueue_function[mma_tile_buffers_kernel](
                        a_tt,
                        b_transposed_tt,
                        out_tt,
                        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                        block_dim=(NUM_THREADS, 1),
                    )
            else:
                raise Error("No known matmul algorithm:", Self.algorithm)

            if perform_validation:
                var reference_buf = gpu_ctx.enqueue_create_buffer[output.dtype](
                    M * N
                )
                var reference = TileTensor(
                    reference_buf, out_tt.layout
                ).as_unsafe_any_origin()

                gpu_ctx.synchronize()

                comptime BM = 32
                comptime BN = 32
                comptime BK = 32
                comptime WM = BM // 2
                comptime WN = BN // 2
                comptime MMA_M = 16
                comptime MMA_N = 8 if gpu_ctx.target.is_nvidia_gpu() else 16
                comptime MMA_K = 16
                comptime NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
                comptime NUM_THREADS = NUM_WARPS * warp_size
                comptime naive_tensor_kernel = naive_tensor[
                    a.dtype,
                    output.dtype,
                    type_of(a_tt).LayoutType,
                    type_of(b_tt).LayoutType,
                    type_of(reference).LayoutType,
                    BM,
                    BN,
                    BK,
                    MMA_M,
                    MMA_N,
                    MMA_K,
                ]

                gpu_ctx.enqueue_function[naive_tensor_kernel](
                    a_tt,
                    b_tt,
                    reference,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(NUM_THREADS, 1),
                )

                gpu_ctx.synchronize()

                comptime if gpu_ctx.target.is_nvidia_gpu():
                    var reference_host = gpu_ctx.enqueue_create_host_buffer[
                        output.dtype
                    ](M * N)
                    var computed_host = gpu_ctx.enqueue_create_host_buffer[
                        output.dtype
                    ](M * N)
                    var computed_device = DeviceBuffer[output.dtype](
                        gpu_ctx, out_tt.unsafe_ptr(), M * N, owning=False
                    )
                    gpu_ctx.enqueue_copy(
                        reference_host.unsafe_ptr(), reference_buf
                    )
                    gpu_ctx.enqueue_copy(
                        computed_host.unsafe_ptr(), computed_device
                    )
                    gpu_ctx.synchronize()
                    # Different FP32 accumulation associations need not agree bitwise.
                    comptime validation_rtol: Float32 = 1e-3
                    comptime validation_atol: Float32 = 1e-4
                    var max_abs_diff = Float32(0)
                    for offset in range(M * N):
                        var expected = reference_host[offset]
                        var actual = computed_host[offset]
                        if not isfinite(expected) or not isfinite(actual):
                            raise Error(
                                (
                                    "Non-finite NVIDIA tensor-core validation"
                                    " value at element "
                                ),
                                offset,
                            )
                        var difference = abs(actual - expected)
                        if (
                            difference
                            > validation_atol + validation_rtol * abs(expected)
                        ):
                            raise Error(
                                (
                                    "NVIDIA tensor-core validation mismatch at"
                                    " element "
                                ),
                                offset,
                            )
                        if difference > max_abs_diff:
                            max_abs_diff = difference
                    print("NVIDIA full reference check:", M * N)
                    print(
                        "NVIDIA full reference max diff bits:",
                        bitcast[.uint32, 1](max_abs_diff),
                    )
                else:
                    # `compare_equal` takes `TileTensor`; pass the reference and
                    # computed views directly.
                    var print_results = True
                    compare_equal[output.dtype, type_of(reference).LayoutType](
                        reference, out_tt, print_results
                    )

                    gpu_ctx.synchronize()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def naive_tensor[
    input_type: DType,
    output_type: DType,
    layout_a: TensorLayout,
    layout_b: TensorLayout,
    layout_c: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: TileTensor[input_type, layout_a, MutAnyOrigin],
    B: TileTensor[input_type, layout_b, MutAnyOrigin],
    C: TileTensor[output_type, layout_c, MutAnyOrigin],
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
    comptime M = C.static_shape[0]  # Number of rows in matrix C
    comptime N = C.static_shape[1]  # Number of columns in matrix C
    comptime K = A.static_shape[1]  # Number of columns in matrix A

    # Calculate thread configuration from compile-time constants
    comptime NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
    comptime NUM_THREADS = NUM_WARPS * WARP_SIZE
    comptime simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    var warp_y, warp_x = divmod(warp_id(), BN // MMA_N)

    # Get the warp tile of the output matrix C
    var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[
        MMA_M, MMA_N
    ](warp_y, warp_x)

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    var mma_op = TensorCore[
        output_type, input_type, Index(MMA_M, MMA_N, MMA_K)
    ]()

    # Calculate correct accumulator fragment size based on MMA configuration
    comptime frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate only small register tile for accumulating partial results
    var c_reg = stack_allocation[dtype=output_type, address_space=.LOCAL](
        row_major[1, frag_size]()
    ).fill(0)

    # Naive approach: Load directly from global memory for each tensor core operation
    # No intermediate tile caching - simpler but less efficient
    for k_i in range(ceildiv(K, BK)):
        # Get the tiles of A and B for the current iteration
        var A_block_tile = A.tile[BM, BK](block_idx.y, k_i)
        var B_block_tile = B.tile[BK, BN](k_i, block_idx.x)

        comptime if current_accelerator().is_nvidia_gpu():
            comptime for mma_k in range(BK // MMA_K):
                var A_mma_tile = A_block_tile.tile[MMA_M, MMA_K](warp_y, mma_k)
                var B_mma_tile = B_block_tile.tile[MMA_K, MMA_N](mma_k, warp_x)
                var a_reg = mma_op.load_a(A_mma_tile)
                var b_reg = mma_op.load_b(B_mma_tile)
                c_reg.copy_from(mma_op.mma_op(a_reg, b_reg, c_reg))
        else:
            # Get the warp tiles directly from global memory (naive approach)
            var A_warp_tile = A_block_tile.tile[MMA_M, MMA_K](warp_y, 0)
            var B_warp_tile = B_block_tile.tile[MMA_K, MMA_N](0, warp_x)

            # Load fragments directly from global memory
            var a_reg = mma_op.load_a(A_warp_tile)
            var b_reg = mma_op.load_b(B_warp_tile)
            c_reg.copy_from(mma_op.mma_op(a_reg, b_reg, c_reg))
    comptime if current_accelerator().is_nvidia_gpu():
        mma_op.store_d(C_warp_tile, c_reg)
    else:
        # Write the final accumulated results to the output matrix (f32 -> f32)
        # Manual store: copy register values directly to global memory
        comptime warp_layout = row_major[MMA_M // frag_size, MMA_N]()

        var dst = C_warp_tile.vectorize[4, 1]().distribute[warp_layout](
            lane_id()
        )
        dst.copy_from(c_reg.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def basic_shared_mem[
    input_type: DType,
    output_type: DType,
    layout_a: TensorLayout,
    layout_b: TensorLayout,
    layout_c: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: TileTensor[input_type, layout_a, MutAnyOrigin],
    B: TileTensor[input_type, layout_b, MutAnyOrigin],
    C: TileTensor[output_type, layout_c, MutAnyOrigin],
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
    comptime M = C.static_shape[0]  # Number of rows in matrix C
    comptime N = C.static_shape[1]  # Number of columns in matrix C
    comptime K = A.static_shape[1]  # Number of columns in matrix A

    # Calculate thread configuration from compile-time constants
    comptime NUM_WARPS = (BM // MMA_M) * (BN // MMA_N)
    comptime NUM_THREADS = NUM_WARPS * WARP_SIZE
    comptime simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    var warp_y, warp_x = divmod(warp_id(), BN // MMA_N)

    # Get the warp tile of the output matrix C
    var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[
        MMA_M, MMA_N
    ](warp_y, warp_x)

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    var mma_op = TensorCore[
        output_type, input_type, Index(MMA_M, MMA_N, MMA_K)
    ]()

    # Allocate shared memory for tiles of A and B
    var A_sram_tile = stack_allocation[dtype=input_type, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var B_sram_tile = stack_allocation[dtype=input_type, address_space=.SHARED](
        row_major[BK, BN]()
    )

    # Calculate correct accumulator fragment size based on MMA configuration
    comptime frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    var c_reg = stack_allocation[dtype=output_type, address_space=.LOCAL](
        row_major[1, frag_size]()
    ).fill(0)

    # Iterate over tiles of A and B in the K dimension
    for k_i in range(ceildiv(K, BK)):
        # Use separate optimized thread layouts for A and B tiles
        comptime load_a_layout = row_major[NUM_THREADS // BK, BK]()
        comptime load_b_layout = row_major[BK, NUM_THREADS // BK]()

        # Get the tiles of A and B for the current iteration
        var A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles using properly sized thread layouts to avoid out-of-bounds access
        copy_dram_to_sram[thread_layout=load_a_layout](A_sram_tile, A_dram_tile)
        copy_dram_to_sram[thread_layout=load_b_layout](B_sram_tile, B_dram_tile)
        barrier()  # Synchronize after loading tiles

        comptime if current_accelerator().is_nvidia_gpu():
            comptime for mma_k in range(BK // MMA_K):
                var A_mma_tile = A_sram_tile.tile[MMA_M, MMA_K](warp_y, mma_k)
                var B_mma_tile = B_sram_tile.tile[MMA_K, MMA_N](mma_k, warp_x)
                var a_reg = mma_op.load_a(A_mma_tile)
                var b_reg = mma_op.load_b(B_mma_tile)
                c_reg.copy_from(mma_op.mma_op(a_reg, b_reg, c_reg))
        else:
            # Get the warp tiles of A and B from shared memory
            var A_warp_tile = A_sram_tile.tile[MMA_M, MMA_K](warp_y, 0)
            var B_warp_tile = B_sram_tile.tile[MMA_K, MMA_N](0, warp_x)

            # Load fragments
            var a_reg = mma_op.load_a(A_warp_tile)
            var b_reg = mma_op.load_b(B_warp_tile)
            c_reg.copy_from(mma_op.mma_op(a_reg, b_reg, c_reg))
        comptime if current_accelerator().is_nvidia_gpu():
            # All warps must finish reading before the next shared tile is loaded.
            barrier()

    comptime if current_accelerator().is_nvidia_gpu():
        mma_op.store_d(C_warp_tile, c_reg)
    else:
        # Write the final accumulated results to the output matrix (f32 -> f32)
        # Manual store: copy register values directly to global memory
        comptime warp_layout = row_major[MMA_M // frag_size, MMA_N]()

        var dst = C_warp_tile.vectorize[4, 1]().distribute[warp_layout](
            lane_id()
        )
        dst.copy_from(c_reg.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def multi_block_tiled[
    input_type: DType,
    output_type: DType,
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
    A: TileTensor[input_type, layout_a, MutAnyOrigin],
    B: TileTensor[input_type, layout_b, MutAnyOrigin],
    C: TileTensor[output_type, layout_c, MutAnyOrigin],
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
    comptime M = C.static_shape[0]  # Number of rows in matrix C
    comptime N = C.static_shape[1]  # Number of columns in matrix C
    comptime K = A.static_shape[1]  # Number of columns in matrix A

    # Calculate thread configuration from compile-time constants
    comptime NUM_WARPS = (BM // WM) * (BN // WN)
    comptime NUM_THREADS = NUM_WARPS * WARP_SIZE
    comptime simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    var warp_y, warp_x = divmod(
        warp_id(),
        BN // WN if current_accelerator().is_nvidia_gpu() else BN // MMA_N,
    )

    # Get the warp tile of the output matrix C
    var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0
    ), "Warp tile should be an integer multiple of instruction shape"

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    var mma_op = TensorCore[
        output_type, input_type, Index(MMA_M, MMA_N, MMA_K)
    ]()

    # Allocate shared memory for tiles of A and B
    var A_sram_tile = stack_allocation[dtype=input_type, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var B_sram_tile = stack_allocation[dtype=input_type, address_space=.SHARED](
        row_major[BK, BN]()
    )

    # Calculate correct accumulator fragment size based on MMA configuration
    comptime frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    var c_reg = stack_allocation[dtype=output_type, address_space=.LOCAL](
        row_major[WM // MMA_M, (WN * frag_size) // MMA_N]()
    ).fill(0)

    # Thread layout for memory transfers
    comptime load_layout = row_major[16, NUM_THREADS // 16]()

    # Iterate over tiles of A and B in the K dimension
    for k_i in range(ceildiv(K, BK)):
        # Get the tiles of A and B for the current iteration
        var A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles using non-vectorized synchronous copy (working version)
        copy_dram_to_sram[thread_layout=load_layout](A_sram_tile, A_dram_tile)
        copy_dram_to_sram[thread_layout=load_layout](B_sram_tile, B_dram_tile)
        barrier()  # Synchronize after loading tiles

        # Get the warp tiles of A and B from shared memory
        var A_warp_tile = A_sram_tile.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_sram_tile.tile[BK, WN](0, warp_x)

        # Iterate over the elements in the K dimension within the tiles
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    # Get the MMA tiles directly from shared memory
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](
                        mma_m, mma_k
                    )
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](
                        mma_k, mma_n
                    )

                    var c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)
                    comptime if current_accelerator().is_nvidia_gpu():
                        var a_reg = mma_op.load_a(A_mma_tile)
                        var b_reg = mma_op.load_b(B_mma_tile)
                        c_reg_m_n.copy_from(
                            mma_op.mma_op(a_reg, b_reg, c_reg_m_n)
                        )
                    else:
                        # Load fragments
                        var a_reg = mma_op.load_a(A_mma_tile)
                        var b_reg = mma_op.load_b(B_mma_tile)

                        # Perform MMA operation using f32 accumulator
                        c_reg_m_n.copy_from(
                            mma_op.mma_op(a_reg, b_reg, c_reg_m_n)
                        )
        comptime if current_accelerator().is_nvidia_gpu():
            # All warps must finish reading before the next shared tile is loaded.
            barrier()

    # Write the final accumulated results to the output matrix (f32 -> f32)
    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)

            # Manual store: copy register values directly to global memory
            comptime if current_accelerator().is_nvidia_gpu():
                mma_op.store_d(C_mma_tile, c_reg_m_n)
            else:
                comptime warp_layout = row_major[MMA_M // frag_size, MMA_N]()

                var dst = C_mma_tile.vectorize[4, 1]().distribute[warp_layout](
                    lane_id()
                )
                dst.copy_from(c_reg_m_n.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def scheduler_hints[
    input_type: DType,
    output_type: DType,
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
    A: TileTensor[input_type, layout_a, MutAnyOrigin],
    B: TileTensor[input_type, layout_b, MutAnyOrigin],
    C: TileTensor[output_type, layout_c, MutAnyOrigin],
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
    comptime M = C.static_shape[0]  # Number of rows in matrix C
    comptime N = C.static_shape[1]  # Number of columns in matrix C
    comptime K = A.static_shape[1]  # Number of columns in matrix A

    # Calculate thread configuration from compile-time constants
    comptime NUM_WARPS = (BM // WM) * (BN // WN)
    comptime NUM_THREADS = NUM_WARPS * WARP_SIZE
    comptime simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    var warp_y, warp_x = divmod(
        warp_id(),
        BN // WN if current_accelerator().is_nvidia_gpu() else BN // MMA_N,
    )

    # Get the warp tile of the output matrix C
    var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0
    ), "Warp tile should be an integer multiple of instruction shape"

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    var mma_op = TensorCore[
        output_type, input_type, Index(MMA_M, MMA_N, MMA_K)
    ]()

    # Allocate single set of shared memory buffers (single buffering to fit memory limit)
    var A_sram_tile = stack_allocation[dtype=input_type, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var B_sram_tile = stack_allocation[dtype=input_type, address_space=.SHARED](
        row_major[BK, BN]()
    )

    # Calculate correct accumulator fragment size based on MMA configuration
    comptime frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    var c_reg = stack_allocation[dtype=output_type, address_space=.LOCAL](
        row_major[WM // MMA_M, (WN * frag_size) // MMA_N]()
    ).fill(0)

    # Thread layout for memory transfers
    comptime load_layout = row_major[16, NUM_THREADS // 16]()

    # Simplified single-buffer pipeline (similar to basic_shared_mem but with AMD scheduling)
    for k_i in range(ceildiv(K, BK)):
        # Get the tiles of A and B for the current iteration
        var A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        # Load tiles using synchronous copy (single buffering)
        copy_dram_to_sram[thread_layout=load_layout](A_sram_tile, A_dram_tile)
        copy_dram_to_sram[thread_layout=load_layout](B_sram_tile, B_dram_tile)
        barrier()  # Synchronize after loading tiles

        # Schedule barrier after loading
        comptime if current_accelerator().is_amd_gpu():
            amd_schedule_barrier()

        # Get the warp tiles from shared memory
        var A_warp_tile = A_sram_tile.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_sram_tile.tile[BK, WN](0, warp_x)

        # Perform MMA operations on current tile with AMD scheduling hints
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    # Get the MMA tiles from shared memory
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](
                        mma_m, mma_k
                    )
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](
                        mma_k, mma_n
                    )

                    var c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)
                    comptime if current_accelerator().is_nvidia_gpu():
                        var a_reg = mma_op.load_a(A_mma_tile)
                        var b_reg = mma_op.load_b(B_mma_tile)
                        c_reg_m_n.copy_from(
                            mma_op.mma_op(a_reg, b_reg, c_reg_m_n)
                        )
                    else:
                        # Load fragments and perform MMA
                        var a_reg = mma_op.load_a(A_mma_tile)
                        var b_reg = mma_op.load_b(B_mma_tile)

                        # Manual accumulation for 32x32x8
                        c_reg_m_n.copy_from(
                            mma_op.mma_op(a_reg, b_reg, c_reg_m_n)
                        )

        comptime if current_accelerator().is_nvidia_gpu():
            # All warps must finish reading before the next shared tile is loaded.
            barrier()

        # Add AMD scheduling hints between tiles
        comptime if current_accelerator().is_amd_gpu():
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
    comptime if current_accelerator().is_amd_gpu():
        amd_schedule_barrier()

    # === OUTPUT PHASE ===
    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_tile = c_reg.tile[1, frag_size](mma_m, mma_n)

            comptime if current_accelerator().is_nvidia_gpu():
                mma_op.store_d(C_mma_tile, c_reg_tile)
            else:
                comptime warp_layout = row_major[MMA_M // frag_size, MMA_N]()

                var dst = C_mma_tile.vectorize[4, 1]().distribute[warp_layout](
                    lane_id()
                )
                dst.copy_from(c_reg_tile.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def double_buffer[
    input_type: DType,
    output_type: DType,
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
    A: TileTensor[input_type, layout_a, MutAnyOrigin],
    B: TileTensor[input_type, layout_b, MutAnyOrigin],
    C: TileTensor[output_type, layout_c, MutAnyOrigin],
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
    comptime M = C.static_shape[0]  # Number of rows in matrix C
    comptime N = C.static_shape[1]  # Number of columns in matrix C
    comptime K = A.static_shape[1]  # Number of columns in matrix A

    # Calculate thread configuration from compile-time constants
    comptime NUM_WARPS = (BM // WM) * (BN // WN)
    comptime NUM_THREADS = NUM_WARPS * WARP_SIZE
    comptime simd_width = simd_width_of[input_type]()

    # Calculate warp tile coordinates within the block
    var warp_y, warp_x = divmod(
        warp_id(),
        BN // WN if current_accelerator().is_nvidia_gpu() else BN // MMA_N,
    )

    # Get the warp tile of the output matrix C
    var C_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_y, warp_x
    )

    # Ensure warp tile dimensions are multiples of instruction shape
    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and K % MMA_K == 0
    ), "Warp tile should be an integer multiple of instruction shape"

    # Create tensor core operation object with mixed precision: f16 input, f32 accumulator
    var mma_op = TensorCore[
        output_type, input_type, Index(MMA_M, MMA_N, MMA_K)
    ]()

    # Allocate two sets of shared memory buffers for double buffering
    var A_sram_buffer_0 = stack_allocation[
        dtype=input_type, address_space=.SHARED
    ](row_major[BM, BK]())
    var A_sram_buffer_1 = stack_allocation[
        dtype=input_type, address_space=.SHARED
    ](row_major[BM, BK]())
    var B_sram_buffer_0 = stack_allocation[
        dtype=input_type, address_space=.SHARED
    ](row_major[BK, BN]())
    var B_sram_buffer_1 = stack_allocation[
        dtype=input_type, address_space=.SHARED
    ](row_major[BK, BN]())

    # Calculate correct accumulator fragment size based on MMA configuration
    comptime frag_size = MMA_M * MMA_N // WARP_SIZE

    # Allocate register tile for accumulating partial results
    var c_reg = stack_allocation[dtype=output_type, address_space=.LOCAL](
        row_major[WM // MMA_M, (WN * frag_size) // MMA_N]()
    ).fill(0)
    # Thread layout for memory transfers
    comptime load_layout = row_major[NUM_THREADS // 8, 8]()

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
            # Vectorize to create 4-byte elements (2 x f16 = 4 bytes)
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
            # Commit the issued async copies as a group so the matching
            # `async_copy_wait_all()` below can synchronize on them.
            async_copy_commit_group()

        # === COMPUTE PHASE: Use current buffers for computation ===
        # Get the warp tiles from the current compute buffers
        var A_warp_tile = A_compute_buffer.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_compute_buffer.tile[BK, WN](0, warp_x)

        # Perform MMA operations on current tile
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    # Get the MMA tiles from shared memory
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](
                        mma_m, mma_k
                    )
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](
                        mma_k, mma_n
                    )

                    var c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)
                    comptime if current_accelerator().is_nvidia_gpu():
                        var a_reg = mma_op.load_a(A_mma_tile)
                        var b_reg = mma_op.load_b(B_mma_tile)
                        c_reg_m_n.copy_from(
                            mma_op.mma_op(a_reg, b_reg, c_reg_m_n)
                        )
                    else:
                        # Load fragments and perform MMA
                        var a_reg = mma_op.load_a(A_mma_tile)
                        var b_reg = mma_op.load_b(B_mma_tile)

                        # Manual accumulation: bypass TensorCore store_d
                        # Copy result directly to register tile
                        c_reg_m_n.copy_from(
                            mma_op.mma_op(a_reg, b_reg, c_reg_m_n)
                        )

        # === SYNC: Ensure next iteration's data is ready ===
        if next_k < k_iterations:
            async_copy_wait_all()  # Wait for async loads to complete
            barrier()  # Ensure all threads see the loaded data

    # === OUTPUT PHASE: Write results to global memory manually ===
    # Bypass TensorCore store_d and use direct register-to-memory copy
    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, frag_size](mma_m, mma_n)

            # Manual store: copy register values directly to global memory
            comptime if current_accelerator().is_nvidia_gpu():
                mma_op.store_d(C_mma_tile, c_reg_m_n)
            else:
                comptime warp_layout = row_major[MMA_M // frag_size, MMA_N]()

                var dst = C_mma_tile.vectorize[4, 1]().distribute[warp_layout](
                    lane_id()
                )
                dst.copy_from(c_reg_m_n.vectorize[1, 4]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def mma_tile_buffers[
    input_type: DType,
    output_type: DType,
    layout_a: TensorLayout,
    layout_b: TensorLayout,
    layout_c: TensorLayout,
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
    A: TileTensor[input_type, layout_a, MutAnyOrigin],
    B: TileTensor[input_type, layout_b, MutAnyOrigin],
    C: TileTensor[output_type, layout_c, MutAnyOrigin],
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
    comptime transpose_b = True
    comptime assert (
        transpose_b
    ), "Transpose b must be true for this implementation"
    comptime assert (
        WK == BK
    ), "This implementation does not split K across warps"
    comptime assert MMA_M == MMA_N, "The MFMA shape is square"

    var M = Int(A.dim[0]())
    var N = Int(B.dim[0]())
    var K = Int(B.dim[1]())

    comptime accum_type = DType.float32
    comptime simd_width = simd_width_of[input_type]()
    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    comptime num_threads = num_warps_m * num_warps_n * WARP_SIZE

    # MMA instruction tiling
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime c_frag_size = MMA_M * MMA_N // WARP_SIZE

    # K dimension tiling: each k-tile is one 16-byte vector per lane wide.
    comptime k_group_size = 16 // simd_width
    comptime k_tile_size = MMA_K * k_group_size
    comptime num_k_tiles = WK // k_tile_size

    var warp_m, warp_n = udivmod(warp_id(), num_warps_n)

    # Shared memory holds the BK-wide block of A and B as a sequence of
    # k-tile blocks, each `rows x k_tile_size` and contiguous, so one lane's
    # fragment for an MMA is a single vector read. The swizzle spreads the
    # per-lane vector columns of each block across LDS banks.
    comptime swizzle = Optional[Swizzle](
        Swizzle(log2_floor(WARP_SIZE // MMA_M) + 1, 0, 1)
    )
    var a_smem = stack_allocation[input_type, address_space=.SHARED](
        row_major[BM, BK]()
    )
    var b_smem = stack_allocation[input_type, address_space=.SHARED](
        row_major[BN, BK]()
    )

    # Global memory reaches shared memory through registers: every thread
    # loads one vector per row of its thread-layout slice with a buffer load,
    # then writes it into the blocked shared layout.
    comptime load_thread_cols = BK // simd_width
    comptime load_thread_rows = num_threads // load_thread_cols
    comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
    var a_load_reg = stack_allocation[input_type, address_space=.LOCAL](
        row_major[1, BM * BK // num_threads]()
    )
    var b_load_reg = stack_allocation[input_type, address_space=.LOCAL](
        row_major[1, BN * BK // num_threads]()
    )
    var a_loader = RegTileLoader[input_type, load_layout](
        A.tile[BM, BK](block_idx.y, 0), bounds_from=A
    )
    var b_loader = RegTileLoader[input_type, load_layout](
        B.tile[BN, BK](block_idx.x, 0), bounds_from=B
    )
    comptime LdsWriter = RegTileWriterLDS[
        load_layout, swizzle=swizzle, num_threads=num_threads
    ]

    # The MMA operator owns the A and B fragment registers and the
    # accumulator, and reads fragments back out of the swizzled shared tiles.
    var mma_op = MmaOp[
        out_type=accum_type,
        in_type=input_type,
        shape=IndexList[3](MMA_M, MMA_N, MMA_K),
        k_group_size=k_group_size,
        num_k_tiles=num_k_tiles,
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        swizzle=swizzle,
    ]()

    var k_counter = 0

    @inline(.always)
    def load_tiles_from_dram() {mut k_counter, imm}:
        a_loader.load(
            a_load_reg,
            A.tile[BM, BK](block_idx.y, k_counter).vectorize[1, simd_width](),
        )
        b_loader.load(
            b_load_reg,
            B.tile[BN, BK](block_idx.x, k_counter).vectorize[1, simd_width](),
        )
        k_counter += 1

    @inline(.always)
    def copy_tiles_to_shared() {imm}:
        LdsWriter.copy_blocked[k_tile_size](a_smem, a_load_reg)
        LdsWriter.copy_blocked[k_tile_size](b_smem, b_load_reg)

    @inline(.always)
    def load_tiles_from_shared[k_tile_idx: Int]() {imm}:
        var a_block = TileTensor(
            a_smem.ptr.unsafe_offset(k_tile_idx * BM * k_tile_size),
            row_major[BM, k_tile_size](),
        )
        var b_block = TileTensor(
            b_smem.ptr.unsafe_offset(k_tile_idx * BN * k_tile_size),
            row_major[BN, k_tile_size](),
        )
        mma_op.load_frag[k_tile_idx](
            a_block.tile[WM, k_tile_size](warp_m, 0),
            b_block.tile[WN, k_tile_size](warp_n, 0),
        )

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

    # Stage 3: Main computation loop with pipelined operations
    var k_iterations = K // BK
    if k_iterations > 2:
        for _ in range(2, k_iterations):
            comptime for k_tile_idx in range(1, num_k_tiles):
                load_tiles_from_shared[k_tile_idx]()
            mma_op.mma[0]()

            barrier()
            copy_tiles_to_shared()
            load_tiles_from_dram()

            comptime for k_tile_idx in range(1, num_k_tiles):
                mma_op.mma[k_tile_idx]()

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

    # Stage 4: Process the last two tiles
    amd_schedule_barrier()
    comptime for k_tile_idx in range(1, num_k_tiles):
        load_tiles_from_shared[k_tile_idx]()
    barrier()
    copy_tiles_to_shared()
    comptime for k_tile_idx in range(num_k_tiles):
        mma_op.mma[k_tile_idx]()
    amd_schedule_barrier()

    barrier()
    comptime for k_tile_idx in range(num_k_tiles):
        load_tiles_from_shared[k_tile_idx]()
    comptime for k_tile_idx in range(num_k_tiles):
        mma_op.mma[k_tile_idx]()
    amd_schedule_barrier()

    # Stage 5: Write the accumulated result back to global memory, one MMA
    # tile per store so the writer can apply the MFMA register permutation.
    var c_reg = mma_op.accum_tile()
    var c_writer = RegTileWriter[output_type, MMA_M, WARP_SIZE // MMA_M](C)
    var c_warp_tile = C.tile[BM, BN](block_idx.y, block_idx.x).tile[WM, WN](
        warp_m, warp_n
    )
    comptime vec_width = 4 if MMA_M == 32 else c_frag_size
    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            c_writer.store[mfma32=MMA_M == 32](
                c_warp_tile.tile[MMA_M, MMA_N](m_mma, n_mma).vectorize[
                    1, vec_width
                ](),
                c_reg.tile[1, c_frag_size](m_mma, n_mma),
            )
