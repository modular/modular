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
"""Multi-GPU reducescatter implementation for distributed tensor reduction across GPUs.
"""

from collections import InlineArray
from collections.optional import Optional

from buffer import NDBuffer
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.primitives.grid_controls import PDLLevel
from gpu.host import DeviceContext, get_gpu_target
from gpu.memory import Consistency, ReduceOp, multimem_ld_reduce
from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from gpu.intrinsics import (
    Scope,
)
from math import ceildiv
from sys import simd_width_of, align_of, is_amd_gpu

from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)

# On AMD Systems, the loads from GLOBAL addressspace gives an improvement
# to the performance.
comptime _target_address_space = AddressSpace.GLOBAL if is_amd_gpu() else AddressSpace.GENERIC

comptime elementwise_epilogue_type = fn[
    dtype: DType, rank: Int, width: Int, *, alignment: Int
](IndexList[rank], SIMD[dtype, size=width]) capturing -> None


@always_inline
fn _load_reduce[
    dtype: DType,
    //,
    ngpus: Int,
    simd_width: Int,
    alignment: Int,
    accum_type: DType,
    *,
    use_multimem: Bool = False,
](
    elem_idx: Int,
    ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
) -> SIMD[dtype, simd_width]:
    comptime if use_multimem:
        # Multimem mode: use optimized reduction
        return multimem_ld_reduce[
            dtype,
            simd_width=simd_width,
            reduction = ReduceOp.ADD,
            scope = Scope.GPU,
            consistency = Consistency.RELAXED,
            accum_type=accum_type,
        ]((ptrs[0] + elem_idx).address_space_cast[AddressSpace.GLOBAL]())
    else:
        # Regular mode: manual accumulation
        # Initialize with first load to avoid extra zero-add operation
        var accum = (
            ptrs[0]
            .address_space_cast[_target_address_space]()
            .load[width=simd_width, alignment=alignment, invariant=True](
                elem_idx
            )
            .cast[accum_type]()
        )

        comptime for gpu_idx in range(1, ngpus):
            accum += (
                ptrs[gpu_idx]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment, invariant=True](
                    elem_idx
                )
                .cast[accum_type]()
            )

        return accum.cast[dtype]()


struct ReduceScatterConfig[
    dtype: DType,
    ngpus: Int,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
    alignment: Int = align_of[SIMD[dtype, simd_width]](),
    accum_type: DType = get_accum_type[dtype](),
](TrivialRegisterPassable):
    var stride: Int
    var part: Int
    var remainder: Int

    @always_inline
    fn __init__(
        out self,
        num_elements: Int,
        threads_per_gpu: Int,
    ):
        comptime assert Self.ngpus > 1, "ngpus must be greater than 1"
        self.stride = threads_per_gpu * Self.simd_width
        # --- Data Partitioning ---
        # Data are divided as evenly as possible amongst ngpus.
        # We consider a simd vector as the basic unit of work.
        # For ragged cases, lower ranks receive an extra simd vector.
        var num_simd_vectors = num_elements // Self.simd_width
        self.part = num_simd_vectors // Self.ngpus
        self.remainder = num_simd_vectors % Self.ngpus

    @always_inline
    fn rank_start(self, rank: Int) -> Int:
        return (rank * self.part + min(rank, self.remainder)) * Self.simd_width

    @always_inline
    fn rank_end(self, rank: Int) -> Int:
        return self.rank_start(rank + 1)

    @always_inline
    fn rank_part(self, rank: Int) -> Int:
        return (self.part + Int(rank < self.remainder)) * Self.simd_width

    @always_inline
    fn thr_local_start(self, thread_idx: UInt) -> Int:
        return Int(thread_idx) * Self.simd_width


@always_inline
fn _reduce_scatter_impl[
    dtype: DType,
    rank: Int,
    simd_width: Int,
    alignment: Int,
    accum_type: DType,
    //,
    ngpus: Int,
    *,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
](
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    out_buf: NDBuffer[dtype, rank, MutAnyOrigin],
    my_rank: Int,
    config: ReduceScatterConfig[
        dtype, ngpus, simd_width, alignment, accum_type
    ],
):
    # Grid-strided loop with vectorized reduction:
    # - Each thread processes partition elements using 128-bit accesses.
    # - Accumulates in higher precision (float32) for numerical stability.
    for idx in range(
        config.rank_start(my_rank) + config.thr_local_start(global_idx.x),
        config.rank_end(my_rank),
        config.stride,
    ):
        # float32 accumulator for numerical stability.
        var reduced_result = _load_reduce[
            ngpus,
            simd_width = config.simd_width,
            alignment = config.alignment,
            accum_type = config.accum_type,
            use_multimem=use_multimem,
        ](idx, src_ptrs)

        # Apply epilogue and store result.
        output_lambda[width = config.simd_width, alignment = config.alignment](
            out_buf.get_nd_index(idx - config.rank_start(my_rank)),
            reduced_result,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn _reducescatter_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    out_buf: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
):
    """Reduce-scatter kernel for bandwidth-bound transfers.

    This kernel implements the reduce-scatter phase where each GPU reduces
    its assigned partition from all input buffers.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: Elementwise epilogue function to apply to reduced values.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: Whether multimem optimization is enabled.

    Args:
        src_ptrs: Input buffers from all GPUs.
        out_buf: Output buffer for this GPU's partition of reduced data.
        rank_sigs: Signal pointers for synchronization.
        num_elements: Total number of elements across all GPUs.
        my_rank: Current GPU rank.
    """
    comptime num_buffers = 1 if use_multimem else ngpus
    var my_sig = rank_sigs[my_rank]

    # --- Thread Indexing ---
    var threads_per_gpu = Int(grid_dim.x) * BLOCK_SIZE

    var reduce_scatter_config = ReduceScatterConfig[dtype, ngpus](
        num_elements, threads_per_gpu
    )

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Round-robin access pattern to balance NVLink traffic across GPUs.
    var ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], num_buffers
    ](uninitialized=True)

    comptime for i in range(num_buffers):
        var target = 0 if num_buffers == 1 else (my_rank + i) % num_buffers
        ptrs[i] = src_ptrs[target]

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    _reduce_scatter_impl[
        ngpus, output_lambda=output_lambda, use_multimem=use_multimem
    ](ptrs, out_buf, my_rank, reduce_scatter_config)

    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@always_inline
fn _reducescatter_p2p[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    list_of_in_bufs: InlineArray[
        NDBuffer[dtype, rank, ImmutAnyOrigin], 1 if use_multimem else ngpus
    ],
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    max_num_blocks: Int,
    ctx: DeviceContext,
) raises:
    """
    Performs reducescatter using peer-to-peer access for a single GPU.

    This implements the reduce-scatter phase: each GPU reduces its assigned
    partition and writes the result to its output buffer.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        output_lambda: Elementwise epilogue function to apply to reduced values.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: Whether multimem optimization is enabled.

    Args:
        list_of_in_bufs: Input buffers from all GPUs (peer access required).
        output_buffer: Output buffer for this GPU's partition of reduced data.
        rank_sigs: Signal pointers for synchronization.
        max_num_blocks: Maximum number of thread blocks to launch.
        ctx: Device context for THIS GPU.
    """

    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    var num_elements = list_of_in_bufs[0].num_elements()

    # TODO(KERN-2337): generalize this check, ensure the *last-axis* % simd_width = 0
    if num_elements % simd_width != 0:
        raise Error(
            "non SIMD-width multiple number of elements unsupported by"
            " reducescatter"
        )

    # Pass a stack-allocated array of pointers to the device kernel.
    comptime num_buffers = 1 if use_multimem else ngpus
    var list_of_in_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], num_buffers
    ](uninitialized=True)

    comptime for i in range(num_buffers):
        list_of_in_ptrs[i] = list_of_in_bufs[i].data

    # Block size configuration
    comptime BLOCK_SIZE = 256

    # Define grid size - each GPU processes 1/ngpus of the elements
    var grid_size = min(
        max_num_blocks,
        ceildiv(num_elements // (simd_width * ngpus), BLOCK_SIZE),
    )

    # Define the reduce-scatter kernel
    comptime kernel = _reducescatter_kernel[
        dtype,
        rank,
        ngpus,
        BLOCK_SIZE=BLOCK_SIZE,
        output_lambda=output_lambda,
        pdl_level=pdl_level,
        use_multimem=use_multimem,
    ]

    # Launch the kernel
    ctx.enqueue_function[kernel, kernel](
        list_of_in_ptrs,
        output_buffer,
        rank_sigs,
        num_elements,
        Int(ctx.id()),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )


@parameter
fn reducescatter[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    use_multimem: Bool = False,
](
    input_buffers: InlineArray[
        NDBuffer[dtype, rank, ImmutAnyOrigin], 1 if use_multimem else ngpus
    ],
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device reducescatter operation.

    Performs a reduce-scatter across multiple GPUs: each GPU reduces its assigned
    partition from all input buffers and writes the result to its output buffer.

    This is equivalent to the reduce-scatter phase of the 2-stage allreduce algorithm.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        output_lambda: Optional elementwise epilogue function. If not provided,
            reduced values are stored directly to output_buffer.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: If True, use multimem optimization (reserved for future use).

    Args:
        input_buffers: Input buffers from all GPUs (peer access required).
            When use_multimem is False (default), expects ngpus buffers.
            When use_multimem is True, expects a single buffer.
        output_buffer: Output buffer for THIS GPU's partition of reduced data.
            Size should be approximately 1/ngpus of the input size.
        rank_sigs: Signal pointers for synchronization between GPUs.
        ctx: Device context for THIS GPU.
        _max_num_blocks: Optional maximum number of thread blocks to launch.
            If not specified, uses MAX_NUM_BLOCKS_UPPER_BOUND.

    Raises:
        Error: If P2P access is not available between GPUs.
        Error: If input buffer size is not a multiple of SIMD width.
    """
    comptime assert ngpus >= 2, "reducescatter requires at least 2 GPUs"

    # Return early if the input buffer is empty
    var num_elements = input_buffers[0].num_elements()
    if num_elements == 0:
        return

    if not is_p2p_enabled():
        raise Error("Reducescatter currently requires P2P access between GPUs")

    # Determine max number of blocks
    var max_num_blocks = (
        _max_num_blocks.value() if _max_num_blocks else MAX_NUM_BLOCKS_UPPER_BOUND
    )

    # Default epilogue: store directly to output buffer
    @always_inline
    @parameter
    @__copy_capture(output_buffer)
    fn default_output_lambda[
        _dtype: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
        output_buffer.store[width=_width, alignment=_alignment](
            rebind[IndexList[rank]](coords), rebind[SIMD[dtype, _width]](val)
        )

    comptime actual_output_lambda = default_output_lambda if not output_lambda else output_lambda.value()

    # Launch the reduce-scatter kernel via P2P
    _reducescatter_p2p[
        dtype,
        rank,
        ngpus,
        output_lambda=actual_output_lambda,
        pdl_level=pdl_level,
        use_multimem=use_multimem,
    ](
        input_buffers,
        output_buffer,
        rank_sigs,
        max_num_blocks,
        ctx,
    )
