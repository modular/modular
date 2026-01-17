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
"""Multi-GPU broadcast kernel implementation."""

from math import ceildiv
from gpu.host import DeviceContext, get_gpu_target
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)

from sys import simd_width_of
from buffer import NDBuffer
from gpu.memory import Consistency, multimem_st
from gpu.intrinsics import Scope
from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    can_enable_p2p,
)
from .device_query import _dispatch_max_num_blocks, get_sm_version

from utils import IndexList, StaticTuple


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn broadcast_multimem_kernel[
    dtype: DType,
    rank: Int,
    BLOCK_SIZE: Int,
    ngpus: Int,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
    pdl_level: PDLLevel = PDLLevel(),
](
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    input_buffer: NDBuffer[dtype, rank, ImmutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    root: Int,
):
    """Broadcast kernel using multimem.st for multicast writes.

    Root GPU writes to multicast address, data appears on all GPUs.
    Only root performs the stores; other GPUs just participate in barriers.
    """
    var my_sig = rank_sigs[my_rank]

    # --- Thread Indexing ---
    var global_tid = Int(global_idx.x)
    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = Int(grid_dim.x) * BLOCK_SIZE

    @parameter
    if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    var num_elements = output_buffer.num_elements()
    var num_simd_vectors = num_elements // simd_width

    # Only root GPU performs the multicast store
    if my_rank == root:
        # Get multicast output pointer
        var out_ptr = output_buffer.data.address_space_cast[
            AddressSpace.GLOBAL
        ]()

        # Grid-strided loop to cover all elements (vectorized)
        for idx in range(global_tid, num_simd_vectors, stride):
            var elem_idx = idx * simd_width
            # Load from input buffer
            var data = input_buffer.load[width=simd_width](
                input_buffer.get_nd_index(elem_idx)
            )
            # Store to multicast address - all GPUs receive this write
            multimem_st[
                dtype,
                simd_width=simd_width,
                scope = Scope.GPU,
                consistency = Consistency.RELAXED,
            ](out_ptr + elem_idx, data)

    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn broadcast_pull_kernel[
    dtype: DType,
    rank: Int,
    BLOCK_SIZE: Int,
    ngpus: Int,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
    pdl_level: PDLLevel = PDLLevel(),
](
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    input_buffer: NDBuffer[dtype, rank, ImmutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
):
    var my_sig = rank_sigs[my_rank]

    # --- Thread Indexing ---
    var global_tid = Int(global_idx.x)
    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = Int(grid_dim.x) * BLOCK_SIZE

    @parameter
    if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    var num_elements = output_buffer.num_elements()
    var num_simd_vectors = num_elements // simd_width

    # Grid-strided loop to cover all elements (vectorized).
    for idx in range(global_tid, num_simd_vectors, stride):
        var elem_idx = idx * simd_width
        output_buffer.store[width=simd_width](
            output_buffer.get_nd_index(elem_idx),
            input_buffer.load[width=simd_width](
                input_buffer.get_nd_index(elem_idx)
            ),
        )
    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@parameter
fn broadcast[
    dtype: DType,
    rank: Int,
    //,
    ngpus: Int,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    input_buffer: NDBuffer[dtype, rank, ImmutAnyOrigin],
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    root: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    var my_rank: Int = Int(ctx.id())

    var num_elements = output_buffer.num_elements()
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    # Do nothing if there are no elements to reduce.
    if num_elements == 0:
        return

    debug_assert(
        output_buffer.num_elements() == input_buffer.num_elements(),
        "Buffer shapes don't match",
    )

    comptime BLOCK_SIZE = 256
    # Default max blocks if not specified.
    comptime sm_version = get_sm_version()
    # TODO: _dispatch_max_num_blocks was tuned for allreduce; may need separate tuning for broadcast
    var max_num_blocks = _max_num_blocks.or_else(
        _dispatch_max_num_blocks[ngpus, sm_version](input_buffer.bytecount())
    )

    var grid_size = min(
        max_num_blocks,
        ceildiv(num_elements // simd_width, BLOCK_SIZE),
    )

    @parameter
    if use_multimem:
        comptime bcast_kernel = broadcast_multimem_kernel[
            dtype,
            rank,
            BLOCK_SIZE,
            ngpus,
            pdl_level=pdl_level,
        ]

        ctx.enqueue_function[bcast_kernel, bcast_kernel](
            output_buffer,
            input_buffer,
            rank_sigs,
            my_rank,
            root,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        comptime bcast_kernel = broadcast_pull_kernel[
            dtype,
            rank,
            BLOCK_SIZE,
            ngpus,
            pdl_level=pdl_level,
        ]

        ctx.enqueue_function[bcast_kernel, bcast_kernel](
            output_buffer,
            input_buffer,
            rank_sigs,
            my_rank,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )
