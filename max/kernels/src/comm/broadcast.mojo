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

from collections import InlineArray
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

from sys import align_of, is_amd_gpu, simd_width_of, size_of
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

# On AMD Systems, loads from GLOBAL addressspace give better performance.
comptime _target_address_space = AddressSpace.GLOBAL if is_amd_gpu() else AddressSpace.GENERIC


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
fn broadcast_pull_1stage_kernel[
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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn broadcast_pull_2stage_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    pdl_level: PDLLevel = PDLLevel(),
](
    result: NDBuffer[dtype, rank, MutAnyOrigin],
    root_input_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
    root: Int,
):
    """Two-stage broadcast: scatter from root, then allgather among non-root GPUs.

    Stage 1 (Scatter): Root's data is split into (ngpus-1) chunks. Each non-root
    GPU reads its assigned chunk directly from root's input buffer and writes
    it to its signal payload.

    Stage 2 (Allgather): Each non-root GPU gathers the remaining chunks from
    other non-root GPUs' signal payloads. Root already has all data.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        BLOCK_SIZE: Number of threads per block.
        pdl_level: Control PDL behavior for the kernel.

    Args:
        result: Output buffer for broadcast result.
        root_input_ptr: Pointer to root's input data (all GPUs read from this).
        rank_sigs: Signal pointers for synchronization.
            IMPORTANT: Signal pointers have trailing buffers for communication.
        num_elements: Number of elements to broadcast.
        my_rank: Current GPU rank.
        root: Root GPU rank (source of broadcast).
    """
    var my_sig = rank_sigs[my_rank]

    # Thread indexing
    var global_tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK_SIZE

    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    # Partition data among (ngpus-1) non-root GPUs
    comptime num_participants = ngpus - 1
    var part_size = (
        num_elements // simd_width // num_participants
    ) * simd_width
    var thr_local_start = global_tid * simd_width
    var elem_stride = stride * simd_width  # Stride in elements, not threads

    @parameter
    if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Get payload buffers from signal pointers (skip Signal header)
    # These are used as scratch space for the scatter-gather pattern
    var payloads = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)

    @parameter
    for i in range(ngpus):
        payloads[i] = (
            rank_sigs[i].address_space_cast[AddressSpace.GENERIC]() + 1
        ).bitcast[Scalar[dtype]]()

    # Map my_rank to participant index (0-indexed among non-root GPUs)
    # For root=0: GPU1->participant0, GPU2->participant1, etc.
    # For root=k: GPUs 0..k-1 map to participants 0..k-1,
    #             GPUs k+1..n-1 map to participants k..n-2
    var my_participant_idx = my_rank - 1 if my_rank > root else my_rank
    var is_root = my_rank == root

    # === Stage 1: Scatter from root ===
    # Initial barrier to ensure all GPUs are ready
    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    if is_root:
        # Root: copy input directly to output (root already has all data)
        var num_simd_vectors = num_elements // simd_width
        for idx in range(global_tid, num_simd_vectors, stride):
            var elem_idx = idx * simd_width
            var data = root_input_ptr.address_space_cast[
                _target_address_space
            ]().load[width=simd_width, alignment=alignment, invariant=True](
                elem_idx
            )
            result.store[width=simd_width, alignment=alignment](
                result.get_nd_index(elem_idx), data
            )
    else:
        # Non-root: read my chunk from root's input
        # Write to BOTH my payload (for others to read) AND my output (for myself)
        var rank_start = my_participant_idx * part_size
        var rank_end = (
            num_elements if my_participant_idx
            == num_participants - 1 else rank_start + part_size
        )
        var my_payload = payloads[my_rank]

        for idx in range(rank_start + thr_local_start, rank_end, elem_stride):
            var data = root_input_ptr.address_space_cast[
                _target_address_space
            ]().load[width=simd_width, alignment=alignment, invariant=True](idx)
            # Store to my payload (for other GPUs to read in Stage 2)
            my_payload.address_space_cast[_target_address_space]().store[
                alignment=alignment
            ](idx - rank_start, data)
            # Also store directly to my output (skip redundant copy in Stage 2)
            result.store[width=simd_width, alignment=alignment](
                result.get_nd_index(idx), data
            )

    # Barrier with memory fence to ensure scatter is complete
    _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
        rank_sigs, my_sig, my_rank
    )

    # === Stage 2: Gather remaining chunks ===
    # Non-root GPUs gather the chunks they don't have from other non-root GPUs.
    # Use round-robin access pattern to balance NVLink traffic.
    if not is_root:
        # Gather (ngpus-2) chunks from other non-root GPUs
        # Swap loop order: iterate elements in outer loop, peers in inner loop
        # This interleaves reads across all peers for better parallelism

        # Calculate max chunk size (last chunk may have remainder)
        var max_chunk_size = num_elements - (num_participants - 1) * part_size

        for idx in range(thr_local_start, max_chunk_size, elem_stride):

            @parameter
            for offset in range(1, num_participants):
                # Round-robin: each GPU starts gathering from a different peer
                var participant_idx = (
                    my_participant_idx + offset
                ) % num_participants

                var chunk_start = participant_idx * part_size
                var chunk_end = (
                    num_elements if participant_idx
                    == num_participants - 1 else chunk_start + part_size
                )

                # Map participant index to GPU rank
                var src_rank = (
                    participant_idx if participant_idx
                    < root else participant_idx + 1
                )
                var src_payload = payloads[src_rank]

                # Check if idx is within this chunk's bounds
                if idx < (chunk_end - chunk_start):
                    var data = src_payload.address_space_cast[
                        _target_address_space
                    ]().load[
                        width=simd_width, alignment=alignment, invariant=True
                    ](
                        idx
                    )
                    # Write to final position in result
                    result.store[width=simd_width, alignment=alignment](
                        result.get_nd_index(chunk_start + idx), data
                    )

    # Final barrier to ensure all GPUs complete before returning
    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


fn _should_use_2stage[ngpus: Int](num_bytes: Int) -> Bool:
    """Determine if 2-stage broadcast should be used based on GPU count and size.

    Crossover points determined empirically:
    - 2 GPUs: Always use 1-stage (2-stage has no benefit)
    - 4 GPUs: 2-stage wins for >= 6 MiB
    - 8 GPUs: 2-stage wins for >= 2 MiB
    """

    @parameter
    if ngpus == 2:
        return False
    elif ngpus == 4:
        return num_bytes >= 6 * 1024 * 1024  # 6 MiB
    elif ngpus == 8:
        return num_bytes >= 2 * 1024 * 1024  # 2 MiB
    else:
        # For other GPU counts, use 2-stage for large sizes as a reasonable default
        return num_bytes >= 4 * 1024 * 1024  # 4 MiB


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
        # Dispatch between 1-stage and 2-stage based on size and GPU count
        var num_bytes = input_buffer.bytecount()
        if _should_use_2stage[ngpus](num_bytes):
            # Use 2-stage for large data with multiple GPUs
            broadcast_2stage[
                dtype=dtype, rank=rank, ngpus=ngpus, pdl_level=pdl_level
            ](
                input_buffer,
                output_buffer,
                rank_sigs,
                ctx,
                root,
                _max_num_blocks,
            )
        else:
            comptime bcast_kernel = broadcast_pull_1stage_kernel[
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


@parameter
fn broadcast_2stage[
    dtype: DType,
    rank: Int,
    //,
    ngpus: Int,
    pdl_level: PDLLevel = PDLLevel(),
](
    input_buffer: NDBuffer[dtype, rank, ImmutAnyOrigin],
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    root: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Two-stage broadcast: scatter from root, then allgather among non-root GPUs.

    This algorithm achieves better bandwidth than ring broadcast by:
    1. Stage 1 (Scatter): Root sends 1/(ngpus-1) of data to each non-root GPU
       in parallel, utilizing root's full outbound NVLink bandwidth.
    2. Stage 2 (Allgather): Non-root GPUs gather from each other in parallel,
       with each GPU reading (ngpus-2) chunks from other non-root GPUs.

    The root GPU already has all data, so it only participates in stage 1
    (copying input to output) and the barriers.

    IMPORTANT: Signal buffers must be sized to hold at least:
        size_of(Signal) + (num_elements / (ngpus-1)) * size_of(dtype)
    This is the payload space needed for each GPU's chunk.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        pdl_level: Control PDL behavior for the kernel.

    Args:
        input_buffer: Input buffer (only root's is read, but all must be valid).
        output_buffer: Output buffer for THIS GPU.
        rank_sigs: Signal pointers with payload space for staging.
        ctx: Device context for THIS GPU.
        root: Root GPU rank (source of broadcast data).
        _max_num_blocks: Optional maximum number of thread blocks.
    """
    var my_rank: Int = Int(ctx.id())

    var num_elements = output_buffer.num_elements()
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    # Do nothing if there are no elements.
    if num_elements == 0:
        return

    debug_assert(
        output_buffer.num_elements() == input_buffer.num_elements(),
        "Buffer shapes don't match",
    )

    if num_elements % simd_width != 0:
        raise Error(
            "non SIMD-width multiple number of elements unsupported by"
            " broadcast_2stage"
        )

    comptime BLOCK_SIZE = 256
    # Limit blocks - tuning parameter
    comptime MAX_BLOCKS = 512

    # Grid size: each GPU processes 1/(ngpus-1) of the elements in scatter phase
    var grid_size = min(
        _max_num_blocks.or_else(MAX_BLOCKS),
        ceildiv(num_elements // (simd_width * (ngpus - 1)), BLOCK_SIZE),
    )

    comptime kernel = broadcast_pull_2stage_kernel[
        dtype,
        rank,
        ngpus,
        BLOCK_SIZE=BLOCK_SIZE,
        pdl_level=pdl_level,
    ]

    ctx.enqueue_function[kernel, kernel](
        output_buffer,
        input_buffer.data,
        rank_sigs,
        num_elements,
        my_rank,
        root,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )
