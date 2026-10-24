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

from std.collections import Array
from std.math import align_down, ceildiv
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)

from std.sys import align_of, is_amd_gpu, simd_width_of, size_of
from max.gpu.memory import Consistency, multimem_st
from max.gpu.intrinsics import Scope
from layout import Coord, TensorLayout, TensorEngine, TileTensor, row_major

from .sync import (
    Signal,
    _multi_gpu_barrier,
    circular_add,
    is_p2p_enabled,
)
from .device_query import dispatch_select_comm_config, get_sm_version
from .allreduce import allreduce_tuning_table

from std.utils import StaticTuple

# On AMD Systems, loads from GLOBAL addressspace give better performance.
comptime _target_address_space = AddressSpace.GLOBAL if is_amd_gpu() else AddressSpace.GENERIC


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"broadcast_multimem_{dtype}")
def broadcast_multimem_kernel[
    dtype: DType,
    Layout: TensorLayout,
    BLOCK_SIZE: Int,
    ngpus: Int,
    out_engine: TensorEngine,
    in_engine: TensorEngine,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
](
    output: TileTensor[dtype, Layout, MutAnyOrigin, Engine=out_engine],
    input: TileTensor[dtype, Layout, ImmutAnyOrigin, Engine=in_engine],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int32,
    root: Int32,
):
    """Broadcast kernel using multimem.st for multicast writes.

    Root GPU writes to multicast address, data appears on all GPUs.
    Only root performs the stores; other GPUs just participate in barriers.

    Parameters:
        dtype: Element data type of the input and output tensors.
        Layout: `TensorLayout` shared by both tensors.
        BLOCK_SIZE: Number of threads per thread block.
        ngpus: Number of GPUs participating in the broadcast.
        out_engine: Engine of the output tensor.
        in_engine: Engine of the input tensor.
        simd_width: Vector width used for memory access (defaults to the
            device-native SIMD width for `dtype`).

    Args:
        output: Output `TileTensor` for this GPU.
        input: Input `TileTensor` (root's data, readable via P2P).
        rank_sigs: Per-GPU `Signal` pointers for barrier synchronization.
        my_rank: Rank of this GPU in the communicator.
        root: Rank of the source GPU whose data is broadcast to all GPUs.
    """
    var _my_rank = Int(my_rank)
    var _root = Int(root)
    var my_sig = rank_sigs[_my_rank]

    # --- Thread Indexing ---
    var global_tid = global_idx.x
    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = grid_dim.x * BLOCK_SIZE

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, _my_rank)

        var num_elements = input.num_elements()
        var num_simd_vectors = num_elements // simd_width

        # Only root GPU performs the multicast store
        if _my_rank == _root:
            comptime alignment = align_of[SIMD[dtype, simd_width]]()

            # Get multicast output pointer and input pointer
            var out_ptr = output.ptr.unsafe_address_space_cast[.GLOBAL]()
            var in_tensor = input.address_space_cast[
                _target_address_space
            ]().reshape(row_major(num_elements))

            # Grid-strided loop to cover all elements (vectorized)
            for idx in range(global_tid, num_simd_vectors, stride):
                var elem_idx = idx * simd_width
                # Load from input buffer with invariant hint
                var data = in_tensor.load[
                    width=simd_width, alignment=alignment
                ]((elem_idx,))
                # Store to multicast address - all GPUs receive this write
                multimem_st[
                    dtype,
                    simd_width=simd_width,
                    scope=Scope.GPU,
                    consistency=Consistency.RELAXED,
                ](out_ptr.unsafe_offset(elem_idx), data)

            # Handle tail elements (when num_elements is not a multiple of simd_width).
            # multimem_st requires >= 32 bits total, so use a minimum vector width
            # (e.g. 2 for bfloat16/float16, 1 for float32).
            comptime min_mm_width = max(1, 4 // size_of[dtype]())
            var tail_start = num_simd_vectors * simd_width
            var tail_count = num_elements - tail_start
            var num_tail_chunks = tail_count // min_mm_width

            # Spread tail chunks across threads
            if global_tid < num_tail_chunks:
                var tail_elem_idx = tail_start + global_tid * min_mm_width
                var data = in_tensor.load[
                    width=min_mm_width,
                    alignment=align_of[dtype](),
                ](Coord(tail_elem_idx))
                multimem_st[
                    scope=Scope.GPU,
                    consistency=Consistency.RELAXED,
                ](out_ptr.unsafe_offset(tail_elem_idx), data)

            # Handle any remaining sub-chunk elements with an overlapping
            # write that re-stores the last min_mm_width elements of the
            # buffer. The overlap is harmless because the data is identical.
            comptime if min_mm_width > 1:
                if (
                    tail_count % min_mm_width != 0
                    and global_tid == 0
                    and num_elements >= min_mm_width
                ):
                    var overlap_idx = num_elements - min_mm_width
                    var data = in_tensor.load[
                        width=min_mm_width,
                        alignment=align_of[dtype](),
                    ](Coord(overlap_idx))
                    multimem_st[
                        scope=Scope.GPU,
                        consistency=Consistency.RELAXED,
                    ](out_ptr.unsafe_offset(overlap_idx), data)

        _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, _my_rank)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"broadcast_pull_1stage_{dtype}")
def broadcast_pull_1stage_kernel[
    dtype: DType,
    layout: TensorLayout,
    BLOCK_SIZE: Int,
    ngpus: Int,
    out_engine: TensorEngine,
    in_engine: TensorEngine,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
](
    output: TileTensor[dtype, layout, MutAnyOrigin, Engine=out_engine],
    input: TileTensor[dtype, layout, ImmutAnyOrigin, Engine=in_engine],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int32,
):
    """Single-stage pull broadcast kernel: each GPU reads root's input directly.

    All GPUs participate in the start and end barriers; after the start barrier
    every GPU copies the root's input buffer to its own output buffer using a
    grid-strided vectorized load/store loop. This one-stage path is preferred
    for small messages (up to a few MiB) and for 2-GPU configurations where
    the 2-stage scatter/gather overhead is not justified.

    Parameters:
        dtype: Element data type of the input and output tensors.
        layout: `TensorLayout` shared by both tensors.
        BLOCK_SIZE: Number of threads per thread block.
        ngpus: Number of GPUs participating in the broadcast.
        out_engine: Engine of the output tensor.
        in_engine: Engine of the input tensor.
        simd_width: Vector width used for memory access (defaults to the
            device-native SIMD width for `dtype`).

    Args:
        output: Output `TileTensor` for this GPU.
        input: Input `TileTensor` (root's data, readable via P2P).
        rank_sigs: Per-GPU `Signal` pointers for barrier synchronization.
        my_rank: Rank of this GPU in the communicator.
    """
    var _my_rank = Int(my_rank)
    var my_sig = rank_sigs[_my_rank]

    # --- Thread Indexing ---
    var global_tid = global_idx.x
    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = grid_dim.x * BLOCK_SIZE

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, _my_rank)

        comptime alignment = align_of[SIMD[dtype, simd_width]]()
        var num_elements = input.num_elements()
        var num_simd_vectors = num_elements // simd_width
        var in_tensor = input.address_space_cast[
            _target_address_space
        ]().reshape(row_major(num_elements))
        var out_tensor = output.address_space_cast[
            _target_address_space
        ]().reshape(row_major(num_elements))

        # Grid-strided loop to cover all elements (vectorized).
        for idx in range(global_tid, num_simd_vectors, stride):
            var elem_idx = idx * simd_width
            var data = in_tensor.load[width=simd_width, alignment=alignment](
                (elem_idx,)
            )
            out_tensor.store[alignment=alignment]((elem_idx,), data)

        # Handle tail elements (when num_elements is not a multiple of simd_width)
        # Spread across threads instead of just thread 0
        var tail_start = num_simd_vectors * simd_width
        var tail_idx = tail_start + global_tid
        if tail_idx < num_elements:
            var data = rebind[Scalar[dtype]](in_tensor.load((tail_idx,)))
            out_tensor.store((tail_idx,), data)

        _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, _my_rank)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"broadcast_pull_2stage_{dtype}")
def broadcast_pull_2stage_kernel[
    dtype: DType,
    OutputLayout: TensorLayout,
    ngpus: Int,
    result_engine: TensorEngine,
    in_engine: TensorEngine,
    *,
    BLOCK_SIZE: Int,
](
    result: TileTensor[dtype, OutputLayout, MutAnyOrigin, Engine=result_engine],
    root_input: TileTensor[
        dtype, OutputLayout, ImmutAnyOrigin, Engine=in_engine
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    num_elements: Int32,
    my_rank: Int32,
    root: Int32,
):
    """Two-stage broadcast: scatter from root, then allgather among all GPUs.

    Stage 1 (Scatter): Root's data is split into ngpus chunks. Each GPU
    reads its assigned chunk directly from root's input buffer and writes
    it to its signal payload. Non-root GPUs also write to their result buffer.
    Root copies all N elements from source to dest (local operation).

    Stage 2 (Allgather): Non-root GPUs gather the remaining chunks from
    all other GPUs' signal payloads (including root's). Root skips this stage
    since it already has all data.

    Parameters:
        dtype: Data dtype of tensor elements.
        OutputLayout: Layout of the output TileTensor.
        ngpus: Number of GPUs participating.
        result_engine: Engine of the output tensor.
        in_engine: Engine of the root's input tensor.
        BLOCK_SIZE: Number of threads per block.

    Args:
        result: Output TileTensor for broadcast result.
        root_input: Root's input TileTensor (all GPUs read from this via P2P).
        rank_sigs: Signal pointers for synchronization.
            IMPORTANT: Signal pointers have trailing buffers for communication.
        num_elements: Number of elements to broadcast.
        my_rank: Current GPU rank.
        root: Root GPU rank (source of broadcast).
    """
    var _my_rank = Int(my_rank)
    var _root = Int(root)
    var _num_elements = Int(num_elements)
    var my_sig = rank_sigs[_my_rank]

    # Thread indexing
    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    # Partition data among all ngpus GPUs
    var part_size = (_num_elements // simd_width // ngpus) * simd_width
    var thr_local_start = global_tid * simd_width
    var elem_stride = stride * simd_width  # Stride in elements, not threads

    # Get payload buffers from signal pointers (skip Signal header)
    # These are used as scratch space for the scatter-gather pattern
    comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    var payloads = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: (
            rank_sigs[i].unsafe_offset(1)
        ).unsafe_bitcast[Scalar[dtype]]()
    )

    with PDL():
        # === Stage 1: Scatter from root ===
        # Initial barrier to ensure all GPUs are ready
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, _my_rank)

        var is_root = _my_rank == _root
        var in_tile = root_input.address_space_cast[_target_address_space]()
        var result_tt = result.address_space_cast[
            _target_address_space
        ]().reshape(row_major(_num_elements))
        # Each GPU reads its chunk from root's input and writes to payload
        var my_chunk_start = _my_rank * part_size
        var my_chunk_end = (
            _num_elements if _my_rank
            == ngpus - 1 else my_chunk_start + part_size
        )
        var my_payload = payloads[_my_rank]

        # Calculate tail info (only last chunk can have tail elements)
        # Since part_size is SIMD-aligned, only the last GPU's chunk can have tail
        var tail_start = align_down(_num_elements, simd_width)
        var aligned_chunk_end = (
            tail_start if _my_rank == ngpus - 1 else my_chunk_end
        )

        # Stage 1: All GPUs write their chunk to payload + result
        for idx in range(
            my_chunk_start + thr_local_start, aligned_chunk_end, elem_stride
        ):
            var data = in_tile.raw_load[width=simd_width, alignment=alignment](
                idx
            )
            my_payload.unsafe_address_space_cast[
                _target_address_space
            ]().unsafe_store[alignment=alignment](idx - my_chunk_start, data)
            result_tt.store[alignment=alignment]((idx,), data)

        # Handle tail elements (spread across threads)
        var tail_idx = aligned_chunk_end + global_tid
        if tail_idx < my_chunk_end:
            var data = in_tile.raw_load[width=1](tail_idx)
            my_payload.unsafe_address_space_cast[
                _target_address_space
            ]().unsafe_store(tail_idx - my_chunk_start, data)
            result_tt.store((tail_idx,), data)

        # Barrier with memory fence to ensure scatter is complete
        _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
            rank_sigs, my_sig, _my_rank
        )

        # === Stage 2: Gather remaining chunks ===
        # Non-root GPUs gather the chunks they don't have from all other GPUs.
        if not is_root:
            # Calculate max chunk size and its aligned portion
            var last_chunk_size = _num_elements - (ngpus - 1) * part_size
            var last_aligned_size = align_down(last_chunk_size, simd_width)
            # Use the larger of part_size or last_aligned_size for loop bound
            var max_aligned_chunk_size = (
                part_size if part_size
                > last_aligned_size else last_aligned_size
            )

            for idx in range(
                thr_local_start, max_aligned_chunk_size, elem_stride
            ):
                comptime for offset in range(1, ngpus):
                    # Round-robin: each GPU gathers from other peers
                    var src_rank = circular_add[ngpus](_my_rank, offset)

                    var chunk_start = src_rank * part_size
                    # Use aligned size for last chunk, full size for others
                    var chunk_size = (
                        last_aligned_size if src_rank
                        == ngpus - 1 else part_size
                    )

                    var src_payload = payloads[src_rank]

                    # Check if idx is within this chunk's aligned bounds
                    if idx < chunk_size:
                        var data = src_payload.unsafe_address_space_cast[
                            _target_address_space
                        ]().unsafe_load[width=simd_width, alignment=alignment](
                            idx
                        )
                        # Write to final position in result
                        result_tt.store[alignment=alignment](
                            (chunk_start + idx,), data
                        )

            # Handle tail elements from last GPU's chunk (thread 0 only)
            # Skip if we're the last GPU (we already have our tail from Stage 1)
            var last_chunk_start = (ngpus - 1) * part_size
            if (
                global_tid == 0
                and _my_rank != ngpus - 1
                and last_aligned_size < last_chunk_size
            ):
                var last_payload = payloads[ngpus - 1]
                for i in range(last_aligned_size, last_chunk_size):
                    var data = last_payload.unsafe_address_space_cast[
                        _target_address_space
                    ]().unsafe_load(i)
                    result_tt.store((last_chunk_start + i,), data)

        # Root: copy all elements from input to result (after Stage 2)
        # Skip if in-place (input and result point to same memory)
        var is_inplace = in_tile.ptr == result_tt.ptr
        if is_root and not is_inplace:
            var num_simd_vectors = _num_elements // simd_width
            for idx in range(global_tid, num_simd_vectors, stride):
                var elem_idx = idx * simd_width
                var data = in_tile.raw_load[
                    width=simd_width, alignment=alignment
                ](elem_idx)
                result_tt.store[alignment=alignment]((elem_idx,), data)

            # Handle tail elements (spread across threads)
            var root_tail_idx = tail_start + global_tid
            if root_tail_idx < _num_elements:
                var data = in_tile.raw_load[width=1](root_tail_idx)
                result_tt.store((root_tail_idx,), data)

        # Final barrier to ensure all GPUs complete before returning
        _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, _my_rank)


def _should_use_2stage[ngpus: Int](num_bytes: Int) -> Bool:
    """Determine if 2-stage broadcast should be used based on GPU count and size.

    Crossover points determined empirically:
    - 2 GPUs: Always use 1-stage (2-stage has no benefit)
    - 4 GPUs: 2-stage wins for >= 6 MiB
    - 8 GPUs: 2-stage wins for >= 2 MiB
    """

    comptime match ngpus:
        case 2:
            return False
        case 4:
            return num_bytes >= 6 * 1024 * 1024  # 6 MiB
        case 8:
            return num_bytes >= 2 * 1024 * 1024  # 2 MiB
        case _:
            # For other GPU counts, use 2-stage for large sizes as a reasonable default
            return num_bytes >= 4 * 1024 * 1024  # 4 MiB


def broadcast[
    dtype: DType,
    in_layout: TensorLayout,
    in_origin: Origin,
    in_engine: TensorEngine,
    out_engine: TensorEngine,
    //,
    ngpus: Int,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    input_tensor: TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
    output_tensor: TileTensor[mut=True, dtype, in_layout, _, Engine=out_engine],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    ctx: DeviceContext,
    root: Int,
    _max_num_blocks: Optional[Int] = None,
    *,
    rank: Int,
) raises:
    """Broadcast data from root GPU to all participating GPUs.

    Parameters:
        dtype: Data type of the tensor elements.
        in_layout: Layout of the input TileTensor.
        in_origin: Origin of the input TileTensor.
        in_engine: Engine of the input tensor.
        out_engine: Engine of the output tensor.
        ngpus: Number of GPUs participating in the broadcast.
        pdl_level: Controls PDL behavior for P2P kernels.
        use_multimem: Whether to use multimem mode for improved performance.

    Args:
        input_tensor: Input tensor from root GPU as a TileTensor.
        output_tensor: Output tensor for THIS GPU as a TileTensor.
        rank_sigs: Per-GPU Signal pointers, indexed by group rank.
        ctx: Device context for THIS GPU.
        root: Group rank of the source GPU.
        _max_num_blocks: Optional grid limit.
        rank: This GPU's group rank, not its device id: a group placed on
            devices 4..7 still packs `rank_sigs` at 0..ngpus-1.
    """
    comptime assert ngpus >= 2, "broadcast requires at least 2 GPUs"

    var my_rank: Int = rank

    var num_elements = output_tensor.num_elements()
    comptime simd_width = simd_width_of[dtype, target=ctx.target]()

    # Do nothing if there are no elements to reduce.
    if num_elements == 0:
        return

    assert (
        output_tensor.num_elements() == input_tensor.num_elements()
    ), "Tensor shapes don't match"

    if not is_p2p_enabled():
        raise Error("Broadcast currently requires P2P access between GPUs")

    comptime BLOCK_SIZE = 256
    # Default max blocks if not specified.
    comptime sm_version = get_sm_version()
    # TODO: dispatch_select_comm_config was tuned for allreduce; may need separate tuning for broadcast
    var num_bytes = num_elements * size_of[dtype]()
    var max_num_blocks = _max_num_blocks.or_else(
        dispatch_select_comm_config[ngpus, sm_version, allreduce_tuning_table](
            num_bytes
        ).get_num_blocks()
    )

    var grid_size = min(
        max_num_blocks,
        ceildiv(ceildiv(num_elements, simd_width), BLOCK_SIZE),
    )

    comptime if use_multimem:
        comptime bcast_kernel = broadcast_multimem_kernel[
            dtype,
            in_layout,
            BLOCK_SIZE,
            ngpus,
            out_engine,
            in_engine,
        ]

        ctx.enqueue_function[bcast_kernel](
            output_tensor,
            input_tensor.as_imm(),
            rank_sigs,
            Int32(my_rank),
            Int32(root),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        if _should_use_2stage[ngpus](num_bytes):
            # Use 2-stage for large data with multiple GPUs
            broadcast_2stage[ngpus, pdl_level=pdl_level](
                input_tensor,
                output_tensor,
                rank_sigs,
                ctx,
                root,
                _max_num_blocks,
                rank=my_rank,
            )
        else:
            comptime bcast_kernel = broadcast_pull_1stage_kernel[
                dtype,
                in_layout,
                BLOCK_SIZE,
                ngpus,
                out_engine,
                in_engine,
            ]

            ctx.enqueue_function[bcast_kernel](
                output_tensor,
                input_tensor.as_imm(),
                rank_sigs,
                Int32(my_rank),
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
                attributes=pdl_launch_attributes(pdl_level),
            )


def broadcast_2stage[
    dtype: DType,
    in_layout: TensorLayout,
    in_origin: Origin,
    in_engine: TensorEngine,
    out_engine: TensorEngine,
    //,
    ngpus: Int,
    pdl_level: PDLLevel,
](
    input_tensor: TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
    output_tensor: TileTensor[mut=True, dtype, in_layout, _, Engine=out_engine],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    ctx: DeviceContext,
    root: Int,
    _max_num_blocks: Optional[Int] = None,
    *,
    rank: Int,
) raises:
    """Two-stage broadcast: scatter from root, then allgather among all GPUs.

    Note: This path is only used with 3+ GPUs. With 2 GPUs, broadcast uses
    the simpler 1-stage path for better performance.

    This algorithm achieves better bandwidth than simple pull broadcast by:
    1. Stage 1 (Scatter): Each GPU reads 1/ngpus of the data from root and
       writes to its payload buffer, utilizing root's outbound GPU link bandwidth.
    2. Stage 2 (Allgather): All GPUs gather from each other in parallel,
       with each GPU reading (ngpus-1) chunks from other GPUs' payloads.

    All GPUs (including root) participate uniformly in both stages, which
    better utilizes root's GPU link bandwidth and simplifies partitioning.

    IMPORTANT: Signal buffers must be sized to hold at least:
        size_of(Signal) + (num_elements / ngpus) * size_of(dtype)
    This is the payload space needed for each GPU's chunk.

    Parameters:
        dtype: Data dtype of tensor elements.
        in_layout: Layout of the input TileTensor.
        in_origin: Origin of the input TileTensor.
        in_engine: Engine of the input tensor.
        out_engine: Engine of the output tensor.
        ngpus: Number of GPUs participating.
        pdl_level: Control PDL behavior for the kernel.

    Args:
        input_tensor: Input tensor (only root's is read, but all must be valid).
        output_tensor: Output tensor for THIS GPU.
        rank_sigs: Signal pointers with payload space for staging.
        ctx: Device context for THIS GPU.
        root: Root GPU rank (source of broadcast data).
        _max_num_blocks: Optional maximum number of thread blocks.
        rank: This GPU's group rank; see `broadcast`.
    """
    var my_rank: Int = rank

    var num_elements = output_tensor.num_elements()
    comptime simd_width = simd_width_of[dtype, target=ctx.target]()

    # Do nothing if there are no elements.
    if num_elements == 0:
        return

    assert (
        output_tensor.num_elements() == input_tensor.num_elements()
    ), "Tensor shapes don't match"

    comptime BLOCK_SIZE = 256
    # Limit blocks - tuning parameter
    comptime MAX_BLOCKS = 384

    # Grid size based on per-GPU chunk size.
    var grid_size = min(
        _max_num_blocks.or_else(MAX_BLOCKS),
        ceildiv(ceildiv(num_elements, simd_width * ngpus), BLOCK_SIZE),
    )

    # `OutputLayout` and `in_layout` are the same here: broadcast writes the
    # shape it reads.
    comptime kernel = broadcast_pull_2stage_kernel[
        dtype,
        in_layout,
        ngpus,
        out_engine,
        in_engine,
        BLOCK_SIZE=BLOCK_SIZE,
    ]

    ctx.enqueue_function[kernel](
        output_tensor,
        input_tensor.as_imm(),
        rank_sigs,
        Int32(num_elements),
        Int32(my_rank),
        Int32(root),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )
