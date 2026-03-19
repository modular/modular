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
"""Multi-GPU allgather implementation that gathers values from multiple GPUs
into an output buffer.

This module provides an optimized implementation of allgather operations across
multiple GPUs, supporting both peer-to-peer (P2P) and non-P2P communication
patterns. The implementation automatically selects between approaches based on
hardware capabilities:

1. P2P-based implementation (when P2P access is available):
   - Uses direct GPU-to-GPU memory access for better performance.
   - Optimized for NVLink and xGMI bandwidth utilization.
   - Uses vectorized memory access.

2. Non-P2P fallback implementation:
   - Copies data through device memory when direct GPU access isn't possible.
   - Simple but functional approach for systems without P2P support.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.sys import simd_width_of

from layout import TileTensor
from layout.tile_layout import TensorLayout
from std.memory import UnsafePointer
from std.gpu import WARP_SIZE, global_idx, grid_dim
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target

from std.utils import IndexList, StaticTuple

from .sync import MAX_GPUS, Signal, _multi_gpu_barrier, is_p2p_enabled


@always_inline
def _allgather_naive[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
](
    input_buffers: InlineArray[TileTensor[dtype, in_layout, in_origin], ngpus],
    output_buffers: InlineArray[
        TileTensor[mut=True, dtype, out_layout, out_origin], ngpus
    ],
    ctx: DeviceContext,
) raises:
    """Per-device allgather fallback when P2P access is not available.

    One instance runs per GPU. Each instance copies data from all GPUs
    into its own output buffers using device-to-device memory copies.
    """
    var device_buffers = List[DeviceBuffer[dtype]](capacity=ngpus)

    for i in range(ngpus):
        var rctx = DeviceContext(device_id=i)
        device_buffers.append(
            DeviceBuffer(
                rctx,
                rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                    input_buffers[i].ptr
                ),
                input_buffers[i].num_elements(),
                owning=False,
            )
        )

    for input_idx in range(ngpus):
        var output_device_buffer = DeviceBuffer(
            ctx,
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                output_buffers[input_idx].ptr
            ),
            output_buffers[input_idx].num_elements(),
            owning=False,
        )

        ctx.enqueue_copy(
            output_device_buffer,
            device_buffers[input_idx],
        )


def _allgather_p2p_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
](
    outputs: StaticTuple[UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus],
    src_ptrs: StaticTuple[UnsafePointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    lengths: StaticTuple[Int, ngpus],
    max_num_blocks: Int,
    my_rank: Int,
):
    """P2P kernel for allgather operation.

    Each GPU directly reads from all other GPUs and writes to its output buffers.
    Uses round-robin access pattern to balance NVLink traffic.
    """
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()

    var global_tid = global_idx.x
    var stride = grid_dim.x * UInt(BLOCK_SIZE)
    var my_sig = rank_sigs[my_rank]

    # Synchronize before reading.
    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    # Copy data from each source GPU to corresponding output buffer.
    # outputs[i] should contain data from GPU i.
    comptime for src_gpu in range(ngpus):
        var length = lengths[src_gpu]
        var num_simd_vectors, remainder = divmod(length, simd_width)

        # Grid-strided loop for this source (vectorized).
        for idx in range(global_tid, num_simd_vectors, stride):
            var elem_idx = idx * simd_width
            # Read directly from source GPU.
            var data = src_ptrs[src_gpu].load[width=simd_width](elem_idx)
            # Write to output buffer for this source GPU.
            outputs[src_gpu].store(elem_idx, data)

        # Handle remainder elements with scalar operations.
        if remainder > 0:
            var tail_start = num_simd_vectors * simd_width
            # Use first warp to handle tail to minimize divergence.
            if global_tid < UInt(WARP_SIZE):
                for i in range(global_tid, remainder, WARP_SIZE):
                    var elem_idx = tail_start + i
                    outputs[src_gpu][elem_idx] = src_ptrs[src_gpu][elem_idx]

    # Synchronize after writing.
    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@always_inline
def _allgather_p2p[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
](
    input_buffers: InlineArray[TileTensor[dtype, in_layout, in_origin], ngpus],
    output_buffers: InlineArray[
        TileTensor[mut=True, dtype, out_layout, out_origin], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    max_num_blocks: Int,
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """Per-device P2P allgather: each GPU reads from all peers directly."""

    # Extract raw pointers and sizes from TileTensors.
    var list_of_in_ptrs = StaticTuple[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ]()
    var lengths = StaticTuple[Int, ngpus]()

    comptime for i in range(ngpus):
        list_of_in_ptrs[i] = rebind[
            UnsafePointer[Scalar[dtype], ImmutAnyOrigin]
        ](input_buffers[i].ptr)
        lengths[i] = input_buffers[i].num_elements()

    comptime BLOCK_SIZE = 256

    # Prepare output pointers.
    var output_ptrs = StaticTuple[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus
    ]()

    comptime for src_idx in range(ngpus):
        output_ptrs[src_idx] = rebind[
            UnsafePointer[Scalar[dtype], MutAnyOrigin]
        ](output_buffers[src_idx].ptr)

    # Calculate grid size.
    var max_length = 0
    for i in range(ngpus):
        max_length = max(max_length, lengths[i])

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    # Use ceildiv for max_length to ensure we have enough threads.
    var grid_size = min(
        max_num_blocks,
        ceildiv(ceildiv(max_length, simd_width), BLOCK_SIZE),
    )

    # Launch kernel.
    comptime allgather_p2p_kernel = _allgather_p2p_kernel[
        dtype,
        rank,
        ngpus,
        BLOCK_SIZE=BLOCK_SIZE,
    ]
    ctx.enqueue_function_experimental[allgather_p2p_kernel](
        output_ptrs,
        list_of_in_ptrs,
        rank_sigs,
        lengths,
        max_num_blocks,
        my_rank,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )


@always_inline
def allgather[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
](
    input_buffers: InlineArray[TileTensor[dtype, in_layout, in_origin], ngpus],
    output_buffers: InlineArray[
        TileTensor[mut=True, dtype, out_layout, out_origin], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    my_rank: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device all-gather: one instance per GPU builds its own outputs.

    Each instance reads all input buffers and writes to its own ngpus output
    buffers. The caller is responsible for launching one instance per device
    in parallel (e.g. via _launch_device_collective).

    The implementation automatically selects between P2P and non-P2P paths
    based on hardware capabilities.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs participating in all-gather.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensors.
        out_origin: Origin of the output TileTensors.

    Args:
        input_buffers: Input buffers from ALL GPUs as TileTensors.
        output_buffers: Output buffers for THIS GPU (ngpus TileTensors).
                       output_buffers[i] receives the data from GPU i.
        rank_sigs: Per-GPU Signal pointers for P2P synchronization.
        ctx: Device context for THIS GPU.
        my_rank: Index of this GPU among the participants.
        _max_num_blocks: Maximum number of blocks for kernel launch (optional).
    """
    comptime assert ngpus >= 2, "allgather requires at least 2 GPUs"

    # Return early if all input buffers are empty.
    var all_empty = True

    comptime for i in range(ngpus):
        if input_buffers[i].num_elements() > 0:
            all_empty = False
            break
    if all_empty:
        return

    var max_num_blocks = _max_num_blocks.or_else(216)

    # Check P2P availability.
    if not is_p2p_enabled():
        return _allgather_naive(input_buffers, output_buffers, ctx)
    else:
        return _allgather_p2p[rank=1](
            input_buffers,
            output_buffers,
            rank_sigs,
            max_num_blocks,
            ctx,
            my_rank,
        )
