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

from collections import InlineArray
from math import ceildiv
from sys import simd_width_of

from buffer import NDBuffer
from gpu import (
    block_dim,
    global_idx,
    grid_dim,
    WARP_SIZE,
)
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host import get_gpu_target

from utils import StaticTuple

# Import P2P detection and synchronization from allreduce
from .allreduce import MAX_GPUS, Signal, can_enable_p2p, _multi_gpu_barrier


@always_inline
fn _allgather_naive[
    dtype: DType,
    rank: Int,
    ngpus: Int,
](
    input_buffers: InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus],
    output_buffers: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus * ngpus
    ],
    ctxs: List[DeviceContext],
) raises:
    """Performs allgather across GPUs without using peer-to-peer access.

    This is the fallback implementation when P2P is not available.
    Each device copies data from all other devices through device memory.
    """
    var device_buffers = List[DeviceBuffer[dtype]](capacity=ngpus)

    # Assemble input buffers from all devices.
    for device_idx in range(ngpus):
        device_buffers.append(
            DeviceBuffer(
                ctxs[device_idx],
                input_buffers[device_idx].data,
                input_buffers[device_idx].num_elements(),
                owning=False,
            )
        )

    for device_idx in range(ngpus):
        var curr_ctx = ctxs[device_idx]

        # Copy each input to this device as a separate buffer.
        for input_idx in range(ngpus):
            # Calculate flat index for this output buffer.
            var output_idx = device_idx * ngpus + input_idx

            var output_device_buffer = DeviceBuffer(
                curr_ctx,
                output_buffers[output_idx].data,
                output_buffers[output_idx].num_elements(),
                owning=False,
            )

            # Copy from input device to current device.
            curr_ctx.enqueue_copy(
                output_device_buffer,
                device_buffers[input_idx],
            )


fn _allgather_p2p_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    my_rank: Int,
    *,
    BLOCK_SIZE: Int,
](
    outputs: StaticTuple[UnsafePointer[Scalar[dtype]], ngpus],
    src_ptrs: StaticTuple[UnsafePointer[Scalar[dtype]], ngpus],
    rank_sigs: StaticTuple[UnsafePointer[Signal], MAX_GPUS],
    lengths: StaticTuple[Int, ngpus],
    max_num_blocks: Int,
):
    """P2P kernel for allgather operation.

    Each GPU directly reads from all other GPUs and writes to its output buffers.
    Uses round-robin access pattern to balance NVLink traffic.
    """
    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE
    var my_sig: UnsafePointer[Signal] = rank_sigs[my_rank]

    # Synchronize before reading.
    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    # Copy data from each source GPU to corresponding output buffer.
    # outputs[i] should contain data from GPU i.
    @parameter
    for src_gpu in range(ngpus):
        var length = lengths[src_gpu]
        var num_simd_vectors = length // simd_width
        var remainder = length % simd_width

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
            if global_tid < WARP_SIZE:
                for i in range(global_tid, remainder, WARP_SIZE):
                    var elem_idx = tail_start + i
                    outputs[src_gpu][elem_idx] = src_ptrs[src_gpu][elem_idx]

    # Synchronize after writing.
    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@always_inline
fn _allgather_p2p[
    dtype: DType,
    rank: Int,
    ngpus: Int,
](
    input_buffers: InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus],
    output_buffers: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus * ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    max_num_blocks: Int,
    ctxs: List[DeviceContext],
) raises:
    """Performs allgather using peer-to-peer access between GPUs."""

    # Prepare input pointers
    var list_of_in_ptrs = StaticTuple[UnsafePointer[Scalar[dtype]], ngpus]()
    var lengths = StaticTuple[Int, ngpus]()

    @parameter
    for i in range(ngpus):
        list_of_in_ptrs[i] = input_buffers[i].data
        lengths[i] = input_buffers[i].num_elements()

    # Prepare signal pointers
    var rank_sigs_tuple = StaticTuple[UnsafePointer[Signal], MAX_GPUS](
        UnsafePointer[Signal]()
    )

    @parameter
    for i in range(ngpus):
        rank_sigs_tuple[i] = rank_sigs[i]

    alias BLOCK_SIZE = 256

    # Launch kernel on each GPU.
    @parameter
    for gpu_idx in range(ngpus):
        var curr_ctx = ctxs[gpu_idx]

        # Prepare output pointers for this GPU.
        var output_ptrs = StaticTuple[UnsafePointer[Scalar[dtype]], ngpus]()

        @parameter
        for src_idx in range(ngpus):
            var output_idx = gpu_idx * ngpus + src_idx
            output_ptrs[src_idx] = output_buffers[output_idx].data

        # Calculate grid size.
        var max_length = 0
        for i in range(ngpus):
            max_length = max(max_length, lengths[i])

        alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
        # Use ceildiv for max_length to ensure we have enough threads.
        var grid_size = min(
            max_num_blocks,
            ceildiv(ceildiv(max_length, simd_width), BLOCK_SIZE),
        )

        # Launch kernel.
        curr_ctx.enqueue_function[
            _allgather_p2p_kernel[
                dtype,
                rank,
                ngpus,
                my_rank=gpu_idx,
                BLOCK_SIZE=BLOCK_SIZE,
            ]
        ](
            output_ptrs,
            list_of_in_ptrs,
            rank_sigs_tuple,
            lengths,
            max_num_blocks,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )


@always_inline
fn allgather[
    dtype: DType,
    rank: Int,
    ngpus: Int,
](
    input_buffers: InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus],
    output_buffers: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus * ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
    _max_num_blocks: Optional[Int] = None,
) raises:
    """
    Performs all-gather across GPUs with variadic output.

    Each device receives individual copies of all input buffers.

    The implementation automatically selects between P2P and non-P2P paths
    based on hardware capabilities.

    Parameters:
        dtype: DType - The data type of tensor elements.
        rank: Int - Number of dimensions in input tensors.
        ngpus: Int - Number of GPUs participating in all-gather.

    Args:
        input_buffers: Input buffers from each GPU.
        output_buffers: Flat array of ngpus * ngpus output buffers.
                       Layout: output_buffers[device_idx * ngpus + input_idx]
                       contains device_idx's copy of input_idx's data.
        rank_sigs: Signal pointers for P2P synchronization.
        ctxs: List of device contexts for participating GPUs.
        _max_num_blocks: Maximum number of blocks for kernel launch (optional).
    """

    # Default max blocks if not specified
    var max_num_blocks = _max_num_blocks.or_else(216)

    # Check P2P availability.
    if not can_enable_p2p():
        return _allgather_naive(input_buffers, output_buffers, ctxs)
    else:
        return _allgather_p2p(
            input_buffers, output_buffers, rank_sigs, max_num_blocks, ctxs
        )


# Backward compatibility overload without rank_sigs.
@deprecated("Use the `signal_buffers` overload of `allgather` instead.")
@always_inline
fn allgather[
    dtype: DType,
    rank: Int,
    ngpus: Int,
](
    input_buffers: InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus],
    output_buffers: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus * ngpus
    ],
    ctxs: List[DeviceContext],
) raises:
    """Backward compatible version without rank_sigs parameter.

    This version uses the naive implementation since we can't allocate signal
    buffers with proper lifetime in this function.
    """
    # Use naive implementation for backward compatibility.
    _allgather_naive(input_buffers, output_buffers, ctxs)
