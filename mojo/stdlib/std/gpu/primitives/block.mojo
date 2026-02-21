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
"""GPU block-level operations and utilities.

This module provides block-level operations for NVIDIA and AMD GPUs, including:

- Block-wide reductions:
  - sum: Compute sum across block
  - max: Find maximum value across block
  - min: Find minimum value across block
  - broadcast: Broadcast value to all threads

The module builds on warp-level operations from the warp module, extending them
to work across a full thread block (potentially multiple warps). It handles both
NVIDIA and AMD GPU architectures and supports various data types with SIMD
vectorization.
"""

from math import align_up, ceildiv

from memory import stack_allocation

from gpu import WARP_SIZE, lane_id, thread_idx, warp_id, barrier
import .warp

# ===-----------------------------------------------------------------------===#
# Block Reduction Core
# ===-----------------------------------------------------------------------===#


@always_inline
fn _block_reduce_with_padding[
    dtype: DType,
    //,
    *,
    n_warps: Int,
    padding: Int,
    warp_reduce_fn: fn[dtype: DType, width: Int](SIMD[dtype, width]) -> Scalar[
        dtype
    ],
    broadcast: Bool = False,
](val: Scalar[dtype], *, initial_val: Scalar[dtype]) -> Scalar[dtype]:
    # Add padding to avoid bank conflicts
    var shared_mem = stack_allocation[
        n_warps + padding, dtype, address_space = AddressSpace.SHARED
    ]()

    var wid = Int(warp_id())
    var lid = Int(lane_id())

    # Step 1: Perform warp-level reduction.
    var warp_result = warp_reduce_fn(val)

    @always_inline
    fn compute_offset(offset: Int) -> Int:
        """Computes the offset with the padding if needed."""

        comptime if padding > 0:
            return offset + Int(UInt(offset) // UInt(WARP_SIZE))
        else:
            return offset

    # Step 2: Store warp results to shared memory with padding consideration
    # Each leader thread (lane 0) is responsible for its warp.
    # Account for padding when storing to avoid bank conflicts
    if lid == 0:
        shared_mem[compute_offset(wid)] = warp_result

    barrier()

    # Step 3: Have the first warp reduce all warp results.
    if wid == 0:
        # Make sure that the "ghost" warps do not contribute to the sum.
        var block_val = initial_val
        # Load values from the shared memory (ith lane will have ith warp's
        # value). Account for padding when loading.
        if lid < n_warps:
            block_val = shared_mem[compute_offset(lid)]

        # Reduce across the first warp
        warp_result = warp_reduce_fn(block_val)

        comptime if broadcast:
            # Store the final result back to shared memory for broadcast
            if lid == 0:
                shared_mem[] = warp_result

    comptime if broadcast:
        # Synchronize and broadcast the result to all threads
        barrier()
        # All threads read the final result from shared memory
        warp_result = shared_mem[]

    return warp_result


@always_inline
fn _block_reduce[
    dtype: DType,
    //,
    block_size: Int,
    warp_reduce_fn: fn[dtype: DType, width: Int](SIMD[dtype, width]) -> Scalar[
        dtype
    ],
    broadcast: Bool = False,
](val: Scalar[dtype], *, initial_val: Scalar[dtype]) -> Scalar[dtype]:
    """Performs a generic block-level reduction operation.

    This function implements a block-level reduction using warp-level operations
    and shared memory for inter-warp communication. All threads in the block
    participate to compute the final reduced value.

    Parameters:
        dtype: The data type of the SIMD elements.
        block_size: The number of threads in the block.
        warp_reduce_fn: A function that performs warp-level reduction.
        broadcast: If True, the final reduced value is broadcast to all
            threads in the block. If False, only the first thread will have the
            complete result.

    Args:
        val: The input value from each thread to include in the reduction.
        initial_val: The initial value for the reduction.

    Returns:
        If broadcast is True, each thread in the block will receive the reduced
        value. Otherwise, only the first thread will have the complete result.
    """
    comptime assert (
        block_size >= WARP_SIZE
    ), "Block size must be a greater than warp size"
    comptime assert (
        block_size % WARP_SIZE == 0
    ), "Block size must be a multiple of warp size"

    # Allocate shared memory for inter-warp communication.
    comptime n_warps = block_size // WARP_SIZE

    comptime if n_warps == 1:
        # Single warp optimization: no shared memory or barriers needed
        # Warp shuffle operations are sufficient and much faster
        var warp_result = warp_reduce_fn(val)

        comptime if broadcast:
            # Use efficient warp broadcast (shuffle to lane 0)
            warp_result = warp.broadcast(warp_result)

        return warp_result

    comptime if n_warps == 2:
        return _block_reduce_with_padding[
            n_warps=n_warps,
            padding=0,
            warp_reduce_fn=warp_reduce_fn,
            broadcast=broadcast,
        ](val, initial_val=initial_val)

    # General case with bank conflict optimization
    # Add padding to avoid bank conflicts
    comptime padding = ceildiv(n_warps, WARP_SIZE) if n_warps > WARP_SIZE else 0
    return _block_reduce_with_padding[
        n_warps=n_warps,
        padding=padding,
        warp_reduce_fn=warp_reduce_fn,
        broadcast=broadcast,
    ](val, initial_val=initial_val)


# ===-----------------------------------------------------------------------===#
# Block Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn sum[
    dtype: DType, width: Int, //, *, block_size: Int, broadcast: Bool = True
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the sum of values across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global sum across all threads in the block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.
        broadcast: If True, the final sum is broadcast to all threads in the
            block. If False, only the first thread will have the complete sum.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to the
             sum.

    Returns:
        If broadcast is True, each thread in the block will receive the final
        sum. Otherwise, only the first thread will have the complete sum.
    """

    return _block_reduce[block_size, warp.sum, broadcast=broadcast](
        val.reduce_add(), initial_val=0
    )


# ===-----------------------------------------------------------------------===#
# Block Max
# ===-----------------------------------------------------------------------===#


@always_inline
fn max[
    dtype: DType, width: Int, //, *, block_size: Int, broadcast: Bool = True
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the maximum value across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global maximum across all threads in the block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.
        broadcast: If True, the final reduced value is broadcast to all
            threads in the block. If False, only the first thread will have the
            complete result.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the maximum.

    Returns:
        If broadcast is True, each thread in the block will receive the maximum
        value across the entire block. Otherwise, only the first thread will
        have the complete result.
    """

    return _block_reduce[block_size, warp.max, broadcast=broadcast](
        val.reduce_max(), initial_val=Scalar[dtype].MIN_FINITE
    )


# ===-----------------------------------------------------------------------===#
# Block Min
# ===-----------------------------------------------------------------------===#


@always_inline
fn min[
    dtype: DType, width: Int, //, *, block_size: Int, broadcast: Bool = True
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the minimum value across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global minimum across all threads in the block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.
        broadcast: If True, the final minimum is broadcast to all threads in the
            block. If False, only the first thread will have the complete min.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the minimum.

    Returns:
        If broadcast is True, each thread in the block will receive the minimum
        value across the entire block. Otherwise, only the first thread will
        have the complete result.
    """

    return _block_reduce[block_size, warp.min, broadcast=broadcast](
        val.reduce_min(), initial_val=Scalar[dtype].MAX_FINITE
    )


# ===-----------------------------------------------------------------------===#
# Block Broadcast
# ===-----------------------------------------------------------------------===#


@always_inline
fn broadcast[
    dtype: DType, width: Int, //, *, block_size: Int
](val: SIMD[dtype, width], src_thread: UInt = 0) -> SIMD[dtype, width]:
    """Broadcasts a value from a source thread to all threads in a block.

    This function takes a SIMD value from the specified source thread and
    copies it to all other threads in the block, effectively broadcasting
    the value across the entire block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.

    Args:
        val: The SIMD value to broadcast from the source thread.
        src_thread: The thread ID of the source thread (default: 0).

    Returns:
        A SIMD value where all threads contain a copy of the input value from
        the source thread.
    """
    comptime assert (
        block_size >= WARP_SIZE
    ), "Block size must be greater than or equal to warp size"
    comptime assert (
        block_size % WARP_SIZE == 0
    ), "Block size must be a multiple of warp size"

    comptime if block_size == WARP_SIZE:
        # Single warp - use warp shuffle for better performance
        return warp.broadcast(val)

    # Multi-warp block - use shared memory
    var shared_mem = stack_allocation[
        width, dtype, address_space = AddressSpace.SHARED
    ]()

    # Source thread writes its value to shared memory
    if thread_idx.x == src_thread:
        shared_mem.store(val)

    barrier()

    # All threads read the same value from shared memory
    return shared_mem.load[width=width]()


# ===-----------------------------------------------------------------------===#
# Block Prefix Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn prefix_sum[
    dtype: DType,
    //,
    *,
    block_size: Int,
    exclusive: Bool = False,
](val: Scalar[dtype]) -> Scalar[dtype]:
    """Performs a prefix sum (scan) operation across all threads in a block.

    This function implements a block-level inclusive or exclusive scan,
    efficiently computing the cumulative sum for each thread based on
    thread indices.

    Parameters:
        dtype: The data type of the Scalar elements.
        block_size: The total number of threads in the block.
        exclusive: If True, perform exclusive scan instead of inclusive.

    Args:
        val: The Scalar value from each thread to include in the scan.

    Returns:
        A Scalar value containing the result of the scan operation for each
        thread.
    """
    comptime assert (
        block_size % WARP_SIZE == 0
    ), "Block size must be a multiple of warp size"

    # Allocate shared memory for inter-warp communication
    # We need one slot per warp to store warp-level scan results
    comptime n_warps = block_size // WARP_SIZE
    var warp_mem = stack_allocation[
        align_up(n_warps, WARP_SIZE), dtype, address_space = AddressSpace.SHARED
    ]()

    var thread_result = warp.prefix_sum[exclusive=exclusive](val)

    # Step 2: Store last value from each warp to shared memory
    var wid = warp_id()
    if lane_id() == UInt(WARP_SIZE - 1):
        var inclusive_warp_sum: Scalar[dtype] = thread_result

        comptime if exclusive:
            # For exclusive scan, thread_result is the sum of elements 0 to
            # WARP_SIZE-2. 'val' is the value of the element at WARP_SIZE-1.
            # Adding them gives the inclusive sum of the warp.
            inclusive_warp_sum += val

        warp_mem[wid] = inclusive_warp_sum

    barrier()

    # Step 3: Have the first warp perform a scan on the warp results
    var lid = lane_id()
    if wid == 0:
        var previous_warps_prefix = warp.prefix_sum[exclusive=False](
            warp_mem[lid]
        )
        if lid < UInt(n_warps):
            warp_mem[lid] = previous_warps_prefix
    barrier()

    # Step 4: Add the prefix from previous warps
    if wid > 0:
        thread_result += warp_mem[wid - 1]

    return thread_result
