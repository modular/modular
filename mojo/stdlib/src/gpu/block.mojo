# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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


from builtin.math import max as _max
from builtin.math import min as _min
from memory import stack_allocation, UnsafePointer

from .id import block_idx, thread_idx, lane_id, warp_id
from .sync import barrier
from .memory import AddressSpace
from .warp import (
    sum as warp_sum,
    max as warp_max,
    min as warp_min,
    prefix_sum as warp_prefix_sum,
)
from .warp import shuffle_idx, shuffle_up
from .globals import WARP_SIZE


# ===-----------------------------------------------------------------------===#
# Block Reduction Core
# ===-----------------------------------------------------------------------===#


@always_inline
fn _block_reduce[
    type: DType,
    simd_width: Int, //,
    block_size: Int,
    warp_reduce_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width]
    ) capturing -> SIMD[dtype, width],
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Performs a generic block-level reduction operation.

    This function implements a block-level reduction using warp-level operations
    and shared memory for inter-warp communication. All threads in the block
    participate to compute the final reduced value.

    Parameters:
        type: The data type of the SIMD elements.
        simd_width: The SIMD width for vector operations.
        block_size: The number of threads in the block.
        warp_reduce_fn: A function that performs warp-level reduction.

    Args:
        val: The input value from each thread to include in the reduction.

    Returns:
        The result of the reduction operation for each thread.
    """
    constrained[
        block_size % WARP_SIZE == 0,
        "Block size must be a multiple of warp size",
    ]()

    # Allocate shared memory for inter-warp communication
    alias n_warps = block_size // WARP_SIZE
    var shared_mem = stack_allocation[
        n_warps * simd_width, type, address_space = AddressSpace.SHARED
    ]()

    # Step 1: Perform warp-level reduction
    var warp_result = warp_reduce_fn(val)

    # Step 2: Store warp results to shared memory
    if lane_id() == 0:
        shared_mem.store(warp_id(), warp_result)

    barrier()

    # Step 3: Have the first warp reduce the warp results
    if warp_id() == 0:
        # Load this warp's share of the warp results
        var block_val = SIMD[type, simd_width](0)
        if lane_id() < n_warps:
            block_val = shared_mem.load[width=simd_width](lane_id())

        # Reduce across the first warp
        warp_result = warp_reduce_fn(block_val)

    # Step 4: Broadcast the result from thread 0 to all threads
    return shuffle_idx(warp_result, 0)


# ===-----------------------------------------------------------------------===#
# Block Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn sum[
    type: DType, simd_width: Int, //, *, block_size: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes the sum of values across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global sum across all threads in the block.

    Parameters:
        type: The data type of the SIMD elements.
        simd_width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to the
             sum.

    Returns:
        A SIMD value where all threads contain the sum found across the entire
        block.
    """

    @parameter
    fn _warp_sum[
        dtype: DType, width: Int
    ](x: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return warp_sum(x)

    return _block_reduce[block_size, _warp_sum](val)


# ===-----------------------------------------------------------------------===#
# Block Max
# ===-----------------------------------------------------------------------===#


@always_inline
fn max[
    type: DType, simd_width: Int, //, *, block_size: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes the maximum value across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global maximum across all threads in the block.

    Parameters:
        type: The data type of the SIMD elements.
        simd_width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the maximum.

    Returns:
        A SIMD value where all threads contain the maximum value found across
        the entire block.
    """

    @parameter
    fn _warp_max[
        dtype: DType, width: Int
    ](x: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return warp_max(x)

    return _block_reduce[block_size, _warp_max](val)


# ===-----------------------------------------------------------------------===#
# Block Min
# ===-----------------------------------------------------------------------===#


@always_inline
fn min[
    type: DType, simd_width: Int, //, *, block_size: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes the minimum value across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global minimum across all threads in the block.

    Parameters:
        type: The data type of the SIMD elements.
        simd_width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the minimum.

    Returns:
        A SIMD value where all threads contain the minimum value found across
        the entire block.
    """

    @parameter
    fn _warp_min[
        dtype: DType, width: Int
    ](x: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return warp_min(x)

    return _block_reduce[block_size, _warp_min](val)


# ===-----------------------------------------------------------------------===#
# Block Broadcast
# ===-----------------------------------------------------------------------===#


@always_inline
fn broadcast[
    type: DType, simd_width: Int, //, *, block_size: Int
](val: SIMD[type, simd_width], src_thread: UInt = 0) -> SIMD[type, simd_width]:
    """Broadcasts a value from a source thread to all threads in a block.

    This function takes a SIMD value from the specified source thread and
    copies it to all other threads in the block, effectively broadcasting
    the value across the entire block.

    Parameters:
        type: The data type of the SIMD elements.
        simd_width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.

    Args:
        val: The SIMD value to broadcast from the source thread.
        src_thread: The thread ID of the source thread (default: 0).

    Returns:
        A SIMD value where all threads contain a copy of the input value from
        the source thread.
    """
    # Allocate shared memory for broadcasting
    var shared_mem = stack_allocation[
        simd_width, type, address_space = AddressSpace.SHARED
    ]()

    # Source thread writes its value to shared memory
    if thread_idx.x == src_thread:
        shared_mem.store(0, val)

    barrier()

    # All threads read the same value from shared memory
    return shared_mem.load[width=simd_width](0)


# ===-----------------------------------------------------------------------===#
# Block Prefix Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn prefix_sum[
    type: DType, //,
    *,
    block_size: Int,
    exclusive: Bool = False,
](val: Scalar[type]) -> Scalar[type]:
    """Performs a prefix sum (scan) operation across all threads in a block.

    This function implements a block-level inclusive or exclusive scan,
    efficiently computing the cumulative sum for each thread based on
    thread indices.

    Parameters:
        type: The data type of the Scalar elements.
        block_size: The total number of threads in the block.
        exclusive: If True, perform exclusive scan instead of inclusive.

    Args:
        val: The Scalar value from each thread to include in the scan.

    Returns:
        A Scalar value containing the result of the scan operation for each
        thread.
    """
    constrained[
        block_size % WARP_SIZE == 0,
        "Block size must be a multiple of warp size",
    ]()

    # Allocate shared memory for inter-warp communication
    # We need one slot per warp to store warp-level scan results
    alias n_warps = block_size // WARP_SIZE
    var warp_mem = stack_allocation[
        n_warps, type, address_space = AddressSpace.SHARED
    ]()

    var thread_result = warp_prefix_sum[exclusive=exclusive](val)

    # Step 2: Store last value from each warp to shared memory
    if lane_id() == WARP_SIZE - 1:
        warp_mem[warp_id()] = thread_result

    barrier()

    # Step 3: Have the first warp perform a scan on the warp results
    if warp_id() == 0 and lane_id() < n_warps:
        var warp_val = warp_mem[lane_id()]

        # Perform scan on warp results
        @parameter
        for offset in range(1, n_warps):
            var addend = shuffle_up(warp_val, offset)
            if lane_id() >= offset:
                warp_val += addend

        # Store scanned warp results back
        warp_mem[lane_id()] = warp_val

    barrier()

    # Step 4: Add the prefix from previous warps
    if warp_id() > 0:
        var warp_prefix = warp_mem[warp_id() - 1]
        thread_result += warp_prefix

    return thread_result
