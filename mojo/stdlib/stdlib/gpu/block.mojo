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


from memory import stack_allocation
from math import align_up

from .globals import WARP_SIZE
from .id import lane_id, thread_idx, warp_id
from .memory import AddressSpace
from .sync import barrier
from .warp import broadcast as warp_broadcast
from .warp import max as warp_max
from .warp import min as warp_min
from .warp import prefix_sum as warp_prefix_sum
from .warp import sum as warp_sum

# ===-----------------------------------------------------------------------===#
# Block Reduction Core
# ===-----------------------------------------------------------------------===#


@always_inline
fn _block_reduce[
    dtype: DType,
    width: Int, //,
    block_size: Int,
    warp_reduce_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width]
    ) capturing -> SIMD[dtype, width],
    broadcast: Bool = False,
](val: SIMD[dtype, width], *, initial_val: SIMD[dtype, width]) -> SIMD[
    dtype, width
]:
    """Performs a generic block-level reduction operation.

    This function implements a block-level reduction using warp-level operations
    and shared memory for inter-warp communication. All threads in the block
    participate to compute the final reduced value.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The SIMD width for vector operations.
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
    constrained[
        block_size >= WARP_SIZE,
        "Block size must be a greater than warp size",
    ]()
    constrained[
        block_size % WARP_SIZE == 0,
        "Block size must be a multiple of warp size",
    ]()

    # Allocate shared memory for inter-warp communication.
    alias n_warps = block_size // WARP_SIZE

    @parameter
    if n_warps == 1:
        # There is a single warp, so we do not need to use shared memory
        # and warp shuffle operations are sufficient.
        var warp_result = warp_reduce_fn(val)

        @parameter
        if broadcast:
            # Broadcast the result to all threads in the warp
            warp_result = warp_broadcast(warp_result)

        return warp_result

    var shared_mem = stack_allocation[
        n_warps * width, dtype, address_space = AddressSpace.SHARED
    ]()

    # Step 1: Perform warp-level reduction.
    var warp_result = warp_reduce_fn(val)

    # Step 2: Store warp results to shared memory
    var wid = warp_id()
    var lid = lane_id()
    # Each leader thread (lane 0) is responsible for its warp.
    if lid == 0:
        shared_mem.store(wid, warp_result)

    barrier()

    # Step 3: Have the first warp reduce all warp results.
    if wid == 0:
        # Make sure that the "ghost" warps do not contribute to the sum.
        var block_val = initial_val
        # Load values from the shared memory (ith lane will have ith warp's
        # value).
        if lid < n_warps:
            block_val = shared_mem.load[width=width](lid)

        # Reduce across the first warp
        warp_result = warp_reduce_fn(block_val)

    @parameter
    if broadcast:
        # Broadcast the result to all threads in the block
        warp_result = block.broadcast[block_size=block_size](warp_result, 0)

    return warp_result


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

    @parameter
    fn _warp_sum[
        dtype: DType, width: Int
    ](x: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return warp_sum(x)

    return _block_reduce[block_size, _warp_sum, broadcast=broadcast](
        val, initial_val=0
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

    @parameter
    fn _warp_max[
        dtype: DType, width: Int
    ](x: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return warp_max(x)

    return _block_reduce[block_size, _warp_max, broadcast=broadcast](
        val, initial_val=Scalar[dtype].MIN_FINITE
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

    @parameter
    fn _warp_min[
        dtype: DType, width: Int
    ](x: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return warp_min(x)

    return _block_reduce[block_size, _warp_min, broadcast=broadcast](
        val, initial_val=Scalar[dtype].MAX_FINITE
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
    constrained[
        block_size >= WARP_SIZE,
        "Block size must be greater than or equal to warp size",
    ]()
    constrained[
        block_size % WARP_SIZE == 0,
        "Block size must be a multiple of warp size",
    ]()

    @parameter
    if block_size == WARP_SIZE:
        # Single warp - use warp shuffle for better performance
        return warp_broadcast(val)

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
    dtype: DType, //,
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
    constrained[
        block_size % WARP_SIZE == 0,
        "Block size must be a multiple of warp size",
    ]()

    # Allocate shared memory for inter-warp communication
    # We need one slot per warp to store warp-level scan results
    alias n_warps = block_size // WARP_SIZE
    var warp_mem = stack_allocation[
        align_up(n_warps, WARP_SIZE), dtype, address_space = AddressSpace.SHARED
    ]()

    var thread_result = warp_prefix_sum[exclusive=exclusive](val)

    # Step 2: Store last value from each warp to shared memory
    var wid = warp_id()
    if lane_id() == WARP_SIZE - 1:
        var inclusive_warp_sum: Scalar[dtype] = thread_result

        @parameter
        if exclusive:
            # For exclusive scan, thread_result is the sum of elements 0 to
            # WARP_SIZE-2. 'val' is the value of the element at WARP_SIZE-1.
            # Adding them gives the inclusive sum of the warp.
            inclusive_warp_sum += val

        warp_mem[wid] = inclusive_warp_sum

    barrier()

    # Step 3: Have the first warp perform a scan on the warp results
    var lid = lane_id()
    if wid == 0:
        var previous_warps_prefix = warp_prefix_sum[exclusive=False](
            warp_mem[lid]
        )
        if lid < n_warps:
            warp_mem[lid] = previous_warps_prefix
    barrier()

    # Step 4: Add the prefix from previous warps
    if wid > 0:
        thread_result += warp_mem[wid - 1]

    return thread_result
