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
"""GPU warp-level operations and utilities.

This module provides warp-level operations for NVIDIA and AMD GPUs, including:

- Shuffle operations to exchange values between threads in a warp:
  - shuffle_idx: Copy value from source lane to other lanes
  - shuffle_up: Copy from lower lane IDs
  - shuffle_down: Copy from higher lane IDs
  - shuffle_xor: Exchange values in butterfly pattern

- Warp-wide reductions:
  - sum: Compute sum across warp
  - max: Find maximum value across warp
  - min: Find minimum value across warp
  - broadcast: Broadcast value to all lanes

The module handles both NVIDIA and AMD GPU architectures through architecture-specific
implementations of the core operations. It supports various data types including
integers, floats, and half-precision floats, with SIMD vectorization.
"""

from sys import bitwidthof, is_nvidia_gpu, llvm_intrinsic, sizeof
from sys._assembly import inlined_assembly
from sys.info import _is_sm_100x_or_newer

from bit import log2_floor
from builtin.math import max as _max
from builtin.math import min as _min
from gpu import lane_id
from gpu.globals import WARP_SIZE
from memory import bitcast

from .tensor_ops import tc_reduce

# TODO (#24457): support shuffles with width != 32
alias _WIDTH_MASK = WARP_SIZE - 1
alias _FULL_MASK = 2**WARP_SIZE - 1

# shfl.sync.up.b32 prepares this mask differently from other shuffle intrinsics
alias _WIDTH_MASK_SHUFFLE_UP = 0


# ===-----------------------------------------------------------------------===#
# utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _shuffle[
    mnemonic: StringSlice,
    type: DType,
    simd_width: Int,
    *,
    WIDTH_MASK: Int32,
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    constrained[
        type.is_half_float() or simd_width == 1,
        "Unsupported simd_width",
    ]()

    @parameter
    if type is DType.float32:
        return llvm_intrinsic[
            "llvm.nvvm.shfl.sync." + mnemonic + ".f32", Scalar[type]
        ](Int32(mask), val, offset, WIDTH_MASK)
    elif type in (DType.int32, DType.uint32):
        return llvm_intrinsic[
            "llvm.nvvm.shfl.sync." + mnemonic + ".i32", Scalar[type]
        ](Int32(mask), val, offset, WIDTH_MASK)
    elif type in (DType.int64, DType.uint64):
        var val_bitcast = bitcast[
            new_type = DType.uint32, new_width = simd_width * 2
        ](val)
        var val_half1, val_half2 = val_bitcast.deinterleave()
        var shuffle1 = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val_half1, offset
        )
        var shuffle2 = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val_half2, offset
        )
        var result = shuffle1.interleave(shuffle2)
        return bitcast[type, simd_width](result)
    elif type.is_half_float():

        @parameter
        if simd_width == 1:
            # splat and recurse to meet 32 bitwidth requirements
            var splatted_val = SIMD[type, 2](rebind[Scalar[type]](val))
            return _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
                mask, splatted_val, offset
            )[0]
        else:
            # bitcast and recurse to use i32 intrinsic. Two half values fit
            # into an int32.
            var packed_val = bitcast[DType.int32, simd_width // 2](val)
            var result_packed = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
                mask, packed_val, offset
            )
            return bitcast[type, simd_width](result_packed)
    elif type is DType.bool:
        constrained[simd_width == 1, "unhandled simd width"]()
        return _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val.cast[DType.int32](), offset
        ).cast[type]()

    else:
        constrained[False, "unhandled shuffle type"]()
        return 0


@always_inline
fn _shuffle_amd_helper[
    type: DType, simd_width: Int
](dst_lane: UInt32, val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    @parameter
    if sizeof[SIMD[type, simd_width]]() == 4:
        # Handle int32, float32, float16x2, etc.
        var result_packed = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](
            dst_lane * 4, bitcast[DType.int32, 1](val)
        )
        return bitcast[type, simd_width](result_packed)

    constrained[simd_width == 1, "Unsupported simd width"]()

    @parameter
    if type is DType.bool:
        return _shuffle_amd_helper(dst_lane, val.cast[DType.int32]()).cast[
            type
        ]()
    elif bitwidthof[type]() == 16:
        var val_splatted = SIMD[type, 2](rebind[Scalar[type]](val))
        return _shuffle_amd_helper(dst_lane, val_splatted)[0]
    elif bitwidthof[type]() == 64:
        var val_bitcast = bitcast[
            new_type = DType.uint32, new_width = simd_width * 2
        ](val)
        var val_half1, val_half2 = val_bitcast.deinterleave()
        var shuffle1 = _shuffle_amd_helper(dst_lane, val_half1)
        var shuffle2 = _shuffle_amd_helper(dst_lane, val_half2)
        var result = shuffle1.interleave(shuffle2)
        return bitcast[type, simd_width](result)
    else:
        constrained[False, "unhandled shuffle type"]()
        return 0


# ===-----------------------------------------------------------------------===#
# shuffle_idx
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_idx[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
    """Copies a value from a source lane to other lanes in a warp.

        Broadcasts a value from a source thread in a warp to all participating threads
        without using shared memory. This is a convenience wrapper that uses the full
        warp mask by default.

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32, half).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be broadcast from the source lane.
        offset: The source lane ID to copy the value from.

    Returns:
        A SIMD vector where all lanes contain the value from the source lane specified by offset.

    Example:

        ```mojo
            from gpu.warp import shuffle_idx

            val = SIMD[DType.float32, 16](1.0)

            # Broadcast value from lane 0 to all lanes
            result = shuffle_idx(val, 0)

            # Get value from lane 5
            result = shuffle_idx(val, 5)
        ```
        .
    """
    return shuffle_idx(_FULL_MASK, val, offset)


@always_inline
fn _shuffle_idx_amd[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane: Int32 = lane_id()
    # Godbolt uses 0x3fffffc0. It is masking out the lower 64-bits
    # But it's also masking out the upper two bits. Why?
    # The lane should not be > 64 so the upper 2 bits should always be zero.
    # Use -64 for now.
    var t0 = lane & -WARP_SIZE
    var dst_lane = t0 | offset.cast[DType.int32]()
    return _shuffle_amd_helper(UInt32(dst_lane), val)


@always_inline
fn shuffle_idx[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    """Copies a value from a source lane to other lanes in a warp with explicit mask control.

        Broadcasts a value from a source thread in a warp to participating threads specified by
        the mask. This provides fine-grained control over which threads participate in the shuffle
        operation.

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32, half).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: A bit mask specifying which lanes participate in the shuffle (1 bit per lane).
        val: The SIMD value to be broadcast from the source lane.
        offset: The source lane ID to copy the value from.

    Returns:
        A SIMD vector where participating lanes (set in mask) contain the value from the
        source lane specified by offset. Non-participating lanes retain their original values.

    Example:

        ```mojo
            from gpu.warp import shuffle_idx

            # Only broadcast to first 16 lanes
            var mask = 0xFFFF  # 16 ones
            var val = SIMD[DType.float32, 32](1.0)
            var result = shuffle_idx(mask, val, 5)
        ```
        .
    """

    @parameter
    if is_nvidia_gpu():
        return _shuffle[
            "idx",
            WIDTH_MASK=_WIDTH_MASK,
        ](mask, val, offset)
    else:
        return _shuffle_idx_amd(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_up
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_up[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
    """Copies values from threads with lower lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    lower lane ID, offset by the specified amount. Uses the full warp mask by default.

    For example, with offset=1:
    - Thread N gets value from thread N-1
    - Thread 1 gets value from thread 0
    - Thread 0 gets undefined value

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be shuffled up the warp.
        offset: The number of lanes to shift values up by.

    Returns:
        The SIMD value from the thread offset lanes lower in the warp.
        Returns undefined values for threads where lane_id - offset < 0.
    """
    return shuffle_up(_FULL_MASK, val, offset)


@always_inline
fn _shuffle_up_amd[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane: Int32 = lane_id()
    var t0 = lane - offset.cast[DType.int32]()
    var t1 = lane & -WARP_SIZE
    var dst_lane = (t0 < t1).select(lane, t0)
    return _shuffle_amd_helper(UInt32(dst_lane), val)


@always_inline
fn shuffle_up[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    """Copies values from threads with lower lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    lower lane ID, offset by the specified amount. The operation is performed only for
    threads specified in the mask.

    For example, with offset=1:
    - Thread N gets value from thread N-1 if both threads are in the mask
    - Thread 1 gets value from thread 0 if both threads are in the mask
    - Thread 0 gets undefined value
    - Threads not in the mask get undefined values

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: The warp mask specifying which threads participate in the shuffle.
        val: The SIMD value to be shuffled up the warp.
        offset: The number of lanes to shift values up by.

    Returns:
        The SIMD value from the thread offset lanes lower in the warp.
        Returns undefined values for threads where lane_id - offset < 0 or
        threads not in the mask.
    """

    @parameter
    if is_nvidia_gpu():
        return _shuffle["up", WIDTH_MASK=_WIDTH_MASK_SHUFFLE_UP](
            mask, val, offset
        )
    else:
        return _shuffle_up_amd(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_down
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_down[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
    """Copies values from threads with higher lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    higher lane ID, offset by the specified amount. Uses the full warp mask by default.

    For example, with offset=1:
    - Thread 0 gets value from thread 1
    - Thread 1 gets value from thread 2
    - Thread N gets value from thread N+1
    - Last N threads get undefined values

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be shuffled down the warp.
        offset: The number of lanes to shift values down by. Must be positive.

    Returns:
        The SIMD value from the thread offset lanes higher in the warp.
        Returns undefined values for threads where lane_id + offset >= WARP_SIZE.
    """
    return shuffle_down(_FULL_MASK, val, offset)


@always_inline
fn _shuffle_down_amd[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane = lane_id()
    # set the offset to 0 if lane + offset >= WARP_SIZE
    var dst_lane = (lane + offset > _WIDTH_MASK).select(0, offset) + lane
    return _shuffle_amd_helper(dst_lane, val)


@always_inline
fn shuffle_down[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    """Copies values from threads with higher lane IDs in the warp using a custom mask.

    Performs a shuffle operation where each thread receives a value from a thread with a
    higher lane ID, offset by the specified amount. The mask parameter controls which
    threads participate in the shuffle.

    For example, with offset=1:
    - Thread 0 gets value from thread 1
    - Thread 1 gets value from thread 2
    - Thread N gets value from thread N+1
    - Last N threads get undefined values

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: A bitmask controlling which threads participate in the shuffle.
             Only threads with their corresponding bit set will exchange values.
        val: The SIMD value to be shuffled down the warp.
        offset: The number of lanes to shift values down by. Must be positive.

    Returns:
        The SIMD value from the thread offset lanes higher in the warp.
        Returns undefined values for threads where lane_id + offset >= WARP_SIZE
        or where the corresponding mask bit is not set.
    """

    @parameter
    if is_nvidia_gpu():
        return _shuffle["down", WIDTH_MASK=_WIDTH_MASK](mask, val, offset)
    else:
        return _shuffle_down_amd(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_xor
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_xor[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
    """Exchanges values between threads in a warp using a butterfly pattern.

    Performs a butterfly exchange pattern where each thread swaps values with another thread
    whose lane ID differs by a bitwise XOR with the given offset. This creates a butterfly
    communication pattern useful for parallel reductions and scans.

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be exchanged with another thread.
        offset: The lane offset to XOR with the current thread's lane ID to determine
               the exchange partner. Common values are powers of 2 for butterfly patterns.

    Returns:
        The SIMD value from the thread at lane (current_lane XOR offset).
    """
    return shuffle_xor(_FULL_MASK, val, offset)


@always_inline
fn _shuffle_xor_amd[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane: UInt32 = lane_id()
    var t0 = lane ^ offset
    var t1 = lane & -WARP_SIZE
    # This needs to be "add nsw" = add no sign wrap
    var t2 = t1 + WARP_SIZE
    var dst_lane = (t0 < t2).select(t0, lane)
    return _shuffle_amd_helper(dst_lane, val)


@always_inline
fn shuffle_xor[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    """Exchanges values between threads in a warp using a butterfly pattern with masking.

    Performs a butterfly exchange pattern where each thread swaps values with another thread
    whose lane ID differs by a bitwise XOR with the given offset. The mask parameter allows
    controlling which threads participate in the exchange.

    Parameters:
        type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: A bit mask specifying which threads participate in the exchange.
             Only threads with their corresponding bit set in the mask will exchange values.
        val: The SIMD value to be exchanged with another thread.
        offset: The lane offset to XOR with the current thread's lane ID to determine
               the exchange partner. Common values are powers of 2 for butterfly patterns.

    Returns:
        The SIMD value from the thread at lane (current_lane XOR offset) if both threads
        are enabled by the mask, otherwise the original value is preserved.

    Example:

        ```mojo
            from gpu.warp import shuffle_xor

            # Exchange values between even-numbered threads 4 lanes apart
            mask = 0xAAAAAAAA  # Even threads only
            var val = SIMD[DType.float32, 16](42.0)  # Example value
            result = shuffle_xor(mask, val, 4.0)
        ```
        .
    """

    @parameter
    if is_nvidia_gpu():
        return _shuffle["bfly", WIDTH_MASK=_WIDTH_MASK](mask, val, offset)
    else:
        return _shuffle_xor_amd(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# Warp Reduction
# ===-----------------------------------------------------------------------===#


@always_inline
fn lane_group_reduce[
    val_type: DType,
    simd_width: Int, //,
    shuffle: fn[type: DType, simd_width: Int] (
        val: SIMD[type, simd_width], offset: UInt32
    ) -> SIMD[type, simd_width],
    func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    num_lanes: Int,
    *,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Performs a generic warp-level reduction operation using shuffle operations.

    This function implements a parallel reduction across threads in a warp using a butterfly
    pattern. It allows customizing both the shuffle operation and reduction function.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        shuffle: A function that performs the warp shuffle operation. Takes a SIMD value and
                offset and returns the shuffled result.
        func: A binary function that combines two SIMD values during reduction. This defines
              the reduction operation (e.g. add, max, min).
        num_lanes: The number of lanes in a group. The reduction is done within each group. Must be a power of 2.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value.

    Returns:
        A SIMD value containing the reduction result.

    Example:

        ```mojo
            from gpu.warp import lane_group_reduce, shuffle_down

            # Compute sum across 16 threads using shuffle down
            @parameter
            fn add[type: DType, width: Int](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
                return x + y
            var val = SIMD[DType.float32, 16](42.0)
            var result = lane_group_reduce[shuffle_down, add, num_lanes=16](val)
        ```
        .
    """
    var res = val

    alias limit = log2_floor(num_lanes)

    @parameter
    for i in reversed(range(limit)):
        alias offset = 1 << i
        res = func(res, shuffle(res, offset * stride))

    return res


@always_inline
fn reduce[
    val_type: DType,
    simd_width: Int, //,
    shuffle: fn[type: DType, simd_width: Int] (
        val: SIMD[type, simd_width], offset: UInt32
    ) -> SIMD[type, simd_width],
    func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Performs a generic warp-wide reduction operation using shuffle operations.

    This is a convenience wrapper around lane_group_reduce that operates on the entire warp.
    It allows customizing both the shuffle operation and reduction function.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        shuffle: A function that performs the warp shuffle operation. Takes a SIMD value and
                offset and returns the shuffled result.
        func: A binary function that combines two SIMD values during reduction. This defines
              the reduction operation (e.g. add, max, min).

    Args:
        val: The SIMD value to reduce. Each lane contributes its value.

    Returns:
        A SIMD value containing the reduction result broadcast to all lanes in the warp.

    Example:

    ```mojo
        from gpu.warp import reduce, shuffle_down

        # Compute warp-wide sum using shuffle down
        @parameter
        fn add[type: DType, width: Int](x: SIMD[type, width], y: SIMD[type, width]) capturing -> SIMD[type, width]:
            return x + y

        val = SIMD[DType.float32, 4](2.0, 4.0, 6.0, 8.0)
        result = reduce[shuffle_down, add](val)
    ```
    .
    """
    return lane_group_reduce[shuffle, func, num_lanes=WARP_SIZE](val)


# ===-----------------------------------------------------------------------===#
# Warp Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn lane_group_sum[
    val_type: DType,
    simd_width: Int, //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the sum of values across a group of lanes using warp-level operations.

    This function performs a parallel reduction across a group of lanes to compute their sum.
    The reduction is done using warp shuffle operations for efficient communication between lanes.
    The result is stored in all participating lanes.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to the sum.

    Returns:
        A SIMD value where all participating lanes contain the sum found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """

    @parameter
    fn _reduce_add(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return x + y

    return lane_group_reduce[
        shuffle_down, _reduce_add, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn lane_group_sum_and_broadcast[
    val_type: DType,
    simd_width: Int, //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the sum across a lane group and broadcasts the result to all lanes.

    This function performs a parallel reduction using a butterfly pattern to compute the sum,
    then broadcasts the result to all participating lanes. The butterfly pattern ensures
    efficient communication between lanes through warp shuffle operations.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to the sum.

    Returns:
        A SIMD value where all participating lanes contain the sum found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """

    @parameter
    fn _reduce_add(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return x + y

    return lane_group_reduce[
        shuffle_xor, _reduce_add, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn sum[
    val_type: DType, simd_width: Int, //
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the sum of values across all lanes in a warp.

    This is a convenience wrapper around lane_group_sum_and_broadcast that
    operates on the entire warp.  It performs a parallel reduction using warp
    shuffle operations to find the global sum across all lanes in the warp.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to the sum.

    Returns:
        A SIMD value where all lanes contain the sum found across the entire warp.
        The sum is broadcast to all lanes.
    """
    return lane_group_sum_and_broadcast[num_lanes=WARP_SIZE](val)


@fieldwise_init
@register_passable("trivial")
struct ReductionMethod:
    """Enumerates the supported reduction methods."""

    var _value: Int

    alias TENSOR_CORE = Self(0)
    """Use tensor core for reduction."""
    alias WARP = Self(1)
    """Use warp shuffle for reduction."""

    fn __eq__(self, other: Self) -> Bool:
        """Checks if two ReductionMethod are equal.

        Args:
            other: The other ReductionMethod to compare.

        Returns:
            True if the ReductionMethod are equal, false otherwise.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Checks if two ReductionMethod are not equal.

        Args:
            other: The other ReductionMethod to compare.

        Returns:
            True if the ReductionMethod are not equal, false otherwise.
        """
        return self._value != other._value

    fn __is__(self, other: Self) -> Bool:
        """Checks if two ReductionMethod are identical.

        Args:
            other: The other ReductionMethod to compare.

        Returns:
            True if the ReductionMethod are identical, false otherwise.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Checks if two ReductionMethod are not identical.

        Args:
            other: The other ReductionMethod to compare.

        Returns:
            True if the ReductionMethod are not identical, false otherwise.
        """
        return self != other


@always_inline
fn sum[
    intermediate_type: DType,
    *,
    reduction_method: warp.ReductionMethod,
    output_type: DType,
](x: SIMD) -> Scalar[output_type]:
    """Performs a warp-level reduction to compute the sum of values across threads.

    This function provides two reduction methods:
    1. Warp shuffle: Uses warp shuffle operations to efficiently sum values across threads
    2. Tensor core: Leverages tensor cores for high-performance reductions, with type casting

    The tensor core method will cast the input to the specified intermediate type before
    reduction to ensure compatibility with tensor core operations. The warp shuffle method
    requires the output type to match the input type.

    Parameters:
        intermediate_type: The data type to cast to when using tensor core reduction.
        reduction_method: `WARP` for warp shuffle or `TENSOR_CORE` for tensor core reduction.
        output_type: The desired output data type for the reduced value.

    Args:
        x: The SIMD value to reduce across the warp.

    Returns:
        A scalar containing the sum of the input values across all threads in the warp,
        cast to the specified output type.

    Constraints:
        - For warp shuffle reduction, output_type must match the input value type.
        - For tensor core reduction, input will be cast to intermediate_type.
    """

    @parameter
    if reduction_method is ReductionMethod.WARP:
        constrained[
            output_type == x.dtype,
            (
                "the output type must match the input value for warp-level"
                " reduction"
            ),
        ]()

        return rebind[Scalar[output_type]](sum(x.reduce_add()))
    else:
        return tc_reduce[output_type](x.cast[intermediate_type]())


# ===-----------------------------------------------------------------------===#
# Warp Prefix Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn prefix_sum[
    type: DType, //,
    intermediate_type: DType = type,
    *,
    output_type: DType = type,
    exclusive: Bool = False,
](x: Scalar[type]) -> Scalar[output_type]:
    """Computes a warp-level prefix sum (scan) operation.

    Performs an inclusive or exclusive prefix sum across threads in a warp using
    a parallel scan algorithm with warp shuffle operations. This implements an
    efficient parallel scan with logarithmic complexity.

    For example, if we have a warp with the following elements:
    $$$
    [x_0, x_1, x_2, x_3, x_4]
    $$$

    The prefix sum is:
    $$$
    [x_0, x_0 + x_1, x_0 + x_1 + x_2, x_0 + x_1 + x_2 + x_3, x_0 + x_1 + x_2 + x_3 + x_4]
    $$$

    Parameters:
        type: The data type of the input SIMD elements.
        intermediate_type: Type used for intermediate calculations (defaults to
                          input type).
        output_type: The desired output data type (defaults to input type).
        exclusive: If True, performs exclusive scan where each thread receives
                   the sum of all previous threads. If False (default), performs
                   inclusive scan where each thread receives the sum including
                   its own value.

    Args:
        x: The SIMD value to include in the prefix sum.

    Returns:
        A scalar containing the prefix sum at the current thread's position in
        the warp, cast to the specified output type.
    """
    var res = x.cast[intermediate_type]().reduce_add()

    @parameter
    for i in range(log2_floor(WARP_SIZE)):
        alias offset = 1 << i
        var n = shuffle_up(res, offset)
        if lane_id() >= offset:
            res += n

    @parameter
    if exclusive:
        res = shuffle_up(res, 1)
        if lane_id() == 0:
            res = 0

    return res.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Warp Max
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _has_redux_f32_support[dtype: DType, simd_width: Int]() -> Bool:
    return _is_sm_100x_or_newer() and dtype is DType.float32 and simd_width == 1


@always_inline("nodebug")
fn _redux_f32_max_min[direction: StaticString](val: SIMD) -> __type_of(val):
    alias instruction = StaticString("redux.sync.") + direction + ".NaN.f32"
    return inlined_assembly[
        instruction + " $0, $1, $2;",
        __type_of(val),
        constraints="=r,r,i",
        has_side_effect=True,
    ](val, Int32(_FULL_MASK))


@always_inline
fn lane_group_max[
    val_type: DType,
    simd_width: Int, //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its maximum within a lane group using warp-level operations.

    This function performs a parallel reduction across a group of lanes to find the maximum value.
    The reduction is done using warp shuffle operations for efficient communication between lanes.
    The result is stored in all participating lanes.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the maximum.

    Returns:
        A SIMD value where all participating lanes contain the maximum value found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """

    @parameter
    if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["max"](val)

    @parameter
    fn _reduce_max(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return _max(x, y)

    return lane_group_reduce[
        shuffle_down, _reduce_max, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn lane_group_max_and_broadcast[
    val_type: DType,
    simd_width: Int, //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces and broadcasts the maximum value within a lane group using warp-level operations.

    This function performs a parallel reduction to find the maximum value and broadcasts it to all lanes.
    The reduction and broadcast are done using warp shuffle operations in a butterfly pattern for
    efficient all-to-all communication between lanes.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce and broadcast. Each lane contributes its value.

    Returns:
        A SIMD value where all participating lanes contain the maximum value found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """

    @parameter
    if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["max"](val)

    @parameter
    fn _reduce_max(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return _max(x, y)

    return lane_group_reduce[
        shuffle_xor, _reduce_max, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn max[
    val_type: DType,
    simd_width: Int, //,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the maximum value across all lanes in a warp.

    This is a convenience wrapper around lane_group_max that operates on the entire warp.
    It performs a parallel reduction using warp shuffle operations to find the global maximum
    value across all lanes in the warp.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the maximum.

    Returns:
        A SIMD value where all lanes contain the maximum value found across the entire warp.
    """
    return lane_group_max[num_lanes=WARP_SIZE](val)


# ===-----------------------------------------------------------------------===#
# Warp Min
# ===-----------------------------------------------------------------------===#


@always_inline
fn lane_group_min[
    val_type: DType,
    simd_width: Int, //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its minimum within a lane group using warp-level operations.

    This function performs a parallel reduction across a group of lanes to find the minimum value.
    The reduction is done using warp shuffle operations for efficient communication between lanes.
    The result is stored in all participating lanes.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the minimum.

    Returns:
        A SIMD value where all participating lanes contain the minimum value found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """

    @parameter
    if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["min"](val)

    @parameter
    fn _reduce_min(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return _min(x, y)

    return lane_group_reduce[
        shuffle_down, _reduce_min, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn min[
    val_type: DType, simd_width: Int, //
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the minimum value across all lanes in a warp.

    This is a convenience wrapper around lane_group_min that operates on the entire warp.
    It performs a parallel reduction using warp shuffle operations to find the global minimum
    value across all lanes in the warp.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the minimum.

    Returns:
        A SIMD value where all lanes contain the minimum value found across the entire warp.
        The minimum value is broadcast to all lanes.
    """
    return lane_group_min[num_lanes=WARP_SIZE](val)


# ===-----------------------------------------------------------------------===#
# Warp Broadcast
# ===-----------------------------------------------------------------------===#


@always_inline
fn broadcast[
    val_type: DType, simd_width: Int, //
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Broadcasts a SIMD value from lane 0 to all lanes in the warp.

    This function takes a SIMD value from lane 0 and copies it to all other lanes in the warp,
    effectively broadcasting the value across the entire warp. This is useful for sharing data
    between threads in a warp without using shared memory.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.

    Args:
        val: The SIMD value to broadcast from lane 0.

    Returns:
        A SIMD value where all lanes contain a copy of the input value from lane 0.
    """
    return shuffle_idx(val, 0)


fn broadcast(val: Int) -> Int:
    """Broadcasts an integer value from lane 0 to all lanes in the warp.

    This function takes an integer value from lane 0 and copies it to all other lanes in the warp.
    It provides a convenient way to share scalar integer data between threads without using shared memory.

    Args:
        val: The integer value to broadcast from lane 0.

    Returns:
        The broadcast integer value, where all lanes receive a copy of the input from lane 0.
    """
    return Int(shuffle_idx(Int32(val), 0))


fn broadcast(val: UInt) -> UInt:
    """Broadcasts an unsigned integer value from lane 0 to all lanes in the warp.

    This function takes an unsigned integer value from lane 0 and copies it to all other lanes in the warp.
    It provides a convenient way to share scalar unsigned integer data between threads without using shared memory.

    Args:
        val: The unsigned integer value to broadcast from lane 0.

    Returns:
        The broadcast unsigned integer value, where all lanes receive a copy of the input from lane 0.
    """
    return Int(shuffle_idx(Int32(val), 0))
