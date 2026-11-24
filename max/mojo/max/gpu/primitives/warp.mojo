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

from std.sys import (
    bit_width_of,
    is_amd_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    size_of,
)
from std.sys._assembly import inlined_assembly
from std.sys.info import _cdna_4_or_newer, _cdna_5_or_newer

from std.bit import log2_floor
from std.math.math import max as _max, min as _min
from std._gpu import lane_id
from std.memory import bitcast

from max._plugin import CurrentMaxPlugin
from max.gpu.globals import WARP_SIZE
from max.gpu.intrinsics import permlane_shuffle

comptime _FULL_MASK = UInt(2**WARP_SIZE - 1)

# Common function type for binary SIMD reduction operations (add, max, min).
comptime _ReduceFn = def[dtype: DType, width: SIMDLength](
    SIMD[dtype, width], SIMD[dtype, width]
) thin -> SIMD[dtype, width]

comptime _reduce_fn_signature = def[dtype: DType, width: SIMDLength](
    SIMD[dtype, width], SIMD[dtype, width]
) -> SIMD[dtype, width]

# Value form of `_ReduceFn`, taken as a runtime closure argument.
comptime _ReduceClosure = (
    ImplicitlyCopyable & RegisterPassable & _reduce_fn_signature
)


# ===-----------------------------------------------------------------------===#
# AMD DPP (Data Parallel Primitives) intrinsics
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _dpp_update_i32[
    dpp_ctrl: Int,
    row_mask: Int = 0xF,
    bank_mask: Int = 0xF,
    bound_ctrl: Bool = True,
](old: Int32, src: Int32) -> Int32:
    """Performs a DPP (Data Parallel Primitives) cross-lane operation on AMD GPUs.

    This wraps llvm.amdgcn.update.dpp.i32 to move data between lanes at
    register-level bandwidth, avoiding LDS-based ds_bpermute.

    Parameters:
        dpp_ctrl: DPP control word specifying the cross-lane pattern.
        row_mask: 4-bit mask selecting which rows (of 16 lanes) participate.
        bank_mask: 4-bit mask selecting which banks (of 4 lanes) participate.
        bound_ctrl: If True, out-of-range source lanes produce 0 instead of
            old.

    Args:
        old: Fallback value for masked-out or out-of-range lanes.
        src: Source value to read from neighboring lanes via DPP.

    Returns:
        The value read from the source lane specified by dpp_ctrl, or the
        fallback value.
    """
    return llvm_intrinsic[
        "llvm.amdgcn.update.dpp.i32",
        Int32,
    ](
        old,
        src,
        Int32(dpp_ctrl),
        Int32(row_mask),
        Int32(bank_mask),
        bound_ctrl,
    )


@inline(.always)
def _dpp_move[
    dtype: DType, simd_width: SIMDLength, //, dpp_ctrl: Int
](val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Returns a neighboring lane's value via a DPP cross-lane operation.

    This is the pure data-movement primitive: it applies the DPP control word
    and returns the value from the source lane without combining it with the
    current value. The caller applies its own reduction function.

    Since this uses bound_ctrl=True, out-of-range sources return 0. For the
    rotation-based reduction pattern (quad_perm, row_ror) all sources are
    always in-range so the fallback is never reached.

    Parameters:
        dtype: The data type of the SIMD elements.
        simd_width: The number of elements in the SIMD vector.
        dpp_ctrl: DPP control word specifying the cross-lane pattern.

    Args:
        val: The value whose neighboring lane copy is requested.

    Returns:
        The value from the source lane specified by dpp_ctrl.
    """
    comptime if size_of[SIMD[dtype, simd_width]]() == 4:
        var src = bitcast[.int32, 1](val)
        var neighbor = _dpp_update_i32[dpp_ctrl](Int32(0), src)
        return bitcast[dtype, simd_width](neighbor)
    elif bit_width_of[dtype]() == 16 and simd_width == 1:
        var splatted = SIMD[dtype, 2](val._refine[new_size=1]())
        var result = _dpp_move[dpp_ctrl](splatted)
        return result[0]
    elif bit_width_of[dtype]() == 64 and simd_width == 1:
        var parts = bitcast[.int32, 2](val)
        var lo = _dpp_update_i32[dpp_ctrl](Int32(0), parts[0])
        var hi = _dpp_update_i32[dpp_ctrl](Int32(0), parts[1])
        return bitcast[dtype, 1](SIMD[.int32, 2](lo, hi))
    else:
        comptime assert False, "unsupported type for DPP move"


@inline(.always)
def _dpp_reduce_and_broadcast[
    dtype: DType,
    simd_width: SIMDLength,
    FuncType: _ReduceClosure,
    //,
    num_lanes: Int = WARP_SIZE,
](val: SIMD[dtype, simd_width], func: FuncType) -> SIMD[dtype, simd_width]:
    """Performs a DPP-based reduction and broadcast on AMD GPUs.

    Uses AMD DPP instructions for intra-row (16-lane) reduction and shuffle_xor
    for cross-row reduction. This is significantly lower latency compared to
    ds_bpermute-based shuffles for the intra-row portion.

    The reduction uses a rotation-based pattern where all source lanes are
    always in-range, so this works correctly for any associative+commutative
    reduction (sum, max, min, etc.):
    1. Quad permutations give all lanes 2-wide then 4-wide partial results.
    2. Row rotations (ror:4, ror:8) give all lanes 16-wide row results.
       Rotation wraps within each 16-lane row, so every lane accumulates
       the full row result without needing a separate broadcast step.
    3. Shuffle XOR handles cross-row communication for 32 and 64-wide
       results.

    For sub-warp sizes the chain is truncated: num_lanes=2 uses 1 DPP step,
    num_lanes=4 uses 2, num_lanes=8 uses 3, etc.

    Parameters:
        dtype: The data type of the SIMD elements.
        simd_width: The number of elements in the SIMD vector.
        FuncType: The type of the reduction closure.
        num_lanes: Number of lanes in the reduction group (must be power of 2,
            2..WARP_SIZE).

    Args:
        val: The value to reduce across the lane group.
        func: Binary reduction function (e.g. add, max, min).

    Returns:
        The reduction result across the lane group, broadcast to every lane
        in the group.
    """
    comptime assert num_lanes >= 2 and Bool(
        num_lanes.is_power_of_two()
    ), "num_lanes must be a power of 2 >= 2"

    comptime assert num_lanes <= WARP_SIZE, "num_lanes cannot exceed WARP_SIZE"

    # DPP control constants for the reduction pattern.
    comptime _DPP_QUAD_PERM_1032 = 0xB1  # quad_perm:[1,0,3,2] - swap pairs
    comptime _DPP_QUAD_PERM_2301 = 0x4E  # quad_perm:[2,3,0,1] - swap halves
    comptime _DPP_ROW_HALF_MIRROR = 0x141  # row_half_mirror - mirror within 8-lane halves
    comptime _DPP_ROW_ROR_8 = 0x128  # row_ror:8 - rotate right by 8 within row

    var out = val

    # Step 1: quad_perm swap pairs → 2-wide results.
    out = func(out, _dpp_move[_DPP_QUAD_PERM_1032](out))

    # Step 2: quad_perm swap halves → 4-wide results.
    comptime if num_lanes >= 4:
        out = func(out, _dpp_move[_DPP_QUAD_PERM_2301](out))

    # Steps 3-4: Intra-row reduction.
    # row_half_mirror mirrors within each 8-lane half, producing 8-wide
    # results. For num_lanes >= 16, an additional row_ror:8 yields the
    # full 16-wide row result.
    comptime if num_lanes >= 8:
        out = func(out, _dpp_move[_DPP_ROW_HALF_MIRROR](out))
    comptime if num_lanes >= 16:
        out = func(out, _dpp_move[_DPP_ROW_ROR_8](out))

    # Steps 5-6: Cross-row reduction.
    # On CDNA4+ use permlane_shuffle (register-level) instead of
    # shuffle_xor (ds_bpermute through LDS). permlane_shuffle only
    # supports 32-bit operands, so fall back to shuffle_xor for wider types.
    @inline(.always)
    def _cross_row_step[
        shuffle_width: Int
    ](v: SIMD[dtype, simd_width]) {imm func} -> SIMD[dtype, simd_width]:
        comptime if _cdna_4_or_newer() and size_of[
            SIMD[dtype, simd_width]
        ]() == 4:
            return func(v, permlane_shuffle[shuffle_width](v))
        else:
            return func(v, shuffle_xor(v, UInt32(shuffle_width)))

    comptime if num_lanes >= 32:
        out = _cross_row_step[16](out)
    comptime if num_lanes >= 64:
        out = _cross_row_step[32](out)

    return out


@inline(.always)
def _dpp_prefix_sum[
    dtype: DType, //, exclusive: Bool
](val: Scalar[dtype]) -> Scalar[dtype]:
    comptime assert _cdna_4_or_newer(), "Requires CDNA4 or newer"

    comptime _DPP_ROW_SHR_1 = 0x111
    comptime _DPP_ROW_SHR_2 = 0x112
    comptime _DPP_ROW_SHR_4 = 0x114
    comptime _DPP_ROW_SHR_8 = 0x118

    var out = val
    var lane = lane_id()
    var row_lane = lane % 16

    # Steps 1-4: Intra-row prefix sum.
    var shr_1 = _dpp_move[_DPP_ROW_SHR_1](out)
    if row_lane >= 1:
        out += shr_1

    var shr_2 = _dpp_move[_DPP_ROW_SHR_2](out)
    if row_lane >= 2:
        out += shr_2

    var shr_4 = _dpp_move[_DPP_ROW_SHR_4](out)
    if row_lane >= 4:
        out += shr_4

    var shr_8 = _dpp_move[_DPP_ROW_SHR_8](out)
    if row_lane >= 8:
        out += shr_8

    comptime if _cdna_5_or_newer():
        # Step 5: Cross-row prefix sum propagation.
        var bcast_15 = shuffle_idx(out, 15)
        if lane >= 16:
            out += bcast_15

        # Optionally shift up for exclusive mode.
        comptime if exclusive:
            out = shuffle_up(out, 1)
            if lane == 0:
                out = 0
    else:
        comptime _DPP_WAVE_SHR_1 = 0x138
        comptime _DPP_ROW_BCAST_15 = 0x142
        comptime _DPP_ROW_BCAST_31 = 0x143

        # Steps 5-6: Cross-row prefix sum propagation.
        var bcast_15 = _dpp_move[_DPP_ROW_BCAST_15](out)
        if (lane % 32) >= 16:
            out += bcast_15

        var bcast_31 = _dpp_move[_DPP_ROW_BCAST_31](out)
        if lane >= 32:
            out += bcast_31

        # Optionally shift up for exclusive mode.
        comptime if exclusive:
            out = _dpp_move[_DPP_WAVE_SHR_1](out)

    return out


# ===-----------------------------------------------------------------------===#
# shuffle_idx
# ===-----------------------------------------------------------------------===#


@inline(.always)
def shuffle_idx[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Copies a value from a source lane to other lanes in a warp.

        Broadcasts a value from a source thread in a warp to all participating threads
        without using shared memory. This is a convenience wrapper that uses the full
        warp mask by default.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32, half).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be broadcast from the source lane.
        offset: The source lane ID to copy the value from.

    Returns:
        A SIMD vector where all lanes contain the value from the source lane specified by offset.

    Example:

        ```mojo
            from max.gpu.primitives.warp import shuffle_idx

            val = SIMD[.float32, 16](1.0)

            # Broadcast value from lane 0 to all lanes
            result = shuffle_idx(val, 0)

            # Get value from lane 5
            result = shuffle_idx(val, 5)
        ```
    """
    return shuffle_idx(_FULL_MASK, val, offset)


@inline(.always)
def shuffle_idx[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    """Copies a value from a source lane to other lanes in a warp with explicit mask control.

        Broadcasts a value from a source thread in a warp to participating threads specified by
        the mask. This provides fine-grained control over which threads participate in the shuffle
        operation.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32, half).
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
            from max.gpu.primitives.warp import shuffle_idx

            # Only broadcast to first 16 lanes
            var mask: UInt = 0xFFFF  # 16 ones
            var val = SIMD[.float32, 32](1.0)
            var result = shuffle_idx(mask, val, 5)
        ```
    """

    return CurrentMaxPlugin.shuffle_idx_fn(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_up
# ===-----------------------------------------------------------------------===#


@inline(.always)
def shuffle_up[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Copies values from threads with lower lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    lower lane ID, offset by the specified amount. Uses the full warp mask by default.

    For example, with offset=1:
    - Thread N gets value from thread N-1
    - Thread 1 gets value from thread 0
    - Thread 0 gets undefined value

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be shuffled up the warp.
        offset: The number of lanes to shift values up by.

    Returns:
        The SIMD value from the thread offset lanes lower in the warp.
        Returns undefined values for threads where lane_id - offset < 0.
    """
    return shuffle_up(_FULL_MASK, val, offset)


@inline(.always)
def shuffle_up[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
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
        dtype: The data type of the SIMD elements (e.g. float32, int32).
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

    return CurrentMaxPlugin.shuffle_up_fn(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_down
# ===-----------------------------------------------------------------------===#


@inline(.always)
def shuffle_down[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Copies values from threads with higher lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    higher lane ID, offset by the specified amount. Uses the full warp mask by default.

    For example, with offset=1:
    - Thread 0 gets value from thread 1
    - Thread 1 gets value from thread 2
    - Thread N gets value from thread N+1
    - Last N threads get undefined values

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be shuffled down the warp.
        offset: The number of lanes to shift values down by. Must be positive.

    Returns:
        The SIMD value from the thread offset lanes higher in the warp.
        Returns undefined values for threads where lane_id + offset >= WARP_SIZE.
    """
    return shuffle_down(_FULL_MASK, val, offset)


@inline(.always)
def shuffle_down[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
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
        dtype: The data type of the SIMD elements (e.g. float32, int32).
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

    return CurrentMaxPlugin.shuffle_down_fn(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_xor
# ===-----------------------------------------------------------------------===#


@inline(.always)
def shuffle_xor[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Exchanges values between threads in a warp using a butterfly pattern.

    Performs a butterfly exchange pattern where each thread swaps values with another thread
    whose lane ID differs by a bitwise XOR with the given offset. This creates a butterfly
    communication pattern useful for parallel reductions and scans.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be exchanged with another thread.
        offset: The lane offset to XOR with the current thread's lane ID to determine
               the exchange partner. Common values are powers of 2 for butterfly patterns.

    Returns:
        The SIMD value from the thread at lane (current_lane XOR offset).
    """
    return shuffle_xor(_FULL_MASK, val, offset)


@inline(.always)
def shuffle_xor[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    """Exchanges values between threads in a warp using a butterfly pattern with masking.

    Performs a butterfly exchange pattern where each thread swaps values with another thread
    whose lane ID differs by a bitwise XOR with the given offset. The mask parameter allows
    controlling which threads participate in the exchange.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
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
            from max.gpu.primitives.warp import shuffle_xor

            # Exchange values between even-numbered threads 4 lanes apart
            var mask: UInt = 0xAAAAAAAA  # Even threads only
            var val = SIMD[.float32, 16](42.0)  # Example value
            var result = shuffle_xor(mask, val, 4)
        ```
    """

    return CurrentMaxPlugin.shuffle_xor_fn(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# Warp Reduction
# ===-----------------------------------------------------------------------===#


@inline(.always)
def lane_group_reduce[
    val_type: DType,
    simd_width: SIMDLength,
    FuncType: _ReduceClosure,
    //,
    shuffle: def[dtype: DType, simd_width: SIMDLength](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) thin -> SIMD[dtype, simd_width],
    num_lanes: Int,
    *,
    stride: Int = 1,
](val: SIMD[val_type, simd_width], func: FuncType) -> SIMD[
    val_type, simd_width
]:
    """Performs a generic warp-level reduction operation using shuffle operations.

    This function implements a parallel reduction across threads in a warp using a butterfly
    pattern. It allows customizing both the shuffle operation and reduction function.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        FuncType: The type of the reduction closure.
        shuffle: A function that performs the warp shuffle operation. Takes a SIMD value and
                offset and returns the shuffled result.
        num_lanes: The number of lanes in a group. The reduction is done within each group. Must be a power of 2.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value.
        func: A binary function that combines two SIMD values during reduction. This defines
              the reduction operation (e.g. add, max, min).

    Returns:
        A SIMD value containing the reduction result.

    Example:

        ```mojo
            from max.gpu.primitives.warp import lane_group_reduce, shuffle_down

            # Compute sum across 16 threads using shuffle down
            def add[dtype: DType, width: SIMDLength](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
                return x + y
            var val = SIMD[.float32, 16](42.0)
            var result = lane_group_reduce[shuffle_down, num_lanes=16](val, add)
        ```
    """
    var res = val

    comptime limit = log2_floor(num_lanes)

    comptime for i in reversed(range(limit)):
        comptime offset = 1 << i
        res = func(res, shuffle(res, UInt32(offset * stride)))

    return res


@inline(.always)
def reduce[
    val_type: DType,
    simd_width: SIMDLength,
    FuncType: _ReduceClosure,
    //,
    shuffle: def[dtype: DType, simd_width: SIMDLength](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) thin -> SIMD[dtype, simd_width],
](val: SIMD[val_type, simd_width], func: FuncType) -> SIMD[
    val_type, simd_width
]:
    """Performs a generic warp-wide reduction operation using shuffle operations.

    This is a convenience wrapper around lane_group_reduce that operates on the entire warp.
    It allows customizing both the shuffle operation and reduction function.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        FuncType: The type of the reduction closure.
        shuffle: A function that performs the warp shuffle operation. Takes a SIMD value and
                offset and returns the shuffled result.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value.
        func: A binary function that combines two SIMD values during reduction. This defines
              the reduction operation (e.g. add, max, min).

    Returns:
        A SIMD value containing the reduction result broadcast to all lanes in the warp.

    Example:

    ```mojo
        from max.gpu.primitives.warp import reduce, shuffle_down

        # Compute warp-wide sum using shuffle down
        def add[dtype: DType, width: SIMDLength](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
            return x + y

        val = SIMD[.float32, 4](2.0, 4.0, 6.0, 8.0)
        result = reduce[shuffle_down](val, add)
    ```
    """
    return lane_group_reduce[shuffle, num_lanes=WARP_SIZE](val, func)


# ===-----------------------------------------------------------------------===#
# Shared broadcast-reduce dispatch
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _lane_group_broadcast_reduce[
    val_type: DType,
    simd_width: SIMDLength,
    FuncType: _ReduceClosure,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width], func: FuncType) -> SIMD[
    val_type, simd_width
]:
    """Shared broadcast-reduce dispatch: CDNA4+ permlane, AMD DPP, or
    shuffle_xor fallback."""
    comptime if (
        is_amd_gpu() and num_lanes >= 2 and Bool(num_lanes.is_power_of_two())
    ):
        comptime if (
            num_lanes == WARP_SIZE // stride
            and stride in (16, 32)
            and _cdna_4_or_newer()
        ):
            var out = val

            comptime if WARP_SIZE == 64:
                out = func(out, permlane_shuffle[32](out))

            comptime if stride == 16:
                out = func(out, permlane_shuffle[16](out))

            return out

        comptime if stride == 1:
            return _dpp_reduce_and_broadcast[num_lanes=num_lanes](val, func)

    return lane_group_reduce[shuffle_xor, num_lanes=num_lanes, stride=stride](
        val, func
    )


# ===-----------------------------------------------------------------------===#
# Warp Sum
# ===-----------------------------------------------------------------------===#


@inline(.always)
def lane_group_sum[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the sum of values across a group of lanes and broadcasts to all lanes.

    This function performs a parallel reduction across a group of lanes to compute their sum.
    The result is broadcast to all participating lanes using optimized hardware-specific
    paths (AMD DPP, Blackwell redux, or butterfly shuffle pattern).

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

    def _reduce_add(x: SIMD, y: type_of(x)) -> type_of(x):
        return x + y

    return _lane_group_broadcast_reduce[num_lanes=num_lanes, stride=stride](
        val, _reduce_add
    )


@inline(.always)
def sum(val: SIMD) -> Scalar[val.dtype]:
    """Computes the sum of values across all lanes in a warp.

    This is a convenience wrapper around `lane_group_sum` that operates on the
    entire warp. It performs a parallel reduction using warp shuffle operations
    to find the global sum across all lanes in the warp.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to the sum.

    Returns:
        The scalar sum of values across all lanes in the warp.
    """
    return lane_group_sum[num_lanes=WARP_SIZE](val.reduce_add())


# ===-----------------------------------------------------------------------===#
# Warp Prefix Sum
# ===-----------------------------------------------------------------------===#


@inline(.always)
def prefix_sum[
    dtype: DType,
    //,
    intermediate_type: DType = dtype,
    *,
    output_type: DType = dtype,
    exclusive: Bool = False,
](x: Scalar[dtype]) -> Scalar[output_type]:
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
        dtype: The data type of the input SIMD elements.
        intermediate_type: Type used for intermediate calculations (defaults to
                          input dtype).
        output_type: The desired output data type (defaults to input dtype).
        exclusive: If True, performs exclusive scan where each thread receives
                   the sum of all previous threads. If False (default), performs
                   inclusive scan where each thread receives the sum including
                   its own value.

    Args:
        x: The SIMD value to include in the prefix sum.

    Returns:
        A scalar containing the prefix sum at the current thread's position in
        the warp, cast to the specified output dtype.
    """
    var res = x.cast[intermediate_type]()

    comptime if _cdna_4_or_newer():
        res = _dpp_prefix_sum[exclusive](res)
    else:
        var lane = lane_id()

        comptime for i in range(log2_floor(WARP_SIZE)):
            comptime offset = 1 << i
            var n = shuffle_up(res, UInt32(offset))
            if lane >= offset:
                res += n

        comptime if exclusive:
            res = shuffle_up(res, 1)
            if lane == 0:
                res = 0

    return res.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Warp Max
# ===-----------------------------------------------------------------------===#


@inline(.nodebug)
def _has_redux_f32_support[dtype: DType, simd_width: Int]() -> Bool:
    return (
        (is_nvidia_gpu["sm_100a"]() or is_nvidia_gpu["sm_101a"]())
        and dtype == .float32
        and simd_width == 1
    )


@inline(.nodebug)
def _redux_f32_max_min[direction: StaticString](val: SIMD) -> type_of(val):
    comptime instruction = StaticString("redux.sync.") + direction + ".NaN.f32"
    return inlined_assembly[
        instruction + " $0, $1, $2;",
        type_of(val),
        constraints="=f,f,i",
        has_side_effect=True,
    ](val, Int32(_FULL_MASK))


@inline(.always)
def lane_group_max[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its maximum within a lane group and broadcasts to all lanes.

    This function performs a parallel reduction across a group of lanes to find the maximum value.
    The result is broadcast to all participating lanes using optimized hardware-specific
    paths (AMD DPP, Blackwell redux, or butterfly shuffle pattern).

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

    comptime if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["max"](val)

    def _reduce_max(x: SIMD, y: type_of(x)) -> type_of(x):
        return _max(x, y)

    return _lane_group_broadcast_reduce[num_lanes=num_lanes, stride=stride](
        val, _reduce_max
    )


@inline(.always)
def max(val: SIMD) -> Scalar[val.dtype]:
    """Computes the maximum value across all lanes in a warp.

    This is a convenience wrapper around lane_group_max that operates on the entire warp.
    It performs a parallel reduction using warp shuffle operations to find the global maximum
    value across all lanes in the warp.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the maximum.

    Returns:
        The scalar maximum value across all lanes in the warp.
    """
    return lane_group_max[num_lanes=WARP_SIZE](val.reduce_max())


# ===-----------------------------------------------------------------------===#
# Warp Min
# ===-----------------------------------------------------------------------===#


@inline(.always)
def lane_group_min[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its minimum within a lane group and broadcasts to all lanes.

    This function performs a parallel reduction across a group of lanes to find the minimum value.
    The result is broadcast to all participating lanes using optimized hardware-specific
    paths (AMD DPP, Blackwell redux, or butterfly shuffle pattern).

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

    comptime if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["min"](val)

    def _reduce_min(x: SIMD, y: type_of(x)) -> type_of(x):
        return _min(x, y)

    return _lane_group_broadcast_reduce[num_lanes=num_lanes, stride=stride](
        val, _reduce_min
    )


@inline(.always)
def min(val: SIMD) -> Scalar[val.dtype]:
    """Computes the minimum value across all lanes in a warp.

    This is a convenience wrapper around lane_group_min that operates on the entire warp.
    It performs a parallel reduction using warp shuffle operations to find the global minimum
    value across all lanes in the warp.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the minimum.

    Returns:
        The scalar minimum value across all lanes in the warp.
    """
    return lane_group_min[num_lanes=WARP_SIZE](val.reduce_min())


# ===-----------------------------------------------------------------------===#
# Warp Broadcast
# ===-----------------------------------------------------------------------===#


@inline(.always)
def broadcast[
    val_type: DType, simd_width: SIMDLength, //
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


# ===-----------------------------------------------------------------------===#
# Warp Vote
# ===-----------------------------------------------------------------------===#


@inline(.always)
def vote[ret_type: DType](val: Bool) -> Scalar[ret_type]:
    """Creates a 32 or 64 bit mask among all threads in the warp, where each bit is set to 1 if the
    corresponding thread voted True, and 0 otherwise.

    This function takes a boolean value which represents the corresponding threads vote.

    NVIDIA supports 32-bit masks; AMD supports 32- and 64-bit masks; Apple
    Silicon (a 32-lane SIMD-group) supports 32-bit masks, and also accepts a
    `DType.uint64` return whose upper 32 bits are always zero.

    Parameters:
        ret_type: Return type for the mask (must be `DType.uint32` or `DType.uint64`).

    Args:
        val: The boolean vote.

    Returns:
        A mask containing the vote of all threads in the warp.
    """

    comptime assert ret_type in (
        DType.uint32,
        DType.uint64,
    ), "vote ret_type must be DType.uint32 or DType.uint64"
    return CurrentMaxPlugin.vote_fn[ret_type](val)


# ===-----------------------------------------------------------------------===#
# match_any
# ===-----------------------------------------------------------------------===#


@inline(.always)
def match_any[
    dtype: DType,
    //,
    mask_type: DType = (DType.uint32 if WARP_SIZE <= 32 else DType.uint64),
](value: Scalar[dtype]) -> Scalar[mask_type]:
    """Finds, for each lane, the mask of warp lanes whose `value` bits match it.

    Returns a per-lane lane mask whose bit `l` is set for every active lane `l`
    whose `value` has the same bit pattern as the calling lane's. The comparison
    is on the bits (matching NVIDIA's `match.any.sync`), so `0.0` and `-0.0` do
    not match while two `NaN`s with equal bits do. This is the fold a warp uses
    to coalesce same-keyed lanes (a histogram or scatter leader handling a whole
    group in one non-atomic update) instead of one atomic per lane.

    All `WARP_SIZE` lanes must reach the call converged.

    Example:

        ```mojo
        from max.gpu.primitives.warp import match_any

        # If lanes 0, 3, 7 hold the same value, each of them gets a mask with
        # bits 0, 3, and 7 set; the remaining lanes get their own groups.
        var my_key = Int32(42)
        var group = match_any(my_key)
        ```

    Parameters:
        dtype: The element type of `value` (inferred from the argument).
        mask_type: The lane-mask return type, `DType.uint32` or `DType.uint64`
            (defaults to the type matching `WARP_SIZE`).

    Args:
        value: The calling lane's value to match against the rest of the warp.

    Returns:
        A `mask_type` lane mask with bit `l` set for each active lane `l` holding
        a bit-equal `value`.

    Constraints:
        Requires a GPU target with warp shuffles. `dtype` must be a 32- or
        64-bit type and `mask_type` must be `DType.uint32` or `DType.uint64`
        (NVIDIA returns a 32-bit mask, so `mask_type` must be `DType.uint32`
        there).
    """
    comptime assert mask_type in (
        DType.uint32,
        DType.uint64,
    ), "match_any mask_type must be DType.uint32 or DType.uint64"
    comptime assert size_of[dtype]() in (
        4,
        8,
    ), "match_any value must be a 32- or 64-bit type"
    comptime bits_type = DType.uint32 if size_of[dtype]() == 4 else DType.uint64
    var bits = bitcast[bits_type](value)

    return CurrentMaxPlugin.match_any_fn[mask_type](bits)


# ===-----------------------------------------------------------------------===#
# match_all
# ===-----------------------------------------------------------------------===#


@inline(.always)
def match_all[
    dtype: DType,
    //,
    mask_type: DType = (DType.uint32 if WARP_SIZE <= 32 else DType.uint64),
](value: Scalar[dtype]) -> Scalar[mask_type]:
    """Returns the warp's active-lane mask if all lanes share `value`, else 0.

    When every active lane holds the same bits as the calling lane, returns the
    mask of those lanes (so a non-zero result is the "all agree" predicate that
    NVIDIA's `match.all.sync` also exposes); otherwise returns 0. The comparison
    is on the bits, so `0.0` and `-0.0` are treated as different. This is the
    dual of `match_any`: it reports warp-wide agreement on a key.

    All `WARP_SIZE` lanes must reach the call converged.

    Example:

        ```mojo
        from max.gpu.primitives.warp import match_all

        # `agreed` is non-zero (the active-lane mask) iff every lane passed the
        # same `key`.
        var key = Int32(42)
        var agreed = match_all(key)
        ```

    Parameters:
        dtype: The element type of `value` (inferred from the argument).
        mask_type: The lane-mask return type, `DType.uint32` or `DType.uint64`
            (defaults to the type matching `WARP_SIZE`).

    Args:
        value: The calling lane's value to compare against the rest of the warp.

    Returns:
        A `mask_type` lane mask of the active lanes when they all hold a
        bit-equal `value`, otherwise 0.

    Constraints:
        Requires a GPU target with warp shuffles. `dtype` must be a 32- or
        64-bit type and `mask_type` must be `DType.uint32` or `DType.uint64`
        (NVIDIA returns a 32-bit mask, so `mask_type` must be `DType.uint32`
        there).
    """
    comptime assert mask_type in (
        DType.uint32,
        DType.uint64,
    ), "match_all mask_type must be DType.uint32 or DType.uint64"
    comptime assert size_of[dtype]() in (
        4,
        8,
    ), "match_all value must be a 32- or 64-bit type"
    comptime bits_type = DType.uint32 if size_of[dtype]() == 4 else DType.uint64
    var bits = bitcast[bits_type](value)

    return CurrentMaxPlugin.match_all_fn[mask_type](bits)
