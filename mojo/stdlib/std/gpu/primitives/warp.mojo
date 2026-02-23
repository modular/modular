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

from sys import (
    CompilationTarget,
    bit_width_of,
    is_amd_gpu,
    is_apple_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    size_of,
    _RegisterPackType,
)
from sys._assembly import inlined_assembly
from sys.info import _is_sm_100x_or_newer, _cdna_4_or_newer

from bit import log2_floor
from math.math import max as _max, min as _min
from gpu import lane_id
from gpu.intrinsics import permlane_shuffle
from gpu.globals import WARP_SIZE
from memory import bitcast

from ..compute.tensor_ops import tc_reduce

# TODO (#24457): support shuffles with width != 32
comptime _WIDTH_MASK = WARP_SIZE - 1
comptime _FULL_MASK = UInt(2**WARP_SIZE - 1)

# shfl.sync.up.b32 prepares this mask differently from other shuffle intrinsics
comptime _WIDTH_MASK_SHUFFLE_UP = 0

# Common function type for binary SIMD reduction operations (add, max, min).
comptime _ReduceFn = fn[dtype: DType, width: Int](
    SIMD[dtype, width], SIMD[dtype, width]
) capturing -> SIMD[dtype, width]


# ===-----------------------------------------------------------------------===#
# AMD DPP (Data Parallel Primitives) intrinsics
# ===-----------------------------------------------------------------------===#


@always_inline
fn _dpp_update_i32[
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


@always_inline
fn _dpp_move[
    dtype: DType, simd_width: Int, //, dpp_ctrl: Int
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
        var src = bitcast[DType.int32, 1](val)
        var neighbor = _dpp_update_i32[dpp_ctrl](Int32(0), src)
        return bitcast[dtype, simd_width](neighbor)
    elif bit_width_of[dtype]() == 16 and simd_width == 1:
        var splatted = SIMD[dtype, 2](val._refine[new_size=1]())
        var result = _dpp_move[dpp_ctrl](splatted)
        return result[0]
    elif bit_width_of[dtype]() == 64 and simd_width == 1:
        var parts = bitcast[DType.int32, 2](val)
        var lo = _dpp_update_i32[dpp_ctrl](Int32(0), parts[0])
        var hi = _dpp_update_i32[dpp_ctrl](Int32(0), parts[1])
        return bitcast[dtype, 1](SIMD[DType.int32, 2](lo, hi))
    else:
        comptime assert False, "unsupported type for DPP move"
        return val


@always_inline
fn _dpp_reduce_and_broadcast[
    dtype: DType,
    simd_width: Int,
    //,
    func: _ReduceFn,
    num_lanes: Int = WARP_SIZE,
](val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
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
        func: Binary reduction function (e.g. add, max, min).
        num_lanes: Number of lanes in the reduction group (must be power of 2,
            2..WARP_SIZE).

    Args:
        val: The value to reduce across the lane group.

    Returns:
        The reduction result across the lane group, broadcast to every lane
        in the group.
    """
    constrained[
        num_lanes >= 2 and num_lanes.is_power_of_two(),
        "num_lanes must be a power of 2 >= 2",
    ]()
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
    @always_inline
    fn _cross_row_step[
        shuffle_width: Int
    ](v: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
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


# ===-----------------------------------------------------------------------===#
# utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _shuffle[
    mnemonic: StringSlice,
    dtype: DType,
    simd_width: Int,
    *,
    WIDTH_MASK: Int32,
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    comptime assert (
        dtype.is_half_float() or simd_width == 1
    ), "Unsupported simd_width"

    comptime if dtype == DType.float32:
        return llvm_intrinsic[
            "llvm.nvvm.shfl.sync." + mnemonic + ".f32", Scalar[dtype]
        ](Int32(mask), val, offset, WIDTH_MASK)
    elif dtype in (DType.int32, DType.uint32):
        return llvm_intrinsic[
            "llvm.nvvm.shfl.sync." + mnemonic + ".i32", Scalar[dtype]
        ](Int32(mask), val, offset, WIDTH_MASK)
    elif dtype in (DType.int64, DType.uint64):
        var val_bitcast = bitcast[DType.uint32, simd_width * 2](val)
        var val_half1, val_half2 = val_bitcast.deinterleave()
        var shuffle1 = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val_half1, offset
        )
        var shuffle2 = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val_half2, offset
        )
        var result = shuffle1.interleave(shuffle2)
        return bitcast[dtype, simd_width](result)
    elif dtype.is_half_float():
        comptime if simd_width == 1:
            # splat and recurse to meet 32 bitwidth requirements
            var splatted_val = SIMD[dtype, 2](val._refine[new_size=1]())
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
            return bitcast[dtype, simd_width](result_packed)
    elif dtype == DType.bool:
        comptime assert simd_width == 1, "unhandled simd width"
        return _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val.cast[DType.int32](), offset
        ).cast[dtype]()

    else:
        comptime assert False, "unhandled shuffle dtype"
        return 0


@always_inline
fn _shuffle_amd_helper[
    dtype: DType, simd_width: Int
](dst_lane: UInt32, val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime if size_of[SIMD[dtype, simd_width]]() == 4:
        # Handle int32, float32, float16x2, etc.
        var result_packed = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](
            dst_lane * 4, bitcast[DType.int32, 1](val)
        )
        return bitcast[dtype, simd_width](result_packed)
    else:
        comptime assert simd_width == 1, "Unsupported simd width"

        comptime if dtype == DType.bool:
            return _shuffle_amd_helper(dst_lane, val.cast[DType.int32]()).cast[
                dtype
            ]()
        elif bit_width_of[dtype]() == 16:
            var val_splatted = SIMD[dtype, 2](val._refine[new_size=1]())
            return _shuffle_amd_helper(dst_lane, val_splatted)[0]
        elif bit_width_of[dtype]() == 64:
            var val_bitcast = bitcast[DType.uint32, simd_width * 2](val)
            var val_half1, val_half2 = val_bitcast.deinterleave()
            var shuffle1 = _shuffle_amd_helper(dst_lane, val_half1)
            var shuffle2 = _shuffle_amd_helper(dst_lane, val_half2)
            var result = shuffle1.interleave(shuffle2)
            return bitcast[dtype, simd_width](result)
        else:
            comptime assert False, "unhandled shuffle dtype"
            return 0


@always_inline
fn _shuffle_apple_helper[
    op: StringSlice, dtype: DType, simd_width: Int
](
    mask: UInt,  # Ignored, for API parity
    val: SIMD[dtype, simd_width],
    offset: UInt32,
) -> SIMD[dtype, simd_width]:
    """
    Mapping from Metal stdlib to AIR (LLVM) intrinsics:
      Metal                         → AIR intrinsic stem
      ----------------------------------------------------------
      simd_shuffle                  → llvm.air.simd_shuffle
      simd_shuffle_down             → llvm.air.simd_shuffle_down
      simd_shuffle_up               → llvm.air.simd_shuffle_up
      simd_shuffle_xor              → llvm.air.simd_shuffle_xor
    """

    comptime assert (
        dtype.is_half_float() or simd_width == 1
    ), "Unsupported simd_width"

    var arg = UInt16(offset)  # AIR intrinsics use 16-bit offsets

    comptime if dtype in (DType.int64, DType.uint64):
        var bits = bitcast[DType.uint32, simd_width * 2](val)
        var half1, half2 = bits.deinterleave()

        var half1_n = rebind[SIMD[DType.uint32, simd_width]](half1)
        var half2_n = rebind[SIMD[DType.uint32, simd_width]](half2)
        var s1 = _shuffle_apple_helper[op, DType.uint32, simd_width](
            mask, half1_n, offset
        )
        var s2 = _shuffle_apple_helper[op, DType.uint32, simd_width](
            mask, half2_n, offset
        )

        var merged = s1.interleave(s2)
        return bitcast[dtype, simd_width](merged)
    elif dtype == DType.bool:
        var val1 = rebind[SIMD[DType.int32, 1]](val.cast[DType.int32]())
        var tmp = _shuffle_apple_helper[op, DType.int32, 1](mask, val1, offset)
        return tmp.cast[dtype]()
    elif (
        dtype == DType.bfloat16
    ):  # bfloat16 is declared in MSL but actually causes a backend error.
        comptime if simd_width == 1:
            var pair = SIMD[dtype, 2](val._refine[new_size=1]())
            var pair_i32 = bitcast[DType.int32, 1](pair)
            var y_i32 = _shuffle_apple_helper[op, DType.int32, 1](
                mask, pair_i32, offset
            )
            return bitcast[dtype, 2](y_i32)[0]
        else:
            var packed = bitcast[DType.int32, simd_width // 2](val)
            var packed_shuf = _shuffle_apple_helper[
                op, DType.int32, simd_width // 2
            ](mask, packed, offset)
            return bitcast[dtype, simd_width](packed_shuf)
    else:
        comptime name = "llvm.air.simd_shuffle" + (
            "" if op == "indexed" else "_" + op
        )
        return llvm_intrinsic[name, SIMD[dtype, simd_width]](val, arg)


# ===-----------------------------------------------------------------------===#
# shuffle_idx
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_idx[
    dtype: DType, simd_width: Int, //
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
            from gpu import shuffle_idx

            val = SIMD[DType.float32, 16](1.0)

            # Broadcast value from lane 0 to all lanes
            result = shuffle_idx(val, 0)

            # Get value from lane 5
            result = shuffle_idx(val, 5)
        ```
    """
    return shuffle_idx(_FULL_MASK, val, offset)


@always_inline
fn _shuffle_idx_amd[
    dtype: DType, simd_width: Int, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane = Int32(lane_id())
    # Godbolt uses 0x3fffffc0. It is masking out the lower 64-bits
    # But it's also masking out the upper two bits. Why?
    # The lane should not be > 64 so the upper 2 bits should always be zero.
    # Use -64 for now.
    var t0 = lane & Int32(-WARP_SIZE)
    var dst_lane = t0 | offset.cast[DType.int32]()
    return _shuffle_amd_helper(UInt32(dst_lane), val)


@always_inline
fn shuffle_idx[
    dtype: DType, simd_width: Int, //
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
            from gpu import shuffle_idx

            # Only broadcast to first 16 lanes
            var mask = 0xFFFF  # 16 ones
            var val = SIMD[DType.float32, 32](1.0)
            var result = shuffle_idx(mask, val, 5)
        ```
    """

    comptime if is_nvidia_gpu():
        return _shuffle[
            "idx",
            WIDTH_MASK = Int32(_WIDTH_MASK),
        ](mask, val, offset)
    elif is_amd_gpu():
        return _shuffle_idx_amd(mask, val, offset)
    elif is_apple_gpu():
        return _shuffle_apple_helper["indexed", dtype, simd_width](
            mask, val, offset
        )
    else:
        return CompilationTarget.unsupported_target_error[
            SIMD[dtype, simd_width],
            operation = __get_current_function_name(),
        ]()


# ===-----------------------------------------------------------------------===#
# shuffle_up
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_up[
    dtype: DType, simd_width: Int, //
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


@always_inline
fn _shuffle_up_amd[
    dtype: DType, simd_width: Int, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane = Int32(lane_id())
    var t0 = lane - offset.cast[DType.int32]()
    var t1 = lane & Int32(-WARP_SIZE)
    var dst_lane = t0.lt(t1).select(lane, t0)
    return _shuffle_amd_helper(UInt32(dst_lane), val)


@always_inline
fn shuffle_up[
    dtype: DType, simd_width: Int, //
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

    comptime if is_nvidia_gpu():
        return _shuffle["up", WIDTH_MASK=_WIDTH_MASK_SHUFFLE_UP](
            mask, val, offset
        )
    elif is_amd_gpu():
        return _shuffle_up_amd(mask, val, offset)
    elif is_apple_gpu():
        return _shuffle_apple_helper["up", dtype, simd_width](mask, val, offset)
    else:
        return CompilationTarget.unsupported_target_error[
            SIMD[dtype, simd_width],
            operation = __get_current_function_name(),
        ]()


# ===-----------------------------------------------------------------------===#
# shuffle_down
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_down[
    dtype: DType, simd_width: Int, //
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


@always_inline
fn _shuffle_down_amd[
    dtype: DType, simd_width: Int, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane = UInt32(lane_id())
    # set the offset to 0 if lane + offset >= WARP_SIZE
    var dst_lane = (lane + offset).le(UInt32(_WIDTH_MASK)).select(
        offset, 0
    ) + lane
    return _shuffle_amd_helper(dst_lane, val)


@always_inline
fn shuffle_down[
    dtype: DType, simd_width: Int, //
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

    comptime if is_nvidia_gpu():
        return _shuffle["down", WIDTH_MASK = Int32(_WIDTH_MASK)](
            mask, val, offset
        )
    elif is_amd_gpu():
        return _shuffle_down_amd(mask, val, offset)
    elif is_apple_gpu():
        return _shuffle_apple_helper["down", dtype, simd_width](
            mask, val, offset
        )
    else:
        return CompilationTarget.unsupported_target_error[
            SIMD[dtype, simd_width],
            operation = __get_current_function_name(),
        ]()


# ===-----------------------------------------------------------------------===#
# shuffle_xor
# ===-----------------------------------------------------------------------===#


@always_inline
fn shuffle_xor[
    dtype: DType, simd_width: Int, //
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


@always_inline
fn _shuffle_xor_amd[
    dtype: DType, simd_width: Int, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane = UInt32(lane_id())
    var t0 = lane ^ offset
    var t1 = lane & UInt32(-WARP_SIZE)
    # This needs to be "add nsw" = add no sign wrap
    var t2 = t1 + UInt32(WARP_SIZE)
    var dst_lane = t0.lt(t2).select(t0, lane)
    return _shuffle_amd_helper(dst_lane, val)


@always_inline
fn shuffle_xor[
    dtype: DType, simd_width: Int, //
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
            from gpu import shuffle_xor

            # Exchange values between even-numbered threads 4 lanes apart
            mask = 0xAAAAAAAA  # Even threads only
            var val = SIMD[DType.float32, 16](42.0)  # Example value
            result = shuffle_xor(mask, val, 4.0)
        ```
    """

    comptime if is_nvidia_gpu():
        return _shuffle["bfly", WIDTH_MASK = Int32(_WIDTH_MASK)](
            mask, val, offset
        )
    elif is_amd_gpu():
        return _shuffle_xor_amd(mask, val, offset)
    elif is_apple_gpu():
        return _shuffle_apple_helper["xor", dtype, simd_width](
            mask, val, offset
        )
    else:
        return CompilationTarget.unsupported_target_error[
            SIMD[dtype, simd_width],
            operation = __get_current_function_name(),
        ]()


# ===-----------------------------------------------------------------------===#
# Warp Reduction
# ===-----------------------------------------------------------------------===#


@always_inline
fn lane_group_reduce[
    val_type: DType,
    simd_width: Int,
    //,
    shuffle: fn[dtype: DType, simd_width: Int](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) -> SIMD[dtype, simd_width],
    func: _ReduceFn,
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
            from gpu import lane_group_reduce, shuffle_down

            # Compute sum across 16 threads using shuffle down
            @parameter
            fn add[dtype: DType, width: Int](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
                return x + y
            var val = SIMD[DType.float32, 16](42.0)
            var result = lane_group_reduce[shuffle_down, add, num_lanes=16](val)
        ```
    """
    var res = val

    comptime limit = log2_floor(num_lanes)

    comptime for i in reversed(range(limit)):
        comptime offset = 1 << i
        res = func(res, shuffle(res, UInt32(offset * stride)))

    return res


@always_inline
fn reduce[
    val_type: DType,
    simd_width: Int,
    //,
    shuffle: fn[dtype: DType, simd_width: Int](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) -> SIMD[dtype, simd_width],
    func: _ReduceFn,
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
        from gpu import reduce, shuffle_down

        # Compute warp-wide sum using shuffle down
        @parameter
        fn add[dtype: DType, width: Int](x: SIMD[dtype, width], y: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
            return x + y

        val = SIMD[DType.float32, 4](2.0, 4.0, 6.0, 8.0)
        result = reduce[shuffle_down, add](val)
    ```
    """
    return lane_group_reduce[shuffle, func, num_lanes=WARP_SIZE](val)


# ===-----------------------------------------------------------------------===#
# Warp Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn lane_group_sum[
    val_type: DType,
    simd_width: Int,
    //,
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
    fn _reduce_add(x: SIMD, y: type_of(x)) -> type_of(x):
        return x + y

    return lane_group_reduce[
        shuffle_down, _reduce_add, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn lane_group_sum_and_broadcast[
    val_type: DType,
    simd_width: Int,
    //,
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
    fn _reduce_add(x: SIMD, y: type_of(x)) -> type_of(x):
        return x + y

    comptime if (
        num_lanes == WARP_SIZE // stride
        and stride in (16, 32)
        and _cdna_4_or_newer()
    ):
        var out = _reduce_add(val, permlane_shuffle[32](val))

        comptime if stride == 16:
            out = _reduce_add(out, permlane_shuffle[16](out))

        return out
    elif (
        stride == 1
        and num_lanes >= 2
        and num_lanes.is_power_of_two()
        and is_amd_gpu()
    ):
        return _dpp_reduce_and_broadcast[_reduce_add, num_lanes=num_lanes](val)
    else:
        return lane_group_reduce[
            shuffle_xor, _reduce_add, num_lanes=num_lanes, stride=stride
        ](val)


@always_inline
fn sum(val: SIMD) -> Scalar[val.dtype]:
    """Computes the sum of values across all lanes in a warp.

    This is a convenience wrapper around lane_group_sum_and_broadcast that
    operates on the entire warp.  It performs a parallel reduction using warp
    shuffle operations to find the global sum across all lanes in the warp.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to the sum.

    Returns:
        The scalar sum of values across all lanes in the warp.
    """
    return lane_group_sum_and_broadcast[num_lanes=WARP_SIZE](val.reduce_add())


# ===-----------------------------------------------------------------------===#
# Warp Prefix Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn prefix_sum[
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
    var res = x.cast[intermediate_type]().reduce_add()

    var lane = lane_id()

    comptime for i in range(log2_floor(WARP_SIZE)):
        comptime offset = 1 << i
        var n = shuffle_up(res, UInt32(offset))
        if lane >= UInt(offset):
            res += n

    comptime if exclusive:
        res = shuffle_up(res, 1)
        if lane == 0:
            res = 0

    return res.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Warp Max
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _has_redux_f32_support[dtype: DType, simd_width: Int]() -> Bool:
    return _is_sm_100x_or_newer() and dtype == DType.float32 and simd_width == 1


@always_inline("nodebug")
fn _redux_f32_max_min[direction: StaticString](val: SIMD) -> type_of(val):
    comptime instruction = StaticString("redux.sync.") + direction + ".NaN.f32"
    return inlined_assembly[
        instruction + " $0, $1, $2;",
        type_of(val),
        constraints="=r,r,i",
        has_side_effect=True,
    ](val, Int32(_FULL_MASK))


@always_inline
fn lane_group_max[
    val_type: DType,
    simd_width: Int,
    //,
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

    comptime if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["max"](val)

    @parameter
    fn _reduce_max(x: SIMD, y: type_of(x)) -> type_of(x):
        return _max(x, y)

    return lane_group_reduce[
        shuffle_down, _reduce_max, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn lane_group_max_and_broadcast[
    val_type: DType,
    simd_width: Int,
    //,
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

    comptime if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["max"](val)

    @parameter
    fn _reduce_max(x: SIMD, y: type_of(x)) -> type_of(x):
        return _max(x, y)

    comptime if (
        num_lanes == WARP_SIZE // stride
        and stride in (16, 32)
        and _cdna_4_or_newer()
    ):
        var out = _reduce_max(val, permlane_shuffle[32](val))

        comptime if stride == 16:
            out = _reduce_max(out, permlane_shuffle[16](out))

        return out
    elif (
        stride == 1
        and num_lanes >= 2
        and num_lanes.is_power_of_two()
        and is_amd_gpu()
    ):
        return _dpp_reduce_and_broadcast[_reduce_max, num_lanes=num_lanes](val)
    else:
        return lane_group_reduce[
            shuffle_xor, _reduce_max, num_lanes=num_lanes, stride=stride
        ](val)


@always_inline
fn max(val: SIMD) -> Scalar[val.dtype]:
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


@always_inline
fn lane_group_min[
    val_type: DType,
    simd_width: Int,
    //,
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

    comptime if (
        _has_redux_f32_support[val_type, simd_width]()
        and num_lanes == WARP_SIZE
    ):
        return _redux_f32_max_min["min"](val)

    @parameter
    fn _reduce_min(x: SIMD, y: type_of(x)) -> type_of(x):
        return _min(x, y)

    return lane_group_reduce[
        shuffle_down, _reduce_min, num_lanes=num_lanes, stride=stride
    ](val)


@always_inline
fn min(val: SIMD) -> Scalar[val.dtype]:
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
    return UInt(Int(shuffle_idx(Int32(val), 0)))


# ===-----------------------------------------------------------------------===#
# Warp Vote
# ===-----------------------------------------------------------------------===#


@always_inline
fn _vote_nvidia_helper(vote: Bool) -> UInt32:
    return llvm_intrinsic[
        "llvm.nvvm.vote.ballot.sync",
        UInt32,
        UInt32,
        Bool,
        has_side_effect=False,
    ](0xFFFFFFFF, vote).cast[DType.uint32]()


@always_inline
fn _vote_amd_helper[ret_type: DType](vote: Bool) -> Scalar[ret_type]:
    comptime assert ret_type in (
        DType.uint32,
        DType.uint64,
    ), "Unsupported return type"

    comptime instruction = String(
        "llvm.amdgcn.ballot.i", bit_width_of[ret_type]()
    )
    return llvm_intrinsic[
        instruction,
        Scalar[ret_type],
        has_side_effect=False,
    ](vote)


@always_inline
fn vote[ret_type: DType](val: Bool) -> Scalar[ret_type]:
    """Creates a 32 or 64 bit mask among all threads in the warp, where each bit is set to 1 if the
    corresponding thread voted True, and 0 otherwise.

    This function takes a boolean value which represents the cooresponding threads vote.

    Nvidia only supports 32 bit masks, while AMD supports 32 and 64 bit masks.

    Parameters:
        ret_type: Return type for the mask (must be `DType.uint32` or `DType.uint64`).

    Args:
        val: The boolean vote.

    Returns:
        A mask containing the vote of all threads in the warp.
    """

    comptime if is_nvidia_gpu():
        comptime assert ret_type == DType.uint32, "Unsupported return type"
        return rebind[Scalar[ret_type]](_vote_nvidia_helper(val))
    elif is_amd_gpu():
        return _vote_amd_helper[ret_type](val)
    else:
        return CompilationTarget.unsupported_target_error[
            Scalar[ret_type], operation = __get_current_function_name()
        ]()
