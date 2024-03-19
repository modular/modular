# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs shuffle instructions."""

from sys import llvm_intrinsic

from .globals import WARP_SIZE

# TODO (#24457): support shuffles with width != 32
alias _WIDTH_MASK = WARP_SIZE - 1

# shfl.sync.up.b32 prepares this mask differently from other shuffle intrinsics
alias _WIDTH_MASK_SHUFFLE_UP = 0


@always_inline("nodebug")
fn _is_half_like[type: DType]() -> Bool:
    return type.is_bfloat16() or type.is_float16()


# ===----------------------------------------------------------------------===#
# shuffle_idx
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_idx[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width], src_lane: Int) -> SIMD[type, simd_width]:
    """Copies a value from a source lane to other lanes in a warp.

    Broadcasts a value from a source thread in a warp to all the participating
    threads without the use of shared memory

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      val: The value to be shuffled.
      src_lane: The offset warp lane ID.

    Returns:
      The value from the src_lane.
    """
    return shuffle_idx(0xFFFFFFFF, val, src_lane)


@always_inline("nodebug")
fn shuffle_idx[
    type: DType, simd_width: Int
](
    mask: Int,
    val: SIMD[type, simd_width],
    src_lane: Int,
    width: Int = WARP_SIZE - 1,
) -> SIMD[type, simd_width]:
    """Copies a value from a source lane to other lanes in a warp.

    Broadcasts a value from a source thread in a warp to all the participating
    threads without the use of shared memory

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      src_lane: The source warp lane ID.
      width: The warp width which must be a power of 2.

    Returns:
      The value from the src_lane.
    """
    constrained[
        _is_half_like[type]() or simd_width == 1,
        "Unsupported simd_width",
    ]()

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.idx.f32", Scalar[type]](
            Int32(mask), val, UInt32(src_lane), Int32(_WIDTH_MASK)
        )
    elif type.is_int32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.idx.i32", Scalar[type]](
            Int32(mask), val, UInt32(src_lane), Int32(_WIDTH_MASK)
        )
    elif _is_half_like[type]():

        @parameter
        if simd_width == 1:
            # splat and recurse to meet 32 bitwidth requirements
            var splatted_val = SIMD[type, 2](rebind[SIMD[type, 1]](val))
            return shuffle_idx(mask, splatted_val, src_lane, width)[0]
        else:
            # bitcast and recurse to use i32 intrinsic
            var packed_val = bitcast[DType.int32, 1](val)
            var result_packed = shuffle_idx(mask, packed_val, src_lane, width)
            return bitcast[type, simd_width](result_packed)

    else:
        constrained[False, "unhandled type"]()
        return 0


# ===----------------------------------------------------------------------===#
# shuffle_up
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_up[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width], offset: Int) -> SIMD[type, simd_width]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    lower lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the specified offset.
    """
    return shuffle_up(0xFFFFFFFF, val, offset)


@always_inline("nodebug")
fn shuffle_up[
    type: DType, simd_width: Int
](
    mask: Int,
    val: SIMD[type, simd_width],
    offset: Int,
    width: Int = WARP_SIZE - 1,
) -> SIMD[type, simd_width]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    lower lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      offset: The offset warp lane ID.
      width: The warp width which must be a power of 2.

    Returns:
      The value at the specified offset.
    """
    constrained[
        _is_half_like[type]() or simd_width == 1,
        "Unsupported simd_width",
    ]()

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.up.f32", Scalar[type]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK_SHUFFLE_UP)
        )
    elif type.is_int32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.up.i32", Scalar[type]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK_SHUFFLE_UP)
        )
    elif _is_half_like[type]():

        @parameter
        if simd_width == 1:
            # splat and recurse to meet 32 bitwidth requirements
            var splatted_val = SIMD[type, 2](rebind[SIMD[type, 1]](val))
            return shuffle_up(mask, splatted_val, offset, width)[0]
        else:
            # bitcast and recurse to use i32 intrinsic
            var packed_val = bitcast[DType.int32, 1](val)
            var result_packed = shuffle_up(mask, packed_val, offset, width)
            return bitcast[type, simd_width](result_packed)
    else:
        constrained[False, "unhandled type"]()
        return 0


# ===----------------------------------------------------------------------===#
# shuffle_down
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_down[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width], offset: Int) -> SIMD[type, simd_width]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    higher lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the specified offset.
    """
    return shuffle_down(0xFFFFFFFF, val, offset)


@always_inline("nodebug")
fn shuffle_down[
    type: DType, simd_width: Int
](mask: Int, val: SIMD[type, simd_width], offset: Int) -> SIMD[
    type, simd_width
]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    higher lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the specified offset.
    """
    constrained[
        _is_half_like[type]() or simd_width == 1,
        "Unsupported simd_width",
    ]()

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.down.f32", Scalar[type]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )
    elif type.is_int32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.down.i32", Scalar[type]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )
    elif _is_half_like[type]():

        @parameter
        if simd_width == 1:
            # splat and recurse to meet 32 bitwidth requirements
            var splatted_val = SIMD[type, 2](rebind[SIMD[type, 1]](val))
            return shuffle_down(mask, splatted_val, offset)[0]
        else:
            # bitcast and recurse to use i32 intrinsic:
            var packed_val = bitcast[DType.int32, 1](val)
            var result_packed = shuffle_down(mask, packed_val, offset)
            return bitcast[type, simd_width](result_packed)
    else:
        constrained[False, "unhandled type"]()
        return 0


# ===----------------------------------------------------------------------===#
# shuffle_xor
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_xor[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width], offset: Int,) -> SIMD[type, simd_width]:
    """Copies values from between lanes (butterfly pattern).

    Exchange a value between threads within a warp by copying from a thread
    based on the bitwise xor of the offset with the thread's own lane id.

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the lane based on bitwise XOR of own lane id.
    """
    return shuffle_xor(0xFFFFFFFF, val, offset)


@always_inline("nodebug")
fn shuffle_xor[
    type: DType, simd_width: Int
](mask: Int, val: SIMD[type, simd_width], offset: Int) -> SIMD[
    type, simd_width
]:
    """Copies values from between lanes (butterfly pattern).

    Exchange a value between threads within a warp by copying from a thread
    based on the bitwise xor of the offset with the thread's own lane id.

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the lane based on bitwise XOR of own lane id.
    """
    constrained[
        _is_half_like[type]() or simd_width == 1,
        "Unsupported simd_width",
    ]()

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.bfly.f32", Scalar[type]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )
    elif type.is_int32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.bfly.i32", Scalar[type]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )
    elif _is_half_like[type]():

        @parameter
        if simd_width == 1:
            # splat and recurse to meet 32 bitwidth requirements
            var splatted_val = SIMD[type, 2](rebind[SIMD[type, 1]](val))
            return shuffle_xor(mask, splatted_val, offset)[0]
        else:
            # bitcast and recurse to use i32 intrinsic:
            var packed_val = bitcast[DType.int32, 1](val)
            var result_packed = shuffle_xor(mask, packed_val, offset)
            return bitcast[type, simd_width](result_packed)
    else:
        constrained[False, "unhandled type"]()
        return 0


# ===----------------------------------------------------------------------===#
# Warp Reduction
# ===----------------------------------------------------------------------===#


@always_inline
fn _floorlog2[n: Int]() -> Int:
    return 0 if n <= 1 else 1 + _floorlog2[n >> 1]()


@always_inline
fn _static_log2[n: Int]() -> Int:
    return 0 if n <= 1 else _floorlog2[n - 1]() + 1


@always_inline("nodebug")
fn warp_reduce[
    shuffle: fn[type: DType, simd_width: Int] (
        val: SIMD[type, simd_width], offset: Int
    ) -> SIMD[type, simd_width],
    func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    val_type: DType,
    simd_width: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Takes in an input function to computes warp shuffle based reduction operation.
    """
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    @unroll
    for mask in range(limit - 1, -1, -1):
        res = func(res, shuffle(res, 1 << mask))

    return res
