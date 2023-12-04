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

# ===----------------------------------------------------------------------===#
# shuffle_idx
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_idx[type: DType](val: SIMD[type, 1], srcLane: Int) -> SIMD[type, 1]:
    """Copies a value from a source lane to other lanes in a warp.

    Broadcasts a value from a source thread in a warp to all the participating
    threads without the use of shared memory

    Parameters:
      type: The type of the scalar value.

    Args:
      val: The value to be shuffled.
      srcLane: The offset warp lane ID.

    Returns:
      The value from the srcLane.
    """
    return shuffle_idx(0xFFFFFFFF, val, srcLane)


@always_inline("nodebug")
fn shuffle_idx[
    type: DType
](
    mask: Int, val: SIMD[type, 1], srcLane: Int, width: Int = WARP_SIZE - 1
) -> SIMD[type, 1]:
    """Copies a value from a source lane to other lanes in a warp.

    Broadcasts a value from a source thread in a warp to all the participating
    threads without the use of shared memory

    Parameters:
      type: The type of the scalar value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      srcLane: The source warp lane ID.
      width: The warp width which must be a power of 2.

    Returns:
      The value from the srcLane.
    """

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.idx.f32", SIMD[type, 1]](
            Int32(mask), val, UInt32(srcLane), Int32(_WIDTH_MASK)
        )
    else:
        return llvm_intrinsic["llvm.nvvm.shfl.sync.idx.i32", SIMD[type, 1]](
            Int32(mask), val, UInt32(srcLane), Int32(_WIDTH_MASK)
        )


# ===----------------------------------------------------------------------===#
# shuffle_up
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_up[type: DType](val: SIMD[type, 1], offset: Int) -> SIMD[type, 1]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    lower lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the scalar value.

    Args:
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the specified offset.
    """
    return shuffle_up(0xFFFFFFFF, val, offset)


@always_inline("nodebug")
fn shuffle_up[
    type: DType
](
    mask: Int, val: SIMD[type, 1], offset: Int, width: Int = WARP_SIZE - 1
) -> SIMD[type, 1]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    lower lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the scalar value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      offset: The offset warp lane ID.
      width: The warp width which must be a power of 2.

    Returns:
      The value at the specified offset.
    """

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.up.f32", SIMD[type, 1]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )
    else:
        return llvm_intrinsic["llvm.nvvm.shfl.sync.up.i32", SIMD[type, 1]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )


# ===----------------------------------------------------------------------===#
# shuffle_down
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_down[type: DType](val: SIMD[type, 1], offset: Int) -> SIMD[type, 1]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    higher lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the scalar value.

    Args:
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the specified offset.
    """
    return shuffle_down(0xFFFFFFFF, val, offset)


@always_inline("nodebug")
fn shuffle_down[
    type: DType
](mask: Int, val: SIMD[type, 1], offset: Int) -> SIMD[type, 1]:
    """Copies values from other lanes in the warp.

    Exchange a value between threads within a warp by copying from a thread with
    higher lane id relative to the caller without the use of shared memory.

    Parameters:
      type: The type of the scalar value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the specified offset.
    """

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.down.f32", SIMD[type, 1]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )
    else:
        return llvm_intrinsic["llvm.nvvm.shfl.sync.down.i32", SIMD[type, 1]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )


# ===----------------------------------------------------------------------===#
# shuffle_xor
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_xor[
    type: DType
](val: SIMD[type, 1], offset: Int,) -> SIMD[type, 1]:
    """Copies values from between lanes (butterfly pattern).

    Exchange a value between threads within a warp by copying from a thread
    based on the bitwise xor of the offset with the thread's own lane id.

    Parameters:
      type: The type of the scalar value.

    Args:
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the lane based on bitwise XOR of own lane id.
    """
    return shuffle_xor(0xFFFFFFFF, val, offset)


@always_inline("nodebug")
fn shuffle_xor[
    type: DType
](mask: Int, val: SIMD[type, 1], offset: Int) -> SIMD[type, 1]:
    """Copies values from between lanes (butterfly pattern).

    Exchange a value between threads within a warp by copying from a thread
    based on the bitwise xor of the offset with the thread's own lane id.

    Parameters:
      type: The type of the scalar value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value at the lane based on bitwise XOR of own lane id.
    """

    @parameter
    if type.is_float32():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.bfly.f32", SIMD[type, 1]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )
    else:
        return llvm_intrinsic["llvm.nvvm.shfl.sync.bfly.i32", SIMD[type, 1]](
            Int32(mask), val, UInt32(offset), Int32(_WIDTH_MASK)
        )


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
    shuffle: fn[type: DType] (val: SIMD[type, 1], offset: Int) -> SIMD[type, 1],
    func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    val_type: DType,
](val: SIMD[val_type, 1]) -> SIMD[val_type, 1]:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    @unroll
    for mask in range(limit - 1, -1, -1):
        res = func(res, shuffle(res, 1 << mask))

    return res
