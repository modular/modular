# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs shuffle instructions."""

from sys import llvm_intrinsic

from memory import bitcast

from .globals import WARP_SIZE
from .tensor_ops import tc_reduce

# TODO (#24457): support shuffles with width != 32
alias _WIDTH_MASK = WARP_SIZE - 1

# shfl.sync.up.b32 prepares this mask differently from other shuffle intrinsics
alias _WIDTH_MASK_SHUFFLE_UP = 0


# ===----------------------------------------------------------------------===#
# utilities
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _shuffle[
    mnemonic: StringLiteral,
    type: DType,
    simd_width: Int,
    *,
    WIDTH_MASK: Int32,
](mask: Int, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
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
    elif type is DType.int32:
        return llvm_intrinsic[
            "llvm.nvvm.shfl.sync." + mnemonic + ".i32", Scalar[type]
        ](Int32(mask), val, offset, WIDTH_MASK)
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

    else:
        constrained[False, "unhandled type"]()
        return 0


# ===----------------------------------------------------------------------===#
# shuffle_idx
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_idx[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
    """Copies a value from a source lane to other lanes in a warp.

    Broadcasts a value from a source thread in a warp to all the participating
    threads without the use of shared memory

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      val: The value to be shuffled.
      offset: The offset warp lane ID.

    Returns:
      The value from the offset.
    """
    return shuffle_idx(0xFFFFFFFF, val, offset)


@always_inline("nodebug")
fn shuffle_idx[
    type: DType, simd_width: Int, //
](mask: Int, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    """Copies a value from a source lane to other lanes in a warp.

    Broadcasts a value from a source thread in a warp to all the participating
    threads without the use of shared memory

    Parameters:
      type: The type of the simd value.
      simd_width: The width of the simd value.

    Args:
      mask: The mask of the warp lanes.
      val: The value to be shuffled.
      offset: The source warp lane ID.

    Returns:
      The value from the offset.
    """
    return _shuffle[
        "idx",
        WIDTH_MASK=_WIDTH_MASK,
    ](mask, val, offset)


# ===----------------------------------------------------------------------===#
# shuffle_up
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_up[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
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
    type: DType, simd_width: Int, //
](mask: Int, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
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

    Returns:
      The value at the specified offset.
    """
    return _shuffle["up", WIDTH_MASK=_WIDTH_MASK_SHUFFLE_UP](mask, val, offset)


# ===----------------------------------------------------------------------===#
# shuffle_down
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_down[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
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
    type: DType, simd_width: Int, //
](mask: Int, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
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
    return _shuffle["down", WIDTH_MASK=_WIDTH_MASK](mask, val, offset)


# ===----------------------------------------------------------------------===#
# shuffle_xor
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn shuffle_xor[
    type: DType, simd_width: Int, //
](val: SIMD[type, simd_width], offset: UInt32) -> SIMD[type, simd_width]:
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
    type: DType, simd_width: Int, //
](mask: Int, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
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
    return _shuffle["bfly", WIDTH_MASK=_WIDTH_MASK](mask, val, offset)


# ===----------------------------------------------------------------------===#
# Warp Reduction
# ===----------------------------------------------------------------------===#


@always_inline
fn _floorlog2[n: Int]() -> Int:
    return 0 if n <= 1 else 1 + _floorlog2[n >> 1]()


@always_inline
fn _static_log2[n: Int]() -> Int:
    return 0 if n <= 1 else _floorlog2[n - 1]() + 1


@always_inline
fn lane_group_reduce[
    shuffle: fn[type: DType, simd_width: Int] (
        val: SIMD[type, simd_width], offset: UInt32
    ) -> SIMD[type, simd_width],
    func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    val_type: DType,
    simd_width: Int,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Takes in an input function to computes warp shuffle based reduction operation.
    """
    var res = val

    alias limit = _static_log2[nthreads]()

    @parameter
    for i in reversed(range(limit)):
        alias offset = 1 << i
        res = func(res, shuffle(res, offset))

    return res


@always_inline
fn warp_reduce[
    shuffle: fn[type: DType, simd_width: Int] (
        val: SIMD[type, simd_width], offset: UInt32
    ) -> SIMD[type, simd_width],
    func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    val_type: DType,
    simd_width: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Takes in an input function to computes warp shuffle based reduction operation.
    """
    return lane_group_reduce[
        shuffle, func, val_type, simd_width, nthreads=WARP_SIZE
    ](val)


# ===----------------------------------------------------------------------===#
# Warp Sum
# ===----------------------------------------------------------------------===#


@always_inline
fn lane_group_sum[
    val_type: DType,
    simd_width: Int,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    @parameter
    fn _reduce_add(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return x + y

    return lane_group_reduce[shuffle_down, _reduce_add, nthreads=nthreads](val)


@always_inline
fn lane_group_max[
    val_type: DType,
    simd_width: Int,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    @parameter
    fn _reduce_max(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return max(x, y)

    return lane_group_reduce[shuffle_down, _reduce_max, nthreads=nthreads](val)


@always_inline
fn warp_sum[
    val_type: DType, simd_width: Int
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    return lane_group_sum[nthreads=WARP_SIZE](val)


@always_inline
fn warp_max[
    val_type: DType, simd_width: Int
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    return lane_group_max[nthreads=WARP_SIZE](val)


# ===----------------------------------------------------------------------===#
# Block Sum
# ===----------------------------------------------------------------------===#


@value
struct ReductionMethod:
    var _value: Int

    alias TENSOR_CORE = Self(0)
    alias WARP = Self(1)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other


@always_inline
fn warp_sum[
    intermediate_type: DType,
    *,
    reduction_method: ReductionMethod,
    output_type: DType,
](x: SIMD) -> Scalar[output_type]:
    """Performs a warp level reduction using either a warp shuffle or tensor
    core operation. If the tensor core method is chosen, then the input value
    is cast to the intermediate type to make the value consumable by the
    tensor core op."""

    @parameter
    if reduction_method is ReductionMethod.WARP:
        constrained[
            output_type == x.type,
            (
                "the output type must match the input value for warp-level"
                " reduction"
            ),
        ]()

        return rebind[Scalar[output_type]](warp_sum(x.reduce_add()))
    else:
        return tc_reduce[output_type](x.cast[intermediate_type]())
