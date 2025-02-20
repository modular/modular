# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs shuffle instructions."""

from sys import is_nvidia_gpu, llvm_intrinsic

from bit import log2_floor
from gpu import lane_id
from memory import bitcast
from builtin.math import max as _max, min as _min

from .globals import WARP_SIZE
from .tensor_ops import tc_reduce

# TODO (#24457): support shuffles with width != 32
alias _WIDTH_MASK = WARP_SIZE - 1
alias FULL_MASK = 2**WARP_SIZE - 1

# shfl.sync.up.b32 prepares this mask differently from other shuffle intrinsics
alias _WIDTH_MASK_SHUFFLE_UP = 0


# ===-----------------------------------------------------------------------===#
# utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _shuffle[
    mnemonic: StringLiteral,
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


# ===-----------------------------------------------------------------------===#
# shuffle_idx
# ===-----------------------------------------------------------------------===#


@always_inline
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
    return shuffle_idx(FULL_MASK, val, offset)


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
    var v = t0 | offset.cast[DType.int32]()
    v <<= 2
    var result_packed = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](
        v, bitcast[DType.int32, 1](val)
    )
    return bitcast[type, simd_width](result_packed)


@always_inline
fn shuffle_idx[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
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

    @parameter
    if is_nvidia_gpu():
        return _shuffle[
            "idx",
            WIDTH_MASK=_WIDTH_MASK,
        ](mask, val, offset)
    else:

        @parameter
        if bitwidthof[type]() == 16 and simd_width == 1:
            var val_splatted = SIMD[type, 2](rebind[Scalar[type]](val))
            return _shuffle_idx_amd(mask, val_splatted, offset)[0]
        else:
            return _shuffle_idx_amd(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_up
# ===-----------------------------------------------------------------------===#


@always_inline
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
    return shuffle_up(FULL_MASK, val, offset)


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
    var v = (t0 < t1).select(lane, t0)
    # The address needs to be in bytes
    v <<= 2
    var result_packed = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](
        v, bitcast[DType.int32, 1](val)
    )
    return bitcast[type, simd_width](result_packed)


@always_inline
fn shuffle_up[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
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

    @parameter
    if is_nvidia_gpu():
        return _shuffle["up", WIDTH_MASK=_WIDTH_MASK_SHUFFLE_UP](
            mask, val, offset
        )
    else:

        @parameter
        if bitwidthof[type]() == 16 and simd_width == 1:
            var val_splatted = SIMD[type, 2](rebind[Scalar[type]](val))
            return _shuffle_up_amd(mask, val_splatted, offset)[0]
        else:
            return _shuffle_up_amd(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_down
# ===-----------------------------------------------------------------------===#


@always_inline
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
    return shuffle_down(FULL_MASK, val, offset)


@always_inline
fn _shuffle_down_amd[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
    type, simd_width
]:
    # FIXME: Set the EXECute mask register to the mask
    var lane = lane_id()
    # set the offset to 0 if lane + offset >= WARP_SIZE
    var v = (lane + offset > _WIDTH_MASK).select(0, offset)
    v += lane
    # The address needs to be in bytes
    v <<= 2
    var result_packed = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](
        v, bitcast[DType.int32, 1](val)
    )
    return bitcast[type, simd_width](result_packed)


@always_inline
fn shuffle_down[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
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

    @parameter
    if is_nvidia_gpu():
        return _shuffle["down", WIDTH_MASK=_WIDTH_MASK](mask, val, offset)
    else:

        @parameter
        if bitwidthof[type]() == 16 and simd_width == 1:
            var val_splatted = SIMD[type, 2](rebind[Scalar[type]](val))
            return _shuffle_down_amd(mask, val_splatted, offset)[0]
        elif type is DType.bool and simd_width == 1:
            return _shuffle_down_amd(
                mask, val.cast[DType.int32](), offset
            ).cast[type]()
        else:
            return _shuffle_down_amd(mask, val, offset)


# ===-----------------------------------------------------------------------===#
# shuffle_xor
# ===-----------------------------------------------------------------------===#


@always_inline
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
    return shuffle_xor(FULL_MASK, val, offset)


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
    var v = (t0 < t2).select(t0, lane)
    # The address needs to be in bytes
    v <<= 2
    var result_packed = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](
        v, bitcast[DType.int32, 1](val)
    )
    return bitcast[type, simd_width](result_packed)


@always_inline
fn shuffle_xor[
    type: DType, simd_width: Int, //
](mask: UInt, val: SIMD[type, simd_width], offset: UInt32) -> SIMD[
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

    @parameter
    if is_nvidia_gpu():
        return _shuffle["bfly", WIDTH_MASK=_WIDTH_MASK](mask, val, offset)
    else:

        @parameter
        if bitwidthof[type]() == 16 and simd_width == 1:
            var val_splatted = SIMD[type, 2](rebind[Scalar[type]](val))
            return _shuffle_xor_amd(mask, val_splatted, offset)[0]
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
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Takes in an input function to computes warp shuffle based reduction
    operation.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.
        shuffle: The shuffle function to use.
        func: The function to apply to the SIMD value.
        nthreads: The number of threads in the warp.

    Args:
        val: The SIMD value to reduce.

    Returns:
        A SIMD value with the reduction result.
    """
    var res = val

    alias limit = log2_floor(nthreads)

    @parameter
    for i in reversed(range(limit)):
        alias offset = 1 << i
        res = func(res, shuffle(res, offset))

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
    """Takes in an input function to computes warp shuffle based reduction
    operation.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.
        shuffle: The shuffle function to use.
        func: The function to apply to the SIMD value.

    Args:
        val: The SIMD value to reduce.

    Returns:
        A SIMD value with the reduction result.
    """
    return lane_group_reduce[shuffle, func, nthreads=WARP_SIZE](val)


# ===-----------------------------------------------------------------------===#
# Warp Sum
# ===-----------------------------------------------------------------------===#


@always_inline
fn lane_group_sum[
    val_type: DType,
    simd_width: Int, //,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the sum within a lane group.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.
        nthreads: The number of threads in the warp.

    Args:
        val: The SIMD value to sum.

    Returns:
        A SIMD value with the sum of all lanes.
    """

    @parameter
    fn _reduce_add(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return x + y

    return lane_group_reduce[shuffle_down, _reduce_add, nthreads=nthreads](val)


@always_inline
fn lane_group_sum_and_broadcast[
    val_type: DType,
    simd_width: Int, //,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes sum and broadcasts within a lane group.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.
        nthreads: The number of threads in the warp.

    Args:
        val: The SIMD value to sum.

    Returns:
        A SIMD value with the sum of all lanes.
    """

    @parameter
    fn _reduce_add(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return x + y

    return lane_group_reduce[shuffle_xor, _reduce_add, nthreads=nthreads](val)


@always_inline
fn sum[
    val_type: DType, simd_width: Int, //
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the sum across the warp.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.

    Args:
        val: The SIMD value to sum.

    Returns:
        A SIMD value with the sum of all lanes.
    """
    return lane_group_sum[nthreads=WARP_SIZE](val)


@value
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
    """Performs a warp level reduction using either a warp shuffle or tensor
    core operation. If the tensor core method is chosen, then the input value
    is cast to the intermediate type to make the value consumable by the
    tensor core op.

    Parameters:
        intermediate_type: The type of the intermediate value.
        reduction_method: The method to use for the reduction.
        output_type: The type of the output value.

    Args:
        x: The value to reduce.

    Returns:
        The reduced value.
    """

    @parameter
    if reduction_method is ReductionMethod.WARP:
        constrained[
            output_type == x.type,
            (
                "the output type must match the input value for warp-level"
                " reduction"
            ),
        ]()

        return rebind[Scalar[output_type]](sum(x.reduce_add()))
    else:
        return tc_reduce[output_type](x.cast[intermediate_type]())


# ===-----------------------------------------------------------------------===#
# Warp Max
# ===-----------------------------------------------------------------------===#


fn lane_group_max[
    val_type: DType,
    simd_width: Int, //,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its maximum within a lane group.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.
        nthreads: The number of threads in the warp.

    Args:
        val: The SIMD value to reduce.

    Returns:
        A SIMD value with the maximum value in all lanes.
    """

    @parameter
    fn _reduce_max(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return _max(x, y)

    return lane_group_reduce[shuffle_down, _reduce_max, nthreads=nthreads](val)


fn lane_group_max_and_broadcast[
    val_type: DType,
    simd_width: Int, //,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces and broadcasts max within a lane group.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.
        nthreads: The number of threads in the warp.

    Args:
        val: The SIMD value to reduce.

    Returns:
        A SIMD value with the maximum value in all lanes.
    """

    @parameter
    fn _reduce_max(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return _max(x, y)

    return lane_group_reduce[shuffle_xor, _reduce_max, nthreads=nthreads](val)


@always_inline
fn max[
    val_type: DType,
    simd_width: Int, //,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the maximum value across the warp.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.

    Args:
        val: The SIMD value to reduce.

    Returns:
        A SIMD value with the maximum value in all lanes.
    """
    return lane_group_max[nthreads=WARP_SIZE](val)


# ===-----------------------------------------------------------------------===#
# Warp Min
# ===-----------------------------------------------------------------------===#


@always_inline
fn lane_group_min[
    val_type: DType,
    simd_width: Int, //,
    nthreads: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its minimum within a lane group.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.
        nthreads: The number of threads in the warp.

    Args:
        val: The SIMD value to reduce.

    Returns:
        A SIMD value with the minimum value in all lanes.
    """

    @parameter
    fn _reduce_min(x: SIMD, y: __type_of(x)) -> __type_of(x):
        return _min(x, y)

    return lane_group_reduce[shuffle_down, _reduce_min, nthreads=nthreads](val)


@always_inline
fn min[
    val_type: DType, simd_width: Int, //
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the minimum value across the warp.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.

    Args:
        val: The SIMD value to reduce.

    Returns:
        A SIMD value with the minimum value in all lanes.
    """
    return lane_group_min[nthreads=WARP_SIZE](val)


# ===-----------------------------------------------------------------------===#
# Warp Broadcast
# ===-----------------------------------------------------------------------===#


@always_inline
fn broadcast[
    val_type: DType, simd_width: Int, //
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Broadcasts a SIMD value across the warp.

    Parameters:
        val_type: The type of the SIMD value.
        simd_width: The width of the SIMD value.

    Args:
        val: The SIMD value to broadcast.

    Returns:
        A SIMD value with all lanes equal to the input value.
    """
    return shuffle_idx(val, 0)


fn broadcast(val: Int) -> Int:
    """Broadcasts an Int value across the warp.

    Args:
        val: The Int value to broadcast.

    Returns:
        An Int value with all lanes equal to the input value.
    """
    return Int(shuffle_idx(Int32(val), 0))


fn broadcast(val: UInt) -> UInt:
    """Broadcasts a UInt value across the warp.

    Args:
        val: The UInt value to broadcast.

    Returns:
        A UInt value with all lanes equal to the input value.
    """
    return Int(shuffle_idx(Scalar[DType.int32](val), 0))
