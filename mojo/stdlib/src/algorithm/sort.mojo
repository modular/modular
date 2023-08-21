# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements sorting functions."""

from math.bit import ctlz
from sys.info import bitwidthof

from memory.unsafe import DTypePointer, Pointer

from utils.vector import DynamicVector

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _swap[type: AnyType](inout lhs: type, inout rhs: type):
    let tmp = lhs
    lhs = rhs
    rhs = tmp


@always_inline
fn _swap[type: AnyType](array: Pointer[type], i0: Int, i1: Int):
    let tmp = array.load(i0)
    array.store(i0, array.load(i1))
    array.store(i1, tmp)


# ===----------------------------------------------------------------------===#
# sort
# ===----------------------------------------------------------------------===#

alias _cmp_fn_type = fn[type: AnyType] (type, type) capturing -> Bool


fn _insertion_sort[
    type: AnyType, cmp_fn: _cmp_fn_type
](array: Pointer[type], start: Int, end: Int):
    """Sort the array[start:end] slice"""

    for i in range(start + 1, end):
        let value = array.load(i)
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > start and not cmp_fn[type](array.load(j - 1), value):
            array.store(j, array.load(j - 1))
            j -= 1

        array.store(j, value)


fn _partition[
    type: AnyType, cmp_fn: _cmp_fn_type
](array: Pointer[type], start: Int, end: Int) -> Int:
    let pivot = start + (end - start) // 2

    let pivot_value = array.load(pivot)

    var left = start
    var right = end - 2

    _swap(array, pivot, end - 1)

    while left < right:
        if cmp_fn[type](array.load(left), pivot_value):
            left += 1
        elif cmp_fn[type](pivot_value, array.load(right)):
            right -= 1
        else:
            _swap(array, left, right)

    if cmp_fn[type](array.load(right), pivot_value):
        right += 1
    _swap(array, end - 1, right)
    return right


fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    return bitwidthof[DType.index]() - ctlz(size | 1) - 1


fn _quicksort[
    type: AnyType, cmp_fn: _cmp_fn_type
](array: Pointer[type], size: Int):
    if size == 0:
        return

    var stack = DynamicVector[Int](_estimate_initial_height(size))
    stack.push_back(0)
    stack.push_back(size)
    while stack.__len__() > 0:
        let end = stack.pop_back()
        let start = stack.pop_back()

        let len = end - start
        if len < 2:
            continue

        if len == 2:
            _small_sort[2, type, cmp_fn](array + start)
            continue

        if len == 3:
            _small_sort[3, type, cmp_fn](array + start)
            continue

        if len == 4:
            _small_sort[4, type, cmp_fn](array + start)
            continue

        if len == 5:
            _small_sort[5, type, cmp_fn](array + start)
            continue

        if len < 32:
            _insertion_sort[type, cmp_fn](array, start, end)
            continue

        let pivot = _partition[type, cmp_fn](array, start, end)

        stack.push_back(pivot + 1)
        stack.push_back(end)

        stack.push_back(start)
        stack.push_back(pivot)
    stack._del_old()


# ===----------------------------------------------------------------------===#
# sort
# ===----------------------------------------------------------------------===#


fn sort(inout buff: Pointer[Int], len: Int):
    """Sort the vector inplace.

    The function doesn't return anything, the vector is updated inplace.

    Args:
        buff: Input buffer.
        len: The length of the buffer.
    """

    @parameter
    fn _less_than_equal[type: AnyType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    _quicksort[Int, _less_than_equal](buff, len)


fn sort[type: DType](inout buff: Pointer[SIMD[type, 1]], len: Int):
    """Sort the vector inplace.

    The function doesn't return anything, the vector is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        buff: Input buffer.
        len: The length of the buffer.
    """

    @parameter
    fn _less_than_equal[ty: AnyType](lhs: ty, rhs: ty) -> Bool:
        return rebind[SIMD[type, 1]](lhs) <= rebind[SIMD[type, 1]](rhs)

    _quicksort[SIMD[type, 1], _less_than_equal](buff, len)


fn sort(inout v: DynamicVector[Int]):
    """Sort the vector inplace.

    The function doesn't return anything, the vector is updated inplace.

    Args:
        v: Input integer vector to sort.
    """
    sort(v.data, v.__len__())


fn sort[type: DType](inout v: DynamicVector[SIMD[type, 1]]):
    """Sort the vector inplace.

    The function doesn't return anything, the vector is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        v: Input vector to sort.
    """

    sort[type](v.data, v.__len__())


# ===----------------------------------------------------------------------===#
# sort networks
# ===----------------------------------------------------------------------===#


@always_inline
fn _sort2[
    type: AnyType, cmp_fn: _cmp_fn_type
](array: Pointer[type], offset0: Int, offset1: Int):
    let a = array.load(offset0)
    let b = array.load(offset1)
    if not cmp_fn[type](a, b):
        array.store(offset0, b)
        array.store(offset1, a)


@always_inline
fn _sort_partial_3[
    type: AnyType, cmp_fn: _cmp_fn_type
](array: Pointer[type], offset0: Int, offset1: Int, offset2: Int):
    let a = array.load(offset0)
    let b = array.load(offset1)
    let c = array.load(offset2)
    var r = cmp_fn[type](a, c)
    array.store(offset2, c if r else a)
    let tmp = a if r else c
    r = cmp_fn[type](tmp, b)
    if not r:
        array.store(offset0, a if r else b)
        array.store(offset1, b if r else tmp)


@always_inline
fn _small_sort[
    n: Int, type: AnyType, cmp_fn: _cmp_fn_type
](array: Pointer[type]):
    @parameter
    if n == 2:
        _sort2[type, cmp_fn](array, 0, 1)
        return

    @parameter
    if n == 3:
        _sort2[type, cmp_fn](array, 1, 2)
        _sort_partial_3[type, cmp_fn](array, 0, 1, 2)
        return

    @parameter
    if n == 4:
        _sort2[type, cmp_fn](array, 0, 2)
        _sort2[type, cmp_fn](array, 1, 3)
        _sort2[type, cmp_fn](array, 0, 1)
        _sort2[type, cmp_fn](array, 2, 3)
        _sort2[type, cmp_fn](array, 1, 2)
        return

    @parameter
    if n == 5:
        _sort2[type, cmp_fn](array, 0, 1)
        _sort2[type, cmp_fn](array, 3, 4)
        _sort_partial_3[type, cmp_fn](array, 2, 3, 4)
        _sort2[type, cmp_fn](array, 1, 4)
        _sort_partial_3[type, cmp_fn](array, 0, 2, 3)
        _sort_partial_3[type, cmp_fn](array, 1, 2, 3)
        return
