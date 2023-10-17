# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements sorting functions.

You can import these APIs from the `algorithm` package. For example:

```mojo
from algorithm.sort import sort
```
"""

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
    if start == end:
        return end

    let pivot = start + (end - start) // 2

    let pivot_value = array.load(pivot)

    var left = start
    var right = end - 2

    _swap(array, pivot, end - 1)

    while left < right:
        if cmp_fn[type](array.load(left), pivot_value):
            left += 1
        elif not cmp_fn[type](array.load(right), pivot_value):
            right -= 1
        else:
            _swap(array, left, right)

    if cmp_fn[type](array.load(right), pivot_value):
        right += 1
    _swap(array, end - 1, right)
    return right


fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    return max(2, bitwidthof[DType.index]() - ctlz(size | 1) - 1)


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
# partition
# ===----------------------------------------------------------------------===#
fn partition[
    type: AnyType, cmp_fn: _cmp_fn_type
](buff: Pointer[type], k: Int, size: Int):
    """Partition the input vector inplace such that first k elements are the
    largest (or smallest if cmp_fn is <= operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        type: DType of the underlying data.
        cmp_fn: Comparison functor of type, type) capturing -> Bool type.

    Args:
        buff: Input buffer.
        k: Index of the partition element.
        size: The length of the buffer.
    """
    var stack = DynamicVector[Int](_estimate_initial_height(size))
    stack.push_back(0)
    stack.push_back(size)
    while stack.__len__() > 0:
        let end = stack.pop_back()
        let start = stack.pop_back()
        let pivot = _partition[type, cmp_fn](buff, start, end)
        if pivot == k:
            break
        elif k < pivot:
            stack.push_back(start)
            stack.push_back(pivot)
        else:
            stack.push_back(pivot + 1)
            stack.push_back(end)
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
    let r = cmp_fn[type](c, a)
    let t = c if r else a
    if r:
        array.store(offset2, a)
    if cmp_fn[type](b, t):
        array.store(offset0, b)
        array.store(offset1, t)


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
