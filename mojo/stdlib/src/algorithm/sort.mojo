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

from collections.vector import DynamicVector

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _swap[type: AnyRegType](array: Pointer[type], i0: Int, i1: Int):
    let tmp = array.load(i0)
    array.store(i0, array.load(i1))
    array.store(i1, tmp)


# ===----------------------------------------------------------------------===#
# sort
# ===----------------------------------------------------------------------===#

alias _cmp_fn_type = fn[type: AnyRegType] (type, type) capturing -> Bool


fn _insertion_sort[
    type: AnyRegType, cmp_fn: _cmp_fn_type
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
    type: AnyRegType, cmp_fn: _cmp_fn_type
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
    type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type], size: Int):
    if size == 0:
        return

    var stack = DynamicVector[Int](_estimate_initial_height(size))
    stack.push_back(0)
    stack.push_back(size)
    while len(stack) > 0:
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


# ===----------------------------------------------------------------------===#
# partition
# ===----------------------------------------------------------------------===#
fn partition[
    type: AnyRegType, cmp_fn: _cmp_fn_type
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
    while len(stack) > 0:
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
    fn _less_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    _quicksort[Int, _less_than_equal](buff, len)


fn sort[type: DType](inout buff: Pointer[Scalar[type]], len: Int):
    """Sort the vector inplace.

    The function doesn't return anything, the vector is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        buff: Input buffer.
        len: The length of the buffer.
    """

    @parameter
    fn _less_than_equal[ty: AnyRegType](lhs: ty, rhs: ty) -> Bool:
        return rebind[Scalar[type]](lhs) <= rebind[Scalar[type]](rhs)

    _quicksort[Scalar[type], _less_than_equal](buff, len)


fn sort(inout v: DynamicVector[Int]):
    """Sort the vector inplace.

    The function doesn't return anything, the vector is updated inplace.

    Args:
        v: Input integer vector to sort.
    """
    # Downcast any pointer to register-passable pointer.
    var ptr = rebind[Pointer[Int]](v.data)
    sort(ptr, len(v))


fn sort[type: DType](inout v: DynamicVector[Scalar[type]]):
    """Sort the vector inplace.

    The function doesn't return anything, the vector is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        v: Input vector to sort.
    """

    var ptr = rebind[Pointer[Scalar[type]]](v.data)
    sort[type](ptr, len(v))


# ===----------------------------------------------------------------------===#
# sort networks
# ===----------------------------------------------------------------------===#


@always_inline
fn _sort2[
    type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type], offset0: Int, offset1: Int):
    let a = array.load(offset0)
    let b = array.load(offset1)
    if not cmp_fn[type](a, b):
        array.store(offset0, b)
        array.store(offset1, a)


@always_inline
fn _sort_partial_3[
    type: AnyRegType, cmp_fn: _cmp_fn_type
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
    elif r:
        array.store(offset0, t)


@always_inline
fn _small_sort[
    n: Int, type: AnyRegType, cmp_fn: _cmp_fn_type
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
