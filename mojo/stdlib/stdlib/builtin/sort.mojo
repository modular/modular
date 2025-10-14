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
"""Implements the built-in `sort` function.

These are Mojo built-ins, so you don't need to import them.
"""

from math import ceil

from bit import count_leading_zeros
from memory import Span

# ===-----------------------------------------------------------------------===#
# sort
# ===-----------------------------------------------------------------------===#

alias insertion_sort_threshold = 32


@always_inline
fn _insertion_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]):
    """Sort the array[start:end] slice"""
    var array = span.unsafe_ptr().as_any_origin()
    var size = len(span)

    for i in range(1, size):
        var value = (array + i).take_pointee()
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > 0 and cmp_fn(value, array[j - 1]):
            (array + j).init_pointee_move_from(array + j - 1)
            j -= 1

        (array + j).init_pointee_move(value^)


# put everything thats "<" to the left of pivot
@always_inline
fn _quicksort_partition_right[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]) -> Int:
    var size = len(span)

    var left = 1
    var right = size - 1
    var ref pivot_value = span.unsafe_get(0)

    while True:
        # no need for left < right since quick sort pick median of 3 as pivot
        while cmp_fn(span.unsafe_get(left), pivot_value):
            left += 1
        while left < right and not cmp_fn(span.unsafe_get(right), pivot_value):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            span.unsafe_swap_elements(pivot_pos, 0)
            return pivot_pos
        span.unsafe_swap_elements(left, right)
        left += 1
        right -= 1


# put everything thats "<=" to the left of pivot
@always_inline
fn _quicksort_partition_left[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]) -> Int:
    var size = len(span)

    var left = 1
    var right = size - 1
    var ref pivot_value = span.unsafe_get(0)

    while True:
        while left < right and not cmp_fn(pivot_value, span.unsafe_get(left)):
            left += 1
        while cmp_fn(pivot_value, span.unsafe_get(right)):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            span.unsafe_swap_elements(pivot_pos, 0)
            return pivot_pos
        span.unsafe_swap_elements(left, right)
        left += 1
        right -= 1


fn _heap_sort_fix_down[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin], idx: Int):
    var size = len(span)
    var i = idx
    var j = i * 2 + 1
    while j < size:  # has left child
        # if right child exist and has higher value, swap with right
        if i * 2 + 2 < size and cmp_fn(
            span.unsafe_get(j), span.unsafe_get(i * 2 + 2)
        ):
            j = i * 2 + 2
        if not cmp_fn(span.unsafe_get(i), span.unsafe_get(j)):
            return
        span.unsafe_swap_elements(j, i)
        i = j
        j = i * 2 + 1


@always_inline
fn _heap_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]):
    var size = len(span)
    # heapify
    for i in range(size // 2 - 1, -1, -1):
        _heap_sort_fix_down[cmp_fn](span, i)
    # sort
    while size > 1:
        size -= 1
        span.unsafe_swap_elements(0, size)
        _heap_sort_fix_down[cmp_fn](span, 0)


@always_inline
fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    var log2 = Int((DType.int.bit_width() - 1) ^ count_leading_zeros(size | 1))
    # The number 1.3 was chosen by experimenting the max stack size for random
    # input. This also depends on insertion_sort_threshold
    return max(2, Int(ceil(1.3 * log2)))


@always_inline
fn _delegate_small_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]):
    var size = len(span)
    if size == 2:
        _small_sort[2, T, cmp_fn](span)

        return
    if size == 3:
        _small_sort[3, T, cmp_fn](span)
        return

    if size == 4:
        _small_sort[4, T, cmp_fn](span)
        return

    if size == 5:
        _small_sort[5, T, cmp_fn](span)
        return


# FIXME (MSTDL-808): Using _Pair over Span results in 1-3% improvement
# struct _Pair[T: AnyType]:
#     var ptr: UnsafePointer[T]
#     var len: Int


@always_inline
fn _quicksort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
    *,
    do_smallsort: Bool = False,
](span: Span[T, origin]):
    var size = len(span)
    if size == 0:
        return

    var stack = List[Span[T, origin]](capacity=_estimate_initial_height(size))
    stack.append(span)
    while len(stack) > 0:
        var interval = stack.pop()
        var len = len(interval)

        @parameter
        if do_smallsort:
            if len <= 5:
                _delegate_small_sort[cmp_fn](interval)
                continue

        if len < insertion_sort_threshold:
            _insertion_sort[cmp_fn](interval)
            continue

        # pick median of 3 as pivot
        _sort3[T, cmp_fn](interval, len >> 1, 0, len - 1)

        # if ptr[-1] == pivot_value, then everything in between will
        # be the same, so no need to recurse that interval
        # already have array[-1] <= array[0]
        var interval_ptr = interval.unsafe_ptr()
        if interval_ptr > span.unsafe_ptr() and not cmp_fn(
            interval_ptr[-1], interval_ptr[0]
        ):
            var pivot = _quicksort_partition_left[cmp_fn](interval)
            if len > pivot + 2:
                stack.append(
                    interval.unsafe_subspan(
                        offset=pivot + 1, length=len - pivot - 1
                    )
                )
            continue

        var pivot = _quicksort_partition_right[cmp_fn](interval)

        if len > pivot + 2:
            stack.append(
                interval.unsafe_subspan(
                    offset=pivot + 1, length=len - pivot - 1
                )
            )

        if pivot > 1:
            stack.append(interval.unsafe_subspan(offset=0, length=pivot))


# ===-----------------------------------------------------------------------===#
# stable sort
# ===-----------------------------------------------------------------------===#


fn _merge[
    T: Copyable & Movable,
    span_origin: MutableOrigin,
    result_origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](
    span1: Span[T, span_origin],
    span2: Span[T, span_origin],
    result: Span[T, result_origin],
):
    """Merge span1 and span2 into result using the given cmp_fn. The function
    will crash if result is not large enough to hold both span1 and span2.

    Note that if result contains data previously, its destructor will not be called.

    Parameters:
        T: Type of the spans.
        span_origin: Origin of the input spans.
        result_origin: Origin of the result Span.
        cmp_fn: Comparison functor of (type, type) capturing [_] -> Bool type.

    Args:
        span1: The first span to be merged.
        span2: The second span to be merged.
        result: The output span.
    """
    var span1_size = len(span1)
    var span2_size = len(span2)
    var span1_ptr = span1.unsafe_ptr()
    var span2_ptr = span2.unsafe_ptr()
    var res_ptr = result.unsafe_ptr()

    debug_assert(
        span1_size + span2_size <= len(result),
        "The merge result does not fit in the span provided",
    )
    var i = 0
    var j = 0
    var k = 0
    while i < span1_size:
        if j == span2_size:
            while i < span1_size:
                (res_ptr + k).init_pointee_move_from(span1_ptr + i)
                k += 1
                i += 1
            return
        if cmp_fn(span2.unsafe_get(j), span1.unsafe_get(i)):
            (res_ptr + k).init_pointee_move_from(span2_ptr + j)
            j += 1
        else:
            (res_ptr + k).init_pointee_move_from(span1_ptr + i)
            i += 1
        k += 1

    while j < span2_size:
        (res_ptr + k).init_pointee_move_from(span2_ptr + j)
        k += 1
        j += 1


fn _stable_sort_impl[
    T: Copyable & Movable,
    span_life: MutableOrigin,
    tmp_life: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, span_life], temp_buff: Span[T, tmp_life]):
    var size = len(span)
    if size <= 1:
        return
    var i = 0
    while i < size:
        _insertion_sort[cmp_fn](
            span.unsafe_subspan(
                offset=i, length=min(insertion_sort_threshold, size - i)
            )
        )
        i += insertion_sort_threshold
    var merge_size = insertion_sort_threshold
    while merge_size < size:
        var j = 0
        while j + merge_size < size:
            var span1 = span.unsafe_subspan(offset=j, length=merge_size)
            var span2 = span.unsafe_subspan(
                offset=j + merge_size,
                length=min(merge_size, max(0, size - (j + merge_size))),
            )
            _merge[cmp_fn](span1, span2, temp_buff)
            for i in range(merge_size + len(span2)):
                UnsafePointer(to=span.unsafe_get(j + i)).init_pointee_move_from(
                    UnsafePointer(to=temp_buff.unsafe_get(i))
                )
            j += 2 * merge_size
        merge_size *= 2


fn _stable_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]):
    var temp_buff = UnsafePointer[T].alloc(len(span))
    var temp_buff_span = Span(ptr=temp_buff, length=UInt(len(span)))
    _stable_sort_impl[cmp_fn](span, temp_buff_span)
    temp_buff.free()


# ===-----------------------------------------------------------------------===#
# partition
# ===-----------------------------------------------------------------------===#


@always_inline
fn _partition[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]) -> Int:
    var size = len(span)
    if size <= 1:
        return 0

    var pivot = size // 2

    var left = 0
    var right = size - 2

    var pivot_index = size - 1
    span.unsafe_swap_elements(pivot, pivot_index)

    while left < right:
        if cmp_fn(span.unsafe_get(left), span.unsafe_get(pivot_index)):
            left += 1
        elif not cmp_fn(span.unsafe_get(right), span.unsafe_get(pivot_index)):
            right -= 1
        else:
            span.unsafe_swap_elements(left, right)

    if cmp_fn(span.unsafe_get(right), span.unsafe_get(pivot_index)):
        right += 1
    span.unsafe_swap_elements(pivot_index, right)
    return right


fn _partition[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](var span: Span[T, origin], var k: Int):
    while True:
        var pivot = _partition[cmp_fn](span)
        if pivot == k:
            return
        elif k < pivot:
            span = span.unsafe_subspan(offset=0, length=pivot)
        else:
            span._data += pivot + 1
            span._len -= pivot + 1
            k -= pivot + 1


fn partition[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin], k: Int):
    """Partition the input buffer inplace such that first k elements are the
    largest (or smallest if cmp_fn is < operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        T: Type of the underlying data.
        origin: Origin of span.
        cmp_fn: Comparison functor of (T, T) capturing [_] -> Bool type.

    Args:
        span: Input buffer.
        k: Index of the partition element.
    """

    _partition[cmp_fn](span, k)


# ===-----------------------------------------------------------------------===#
# sort
# ===-----------------------------------------------------------------------===#


# Junction from public to private API
fn _sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
    *,
    stable: Bool = False,
    do_smallsort: Bool = False,
](span: Span[T, origin]):
    @parameter
    if do_smallsort:
        if len(span) <= 5:
            _delegate_small_sort[cmp_fn](span)
            return

    if len(span) < insertion_sort_threshold:
        _insertion_sort[cmp_fn](span)
        return

    @parameter
    if stable:
        _stable_sort[cmp_fn](span)
    else:
        _quicksort[cmp_fn, do_smallsort=do_smallsort](span)


# TODO (MSTDL-766): The Int and Scalar[T] overload should be remove
# (same for partition)
# Eventually we want a sort that takes a Span and one that takes a List with
# optional cmp_fn.
fn sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
    *,
    stable: Bool = False,
    __disambiguate: NoneType = None,
](span: Span[T, origin]):
    """Sort a span in-place.
    The function doesn't return anything, the span is updated in-place.

    Parameters:
        T: Copyable & Movable type of the underlying data.
        origin: Origin of span.
        cmp_fn: The comparison function.
        stable: Whether the sort should be stable.
        __disambiguate: Give the Scalar overload higher priority. Do not pass explicitly.


    Args:
        span: The span to be sorted.
    """

    _sort[cmp_fn, stable=stable](span)


fn sort[
    dtype: DType,
    origin: MutableOrigin, //,
    cmp_fn: fn (Scalar[dtype], Scalar[dtype]) capturing [_] -> Bool,
    *,
    stable: Bool = False,
](span: Span[Scalar[dtype], origin]):
    """Sort a span of Scalar elements in-place.
    The function doesn't return anything, the list is updated in-place.

    Parameters:
        dtype: Type of elements.
        origin: Origin of span.
        cmp_fn: The comparison function.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """
    _sort[cmp_fn, stable=stable, do_smallsort=True](span)


fn sort[
    origin: MutableOrigin, //,
    cmp_fn: fn (Int, Int) capturing [_] -> Bool,
    *,
    stable: Bool = False,
](span: Span[Int, origin]):
    """Sort a span in-place.
    The function doesn't return anything, the span is updated in-place.

    Parameters:
        origin: Origin of span.
        cmp_fn: The comparison function.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    _sort[cmp_fn, stable=stable, do_smallsort=True](span)


fn sort[
    origin: MutableOrigin, //,
    *,
    stable: Bool = False,
](span: Span[Int, origin]):
    """Sort a span inplace.
    The function doesn't return anything, the span is updated in-place.

    Parameters:
        origin: Origin of span.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    sort[_cmp_fn, stable=stable](span)


fn sort[
    T: Copyable & Movable & Comparable,
    origin: MutableOrigin, //,
    *,
    stable: Bool = False,
](span: Span[T, origin]):
    """Sort a span of comparable elements in-place.

    Parameters:
        T: The order comparable collection element type.
        origin: Origin of span.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    @parameter
    fn _cmp_fn(a: T, b: T) -> Bool:
        return a < b

    sort[_cmp_fn, stable=stable](span)


# ===-----------------------------------------------------------------------===#
# sort networks
# ===-----------------------------------------------------------------------===#


@always_inline
fn _sort2[
    origin: MutableOrigin, //,
    T: Copyable & Movable,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin], offset0: Int, offset1: Int,):
    if not cmp_fn(span.unsafe_get(offset0), span.unsafe_get(offset1)):
        span.unsafe_swap_elements(offset0, offset1)


@always_inline
fn _sort3[
    origin: MutableOrigin, //,
    T: Copyable & Movable,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin], offset0: Int, offset1: Int, offset2: Int,):
    _sort2[T, cmp_fn](span, offset0, offset1)
    _sort2[T, cmp_fn](span, offset1, offset2)
    _sort2[T, cmp_fn](span, offset0, offset1)


@always_inline
fn _sort_partial_3[
    origin: MutableOrigin, //,
    T: Copyable & Movable,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin], offset0: Int, offset1: Int, offset2: Int):
    """Sorts [a, b, c] assuming [b, c] is already sorted."""
    if cmp_fn(span.unsafe_get(offset0), span.unsafe_get(offset1)):
        return

    span.unsafe_swap_elements(offset0, offset1)
    if not cmp_fn(span.unsafe_get(offset1), span.unsafe_get(offset2)):
        span.unsafe_swap_elements(offset1, offset2)


@always_inline
fn _small_sort[
    origin: MutableOrigin, //,
    n: Int,
    T: Copyable & Movable,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin]):
    @parameter
    if n == 2:
        _sort2[T, cmp_fn](span, 0, 1)
        return

    @parameter
    if n == 3:
        _sort2[T, cmp_fn](span, 1, 2)
        _sort_partial_3[T, cmp_fn](span, 0, 1, 2)
        return

    @parameter
    if n == 4:
        _sort2[T, cmp_fn](span, 0, 2)
        _sort2[T, cmp_fn](span, 1, 3)
        _sort2[T, cmp_fn](span, 0, 1)
        _sort2[T, cmp_fn](span, 2, 3)
        _sort2[T, cmp_fn](span, 1, 2)
        return

    @parameter
    if n == 5:
        _sort2[T, cmp_fn](span, 0, 1)
        _sort2[T, cmp_fn](span, 3, 4)
        _sort_partial_3[T, cmp_fn](span, 2, 3, 4)
        _sort2[T, cmp_fn](span, 1, 4)
        _sort_partial_3[T, cmp_fn](span, 0, 2, 3)
        _sort_partial_3[T, cmp_fn](span, 1, 2, 3)
        return
