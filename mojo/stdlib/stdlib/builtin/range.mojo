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
"""Implements a 'range' call.

These are Mojo built-ins, so you don't need to import them.
"""


from math import ceildiv

from python import PythonObject

from utils._select import _select_register_value as select

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _sign(x: Int) -> Int:
    var result = 0
    result = select(x > 0, 1, result)
    result = select(x < 0, -1, result)
    return result


# ===----------------------------------------------------------------------=== #
# Range
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _ZeroStartingRange(Iterator, Movable, ReversibleRange, Sized):
    alias Element = Int
    var curr: Int
    var end: Int

    @always_inline
    fn __init__(out self, end: Int):
        self.curr = max(0, end)
        self.end = self.curr

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(mut self) -> Int:
        var curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return self.curr

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> Int:
        var i = Int(index(idx))
        debug_assert(i < self.__len__(), "index out of range")
        return i

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        return range(self.end - 1, -1, -1)


@fieldwise_init
@register_passable("trivial")
struct _SequentialRange(Iterator, ReversibleRange, Sized):
    alias Element = Int
    var start: Int
    var end: Int

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(mut self) -> Int:
        var start = self.start
        self.start += 1
        return start

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        return max(0, self.end - self.start)

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> Int:
        debug_assert(self.__len__() > index(idx), "index out of range")
        return self.start + index(idx)

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        return range(self.end - 1, self.start - 1, -1)


@fieldwise_init
@register_passable("trivial")
struct _StridedRangeIterator(Iterator, Sized):
    alias Element = Int
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    fn __len__(self) -> Int:
        if self.step > 0 and self.start < self.end:
            return self.end - self.start
        elif self.step < 0 and self.start > self.end:
            return self.start - self.end
        else:
            return 0

    @always_inline
    fn __next__(mut self) -> Int:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0


@fieldwise_init
@register_passable("trivial")
struct _StridedRange(Iterator, ReversibleRange, Sized):
    alias Element = Int
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    fn __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end
        self.step = 1

    @always_inline
    fn __iter__(self) -> _StridedRangeIterator:
        return _StridedRangeIterator(self.start, self.end, self.step)

    @always_inline
    fn __next__(mut self) -> Int:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        # If the step is positive we want to check that the start is smaller
        # than the end, if the step is negative we want to check the reverse.
        # We break this into selects to avoid generating branches.
        var c1 = (self.step > 0) & (self.start > self.end)
        var c2 = (self.step < 0) & (self.start < self.end)
        var cnd = c1 | c2

        var numerator = abs(self.start - self.end)
        var denominator = abs(self.step)

        # If the start is after the end and step is positive then we
        # are generating an empty range. In this case divide 0/1 to
        # return 0 without a branch.
        return ceildiv(select(cnd, 0, numerator), select(cnd, 1, denominator))

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> Int:
        debug_assert(self.__len__() > index(idx), "index out of range")
        return self.start + index(idx) * self.step

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        var shifted_end = self.end - _sign(self.step)
        var start = shifted_end - ((shifted_end - self.start) % self.step)
        var end = self.start - self.step
        var step = -self.step
        return range(start, end, step)


@always_inline
fn range[T: Indexer, //](end: T) -> _ZeroStartingRange:
    """Constructs a [0; end) Range.

    Parameters:
        T: The type of the end value.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _ZeroStartingRange(index(end))


@always_inline
fn range[T: IntableRaising, //](end: T) raises -> _ZeroStartingRange:
    """Constructs a [0; end) Range.

    Parameters:
        T: The type of the end value.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.

    Raises:
        An error if the conversion to an `Int` failed.
    """
    return _ZeroStartingRange(Int(end))


@always_inline
fn range(end: PythonObject) raises -> _ZeroStartingRange:
    """Constructs a [0; end) Range from a Python `int`.

    Args:
        end: The end of the range as a Python `int`.

    Returns:
        The constructed range.

    Raises:
        An error if converting `end` to an `Int` failed.
    """
    return range(Int(end))


@always_inline
fn range[T0: Indexer, T1: Indexer, //](start: T0, end: T1) -> _SequentialRange:
    """Constructs a [start; end) Range.

    Parameters:
        T0: The type of the start value.
        T1: The type of the end value.

    Args:
        start: The start of the range.
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _SequentialRange(index(start), index(end))


@always_inline
fn range[
    T0: IntableRaising, T1: IntableRaising
](start: T0, end: T1) raises -> _SequentialRange:
    """Constructs a [start; end) Range.

    Parameters:
        T0: The type of the start value.
        T1: The type of the end value.

    Args:
        start: The start of the range.
        end: The end of the range.

    Returns:
        The constructed range.

    Raises:
        An error if converting `start` or `end` to an `Int` failed.
    """
    return _SequentialRange(Int(start), Int(end))


@always_inline
fn range(start: PythonObject, end: PythonObject) raises -> _SequentialRange:
    """Constructs a [start; end) Range from Python `int` objects.

    Args:
        start: The start of the range as a Python `int`.
        end: The end of the range as a Python `int`.

    Returns:
        The constructed range.

    Raises:
        An error if converting `start` or `end` to an `Int` failed.
    """
    return range(Int(start), Int(end))


@always_inline
fn range[
    T0: Indexer, T1: Indexer, T2: Indexer, //
](start: T0, end: T1, step: T2) -> _StridedRange:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        T0: The type of the start value.
        T1: The type of the end value.
        T2: The type of the step value.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.

    Returns:
        The constructed range.
    """
    return _StridedRange(index(start), index(end), index(step))


@always_inline
fn range[
    T0: IntableRaising, T1: IntableRaising, T2: IntableRaising, //
](start: T0, end: T1, step: T2) raises -> _StridedRange:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        T0: The type of the start value.
        T1: The type of the end value.
        T2: The type of the step value.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.

    Returns:
        The constructed range.

    Raises:
        An error if converting `start`, `end`, or `step` to an `Int` failed.
    """
    return _StridedRange(Int(start), Int(end), Int(step))


@always_inline
fn range(
    start: PythonObject, end: PythonObject, step: PythonObject
) raises -> _StridedRange:
    """Constructs a [start; end) Range from Python `int` objects with a given
    step.

    Args:
        start: The start of the range as a Python `int`.
        end: The end of the range as a Python `int`.
        step: The step for the range as a Python `int`.

    Returns:
        The constructed range.

    Raises:
        An error if converting `start`, `end`, or `step` to an `Int` failed.
    """
    return range(Int(start), Int(end), Int(step))


# ===----------------------------------------------------------------------=== #
# Range UInt
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _UIntZeroStartingRange(Iterator, UIntSized):
    alias Element = UInt
    var curr: UInt
    var end: UInt

    @always_inline
    fn __init__(out self, end: UInt):
        self.curr = end
        self.end = self.curr

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(mut self) -> UInt:
        var curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> UInt:
        return self.curr

    @always_inline
    fn __getitem__(self, idx: UInt) -> UInt:
        debug_assert(idx < self.__len__(), "index out of range")
        return idx


@fieldwise_init
@register_passable("trivial")
struct _UIntStridedRangeIterator(Iterator, UIntSized):
    alias Element = UInt
    var start: UInt
    var end: UInt
    var step: UInt

    @always_inline
    fn __len__(self) -> UInt:
        return select(self.start < self.end, self.end - self.start, 0)

    @always_inline
    fn __next__(mut self) -> UInt:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0


@register_passable("trivial")
struct _UIntStridedRange(Iterator, UIntSized):
    alias Element = UInt
    var start: UInt
    var end: UInt
    var step: UInt

    @always_inline
    fn __init__(out self, start: UInt, end: UInt, step: UInt):
        self.start = start
        self.end = end
        debug_assert(
            step != 0, "range() arg 3 (the step size) must not be zero"
        )
        debug_assert(
            step != UInt(Int(-1)),
            (
                "range() arg 3 (the step size) cannot be -1.  Reverse range is"
                " not supported yet for UInt ranges."
            ),
        )
        self.step = step

    @always_inline
    fn __iter__(self) -> _UIntStridedRangeIterator:
        return _UIntStridedRangeIterator(self.start, self.end, self.step)

    @always_inline
    fn __next__(mut self) -> UInt:
        if self.start >= self.end:
            return self.end
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> UInt:
        if self.start >= self.end:
            return 0
        return ceildiv(self.end - self.start, self.step)

    @always_inline
    fn __getitem__(self, idx: UInt) -> UInt:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + idx * self.step


@always_inline
fn range(end: UInt) -> _UIntZeroStartingRange:
    """Constructs a [0; end) Range.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _UIntZeroStartingRange(end)


@always_inline
fn range(start: UInt, end: UInt, step: UInt = 1) -> _UIntStridedRange:
    """Constructs a [start; end) Range with a given step.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.  Defaults to 1.

    Returns:
        The constructed range.
    """
    return _UIntStridedRange(start, end, step)


# ===----------------------------------------------------------------------=== #
# Range Scalar
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _ZeroStartingScalarRange[dtype: DType](Iterator & Copyable):
    alias Element = Scalar[dtype]
    var curr: Scalar[dtype]
    var end: Scalar[dtype]

    @always_inline
    fn __init__(out self, end: Scalar[dtype]):
        self.curr = max(0, end)
        self.end = self.curr

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(mut self) -> Scalar[dtype]:
        var curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Scalar[dtype]:
        return self.curr

    @always_inline
    fn __getitem__(self, idx: Scalar[dtype]) -> Scalar[dtype]:
        debug_assert(idx < self.__len__(), "index out of range")
        return idx

    @always_inline
    fn __reversed__(self) -> _StridedScalarRange[dtype]:
        constrained[
            not dtype.is_unsigned(), "cannot reverse an unsigned range"
        ]()
        return range(self.end - 1, Scalar[dtype](-1), Scalar[dtype](-1))


@fieldwise_init
@register_passable("trivial")
struct _SequentialScalarRange[dtype: DType](Iterator & Copyable):
    alias Element = Scalar[dtype]
    var start: Scalar[dtype]
    var end: Scalar[dtype]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(mut self) -> Scalar[dtype]:
        var start = self.start
        self.start += 1
        return start

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Scalar[dtype]:
        return max(0, self.end - self.start)

    @always_inline
    fn __getitem__(self, idx: Scalar[dtype]) -> Scalar[dtype]:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + idx

    @always_inline
    fn __reversed__(self) -> _StridedScalarRange[dtype]:
        constrained[
            not dtype.is_unsigned(), "cannot reverse an unsigned range"
        ]()
        return range(self.end - 1, self.start - 1, Scalar[dtype](-1))


@fieldwise_init
@register_passable("trivial")
struct _StridedScalarRangeIterator[dtype: DType](Iterator & Copyable):
    alias Element = Scalar[dtype]
    var start: Scalar[dtype]
    var end: Scalar[dtype]
    var step: Scalar[dtype]

    @always_inline
    fn __has_next__(self) -> Bool:
        # If the type is unsigned, then 'step' cannot be negative.
        @parameter
        if dtype.is_unsigned():
            return self.start < self.end
        else:
            if self.step > 0:
                return self.start < self.end
            return self.end < self.start

    @always_inline
    fn __next__(mut self) -> Scalar[dtype]:
        var result = self.start
        self.start += self.step
        return result


@fieldwise_init
@register_passable("trivial")
struct _StridedScalarRange[dtype: DType]:
    alias Element = Scalar[dtype]
    var start: Scalar[dtype]
    var end: Scalar[dtype]
    var step: Scalar[dtype]

    @always_inline
    fn __iter__(self) -> _StridedScalarRangeIterator[dtype]:
        return _StridedScalarRangeIterator(self.start, self.end, self.step)


@always_inline
fn range[
    dtype: DType, //
](end: Scalar[dtype]) -> _ZeroStartingScalarRange[dtype]:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        dtype: The range dtype.

    Args:
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _ZeroStartingScalarRange(end)


@always_inline
fn range[
    dtype: DType, //
](start: Scalar[dtype], end: Scalar[dtype]) -> _SequentialScalarRange[dtype]:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        dtype: The range dtype.

    Args:
        start: The start of the range.
        end: The end of the range.

    Returns:
        The constructed range.
    """
    return _SequentialScalarRange(start, end)


@always_inline
fn range[
    dtype: DType, //
](
    start: Scalar[dtype], end: Scalar[dtype], step: Scalar[dtype]
) -> _StridedScalarRange[dtype]:
    """Constructs a [start; end) Range with a given step.

    Parameters:
        dtype: The range dtype.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step for the range.  Defaults to 1.

    Returns:
        The constructed range.
    """
    return _StridedScalarRange(start, end, step)
