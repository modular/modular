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
"""Implements a 'range' call.

These are Mojo built-ins, so you don't need to import them.
"""


from math import ceildiv
from sys.info import size_of
from sys.intrinsics import unlikely

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


struct _ZeroStartingRange(
    Iterable, Iterator, ReversibleRange, Sized, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Int
    var curr: Int
    var end: Int

    @always_inline
    fn __init__(out self, end: Int):
        self.curr = max(end, 0)
        self.end = self.curr

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __next__(mut self) raises StopIteration -> Int:
        var curr = self.curr
        if curr == 0:
            raise StopIteration()
        self.curr = curr - 1
        return self.end - curr

    @always_inline
    fn __len__(self) -> Int:
        return self.curr

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> Int:
        var i = index(idx)
        debug_assert(i < self.__len__(), "index out of range")
        return i

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        return range(self.end - 1, -1, -1)

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self)
        return (len, {len})


struct _SequentialRange(
    Iterable, Iterator, ReversibleRange, Sized, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Int
    var start: Int
    var end: Int

    @always_inline
    fn __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = max(start, end)

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __next__(mut self) raises StopIteration -> Int:
        var start = self.start
        if start == self.end:
            raise StopIteration()
        self.start = start + 1
        return start

    @always_inline
    fn __len__(self) -> Int:
        return self.end - self.start

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> Int:
        debug_assert(self.__len__() > index(idx), "index out of range")
        return self.start + index(idx)

    @always_inline
    fn __reversed__(self) -> _StridedRange:
        return range(self.end - 1, self.start - 1, -1)

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self)
        return (len, {len})


@fieldwise_init
struct _StridedRangeIterator(
    Iterable, Iterator, Sized, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Int
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __len__(self) -> Int:
        if self.step > 0 and self.start < self.end:
            return self.end - self.start
        elif self.step < 0 and self.start > self.end:
            return self.start - self.end
        else:
            return 0

    @always_inline
    fn __next__(mut self) raises StopIteration -> Int:
        if self.__len__() <= 0:
            raise StopIteration()
        var result = self.start
        self.start += self.step
        return result

    # FIXME(GENAI-359): Opt'ing into old-style foreach code generation is
    # necessary to get an AMD355 test working.
    @always_inline
    fn __next_old__(mut self) -> Self.Element:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self)
        return (len, {len})


struct _StridedRange(
    Iterable, Iterator, ReversibleRange, Sized, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _StridedRangeIterator
    comptime Element = Int
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    fn __init__(out self, start: Int, end: Int, step: Int = 1):
        self.start = start
        self.end = end
        self.step = step

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return _StridedRangeIterator(self.start, self.end, self.step)

    @always_inline
    fn __next__(mut self) raises StopIteration -> Int:
        if self.__len__() <= 0:
            raise StopIteration()
        var result = self.start
        self.start += self.step
        return result

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
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self)
        return (len, {len})


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


# ===----------------------------------------------------------------------=== #
# Range Scalar
# ===----------------------------------------------------------------------=== #


fn _scalar_range_bounds[
    dtype: DType
](len: Scalar[dtype]) -> Tuple[Int, Optional[Int]]:
    comptime if size_of[Scalar[dtype]]() >= size_of[Int]():
        if unlikely(UInt(len) > UInt(Int.MAX)):
            return (Int.MAX, None)

    return (Int(len), {Int(len)})


struct _ZeroStartingScalarRange[dtype: DType](
    Iterable, TrivialRegisterPassable, Iterator & ImplicitlyCopyable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    var curr: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]

    @always_inline
    fn __init__(out self, end: Scalar[Self.dtype]):
        self.curr = max(end, 0)
        self.end = self.curr

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        var curr = self.curr
        if curr == 0:
            raise StopIteration()
        self.curr = curr - 1
        return self.end - curr

    # FIXME(GENAI-359): Remove __next_old__ and __has_next__ once we figure out
    # why doing so regresses code generation.
    @always_inline
    fn __next_old__(mut self) -> Scalar[Self.dtype]:
        var curr = self.curr
        self.curr -= 1
        return self.end - curr

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Scalar[Self.dtype]:
        return self.curr

    @always_inline
    fn __getitem__(self, idx: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        debug_assert(idx < self.__len__(), "index out of range")
        return idx

    @always_inline
    fn __reversed__(self) -> _StridedScalarRange[Self.dtype]:
        comptime assert (
            not Self.dtype.is_unsigned()
        ), "cannot reverse an unsigned range"
        return range(
            self.end - 1, Scalar[Self.dtype](-1), Scalar[Self.dtype](-1)
        )

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.__len__())


struct _SequentialScalarRange[dtype: DType](
    ImplicitlyCopyable, Iterable, Iterator, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    var start: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]

    @always_inline
    fn __init__(out self, start: Scalar[Self.dtype], end: Scalar[Self.dtype]):
        self.start = start
        self.end = max(start, end)

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        var start = self.start
        if start == self.end:
            raise StopIteration()
        self.start = start + 1
        return start

    @always_inline
    fn __len__(self) -> Scalar[Self.dtype]:
        return self.end - self.start

    @always_inline
    fn __getitem__(self, idx: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + idx

    @always_inline
    fn __reversed__(self) -> _StridedScalarRange[Self.dtype]:
        comptime assert (
            not Self.dtype.is_unsigned()
        ), "cannot reverse an unsigned range"
        return range(self.end - 1, self.start - 1, Scalar[Self.dtype](-1))

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.__len__())


@fieldwise_init
struct _StridedScalarRange[dtype: DType](
    ImplicitlyCopyable, Iterable, Iterator, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    var start: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]
    var step: Scalar[Self.dtype]

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        # If the type is unsigned, then 'step' cannot be negative.
        comptime if Self.dtype.is_unsigned():
            if self.start >= self.end:
                raise StopIteration()
        else:
            if self.step > 0:
                if self.start >= self.end:
                    raise StopIteration()
            elif self.end >= self.start:
                raise StopIteration()

        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __len__(self) -> Scalar[Self.dtype]:
        comptime assert Self.dtype.is_integral(), "dtype must be integral"

        comptime if Self.dtype.is_unsigned():
            return Scalar[Self.dtype](
                select(
                    self.start < self.end,
                    ceildiv(self.end - self.start, self.step),
                    0,
                )
            )
        else:  # is_signed
            return Scalar[Self.dtype](
                range(Int(self.start), Int(self.end), Int(self.step)).__len__()
            )

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.__len__())

    @always_inline
    fn __getitem__(self, idx: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        debug_assert(idx < self.__len__(), "index out of range")
        return self.start + idx * self.step


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
