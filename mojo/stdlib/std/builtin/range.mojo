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
"""Defines Mojo's built-in `range()` function.

In Mojo, ranges are values, not loop constructs, generators, or lists. Every
range is a half-open interval, [start, end).

The stand-alone `range()` function constructs zero-based, sequential, and
strided ranges.

`range()` is built in. You don't need to import it.
"""

from std.math import ceil, ceildiv, fma
from std.sys.info import size_of
from std.sys.intrinsics import unlikely

from std.python import PythonObject

from std.utils._select import _select_register_value as select

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
def _sign[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    var result = Scalar[dtype](0)
    result = select(x > 0, Scalar[dtype](1), result)
    result = select(x < 0, Scalar[dtype](-1), result)
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
    comptime ReversedType = _StridedRange
    var curr: Int
    var end: Int

    @always_inline
    def __init__(out self, end: Int):
        self.curr = max(end, 0)
        self.end = self.curr

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        var curr = self.curr
        if curr == 0:
            raise StopIteration()
        self.curr = curr - 1
        return self.end - curr

    @always_inline
    def __len__(self) -> Int:
        return self.curr

    @always_inline
    def __getitem__[I: Indexer](self, idx: I) -> Int:
        var i = index(idx)
        assert i < self.__len__(), "index out of range"
        return i

    @always_inline
    def __reversed__(self) -> _StridedRange:
        return _StridedRange(self.end - 1, -1, -1)

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self)
        return (len, {len})


struct _SequentialRange(
    Iterable, Iterator, ReversibleRange, Sized, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Int
    comptime ReversedType = _StridedRange
    var start: Int
    var end: Int

    @always_inline
    def __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = max(start, end)

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        var start = self.start
        if start == self.end:
            raise StopIteration()
        self.start = start + 1
        return start

    @always_inline
    def __len__(self) -> Int:
        return self.end - self.start

    @always_inline
    def __getitem__[I: Indexer](self, idx: I) -> Int:
        assert self.__len__() > index(idx), "index out of range"
        return self.start + index(idx)

    @always_inline
    def __reversed__(self) -> _StridedRange:
        return _StridedRange(self.end - 1, self.start - 1, -1)

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
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
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __len__(self) -> Int:
        if self.step > 0 and self.start < self.end:
            return self.end - self.start
        elif self.step < 0 and self.start > self.end:
            return self.start - self.end
        else:
            return 0

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        if self.__len__() <= 0:
            raise StopIteration()
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self)
        return (len, {len})


struct _StridedRange(
    Iterable, Iterator, ReversibleRange, Sized, TrivialRegisterPassable
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _StridedRangeIterator
    comptime Element = Int
    comptime ReversedType = _StridedRange
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    def __init__(out self, start: Int, end: Int, step: Int = 1):
        self.start = start
        self.end = end
        self.step = step

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return _StridedRangeIterator(self.start, self.end, self.step)

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        if self.__len__() <= 0:
            raise StopIteration()
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    def __len__(self) -> Int:
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
    def __getitem__[I: Indexer](self, idx: I) -> Int:
        assert self.__len__() > index(idx), "index out of range"
        return self.start + index(idx) * self.step

    @always_inline
    def __reversed__(self) -> _StridedRange:
        var shifted_end = self.end - _sign(self.step)
        var start = shifted_end - ((shifted_end - self.start) % self.step)
        var end = self.start - self.step
        var step = -self.step
        return _StridedRange(start, end, step)

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self)
        return (len, {len})


@always_inline
def range[T: Indexer, //](end: T) -> _ZeroStartingRange:
    """Returns the integer sequence `[0, end)`.

    Integer ranges are values. They support `len()`, O(1) indexing, and
    `reversed()` without allocating. `reversed(range(n))` iterates from
    `n - 1` down to `0`.

    Parameters:
        T: The type of the end value. Constrained to `Indexer`.

    Args:
        end: The exclusive upper bound. Negative values produce an empty range.

    Returns:
        A zero-based integer range over `[0, end)`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(5):
        print(i)  # 0, 1, 2, 3, 4

    var steps = range(1000)
    print(steps[499])  # 499
    print(len(steps))  # 1000

    for i in reversed(range(5)):
        print(i)  # 4, 3, 2, 1, 0
    ```
    """
    return _ZeroStartingRange(index(end))


@always_inline
def range[T: Indexer, //](start: T, end: T) -> _SequentialRange:
    """Returns the integer sequence `[start, end)`.

    **The two-argument form never counts down.** `range(7, 3)` is empty,
    not `[7, 6, 5, 4]`. Use the three-argument form with a negative step to
    count downward. The range supports `len()`, O(1) indexing, and
    `reversed()`.

    Parameters:
        T: The type of the start and end values. Constrained to `Indexer`.

    Args:
        start: The inclusive lower bound.
        end: The exclusive upper bound. When `end <= start`, the range is empty.

    Returns:
        A sequential integer range over `[start, end)`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(3, 7):
        print(i)  # 3, 4, 5, 6

    for i in range(-3, 4):
        print(i)  # -3, -2, -1, 0, 1, 2, 3

    print(len(range(7, 3)))  # 0
    ```
    """
    return _SequentialRange(index(start), index(end))


@always_inline
def range[T: Indexer, //](start: T, end: T, step: T) -> _StridedRange:
    """Returns the integer sequence `[start, end)` with a given step.

    When you don't know which bound is larger, choose the direction with an
    inline conditional:

    ```mojo
    var step = 1 if end > start else -1
    for i in range(start, end, step):
        ...
    ```

    Parameters:
        T: The type of the start, end, and step values. Constrained to
            `Indexer`.

    Args:
        start: The inclusive lower bound when stepping forward, or the
            inclusive upper bound when stepping backward.
        end: The exclusive bound in the direction of the step.
        step: The increment per iteration. A positive step counts up, and a
            negative step counts down. A zero step produces an empty range.

    Returns:
        A strided integer range over `[start, end)` by `step`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(0, 10, 2):
        print(i)  # 0, 2, 4, 6, 8

    for i in range(7, 3, -1):
        print(i)  # 7, 6, 5, 4

    var evens = range(0, 2_000_000, 2)
    print(evens[999_999])  # 1_999_998
    print(len(evens))      # 1_000_000
    ```
    """
    return _StridedRange(index(start), index(end), index(step))


@always_inline
def range(end: Int) -> _ZeroStartingRange:
    """Returns the `Int` sequence `[0, end)`.

    Integer ranges are values. They support `len()`, O(1) indexing, and
    `reversed()` without allocating. `reversed(range(n))` iterates from
    `n - 1` down to `0`.

    Args:
        end: The exclusive upper bound. Negative values produce an empty range.

    Returns:
        A zero-based integer range over `[0, end)`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(5):
        print(i)  # 0, 1, 2, 3, 4

    var steps = range(1000)
    print(steps[499])  # 499
    print(len(steps))  # 1000

    for i in reversed(range(5)):
        print(i)  # 4, 3, 2, 1, 0
    ```
    """
    return _ZeroStartingRange(end)


@always_inline
def range(start: Int, end: Int) -> _SequentialRange:
    """Returns the `Int` sequence `[start, end)`.

    **The two-argument form never counts down.** `range(7, 3)` is empty,
    not `[7, 6, 5, 4]`. Use the three-argument form with a negative step to
    count downward. The range supports `len()`, O(1) indexing, and
    `reversed()`.

    Args:
        start: The inclusive lower bound.
        end: The exclusive upper bound. When `end <= start`, the range is empty.

    Returns:
        A sequential integer range over `[start, end)`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(3, 7):
        print(i)  # 3, 4, 5, 6

    for i in range(-3, 4):
        print(i)  # -3, -2, -1, 0, 1, 2, 3

    print(len(range(7, 3)))  # 0
    ```
    """
    return _SequentialRange(start, end)


@always_inline
def range(start: Int, end: Int, step: Int) -> _StridedRange:
    """Returns the `Int` sequence `[start, end)` with a given step.

    The range supports `len()`, O(1) indexing, and `reversed()`. When you
    don't know which bound is larger, choose the direction with an inline
    conditional:

    ```mojo
    var step = 1 if end > start else -1
    for i in range(start, end, step):
        ...
    ```

    Args:
        start: The inclusive lower bound when stepping forward, or the
            inclusive upper bound when stepping backward.
        end: The exclusive bound in the direction of the step.
        step: The increment per iteration. A positive step counts up, and a
            negative step counts down. A zero step produces an empty range.

    Returns:
        A strided integer range over `[start, end)` by `step`.

    Performance:
        O(1) construction. O(1) indexing. No list allocation.

    Examples:

    ```mojo
    for i in range(0, 10, 2):
        print(i)  # 0, 2, 4, 6, 8

    for i in range(7, 3, -1):
        print(i)  # 7, 6, 5, 4

    var evens = range(0, 2_000_000, 2)
    print(evens[999_999])  # 1_999_998
    print(len(evens))      # 1_000_000
    ```
    """
    return _StridedRange(start, end, step)


# ===----------------------------------------------------------------------=== #
# Range Scalar
# ===----------------------------------------------------------------------=== #


def _scalar_range_bounds[
    dtype: DType
](len: Scalar[dtype]) -> Tuple[Int, Optional[Int]]:
    comptime if size_of[Scalar[dtype]]() >= size_of[Int]():
        if unlikely(UInt(len) > UInt(Int.MAX)):
            return (Int.MAX, None)

    return (Int(len), {Int(len)})


struct _ZeroStartingScalarRange[dtype: DType](
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    ReversibleRange,
    TrivialRegisterPassable,
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    comptime ReversedType = _StridedScalarRange[Self.dtype]
    var curr: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]

    @always_inline
    def __init__(out self, end: Scalar[Self.dtype]):
        self.curr = max(end, 0)
        self.end = self.curr

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        var curr = self.curr
        self.curr = curr - 1
        if curr == 0:
            raise StopIteration()
        return self.end - curr

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    def __len__(self) -> Scalar[Self.dtype]:
        return self.curr

    @always_inline
    def __getitem__(self, idx: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        assert idx < self.__len__(), "index out of range"
        return idx

    @always_inline
    def __reversed__(self) -> _StridedScalarRange[Self.dtype]:
        comptime assert (
            not Self.dtype.is_unsigned()
        ), "cannot reverse an unsigned range"
        return range(
            self.end - 1, Scalar[Self.dtype](-1), Scalar[Self.dtype](-1)
        )

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.__len__())


struct _SequentialScalarRange[dtype: DType](
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    ReversibleRange,
    TrivialRegisterPassable,
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    comptime ReversedType = _StridedScalarRange[Self.dtype]
    var start: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]

    @always_inline
    def __init__(out self, start: Scalar[Self.dtype], end: Scalar[Self.dtype]):
        self.start = start
        self.end = max(start, end)

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        var start = self.start
        if start == self.end:
            raise StopIteration()
        self.start = start + 1
        return start

    @always_inline
    def __len__(self) -> Scalar[Self.dtype]:
        return self.end - self.start

    @always_inline
    def __getitem__(self, idx: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        assert idx < self.__len__(), "index out of range"
        return self.start + idx

    @always_inline
    def __reversed__(self) -> _StridedScalarRange[Self.dtype]:
        comptime assert (
            not Self.dtype.is_unsigned()
        ), "cannot reverse an unsigned range"
        return range(self.end - 1, self.start - 1, Scalar[Self.dtype](-1))

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.__len__())


@always_inline
def _fp_range_count[
    dtype: DType, //
](start: Scalar[dtype], end: Scalar[dtype], step: Scalar[dtype]) -> Int:
    # A zero step is empty.
    if step == 0:
        return 0
    # This calculation avoids `// + 1`, which overcounts by one when `end`
    # lands on the grid. `ceil` and `/` are correct for forward and backward
    # ranges.
    var raw = ceil((end - start) / step)
    return Int(raw) if raw > 0 else 0


# Floating-point ranges iterate by index (`fma(k, step, start)`), avoiding
# drift. Reverse iteration mirrors forward. One extra `Int` cursor carries
# both position and direction; integer ranges ignore it.
struct _StridedScalarRange[dtype: DType](
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    ReversibleRange,
    Sized,
    TrivialRegisterPassable,
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Scalar[Self.dtype]
    comptime ReversedType = Self
    var start: Scalar[Self.dtype]
    var end: Scalar[Self.dtype]
    var step: Scalar[Self.dtype]
    var idx: Int  # fp iteration cursor; sign is the direction (>= 0 fwd, < 0 rev)

    @always_inline
    def __init__(
        out self,
        start: Scalar[Self.dtype],
        end: Scalar[Self.dtype],
        step: Scalar[Self.dtype],
        idx: Int = 0,
    ):
        self.start = start
        self.end = end
        self.step = step
        self.idx = idx

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Scalar[Self.dtype]:
        comptime if Self.dtype.is_floating_point():
            var count = _fp_range_count(self.start, self.end, self.step)
            if self.idx >= 0:
                if self.idx >= count:
                    raise StopIteration()
                var result = fma(
                    Scalar[Self.dtype](self.idx), self.step, self.start
                )
                self.idx += 1
                return result
            else:
                var i = count + self.idx
                if i < 0:
                    raise StopIteration()
                var result = fma(Scalar[Self.dtype](i), self.step, self.start)
                self.idx -= 1
                return result
        else:
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
    def __len__(self) -> Int:
        comptime assert Self.dtype.is_integral(), "dtype must be integral"

        comptime if Self.dtype.is_unsigned():
            return Int(
                select(
                    self.start < self.end,
                    ceildiv(self.end - self.start, self.step),
                    0,
                )
            )
        else:  # is_signed
            return _StridedRange(
                Int(self.start), Int(self.end), Int(self.step)
            ).__len__()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return _scalar_range_bounds(self.__len__())

    @always_inline
    def __getitem__(self, idx: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        assert Int(idx) < self.__len__(), "index out of range"
        return self.start + idx * self.step

    @always_inline
    def __reversed__(self) -> Self.ReversedType:
        comptime if Self.dtype.is_integral():
            # Integer spacing guarantees that `end - ±step` snaps to the last
            # produced element.
            var shifted_end = self.end - _sign(self.step)
            var start = shifted_end - ((shifted_end - self.start) % self.step)
            return Self(start, self.start - self.step, -self.step)
        else:
            # Reverse starts the cursor at -1; `__next__` maps it to
            # count - 1.
            return Self(self.start, self.end, self.step, -1)


@always_inline
def range[
    dtype: DType, //
](end: Scalar[dtype]) -> _ZeroStartingScalarRange[dtype]:
    """Returns the scalar sequence `[0, end)` with elements of type `dtype`.

    Use this overload when you need typed scalar elements.

    The `dtype` is inferred from the argument, so `Int32(8)` produces
    `Int32` elements. This form requires an integer `dtype`; floating-point
    ranges require an explicit step. Signed-integer ranges can be reversed;
    reversing an unsigned range is a compile-time error. Only the
    three-argument form supports `len()`.

    Parameters:
        dtype: The `DType` of the sequence elements. Inferred from `end`.

    Args:
        end: The exclusive upper bound. Negative values produce an empty range.

    Returns:
        A zero-based scalar range over `[0, end)`.

    Performance:
        O(1). No list allocation.

    Examples:

    ```mojo
    for i in range(UInt8(4)):
        print(i)  # 0, 1, 2, 3 (each value is UInt8)
    ```
    """
    comptime assert dtype.is_numeric(), "range requires a numeric dtype"
    comptime assert dtype.is_integral(), (
        "a floating-point range requires an explicit step; use range(start,"
        " end, step)"
    )
    return _ZeroStartingScalarRange(end)


@always_inline
def range[
    dtype: DType, //
](start: Scalar[dtype], end: Scalar[dtype]) -> _SequentialScalarRange[dtype]:
    """Returns the scalar sequence `[start, end)` with elements of type `dtype`.

        **The two-argument form never counts down.** The range is empty
        when `end <= start`. Use the three-argument form with a negative
        step to count downward. This form requires an integer `dtype`.
        Floating-point ranges require an explicit step. Signed-integer
        ranges can be reversed; reversing an unsigned range is a compile-time
        error. Only the three-argument form supports `len()`.

    Parameters:
        dtype: The `DType` of the sequence elements. Inferred from the arguments.

    Args:
        start: The inclusive lower bound.
        end: The exclusive upper bound. When `end <= start`, the range is empty.

    Returns:
        A sequential scalar range over `[start, end)`.

    Performance:
        O(1). No list allocation.

    Examples:

    ```mojo
    for i in range(Int32(3), Int32(7)):
        print(i)  # 3, 4, 5, 6  — each value is Int32
    ```
    """
    comptime assert dtype.is_numeric(), "range requires a numeric dtype"
    comptime assert dtype.is_integral(), (
        "a floating-point range requires an explicit step; use range(start,"
        " end, step)"
    )
    return _SequentialScalarRange(start, end)


@always_inline
def range[
    dtype: DType, //
](
    start: Scalar[dtype], end: Scalar[dtype], step: Scalar[dtype]
) -> _StridedScalarRange[dtype]:
    """Returns the scalar sequence `[start, end)` with a given step.

    Integer scalar ranges support `len()`, O(1) indexing, and `reversed()`,
    including unsigned ranges. Float ranges are iteration-only. Each element
    is computed as `fma(i, step, start)`, and `reversed()` is a bit-for-bit
    mirror of forward iteration.

    **Float endpoints are exclusive.** To include a specific endpoint, push
    `end` past it by a fraction of the step:

    ```mojo
    # [0.0, 0.25, 0.5, 0.75] — 1.0 not included
    for t in range(Float64(0.0), Float64(1.0), Float64(0.25)):
        print(t)

    # [0.0, 0.25, 0.5, 0.75, 1.0] — endpoint included
    var tolerance = Float64(0.1)
    for t in range(Float64(0.0), Float64(1) + tolerance, Float64(0.25)):
        print(t)
    ```

    **A zero step yields an empty float range.** For integer element types a
    zero step can still iterate forever when the bounds and step disagree (a
    signed range with `end < start`, or an unsigned range with `start < end`),
    so pass a nonzero step for integer ranges.

    Parameters:
        dtype: The `DType` of the sequence elements. Inferred from the arguments.

    Args:
        start: The inclusive lower bound when stepping forward, or the
            inclusive upper bound when stepping backward.
        end: The exclusive bound in the direction of the step.
        step: The increment per iteration. A positive step counts up, and a
            negative step counts down. A zero step yields an empty float range;
            avoid it for integer ranges.

    Returns:
        A strided scalar range over `[start, end)` by `step`.

    Performance:
        O(1). No list allocation.

    Examples:

    ```mojo
    # Walk t over [0, 1)
    for t in range(Float64(0.0), Float64(1.0), Float64(0.25)):
        print(t)  # 0.0, 0.25, 0.5, 0.75

    # Integer scalar range — len() and indexed access work
    var r = range(Int32(0), Int32(20), Int32(2))
    print(len(r))       # 10
    print(r[Int32(3)])  # 6
    ```
    """
    comptime assert dtype.is_numeric(), "range requires a numeric dtype"
    return _StridedScalarRange(start, end, step)
