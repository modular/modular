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
"""Provides traits and utilities for iteration.

This module defines the core iteration protocol for Mojo through the `Iterable`
and `Iterator` traits. Types that conform to these traits can be used with
`for` loops and iteration utilities like `enumerate()`, `zip()`, and `map()`.

The iteration protocol consists of two key traits:

- `Iterable`: Types that can be converted into an iterator
- `Iterator`: Types that can produce a sequence of values

Examples:

```mojo
from iter import enumerate, zip, map

# Enumerate with index
var items = ["a", "b", "c"]
for index, value in enumerate(items):
    print(index, value)

# Zip multiple iterables
var numbers = [1, 2, 3]
var letters = ["x", "y", "z"]
for num, letter in zip(numbers, letters):
    print(num, letter)

# Map a function over an iterable
fn square(x: Int) -> Int:
    return x * x
var values = [1, 2, 3, 4]
for squared in map[square](values):
    print(squared)
```
"""


# ===-----------------------------------------------------------------------===#
# Iterable
# ===-----------------------------------------------------------------------===#


trait Iterable:
    """The `Iterable` trait describes a type that can be turned into an
    iterator.
    """

    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        """Returns an iterator over the elements of this iterable.

        Returns:
            An iterator over the elements.
        """
        ...


# ===-----------------------------------------------------------------------===#
# Iterator
# ===-----------------------------------------------------------------------===#


trait Iterator(Copyable, Movable):
    """The `Iterator` trait describes a type that can be used as an
    iterator, e.g. in a `for` loop.
    """

    alias Element: Copyable & Movable

    fn __has_next__(self) -> Bool:
        """Checks if there are more elements in the iterator.

        Returns:
            True if there are more elements, False otherwise.
        """
        ...

    fn __next__(mut self) -> Self.Element:
        """Returns the next element from the iterator.

        Returns:
            The next element.
        """
        ...

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        """Returns bounds `[lower, upper]` for the remaining iterator length.

        This helps collections pre-allocate memory when constructed from iterators.
        The default implementation returns `(0, None)`.

        Returns:
            A tuple where the first element is the lower bound and the second
            is an optional upper bound (`None` means unknown or `upper > Int.MAX`).

        Safety:

        If the upper bound is not None, implementations must ensure that `lower <= upper`.
        The bounds are hints only - iterators may not comply with them. Never omit safety
        checks when using `bounds` to build collections.

        Examples:

        ```mojo
        fn to_int_list[I: Iterable](iter: I) -> List[Int]:
            var lower, _upper = iter.bounds()
            var list = List[Int](capacity=lower)
            for element in iter:
                list.append(rebind[Int](element))
            return list^
        ```
        """
        return (0, None)


@always_inline
fn iter[
    IterableType: Iterable
](ref iterable: IterableType) -> IterableType.IteratorType[
    __origin_of(iterable)
]:
    """Constructs an iterator from an iterable.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to construct the iterator from.

    Returns:
        An iterator for the given iterable.
    """
    return iterable.__iter__()


@always_inline
fn next[
    IteratorType: Iterator
](mut iterator: IteratorType) -> IteratorType.Element:
    """Advances the iterator and returns the next element.

    Parameters:
        IteratorType: The type of the iterator.

    Args:
        iterator: The iterator to advance.

    Returns:
        The next element from the iterator.
    """
    return iterator.__next__()


# ===-----------------------------------------------------------------------===#
# enumerate
# ===-----------------------------------------------------------------------===#


struct _Enumerate[InnerIteratorType: Iterator](
    Copyable, Iterable, Iterator, Movable
):
    """An iterator that yields tuples of the index and the element of the
    original iterator.
    """

    alias Element = Tuple[Int, InnerIteratorType.Element]
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self
    var _inner: InnerIteratorType
    var _count: Int

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        return self.copy()

    fn __init__(out self, var iterator: InnerIteratorType, *, start: Int = 0):
        self._inner = iterator^
        self._count = start

    fn __has_next__(self) -> Bool:
        return self._inner.__has_next__()

    fn __next__(mut self) -> Self.Element:
        var count = self._count
        self._count += 1
        return count, next(self._inner)

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
fn enumerate[
    IterableType: Iterable
](ref iterable: IterableType, *, start: Int = 0) -> _Enumerate[
    IterableType.IteratorType[__origin_of(iterable)]
]:
    """Returns an iterator that yields tuples of the index and the element of
    the original iterator.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: An iterable object (e.g., list, string, etc.).
        start: The starting index for enumeration (default is 0).

    Returns:
        An enumerate iterator that yields tuples of `(index, element)`.

    Examples:

    ```mojo
    var l = ["hey", "hi", "hello"]
    for i, elem in enumerate(l):
        print(i, elem)
    ```
    """
    return _Enumerate(iter(iterable), start=start)


# ===-----------------------------------------------------------------------===#
# zip
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Zip2[IteratorTypeA: Iterator, IteratorTypeB: Iterator](
    Copyable, Iterable, Iterator, Movable
):
    alias Element = Tuple[IteratorTypeA.Element, IteratorTypeB.Element]
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    var _inner_a: IteratorTypeA
    var _inner_b: IteratorTypeB

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(self._inner_a.copy(), self._inner_b.copy())

    fn __has_next__(self) -> Bool:
        return self._inner_a.__has_next__() and self._inner_b.__has_next__()

    fn __next__(mut self) -> Self.Element:
        return next(self._inner_a), next(self._inner_b)

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return _min_bounds(self._inner_a.bounds(), self._inner_b.bounds())


@fieldwise_init
struct _Zip3[
    IteratorTypeA: Iterator, IteratorTypeB: Iterator, IteratorTypeC: Iterator
](Copyable, Iterable, Iterator, Movable):
    alias Element = Tuple[
        IteratorTypeA.Element, IteratorTypeB.Element, IteratorTypeC.Element
    ]
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    var _inner_a: IteratorTypeA
    var _inner_b: IteratorTypeB
    var _inner_c: IteratorTypeC

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(
            self._inner_a.copy(), self._inner_b.copy(), self._inner_c.copy()
        )

    fn __has_next__(self) -> Bool:
        return (
            self._inner_a.__has_next__()
            and self._inner_b.__has_next__()
            and self._inner_c.__has_next__()
        )

    fn __next__(mut self) -> Self.Element:
        return next(self._inner_a), next(self._inner_b), next(self._inner_c)

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return _min_bounds(
            self._inner_a.bounds(),
            self._inner_b.bounds(),
            self._inner_c.bounds(),
        )


@fieldwise_init
struct _Zip4[
    IteratorTypeA: Iterator,
    IteratorTypeB: Iterator,
    IteratorTypeC: Iterator,
    IteratorTypeD: Iterator,
](Copyable, Iterable, Iterator, Movable):
    alias Element = Tuple[
        IteratorTypeA.Element,
        IteratorTypeB.Element,
        IteratorTypeC.Element,
        IteratorTypeD.Element,
    ]
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    var _inner_a: IteratorTypeA
    var _inner_b: IteratorTypeB
    var _inner_c: IteratorTypeC
    var _inner_d: IteratorTypeD

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(
            self._inner_a.copy(),
            self._inner_b.copy(),
            self._inner_c.copy(),
            self._inner_d.copy(),
        )

    fn __has_next__(self) -> Bool:
        return (
            self._inner_a.__has_next__()
            and self._inner_b.__has_next__()
            and self._inner_c.__has_next__()
            and self._inner_d.__has_next__()
        )

    fn __next__(mut self) -> Self.Element:
        return (
            next(self._inner_a),
            next(self._inner_b),
            next(self._inner_c),
            next(self._inner_d),
        )

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return _min_bounds(
            self._inner_a.bounds(),
            self._inner_b.bounds(),
            self._inner_c.bounds(),
            self._inner_d.bounds(),
        )


@always_inline
fn zip[
    IterableTypeA: Iterable, IterableTypeB: Iterable
](ref iterable_a: IterableTypeA, ref iterable_b: IterableTypeB) -> _Zip2[
    IterableTypeA.IteratorType[__origin_of(iterable_a)],
    IterableTypeB.IteratorType[__origin_of(iterable_b)],
]:
    """Returns an iterator that yields tuples of the elements of the original
    iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.

    Returns:
        A zip iterator that yields tuples of elements from both iterables.

    Examples:

    ```mojo
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    for a, b in zip(l, l2):
        print(a, b)
    ```
    """
    return _Zip2(iter(iterable_a), iter(iterable_b))


@always_inline
fn zip[
    IterableTypeA: Iterable, IterableTypeB: Iterable, IterableTypeC: Iterable
](
    ref iterable_a: IterableTypeA,
    ref iterable_b: IterableTypeB,
    ref iterable_c: IterableTypeC,
) -> _Zip3[
    IterableTypeA.IteratorType[__origin_of(iterable_a)],
    IterableTypeB.IteratorType[__origin_of(iterable_b)],
    IterableTypeC.IteratorType[__origin_of(iterable_c)],
]:
    """Returns an iterator that yields tuples of the elements of the original
    iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.
        IterableTypeC: The type of the third iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.
        iterable_c: The third iterable.

    Returns:
        A zip iterator that yields tuples of elements from all three iterables.

    Examples:

    ```mojo
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var l3 = [100, 200, 300]
    for a, b, c in zip(l, l2, l3):
        print(a, b, c)
    ```
    """
    return _Zip3(iter(iterable_a), iter(iterable_b), iter(iterable_c))


@always_inline
fn zip[
    IterableTypeA: Iterable,
    IterableTypeB: Iterable,
    IterableTypeC: Iterable,
    IterableTypeD: Iterable,
](
    ref iterable_a: IterableTypeA,
    ref iterable_b: IterableTypeB,
    ref iterable_c: IterableTypeC,
    ref iterable_d: IterableTypeD,
) -> _Zip4[
    IterableTypeA.IteratorType[__origin_of(iterable_a)],
    IterableTypeB.IteratorType[__origin_of(iterable_b)],
    IterableTypeC.IteratorType[__origin_of(iterable_c)],
    IterableTypeD.IteratorType[__origin_of(iterable_d)],
]:
    """Returns an iterator that yields tuples of the elements of the original
    iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.
        IterableTypeC: The type of the third iterable.
        IterableTypeD: The type of the fourth iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.
        iterable_c: The third iterable.
        iterable_d: The fourth iterable.

    Returns:
        A zip iterator that yields tuples of elements from all four iterables.

    Examples:

    ```mojo
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var l3 = [100, 200, 300]
    var l4 = [1000, 2000, 3000]
    for a, b, c, d in zip(l, l2, l3, l4):
        print(a, b, c, d)
    ```
    """
    return _Zip4(
        iter(iterable_a), iter(iterable_b), iter(iterable_c), iter(iterable_d)
    )


# ===-----------------------------------------------------------------------===#
# map
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _MapIterator[
    OutputType: Copyable & Movable,
    InnerIteratorType: Iterator, //,
    function: fn (var InnerIteratorType.Element) -> OutputType,
](Copyable, Iterable, Iterator, Movable):
    alias Element = OutputType
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    var _inner: InnerIteratorType

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        return self.copy()

    fn __has_next__(self) -> Bool:
        return self._inner.__has_next__()

    fn __next__(mut self) -> Self.Element:
        return function(next(self._inner))

    fn copy(self) -> Self:
        return Self(self._inner.copy())

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
fn map[
    origin: ImmutableOrigin,
    IterableType: Iterable,
    ResultType: Copyable & Movable, //,
    function: fn (var IterableType.IteratorType[origin].Element) -> ResultType,
](ref [origin]iterable: IterableType) -> _MapIterator[
    OutputType=ResultType, function=function
]:
    """Returns an iterator that applies `function` to each element of the input
    iterable.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.
        ResultType: The return type of the function.
        function: The function to apply to each element.

    Args:
        iterable: The iterable to map over.

    Returns:
        A map iterator that yields the results of applying `function` to each
        element.

    Examples:

    ```mojo
    var l = [1, 2, 3]
    fn add_one(x: Int) -> Int:
        return x + 1
    var m = map[add_one](l)

    # outputs:
    # 2
    # 3
    # 4
    for elem in m:
        print(elem)
    ```
    """
    return {iter(iterable)}


# ===-----------------------------------------------------------------------===#
# utilities
# ===-----------------------------------------------------------------------===#


fn _min_bounds(*bounds: Tuple[Int, Optional[Int]]) -> Tuple[Int, Optional[Int]]:
    var res_lower = Int.MAX
    var res_upper = Optional[Int](None)

    for bound in bounds:
        var lower, upper = bound
        res_lower = min(res_lower, lower)
        if upper:
            res_upper = min(res_upper.or_else(Int.MAX), upper.value())

    return (res_lower, res_upper)
