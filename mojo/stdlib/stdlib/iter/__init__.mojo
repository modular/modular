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


trait Iterable:
    """The `Iterator` trait describes a type that can be turned into an
    iterator.
    """

    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        ...


trait Iterator(Copyable, Movable):
    """The `Iterator` trait describes a type that can be used as an
    iterator, e.g. in a `for` loop.
    """

    alias Element: Copyable & Movable

    fn __has_next__(self) -> Bool:
        ...

    fn __next__(mut self) -> Self.Element:
        ...


@always_inline
fn iter[
    IterableType: Iterable
](ref iterable: IterableType) -> IterableType.IteratorType[
    __origin_of(iterable)
]:
    return iterable.__iter__()


@always_inline
fn next[
    IteratorType: Iterator
](mut iterator: IteratorType) -> IteratorType.Element:
    return iterator.__next__()


struct _Enumerate[InnerIteratorType: Iterator](
    Copyable, Iterable, Iterator, Movable
):
    """The `enumerate` function returns an iterator that yields tuples of the
    index and the element of the original iterator.
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


@always_inline
fn enumerate[
    IterableType: Iterable
](ref iterable: IterableType, *, start: Int = 0) -> _Enumerate[
    IterableType.IteratorType[__origin_of(iterable)]
]:
    """The `enumerate` function returns an iterator that yields tuples of the
    index and the element of the original iterator.

    Args:
        iterable: An iterable object (e.g., list, string, etc.).
        start: The starting index for enumeration (default is 0).

    Example:

    ```mojo
    var l = ["hey", "hi", "hello"]
    for i, elem in enumerate(l):
        print(i, elem)
    ```
    """
    return _Enumerate(iter(iterable), start=start)


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


@always_inline
fn zip[
    IterableTypeA: Iterable, IterableTypeB: Iterable
](ref iterable_a: IterableTypeA, ref iterable_b: IterableTypeB) -> _Zip2[
    IterableTypeA.IteratorType[__origin_of(iterable_a)],
    IterableTypeB.IteratorType[__origin_of(iterable_b)],
]:
    """Returns an iterator that yields tuples of the
    elements of the original iterables.

    # Examples
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
    """Returns an iterator that yields tuples of the
    elements of the original iterables.

    # Examples
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
    """Returns an iterator that yields tuples of the
    elements of the original iterables.

    # Examples
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
