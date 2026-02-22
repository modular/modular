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
"""Iteration traits and utilities: Iterable, Iterator, enumerate, zip, map.

This package defines the core iteration protocol for Mojo through the `Iterable`
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

from builtin.constrained import _constrained_conforms_to

# ===-----------------------------------------------------------------------===#
# Iterable
# ===-----------------------------------------------------------------------===#


trait Iterable:
    """The `Iterable` trait describes a type that can be turned into an
    iterator.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Returns an iterator over the elements of this iterable.

        Returns:
            An iterator over the elements.
        """
        ...


# ===-----------------------------------------------------------------------===#
# Iterator
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct StopIteration(TrivialRegisterPassable, Writable):
    """A custom error type for Iterator's that run out of elements."""

    fn write_to(self, mut writer: Some[Writer]):
        """This always writes "StopIteration".

        Args:
            writer: The writer to write to.
        """
        writer.write("StopIteration")


trait Iterator(ImplicitlyDestructible, Movable):
    """The `Iterator` trait describes a type that can be used as an
    iterator, e.g. in a `for` loop.
    """

    comptime Element: Movable

    fn __next__(mut self) raises StopIteration -> Self.Element:
        """Returns the next element from the iterator.

        Raises:
            StopIteration if there are no more elements.

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
](ref iterable: IterableType) -> IterableType.IteratorType[origin_of(iterable)]:
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
](mut iterator: IteratorType) raises StopIteration -> IteratorType.Element:
    """Advances the iterator and returns the next element.

    Parameters:
        IteratorType: The type of the iterator.

    Args:
        iterator: The iterator to advance.

    Returns:
        The next element from the iterator.

    Raises:
        StopIteration: If the iterator is exhausted.
    """
    return iterator.__next__()


# ===-----------------------------------------------------------------------===#
# enumerate
# ===-----------------------------------------------------------------------===#


struct _Enumerate[InnerIteratorType: Iterator](Copyable, Iterable, Iterator):
    """An iterator that yields tuples of the index and the element of the
    original iterator.
    """

    comptime Element = Tuple[Int, Self.InnerIteratorType.Element]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    var _inner: Self.InnerIteratorType
    var _count: Int

    fn __init__(
        out self, var iterator: Self.InnerIteratorType, *, start: Int = 0
    ):
        self._inner = iterator^
        self._count = start

    fn __init__(out self, *, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.InnerIteratorType, Copyable),
            Parent=Self,
            Element = Self.InnerIteratorType,
            ParentConformsTo="Copyable",
        ]()
        self._inner = rebind_var[Self.InnerIteratorType](
            trait_downcast[Copyable](copy._inner).copy()
        )
        self._count = copy._count

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __next__(mut self) raises StopIteration -> Self.Element:
        # This raises on error.
        var elt = next(self._inner)
        var count = self._count
        self._count += 1
        return count, elt^

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
fn enumerate[
    IterableType: Iterable
](ref iterable: IterableType, *, start: Int = 0) -> _Enumerate[
    IterableType.IteratorType[origin_of(iterable)]
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
    Copyable, Iterable, Iterator
):
    comptime Element = Tuple[
        Self.IteratorTypeA.Element, Self.IteratorTypeB.Element
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _inner_a: Self.IteratorTypeA
    var _inner_b: Self.IteratorTypeB

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __init__(out self, *, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeA,
            ParentConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeB,
            ParentConformsTo="Copyable",
        ]()
        self._inner_a = rebind_var[Self.IteratorTypeA](
            trait_downcast[Copyable](copy._inner_a).copy()
        )
        self._inner_b = rebind_var[Self.IteratorTypeB](
            trait_downcast[Copyable](copy._inner_b).copy()
        )

    fn __next__(mut self) raises StopIteration -> Self.Element:
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeA,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeB,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()

        var a = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_a)
        )
        var b = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_b)
        )
        return (
            rebind_var[Self.IteratorTypeA.Element](a^),
            rebind_var[Self.IteratorTypeB.Element](b^),
        )

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return _min_bounds(self._inner_a.bounds(), self._inner_b.bounds())


@fieldwise_init
struct _Zip3[
    IteratorTypeA: Iterator, IteratorTypeB: Iterator, IteratorTypeC: Iterator
](Copyable, Iterable, Iterator):
    comptime Element = Tuple[
        Self.IteratorTypeA.Element,
        Self.IteratorTypeB.Element,
        Self.IteratorTypeC.Element,
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _inner_a: Self.IteratorTypeA
    var _inner_b: Self.IteratorTypeB
    var _inner_c: Self.IteratorTypeC

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __init__(out self, *, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeA,
            ParentConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeB,
            ParentConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeC, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeC,
            ParentConformsTo="Copyable",
        ]()
        self._inner_a = rebind_var[Self.IteratorTypeA](
            trait_downcast[Copyable](copy._inner_a).copy()
        )
        self._inner_b = rebind_var[Self.IteratorTypeB](
            trait_downcast[Copyable](copy._inner_b).copy()
        )
        self._inner_c = rebind_var[Self.IteratorTypeC](
            trait_downcast[Copyable](copy._inner_c).copy()
        )

    fn __next__(mut self) raises StopIteration -> Self.Element:
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeA,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeB,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeC, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeC,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()

        var a = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_a)
        )
        var b = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_b)
        )
        var c = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_c)
        )
        return (
            rebind_var[Self.IteratorTypeA.Element](a^),
            rebind_var[Self.IteratorTypeB.Element](b^),
            rebind_var[Self.IteratorTypeC.Element](c^),
        )

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
](Copyable, Iterable, Iterator):
    comptime Element = Tuple[
        Self.IteratorTypeA.Element,
        Self.IteratorTypeB.Element,
        Self.IteratorTypeC.Element,
        Self.IteratorTypeD.Element,
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _inner_a: Self.IteratorTypeA
    var _inner_b: Self.IteratorTypeB
    var _inner_c: Self.IteratorTypeC
    var _inner_d: Self.IteratorTypeD

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __init__(out self, *, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeA,
            ParentConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeB,
            ParentConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeC, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeC,
            ParentConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeD, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeD,
            ParentConformsTo="Copyable",
        ]()
        self._inner_a = rebind_var[Self.IteratorTypeA](
            trait_downcast[Copyable](copy._inner_a).copy()
        )
        self._inner_b = rebind_var[Self.IteratorTypeB](
            trait_downcast[Copyable](copy._inner_b).copy()
        )
        self._inner_c = rebind_var[Self.IteratorTypeC](
            trait_downcast[Copyable](copy._inner_c).copy()
        )
        self._inner_d = rebind_var[Self.IteratorTypeD](
            trait_downcast[Copyable](copy._inner_d).copy()
        )

    fn __next__(mut self) raises StopIteration -> Self.Element:
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeA,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeB,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeC, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeC,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeD, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.IteratorTypeD,
            ParentConformsTo="Iterator",
            ElementConformsTo="ImplicitlyDestructible",
        ]()

        var a = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_a)
        )
        var b = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_b)
        )
        var c = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_c)
        )
        var d = trait_downcast_var[Movable & ImplicitlyDestructible](
            next(self._inner_d)
        )
        return (
            rebind_var[Self.IteratorTypeA.Element](a^),
            rebind_var[Self.IteratorTypeB.Element](b^),
            rebind_var[Self.IteratorTypeC.Element](c^),
            rebind_var[Self.IteratorTypeD.Element](d^),
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
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    IterableTypeB.IteratorType[origin_of(iterable_b)],
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
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    IterableTypeB.IteratorType[origin_of(iterable_b)],
    IterableTypeC.IteratorType[origin_of(iterable_c)],
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
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    IterableTypeB.IteratorType[origin_of(iterable_b)],
    IterableTypeC.IteratorType[origin_of(iterable_c)],
    IterableTypeD.IteratorType[origin_of(iterable_d)],
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
    OutputType: Copyable,
    InnerIteratorType: Iterator,
    //,
    function: fn(var InnerIteratorType.Element) -> OutputType,
](Copyable, Iterable, Iterator):
    comptime Element = Self.OutputType
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _inner: Self.InnerIteratorType

    fn __init__(out self, *, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.InnerIteratorType, Copyable),
            Parent=Self,
            Element = Self.InnerIteratorType,
            ParentConformsTo="Copyable",
        ]()
        self._inner = rebind_var[Self.InnerIteratorType](
            trait_downcast[Copyable](copy._inner).copy()
        )

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __next__(mut self) raises StopIteration -> Self.Element:
        return Self.function(next(self._inner))

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
fn map[
    origin: ImmutOrigin,
    IterableType: Iterable,
    ResultType: Copyable,
    //,
    function: fn(var IterableType.IteratorType[origin].Element) -> ResultType,
](ref[origin] iterable: IterableType) -> _MapIterator[function]:
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
    # FIXME(MOCO-3238): This rebind shouldn't ve needed, something isn't getting
    # substituted through associated types right.
    return {
        rebind_var[_MapIterator[function].InnerIteratorType](iter(iterable))
    }


# ===-----------------------------------------------------------------------===#
# peekable
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _PeekableIterator[InnerIterator: Iterator](Copyable, Iterable, Iterator):
    comptime Element = Self.InnerIterator.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _inner: Self.InnerIterator
    var _next: Optional[Self.Element]

    fn __init__(out self, var inner: Self.InnerIterator):
        self._inner = inner^
        self._next = None

    fn __init__(out self, *, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.InnerIterator, Copyable),
            Parent=Self,
            Element = Self.InnerIterator,
            ParentConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.Element, Copyable),
            Parent=Self,
            Element = Self.Element,
            ParentConformsTo="Copyable",
        ]()

        self._inner = rebind_var[Self.InnerIterator](
            trait_downcast[Copyable](copy._inner).copy()
        )
        self._next = copy._next.copy()

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __next__(mut self) raises StopIteration -> Self.Element:
        if self._next:
            return self._next.unsafe_take()
        return next(self._inner)

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var peek_len = 1 if self._next else 0
        var lower, upper = self._inner.bounds()
        if upper:
            return (lower + peek_len, upper.value() + peek_len)
        else:
            return (lower + peek_len, None)

    fn peek(
        mut self,
    ) -> Optional[Pointer[Self.Element, ImmutOrigin(origin_of(self._next[]))]]:
        if not self._next:
            try:
                self._next = next(self._inner)
            except:
                return None
        return Pointer(to=self._next.unsafe_value()).get_immutable()


fn peekable(
    ref iterable: Some[Iterable],
) -> _PeekableIterator[type_of(iterable).IteratorType[origin_of(iterable)]]:
    """Returns a peekable iterator that can use the `peek` method to look ahead
    at the next element without advancing the iterator.

    Args:
        iterable: The iterable to create a peekable iterator from.

    Returns:
        A peekable iterator.
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
