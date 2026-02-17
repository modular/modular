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
"""Provides iterator utilities for common iteration patterns.

This module includes functions for creating specialized iterators:

- `count()` - Creates an infinite counter with customizable start and step values
- `cycle()` - Cycles through an iterable indefinitely
- `drop_while()` - Drops elements while predicate is true, then yields the rest
- `product()` - Computes the Cartesian product of two, three, or four iterables
- `repeat()` - Repeats an element a specified number of times
- `take_while()` - Yields elements while predicate is true

These utilities enable functional-style iteration patterns and composable iterator
operations.
"""

# ===-----------------------------------------------------------------------===#
# count
# ===-----------------------------------------------------------------------===#

from builtin.constrained import _constrained_conforms_to
from builtin.rebind import downcast
from builtin.variadics import Variadic


@fieldwise_init
struct _CountIterator(Iterable, Iterator, TrivialRegisterPassable):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Int
    var start: Int
    var step: Int

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __next__(mut self) raises StopIteration -> Int:
        var result = self.start
        self.start += self.step
        return result


@always_inline
fn count(start: Int = 0, step: Int = 1) -> _CountIterator:
    """Constructs an iterator that starts at the value `start` with a stride of
    `step`.

    Args:
        start: The start of the iterator.
        step: The stride of the iterator.

    Returns:
        The constructed iterator.
    """
    return {start, step}


# ===-----------------------------------------------------------------------===#
# product
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product2[IteratorTypeA: Iterator, IteratorTypeB: Copyable & Iterator](
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
    var _inner_a_elem: Optional[Self.IteratorTypeA.Element]
    var _initial_inner_b: Self.IteratorTypeB

    fn __init__(
        out self,
        var inner_a: Self.IteratorTypeA,
        var inner_b: Self.IteratorTypeB,
    ):
        self._inner_a = inner_a^
        self._inner_b = inner_b.copy()
        self._inner_a_elem = None
        self._initial_inner_b = inner_b^

    fn __copyinit__(out self, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeA,
            ParentConformsTo="Copyable",
        ]()
        self._inner_a = rebind_var[Self.IteratorTypeA](
            trait_downcast[Copyable](copy._inner_a).copy()
        )
        self._inner_b = copy._inner_b.copy()
        self._inner_a_elem = copy._inner_a_elem.copy()
        self._initial_inner_b = copy._initial_inner_b.copy()

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __next__(mut self) raises StopIteration -> Self.Element:
        # Take the first element from 'a' if we haven't got it yet.
        if not self._inner_a_elem:
            self._inner_a_elem = next(self._inner_a)

        _constrained_conforms_to[
            conforms_to(type_of(self._inner_a_elem).T, Copyable),
            Parent=Self,
            Element = type_of(self._inner_a_elem).T,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()

        try:
            # Get the next element from 'b' if it exists.
            var b_val = next(self._inner_b)

            var elem = trait_downcast[Copyable](
                self._inner_a_elem.unsafe_value()
            ).copy()
            return rebind_var[Self.IteratorTypeA.Element](elem^), b_val^
        except:
            # reset if we reach the end of the B iterator and grab the next
            # item from the A iterator.
            self._inner_b = self._initial_inner_b.copy()
            self._inner_a_elem = next(self._inner_a)
            var b_val = next(self._inner_b)
            # If a and b iterators had more elements, return this one.
            var elem = trait_downcast[Copyable](
                self._inner_a_elem.unsafe_value()
            ).copy()
            return rebind_var[Self.IteratorTypeA.Element](elem^), b_val^

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        # compute a * initial_b + b for lower and upper

        var a_bounds = self._inner_a.bounds()
        var b_bounds = self._inner_b.bounds() if self._inner_a_elem else (
            0,
            Optional[Int](0),
        )
        var initial_b_bounds = self._initial_inner_b.bounds()

        var lower_bound = a_bounds[0] * initial_b_bounds[0] + b_bounds[0]
        if not a_bounds[1] or not initial_b_bounds[1]:
            return (lower_bound, None)

        var upper_bound = a_bounds[1].unsafe_value() * initial_b_bounds[
            1
        ].unsafe_value() + b_bounds[1].or_else(0)
        return (lower_bound, upper_bound)


@always_inline
fn product[
    IterableTypeA: Iterable, IterableTypeB: Iterable
](ref iterable_a: IterableTypeA, ref iterable_b: IterableTypeB) -> _Product2[
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    downcast[type_of(iter(iterable_b)), Copyable & Iterator],
]:
    """Returns an iterator that yields tuples of the elements of the outer
    product of the iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.

    Returns:
        A product iterator that yields outer product tuples of elements from both
        iterables.

    Examples:

    ```mojo
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    for a, b in product(l, l2):
        print(a, b)
    ```
    """
    return {
        iter(iterable_a),
        rebind_var[downcast[type_of(iter(iterable_b)), Copyable & Iterator]](
            iter(iterable_b)
        ),
    }


# ===-----------------------------------------------------------------------===#
# product (3 iterables)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product3[
    IteratorTypeA: Iterator,
    IteratorTypeB: Copyable & Iterator,
    IteratorTypeC: Copyable & Iterator,
](Copyable, Iterable, Iterator):
    comptime Element = Tuple[
        Self.IteratorTypeA.Element,
        Self.IteratorTypeB.Element,
        Self.IteratorTypeC.Element,
    ]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime _Product2Type = _Product2[Self.IteratorTypeB, Self.IteratorTypeC]
    comptime _OuterProduct2Type = _Product2[
        Self.IteratorTypeA, Self._Product2Type
    ]

    var _inner: Self._OuterProduct2Type

    fn __init__(
        out self,
        var inner_a: Self.IteratorTypeA,
        var inner_b: Self.IteratorTypeB,
        var inner_c: Self.IteratorTypeC,
    ):
        var product2 = Self._Product2Type(inner_b^, inner_c^)
        self._inner = Self._OuterProduct2Type(inner_a^, product2^)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(_inner=self._inner.copy())

    fn __next__(mut self) raises StopIteration -> Self.Element:
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA.Element, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeA.Element,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB.Element, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeB.Element,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeC.Element, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeC.Element,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()

        var nested = next(self._inner)  # Returns (a, (b, c))
        var a = rebind_var[Self.IteratorTypeA.Element](
            trait_downcast[Copyable](nested[0]).copy()
        )
        var b = rebind_var[Self.IteratorTypeB.Element](
            trait_downcast[Copyable](nested[1][0]).copy()
        )
        var c = rebind_var[Self.IteratorTypeC.Element](
            trait_downcast[Copyable](nested[1][1]).copy()
        )
        # Flatten to (a, b, c)
        return (a^, b^, c^)

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
fn product[
    IterableTypeA: Iterable, IterableTypeB: Iterable, IterableTypeC: Iterable
](
    ref iterable_a: IterableTypeA,
    ref iterable_b: IterableTypeB,
    ref iterable_c: IterableTypeC,
) -> _Product3[
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    downcast[type_of(iter(iterable_b)), Copyable & Iterator],
    downcast[type_of(iter(iterable_c)), Copyable & Iterator],
]:
    """Returns an iterator that yields tuples of the elements of the outer
    product of three iterables.

    Parameters:
        IterableTypeA: The type of the first iterable.
        IterableTypeB: The type of the second iterable.
        IterableTypeC: The type of the third iterable.

    Args:
        iterable_a: The first iterable.
        iterable_b: The second iterable.
        iterable_c: The third iterable.

    Returns:
        A product iterator that yields outer product tuples of elements from all
        three iterables.

    Examples:

    ```mojo
    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    for a, b, c in product(l1, l2, l3):
        print(a, b, c)
    ```
    """
    return {
        iter(iterable_a),
        rebind_var[downcast[type_of(iter(iterable_b)), Copyable & Iterator]](
            iter(iterable_b)
        ),
        rebind_var[downcast[type_of(iter(iterable_c)), Copyable & Iterator]](
            iter(iterable_c)
        ),
    }


# ===-----------------------------------------------------------------------===#
# product (4 iterables)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product4[
    IteratorTypeA: Iterator,
    IteratorTypeB: Copyable & Iterator,
    IteratorTypeC: Copyable & Iterator,
    IteratorTypeD: Copyable & Iterator,
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

    comptime _Product3Type = _Product3[
        Self.IteratorTypeB, Self.IteratorTypeC, Self.IteratorTypeD
    ]
    comptime _Product2Type = _Product2[Self.IteratorTypeA, Self._Product3Type]

    var _inner: Self._Product2Type

    fn __init__(
        out self,
        var inner_a: Self.IteratorTypeA,
        var inner_b: Self.IteratorTypeB,
        var inner_c: Self.IteratorTypeC,
        var inner_d: Self.IteratorTypeD,
    ):
        var product3 = Self._Product3Type(inner_b^, inner_c^, inner_d^)
        self._inner = Self._Product2Type(inner_a^, product3^)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(_inner=self._inner.copy())

    fn __next__(mut self) raises StopIteration -> Self.Element:
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeA.Element, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeA.Element,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeB.Element, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeB.Element,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeC.Element, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeC.Element,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()
        _constrained_conforms_to[
            conforms_to(Self.IteratorTypeD.Element, Copyable),
            Parent=Self,
            Element = Self.IteratorTypeD.Element,
            ParentConformsTo="Iterator",
            ElementConformsTo="Copyable",
        ]()

        var nested = next(self._inner)  # Returns (a, (b, c, d))
        # Flatten to (a, b, c, d)

        var a = rebind_var[Self.IteratorTypeA.Element](
            trait_downcast[Copyable](nested[0]).copy()
        )
        var b = rebind_var[Self.IteratorTypeB.Element](
            trait_downcast[Copyable](nested[1][0]).copy()
        )
        var c = rebind_var[Self.IteratorTypeC.Element](
            trait_downcast[Copyable](nested[1][1]).copy()
        )
        var d = rebind_var[Self.IteratorTypeD.Element](
            trait_downcast[Copyable](nested[1][2]).copy()
        )
        return (a^, b^, c^, d^)

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@always_inline
fn product[
    IterableTypeA: Iterable,
    IterableTypeB: Iterable,
    IterableTypeC: Iterable,
    IterableTypeD: Iterable,
](
    ref iterable_a: IterableTypeA,
    ref iterable_b: IterableTypeB,
    ref iterable_c: IterableTypeC,
    ref iterable_d: IterableTypeD,
) -> _Product4[
    IterableTypeA.IteratorType[origin_of(iterable_a)],
    downcast[type_of(iter(iterable_b)), Copyable & Iterator],
    downcast[type_of(iter(iterable_c)), Copyable & Iterator],
    downcast[type_of(iter(iterable_d)), Copyable & Iterator],
]:
    """Returns an iterator that yields tuples of the elements of the outer
    product of four iterables.

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
        A product iterator that yields outer product tuples of elements from all
        four iterables.

    Examples:

    ```mojo
    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    var l4 = [7, 8]
    for a, b, c, d in product(l1, l2, l3, l4):
        print(a, b, c, d)
    ```
    """
    return {
        iter(iterable_a),
        rebind_var[downcast[type_of(iter(iterable_b)), Copyable & Iterator]](
            iter(iterable_b)
        ),
        rebind_var[downcast[type_of(iter(iterable_c)), Copyable & Iterator]](
            iter(iterable_c)
        ),
        rebind_var[downcast[type_of(iter(iterable_d)), Copyable & Iterator]](
            iter(iterable_d)
        ),
    }


# ===-----------------------------------------------------------------------===#
# cycle
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _CycleIterator[InnerIteratorType: Iterator & Copyable](
    Copyable, Iterable, Iterator
):
    """Iterator that cycles through an iterable indefinitely.

    This iterator keeps a copy of the original iterator and resets to it
    when the current iterator is exhausted. This is a lazy implementation
    that does no work at construction time.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _orig: Self.InnerIteratorType
    var _iter: Self.InnerIteratorType

    fn __init__(out self, var iterator: Self.InnerIteratorType):
        """Creates a cycle iterator from an iterator.

        Args:
            iterator: The iterator to cycle through.
        """
        self._orig = iterator.copy()
        self._iter = iterator^

    fn __copyinit__(out self, copy: Self):
        self._orig = copy._orig.copy()
        self._iter = copy._iter.copy()

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn copy(self) -> Self:
        var result = Self(self._orig.copy())
        result._iter = self._iter.copy()
        return result^

    @always_inline
    fn __next__(mut self) raises StopIteration -> Self.Element:
        try:
            return next(self._iter)
        except StopIteration:
            # Reset to original and try again
            self._iter = self._orig.copy()
            # If this also raises StopIteration, the original was empty
            return next(self._iter)


@always_inline
fn cycle[
    IterableType: Iterable
](ref iterable: IterableType) -> _CycleIterator[
    downcast[type_of(iter(iterable)), Copyable & Iterator]
]:
    """Creates an iterator that cycles through an iterable indefinitely.

    This function returns an iterator that yields elements from the input
    iterable repeatedly in an infinite loop. The elements are yielded in
    the same order as they appear in the original iterable.

    This is a lazy implementation - no work is done at construction time.
    The iterator keeps a copy of the original iterator and resets to it
    when exhausted.

    Parameters:
        IterableType: The type of the iterable.

    Args:
        iterable: The iterable to cycle through.

    Returns:
        An iterator that yields elements from the iterable forever.

    Examples:

    ```mojo
    # Cycle through a list
    var colors = ["red", "green", "blue"]
    var color_cycle = cycle(colors)

    # Get 6 elements (cycles twice)
    var count = 0
    for color in color_cycle:
        print(color)
        count += 1
        if count >= 6:
            break
    # Output: red, green, blue, red, green, blue
    ```
    """
    return _CycleIterator(
        rebind_var[downcast[type_of(iter(iterable)), Copyable & Iterator]](
            iter(iterable)
        )
    )


# ===-----------------------------------------------------------------------===#
# take_while
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _TakeWhileIterator[
    InnerIteratorType: Iterator,
    //,
    predicate: fn(InnerIteratorType.Element) -> Bool,
](Copyable, Iterable, Iterator):
    """Iterator that yields elements while predicate returns True.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
        predicate: A function that takes an element and returns True if the
            element should be yielded.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _inner: Self.InnerIteratorType
    var _exhausted: Bool

    fn __init__(out self, var inner: Self.InnerIteratorType):
        """Creates a take_while iterator from an inner iterator.

        Args:
            inner: The inner iterator to wrap.
        """
        self._inner = inner^
        self._exhausted = False

    fn __copyinit__(out self, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.InnerIteratorType, Copyable),
            Parent=Self,
            Element = Self.InnerIteratorType,
            ParentConformsTo="Copyable",
        ]()
        self._inner = rebind_var[Self.InnerIteratorType](
            trait_downcast[Copyable](copy._inner).copy()
        )
        self._exhausted = copy._exhausted

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn __next__(mut self) raises StopIteration -> Self.Element:
        if self._exhausted:
            raise StopIteration()
        var elem = next(self._inner)
        if not Self.predicate(elem):
            self._exhausted = True
            # Discard the element that failed the predicate
            _ = rebind_var[
                downcast[Self.Element, Movable & ImplicitlyDestructible]
            ](elem^)
            raise StopIteration()
        return elem^


@always_inline
fn take_while[
    origin: ImmutOrigin,
    IterableType: Iterable,
    //,
    predicate: fn(IterableType.IteratorType[origin].Element) -> Bool,
](ref[origin] iterable: IterableType) -> _TakeWhileIterator[
    InnerIteratorType = IterableType.IteratorType[origin],
    predicate=predicate,
] where conforms_to(
    IterableType.IteratorType[origin].Element,
    ImplicitlyDestructible,
):
    """Creates an iterator that yields elements while predicate returns True.

    This function returns an iterator that yields elements from the input
    iterable as long as the predicate function returns True for each element.
    Once the predicate returns False, the iterator stops immediately and does
    not yield any more elements.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.
        predicate: A function that takes an element and returns True if the
            element should be yielded.

    Args:
        iterable: The iterable to take elements from.

    Returns:
        An iterator that yields elements while predicate returns True.

    Examples:

    ```mojo
    from itertools import take_while

    # Take while less than 5
    fn less_than_5(x: Int) -> Bool:
        return x < 5

    var nums = [1, 2, 3, 4, 5, 6, 7]
    for num in take_while[less_than_5](nums):
        print(num)  # Prints: 1, 2, 3, 4
    ```
    """
    # FIXME(MOCO-3238): This rebind shouldn't ve needed, something isn't getting
    # substituted through associated types right.
    return _TakeWhileIterator[predicate=predicate](
        rebind_var[IterableType.IteratorType[origin]](iter(iterable))
    )


# ===-----------------------------------------------------------------------===#
# drop_while
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _DropWhileIterator[
    InnerIteratorType: Iterator,
    //,
    predicate: fn(InnerIteratorType.Element) -> Bool,
](Copyable, Iterable, Iterator):
    """Iterator that drops elements while predicate returns True, then yields rest.

    Parameters:
        InnerIteratorType: The type of the inner iterator.
        predicate: A function that takes an element and returns True if the
            element should be dropped.
    """

    comptime Element = Self.InnerIteratorType.Element
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _inner: Self.InnerIteratorType
    var _dropping: Bool

    fn __init__(out self, var inner: Self.InnerIteratorType):
        """Creates a drop_while iterator from an inner iterator.

        Args:
            inner: The inner iterator to wrap.
        """
        self._inner = inner^
        self._dropping = True

    fn __copyinit__(out self, copy: Self):
        _constrained_conforms_to[
            conforms_to(Self.InnerIteratorType, Copyable),
            Parent=Self,
            Element = Self.InnerIteratorType,
            ParentConformsTo="Copyable",
        ]()
        self._inner = rebind_var[Self.InnerIteratorType](
            trait_downcast[Copyable](copy._inner).copy()
        )
        self._dropping = copy._dropping

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn __next__(mut self) raises StopIteration -> Self.Element:
        if self._dropping:
            while True:
                var elem = next(self._inner)
                if Self.predicate(elem):
                    # Discard the element that matched the predicate
                    _ = rebind_var[
                        downcast[Self.Element, Movable & ImplicitlyDestructible]
                    ](elem^)
                    continue
                self._dropping = False
                return elem^
        return next(self._inner)


@always_inline
fn drop_while[
    origin: ImmutOrigin,
    IterableType: Iterable,
    //,
    predicate: fn(IterableType.IteratorType[origin].Element) -> Bool,
](ref[origin] iterable: IterableType) -> _DropWhileIterator[
    InnerIteratorType = IterableType.IteratorType[origin],
    predicate=predicate,
] where conforms_to(
    IterableType.IteratorType[origin].Element,
    ImplicitlyDestructible,
):
    """Creates an iterator that drops elements while predicate returns True.

    This function returns an iterator that drops elements from the input
    iterable as long as the predicate function returns True for each element.
    Once the predicate returns False, the iterator starts yielding all
    remaining elements unconditionally.

    Parameters:
        origin: The origin of the iterable.
        IterableType: The type of the iterable.
        predicate: A function that takes an element and returns True if the
            element should be dropped.

    Args:
        iterable: The iterable to drop elements from.

    Returns:
        An iterator that drops elements while predicate returns True, then
        yields all remaining elements.

    Examples:

    ```mojo
    from itertools import drop_while

    # Drop while less than 5
    fn less_than_5(x: Int) -> Bool:
        return x < 5

    var nums = [1, 2, 3, 4, 5, 6, 1, 2]
    for num in drop_while[less_than_5](nums):
        print(num)  # Prints: 5, 6, 1, 2
    ```
    """
    # FIXME(MOCO-3238): This rebind shouldn't ve needed, something isn't getting
    # substituted through associated types right.
    return _DropWhileIterator[predicate=predicate](
        rebind_var[IterableType.IteratorType[origin]](iter(iterable))
    )


# ===-----------------------------------------------------------------------===#
# repeat
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _RepeatIterator[ElementType: Copyable & ImplicitlyDestructible](
    Copyable, Iterable, Iterator
):
    """Iterator that repeats an element a specified number of times.

    Parameters:
        ElementType: The type of the element to repeat.
    """

    comptime Element = Self.ElementType
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var element: Self.ElementType
    var remaining: Int

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn copy(self) -> Self:
        return Self(self.element.copy(), self.remaining)

    @always_inline
    fn __next__(mut self) raises StopIteration -> Self.ElementType:
        if self.remaining <= 0:
            raise StopIteration()
        self.remaining -= 1
        return self.element.copy()


@always_inline
fn repeat[
    ElementType: Copyable & ImplicitlyDestructible
](element: ElementType, *, times: Int) -> _RepeatIterator[ElementType]:
    """Constructs an iterator that repeats the given element a specified number of times.

    This function creates an iterator that returns the same element over and over
    for the specified number of times.

    Parameters:
        ElementType: The type of the element to repeat.

    Args:
        element: The element to repeat.
        times: The number of times to repeat the element.

    Returns:
        An iterator that repeats the element the specified number of times.

    Examples:

    ```mojo
    # Repeat a value 3 times
    var it = repeat(42, times=3)
    for val in it:
        print(val)  # Prints: 42, 42, 42

    # Repeat a string 5 times
    var str_it = repeat("hello", times=5)
    for s in str_it:
        print(s)  # Prints: hello, hello, hello, hello, hello
    ```
    """
    debug_assert(times >= 0, "The `times` argument must be non-negative")
    return {element.copy(), times}
