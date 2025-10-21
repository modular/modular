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
# ===-----------------------------------------------------------------------===#
# count
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct _CountIterator(Iterable, Iterator):
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self
    alias Element = Int
    var start: Int
    var step: Int

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn __next__(mut self) -> Int:
        var result = self.start
        self.start += self.step
        return result

    @always_inline
    fn __has_next__(self) -> Bool:
        return True


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
struct _Product2[IteratorTypeA: Iterator, IteratorTypeB: Iterator](
    Copyable, Iterable, Iterator, Movable
):
    alias Element = Tuple[IteratorTypeA.Element, IteratorTypeB.Element]
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    var _inner_a: IteratorTypeA
    var _inner_b: IteratorTypeB
    var _inner_a_elem: Optional[IteratorTypeA.Element]
    var _initial_inner_b: IteratorTypeB

    fn __init__(out self, inner_a: IteratorTypeA, inner_b: IteratorTypeB):
        self._inner_a = inner_a.copy()
        self._inner_b = inner_b.copy()
        self._inner_a_elem = None
        self._initial_inner_b = inner_b.copy()

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(
            self._inner_a.copy(),
            self._inner_b.copy(),
            self._inner_a_elem.copy(),
            self._initial_inner_b.copy(),
        )

    fn __has_next__(self) -> Bool:
        if not self._inner_a_elem:
            return self._inner_a.__has_next__()
        if self._inner_b.__has_next__():
            return True
        # If _inner_b is exhausted, but _inner_a has more elements, we can reset _inner_b and continue
        return self._inner_a.__has_next__()

    fn __next__(mut self) -> Self.Element:
        if not self._inner_a_elem:
            self._inner_a_elem = next(self._inner_a)
        if not self._inner_b.__has_next__():
            # reset if we reach the end of the B iterator and grab the next
            # item from the A iterator.
            self._inner_b = self._initial_inner_b.copy()
            self._inner_a_elem = next(self._inner_a)
        return self._inner_a_elem.unsafe_value().copy(), next(self._inner_b)

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
    IterableTypeB.IteratorType[origin_of(iterable_b)],
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
    return {iter(iterable_a), iter(iterable_b)}


# ===-----------------------------------------------------------------------===#
# product (3 iterables)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product3[
    IteratorTypeA: Iterator, IteratorTypeB: Iterator, IteratorTypeC: Iterator
](Copyable, Iterable, Iterator, Movable):
    alias Element = Tuple[
        IteratorTypeA.Element, IteratorTypeB.Element, IteratorTypeC.Element
    ]
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    alias _Product2Type = _Product2[IteratorTypeB, IteratorTypeC]
    alias _OuterProduct2Type = _Product2[IteratorTypeA, Self._Product2Type]

    var _inner: Self._OuterProduct2Type

    fn __init__(
        out self,
        inner_a: IteratorTypeA,
        inner_b: IteratorTypeB,
        inner_c: IteratorTypeC,
    ):
        var product2 = Self._Product2Type(inner_b, inner_c)
        self._inner = Self._OuterProduct2Type(inner_a, product2)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(_inner=self._inner.copy())

    fn __has_next__(self) -> Bool:
        return self._inner.__has_next__()

    fn __next__(mut self) -> Self.Element:
        var nested = next(self._inner)  # Returns (a, (b, c))
        # Flatten to (a, b, c)
        return (nested[0].copy(), nested[1][0].copy(), nested[1][1].copy())

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
    IterableTypeB.IteratorType[origin_of(iterable_b)],
    IterableTypeC.IteratorType[origin_of(iterable_c)],
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
    return {iter(iterable_a), iter(iterable_b), iter(iterable_c)}


# ===-----------------------------------------------------------------------===#
# product (4 iterables)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _Product4[
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

    alias _Product3Type = _Product3[IteratorTypeB, IteratorTypeC, IteratorTypeD]
    alias _Product2Type = _Product2[IteratorTypeA, Self._Product3Type]

    var _inner: Self._Product2Type

    fn __init__(
        out self,
        inner_a: IteratorTypeA,
        inner_b: IteratorTypeB,
        inner_c: IteratorTypeC,
        inner_d: IteratorTypeD,
    ):
        var product3 = Self._Product3Type(inner_b, inner_c, inner_d)
        self._inner = Self._Product2Type(inner_a, product3)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn copy(self) -> Self:
        return Self(_inner=self._inner.copy())

    fn __has_next__(self) -> Bool:
        return self._inner.__has_next__()

    fn __next__(mut self) -> Self.Element:
        var nested = next(self._inner)  # Returns (a, (b, c, d))
        # Flatten to (a, b, c, d)
        return (
            nested[0].copy(),
            nested[1][0].copy(),
            nested[1][1].copy(),
            nested[1][2].copy(),
        )

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
    IterableTypeB.IteratorType[origin_of(iterable_b)],
    IterableTypeC.IteratorType[origin_of(iterable_c)],
    IterableTypeD.IteratorType[origin_of(iterable_d)],
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
        iter(iterable_b),
        iter(iterable_c),
        iter(iterable_d),
    }


# ===-----------------------------------------------------------------------===#
# repeat
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _RepeatIterator[ElementType: Copyable & Movable](
    Copyable, Iterable, Iterator, Movable
):
    """Iterator that repeats an element a specified number of times.

    Parameters:
        ElementType: The type of the element to repeat.
    """

    alias Element = ElementType
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    var element: ElementType
    var remaining: Int

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    fn copy(self) -> Self:
        return Self(self.element.copy(), self.remaining)

    @always_inline
    fn __next__(mut self) -> ElementType:
        self.remaining -= 1
        return self.element.copy()

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.remaining > 0


@always_inline
fn repeat[
    ElementType: Copyable & Movable
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
