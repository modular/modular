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
    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
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

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
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
        var a_bounds = self._inner_a.bounds()
        var b_bounds = self._inner_b.bounds()

        var lower_bound = a_bounds[0] * b_bounds[0]
        var upper_bound = Optional[Int](None)
        if a_bounds[1] and b_bounds[1]:
            upper_bound = (
                a_bounds[1].unsafe_value() * b_bounds[1].unsafe_value()
            )
        return (lower_bound, upper_bound)


@always_inline
fn product[
    IterableTypeA: Iterable, IterableTypeB: Iterable
](ref iterable_a: IterableTypeA, ref iterable_b: IterableTypeB) -> _Product2[
    IterableTypeA.IteratorType[__origin_of(iterable_a)],
    IterableTypeB.IteratorType[__origin_of(iterable_b)],
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
