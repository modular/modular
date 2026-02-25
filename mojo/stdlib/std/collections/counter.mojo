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
"""Defines the `Counter` type.

Import these APIs from the `collections` package:

```mojo
from collections import Counter

```

Counters provide convenient tallying objects that use a dictionary to
store keys and their counts. They offer the full functionality of
counted sets, also called bags or multisets, and extend that model by
supporting negative counts.

"""
from builtin.constrained import _constrained_conforms_to
from collections.dict import Dict, _DictEntryIter, _DictKeyIter, _DictValueIter
from hashlib import Hasher, default_hasher

from utils import Variant


@fieldwise_init
struct Counter[V: KeyElement, H: Hasher = default_hasher](
    Boolable,
    Copyable,
    Defaultable,
    Equatable,
    Iterable,
    Sized,
    Writable,
):
    """A container for counting hashable items.

    In other languages, similar types to counters include bags, counted sets,
    and multisets, although their semantics are normally closer to sets (adding,
    removing, intersecting, unions, etc) rather than increasing and decreasing
    counts. Mojo's `Counter` follows Python's model, and adds math versatility by
    supporting negative counts.

    The value type must implement the `KeyElement` trait, as its values are
    stored in a dictionary as keys.
    The keys' uniform value type are specified statically, unlike a Python
    `Counter`, which can accept arbitrary value types.
    They must be hashable for use in the underlying dictionary.

    Example:

    ```mojo
    from collections import Counter

    var counter = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
    print(counter["a"]) # prints 3
    print(counter["b"]) # prints 2
    ```

    Parameters:
        V: The value type to be counted. Currently must be `KeyElement`.
        H: The type of the hasher in the underlying dictionary.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[Self.V, Int, Self.H, iterable_origin]
    """The iterator type for this counter.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # Fields
    var _data: Dict[Self.V, Int, Self.H]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    fn __init__(out self):
        """Create a new, empty `Counter` object."""
        self._data = Dict[Self.V, Int, Self.H]()

    fn __init__(out self, var *values: Self.V):
        """Create a new `Counter` from a list of values.

        Args:
            values: A list of values to count.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
        print(counter["a"])  # print 3
        print(counter["b"])  # print 2
        ```

        Note:
        A counter is not limited to the values used in this initial list.
        You may add new keys as needed or remove them with clear or one
        of the `pop` calls.
        """
        self._data = Dict[Self.V, Int, Self.H]()
        for item in values:
            self._data[item.copy()] = self._data.get(item, 0) + 1

    fn __init__(out self, items: List[Self.V, ...]):
        """Create a `Counter` from an input iterable.

        Args:
            items: A list of items to count.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String](["a", "a", "a", "b", "b", "c", "d", "c", "c"])
        print(counter["a"]) # prints 3
        print(counter["b"]) # prints 2
        ```
        """
        self._data = Dict[Self.V, Int, Self.H]()
        for item in items:
            self._data[item.copy()] = self._data.get(item, 0) + 1

    @staticmethod
    fn fromkeys(keys: List[Self.V, ...], value: Int) -> Self:
        """Create a new `Counter` from a list of keys and a default value.

        Args:
            keys: The keys to create the `Counter` from.
            value: The default value to associate with each key. Must be non-negative.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter["a"]) # output: 1
        ```

        Returns:
            A new `Counter` with the count of each passed key set to `value`.
        """
        debug_assert(
            value >= 0,
            "value must be non-negative",
        )
        var result = Counter[Self.V, Self.H]()
        for key in keys:
            result[key] = value
        return result^

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    fn __getitem__(self, key: Self.V) -> Int:
        """Get the count of a key.

        Args:
            key: The key to get the count of.

        Returns:
            The count of the key.
        """
        return self.get(key, 0)

    fn __setitem__(mut self, value: Self.V, count: Int):
        """Set a value in the keyword `Counter` by key.

        Args:
            value: The value to associate with the specified count.
            count: The count to store in the `Counter`.
        """
        self._data[value.copy()] = count

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the `Counter`'s keys as immutable references.

        Returns:
            An iterator of immutable references to the `Counter` values.
        """
        return rebind[Self.IteratorType[origin_of(self)]](self._data.__iter__())

    fn __contains__(self, key: Self.V) -> Bool:
        """Check if a given key is in the `Counter` or not.

        Args:
            key: The key to check.

        Returns:
            `True` if there key exists in the `Counter`, `False` otherwise.
        """
        return key in self._data

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        """Returns the number of elements currently stored in the `Counter`.

        Returns:
            The number of elements in the `Counter`.
        """
        return len(self._data)

    fn __bool__(self) -> Bool:
        """Check if the `Counter` is empty or not.

        Returns:
            `False` if the `Counter` is empty, `True` otherwise.
        """
        return Bool(len(self))

    @deprecated("Representable is deprecated. Use Writable instead.")
    @no_inline
    fn __repr__(self) -> String:
        """Returns a string representation of a `Counter`.

        Returns:
            A string representation of the Counter.
        """
        return String.write(self)

    @deprecated("Stringable is deprecated. Use Writable instead.")
    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of a `Counter`.

        Returns:
            A string representation of the Counter.

        Example:

        ```mojo
        from collections import Counter


        var c = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
        var counter_as_string = String(c)
        print(counter_as_string)
        # prints "Counter({'a': 3, 'c': 3, 'b': 2, 'd': 1})"
        ```
        """
        var output = String()
        self.write_to(output)
        return output^

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Write `my_counter.__str__()` to a `Writer`.

        Constraints:
            `V` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        _constrained_conforms_to[
            conforms_to(Self.V, Writable),
            Parent=Self,
            Element = Self.V,
            ParentConformsTo="Writable",
            ElementConformsTo="Writable",
        ]()

        writer.write("Counter({")

        var items = self.most_common(UInt(len(self)))
        for i in range(len(items)):
            ref item = items[i]
            # Access the value and count from CountTuple
            ref value = item._value
            ref key = trait_downcast[Writable](value)
            var count = item._count
            writer.write(repr(key), ": ", repr(count))
            if i < len(items) - 1:
                writer.write(", ")
        writer.write("})")

    # ===------------------------------------------------------------------=== #
    # Comparison operators
    # ===------------------------------------------------------------------=== #

    fn __eq__(self, other: Self) -> Bool:
        """Check if all counts agree. Missing counts are treated as zero.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if the two `Counter`s are equal, `False` otherwise.
        """

        @parameter
        @always_inline
        fn is_eq(keys: _DictKeyIter[Self.V, Int, _]) -> Bool:
            for e in keys:
                if self.get(e, 0) != other.get(e, 0):
                    return False
            return True

        return is_eq(self.keys()) and is_eq(other.keys())

    fn le(self, other: Self) -> Bool:
        """Check if all counts are less than or equal to those in the other
        `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are less than or equal to the other `Counter`,
            `False` otherwise.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(counter.le(other)) # output: True
        counter[3] += 20
        print(counter.le(other)) # output: False
        ```
        """

        @parameter
        @always_inline
        fn is_le(keys: _DictKeyIter[Self.V, Int, _]) -> Bool:
            for e in keys:
                if self.get(e, 0) > other.get(e, 0):
                    return False
            return True

        return is_le(self.keys())

    fn lt(self, other: Self) -> Bool:
        """Check if all counts are less than those in the other `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are less than in the other `Counter`, `False`
            otherwise.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 3)
        print(counter.lt(other)) # output: True
        counter[1] += 1
        print(counter.lt(other)) # output: False
        ```
        """

        @parameter
        @always_inline
        fn is_lt(keys: _DictKeyIter[Self.V, Int, _]) -> Bool:
            for e in keys:
                if self.get(e, 0) >= other.get(e, 0):
                    return False
            return True

        return is_lt(self.keys())

    fn gt(self, other: Self) -> Bool:
        """Check if all counts are greater than those in the other `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are greater than in the other `Counter`,
            `False` otherwise.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 3)
        print(other.gt(counter)) # output: True
        counter[1] += 1
        print(other.gt(counter)) # output: False
        ```
        """
        return other.lt(self)

    fn ge(self, other: Self) -> Bool:
        """Check if all counts are greater than or equal to those in the other
        `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are greater than or equal to the other
            `Counter`, `False` otherwise.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(other.ge(counter)) # output: True
        counter[3] += 20
        print(other.ge(counter)) # output: False
        ```
        """
        return other.le(self)

    # ===------------------------------------------------------------------=== #
    # Binary operators
    # ===------------------------------------------------------------------=== #

    fn __add__(self, other: Self) -> Self:
        """Add counts from two `Counter`s.

        Args:
            other: The other `Counter` to add to this `Counter`.

        Returns:
            A new `Counter` with the counts from both `Counter`s added together.
        """
        var result = Counter[Self.V, Self.H]()

        result.update(self)
        result.update(other)

        return +result^  # Remove zero and negative counts

    fn __iadd__(mut self, other: Self):
        """Add counts from another `Counter` to this `Counter`.

        Args:
            other: The other `Counter` to add to this `Counter`.
        """
        self.update(other)
        self._keep_positive()

    fn __sub__(self, other: Self) -> Self:
        """Subtract counts, but keep only results with positive counts.

        Args:
            other: The other `Counter` to subtract from this `Counter`.

        Returns:
            A new `Counter` with the counts from the other `Counter` subtracted
            from this `Counter`.
        """
        var result = self.copy()

        result.subtract(other)

        return +result^  # Remove zero and negative counts

    fn __isub__(mut self, other: Self):
        """Subtract counts from another `Counter` from this `Counter`, but kee
        only results with positive counts.

        Args:
            other: The other `Counter` to subtract from this `Counter`.
        """
        self.subtract(other)
        self._keep_positive()

    fn __and__(self, other: Self) -> Self:
        """Intersection: keep common elements with the minimum count.

        Args:
            other: The other `Counter` to intersect with.

        Returns:
            A new `Counter` with the common elements and the minimum count of
            the two `Counter`s.
        """
        var result = Counter[Self.V, Self.H]()

        for key in self.keys():
            if key in other:
                result[key] = min(self.get(key, 0), other.get(key, 0))

        return result^

    fn __iand__(mut self, other: Self):
        """Intersection: keep common elements with the minimum count.

        Args:
            other: The other `Counter` to intersect with.
        """
        for key in self.keys():
            if key not in other:
                try:
                    var key_copy = key.copy()  # Copy due to incorrect origins.
                    _ = self.pop(key_copy)
                except:
                    pass  # this should not happen
            else:
                var key_copy = key.copy()  # Copy due to incorrect origins.
                self[key_copy] = min(self.get(key, 0), other.get(key, 0))

    fn __or__(self, other: Self) -> Self:
        """Union: keep all elements with the maximum count.

        Args:
            other: The other `Counter` to union with.

        Returns:
            A new `Counter` with all elements and the maximum count of the two
            `Counter`s.
        """
        var result = Counter[Self.V, Self.H]()

        for key in self.keys():
            var newcount = max(self.get(key, 0), other.get(key, 0))
            if newcount > 0:
                result[key] = newcount

        for key in other.keys():
            if key not in self and other.get(key, 0) > 0:
                result[key] = other.get(key, 0)

        return result^

    fn __ior__(mut self, other: Self):
        """Union: keep all elements with the maximum count.

        Args:
            other: The other `Counter` to union with.
        """
        for key in other.keys():
            var newcount = max(self.get(key, 0), other.get(key, 0))
            if newcount > 0:
                self[key] = newcount

    fn _keep_positive(mut self):
        """Remove zero and negative counts from the `Counter`."""
        for key in self.keys():
            if self.get(key, 0) <= 0:
                try:
                    var key_copy = key.copy()  # Copy due to incorrect origins.
                    _ = self.pop(key_copy)
                except:
                    pass  # this should not happen

    # ===------------------------------------------------------------------=== #
    # Unary operators
    # ===------------------------------------------------------------------=== #

    fn __pos__(self) -> Self:
        """Return a shallow copy of the `Counter`, stripping non-positive
        counts.

        Returns:
            A shallow copy of the `Counter`.
        """
        var result = Counter[Self.V, Self.H]()
        for item in self.items():
            if item.value > 0:
                result[item.key] = item.value
        return result^

    fn __neg__(self) -> Self:
        """Subtract from an empty `Counter`. Strips positive and zero counts,
        and flips the sign on negative counts.

        Returns:
            A new `Counter` with stripped counts and negative counts.
        """
        var result = Counter[Self.V, Self.H]()
        for item in self.items():
            if item.value < 0:
                result[item.key] = -item.value
        return result^

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    fn get(self, value: Self.V) -> Optional[Int]:
        """Get a value from the `Counter`.

        Args:
            value: The value to search for in the `Counter`.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty `Optional`.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter.get("a").or_else(0)) # output: 1
        print(counter.get("d").or_else(0)) # output: 0
        ```
        """
        return self._data.get(value)

    fn get(self, value: Self.V, default: Int) -> Int:
        """Get a value from the `Counter`.

        Args:
            value: The value to search for in the `Counter`.
            default: Default count to return.

        Returns:
            A copy of the value if it was present, otherwise default.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter.get("a", default=0)) # output: 1
        print(counter.get("d", default=0)) # output: 0
        ```
        """
        return self._data.get(value, default)

    fn pop(mut self, value: Self.V) raises -> Int:
        """Remove a value from the `Counter` by value.

        Args:
            value: The value to remove from the `Counter`.

        Returns:
            The value associated with the key, if it was in the `Counter`.

        Raises:
            "KeyError" if the key was not present in the `Counter`.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter.get("b").or_else(0)) # output: 1
        try:
            var count = counter.pop("b")
            print(count) # output: 1
            print(counter.get("b").or_else(0)) # output: 0
        except e:
            print(e) # KeyError if the key was not in the counter
        ```
        """
        return self._data.pop(value)

    fn pop(mut self, value: Self.V, var default: Int) -> Int:
        """Remove a value from the `Counter` by value.

        Args:
            value: The value to remove from the `Counter`.
            default: Optionally provide a default value to return if the value
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the `Counter`.
            If it wasn't, return the provided default value instead.

        Example:

        ```mojo
        from collections import Counter


        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        var count = counter.pop("b", default=100)
        print(count) # output: 1
        count = counter.pop("not-a-key", default=0)
        print(count) # output 0
        ```
        """
        return self._data.pop(value, default)

    fn keys(
        ref self,
    ) -> _DictKeyIter[Self.V, Int, Self.H, origin_of(self._data)]:
        """Iterate over the `Counter`'s keys as immutable references.

        Returns:
            An iterator of immutable references to the `Counter` keys.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String].fromkeys(["d", "b", "a", "c"], 1)
        var key_list = List[String]()
        for key in counter.keys():
            key_list.append(key)
        sort(key_list[:])
        print(key_list) # output: ['a', 'b', 'c', 'd']
        ```
        """
        return self._data.keys()

    fn values(
        ref self,
    ) -> _DictValueIter[Self.V, Int, Self.H, origin_of(self._data)]:
        """Iterate over the `Counter`'s values as references.

        Returns:
            An iterator of references to the `Counter` values.

        Example:

        ```mojo
        from collections import Counter

        # Construct `counter`
        var counter = Counter[Int]([1, 2, 3, 1, 2, 1, 1, 1, 2, 5, 2, 9])

        # Find most populous key
        var max_count: Int = Int.MIN
        for count in counter.values():
            if count > max_count:
                max_count = count

        # Max count is the five ones
        print(max_count) # output: 5
        ```
        """
        return self._data.values()

    fn items(
        self,
    ) -> _DictEntryIter[Self.V, Int, Self.H, origin_of(self._data)]:
        """Iterate over the `Counter`'s entries as immutable references.

        Returns:
            An iterator of immutable references to the `Counter` entries.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
        for count in counter.items():
            print(count.key, count.value)
        # output: 1 5
        # output: 2 4
        ```
        """
        return self._data.items()

    fn clear(mut self):
        """Remove all elements from the `Counter`.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
        print(counter.total()) # output: 9 (5 ones + 4 twos)
        counter.clear() # Removes both entries
        print(counter.total()) # output: 0
        ```
        """
        self._data.clear()

    fn popitem(mut self) raises -> CountTuple[Self.V]:
        """Remove and return an arbitrary (key, value) pair from the `Counter`.
        Useful for destructively iterating over the `Counter`.
        Returns in LIFO order.

        Returns:
            A `CountTuple` containing the key and value of the removed item.

        Raises:
            "KeyError" if the `Counter` is empty.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 5)
        try:
            var tuple = counter.popitem()
            print(tuple._value, tuple._count)
            # output: probably c 5 since that was last in
        except e:
            print(e) # KeyError if the key was not in the counter
        ```
        """
        var item_ref = self._data.popitem()
        return CountTuple[Self.V](item_ref.key, UInt(item_ref.value))

    # Special methods for counter

    fn total(self) -> UInt:
        """Return the total of all counts in the `Counter`.

        Returns:
            The total of all counts in the `Counter`.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
        print(counter.total()) # output: 9 (5 ones + 4 twos)
        counter.clear() # Removes both entries
        print(counter.total()) # output: 0
        ```
        """
        var total = 0
        for count in self.values():
            total += count
        return UInt(total)

    fn most_common(self, n: UInt) -> List[CountTuple[Self.V]]:
        """Return a list of the `n` most common elements and their counts from
        the most common to the least.

        Args:
            n: The number of most common elements to return.

        Returns:
            A list of the `n` most common elements and their counts.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
        for tuple in counter.most_common(2):
            print(tuple._value, tuple._count)
            # output: 1 5
            # output: 2 4
        ```
        """
        var items: List[CountTuple[Self.V]] = List[CountTuple[Self.V]]()
        for item in self._data.items():
            var t = CountTuple[Self.V](item.key, UInt(item.value))
            items.append(t^)

        @parameter
        fn comparator(a: CountTuple[Self.V], b: CountTuple[Self.V]) -> Bool:
            return a < b

        sort[comparator](items)
        items.shrink(Int(n))
        return items^

    fn elements(self) -> List[Self.V]:
        """Return an iterator over elements repeating each as many times as its
        count.

        Returns:
            An iterator over the elements in the `Counter`.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
        print(counter.elements())
        # output: [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 6, 6, 7]
        ```
        """
        var elements: List[Self.V] = List[Self.V]()
        for item in self._data.items():
            for _ in range(item.value):
                elements.append(item.key.copy())
        return elements^

    fn update(mut self, other: Self):
        """Update the `Counter`, like `Dict.update()` but add counts instead of
        replacing them.

        Args:
            other: The `Counter` to update this `Counter` with.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(counter[1]) # output: 2
        counter.update(other)
        print(counter[1]) # output: 12
        ```
        """
        for item in other.items():
            self._data[item.key.copy()] = (
                self._data.get(item.key, 0) + item.value
            )

    fn subtract(mut self, other: Self):
        """Subtract counts. Both inputs and outputs may be zero or negative.

        Args:
            other: The `Counter` to subtract from this `Counter`.

        Example:

        ```mojo
        from collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(counter[1]) # output: 2
        counter.subtract(other)
        print(counter[1]) # output: -8
        ```
        """
        for item in other.items():
            self[item.key] = self.get(item.key, 0) - item.value


struct CountTuple[V: KeyElement](Comparable, Copyable):
    """A tuple representing a value and its count in a `Counter`.

    Parameters:
        V: The value in the `Counter`.
    """

    # Fields
    var _value: Self.V
    """ The value in the `Counter`."""
    var _count: Int
    """ The count of the value in the `Counter`."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    fn __init__(out self, value: Self.V, count: UInt):
        """Create a new `CountTuple`.

        Args:
            value: The value in the `Counter`.
            count: The count of the value in the `Counter`.
        """
        self._value = value.copy()
        self._count = Int(count)

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    fn __lt__(self, other: Self) -> Bool:
        """Compare two `CountTuple`s by count, then by value.

        Args:
            other: The other `CountTuple` to compare to.

        Returns:
            `True` if this `CountTuple` is less than the other, `False`
            otherwise.
        """
        return self._count > other._count

    fn __eq__(self, other: Self) -> Bool:
        """Compare two `CountTuple`s for equality.

        Args:
            other: The other `CountTuple` to compare to.

        Returns:
            `True` if the two `CountTuple`s are equal, `False` otherwise.
        """
        return self._count == other._count

    @always_inline
    fn __getitem__(self, idx: Int) -> Variant[Self.V, Int]:
        """Get an element in the `CountTuple`.

        Args:
            idx: The element to return.

        Returns:
            The value if `idx` is `0` and the count if `idx` is `1`.
        """
        debug_assert(0 <= idx <= 1, "index must be within bounds")
        if idx == 0:
            return self._value.copy()
        else:
            return self._count
