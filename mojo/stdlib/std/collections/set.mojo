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
"""Implements the  Set datatype."""

from format._utils import (
    write_sequence_to,
    FormatStruct,
    Named,
    TypeNames,
    constrained_conforms_to_writable,
)
from hashlib import Hasher, default_hasher

from .dict import Dict, KeyElement, _DictEntryIter, _DictKeyIter
from builtin.constrained import _constrained_conforms_to


struct Set[T: KeyElement, H: Hasher = default_hasher](
    Boolable,
    Comparable,
    Copyable,
    Hashable,
    Iterable,
    KeyElement,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """A set data type.

    O(1) average-case amortized add, remove, and membership check.

    ```mojo
    from collections import Set

    var set = { 1, 2, 3 }
    print(len(set))  # 3
    set.add(4)

    for element in set:
        print(element)

    set -= Set[Int](3, 4, 5)
    print(set == Set[Int](1, 2))  # True
    print(set | Set[Int](0, 1) == Set[Int](0, 1, 2))  # True
    var element = set.pop()
    print(len(set))  # 1
    ```

    Parameters:
        T: The element type of the set. Must implement KeyElement.
        H: The type of the hasher used to hash keys.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[Self.T, NoneType, Self.H, iterable_origin]
    """The iterator type for this set.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # Fields
    var _data: Dict[Self.T, NoneType, Self.H]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, *ts: Self.T, __set_literal__: () = ()):
        """Construct a set from initial elements.

        Args:
            ts: Variadic of elements to add to the set.
            __set_literal__: Tell Mojo to use this method for set literals.
        """
        # TODO: Reserve space in this set. Also, take the elements as 'owned'
        # and transfer them into the set to eliminate copyability.
        self._data = Dict[Self.T, NoneType, Self.H]()
        for t in ts:
            self.add(t)

    # TODO: Should take the list owned so we can transfer the elements out.
    fn __init__(out self, elements: List[Self.T, ...]):
        """Construct a set from a List of elements.

        Args:
            elements: A vector of elements to add to the set.
        """
        self = Self()
        for e in elements:
            self.add(e)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __contains__(self, t: Self.T) -> Bool:
        """Whether or not the set contains an element.

        Args:
            t: The element to check membership in the set.

        Returns:
            Whether or not the set contains the element.
        """
        return t in self._data

    fn __eq__(self, other: Self) -> Bool:
        """Set equality.

        Args:
            other: Another Set instance to check equality against.

        Returns:
            True if the sets contain the same elements and False otherwise.
        """
        if len(self) != len(other):
            return False
        # Iterate over dict entries directly to reuse cached hash values,
        # avoiding redundant hash recomputation for each lookup in `other`.
        for entry in self._data.items():
            if not other._data._find_slot(entry.hash, entry.key)[0]:
                return False
        return True

    fn __and__(self, other: Self) -> Self:
        """The set intersection operator.

        Args:
            other: Another Set instance to intersect with this one.

        Returns:
            A new set containing only the elements which appear in both
            this set and the `other` set.
        """
        return self.intersection(other)

    fn __iand__(mut self, other: Self):
        """In-place set intersection.

        Updates the set to contain only the elements which are already in
        the set and are also contained in the `other` set.

        Args:
            other: Another Set instance to intersect with this one.
        """
        self.intersection_update(other)

    fn __or__(self, other: Self) -> Self:
        """The set union operator.

        Args:
            other: Another Set instance to union with this one.

        Returns:
            A new set containing any elements which appear in either
            this set or the `other` set.
        """
        return self.union(other)

    fn __ior__(mut self, other: Self):
        """In-place set union.

        Updates the set to contain all elements in the `other` set
        as well as keeping all elements it already contained.

        Args:
            other: Another Set instance to union with this one.
        """
        self.update(other)

    fn __sub__(self, other: Self) -> Self:
        """Set subtraction.

        Args:
            other: Another Set instance to subtract from this one.

        Returns:
            A new set containing elements of this set, but not containing
            any elements which were in the `other` set.
        """
        return self.difference(other)

    fn __isub__(mut self, other: Self):
        """In-place set subtraction.

        Updates the set to remove any elements from the `other` set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        self.difference_update(other)

    fn __le__(self, other: Self) -> Bool:
        """Overloads the <= operator for sets. Works like as `issubset` method.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a subset of the `other` set, False otherwise.
        """
        return self.issubset(other)

    fn __ge__(self, other: Self) -> Bool:
        """Overloads the >= operator for sets. Works like as `issuperset` method.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a superset of the `other` set, False otherwise.
        """
        return self.issuperset(other)

    fn __gt__(self, other: Self) -> Bool:
        """Overloads the > operator for strict superset comparison of sets.

        Args:
            other: The set to compare against for the strict superset relationship.

        Returns:
            True if the set is a strict superset of the `other` set, False otherwise.
        """
        return len(self) > len(other) and other.issubset(self)

    fn __lt__(self, other: Self) -> Bool:
        """Overloads the < operator for strict subset comparison of sets.

        Args:
            other: The set to compare against for the strict subset relationship.

        Returns:
            True if the set is a strict subset of the `other` set, False otherwise.
        """
        return len(self) < len(other) and self.issubset(other)

    fn __xor__(self, other: Self) -> Self:
        """Overloads the ^ operator for sets. Works like as `symmetric_difference` method.

        Args:
            other: The set to find the symmetric difference with.

        Returns:
            A new set containing the symmetric difference of the two sets.
        """
        return self.symmetric_difference(other)

    fn __ixor__(mut self, other: Self):
        """Overloads the ^= operator. Works like as `symmetric_difference_update` method.

        Updates the set with the symmetric difference of itself and another set.

        Args:
            other: The set to find the symmetric difference with.
        """
        self.symmetric_difference_update(other)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __bool__(self) -> Bool:
        """Whether the set is non-empty or not.

        Returns:
            True if the set is non-empty, False if it is empty.
        """
        return len(self).__bool__()

    fn __len__(self) -> Int:
        """The size of the set.

        Returns:
            The number of elements in the set.
        """
        return len(self._data)

    fn __hash__[_H: Hasher](self, mut hasher: _H):
        """Updates hasher with the underlying values.

        The update is order independent, so s1 == s2 -> hash(s1) == hash(s2).

        Parameters:
            _H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        var hash_value: UInt64 = 0
        # Hash combination needs to be commutative so iteration order
        # doesn't impact the hash value.
        for e in self:
            hash_value ^= hash(e)
        hasher.update(hash_value)

    @no_inline
    fn __str__(self) -> String:
        """Returns the string representation of the set.

        Returns:
            The string representation of the set.
        """
        var output = String()
        self.write_to(output)
        return output

    @no_inline
    fn __repr__(self) -> String:
        """Returns the string representation of the set.

        Returns:
            The string representation of the set.
        """
        var output = String()
        self.write_repr_to(output)
        return output

    fn _write_self_to[*, is_repr: Bool](self, mut writer: Some[Writer]):
        constrained_conforms_to_writable[Self.T, Parent=Self]()

        var iterator = self.__iter__()

        @parameter
        fn iterate(mut w: Some[Writer]) raises StopIteration:
            ref element = iterator.__next__()

            comptime if is_repr:
                trait_downcast[Writable](element).write_repr_to(w)
            else:
                trait_downcast[Writable](element).write_to(w)

        write_sequence_to[ElementFn=iterate](writer, start="{", end="}")
        _ = iterator^

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Write this set to a `Writer`.

        Constraints:
            `T` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[is_repr=False](writer)

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write this set to a `Writer`.

        Constraints:
            `T` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        @parameter
        fn write_fields(mut w: Some[Writer]):
            self._write_self_to[is_repr=True](w)

        FormatStruct(writer, "Set").params(
            TypeNames[Self.T](),
            Named("Hasher", TypeNames[Self.H]()),
        ).fields[FieldsFn=write_fields]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the set, returning immutable references.

        Returns:
            An iterator of immutable references to the set elements.
        """
        # here we rely on Set being a trivial wrapper of a Dict
        return rebind[Self.IteratorType[origin_of(self)]](
            _DictKeyIter(_DictEntryIter(0, 0, self._data))
        )

    fn add(mut self, t: Self.T):
        """Add an element to the set.

        Args:
            t: The element to add to the set.
        """
        self._data[t.copy()] = None

    fn remove(mut self, t: Self.T) raises:
        """Remove an element from the set.

        Args:
            t: The element to remove from the set.

        Raises:
            If the element isn't in the set to remove.
        """
        self._data.pop(t)

    fn pop(mut self) raises -> Self.T:
        """Remove any one item from the set, and return it.

        As an implementation detail this will remove the first item
        according to insertion order. This is practically useful
        for breadth-first search implementations.

        Returns:
            The element which was removed from the set.

        Raises:
            If the set is empty.
        """
        if not self:
            raise "Pop on empty set"
        var iter = self.__iter__()
        var first = iter.__next__().copy()
        self.remove(first)
        return first^

    fn union(self, other: Self) -> Self:
        """Set union.

        Args:
            other: Another Set instance to union with this one.

        Returns:
            A new set containing any elements which appear in either
            this set or the `other` set.
        """
        var result = self.copy()
        for o in other:
            result.add(o)

        return result^

    fn intersection(self, other: Self) -> Self:
        """Set intersection.

        Args:
            other: Another Set instance to intersect with this one.

        Returns:
            A new set containing only the elements which appear in both
            this set and the `other` set.
        """
        var result = Set[Self.T, Self.H]()
        for v in self:
            if v in other:
                result.add(v)

        return result^

    fn difference(self, other: Self) -> Self:
        """Set difference.

        Args:
            other: Another Set instance to find the difference with this one.

        Returns:
            A new set containing elements that are in this set but not in
            the `other` set.
        """
        var result = Set[Self.T, Self.H]()
        for e in self:
            if e not in other:
                result.add(e)
        return result^

    fn update(mut self, other: Self):
        """In-place set update.

        Updates the set to contain all elements in the `other` set
        as well as keeping all elements it already contained.

        Args:
            other: Another Set instance to union with this one.
        """
        for e in other:
            self.add(e)

    fn intersection_update(mut self, other: Self):
        """In-place set intersection update.

        Updates the set by retaining only elements found in both this set and the `other` set,
        removing all other elements. The result is the intersection of this set with `other`.

        Args:
            other: Another Set instance to intersect with this one.
        """
        # Possible to do this without an extra allocation, but need to be
        # careful about concurrent iteration + mutation
        self.difference_update(self - other)

    fn difference_update(mut self, other: Self):
        """In-place set subtraction.

        Updates the set by removing all elements found in the `other` set,
        effectively keeping only elements that are unique to this set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        for o in other:
            try:
                self.remove(o)
            except:
                pass

    fn issubset(self, other: Self) -> Bool:
        """Check if this set is a subset of another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a subset of the `other` set, False otherwise.
        """
        if len(self) > len(other):
            return False

        for element in self:
            if element not in other:
                return False

        return True

    fn isdisjoint(self, other: Self) -> Bool:
        """Check if this set is disjoint with another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is disjoint with the `other` set, False otherwise.
        """
        for element in self:
            if element in other:
                return False

        return True

    fn issuperset(self, other: Self) -> Bool:
        """Check if this set is a superset of another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a superset of the `other` set, False otherwise.
        """
        if len(self) < len(other):
            return False

        for element in other:
            if element not in self:
                return False

        return True

    fn symmetric_difference(self, other: Self) -> Self:
        """Returns the symmetric difference of two sets.

        Args:
            other: The set to find the symmetric difference with.

        Returns:
            A new set containing the symmetric difference of the two sets.
        """
        var result = Set[Self.T, Self.H]()

        for element in self:
            if element not in other:
                result.add(element)

        for element in other:
            if element not in self:
                result.add(element)

        return result^

    fn symmetric_difference_update(mut self, other: Self):
        """Updates the set with the symmetric difference of itself and another set.

        Args:
            other: The set to find the symmetric difference with.
        """
        self = self.symmetric_difference(other)

    fn discard(mut self, value: Self.T):
        """Remove a value from the set if it exists. Pass otherwise.

        Args:
            value: The element to remove from the set.
        """
        try:
            self._data.pop(value)
        except:
            pass

    fn clear(mut self):
        """Removes all elements from the set.

        This method modifies the set in-place, removing all of its elements.
        After calling this method, the set will be empty.
        """
        self._data.clear()
