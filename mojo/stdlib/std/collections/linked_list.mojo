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
"""Implements a doubly-linked list data structure.

This module provides the `LinkedList` type, a doubly-linked list where each
element points to both the next and previous elements. This structure enables
efficient insertion and deletion at any position, though random access requires
traversal. The implementation includes iterator support for forward and reverse
traversal.
"""

from collections._index_normalization import normalize_index
import format._utils as fmt
from os import abort

from builtin.constrained import _constrained_conforms_to


struct Node[
    ElementType: Copyable & ImplicitlyDestructible,
](Copyable):
    """A node in a linked list data structure.

    Parameters:
        ElementType: The type of element stored in the node.
    """

    comptime _NodePointer = UnsafePointer[Self, MutExternalOrigin]

    var value: Self.ElementType
    """The value stored in this node."""
    var prev: Self._NodePointer
    """The previous node in the list."""
    var next: Self._NodePointer
    """The next node in the list."""

    fn __init__(
        out self,
        var value: Self.ElementType,
        prev: Optional[Self._NodePointer],
        next: Optional[Self._NodePointer],
    ):
        """Initialize a new Node with the given value and optional prev/next
        pointers.

        Args:
            value: The value to store in this node.
            prev: Optional pointer to the previous node.
            next: Optional pointer to the next node.
        """
        self.value = value^
        self.prev = prev.value() if prev else Self._NodePointer()
        self.next = next.value() if next else Self._NodePointer()

    fn __str__[
        _ElementType: Copyable & ImplicitlyDestructible & Writable
    ](self: Node[_ElementType]) -> String:
        """Convert this node's value to a string representation.

        Parameters:
            _ElementType: Used to conditionally enable this function if
                `_ElementType` is `Writable`.

        Returns:
            String representation of the node's value.
        """
        return String.write(self.value)

    fn _into_value(deinit self) -> Self.ElementType:
        return self.value^

    @no_inline
    fn write_to[
        _ElementType: Copyable & ImplicitlyDestructible & Writable
    ](self: Node[_ElementType], mut writer: Some[Writer]):
        """Write this node's value to the given writer.

        Parameters:
            _ElementType: Used to conditionally enable this function if
                `_ElementType` is `Writable`.

        Args:
            writer: The writer to write the value to.
        """
        writer.write(self.value)


@fieldwise_init
struct _LinkedListIter[
    mut: Bool,
    //,
    ElementType: Copyable & ImplicitlyDestructible,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    var src: Pointer[LinkedList[Self.ElementType], Self.origin]
    var curr: UnsafePointer[Node[Self.ElementType], MutExternalOrigin]

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime Element = Self.ElementType  # FIXME(MOCO-2068): shouldn't be needed.

    fn __init__(out self, src: Pointer[LinkedList[Self.Element], Self.origin]):
        self.src = src

        @parameter
        if Self.forward:
            self.curr = self.src[]._head
        else:
            self.curr = self.src[]._tail

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        if not self.curr:
            raise StopIteration()
        var old = self.curr

        @parameter
        if Self.forward:
            self.curr = self.curr[].next
        else:
            self.curr = self.curr[].prev

        return old[].value


struct LinkedList[ElementType: Copyable & ImplicitlyDestructible](
    Boolable,
    Copyable,
    Defaultable,
    Iterable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """A doubly-linked list implementation.

    Parameters:
        ElementType: The type of elements stored in the list. Must implement the
            `Copyable` trait.

    A doubly-linked list is a data structure where each element points to both
    the next and previous elements, allowing for efficient insertion and deletion
    at any position.
    """

    comptime _NodePointer = UnsafePointer[
        Node[Self.ElementType], MutExternalOrigin
    ]

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _LinkedListIter[Self.ElementType, iterable_origin]
    """The iterator type for this linked list.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    var _head: Self._NodePointer
    """The first node in the list."""
    var _tail: Self._NodePointer
    """The last node in the list."""
    var _size: Int
    """The number of elements in the list."""

    fn __init__(out self):
        """Initialize an empty linked list.

        Notes:
            Time Complexity: O(1).
        """
        self._head = Self._NodePointer()
        self._tail = Self._NodePointer()
        self._size = 0

    fn __init__(
        out self, var *elements: Self.ElementType, __list_literal__: () = ()
    ):
        """Initialize a linked list with the given elements.

        Args:
            elements: Variable number of elements to initialize the list with.
            __list_literal__: Tell Mojo to use this method for list literals.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self(elements=elements^)

    fn __init__(
        out self, *, var elements: VariadicListMem[Self.ElementType, _]
    ):
        """Construct a list from a `VariadicListMem`.

        Args:
            elements: The elements to add to the list.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self()

        # Transfer all of the elements into the list.
        @parameter
        fn init_elt(idx: Int, var elt: Self.ElementType):
            self.append(elt^)

        elements^.consume_elements[init_elt]()

    fn __copyinit__(out self, read copy: Self):
        """Initialize this list as a copy of another list.

        Args:
            copy: The list to copy from.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self()
        var curr = copy._head
        while curr:
            self.append(curr[].value.copy())
            curr = curr[].next

    fn __del__(deinit self):
        """Clean up the list by freeing all nodes.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var curr = self._head
        while curr:
            var next = curr[].next
            curr.destroy_pointee()
            curr.free()
            curr = next

    fn append(mut self, var value: Self.ElementType):
        """Add an element to the end of the list.

        Args:
            value: The value to append.

        Notes:
            Time Complexity: O(1).
        """
        var addr = alloc[Node[Self.ElementType]](1)
        if not addr:
            abort("Out of memory")
        var value_ptr = UnsafePointer(to=addr[].value)
        value_ptr.init_pointee_move(value^)
        addr[].prev = self._tail
        addr[].next = Self._NodePointer()
        if self._tail:
            self._tail[].next = addr
        else:
            self._head = addr
        self._tail = addr
        self._size += 1

    fn prepend(mut self, var value: Self.ElementType):
        """Add an element to the beginning of the list.

        Args:
            value: The value to prepend.

        Notes:
            Time Complexity: O(1).
        """
        var node = Node(value^, None, self._head)
        var addr = alloc[Node[Self.ElementType]](1)
        if not addr:
            abort("Out of memory")
        addr.init_pointee_move(node^)
        if self:
            self._head[].prev = addr
        else:
            self._tail = addr
        self._head = addr
        self._size += 1

    fn reverse(mut self):
        """Reverse the order of elements in the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var prev = Self._NodePointer()
        var curr = self._head
        while curr:
            var next = curr[].next
            curr[].next = prev
            prev = curr
            curr = next
        self._tail = self._head
        self._head = prev

    fn pop(mut self) raises -> Self.ElementType:
        """Remove and return the last element of the list.

        Returns:
            The last element in the list.

        Notes:
            Time Complexity: O(1).

        Raises:
            If the operation fails.
        """
        var elem = self._tail
        if not elem:
            raise "Pop on empty list."

        var node = elem.take_pointee()
        self._tail = node.prev
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail[].next = Self._NodePointer()
        elem.free()
        return node^._into_value()

    fn pop[I: Indexer, //](mut self, var i: I) raises -> Self.ElementType:
        """Remove the ith element of the list, counting from the tail if
        given a negative index.

        Parameters:
            I: The type of index to use.

        Args:
            i: The index of the element to get.

        Returns:
            Ownership of the indicated element.

        Notes:
            Time Complexity: O(n) in len(self).

        Raises:
            If the operation fails.
        """
        var idx = index(i)
        var current = self._get_node_ptr(idx)

        if current:
            var node = current.take_pointee()
            if node.prev:
                node.prev[].next = node.next
            else:
                self._head = node.next
            if node.next:
                node.next[].prev = node.prev
            else:
                self._tail = node.prev

            current.free()
            self._size -= 1
            return node^._into_value()

        raise Error("Invalid index for pop: ", idx)

    fn maybe_pop(mut self) -> Optional[Self.ElementType]:
        """Removes the tail of the list and returns it, if it exists.

        Returns:
            The tail of the list, if it was present.

        Notes:
            Time Complexity: O(1).
        """
        var elem = self._tail
        if not elem:
            return Optional[Self.ElementType]()
        var node = elem.take_pointee()
        self._tail = node.prev
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail[].next = Self._NodePointer()
        elem.free()
        return node^._into_value()

    fn maybe_pop[
        I: Indexer, //
    ](mut self, var i: I) -> Optional[Self.ElementType]:
        """Remove the ith element of the list, counting from the tail if
        given a negative index.

        Parameters:
            I: The type of index to use.

        Args:
            i: The index of the element to get.

        Returns:
            The element, if it was found.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var current = self._get_node_ptr(index(i))

        if not current:
            return Optional[Self.ElementType]()
        else:
            var node = current.take_pointee()
            if node.prev:
                node.prev[].next = node.next
            else:
                self._head = node.next
            if node.next:
                node.next[].prev = node.prev
            else:
                self._tail = node.prev

            current.free()
            self._size -= 1
            return Optional[Self.ElementType](node^._into_value())

    fn clear(mut self):
        """Removes all elements from the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var current = self._head
        while current:
            var old = current
            current = current[].next
            old.destroy_pointee()
            old.free()

        self._head = Self._NodePointer()
        self._tail = Self._NodePointer()
        self._size = 0

    fn insert[I: Indexer](mut self, idx: I, var elem: Self.ElementType) raises:
        """Insert an element `elem` into the list at index `idx`.

        Parameters:
            I: The type of index to use.

        Args:
            idx: The index to insert `elem` at `-len(self) <= idx <= len(self)`.
            elem: The item to insert into the list.

        Raises:
            When given an out of bounds index.

        Notes:
            Time Complexity: O(n) in len(self).
        """

        # TODO: use normalize_index
        var i = index(idx)
        i = max(i if i >= 0 else i + len(self), 0)

        if i == 0:
            var node = alloc[Node[Self.ElementType]](1)
            if not node:
                abort("Out of memory")
            node.init_pointee_move(
                Node[Self.ElementType](
                    elem^, Self._NodePointer(), Self._NodePointer()
                )
            )

            if self._head:
                node[].next = self._head
                self._head[].prev = node

            self._head = node

            if not self._tail:
                self._tail = node

            self._size += 1
            return

        i -= 1

        var current = self._get_node_ptr(i)
        if current:
            var next = current[].next
            var node = alloc[Node[Self.ElementType]](1)
            if not node:
                abort("Out of memory")
            var data = UnsafePointer(to=node[].value)
            data[] = elem^
            node[].next = next
            node[].prev = current
            if next:
                next[].prev = node
            current[].next = node
            if node[].next == Self._NodePointer():
                self._tail = node
            if node[].prev == Self._NodePointer():
                self._head = node
            self._size += 1
        else:
            raise Error("Index ", i, " out of bounds")

    fn extend(mut self, var other: Self):
        """Extends the list with another.

        Args:
            other: The list to append to this one.

        Notes:
            Time Complexity: O(1).
        """
        if self._tail:
            self._tail[].next = other._head
            if other._head:
                other._head[].prev = self._tail
            if other._tail:
                self._tail = other._tail

            self._size += other._size
        else:
            self._head = other._head
            self._tail = other._tail
            self._size = other._size

        other._head = Self._NodePointer()
        other._tail = Self._NodePointer()

    fn count[
        _ElementType: Equatable & Copyable, //
    ](self: LinkedList[_ElementType], read elem: _ElementType) -> UInt:
        """Count the occurrences of `elem` in the list.

        Parameters:
            _ElementType: The list element type, used to conditionally enable the
                function.

        Args:
            elem: The element to search for.

        Returns:
            The number of occurrences of `elem` in the list.

        Notes:
            Time Complexity: O(n) in len(self) compares.
        """
        var current = self._head
        var count = 0
        while current:
            if current[].value == elem:
                count += 1

            current = current[].next

        return UInt(count)

    fn __contains__[
        _ElementType: Equatable & Copyable, //
    ](self: LinkedList[_ElementType], value: _ElementType) -> Bool:
        """Checks if the list contains `value`.

        Parameters:
            _ElementType: The list element type, used to conditionally enable the
                function.

        Args:
            value: The value to search for in the list.

        Returns:
            Whether the list contains `value`.

        Notes:
            Time Complexity: O(n) in len(self) compares.
        """
        var current = self._head
        while current:
            if current[].value == value:
                return True
            current = current[].next

        return False

    fn __eq__[
        _ElementType: Equatable & Copyable, //
    ](
        read self: LinkedList[_ElementType],
        read other: LinkedList[_ElementType],
    ) -> Bool:
        """Checks if the two lists are equal.

        Parameters:
            _ElementType: The list element type, used to conditionally enable the
                function.

        Args:
            other: The list to compare to.

        Returns:
            Whether the lists are equal.

        Notes:
            Time Complexity: O(n) in min(len(self), len(other)) compares.
        """
        if self._size != other._size:
            return False

        var self_cursor = self._head
        var other_cursor = other._head

        while self_cursor:
            if self_cursor[].value != other_cursor[].value:
                return False

            self_cursor = self_cursor[].next
            other_cursor = other_cursor[].next

        return True

    fn __ne__[
        _ElementType: Equatable & Copyable, //
    ](self: LinkedList[_ElementType], other: LinkedList[_ElementType]) -> Bool:
        """Checks if the two lists are not equal.

        Parameters:
            _ElementType: The list element type, used to conditionally enable the
                function.

        Args:
            other: The list to compare to.

        Returns:
            Whether the lists are not equal.

        Notes:
            Time Complexity: O(n) in min(len(self), len(other)) compares.
        """
        return not (self == other)

    fn _get_node_ptr[
        I: Indexer, //
    ](ref self, idx: I) -> UnsafePointer[
        Node[Self.ElementType], MutExternalOrigin
    ]:
        """Get a pointer to the node at the specified index.

        Parameters:
            I: The type of index to use.

        Args:
            idx: The index of the node to get.

        Returns:
            A pointer to the node at the specified index.

        Notes:
            This method optimizes traversal by starting from either the head or
            tail depending on which is closer to the target index.

            Time Complexity: O(n) in len(self).
        """
        var l = len(self)
        var i = normalize_index["LinkedList"](idx, l)
        debug_assert(0 <= i < l, "index out of bounds")
        var mid = l // 2
        if i <= mid:
            var curr = self._head
            for _ in range(i):
                curr = curr[].next
            return curr
        else:
            var curr = self._tail
            for _ in range(l - i - 1):
                curr = curr[].prev
            return curr

    fn __getitem__[I: Indexer](ref self, idx: I) -> ref[self] Self.ElementType:
        """Get the element at the specified index.

        Parameters:
            I: The type of index to use.

        Args:
            idx: The index of the element to get.

        Returns:
            The element at the specified index.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        debug_assert(len(self) > 0, "unable to get item from empty list")
        return self._get_node_ptr(idx)[].value

    fn __len__(self) -> Int:
        """Get the number of elements in the list.

        Returns:
            The number of elements in the list.

        Notes:
            Time Complexity: O(1).
        """
        return self._size

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.

        Notes:
            Time Complexity:
            - O(1) for iterator construction.
            - O(n) in len(self) for a complete iteration of the list.
        """
        return _LinkedListIter(Pointer(to=self))

    fn __reversed__(
        self,
    ) -> _LinkedListIter[Self.ElementType, origin_of(self), forward=False]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.

        Notes:
            Time Complexity:
            - O(1) for iterator construction.
            - O(n) in len(self) for a complete iteration of the list.
        """
        return _LinkedListIter[
            Self.ElementType, origin_of(self), forward=False
        ](Pointer(to=self))

    fn __bool__(self) -> Bool:
        """Check if the list is non-empty.

        Returns:
            True if the list has elements, False otherwise.

        Notes:
            Time Complexity: O(1).
        """
        return len(self) != 0

    fn _write_self_to[
        f: fn(Self.ElementType, mut Some[Writer])
    ](self, mut writer: Some[Writer]):
        fmt.constrained_conforms_to_writable[Self.ElementType, Parent=Self]()

        var iterator = self.__iter__()

        @parameter
        fn iterate(mut w: Some[Writer]) raises StopIteration:
            f(iterator.__next__(), w)

        fmt.write_sequence_to[ElementFn=iterate](writer)
        _ = iterator^

    fn __str__(self) -> String:
        """Convert the list to its string representation.

        Returns:
            String representation of the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var writer = String()
        self.write_to(writer)
        return writer

    fn __repr__(self) -> String:
        """Convert the list to its string representation.

        Returns:
            String representation of the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var writer = String()
        self.write_repr_to(writer)
        return writer

    fn write_to(self, mut writer: Some[Writer]):
        """Write the list to the given writer.

        Constraints:
            ElementType must conform to `Writable`.

        Args:
            writer: The writer to write the list to.
        """
        self._write_self_to[f = fmt.write_to[Self.ElementType]](writer)

    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr representation of this LinkedList to a Writer.

        Constraints:
            ElementType must conform to `Writable`.

        Args:
            writer: The writer to write to.
        """

        @parameter
        fn write_fields(mut w: Some[Writer]):
            self._write_self_to[f = fmt.write_repr_to[Self.ElementType]](w)

        fmt.FormatStruct(writer, "LinkedList").params(
            fmt.TypeNames[Self.ElementType](),
        ).fields[FieldsFn=write_fields]()
