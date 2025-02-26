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

from collections import Optional
from collections._index_normalization import normalize_index
from os import abort

from memory import UnsafePointer


@value
struct Node[
    ElementType: CollectionElement,
]:
    """A node in a linked list data structure.

    Parameters:
        ElementType: The type of element stored in the node.
    """

    alias _NodePointer = UnsafePointer[Self]

    var value: ElementType
    """The value stored in this node."""
    var prev: Self._NodePointer
    """The previous node in the list."""
    var next: Self._NodePointer
    """The next node in the list."""

    fn __init__(
        out self,
        owned value: ElementType,
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
        ElementType: WritableCollectionElement
    ](self: Node[ElementType]) -> String:
        """Convert this node's value to a string representation.

        Parameters:
            ElementType: Used to conditionally enable this function if
              `ElementType` is `Writable`.

        Returns:
            String representation of the node's value.
        """
        return String.write(self.value)

    @no_inline
    fn write_to[
        ElementType: WritableCollectionElement, W: Writer
    ](self: Node[ElementType], mut writer: W):
        """Write this node's value to the given writer.

        Parameters:
            ElementType: Used to conditionally enable this function if
              `ElementType` is `Writable`.
            W: The type of writer to write the value to.

        Args:
            writer: The writer to write the value to.
        """
        writer.write(self.value)


@value
struct _LinkedListIter[
    mut: Bool, //,
    ElementType: CollectionElement,
    origin: Origin[mut],
    forward: Bool = True,
]:
    var src: Pointer[LinkedList[ElementType], origin]
    var curr: UnsafePointer[Node[ElementType]]

    # Used to calculate remaining length of iterator in
    # _LinkedListIter.__len__()
    var seen: Int

    fn __init__(out self, src: Pointer[LinkedList[ElementType], origin]):
        self.src = src

        @parameter
        if forward:
            self.curr = self.src[]._head
        else:
            self.curr = self.src[]._tail
        self.seen = 0

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self, out p: Pointer[ElementType, origin]):
        p = Pointer[ElementType, origin].address_of(self.curr[].value)

        @parameter
        if forward:
            self.curr = self.curr[].next
        else:
            self.curr = self.curr[].prev
        self.seen += 1

    fn __has_next__(self) -> Bool:
        return Bool(self.curr)

    fn __len__(self) -> Int:
        return len(self.src[]) - self.seen


struct LinkedList[
    ElementType: CollectionElement,
]:
    """A doubly-linked list implementation.

    A doubly-linked list is a data structure where each element points to both
    the next and previous elements, allowing for efficient insertion and deletion
    at any position.

    Parameters:
        ElementType: The type of elements stored in the list. Must implement
         CollectionElement.
    """

    alias _NodePointer = UnsafePointer[Node[ElementType]]

    var _head: Self._NodePointer
    """The first node in the list."""
    var _tail: Self._NodePointer
    """The last node in the list."""
    var _size: Int
    """The number of elements in the list."""

    fn __init__(out self):
        """
        Initialize an empty linked list.

        Time Complexity: O(1)
        """
        self._head = Self._NodePointer()
        self._tail = Self._NodePointer()
        self._size = 0

    fn __init__(mut self, owned *elements: ElementType):
        """Initialize a linked list with the given elements.

        Time Complexity: O(n) in len(elements)

        Args:
            elements: Variable number of elements to initialize the list with.
        """
        self = Self(elements=elements^)

    fn __init__(out self, *, owned elements: VariadicListMem[ElementType, _]):
        """
        Construct a list from a `VariadicListMem`.

        Time Complexity: O(n) in len(elements)

        Args:
            elements: The elements to add to the list.
        """
        self = Self()

        var length = len(elements)

        for i in range(length):
            var src = UnsafePointer.address_of(elements[i])
            var node = Self._NodePointer.alloc(1)
            if not node:
                abort("Out of memory")
            var dst = UnsafePointer.address_of(node[].value)
            src.move_pointee_into(dst)
            node[].next = Self._NodePointer()
            node[].prev = self._tail
            if self._tail:
                self._tail[].next = node
                self._tail = node
            else:
                self._head = node
                self._tail = node

        # Do not destroy the elements when their backing storage goes away.
        # FIXME(https://github.com/modular/mojo/issues/3969) this is leaking!
        __disable_del elements

        self._size = length

    fn __copyinit__(mut self, read other: Self):
        """Initialize this list as a copy of another list.

        Time Complexity: O(n) in len(elements)

        Args:
            other: The list to copy from.
        """
        self = other.copy()

    fn __moveinit__(mut self, owned other: Self):
        """Initialize this list by moving elements from another list.

        Time Complexity: O(1)

        Args:
            other: The list to move elements from.
        """
        self._head = other._head
        self._tail = other._tail
        self._size = other._size
        other._head = Self._NodePointer()
        other._tail = Self._NodePointer()
        other._size = 0

    fn __del__(owned self):
        """
        Clean up the list by freeing all nodes.

        Time Complexity: O(n) in len(self)
        """
        var curr = self._head
        while curr:
            var next = curr[].next
            curr.destroy_pointee()
            curr.free()
            curr = next

    fn append(mut self, owned value: ElementType):
        """
        Add an element to the end of the list.

        Time Complexity: O(1)

        Args:
            value: The value to append.
        """
        var addr = Self._NodePointer.alloc(1)
        if not addr:
            abort("Out of memory")
        var value_ptr = UnsafePointer.address_of(addr[].value)
        value_ptr.init_pointee_move(value^)
        addr[].prev = self._tail
        addr[].next = Self._NodePointer()
        if self._tail:
            self._tail[].next = addr
        else:
            self._head = addr
        self._tail = addr
        self._size += 1

    fn prepend(mut self, owned value: ElementType):
        """
        Add an element to the beginning of the list.

        Time Complexity: O(1)

        Args:
            value: The value to prepend.
        """
        var node = Node(value^, None, self._head)
        var addr = Self._NodePointer.alloc(1)
        if not addr:
            abort("Out of memory")
        addr.init_pointee_move(node)
        if self:
            self._head[].prev = addr
        else:
            self._tail = addr
        self._head = addr
        self._size += 1

    fn reverse(mut self):
        """
        Reverse the order of elements in the list.

        Time Complexity: O(n) in len(self)
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

    fn pop(mut self) raises -> ElementType:
        """Remove and return the last element of the list.

        Time Complexity: O(1)

        Returns:
            The last element in the list.
        """
        var elem = self._tail
        if not elem:
            raise "Pop on empty list."

        var value = elem[].value
        self._tail = elem[].prev
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail[].next = Self._NodePointer()
        elem.free()
        return value^

    fn pop[I: Indexer](mut self, owned i: I) raises -> ElementType:
        """
        Remove the ith element of the list, counting from the tail if
        given a negative index.

        Time Complexity: O(1)

        Parameters:
            I: The type of index to use.

        Args:
            i: The index of the element to get.

        Returns:
            Ownership of the indicated element.
        """
        var current = self._get_node_ptr(Int(i))

        if current:
            var node = current[]
            if node.prev:
                node.prev[].next = node.next
            else:
                self._head = node.next
            if node.next:
                node.next[].prev = node.prev
            else:
                self._tail = node.prev

            var data = node.value^

            # Aside from T, destructor is trivial
            __mlir_op.`lit.ownership.mark_destroyed`(
                __get_mvalue_as_litref(node)
            )
            current.free()
            self._size -= 1
            return data^

        raise String("Invalid index for pop: {}").format(Int(i))

    fn maybe_pop(mut self) -> Optional[ElementType]:
        """
        Removes the head of the list and returns it, if it exists.

        Time Complexity: O(1)

        Returns:
            The head of the list, if it was present.
        """
        var elem = self._tail
        if not elem:
            return Optional[ElementType]()
        var value = elem[].value
        self._tail = elem[].prev
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail[].next = Self._NodePointer()
        elem.free()
        return value^

    fn maybe_pop[I: Indexer](mut self, owned i: I) -> Optional[ElementType]:
        """
        Remove the ith element of the list, counting from the tail if
        given a negative index.

        Time Complexity: O(1)

        Parameters:
            I: The type of index to use.

        Args:
            i: The index of the element to get.

        Returns:
            The element, if it was found.
        """
        var current = self._get_node_ptr(Int(i))

        if not current:
            return Optional[ElementType]()
        else:
            var node = current[]
            if node.prev:
                node.prev[].next = node.next
            else:
                self._head = node.next
            if node.next:
                node.next[].prev = node.prev
            else:
                self._tail = node.prev

            var data = node.value^

            # Aside from T, destructor is trivial
            __mlir_op.`lit.ownership.mark_destroyed`(
                __get_mvalue_as_litref(node)
            )
            current.free()
            self._size -= 1
            return Optional[ElementType](data^)

    fn clear(mut self):
        """
        Removes all elements from the list.

        Time Complexity: O(n) in len(self)
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

    fn copy(self) -> Self:
        """Create a deep copy of the list.

        Time Complexity: O(n) in len(self)

        Returns:
            A new list containing copies of all elements.
        """
        var new = Self()
        var curr = self._head
        while curr:
            new.append(curr[].value)
            curr = curr[].next
        return new^

    fn insert[I: Indexer](mut self, idx: I, owned elem: ElementType) raises:
        """
        Insert an element `elem` into the list at index `idx`.

        Time Complexity: O(1)

        Parameters:
            I: The type of index to use.

        Raises:
            When given an out of bounds index.

        Args:
            idx: The index to insert `elem` at. `-len(self) <= idx <= len(self)`.
            elem: The item to insert into the list.
        """
        var i = max(0, index(idx) if Int(idx) >= 0 else index(idx) + len(self))

        if i == 0:
            var node = Self._NodePointer.alloc(1)
            if not node:
                abort("Out of memory")
            node.init_pointee_move(
                Node[ElementType](
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
            var node = Self._NodePointer.alloc(1)
            if not node:
                abort("Out of memory")
            var data = UnsafePointer.address_of(node[].value)
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
            raise String("Index {} out of bounds").format(i)

    fn extend(mut self, owned other: Self):
        """
        Extends the list with another.

        Time Complexity: O(1)

        Args:
            other: The list to append to this one.
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
        ElementType: EqualityComparableCollectionElement
    ](self: LinkedList[ElementType], read elem: ElementType) -> UInt:
        """
        Count the occurrences of `elem` in the list.

        Time Complexity: O(n) in len(self) compares

        Parameters:
            ElementType: The list element type, used to conditionally enable the function.

        Args:
            elem: The element to search for.

        Returns:
            The number of occurrences of `elem` in the list.
        """
        var current = self._head
        var count = 0
        while current:
            if current[].value == elem:
                count += 1

            current = current[].next

        return count

    fn __contains__[
        ElementType: EqualityComparableCollectionElement, //
    ](self: LinkedList[ElementType], value: ElementType) -> Bool:
        """
        Checks if the list contains `value`.

        Time Complexity: O(n) in len(self) compares

        Parameters:
            ElementType: The list element type, used to conditionally enable the function.

        Args:
            value: The value to search for in the list.

        Returns:
            Whether the list contains `value`.
        """
        var current = self._head
        while current:
            if current[].value == value:
                return True
            current = current[].next

        return False

    fn __eq__[
        ElementType: EqualityComparableCollectionElement, //
    ](
        read self: LinkedList[ElementType], read other: LinkedList[ElementType]
    ) -> Bool:
        """
        Checks if the two lists are equal.

        Time Complexity: O(n) in min(len(self), len(other)) compares

        Parameters:
            ElementType: The list element type, used to conditionally enable the function.

        Args:
            other: The list to compare to.

        Returns:
            Whether the lists are equal.
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
        ElementType: EqualityComparableCollectionElement, //
    ](self: LinkedList[ElementType], other: LinkedList[ElementType]) -> Bool:
        """
        Checks if the two lists are not equal.

        Time Complexity: O(n) in min(len(self), len(other)) compares

        Parameters:
            ElementType: The list element type, used to conditionally enable the function.

        Args:
            other: The list to compare to.

        Returns:
            Whether the lists are not equal.
        """
        return not (self == other)

    fn _get_node_ptr[
        I: Indexer
    ](ref self, index: I) -> UnsafePointer[Node[ElementType]]:
        """
        Get a pointer to the node at the specified index.

        This method optimizes traversal by starting from either the head or tail
        depending on which is closer to the target index.

        Time Complexity: O(n) in len(self)

        Parameters:
            I: The type of index to use.

        Args:
            index: The index of the node to get.

        Returns:
            A pointer to the node at the specified index.
        """
        var l = len(self)
        var i = normalize_index["LinkedList"](index, l)
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

    fn __getitem__[I: Indexer](ref self, index: I) -> ref [self] ElementType:
        """
        Get the element at the specified index.

        Time Complexity: O(n) in len(self)

        Parameters:
            I: The type of index to use.

        Args:
            index: The index of the element to get.

        Returns:
            The element at the specified index.
        """
        debug_assert(len(self) > 0, "unable to get item from empty list")
        return self._get_node_ptr(index)[].value

    fn __setitem__[I: Indexer](mut self, index: I, owned value: ElementType):
        """
        Set the element at the specified index.

        Time Complexity: O(n) in len(self)

        Parameters:
            I: The type of index to use.

        Args:
            index: The index of the element to set.
            value: The new value to set.
        """
        debug_assert(len(self) > 0, "unable to set item from empty list")
        self._get_node_ptr(index)[].value = value^

    fn __len__(self) -> Int:
        """Get the number of elements in the list.

        Time Complexity: O(1)

        Returns:
            The number of elements in the list.
        """
        return self._size

    fn __iter__(self) -> _LinkedListIter[ElementType, __origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Time Complexity:
            O(1) for iterator construction.
            O(n) in len(self) for a complete iteration of the list.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return _LinkedListIter(Pointer.address_of(self))

    fn __reversed__(
        self,
    ) -> _LinkedListIter[ElementType, __origin_of(self), forward=False]:
        """Iterate backwards over the list, returning immutable references.

        Time Complexity:
            O(1) for iterator construction.
            O(n) in len(self) for a complete iteration of the list.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        return _LinkedListIter[ElementType, __origin_of(self), forward=False](
            Pointer.address_of(self)
        )

    fn __bool__(self) -> Bool:
        """Check if the list is non-empty.

        Time Complexity: O(1)

        Returns:
            True if the list has elements, False otherwise.
        """
        return len(self) != 0

    fn __str__[
        ElementType: WritableCollectionElement
    ](self: LinkedList[ElementType]) -> String:
        """Convert the list to its string representation.

        Time Complexity: O(n) in len(self)

        Parameters:
            ElementType: Used to conditionally enable this function when
             `ElementType` is `Writable`.

        Returns:
            String representation of the list.
        """
        var writer = String()
        self._write(writer)
        return writer

    fn __repr__[
        ElementType: WritableCollectionElement
    ](self: LinkedList[ElementType]) -> String:
        """Convert the list to its string representation.

        Time Complexity: O(n) in len(self)

        Parameters:
            ElementType: Used to conditionally enable this function when
             `ElementType` is `Writable`.

        Returns:
            String representation of the list.
        """
        var writer = String()
        self._write(writer, prefix="LinkedList(", suffix=")")
        return writer

    fn write_to[
        W: Writer, ElementType: WritableCollectionElement
    ](self: LinkedList[ElementType], mut writer: W):
        """Write the list to the given writer.

        Time Complexity: O(n) in len(self)

        Parameters:
            W: The type of writer to write the list to.
            ElementType: Used to conditionally enable this function when
             `ElementType` is `Writable`.

        Args:
            writer: The writer to write the list to.
        """
        self._write(writer)

    @no_inline
    fn _write[
        W: Writer, ElementType: WritableCollectionElement
    ](
        self: LinkedList[ElementType],
        mut writer: W,
        *,
        prefix: String = "[",
        suffix: String = "]",
    ):
        if not self:
            return writer.write(prefix, suffix)

        var curr = self._head
        writer.write(prefix)
        for i in range(len(self)):
            if i:
                writer.write(", ")
            writer.write(curr[].value)
            curr = curr[].next
        writer.write(suffix)
