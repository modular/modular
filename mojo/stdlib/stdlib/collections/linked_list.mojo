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


from collections._index_normalization import normalize_index
from os import abort


struct Node[
    ElementType: Copyable & Movable,
](Copyable, Movable):
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
        var value: ElementType,
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
        ElementType: Copyable & Movable & Writable
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
        ElementType: Copyable & Movable & Writable
    ](self: Node[ElementType], mut writer: Some[Writer]):
        """Write this node's value to the given writer.

        Parameters:
            ElementType: Used to conditionally enable this function if
                `ElementType` is `Writable`.

        Args:
            writer: The writer to write the value to.
        """
        writer.write(self.value)


@fieldwise_init
struct _LinkedListIter[
    mut: Bool, //,
    ElementType: Copyable & Movable,
    origin: Origin[mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterator, Movable):
    var src: Pointer[LinkedList[ElementType], origin]
    var curr: UnsafePointer[Node[ElementType]]

    alias Element = ElementType  # FIXME(MOCO-2068): shouldn't be needed.

    fn __init__(out self, src: Pointer[LinkedList[Self.Element], origin]):
        self.src = src

        @parameter
        if forward:
            self.curr = self.src[]._head
        else:
            self.curr = self.src[]._tail

    fn __iter__(self) -> Self:
        return self.copy()

    fn __has_next__(self) -> Bool:
        return Bool(self.curr)

    fn __next_ref__(mut self) -> ref [origin] Self.Element:
        var old = self.curr

        @parameter
        if forward:
            self.curr = self.curr[].next
        else:
            self.curr = self.curr[].prev

        return old[].value

    @always_inline
    fn __next__(mut self) -> Self.Element:
        return self.__next_ref__().copy()


struct LinkedList[
    ElementType: Copyable & Movable,
](Boolable, Copyable, Defaultable, Movable, Sized):
    """A doubly-linked list implementation.

    Parameters:
        ElementType: The type of elements stored in the list. Must implement the
            `Copyable` and `Movable` traits.

    A doubly-linked list is a data structure where each element points to both
    the next and previous elements, allowing for efficient insertion and deletion
    at any position.
    """

    alias _NodePointer = UnsafePointer[Node[ElementType]]

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
        out self, var *elements: ElementType, __list_literal__: () = ()
    ):
        """Initialize a linked list with the given elements.

        Args:
            elements: Variable number of elements to initialize the list with.
            __list_literal__: Tell Mojo to use this method for list literals.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self(elements=elements^)

    fn __init__(out self, *, var elements: VariadicListMem[ElementType, _]):
        """Construct a list from a `VariadicListMem`.

        Args:
            elements: The elements to add to the list.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self()

        # Transfer all of the elements into the list.
        @parameter
        fn init_elt(idx: Int, var elt: ElementType):
            self.append(elt^)

        elements^.consume_elements[init_elt]()

    fn __copyinit__(out self, read other: Self):
        """Initialize this list as a copy of another list.

        Args:
            other: The list to copy from.

        Notes:
            Time Complexity: O(n) in len(elements).
        """
        self = Self()
        var curr = other._head
        while curr:
            self.append(curr[].value.copy())
            curr = curr[].next

    fn __moveinit__(out self, deinit other: Self):
        """Initialize this list by moving elements from another list.

        Args:
            other: The list to move elements from.

        Notes:
            Time Complexity: O(1).
        """
        self._head = other._head
        self._tail = other._tail
        self._size = other._size

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

    fn append(mut self, var value: ElementType):
        """Add an element to the end of the list.

        Args:
            value: The value to append.

        Notes:
            Time Complexity: O(1).
        """
        var addr = Self._NodePointer.alloc(1)
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

    fn prepend(mut self, var value: ElementType):
        """Add an element to the beginning of the list.

        Args:
            value: The value to prepend.

        Notes:
            Time Complexity: O(1).
        """
        var node = Node(value^, None, self._head)
        var addr = Self._NodePointer.alloc(1)
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

    fn pop(mut self) raises -> ElementType:
        """Remove and return the last element of the list.

        Returns:
            The last element in the list.

        Notes:
            Time Complexity: O(1).
        """
        var elem = self._tail
        if not elem:
            raise "Pop on empty list."

        # FIXME(MSTDL-1755): Should this copy be a move; are we leaking `value`?
        var value = elem[].value.copy()
        self._tail = elem[].prev
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail[].next = Self._NodePointer()
        elem.free()
        return value^

    fn pop[I: Indexer, //](mut self, var i: I) raises -> ElementType:
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
        """
        var idx = index(i)
        var current = self._get_node_ptr(idx)

        if current:
            var node = current[].copy()
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

        raise Error("Invalid index for pop: ", idx)

    fn maybe_pop(mut self) -> Optional[ElementType]:
        """Removes the tail of the list and returns it, if it exists.

        Returns:
            The tail of the list, if it was present.

        Notes:
            Time Complexity: O(1).
        """
        var elem = self._tail
        if not elem:
            return Optional[ElementType]()
        # FIXME(MSTDL-1755): Should this copy be a move; are we leaking `value`?
        var value = elem[].value.copy()
        self._tail = elem[].prev
        self._size -= 1
        if self._size == 0:
            self._head = Self._NodePointer()
        else:
            self._tail[].next = Self._NodePointer()
        elem.free()
        return value^

    fn maybe_pop[I: Indexer, //](mut self, var i: I) -> Optional[ElementType]:
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
        var current = self._get_node_ptr(Int(index(i)))

        if not current:
            return Optional[ElementType]()
        else:
            var node = current[].copy()
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

    fn insert[I: Indexer](mut self, idx: I, var elem: ElementType) raises:
        """Insert an element `elem` into the list at index `idx`.

        Parameters:
            I: The type of index to use.

        Raises:
            When given an out of bounds index.

        Args:
            idx: The index to insert `elem` at `-len(self) <= idx <= len(self)`.
            elem: The item to insert into the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """

        # TODO: use normalize_index
        var i = index(idx)
        i = max(0, i if i >= 0 else i + len(self))

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
        ElementType: EqualityComparable & Copyable & Movable, //
    ](self: LinkedList[ElementType], read elem: ElementType) -> UInt:
        """Count the occurrences of `elem` in the list.

        Parameters:
            ElementType: The list element type, used to conditionally enable the
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
        ElementType: EqualityComparable & Copyable & Movable, //
    ](self: LinkedList[ElementType], value: ElementType) -> Bool:
        """Checks if the list contains `value`.

        Parameters:
            ElementType: The list element type, used to conditionally enable the
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
        ElementType: EqualityComparable & Copyable & Movable, //
    ](
        read self: LinkedList[ElementType], read other: LinkedList[ElementType]
    ) -> Bool:
        """Checks if the two lists are equal.

        Parameters:
            ElementType: The list element type, used to conditionally enable the
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
        ElementType: EqualityComparable & Copyable & Movable, //
    ](self: LinkedList[ElementType], other: LinkedList[ElementType]) -> Bool:
        """Checks if the two lists are not equal.

        Parameters:
            ElementType: The list element type, used to conditionally enable the
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
    ](ref self, idx: I) -> UnsafePointer[Node[ElementType]]:
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

    fn __getitem__[I: Indexer](ref self, idx: I) -> ref [self] ElementType:
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

    fn __setitem__[I: Indexer](mut self, idx: I, var value: ElementType):
        """Set the element at the specified index.

        Parameters:
            I: The type of index to use.

        Args:
            idx: The index of the element to set.
            value: The new value to set.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        debug_assert(len(self) > 0, "unable to set item from empty list")
        self._get_node_ptr(idx)[].value = value^

    fn __len__(self) -> Int:
        """Get the number of elements in the list.

        Returns:
            The number of elements in the list.

        Notes:
            Time Complexity: O(1).
        """
        return self._size

    fn __iter__(self) -> _LinkedListIter[ElementType, __origin_of(self)]:
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
    ) -> _LinkedListIter[ElementType, __origin_of(self), forward=False]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.

        Notes:
            Time Complexity:
            - O(1) for iterator construction.
            - O(n) in len(self) for a complete iteration of the list.
        """
        return _LinkedListIter[ElementType, __origin_of(self), forward=False](
            Pointer(to=self)
        )

    fn __bool__(self) -> Bool:
        """Check if the list is non-empty.

        Returns:
            True if the list has elements, False otherwise.

        Notes:
            Time Complexity: O(1).
        """
        return len(self) != 0

    fn __str__[
        ElementType: Copyable & Movable & Writable
    ](self: LinkedList[ElementType]) -> String:
        """Convert the list to its string representation.

        Parameters:
            ElementType: Used to conditionally enable this function when
                `ElementType` is `Writable`.

        Returns:
            String representation of the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var writer = String()
        self._write(writer)
        return writer

    fn __repr__[
        ElementType: Copyable & Movable & Writable
    ](self: LinkedList[ElementType]) -> String:
        """Convert the list to its string representation.

        Parameters:
            ElementType: Used to conditionally enable this function when
                `ElementType` is `Writable`.

        Returns:
            String representation of the list.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        var writer = String()
        self._write(writer, prefix="LinkedList(", suffix=")")
        return writer

    fn write_to[
        ElementType: Copyable & Movable & Writable
    ](self: LinkedList[ElementType], mut writer: Some[Writer]):
        """Write the list to the given writer.

        Parameters:
            ElementType: Used to conditionally enable this function when
                `ElementType` is `Writable`.

        Args:
            writer: The writer to write the list to.

        Notes:
            Time Complexity: O(n) in len(self).
        """
        self._write(writer)

    @no_inline
    fn _write[
        W: Writer, ElementType: Copyable & Movable & Writable
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
