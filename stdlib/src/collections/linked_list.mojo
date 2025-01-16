# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

from memory import UnsafePointer
from collections import Optional
from collections._index_normalization import normalize_index


trait WritableCollectionElement(CollectionElement, Writable):
    """A trait that combines CollectionElement and Writable traits.

    This trait requires types to implement both CollectionElement and Writable
    interfaces, allowing them to be used in collections and written to output.
    """

    pass


@value
struct Node[ElementType: WritableCollectionElement]:
    """A node in a linked list data structure.

    Parameters:
        ElementType: The type of element stored in the node.
    """

    var value: ElementType
    """The value stored in this node."""
    var prev: UnsafePointer[Node[ElementType]]
    """The previous node in the list."""
    var next: UnsafePointer[Node[ElementType]]
    """The next node in the list."""

    fn __init__(
        out self,
        owned value: ElementType,
        prev: Optional[UnsafePointer[Node[ElementType]]],
        next: Optional[UnsafePointer[Node[ElementType]]],
    ):
        """Initialize a new Node with the given value and optional prev/next
        pointers.

        Args:
            value: The value to store in this node.
            prev: Optional pointer to the previous node.
            next: Optional pointer to the next node.
        """
        self.value = value^
        self.prev = prev.value() if prev else __type_of(self.prev)()
        self.next = next.value() if next else __type_of(self.next)()

    fn __str__(self) -> String:
        """Convert this node's value to a string representation.

        Returns:
            String representation of the node's value.
        """
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write this node's value to the given writer.

        Parameters:
            W: The type of writer to write the value to.

        Args:
            writer: The writer to write the value to.
        """
        writer.write(self.value)


struct LinkedList[ElementType: WritableCollectionElement]:
    """A doubly-linked list implementation.

    A doubly-linked list is a data structure where each element points to both
    the next and previous elements, allowing for efficient insertion and deletion
    at any position.

    Parameters:
        ElementType: The type of elements stored in the list. Must implement
            WritableCollectionElement.
    """

    var _head: UnsafePointer[Node[ElementType]]
    """The first node in the list."""
    var _tail: UnsafePointer[Node[ElementType]]
    """The last node in the list."""
    var _size: Int
    """The number of elements in the list."""

    fn __init__(out self):
        """Initialize an empty linked list."""
        self._head = __type_of(self._head)()
        self._tail = __type_of(self._tail)()
        self._size = 0

    fn __init__(mut self, owned *elements: ElementType):
        """Initialize a linked list with the given elements.

        Args:
            elements: Variable number of elements to initialize the list with.
        """
        self = Self(elements=elements^)

    fn __init__(out self, *, owned elements: VariadicListMem[ElementType, _]):
        """Initialize a linked list with the given elements.

        Args:
            elements: Variable number of elements to initialize the list with.
        """
        self = Self()

        for elem in elements:
            self.append(elem[])

        # Do not destroy the elements when their backing storage goes away.
        __mlir_op.`lit.ownership.mark_destroyed`(
            __get_mvalue_as_litref(elements)
        )

    fn __copyinit__(mut self, read other: Self):
        """Initialize this list as a copy of another list.

        Args:
            other: The list to copy from.
        """
        self._head = other._head
        self._tail = other._tail
        self._size = other._size

    fn __moveinit__(mut self, owned other: Self):
        """Initialize this list by moving elements from another list.

        Args:
            other: The list to move elements from.
        """
        self._head = other._head
        self._tail = other._tail
        self._size = other._size
        other._head = __type_of(other._head)()
        other._tail = __type_of(other._tail)()
        other._size = 0

    fn __del__(owned self):
        """Clean up the list by freeing all nodes."""
        var curr = self._head
        while curr:
            var next = curr[].next
            curr[].prev = __type_of(self._head)()
            curr[].next = __type_of(self._tail)()
            curr.destroy_pointee()
            curr.free()
            curr = next

    fn append(mut self, owned value: ElementType):
        """Add an element to the end of the list.

        Args:
            value: The value to append.
        """
        var node = Node[ElementType](value^, self._tail, None)
        var addr = UnsafePointer[__type_of(node)].alloc(1)
        addr.init_pointee_move(node)
        if self:
            self._tail[].next = addr
        else:
            self._head = addr
        self._tail = addr
        self._size += 1

    fn prepend(mut self, owned value: ElementType):
        """Add an element to the beginning of the list.

        Args:
            value: The value to prepend.
        """
        var node = Node[ElementType](value^, None, self._head)
        var addr = UnsafePointer[__type_of(node)].alloc(1)
        addr.init_pointee_move(node)
        if self:
            self._head[].prev = addr
        else:
            self._tail = addr
        self._head = addr
        self._size += 1

    fn reverse(mut self):
        """Reverse the order of elements in the list."""
        var prev = __type_of(self._head)()
        var curr = self._head
        while curr:
            var next = curr[].next
            curr[].next = prev
            prev = curr
            curr = next
        self._head = prev
        self._tail = self._head[].prev

    fn pop(mut self) -> ElementType:
        """Remove and return the first element of the list.

        Returns:
            The first element in the list.
        """
        var elem = self._tail
        var value = elem[].value
        self._tail = elem[].prev
        self._size -= 1
        return value^

    fn copy(self) -> Self:
        """Create a deep copy of the list.

        Returns:
            A new list containing copies of all elements.
        """
        var new = Self()
        var curr = self._head
        while curr:
            new.append(curr[].value)
            curr = curr[].next
        return new^

    fn __getitem__(ref self, index: Int) -> ref [self] ElementType:
        """Get the element at the specified index.

        Args:
            index: The index of the element to get.

        Returns:
            The element at the specified index.
        """
        var curr = self._head
        for _ in range(
            normalize_index[container_name="LinkedList"](index, self)
        ):
            curr = curr[].next
        return curr[].value

    fn __setitem__(mut self, index: Int, owned value: ElementType):
        """Set the element at the specified index.

        Args:
            index: The index of the element to set.
            value: The new value to set.
        """
        var curr = self._head
        for _ in range(
            normalize_index[container_name="LinkedList"](index, self)
        ):
            curr = curr[].next
        curr[].value = value^

    fn __len__(self) -> Int:
        """Get the number of elements in the list.

        Returns:
            The number of elements in the list.
        """
        return self._size

    fn __bool__(self) -> Bool:
        """Check if the list is non-empty.

        Returns:
            True if the list has elements, False otherwise.
        """
        return len(self) != 0

    fn __str__(self) -> String:
        """Convert the list to its string representation.

        Returns:
            String representation of the list.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """Convert the list to its string representation.

        Returns:
            String representation of the list.
        """
        var writer = String()
        self._write(writer, prefix="LinkedList(", suffix=")")
        return writer

    fn write_to[W: Writer](self, mut writer: W):
        """Write the list to the given writer.

        Parameters:
            W: The type of writer to write the list to.

        Args:
            writer: The writer to write the list to.
        """
        self._write(writer)

    @no_inline
    fn _write[
        W: Writer
    ](self, mut writer: W, *, prefix: String = "[", suffix: String = "]"):
        if not self:
            return writer.write(prefix, suffix)

        var curr = self._head
        writer.write(prefix)
        for i in range(len(self)):
            if i:
                writer.write(", ")
            writer.write(curr[])
            curr = curr[].next
        writer.write(suffix)
