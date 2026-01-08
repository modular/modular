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

from os import abort


comptime Element = String  # Adapt for your type
comptime ListNode = Node[Element]  # Constructing a LinkedList


struct Node[ElementType: ImplicitlyCopyable & Writable](Movable):
    comptime NodePointer = UnsafePointer[Self, MutExternalOrigin]

    var value: Optional[Self.ElementType]  # The `Node`'s value
    var next: Self.NodePointer  # Pointer to the next `Node`

    # Uses an `Optional` value to allow 'empty' Node construction
    # that can be moved into newly allocated memory
    fn __init__(out self, value: Optional[Self.ElementType] = None):
        self.value = value
        self.next = Self.NodePointer()

    # Constructs a `Node` with a `value` with heap allocation and
    # returns a pointer to the new `Node`.
    @staticmethod
    fn make_node(value: Self.ElementType) -> Self.NodePointer:
        node_ptr = alloc[Self](1)
        node_ptr.init_pointee_move(Self(value))
        return node_ptr

    # Constructs a `Node` with allocated memory, assigns a value, appends
    # the pointer to `self.next`. Replaces any existing `next`.
    fn append(mut self, value: Self.ElementType):
        # Free chain if replacing `next`
        if self.next:
            self.next[].free_chain()
            self.next.destroy_pointee()
            self.next.free()

        self.next = Self.make_node(value)

    # Prints the list starting at this pointer's pointee
    @staticmethod
    fn print_list(node_ptr: Self.NodePointer):
        if not node_ptr:
            print("Empty list")
            return

        current_value: Optional[Self.ElementType] = node_ptr[].value
        if current_value:
            print(current_value.value(), end=" ")

        if node_ptr[].next:
            Self.print_list(node_ptr[].next)
        else:
            print()

    # Releases all successively allocated `Node` pointees. Does not release self.
    fn free_chain(self):
        current = self.next
        while current:
            next_node = current[].next
            current.destroy_pointee()
            current.free()
            current = next_node


fn main():
    values: List[Element] = ["one", "one", "two", "three", "five", "eight"]
    list_head = ListNode.make_node(values[0])

    current = list_head
    for idx in range(1, len(values), 1):
        current[].append(values[idx])
        current = current[].next

    ListNode.print_list(list_head)

    # Demonstrates cleanup. In short-lived programs, the OS reclaims memory at exit
    list_head[].free_chain()
    list_head.destroy_pointee()
    list_head.free()
