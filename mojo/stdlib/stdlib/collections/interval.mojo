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

"""
A self-balancing interval tree is a specialized binary search tree designed to
efficiently store and query intervals.

It maintains intervals sorted by their low endpoints and augments each node with a
`max_high` attribute, representing the maximum high endpoint in its subtree. This
`max_high` value enables efficient overlap searching by pruning the search space.
Self-balancing mechanisms, such as Red-Black or AVL trees, ensure logarithmic time
complexity for operations.

Key Features:
  - Stores intervals (low, high).
  - Nodes ordered by `low` endpoints.
  - `max_high` attribute at each node for efficient overlap search.
  - Self-balancing (e.g., using Red-Black tree logic) for O(log n) operations.

Operations:
  - Insertion: O(log n) - Adds a new interval, maintaining balance and updating
    `max_high`.
  - Overlap Search: O(log n) - Finds intervals overlapping a query interval using
    `max_high` for pruning.
  - Deletion: O(log n) - Removes an interval, maintaining balance and updating
    `max_high`.

Space Complexity: O(n), where n is the number of intervals.

Use Cases:
  - Calendar scheduling
  - Computational geometry
  - Genomics
  - Database indexing
  - Resource allocation

In essence, this data structure provides a fast and efficient way to manage and
query interval data, particularly for finding overlaps.
"""


from builtin.string_literal import StaticString
from memory import UnsafePointer

from .deque import Deque


trait IntervalElement(
    Copyable,
    Movable,
    Writable,
    Intable,
    Comparable,
):
    """The trait denotes a trait composition of the `Copyable`, `Movable`,
    `Writable`, `Intable`, and `Comparable` traits. Which is also subtractable.
    """

    fn __sub__(self, rhs: Self) -> Self:
        """Subtracts rhs from self, must be implemented in concrete types.

        Args:
            rhs: The value to subtract from self.

        Returns:
            The result of subtracting rhs from self.
        """
        ...


struct Interval[T: IntervalElement](Copyable, Movable, Boolable, Writable):
    """A half-open interval [start, end) that represents a range of values.

    The interval includes the start value but excludes the end value.

    Parameters:
        T: The type of the interval bounds.
    """

    var start: T
    """The inclusive start of the interval."""

    var end: T
    """The exclusive end of the interval."""

    fn __init__(out self, start: T, end: T):
        """Initialize an interval with start and end values.

        Args:
            start: The starting value of the interval.
            end: The ending value of the interval. Must be greater than or
              equal to start.
        """
        debug_assert(
            start <= end, "invalid interval '(", start, ", ", end, ")'"
        )
        self.start = start
        self.end = end

    fn __init__(out self, interval: Tuple[T, T], /):
        """Initialize an interval with a tuple of start and end values.

        Args:
            interval: A tuple containing the start and end values.
        """
        self.start = interval[0]
        self.end = interval[1]

    fn __copyinit__(out self, existing: Self, /):
        """Create a new instance of the interval by copying the values
        from an existing one.

        Args:
            existing: The interval to copy values from.
        """
        self.start = existing.start
        self.end = existing.end

    fn __moveinit__(out self, owned existing: Self, /):
        """Create a new instance of the interval by moving the values
        from an existing one.

        Args:
            existing: The interval to move values from.
        """
        self.start = existing.start^
        self.end = existing.end^

    fn overlaps(self, other: Self) -> Bool:
        """Returns whether this interval overlaps with another interval.

        Args:
            other: The interval to check for overlap with.

        Returns:
            True if the intervals overlap, False otherwise.
        """
        return other.start < self.end and other.end > self.start

    fn union(self, other: Self) -> Self:
        """Returns the union of this interval and another interval.

        Args:
            other: The interval to union with.

        Returns:
            The union of this interval and the other interval.
        """
        debug_assert(
            self.overlaps(other),
            "intervals do not overlap when computing the union of '",
            self,
            "' and '",
            other,
            "'",
        )
        var start = self.start if self.start < other.start else other.start
        var end = self.end if self.end > other.end else other.end
        return Self(start, end)

    fn intersection(self, other: Self) -> Self:
        """Returns the intersection of this interval and another interval.

        Args:
            other: The interval to intersect with.

        Returns:
            The intersection of this interval and the other interval.
        """
        debug_assert(
            self.overlaps(other),
            "intervals do not overlap when computing the intersection of '",
            self,
            "' and '",
            other,
            "'",
        )
        var start = self.start if self.start > other.start else other.start
        var end = self.end if self.end < other.end else other.end
        return Self(start, end)

    fn __contains__(self, other: T) -> Bool:
        """Returns whether a value is contained within this interval.

        Args:
            other: The value to check.

        Returns:
            True if the value is within the interval bounds, False otherwise.
        """
        return self.start <= other < self.end

    fn __contains__(self, other: Self) -> Bool:
        """Returns whether another interval is fully contained within this
        interval.

        Args:
            other: The interval to check.

        Returns:
            True if the other interval is fully contained within this interval,
            False otherwise.
        """
        return self.start <= other.start and self.end >= other.end

    fn __eq__(self, other: Self) -> Bool:
        """Returns whether this interval equals another interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if both intervals have the same start and end values.
        """
        return self.start == other.start and self.end == other.end

    fn __ne__(self, other: Self) -> Bool:
        """Returns whether this interval is not equal to another interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if the intervals are not equal, False if they are equal.
        """
        return not (self == other)

    fn __le__(self, other: Self) -> Bool:
        """Returns whether this interval is less than or equal to another
        interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's start is less than or equal to the other interval's start.
        """
        return self.start <= other.start

    fn __ge__(self, other: Self) -> Bool:
        """Returns whether this interval is greater than or equal to another
        interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's end is greater than or equal to the other interval's end.
        """
        return self.end >= other.end

    fn __lt__(self, other: Self) -> Bool:
        """Returns whether this interval is less than another interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's start is less than the other interval's start.
        """
        return self.start < other.start

    fn __gt__(self, other: Self) -> Bool:
        """Returns whether this interval is greater than another interval.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval's end is greater than the other interval's end.
        """
        return self.end > other.end

    fn __len__(self) -> Int:
        """Returns the length of this interval.

        Returns:
            The difference between end and start values as an integer.
        """
        debug_assert(Bool(self), "interval is empty")
        return Int(self.end - self.start)

    fn __bool__(self) -> Bool:
        """Returns whether this interval is empty.

        Returns:
            True if the interval is not empty (start < end), False otherwise.
        """
        return self.start < self.end

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Writes this interval to a writer in the format '(start, end)'.

        Parameters:
            W: The writer type that implements the Writer trait.

        Args:
            writer: The writer to write the interval to.
        """
        writer.write("(", self.start, ", ", self.end, ")")

    fn __str__(self) -> String:
        """Returns a string representation of this interval.

        Returns:
            A string in the format '(start, end)' representing this interval.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """Returns a string representation of this interval suitable for
        debugging.

        Returns:
            A string in the format '(start, end)' representing this interval.
        """
        return "Interval" + String.write(self) + ""


struct _IntervalNode[
    T: IntervalElement, U: Copyable & Movable & Stringable & Comparable
](Copyable, Movable, Stringable, Writable):
    """A node containing an interval and associated data.

    Parameters:
        T: The type of the interval bounds, must support subtraction, integer
          conversion, string conversion, comparison and collection operations.
        U: The type of the associated data, must support string conversion
          and collection operations.
    """

    var interval: Interval[T]
    """The interval contained in this node."""

    var data: U
    """The data associated with this interval."""

    var max_end: T
    """The maximum end value of this node."""

    var left: UnsafePointer[Self]
    """The left child of this node."""

    var right: UnsafePointer[Self]
    """The right child of this node."""

    var parent: UnsafePointer[Self]
    """The parent of this node."""

    var _is_red: Bool
    """Red-black node color."""

    fn __init__(
        out self,
        start: T,
        end: T,
        data: U,
        *,
        left: Optional[UnsafePointer[Self]] = None,
        right: Optional[UnsafePointer[Self]] = None,
        parent: Optional[UnsafePointer[Self]] = None,
        is_red: Bool = True,
    ):
        """Creates a new interval node.

        Args:
            start: The start value of the interval.
            end: The end value of the interval.
            data: The data to associate with this interval.
            left: The left child of this node.
            right: The right child of this node.
            parent: The parent of this node.
            is_red: Whether this node is red in the red-black tree.
        """
        self = Self(
            Interval(start, end),
            data,
            left=left,
            right=right,
            parent=parent,
            is_red=is_red,
        )

    fn __init__(
        out self,
        interval: Interval[T],
        data: U,
        *,
        left: Optional[UnsafePointer[Self]] = None,
        right: Optional[UnsafePointer[Self]] = None,
        parent: Optional[UnsafePointer[Self]] = None,
        is_red: Bool = True,
    ):
        """Creates a new interval node.

        Args:
            interval: The interval to associate with this node.
            data: The data to associate with this interval.
            left: The left child of this node.
            right: The right child of this node.
            parent: The parent of this node.
            is_red: Whether this node is red in the red-black tree.
        """
        self.interval = interval
        self.max_end = interval.end
        self.data = data
        self.left = left.value() if left else __type_of(self.left)()
        self.right = right.value() if right else __type_of(self.right)()
        self.parent = parent.value() if parent else __type_of(self.parent)()
        self._is_red = is_red

    fn __copyinit__(out self, existing: Self, /):
        """Create a new instance of the interval node by copying the values
        from an existing one.

        Args:
            existing: The interval node to copy values from.
        """
        self.interval = existing.interval
        self.data = existing.data
        self.max_end = existing.max_end
        self.left = existing.left
        self.right = existing.right
        self.parent = existing.parent
        self._is_red = existing._is_red

    fn __moveinit__(out self, owned existing: Self, /):
        """Create a new instance of the interval node by moving the values
        from an existing one.

        Args:
            existing: The interval node to move values from.
        """
        self.interval = existing.interval^
        self.data = existing.data^
        self.max_end = existing.max_end^
        self.left = existing.left
        self.right = existing.right
        self.parent = existing.parent
        self._is_red = existing._is_red

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Writes this interval node to a writer in the format
        '(start, end): data'.

        Parameters:
            W: The writer type that implements the Writer trait.

        Args:
            writer: The writer to write the interval node to.
        """
        writer.write(self.interval, "=", String(self.data))
        # writer.write(str(self.data))

    fn __str__(self) -> String:
        """Returns a string representation of this interval node.

        Returns:
            A string in the format '(start, end): data' representing this
            interval node.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """Returns a string representation of this interval node suitable for
        debugging.

        Returns:
            A string in the format '(start, end): data' representing this
            interval node.
        """
        return "IntervalNode(" + String.write(self) + ")"

    fn depth(self) -> Int:
        """Returns the depth of this interval node.

        Returns:
            The depth of this interval node.
        """
        var left_depth = self.left[].depth() if self.left else 0
        var right_depth = self.right[].depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    fn __bool__(self) -> Bool:
        """Returns whether this interval node is empty.

        Returns:
            True if the interval node is empty, False otherwise.
        """
        return Bool(self.interval)

    fn __eq__(self, other: Self) -> Bool:
        """Returns whether this interval node is equal to another interval node.

        Args:
            other: The interval node to compare with.

        Returns:
            True if the interval nodes are equal, False otherwise.
        """
        return self.interval == other.interval and self.data == other.data

    fn __lt__(self, other: Self) -> Bool:
        return self.interval < other.interval

    fn __gt__(self, other: Self) -> Bool:
        return self.interval > other.interval


struct IntervalTree[
    T: IntervalElement, U: Copyable & Movable & Stringable & Comparable
](Writable):
    """An interval tree data structure for efficient range queries.

    Parameters:
        T: The type of the interval bounds, must support subtraction, integer
          conversion, string conversion, comparison and collection operations.
        U: The type of the associated data, must support string conversion
          and collection operations.
    """

    var _root: UnsafePointer[_IntervalNode[T, U]]
    """The root node of the interval tree."""

    var _len: Int
    """The number of elements in the interval tree."""

    fn __init__(out self):
        """Initializes an empty IntervalTree."""
        self._root = __type_of(self._root)()
        self._len = 0

    fn _left_rotate(
        mut self, rotation_node: UnsafePointer[_IntervalNode[T, U]]
    ):
        """Performs a left rotation around node x in the red-black tree.

        This method performs a left rotation around the given node x, which is a
        standard operation in red-black trees used to maintain balance. The rotation
        preserves the binary search tree property while changing the structure.

        Before:          After:
             x            y
            / \\         / \\
           a   y   =>   x   c
              / \\      / \\
             b   c    a   b

        Args:
            rotation_node: A pointer to the node around which to perform the left rotation.

        Note:
            The rotation assumes that x has a right child. The method will assert if
            either the root or x's right child is not set.
        """
        debug_assert(Bool(self._root), "node is not set")
        var rotation_right_child = rotation_node[].right
        debug_assert(Bool(rotation_right_child), "right child is not set")
        rotation_node[].right = rotation_right_child[].left

        if rotation_right_child[].left:
            rotation_right_child[].left[].parent = rotation_node

        rotation_right_child[].parent = rotation_node[].parent

        if not rotation_node[].parent:
            self._root = rotation_right_child
        elif rotation_node == rotation_node[].parent[].left:
            rotation_node[].parent[].left = rotation_right_child
        else:
            rotation_node[].parent[].right = rotation_right_child

        rotation_right_child[].left = rotation_node
        rotation_node[].parent = rotation_right_child

        rotation_node[].max_end = rotation_node[].interval.end
        if rotation_node[].left:
            rotation_node[].max_end = max(
                rotation_node[].max_end,
                rotation_node[].left[].max_end,
            )
        if rotation_node[].right:
            rotation_node[].max_end = max(
                rotation_node[].max_end,
                rotation_node[].right[].max_end,
            )

        rotation_right_child[].max_end = rotation_right_child[].interval.end
        if rotation_right_child[].left:
            rotation_right_child[].max_end = max(
                rotation_right_child[].max_end,
                rotation_right_child[].left[].max_end,
            )
        if rotation_right_child[].right:
            rotation_right_child[].max_end = max(
                rotation_right_child[].max_end,
                rotation_right_child[].right[].max_end,
            )

    fn _right_rotate(
        mut self, rotation_node: UnsafePointer[_IntervalNode[T, U]]
    ):
        """Performs a right rotation around node y in the red-black tree.

        This method performs a right rotation around the given node y, which is a
        standard operation in red-black trees used to maintain balance. The rotation
        preserves the binary search tree property while changing the structure.

        Before:          After:
             y            x
            / \\         / \\
           x   c   =>   a   y
          / \\          / \\
         a   b        b   c

        Args:
            rotation_node: A pointer to the node around which to perform the right rotation.

        Note:
            The rotation assumes that y has a left child. The method will assert if
            either the root or y's left child is not set.
        """
        debug_assert(Bool(self._root), "root node is not set")
        var rotation_left_child = rotation_node[].left
        debug_assert(Bool(rotation_left_child), "left child node is not set")
        rotation_node[].left = rotation_left_child[].right

        if rotation_left_child[].right:
            rotation_left_child[].right[].parent = rotation_node

        rotation_left_child[].parent = rotation_node[].parent

        if not rotation_node[].parent:
            self._root = rotation_left_child
        elif rotation_node == rotation_node[].parent[].right:
            rotation_node[].parent[].right = rotation_left_child
        else:
            rotation_node[].parent[].left = rotation_left_child

        rotation_left_child[].right = rotation_node
        rotation_node[].parent = rotation_left_child

        rotation_node[].max_end = rotation_node[].interval.end
        if rotation_node[].left:
            rotation_node[].max_end = max(
                rotation_node[].max_end, rotation_node[].left[].max_end
            )
        if rotation_node[].right:
            rotation_node[].max_end = max(
                rotation_node[].max_end, rotation_node[].right[].max_end
            )

        rotation_left_child[].max_end = rotation_left_child[].interval.end
        if rotation_left_child[].left:
            rotation_left_child[].max_end = max(
                rotation_left_child[].max_end,
                rotation_left_child[].left[].max_end,
            )

    fn insert(mut self, interval: Tuple[T, T], data: U):
        """Insert a new interval into the tree using a tuple representation.

        Args:
            interval: A tuple containing the start and end values of the interval.
            data: The data value to associate with this interval.
        """
        self.insert(Interval(interval[0], interval[1]), data)

    fn insert(mut self, interval: Interval[T], data: U):
        """Insert a new interval into the tree.

        This method inserts a new interval and its associated data into the interval tree.
        It maintains the binary search tree property based on interval start times and
        updates the tree structure to preserve red-black tree properties.

        Args:
            interval: The interval to insert into the tree.
            data: The data value to associate with this interval.
        """
        # Allocate memory for a new node and initialize it with the interval
        # and data
        var new_node = UnsafePointer[_IntervalNode[T, U]].alloc(1)
        new_node.init_pointee_move(_IntervalNode(interval, data))
        self._len += 1

        # If the tree is empty, set the root to the new node and color it black.
        if not self._root:
            self._root = new_node
            self._root[]._is_red = False
            return

        # Find the insertion point by traversing down the tree
        # parent_node tracks the parent of the current node
        var parent_node = __type_of(self._root)()
        # current_node traverses down the tree until we find an empty spot
        var current_node = self._root
        while current_node:
            parent_node = current_node
            if new_node[] < current_node[]:
                current_node = current_node[].left
            else:
                current_node = current_node[].right
            parent_node[].max_end = max(
                parent_node[].max_end, new_node[].interval.end
            )

        new_node[].parent = parent_node
        if not parent_node:
            self._root = new_node
        elif new_node[] < parent_node[]:
            parent_node[].left = new_node
        else:
            parent_node[].right = new_node

        self._insert_fixup(new_node)

    fn _insert_fixup(
        mut self, current_node0: UnsafePointer[_IntervalNode[T, U]]
    ):
        """Fixes up the red-black tree properties after an insertion.

        This method restores the red-black tree properties that may have been violated
        during insertion of a new node. It performs rotations and color changes to
        maintain the balance and color properties of the red-black tree.

        Args:
            current_node0: A pointer to the newly inserted node that may violate red-black
                properties.
        """
        var current_node = current_node0

        # While the parent of the current node is red, we need to fix violations
        while current_node != self._root and current_node[].parent[]._is_red:
            if current_node[].parent == current_node[].parent[].parent[].left:
                # Get uncle node (parent's sibling)
                var uncle_node = current_node[].parent[].parent[].right
                if uncle_node and uncle_node[]._is_red:
                    # Case 1: Uncle is red - recolor parent, uncle and grandparent
                    current_node[].parent[]._is_red = False
                    uncle_node[]._is_red = False
                    current_node[].parent[].parent[]._is_red = True
                    current_node = current_node[].parent[].parent
                else:
                    # Case 2: Uncle is black and node is a right child
                    if current_node == current_node[].parent[].right:
                        current_node = current_node[].parent
                        self._left_rotate(current_node)
                    # Case 3: Uncle is black and node is a left child
                    current_node[].parent[]._is_red = False
                    current_node[].parent[].parent[]._is_red = True
                    self._right_rotate(current_node[].parent[].parent)
            else:
                # Mirror case - parent is right child of grandparent
                var uncle_node = current_node[].parent[].parent[].left
                if uncle_node and uncle_node[]._is_red:
                    # Case 1: Uncle is red - recolor
                    current_node[].parent[]._is_red = False
                    uncle_node[]._is_red = False
                    current_node[].parent[].parent[]._is_red = True
                    current_node = current_node[].parent[].parent
                else:
                    # Case 2: Uncle is black and node is a left child
                    if current_node == current_node[].parent[].left:
                        current_node = current_node[].parent
                        self._right_rotate(current_node)

        # Ensure root is black to maintain red-black tree properties
        self._root[]._is_red = False

    fn __str__(self) -> String:
        """Returns a string representation of the interval tree.

        Returns:
            A string representation of the interval tree.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """Returns a string representation of the interval tree suitable for
        debugging.

        Returns:
            A string representation of the interval tree.
        """
        return String.write(self)

    fn write_to[w: Writer](self, mut writer: w):
        """Writes the interval tree to a writer.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to write the interval tree to.
        """
        self._draw(writer)

    @no_inline
    fn _draw[w: Writer](self, mut writer: w):
        """Draws the interval tree in a simple ASCII tree format.

        Creates a text representation of the tree using ASCII characters, with each node
        indented according to its depth. Uses '├─' and '└─' characters to show the tree
        structure.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
        """
        self._draw_helper(writer, self._root, "", True)

    @no_inline
    fn _draw_helper[
        w: Writer
    ](
        self,
        mut writer: w,
        node: UnsafePointer[_IntervalNode[T, U]],
        indent: String,
        is_last: Bool,
    ):
        """Helper function to recursively draw the interval tree.

        Recursively traverses the tree and draws each node with proper indentation
        and branch characters to show the tree structure.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
            node: The current node being drawn.
            indent: The current indentation string.
            is_last: Whether this node is the last child of its parent.
        """
        # Handle empty tree case
        if not node:
            return

        writer.write(indent)
        var next_indent = indent
        if is_last:
            writer.write("├─")
            next_indent += "  "
        else:
            writer.write("└─")
            next_indent += "| "
        writer.write(node[], "\n")
        # Recursively draw left and right subtrees
        self._draw_helper(writer, node[].left, next_indent, False)
        self._draw_helper(writer, node[].right, next_indent, True)

    @no_inline
    fn _draw3[w: Writer](self, mut writer: w) raises:
        """Draws the interval tree in a simple ASCII tree format.

        Creates a text representation of the tree using ASCII characters, with each node
        indented according to its depth. Uses '├─' and '└─' characters to show the tree
        structure.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
        """
        # Handle empty tree case
        if not self._root:
            return writer.write("Empty")

        var work_list = Deque[
            Tuple[UnsafePointer[_IntervalNode[T, U]], String, Bool]
        ]()
        work_list.append((self._root, String(), True))

        while work_list:
            var node, indent, is_last = work_list.pop()
            if not node:
                continue
            writer.write(indent)
            if is_last:
                writer.write("├─ ")
                indent += "   "
            else:
                writer.write("└─ ")
                indent += "|  "
            writer.write(node[], "\n")
            work_list.append((node[].left, indent, False))
            work_list.append((node[].right, indent, True))

    @no_inline
    fn _draw2[w: Writer](self, mut writer: w) raises:
        """Draws the interval tree in a visual ASCII art format.

        Creates a grid representation of the tree with nodes and connecting branches.
        Each level of the tree is separated by 3 rows vertically.
        Nodes are connected by '/' and '\' characters for left and right children.

        Parameters:
            w: The writer type that implements the Writer trait.

        Args:
            writer: The writer to output the tree visualization to.
        """
        # Handle empty tree case
        if not self._root:
            return writer.write("Empty")

        # Calculate dimensions needed for the grid
        var height = self._root[].depth()
        var width = 2**height - 1

        # Create 2D grid of spaces to hold the tree visualization
        # Each row is a list of single character strings
        var grid = List[List[String]]()
        for _ in range(3 * height):
            var row = List[String]()
            for _ in range(4 * width):
                row.append(" ")  # Initialize with spaces
            grid.append(row)

        var work_list = Deque[
            Tuple[UnsafePointer[_IntervalNode[T, U]], Int, Int, Int]
        ]()
        work_list.append((self._root, 0, 0, width))

        while work_list:
            # Recursively fills the grid with node values and connecting branches.
            var node, level, left, right = work_list.pop()
            if not node:
                continue

            # Calculate position for current node
            var mid = (left + right) // 2  # Center point between boundaries
            var pos_x = mid * 4  # Scale x position for readability
            var pos_y = level * 3  # Scale y position for branch drawing

            # Draw the current node's value
            var node_str = String(node[])
            var start_pos = max(
                0, pos_x - len(node_str) // 2
            )  # Center the node text
            var i = 0
            for char in node_str.codepoints():
                grid[pos_y][start_pos + i] = String(char)
                i += 1

            # Add drawing left branch to the worklist.
            if node[].left:
                for y in range(1, 3):
                    grid[pos_y + y][pos_x - 2 * y + 1] = "/"  # Draw left branch
                work_list.append((node[].left, level + 1, left, mid))

            # Add drawing right branch to the worklist.
            if node[].right:
                for y in range(1, 3):
                    grid[pos_y + y][pos_x + 2 * y] = "\\"  # Draw right branch
                work_list.append((node[].right, level + 1, mid, right))

        # Output the completed grid row by row
        for row in grid:
            var row_str = String(StaticString("").join(row).rstrip())
            if row_str:
                writer.write(row_str, "\n")

    fn depth(self) -> Int:
        """Returns the depth of the interval tree.

        Returns:
            The depth of the interval tree.
        """
        if not self._root:
            return 0

        return self._root[].depth()

    fn transplant(
        mut self,
        mut u: UnsafePointer[_IntervalNode[T, U]],
        mut v: UnsafePointer[_IntervalNode[T, U]],
    ):
        """Transplants the subtree rooted at node u with the subtree rooted at node v.

        Args:
            u: The node to transplant.
            v: The node to transplant to.
        """
        if not u[].parent:
            self._root = v
        elif u == u[].parent[].left:
            u[].parent[].left = v
        else:
            u[].parent[].right = v

        if v:
            v[].parent = u[].parent

    fn search(self, interval: Tuple[T, T]) raises -> List[U]:
        """Searches for intervals overlapping with the given tuple.

        Args:
            interval: The interval tuple (start, end).

        Returns:
            A list of data associated with overlapping intervals.
        """
        return self.search(Interval(interval[0], interval[1]))

    fn search(self, interval: Interval[T]) raises -> List[U]:
        """Searches for intervals overlapping with the given interval.

        Args:
            interval: The interval to search.

        Returns:
            A list of data associated with overlapping intervals.
        """
        return self._search_helper(self._root, interval)

    fn _search_helper(
        self, node: UnsafePointer[_IntervalNode[T, U]], interval: Interval[T]
    ) raises -> List[U]:
        var result = List[U]()
        var work_list = Deque[UnsafePointer[_IntervalNode[T, U]]]()
        work_list.append(node)

        while work_list:
            var current_node = work_list.pop()
            if not current_node:
                continue
            if current_node[].interval.overlaps(interval):
                result.append(current_node[].data)
            if (
                current_node[].left
                and current_node[].left[].interval.start <= interval.end
            ):
                work_list.append(current_node[].left)
            if (
                current_node[].right
                and current_node[].right[].max_end >= interval.start
            ):
                work_list.append(current_node[].right)

        return result
