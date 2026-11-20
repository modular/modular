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
"""Implements slice.

These are Mojo built-ins, so you don't need to import them.
"""

from std.format._utils import FormatStruct, Named


struct Slice(
    Equatable,
    ImplicitlyCopyable,
    Writable,
):
    """Represents a slice expression.

    Objects of this type are generated when slice syntax is used within square
    brackets, for example:

    ```mojo
    var lst: List[Int] = [0, 1, 2, 3, 4, 5, 6, 7]

    # Both produce the same new list: [6, 7].
    var l1 = List(lst[6:])
    var l2 = lst.__getitem__(Slice(6, len(lst)))
    ```
    """

    # Fields
    var start: Optional[Int]
    """The starting index of the slice."""
    var end: Optional[Int]
    """The end index of the slice."""
    var step: Optional[Int]
    """The step increment value of the slice."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    def __init__(out self, start: Int, end: Int):
        """Constructs a slice with the given start and end bounds.

        Args:
            start: The starting index of the slice.
            end: The end index of the slice.
        """
        self.start = start
        self.end = end
        self.step = None

    @inline(.always)
    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        step: Optional[Int],
        __slice_literal__: NoneType = None,
    ):
        """Constructs a slice with the given start, end, and step values.

        Args:
            start: The starting index of the slice.
            end: The end index of the slice.
            step: The step increment value of the slice.
            __slice_literal__: Enables slice literal syntax.
        """
        self.start = start
        self.end = end
        self.step = step

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        """Writes the slice's string representation to a writer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "Slice").fields(self.start, self.end, self.step)

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the slice's debug representation to a writer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "Slice").fields(
            Named("start", self.start),
            Named("end", self.end),
            Named("step", self.step),
        )

    @inline(.always)
    def __eq__(self, other: Self) -> Bool:
        """Compare this slice to the other.

        Args:
            other: The slice to compare to.

        Returns:
            True if start, end, and step values of this slice match the
            corresponding values of the other slice and False otherwise.
        """
        return (
            self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

    def indices(self, length: Int) -> Tuple[Int, Int, Int]:
        """Resolves the slice's bounds against the length of a container.

        Uses the target container length to resolve negative, out-of-bounds,
        or `None` bounds.

        A negative bound counts back from the end of the container:
        ```mojo
        var s = slice(0, -1, 1)
        print(s.indices(5))  # (0, 4, 1)
        ```

        A `None` bound becomes the start or the end of the container based
        on the direction of the step:
        ```mojo
        var s = slice(None, None, 1)
        print(s.indices(5))  # (0, 5, 1)
        ```

        An out-of-bounds bound is clamped to the nearest valid bound in the
        step's direction:
        ```mojo
        var s = slice(20)
        print(s.indices(5))  # (0, 5, 1)
        ```

        Args:
            length: The length of the container the slice applies to.

        Returns:
            A tuple of the resolved start, end, and step.
        """
        var step = self.step.or_else(1)

        var positive_step = step > 0

        var start: Int
        match self.start:
            case .None:
                start = 0 if positive_step else length - 1
            case .Some(s) if s < 0:
                start = s + length
                if start < 0:
                    start = 0 if positive_step else -1
            case .Some(s) if s >= length:
                start = length if positive_step else length - 1
            case .Some(s):
                start = s

        var end: Int
        match self.end:
            case .None:
                end = length if positive_step else -1
            case .Some(e) if e < 0:
                end = e + length
                if end < 0:
                    end = 0 if positive_step else -1
            case .Some(e) if e >= length:
                end = length if positive_step else length - 1
            case .Some(e):
                end = e

        return (start, end, step)


struct StridedSlice(ImplicitlyCopyable, Writable):
    """Represents a slice expression that has a stride.

    This type is used to support different behavior for strided vs unstrided
    slicing.
    """

    var _inner: Slice

    @implicit
    def __init__(out self, other: Slice):
        """Implicitly convert from a general slice.

        Args:
            other: The other slice.
        """
        self._inner = other

    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        stride: Int,
        __slice_literal__: NoneType = None,
    ):
        """Constructs a strided slice with the given start, end, and stride.

        Args:
            start: The starting index of the slice.
            end: The end index of the slice.
            stride: The step increment value of the slice.
            __slice_literal__: Enables slice literal syntax.
        """

        self._inner = Slice(start, end, stride)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        """Writes the strided slice's string representation to a writer.

        Args:
            writer: The object to write to.
        """
        self._inner.write_to(writer)

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the strided slice's debug representation to a writer.

        Args:
            writer: The object to write to.
        """
        self._inner.write_repr_to(writer)

    def indices(self, length: Int) -> Tuple[Int, Int, Int]:
        """Resolves the slice's bounds against the length of a container.

        Bounds resolve with the same rules as `Slice.indices()`.

        Args:
            length: The length of the container the slice applies to.

        Returns:
            A tuple of the resolved start, end, and step.
        """
        return self._inner.indices(length)


struct ContiguousSlice(ImplicitlyCopyable, Writable):
    """Represents a slice expression without a stride.

    This type is used to support different behavior for strided vs unstrided
    slicing.
    """

    var start: Optional[Int]
    """The starting index of the slice."""
    var end: Optional[Int]
    """The end index of the slice."""

    @inline(.always)
    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        stride: NoneType,
        __slice_literal__: NoneType = None,
    ):
        """Constructs a contiguous slice with the given start and end.

        Args:
            start: The starting index of the slice.
            end: The end index of the slice.
            stride: Always none. Disambiguates from slices with a stride.
            __slice_literal__: Enables slice literal syntax.
        """
        self.start = start
        self.end = end

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        """Writes the contiguous slice's string representation to a writer.

        Args:
            writer: The object to write to.
        """
        Slice(self.start, self.end, None).write_to(writer)

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the contiguous slice's debug representation to a writer.

        Args:
            writer: The object to write to.
        """
        Slice(self.start, self.end, None).write_repr_to(writer)

    def indices(self, length: Int) -> Tuple[Int, Int]:
        """Resolves the slice's start and end against the length of a
        container.

        A negative bound counts back from `length`, a `None` bound becomes
        the start or the end, and an out-of-bounds bound is clamped to the
        range `[0, length]`.

        Args:
            length: The length of the container the slice applies to.

        Returns:
            A tuple of the resolved start and end.
        """
        var start = self.start.or_else(0)
        var end = self.end.or_else(length)
        if start < 0:
            start = max(start + length, 0)
        elif start >= length:
            start = length

        if end < 0:
            end = max(end + length, 0)
        elif end >= length:
            end = length

        return start, end


# ===-----------------------------------------------------------------------===#
# Slice constructor functions
# ===-----------------------------------------------------------------------===#


@inline(.always)
def slice(end: Int) -> Slice:
    """Constructs a slice from the beginning up to `end`.

    A `Slice` holds the bounds that bracket slice syntax like `lst[1:5]`
    expresses. Constructing one directly is useful when the bounds are
    computed at run time. The end bound is exclusive.

    `indices()` resolves the bound against a container's length and counts a
    negative bound back from the end. Subscripting without a step is stricter:
    its bounds must be non-negative, and `lst[:-1]` aborts when bounds
    checking is enabled (the default on CPU) and causes undefined behavior
    otherwise. Compute the bound from the container's length instead, like
    `lst[: len(lst) - 1]` to stop one element short of the end.

    Args:
        end: The exclusive upper bound.

    Returns:
        A `Slice` covering the indices `0` up to `end`.

    For example:

    ```mojo
    var numbers: List[Int] = [10, 20, 30, 40, 50]
    var from_start = slice(3)
    print(numbers[from_start])  # [10, 20, 30]
    print(from_start.indices(5))  # (0, 3, 1)
    ```
    """
    return Slice(None, end, None)


@inline(.always)
def slice(start: Int, end: Int) -> Slice:
    """Constructs a slice from `start` up to `end`.

    The bounds are half-open: the slice covers `start` up to, but not
    including, `end`. `indices()` resolves the bounds against a container's
    length and counts negative bounds back from the end. Subscripting without
    a step is stricter: its bounds must be non-negative, and `lst[1:-1]`
    aborts when bounds checking is enabled (the default on CPU) and causes
    undefined behavior otherwise. Compute bounds from the container's length
    instead, like `lst[1 : len(lst) - 1]` to stop one element short of the end.

    Args:
        start: The inclusive lower bound.
        end: The exclusive upper bound.

    Returns:
        A `Slice` covering the indices `start` up to `end`.

    For example:

    ```mojo
    var numbers: List[Int] = [10, 20, 30, 40, 50]
    var s = slice(1, len(numbers) - 1)

    # Slicing the list copies the covered elements into a new list.
    var middle = numbers[s]
    print(middle)  # [20, 30, 40]

    # The bounds also resolve on their own, without a container.
    print(s.indices(len(numbers)))  # (1, 4, 1)
    ```
    """
    return Slice(start, end)


@inline(.always)
def slice(
    start: Optional[Int], end: Optional[Int], step: Optional[Int]
) -> Slice:
    """Constructs a slice from `start` to `end` stepping by `step`.

    The bounds are half-open. A `None` bound means the beginning or the end
    of the container, in the direction the step moves, and a `None` step
    means 1. `indices()` resolves the bounds against a container's length
    and counts negative bounds back from the end. Step-less slicing doesn't
    resolve negatives: its bounds must be non-negative, so compute them from
    the container's length instead.

    Args:
        start: The inclusive lower bound, or `None` for the beginning in
            the step's direction.
        end: The exclusive upper bound, or `None` for the end in the step's
            direction.
        step: The stride between the covered indices, where `None` means 1.

    Returns:
        A `Slice` covering the indices `start` to `end` by `step`.

    For example:

    ```mojo
    var numbers: List[Int] = [10, 20, 30, 40, 50]
    var backwards = slice(None, None, -1)
    print(numbers[backwards])  # [50, 40, 30, 20, 10]
    print(backwards.indices(5))  # (4, -1, -1)
    ```
    """
    return Slice(start, end, step)
