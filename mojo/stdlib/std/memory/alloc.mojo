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
"""Implements layout-aware memory allocation and deallocation.

This module provides `alloc` and `free` functions that pair with a `Layout`
descriptor to express the size and alignment of an allocation at the call site,
making ownership and layout information explicit and co-located.
"""

from std.format._utils import FormatStruct, Named, TypeNames
from std.memory.memory import _free, _malloc
from std.os import abort
from std.sys import align_of, size_of
from std.sys.intrinsics import unlikely


def _alloc_bytes(
    layout: Layout[Byte],
) -> UnsafePointer[Byte, MutExternalOrigin]:
    var pointer = _malloc[Byte](layout.count(), alignment=layout.alignment())
    if unlikely(not pointer):
        abort("alloc failed: returned a null pointer")
    return pointer.unsafe_value()


@always_inline
def alloc[
    T: AnyType, //
](layout: Layout[T]) -> UnsafePointer[T, MutExternalOrigin]:
    """Allocates contiguous storage for `layout.count()` elements of `T`.

    The allocation uses the alignment specified by `layout`. Use `free` with the
    same `Layout` to release the memory.

    Parameters:
        T: The type of the elements to allocate storage for.

    Args:
        layout: Describes the number of elements and alignment of the allocation.

    Returns:
        A pointer to the newly allocated uninitialized storage.

    Constraints:
        `size_of[T]()` must be greater than zero. `layout.count()` must be
        greater than zero.

    Example:

    ```mojo
    from std.memory.alloc import *

    var layout = Layout[Int32](count=4, alignment=64)

    var ptr = alloc(layout)
    for i in range(layout.count()):
        (ptr + i).init_pointee_move(i)

    free(ptr, layout)
    ```
    """
    comptime size_of_t = size_of[T]()
    comptime type_name = reflect[T].name()
    comptime assert size_of_t > 0, String(
        "Mojo's alloc cannot handle zero-sized types: ", type_name
    )

    # TODO: Cannot use t-string as is causes a recursive reference to `alloc`
    debug_assert(layout.count() > 0, "alloc(", layout, "): count must be > 0")

    return _alloc_bytes(layout.as_byte_layout()).bitcast[T]()


@always_inline
def free[
    T: AnyType, //
](pointer: UnsafePointer[T, MutExternalOrigin], /, layout: Layout[T]):
    """Frees memory previously allocated with `alloc`.

    The `layout` argument must match the one passed to the corresponding `alloc`
    call. Passing a mismatched layout or a pointer not obtained from `alloc` is
    undefined behavior.

    Parameters:
        T: The type of the elements in the allocation.

    Args:
        pointer: A pointer returned by a previous call to `alloc`.
        layout: The layout used when the memory was allocated.

    Safety:

    - `pointer` must have been returned by a matching call to `alloc`.
    - `layout` must equal the layout passed to that `alloc` call.

    Example:

    ```mojo
    var layout = Layout[String].single()
    var ptr = alloc(layout)

    ptr.init_pointee_move("Dynamic allocation!")
    ptr.destroy_pointee()

    free(ptr, layout)
    ```
    """
    _free(pointer)


struct Layout[T: AnyType](TrivialRegisterPassable, Writable):
    """Describes the shape of a memory allocation for elements of type `T`.

    A `Layout` bundles the *count* of elements and the *alignment* of the
    allocation into a single value. Passing a `Layout` to `alloc` and `free`
    keeps the size and alignment requirements explicit and co-located at every
    call site, preventing mismatches between allocation and deallocation.

    Parameters:
        T: The element type the layout describes.

    Example:

    ```mojo
    from std.memory.alloc import alloc, free, Layout

    # Allocate room for 8 Int32 values with default alignment.
    var layout = Layout[Int32](count=8)
    var ptr = alloc(layout)
    # ... use ptr ...
    free(ptr, layout)
    ```
    """

    var _count: Int
    var _alignment: Int

    @always_inline
    @doc_hidden
    def __init__(out self, *, count: Int, unsafe_unchecked_alignment: Int):
        self._count = count
        self._alignment = unsafe_unchecked_alignment

    @always_inline
    def __init__(out self, *, count: Int):
        """Initializes a `Layout` with the given element count and a default alignment.

        Args:
            count: Number of elements of type `T` to describe.
        """
        self = Self(count=count, unsafe_unchecked_alignment=align_of[Self.T]())

    @always_inline
    def __init__(out self, *, count: Int, alignment: Int):
        """Initializes a `Layout` with the given element count and alignment.

        This method will abort if the alignment is invalid.

        Args:
            count: Number of elements of type `T` to describe.
            alignment: Byte alignment of the allocation. Must be a power of two.
        """
        if not Self.is_valid_alignment(alignment):
            abort(
                "Alignment is invalid. Must be a power of two and >= to the"
                " types natural alignment."
            )
        self = Self(count=count, unsafe_unchecked_alignment=alignment)

    @always_inline
    @staticmethod
    def aligned[alignment: Int](*, count: Int) -> Self:
        """Initializes a `Layout` with the given element count and comptime alignment.

        Unlike `Layout[T](count, alignment)`, this validates alignment at compile time.

        Parameters:
            alignment: Byte alignment of the allocation. Must be a power of two.

        Args:
            count: Number of elements of type `T` to describe.

        Returns:
            A `Layout` with the specified `count` and `alignment`.
        """
        comptime assert alignment.is_power_of_two(), String(
            "alignment '", alignment, "' is not a power of two"
        )
        comptime assert alignment >= align_of[Self.T](), String(
            "alignment '",
            alignment,
            "' must be at least align_of[",
            reflect[Self.T].name(),
            "]() '",
            align_of[Self.T](),
            "'",
        )
        return Self(count=count, unsafe_unchecked_alignment=alignment)

    @always_inline
    @staticmethod
    def single() -> Self:
        """Creates a `Layout` for exactly one element of type `T`.

        Returns:
            A `Layout` with `count` equal to 1 and default alignment.

        Example:

        ```mojo
        from std.memory.alloc import alloc, free, Layout

        var layout = Layout[Int64].single()
        var ptr = alloc(layout)
        ptr.init_pointee_move(0)
        free(ptr, layout)
        ```
        """
        return Self(count=1)

    @always_inline
    def as_byte_layout(self) -> Layout[Byte]:
        """Converts this layout to an equivalent byte-level layout.

        Multiplies the element count by `size_of[T]()` to express the same
        allocation in terms of raw bytes, preserving the alignment.

        Returns:
            A `Layout[Byte]` whose `count` is `self.count() * size_of[T]()` and
            whose `alignment` matches `self.alignment()`.
        """
        return Layout[Byte](
            count=self._count * size_of[Self.T](),
            unsafe_unchecked_alignment=self._alignment,
        )

    @always_inline
    def alignment(self) -> Int:
        """Returns the alignment of the allocation described by this layout.

        Returns:
            The byte alignment.
        """
        return self._alignment

    @always_inline
    def count(self) -> Int:
        """Returns the number of elements described by this layout.

        Returns:
            The element count passed at construction time.
        """
        return self._count

    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable representation of this layout to `writer`.

        Args:
            writer: The writer to write to.
        """
        FormatStruct(writer, "Layout").params(reflect[Self.T].name()).fields(
            Named("count", self._count), Named("alignment", self._alignment)
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of this layout to `writer`.

        Args:
            writer: The writer to write to.
        """
        self.write_to(writer)

    @staticmethod
    @always_inline("builtin")
    def is_valid_alignment(alignment: Int) -> Bool:
        """Reports whether `alignment` is a valid alignment for `Layout[T]`.

        An alignment is valid when it is a power of two and is at least the
        natural alignment of `T` (`align_of[T]()`). Under-aligning `T` would
        violate its layout requirements, so requested alignments must meet or
        exceed the natural alignment.

        Args:
            alignment: The candidate byte alignment to check.

        Returns:
            `True` if `alignment` is a power of two and is no smaller than
            `align_of[T]()`, otherwise `False`.

        Example:

        ```mojo
        from std.memory.alloc import Layout

        var ok = Layout[Int32].is_valid_alignment(64)  # True (over-aligned)
        var not_pow2 = Layout[Int32].is_valid_alignment(33)  # False
        var too_small = Layout[Int32].is_valid_alignment(4)  # False
        ```
        """
        return alignment.is_power_of_two() and alignment >= align_of[Self.T]()
