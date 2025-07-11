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

"""Implements the `Span` type.

You can import these APIs from the `memory` module. For example:

```mojo
from memory import Span
```
"""

from sys.info import simdwidthof

from memory import Pointer
from memory.unsafe_pointer import _default_alignment


@fieldwise_init
struct _SpanIter[
    mut: Bool, //,
    T: Copyable & Movable,
    origin: Origin[mut],
    forward: Bool = True,
    address_space: AddressSpace = AddressSpace.GENERIC,
    alignment: Int = _default_alignment[T](),
](Copyable, Movable):
    """Iterator for Span.

    Parameters:
        mut: Whether the reference to the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the `Span`.
        forward: The iteration direction. False is backwards.
        address_space: The address space associated with the underlying allocated memory.
        alignment: The minimum alignment of the underlying pointer known statically.

    """

    var index: Int
    var src: Span[T, origin, address_space=address_space, alignment=alignment]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next_ref__(mut self) -> ref [origin, address_space] T:
        @parameter
        if forward:
            self.index += 1
            return self.src[self.index - 1]
        else:
            self.index -= 1
            return self.src[self.index]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src) - self.index
        else:
            return self.index


@fieldwise_init
@register_passable("trivial")
struct Span[
    mut: Bool, //,
    T: Copyable & Movable,
    origin: Origin[mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    alignment: Int = _default_alignment[T](),
](ExplicitlyCopyable, Copyable, Movable, Sized, Boolable, Defaultable):
    """A non-owning view of contiguous data.

    Parameters:
        mut: Whether the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the Span.
        address_space: The address space associated with the allocated memory.
        alignment: The minimum alignment of the underlying pointer known statically.
    """

    # Aliases
    alias Mutable = Span[T, MutableOrigin.cast_from[origin]]
    """The mutable version of the `Span`."""
    alias Immutable = Span[T, ImmutableOrigin.cast_from[origin]]
    """The immutable version of the `Span`."""
    # Fields
    var _data: UnsafePointer[
        T,
        mut=mut,
        origin=origin,
        address_space=address_space,
        alignment=alignment,
    ]
    var _len: Int

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(out self):
        """Create an empty / zero-length span."""
        self._data = {}
        self._len = 0

    @doc_private
    @implicit
    @always_inline("nodebug")
    fn __init__(
        other: Span[T, _],
        out self: Span[T, ImmutableOrigin.cast_from[other.origin]],
    ):
        """Implicitly cast the mutable origin of self to an immutable one.

        Args:
            other: The Span to cast.
        """
        self = rebind[__type_of(self)](other)

    @always_inline("builtin")
    fn __init__(
        out self,
        *,
        ptr: UnsafePointer[
            T,
            address_space=address_space,
            alignment=alignment,
            mut=mut,
            origin=origin, **_,
        ],
        length: UInt,
    ):
        """Unsafe construction from a pointer and length.

        Args:
            ptr: The underlying pointer of the span.
            length: The length of the view.
        """
        self._data = ptr
        self._len = length

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of the provided `Span`.

        Returns:
            A copy of the `Span`.
        """
        return self

    @always_inline
    @implicit
    fn __init__(out self, ref [origin, address_space]list: List[T, *_]):
        """Construct a `Span` from a `List`.

        Args:
            list: The list to which the span refers.
        """
        self._data = (
            list.data.address_space_cast[address_space]()
            .static_alignment_cast[alignment]()
            .origin_cast[mut, origin]()
        )
        self._len = list._len

    @always_inline
    @implicit
    fn __init__[
        size: Int, //
    ](out self, ref [origin]array: InlineArray[T, size]):
        """Construct a `Span` from an `InlineArray`.

        Parameters:
            size: The size of the `InlineArray`.

        Args:
            array: The array to which the span refers.
        """

        self._data = (
            UnsafePointer(to=array)
            .bitcast[T]()
            .address_space_cast[address_space]()
            .static_alignment_cast[alignment]()
            .origin_cast[mut, origin]()
        )
        self._len = size

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> ref [origin, address_space] T:
        """Get a reference to an element in the span.

        Args:
            idx: The index of the value to return.

        Parameters:
            I: A type that can be used as an index.

        Returns:
            An element reference.
        """
        # TODO: Simplify this with a UInt type.
        debug_assert(
            -self._len <= Int(idx) < self._len, "index must be within bounds"
        )
        # TODO(MSTDL-1086): optimize away SIMD/UInt normalization check
        var offset = Int(idx)
        if offset < 0:
            offset += len(self)
        return self._data[offset]

    @always_inline
    fn __getitem__(self, slc: Slice) -> Self:
        """Get a new span from a slice of the current span.

        Args:
            slc: The slice specifying the range of the new subslice.

        Returns:
            A new span that points to the same data as the current span.

        Allocation:
            This function allocates when the step is negative, to avoid a memory
            leak, take ownership of the value.
        """
        var start, end, step = slc.indices(len(self))

        # TODO: Introduce a new slice type that just has a start+end but no
        # step.  Mojo supports slice type inference that can express this in the
        # static type system instead of debug_assert.
        debug_assert(step == 1, "Slice step must be 1")

        return Self(
            ptr=(self._data + start), length=len(range(start, end, step))
        )

    @always_inline
    fn __iter__(
        self,
    ) -> _SpanIter[
        T,
        origin,
        address_space=address_space,
        alignment=alignment,
    ]:
        """Get an iterator over the elements of the `Span`.

        Returns:
            An iterator over the elements of the `Span`.
        """
        return _SpanIter[
            address_space=address_space,
            alignment=alignment,
        ](0, self)

    @always_inline
    fn __reversed__(
        self,
    ) -> _SpanIter[
        T,
        origin,
        forward=False,
        address_space=address_space,
        alignment=alignment,
    ]:
        """Iterate backwards over the `Span`.

        Returns:
            A reversed iterator of the `Span` elements.
        """
        return _SpanIter[
            forward=False, address_space=address_space, alignment=alignment
        ](len(self), self)

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __len__(self) -> Int:
        """Returns the length of the span. This is a known constant value.

        Returns:
            The size of the span.
        """
        return self._len

    fn __contains__[
        dtype: DType, //
    ](
        self: Span[
            Scalar[dtype],
            origin,
            address_space=address_space,
            alignment=alignment,
        ],
        value: Scalar[dtype],
    ) -> Bool:
        """Verify if a given value is present in the Span.

        Parameters:
            dtype: The DType of the scalars stored in the Span.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the list, False otherwise.
        """

        alias widths = InlineArray[Int, 6](256, 128, 64, 32, 16, 8)
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var processed = 0

        @parameter
        for i in range(len(widths)):
            alias width = widths[i]

            @parameter
            if simdwidthof[dtype]() >= width:
                for _ in range((length - processed) // width):
                    if value in (ptr + processed).load[width=width]():
                        return True
                    processed += width

        for i in range(length - processed):
            if ptr[processed + i] == value:
                return True
        return False

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn get_immutable(self) -> Self.Immutable:
        """Return an immutable version of this `Span`.

        Returns:
            An immutable version of the same `Span`.
        """
        return rebind[Self.Immutable](self)

    @always_inline("builtin")
    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[
        T,
        mut=mut,
        origin=origin,
        address_space=address_space,
        alignment=alignment,
    ]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """
        return self._data

    @always_inline
    fn as_ref(self) -> Pointer[T, origin, address_space=address_space]:
        """
        Gets a `Pointer` to the first element of this span.

        Returns:
            A `Pointer` pointing at the first element of this span.
        """

        return Pointer[T, origin, address_space=address_space](to=self._data[0])

    @always_inline
    fn copy_from[
        origin: MutableOrigin, other_alignment: Int, //
    ](
        self: Span[T, origin, alignment=alignment],
        other: Span[T, _, alignment=other_alignment],
    ):
        """
        Performs an element wise copy from all elements of `other` into all elements of `self`.

        Parameters:
            origin: The inferred mutable origin of the data within the Span.
            other_alignment: The inferred alignment of the data within the Span.

        Args:
            other: The `Span` to copy all elements from.
        """
        debug_assert(len(self) == len(other), "Spans must be of equal length")
        for i in range(len(self)):
            self[i] = other[i]

    fn __bool__(self) -> Bool:
        """Check if a span is non-empty.

        Returns:
           True if a span is non-empty, False otherwise.
        """
        return len(self) > 0

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    fn __eq__[
        T: EqualityComparable & Copyable & Movable, rhs_alignment: Int, //
    ](
        self: Span[T, origin, alignment=alignment],
        rhs: Span[T, _, alignment=rhs_alignment],
    ) -> Bool:
        """Verify if span is equal to another span.

        Parameters:
            T: The type of the elements in the span. Must implement the
              traits `EqualityComparable`, `Copyable` and `Movable`.
            rhs_alignment: The inferred alignment of the rhs span.

        Args:
            rhs: The span to compare against.

        Returns:
            True if the spans are equal in length and contain the same elements, False otherwise.
        """
        # both empty
        if not self and not rhs:
            return True
        if len(self) != len(rhs):
            return False
        # same pointer and length, so equal
        if self.unsafe_ptr() == rhs.unsafe_ptr():
            return True
        for i in range(len(self)):
            if self[i] != rhs[i]:
                return False
        return True

    @always_inline
    fn __ne__[
        T: EqualityComparable & Copyable & Movable, //
    ](self: Span[T, origin, alignment=alignment], rhs: Span[T]) -> Bool:
        """Verify if span is not equal to another span.

        Parameters:
            T: The type of the elements in the span. Must implement the
              traits `EqualityComparable`, `Copyable` and `Movable`.

        Args:
            rhs: The span to compare against.

        Returns:
            True if the spans are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    fn fill[
        origin: MutableOrigin, //
    ](self: Span[T, origin, alignment=alignment], value: T):
        """
        Fill the memory that a span references with a given value.

        Parameters:
            origin: The inferred mutable origin of the data within the Span.

        Args:
            value: The value to assign to each element.
        """
        for ref element in self:
            element = value

    fn swap_elements(
        self: Span[mut=True, T, alignment=alignment], a: UInt, b: UInt
    ) raises:
        """
        Swap the values at indices `a` and `b`.

        Args:
            a: The first argument index.
            b: The second argument index.

        Raises:
            If a or b are larger than the length of the span.
        """
        var length = len(self)
        if a > length or b > length:
            raise Error(
                "index out of bounds (length: ",
                length,
                ", a: ",
                a,
                ", b: ",
                b,
                ")",
            )

        var ptr = self.unsafe_ptr()
        var tmp = InlineArray[T, 1](uninitialized=True)
        ptr.offset(a).move_pointee_into(tmp.unsafe_ptr())
        ptr.offset(b).move_pointee_into(ptr.offset(a))
        tmp.unsafe_ptr().move_pointee_into(ptr.offset(b))

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: __type_of(
            Span[T, _, address_space=address_space, alignment=alignment]
        ),
    ](
        self,
        out result: Span[
            mut = mut & other_type.origin.mut,
            T,
            __origin_of(origin, other_type.origin),
            address_space=address_space,
            alignment=alignment,
        ],
    ):
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return __type_of(result)(
            ptr=self._data.origin_cast[result.mut, result.origin](),
            length=self._len,
        )
