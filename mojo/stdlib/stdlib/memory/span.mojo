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

from builtin._location import __call_location
from collections._index_normalization import normalize_index
from sys import align_of
from sys.info import simd_width_of

from algorithm import vectorize
from memory import Pointer


@fieldwise_init
struct _SpanIter[
    mut: Bool, //,
    T: Copyable & Movable,
    origin: Origin[mut],
    forward: Bool = True,
    address_space: AddressSpace = AddressSpace.GENERIC,
](ImplicitlyCopyable, Movable):
    """Iterator for Span.

    Parameters:
        mut: Whether the reference to the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the `Span`.
        forward: The iteration direction. False is backwards.
        address_space: The address space associated with the underlying allocated memory.

    """

    var index: Int
    var src: Span[T, origin, address_space=address_space]

    @always_inline
    fn __iter__(self) -> Self:
        return self.copy()

    @always_inline
    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < len(self.src)
        else:
            return self.index > 0

    @always_inline
    fn __next_ref__(mut self) -> ref [origin, address_space] T:
        @parameter
        if forward:
            var curr = self.index
            self.index += 1
            return self.src[curr]
        else:
            self.index -= 1
            return self.src[self.index]


@fieldwise_init
@register_passable("trivial")
struct Span[
    mut: Bool, //,
    T: Copyable & Movable,
    origin: Origin[mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](Boolable, Defaultable, ImplicitlyCopyable, Movable, Sized):
    """A non-owning view of contiguous data.

    Parameters:
        mut: Whether the span is mutable.
        T: The type of the elements in the span.
        origin: The origin of the Span.
        address_space: The address space associated with the allocated memory.
    """

    # Aliases
    alias Mutable = Span[T, MutableOrigin.cast_from[origin]]
    """The mutable version of the `Span`."""
    alias Immutable = Span[T, ImmutableOrigin.cast_from[origin]]
    """The immutable version of the `Span`."""
    alias UnsafePointerType = UnsafePointer[
        T,
        mut=mut,
        origin=origin,
        address_space=address_space,
    ]
    """The UnsafePointer type that corresponds to this `Span`."""
    # Fields
    var _data: Self.UnsafePointerType
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
    fn __init__(out self, *, ptr: Self.UnsafePointerType, length: UInt):
        """Unsafe construction from a pointer and length.

        Args:
            ptr: The underlying pointer of the span.
            length: The length of the view.
        """
        self._data = ptr
        self._len = Int(length)

    @always_inline
    @implicit
    fn __init__(out self, ref [origin, address_space]list: List[T, *_]):
        """Construct a `Span` from a `List`.

        Args:
            list: The list to which the span refers.
        """
        self._data = (
            list.unsafe_ptr()
            .address_space_cast[address_space]()
            .unsafe_origin_cast[origin]()
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
            .unsafe_origin_cast[origin]()
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
        var normalized_idx = normalize_index["Span", assert_always=False](
            idx, UInt(len(self))
        )
        return self._data[normalized_idx]

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
            ptr=(self._data + start), length=UInt(len(range(start, end, step)))
        )

    @always_inline
    fn __iter__(
        self,
    ) -> _SpanIter[T, origin, address_space=address_space,]:
        """Get an iterator over the elements of the `Span`.

        Returns:
            An iterator over the elements of the `Span`.
        """
        return _SpanIter[address_space=address_space](0, self)

    @always_inline
    fn __reversed__(
        self,
    ) -> _SpanIter[T, origin, forward=False, address_space=address_space,]:
        """Iterate backwards over the `Span`.

        Returns:
            A reversed iterator of the `Span` elements.
        """
        return _SpanIter[forward=False, address_space=address_space](
            len(self), self
        )

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
            if simd_width_of[dtype]() >= width:
                for _ in range((length - processed) // width):
                    if value in (ptr + processed).load[width=width]():
                        return True
                    processed += width

        for i in range(length - processed):
            if ptr[processed + i] == value:
                return True
        return False

    @no_inline
    fn __str__[
        U: Representable & Copyable & Movable, //
    ](self: Span[U, *_]) -> String:
        """Returns a string representation of a `Span`.

        Parameters:
            U: The type of the elements in the span. Must implement the
              trait `Representable`.

        Returns:
            A string representation of the span.

        Notes:
            Note that since we can't condition methods on a trait yet,
            the way to call this method is a bit special. Here is an example
            below:

            ```mojo
            var my_list = [1, 2, 3]
            var my_span = Span(my_list)
            print(my_span.__str__())
            ```

            When the compiler supports conditional methods, then a simple
            `String(my_span)` will be enough.
        """
        # at least 1 byte per item e.g.: [a, b, c, d] = 4 + 2 * 3 + [] + null
        var l = len(self)
        var output = String(capacity=l + 2 * (l - 1) * Int(l > 1) + 3)
        self.write_to(output)
        return output^

    @no_inline
    fn write_to[
        U: Representable & Copyable & Movable, //
    ](self: Span[U, *_], mut writer: Some[Writer]):
        """Write `my_span.__str__()` to a `Writer`.

        Parameters:
            U: The type of the Span elements. Must have the trait
                `Representable`.

        Args:
            writer: The object to write to.
        """
        writer.write("[")
        for i in range(len(self)):
            writer.write(repr(self[i]))
            if i < len(self) - 1:
                writer.write(", ")
        writer.write("]")

    @no_inline
    fn __repr__[
        U: Representable & Copyable & Movable, //
    ](self: Span[U, *_]) -> String:
        """Returns a string representation of a `Span`.

        Parameters:
            U: The type of the elements in the span. Must implement the
              trait `Representable`.

        Returns:
            A string representation of the span.

        Notes:
            Note that since we can't condition methods on a trait yet, the way
            to call this method is a bit special. Here is an example below:

            ```mojo
            var my_list = [1, 2, 3]
            var my_span = Span(my_list)
            print(my_span.__repr__())
            ```

            When the compiler supports conditional methods, then a simple
            `repr(my_span)` will be enough.
        """
        return self.__str__()

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

    @always_inline
    fn unsafe_get(self, idx: Some[Indexer]) -> ref [origin, address_space] T:
        """Get a reference to the element at `index` without bounds checking.

        Args:
            idx: The index of the element to get.

        Safety:
            - This function does not do bounds checking and assumes the provided
            index is in: [0, len(self)). Not upholding this contract will result
            in undefined behavior.
            - This function does not support wraparound for negative indices.
        """
        debug_assert(
            0 <= index(idx) < len(self),
            "Index out of bounds: ",
            index(idx),
        )
        return self._data[idx]

    @always_inline("builtin")
    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[T, mut=mut, origin=origin, address_space=address_space,]:
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
        _origin: MutableOrigin, //
    ](self: Span[T, _origin], other: Span[T, _],):
        """
        Performs an element wise copy from all elements of `other` into all elements of `self`.

        Parameters:
            _origin: The inferred mutable origin of the data within the Span.

        Args:
            other: The `Span` to copy all elements from.
        """
        debug_assert(
            len(self) == len(other),
            "Spans must be of equal length",
        )
        for i in range(len(self)):
            self[i] = other[i].copy()

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
        _T: EqualityComparable & Copyable & Movable, //,
    ](self: Span[_T, origin], rhs: Span[_T, _],) -> Bool:
        """Verify if span is equal to another span.

        Parameters:
            _T: The type of the elements must implement the
              traits `EqualityComparable`, `Copyable` and `Movable`.

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
        _T: EqualityComparable & Copyable & Movable, //
    ](self: Span[_T, origin], rhs: Span[_T]) -> Bool:
        """Verify if span is not equal to another span.

        Parameters:
            _T: The type of the elements in the span. Must implement the
              traits `EqualityComparable`, `Copyable` and `Movable`.

        Args:
            rhs: The span to compare against.

        Returns:
            True if the spans are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    fn fill[_origin: MutableOrigin, //](self: Span[T, _origin], value: T):
        """
        Fill the memory that a span references with a given value.

        Parameters:
            _origin: The inferred mutable origin of the data within the Span.

        Args:
            value: The value to assign to each element.
        """
        for ref element in self:
            element = value.copy()

    @always_inline
    fn unsafe_swap_elements(self: Span[mut=True, T], a: Int, b: Int):
        """Swap the values at indices `a` and `b` without performing bounds checking.

        Args:
            a: The first element's index.
            b: The second element's index.

        Safety:
            - Both `a` and `b` must be in: [0, len(self)).
        """
        debug_assert(
            0 <= a < len(self),
            "Index `a` out of bounds: ",
            a,
        )
        debug_assert(
            0 <= b < len(self),
            "Index `b` out of bounds: ",
            b,
        )
        var ptr = self.unsafe_ptr()

        # `a` and `b` may be equal, so we cannot use `swap` directly.
        ptr.offset(a).swap_pointees(ptr.offset(b))

    fn swap_elements(self: Span[mut=True, T], a: Int, b: Int) raises:
        """
        Swap the values at indices `a` and `b`.

        Args:
            a: The first argument index.
            b: The second argument index.

        Raises:
            If a or b are larger than the length of the span.
        """
        var length = UInt(len(self))
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

        self.unsafe_swap_elements(a, b)

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: __type_of(Span[T, _, address_space=address_space]),
    ](
        self,
        out result: Span[
            mut = mut & other_type.origin.mut,
            T,
            __origin_of(origin, other_type.origin),
            address_space=address_space,
        ],
    ):
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return {
            ptr = self._data.unsafe_mut_cast[result.mut]().unsafe_origin_cast[
                result.origin
            ](),
            length = UInt(self._len),
        }

    fn reverse[
        dtype: DType, O: MutableOrigin, //
    ](self: Span[Scalar[dtype], O]):
        """Reverse the elements of the `Span` inplace.

        Parameters:
            dtype: The DType of the scalars the `Span` stores.
            O: The origin of the `Span`.
        """

        alias widths = (256, 128, 64, 32, 16, 8, 4, 2)
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var middle = length // 2
        var is_odd = length % 2 != 0
        var processed = 0

        @parameter
        for i in range(len(widths)):
            alias w = widths[i]

            @parameter
            if simd_width_of[dtype]() >= w:
                for _ in range((middle - processed) // w):
                    var lhs_ptr = ptr + processed
                    var rhs_ptr = ptr + length - (processed + w)
                    var lhs_v = lhs_ptr.load[width=w]().reversed()
                    var rhs_v = rhs_ptr.load[width=w]().reversed()
                    lhs_ptr.store(rhs_v)
                    rhs_ptr.store(lhs_v)
                    processed += w

        if is_odd:
            var value = ptr[middle + 1]
            (ptr + middle + 1).init_pointee_move_from(ptr + middle - 1)
            (ptr + middle - 1).init_pointee_move(value)

    fn apply[
        dtype: DType,
        O: MutableOrigin, //,
        func: fn[w: Int] (SIMD[dtype, w]) capturing -> SIMD[dtype, w],
    ](self: Span[Scalar[dtype], O]):
        """Apply the function to the `Span` inplace.

        Parameters:
            dtype: The DType.
            O: The origin of the `Span`.
            func: The function to evaluate.
        """

        alias widths = (256, 128, 64, 32, 16, 8, 4)
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var processed = 0

        @parameter
        for i in range(len(widths)):
            alias w = widths[i]

            @parameter
            if simd_width_of[dtype]() >= w:
                for _ in range((length - processed) // w):
                    var p_curr = ptr + processed
                    p_curr.store(func(p_curr.load[width=w]()))
                    processed += w

        for i in range(length - processed):
            (ptr + processed + i).init_pointee_move(func(ptr[processed + i]))

    fn apply[
        dtype: DType,
        O: MutableOrigin, //,
        func: fn[w: Int] (SIMD[dtype, w]) capturing -> SIMD[dtype, w],
        *,
        cond: fn[w: Int] (SIMD[dtype, w]) capturing -> SIMD[DType.bool, w],
    ](self: Span[Scalar[dtype], O]):
        """Apply the function to the `Span` inplace where the condition is
        `True`.

        Parameters:
            dtype: The DType.
            O: The origin of the `Span`.
            func: The function to evaluate.
            cond: The condition to apply the function.
        """

        alias widths = (256, 128, 64, 32, 16, 8, 4)
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var processed = 0

        @parameter
        for i in range(len(widths)):
            alias w = widths[i]

            @parameter
            if simd_width_of[dtype]() >= w:
                for _ in range((length - processed) // w):
                    var p_curr = ptr + processed
                    var vec = p_curr.load[width=w]()
                    p_curr.store(cond(vec).select(func(vec), vec))
                    processed += w

        for i in range(length - processed):
            var vec = ptr[processed + i]
            if cond(vec):
                (ptr + processed + i).init_pointee_move(func(vec))

    fn count[
        dtype: DType, //,
        func: fn[w: Int] (SIMD[dtype, w]) capturing -> SIMD[DType.bool, w],
    ](self: Span[Scalar[dtype]]) -> UInt:
        """Count the amount of times the function returns `True`.

        Parameters:
            dtype: The DType.
            func: The function to evaluate.

        Returns:
            The amount of times the function returns `True`.
        """

        alias simdwidth = simd_width_of[DType.int]()
        var ptr = self.unsafe_ptr()
        var length = len(self)
        var countv = SIMD[DType.int, simdwidth](0)
        var count = Scalar[DType.int](0)

        @parameter
        fn do_count[width: Int](idx: Int):
            var vec = func(ptr.load[width=width](idx)).cast[DType.int]()

            @parameter
            if width == 1:
                count += rebind[__type_of(count)](vec)
            else:
                countv += rebind[__type_of(countv)](vec)

        vectorize[do_count, simdwidth](length)

        return UInt(countv.reduce_add() + count)

    @always_inline
    fn unsafe_subspan(self, *, offset: Int, length: Int) -> Self:
        """Returns a subspan of the current span.

        Args:
            offset: The starting offset of the subspan (self._data + offset).
            length: The length of the new subspan.

        Safety:
            This function does not do bounds checking and assumes the current
            span contains the specified subspan.
        """
        debug_assert(
            0 <= offset < len(self),
            "offset out of bounds: ",
            offset,
        )
        debug_assert(
            0 <= offset + length <= len(self),
            "subspan out of bounds.",
        )
        return Self(ptr=self._data + offset, length=UInt(length))
