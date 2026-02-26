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
"""Provides a fixed-size array implementation with compile-time size checking.

The `InlineArray` type represents a fixed-size sequence of homogeneous elements
where the size is determined at compile time. It provides efficient memory
layout and bounds checking while maintaining type safety.  The `InlineArray`
type is part of the `prelude` module and therefore does not need to be imported
in order to use it.

Examples:

```mojo
# Create an array of 3 integers
var arr: InlineArray[Int, 3] = [1, 2, 3]

# Access elements
print(arr[0])  # Prints 1

# Fill with a value
var filled = InlineArray[Int, 5](fill=42)
```
"""

import math
from collections._index_normalization import normalize_index

from builtin.device_passable import DevicePassable
from builtin.rebind import downcast
from builtin.constrained import _constrained_conforms_to
from builtin.repr import repr
from compile import get_type_name
import format._utils as fmt
from memory import UnsafeMaybeUninit

# ===-----------------------------------------------------------------------===#
# Array
# ===-----------------------------------------------------------------------===#


fn _inline_array_construction_checks[size: Int]():
    """Checks if the properties in `InlineArray` are valid.

    Validity right now is just ensuring the number of elements is > 0.

    Parameters:
        size: The number of elements.
    """
    comptime assert (
        size >= 0
    ), "number of elements in `InlineArray` must be >= 0"


@fieldwise_init
struct _InlineArrayIter[
    mut: Bool,
    //,
    T: Copyable,
    size: Int,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator for `InlineArray`.

    Parameters:
        mut: A boolean to indicate if the iterator is mutable.
        T: The type of the elements in the iterator.
        size: The size of the array.
        origin: The origin of the iterator.
        forward: A boolean to indicate if the iterator is forward.
    """

    comptime Element = Self.T

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: Pointer[InlineArray[Self.T, Self.size], Self.origin]

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        comptime if Self.forward:
            if self.index >= Self.size:
                raise StopIteration()
            self.index += 1
            return self.src[][self.index - 1]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            return self.src[][self.index]

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        comptime if Self.forward:
            iter_len = Self.size - self.index
        else:
            iter_len = self.index

        return (iter_len, {iter_len})


struct InlineArray[ElementType: Copyable, size: Int](
    Copyable,
    Defaultable,
    DevicePassable,
    Iterable,
    Sized,
    Writable,
):
    """A fixed-size sequence of homogeneous elements where size is a constant
    expression.

    InlineArray provides a fixed-size array implementation with compile-time
    size checking. The array size is determined at compile time and cannot be
    changed. Elements must implement the `Copyable` trait.

    Parameters:
        ElementType: The type of the elements in the array. Must implement
            `Copyable` trait.
        size: The size of the array. Must be a positive integer constant.

    Examples:

    ```mojo
    # Create array of 3 integers
    var arr: InlineArray[Int, 3] = [1, 2, 3]

    # Create array filled with value
    var filled = InlineArray[Int, 5](fill=42)

    # Access elements
    print(arr[0])  # Prints 1
    ```
    """

    comptime __del__is_trivial: Bool = downcast[
        Self.ElementType, ImplicitlyDestructible
    ].__del__is_trivial
    comptime __copy_ctor_is_trivial: Bool = Self.ElementType.__copy_ctor_is_trivial
    comptime __move_ctor_is_trivial: Bool = Self.ElementType.__move_ctor_is_trivial

    # Fields
    comptime type = __mlir_type[
        `!pop.array<`, Self.size._mlir_value, `, `, Self.ElementType, `>`
    ]
    """The underlying MLIR array type."""

    var _array: Self.type
    """The underlying storage for the array."""

    comptime device_type: AnyType = Self
    """The device-side type for this array."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _InlineArrayIter[
        Self.ElementType, Self.size, iterable_origin, True
    ]
    """The iterator type for this array.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self.copy()

    @staticmethod
    fn get_type_name() -> String:
        """Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "InlineArray[",
            get_type_name[Self.ElementType](),
            ", ",
            Self.size,
            "]",
        )

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """This constructor will always cause a compile time error if used.
        It is used to steer users away from uninitialized memory.
        """
        comptime assert False, (
            "Initialize with either a variadic list of arguments, a default"
            " fill element or pass the keyword argument"
            " 'uninitialized=True'."
        )

    @always_inline
    fn __init__(out self, *, uninitialized: Bool):
        """Create an InlineArray with uninitialized memory.

        Args:
            uninitialized: A boolean to indicate if the array should be
                initialized. Always set to `True` (it's not actually used inside
                the constructor).

        Examples:

        ```mojo
        var uninitialized_array = InlineArray[Int, 10](uninitialized=True)
        ```

        Notes:
            This constructor is unsafe and should be used with caution. The
            array elements will be uninitialized and accessing them before
            initialization is undefined behavior.
        """
        _inline_array_construction_checks[Self.size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    fn __init__(
        out self,
        *,
        var unsafe_assume_initialized: InlineArray[
            UnsafeMaybeUninit[Self.ElementType], Self.size
        ],
    ):
        """Constructs an `InlineArray` from an `InlineArray` of
        `UnsafeMaybeUninit`.

        Args:
            unsafe_assume_initialized: The array of `UnsafeMaybeUninit`
                elements. All elements must be initialized.

        Warning:
            This is an unsafe constructor. Only use it if you are certain all
            elements are properly initialized.

        Notes:
            This constructor assumes all elements in the input array are
            initialized. Using uninitialized elements results in undefined
            behavior, even for types that are valid for any bit pattern
            (e.g. `Int` or `Float`).
        """

        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        for i in range(Self.size):
            (self.unsafe_ptr() + i).init_pointee_move_from(
                unsafe_assume_initialized[i].unsafe_ptr()
            )

    @always_inline
    fn __init__[batch_size: Int = 64](out self, *, fill: Self.ElementType):
        """Constructs an array where each element is initialized to the supplied
        value.

        Parameters:
            batch_size: The number of elements to unroll for filling the array.
                Default is 64, which optimizes for AVX512 operations on modern
                CPUs. For large arrays (>2k elements), this batched approach
                significantly improves compile times compared to full unrolling
                while maintaining good runtime performance.

        Args:
            fill: The element value to fill each index with.

        Examples:

        ```mojo
        var filled = InlineArray[Int, 5](fill=42)  # [42, 42, 42, 42, 42]

        # For large arrays, consider adjusting batch_size to balance
        # compile time and runtime performance:
        var large = InlineArray[Int, 10000].__init__[batch_size=32](fill=0)
        ```

        Notes:

        - Full unrolling with large arrays (>2k elements) can cause significant
            compiler slowdowns.
        - Using batch_size=64 balances AVX512 efficiency and instruction cache
            usage.
        - For very large arrays, using smaller batch sizes (e.g., 32 or 16) can
            further improve compilation speed while still maintaining good
            runtime performance.
        """
        _inline_array_construction_checks[Self.size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        comptime unroll_end = math.align_down(Self.size, batch_size)

        var ptr = self.unsafe_ptr()

        for _ in range(0, unroll_end, batch_size):
            comptime for _ in range(batch_size):
                ptr.init_pointee_copy(fill)
                ptr += 1

        # Fill the remainder
        comptime for _ in range(unroll_end, Self.size):
            ptr.init_pointee_copy(fill)
            ptr += 1
        debug_assert(
            ptr == self.unsafe_ptr() + Self.size,
            "error during `InlineArray` initialization , please file a bug",
            " report.",
        )

    @always_inline
    fn __init__(out self, var *elems: Self.ElementType, __list_literal__: ()):
        """Constructs an array from a variadic list of elements.

        Args:
            elems: The elements to initialize the array with. Must match the
                array size.
            __list_literal__: Specifies that this constructor can be used for
                list literals.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        ```
        """
        debug_assert(
            len(elems) == Self.size, "No. of elems must match array size"
        )
        self = Self(storage=elems^)

    @always_inline
    fn __init__[
        origin: MutOrigin,
        //,
    ](
        out self,
        *,
        var storage: VariadicListMem[
            elt_is_mutable=True, origin=origin, Self.ElementType, is_owned=True
        ],
    ):
        """Construct an array from a low-level internal representation.

        Parameters:
            origin: The origin of the storage being passed in.

        Args:
            storage: The variadic list storage to construct from. Must match
                array size.
        """

        debug_assert(
            len(storage) == Self.size,
            "Expected variadic list of length ",
            Self.size,
            ", received ",
            len(storage),
        )
        _inline_array_construction_checks[Self.size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        var ptr = self.unsafe_ptr()

        # Move each element into the array storage.
        comptime for i in range(Self.size):
            # Safety: We own the elements in the variadic list.
            ptr.init_pointee_move_from(
                UnsafePointer(to=storage[i]).unsafe_mut_cast[True]()
            )
            ptr += 1

        # Do not destroy the elements when their backing storage goes away.
        # FIXME: Why doesn't consume_elements work here?
        storage^._anihilate()

    fn __init__(out self, *, copy: Self):
        """Copy constructs the array from another array.

        Args:
            copy: The array to copy from.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        var copy = arr.copy()  # Creates new array [1, 2, 3]
        ```
        """

        comptime if Self.ElementType.__copy_ctor_is_trivial:
            self._array = copy._array
        else:
            self = Self(uninitialized=True)
            for idx in range(Self.size):
                var ptr = self.unsafe_ptr() + idx
                ptr.init_pointee_copy(copy.unsafe_get(idx))

    fn __init__(out self, *, deinit take: Self):
        """Move constructs the array from another array.

        Args:
            take: The array to move from.

        Notes:
            Moves the elements from the source array into this array.
        """

        comptime if Self.ElementType.__move_ctor_is_trivial:
            self._array = take._array
        else:
            self = Self(uninitialized=True)
            for idx in range(Self.size):
                var other_ptr = take.unsafe_ptr() + idx
                (self.unsafe_ptr() + idx).init_pointee_move_from(other_ptr)

    fn __del__(deinit self):
        """Deallocates the array and destroys its elements."""

        _constrained_conforms_to[
            conforms_to(Self.ElementType, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.ElementType,
            ParentConformsTo="ImplicitlyDestructible",
        ]()
        comptime TDestructible = downcast[
            Self.ElementType, ImplicitlyDestructible
        ]

        comptime if not TDestructible.__del__is_trivial:
            comptime for idx in range(Self.size):
                var ptr = self.unsafe_ptr() + idx
                ptr.bitcast[TDestructible]().destroy_pointee()

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[I: Indexer](ref self, idx: I) -> ref[self] Self.ElementType:
        """Gets a reference to the element at the given index.

        Parameters:
            I: The type parameter representing the index type, must implement
                Indexer trait.

        Args:
            idx: The index to access. Can be positive (0 to len-1) or negative
                (-len to -1).

        Returns:
            A reference to the element at the specified index.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        print(arr[0])   # Prints 1 - first element
        print(arr[1])   # Prints 2 - second element
        print(arr[-1])  # Prints 3 - last element
        print(arr[-2])  # Prints 2 - second to last element
        ```

        Notes:
            This method provides array-style indexing access to elements in the
            InlineArray. It supports both positive indices starting from 0 and
            negative indices counting backwards from the end of the array. The
            index is bounds-checked at runtime.
        """
        var normalized_index = normalize_index["InlineArray"](idx, len(self))
        return self.unsafe_get(normalized_index)

    @always_inline
    fn __getitem__[
        I: Indexer, //, idx: I
    ](ref self) -> ref[self] Self.ElementType:
        """Gets a reference to the element at the given index with compile-time
        bounds checking.

        Parameters:
            I: The type parameter representing the index type, must implement
                Indexer trait.
            idx: The compile-time constant index to access. Can be positive
                (0 to len-1) or negative (-len to -1).

        Returns:
            A reference to the element at the specified index.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        print(arr[0])   # Prints 1 - first element
        print(arr[-1])  # Prints 3 - last element
        ```

        Notes:
            This overload provides array-style indexing with compile-time bounds
            checking. The index must be a compile-time constant value. It
            supports both positive indices starting from 0 and negative indices
            counting backwards from the end of the array.
        """
        comptime assert (
            -Self.size <= index(idx) < Self.size
        ), "Index must be within bounds."
        comptime normalized_index = normalize_index["InlineArray"](
            idx, Self.size
        )
        return self.unsafe_get(normalized_index)

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the array.

        Returns:
            The size of the array as an Int.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        print(len(arr))  # Prints 3
        ```

        Notes:
            The length is a compile-time constant value determined by the
            size parameter used when creating the array.
        """
        return Self.size

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn unsafe_get[I: Indexer](ref self, idx: I) -> ref[self] Self.ElementType:
        """Gets a reference to an element without bounds checking.

        Parameters:
            I: A type parameter representing the index type, must implement
                Indexer trait.

        Args:
            idx: The index of the element to get. Must be non-negative and in
                bounds. Using an invalid index will cause undefined behavior.

        Returns:
            A reference to the element at the given index.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        print(arr.unsafe_get(0))  # Prints 1
        ```

        Warning:
            This is an unsafe method. No bounds checking is performed.
            Using an invalid index will cause undefined behavior.
            Negative indices are not supported.

        Notes:
            This is an unsafe method that skips bounds checking for performance.
            Users should prefer `__getitem__` instead for safety.
        """
        var i = index(idx)
        debug_assert(
            0 <= i < Self.size,
            " InlineArray.unsafe_get() index out of bounds: ",
            i,
            " should be greater than or equal to 0 and less than ",
            Self.size,
        )
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer(to=self._array).address,
            i._mlir_value,
        )
        return UnsafePointer[_, origin_of(self)](ptr)[]

    @always_inline
    fn unsafe_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Self.ElementType,
        origin,
        address_space=address_space,
    ]:
        """Gets an unsafe pointer to the underlying array storage.

        Parameters:
            origin: The origin of the reference to self.
            address_space: The address space of the array.

        Returns:
            An `UnsafePointer` to the underlying array storage. The pointer's
            mutability matches that of the array reference.

        Examples:

        ```mojo
        var arr:InlineArray[Int, 3] = [1, 2, 3]
        var ptr = arr.unsafe_ptr()
        print(ptr[0])  # Prints 1
        ```

        Warning:
            This is an unsafe method. The returned pointer:
            - Becomes invalid if the array is moved
            - Must not be used to access memory outside array bounds
            - Must be refreshed after any operation that could move the array

        Notes:
            Returns a raw pointer to the array's memory that can be used for
            direct memory access. The pointer inherits mutability from the array
            reference.
        """
        return (
            UnsafePointer(to=self._array)
            .bitcast[Self.ElementType]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )

    @always_inline
    fn __contains__[
        T: Equatable & Copyable, //
    ](self: InlineArray[T, Self.size], value: T) -> Bool:
        """Tests if a value is present in the array using the `in` operator.

        Parameters:
            T: The element type, must implement both `Equatable` and
                `Copyable`.

        Args:
            value: The value to search for.

        Returns:
            True if the value is found in any position in the array, False
            otherwise.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        print(3 in arr)  # Prints True - value exists
        print(4 in arr)  # Prints False - value not found
        ```

        Notes:
            This method enables using the `in` operator to check if a value
            exists in the array. It performs a linear search comparing each
            element for equality with the given value. The element type must
            implement the `Equatable` and `Copyable` traits
            to support equality comparison.
        """

        comptime for i in range(Self.size):
            if self[i] == value:
                return True
        return False

    # ===-------------------------------------------------------------------===#
    # String representation
    # ===-------------------------------------------------------------------===#

    fn _write_self_to[
        f: fn(Self.ElementType, mut Some[Writer])
    ](self, mut writer: Some[Writer]):
        fmt.constrained_conforms_to_writable[Self.ElementType, Parent=Self]()

        var index = 0

        @parameter
        fn iterate(mut w: Some[Writer]) raises StopIteration:
            if index >= Self.size:
                raise StopIteration()
            f(self.unsafe_get(index), w)
            index += 1

        fmt.write_sequence_to[ElementFn=iterate](writer)
        _ = index

    fn write_to(self, mut writer: Some[Writer]):
        """Writes the InlineArray representation to a Writer.

        Constraints:
            ElementType must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[f = fmt.write_to[Self.ElementType]](writer)

    fn write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr representation of this InlineArray to a Writer.

        Constraints:
            ElementType must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        @parameter
        fn write_fields(mut w: Some[Writer]):
            self._write_self_to[f = fmt.write_repr_to[Self.ElementType]](w)

        fmt.FormatStruct(writer, "InlineArray").params(
            fmt.TypeNames[Self.ElementType](),
            Self.size,
        ).fields[FieldsFn=write_fields]()

    @deprecated("Stringable is deprecated. Use Writable instead.")
    @always_inline
    fn __str__(self) -> String:
        """Returns a string representation of the InlineArray.

        Returns:
            A string representation of the array.
        """
        output = String()
        self.write_to(output)
        return output^

    @deprecated("Representable is deprecated. Use Writable instead.")
    @always_inline
    fn __repr__(self) -> String:
        """Returns a string representation of the InlineArray.

        Returns:
            A string representation of the array.
        """
        output = String()
        self.write_repr_to(output)
        return output^

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the array, returning immutable references.

        Returns:
            An iterator of immutable references to the array elements.
        """
        return {0, Pointer(to=self)}

    fn __reversed__(
        ref self,
    ) -> _InlineArrayIter[Self.ElementType, Self.size, origin_of(self), False]:
        """Iterate over elements of the array in reverse order, returning immutable references.

        Returns:
            An iterator of immutable references to the array elements in reverse order.
        """
        return _InlineArrayIter[forward=False](Self.size, Pointer(to=self))
