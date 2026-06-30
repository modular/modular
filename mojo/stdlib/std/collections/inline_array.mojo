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

import std.math
import std.memory
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.rebind import downcast
from std.collections import check_bounds
import std.format._utils as fmt
from std.reflection import reflect
from std.hashlib.hasher import Hasher
from std.memory import (
    UnsafeMaybeUninit,
    destroy_n,
    forget_deinit,
    is_trivially_copyable,
    is_trivially_destructible,
    is_trivially_movable,
    uninit_move_n,
)
from std.memory.unsafe_maybe_uninit import (
    _is_trivially_copyable,
    _is_trivially_movable,
)
from std.utils.type_functions import ConditionalType

# ===-----------------------------------------------------------------------===#
# Array
# ===-----------------------------------------------------------------------===#


def _inline_array_construction_checks[size: Int]():
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
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
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
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        comptime if Self.forward:
            iter_len = Self.size - self.index
        else:
            iter_len = self.index

        return (iter_len, {iter_len})


struct _InlineArrayIterOwned[T: Movable & ImplicitlyDeletable, size: Int](
    IterableOwned, Iterator, Movable
):
    """An owning iterator for InlineArray.

    Parameters:
        T: The type of the elements in the array.
        size: The size of the array.
    """

    comptime Element = Self.T
    comptime IteratorOwnedType = Self

    var _array: InlineArray[Self.T, Self.size]
    var _index: Int

    def __init__(out self, var array: InlineArray[Self.T, Self.size]):
        """Consume an array and create an iterator over its elements.

        Args:
            array: The array to consume.
        """
        self._array = array^
        self._index = 0

    def __init__(out self, *, deinit move: Self):
        """Move constructor that handles partially consumed array storage.

        After partial iteration some array slots are uninitialized, so
        the default fieldwise move would be unsound.  This constructor
        moves only the unconsumed elements and marks the source
        destroyed.

        Args:
            move: The iterator to move from.
        """
        self._index = move._index
        self._array = InlineArray[Self.T, Self.size](uninitialized=True)
        uninit_move_n[overlapping=False](
            dest=self._array.unsafe_ptr() + move._index,
            src=move._array.unsafe_ptr() + move._index,
            count=Self.size - move._index,
        )

    @always_inline
    def __del__(deinit self):
        # Move fields out of self so we can manage their lifetimes.
        var idx = self._index
        var array = self._array^

        # Destroy the remaining elements that have not yet been
        # iterated over.
        destroy_n(array.unsafe_ptr() + idx, Self.size - idx)

        # Mark the array as destroyed so InlineArray.__del__ doesn't
        # double-destroy the elements we already handled.
        forget_deinit(array^)

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        if self._index >= Self.size:
            raise StopIteration()
        self._index += 1
        return (self._array.unsafe_ptr() + self._index - 1).take_pointee()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = Self.size - self._index
        return (remaining, {remaining})


@explicit_destroy(
    "An `InlineArray` of non-`ImplicitlyDeletable` elements must either be"
    " explicitly destroyed with `destroy_with()`, or have its ownership passed"
    " along by returning it or moving it into another function."
)
struct InlineArray[ElementType: Movable, size: Int](
    Copyable where conforms_to(ElementType, Copyable),
    Defaultable,
    DevicePassable where conforms_to(
        ElementType, DevicePassable
    ) and conforms_to(ElementType, Copyable),
    Equatable where conforms_to(ElementType, Equatable),
    Hashable where conforms_to(ElementType, Hashable),
    ImplicitlyCopyable where conforms_to(ElementType, ImplicitlyCopyable),
    ImplicitlyDeletable where conforms_to(ElementType, ImplicitlyDeletable),
    Iterable,
    IterableOwned where conforms_to(ElementType, ImplicitlyDeletable),
    Movable,
    Sized,
    Writable where conforms_to(ElementType, Writable),
):
    """A fixed-size sequence of homogeneous elements where size is a constant
    expression.

    InlineArray provides a fixed-size array implementation with compile-time
    size checking. The array size is determined at compile time and cannot be
    changed.

    Parameters:
        ElementType: The type of the elements in the array. Must implement
            `Movable`. Copy construction, `fill=` construction, and iteration
            additionally require `Copyable` and are enforced via conditional
            `where` clauses.
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

    comptime __del__is_trivial: Bool = is_trivially_destructible[
        downcast[Self.ElementType, ImplicitlyDeletable]
    ]() if conforms_to(Self.ElementType, ImplicitlyDeletable) else False
    comptime __copy_ctor_is_trivial: Bool = _is_trivially_copyable[
        Self.ElementType
    ]()
    comptime __move_ctor_is_trivial: Bool = _is_trivially_movable[
        Self.ElementType
    ]()

    # Fields
    comptime type = __mlir_type[
        `!pop.array<`, Self.size.__mlir_index__(), `, `, Self.ElementType, `>`
    ]
    """The underlying MLIR array type."""

    var _array: Self.type
    """The underlying storage for the array."""

    comptime _DeviceElementType: Movable = ConditionalType[
        Trait=Movable,
        If=conforms_to(Self.ElementType, DevicePassable),
        Then=downcast[
            downcast[Self.ElementType, DevicePassable].device_type, Movable
        ],
        Else=Self.ElementType,
    ]
    """The device-side element type: the element's `device_type` when it is
    `DevicePassable`, otherwise the element type itself."""

    comptime device_type: AnyType = InlineArray[
        Self._DeviceElementType, Self.size
    ]
    """The device-side type for this array.

    Parametric over the elements' device types, so an array of a `DevicePassable`
    element type encodes to the array of converted elements (and collapses to
    `Self` for identity elements)."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _InlineArrayIter[
        downcast[Self.ElementType, Copyable],
        Self.size,
        iterable_origin,
        True,
    ]
    """The iterator type for this array.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime IteratorOwnedType: Iterator = _InlineArrayIterOwned[
        downcast[Self.ElementType, ImplicitlyDeletable], Self.size
    ]
    """The owned iterator type for this array."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where conforms_to(Self.ElementType, DevicePassable) and conforms_to(
        Self.ElementType, Copyable
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: Target specific device type encoder.
            target: The target address to store the device type.
        """
        # Encode element-wise so a `DevicePassable` element runs its own
        # `_to_device_type` conversion rather than being byte-copied wholesale.
        encoder.encode_inline_array(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "InlineArray[",
            reflect[Self.ElementType].name(),
            ", ",
            Self.size,
            "]",
        )

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline
    def __init__(out self):
        """This constructor will always cause a compile time error if used.
        It is used to steer users away from uninitialized memory.
        """
        comptime assert False, (
            "Initialize with either a variadic list of arguments, a default"
            " fill element or pass the keyword argument"
            " 'uninitialized=True'."
        )

    @always_inline
    def __init__(out self, *, uninitialized: Bool):
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

    def __init__(
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
    def __init__[
        batch_size: SIMDSize = 64
    ](out self, *, fill: Self.ElementType) where conforms_to(
        Self.ElementType, Copyable
    ):
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
        self = Self(uninitialized=True)

        comptime unroll_end = std.math.align_down(Self.size, batch_size)

        var base = self.unsafe_ptr()
        var ptr = base

        for _ in range(0, unroll_end, batch_size):
            comptime for _ in range(batch_size):
                ptr.init_pointee_copy(fill)
                ptr += 1

        # Fill the remainder
        comptime for _ in range(unroll_end, Self.size):
            ptr.init_pointee_copy(fill)
            ptr += 1
        debug_assert(
            ptr == (base + Self.size),
            "error during `InlineArray` initialization , please file a bug",
            " report.",
        )

    @always_inline
    def __init__(
        out self, var *elems: Self.ElementType, __list_literal__: NoneType
    ):
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
        debug_assert[assert_mode="safe"](
            len(elems) == Self.size,
            "InlineArray: expected ",
            Self.size,
            " elements, received ",
            len(elems),
        )
        _inline_array_construction_checks[Self.size]()
        self = Self(uninitialized=True)
        var ptr = self.unsafe_ptr()

        # Move each element into the array storage.
        comptime for i in range(Self.size):
            # Safety: We own the elements in the variadic list.
            ptr.init_pointee_move_from(UnsafePointer(to=elems[i]))
            ptr += 1

        # Do not destroy the elements when their backing storage goes away.
        # FIXME: Why doesn't consume_elements work here?
        elems^._annihilate()

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.ElementType, Copyable):
        """Copy constructs the array from another array.

        Args:
            copy: The array to copy from.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        var copy = arr.copy()  # Creates new array [1, 2, 3]
        ```
        """

        # TODO(MOCO-4058): The `where conforms_to(Self.ElementType, Copyable)`
        # clause above should make the downcasts below redundant, but the
        # compiler does not narrow `Self.ElementType`'s bound from `Movable`
        # to `Copyable` when resolving downstream parametric overloads (e.g.
        # `UnsafePointer.init_pointee_copy[T: Copyable]`). Drop the downcasts
        # once the compiler propagates `where`-clause evidence.
        comptime if is_trivially_copyable[Self.ElementType]():
            self._array = copy._array
        else:
            self = Self(uninitialized=True)
            var base = self.unsafe_ptr()
            for idx in range(Self.size):
                (base + idx).init_pointee_copy(copy.unsafe_get(idx))

    def __init__(out self, *, deinit move: Self):
        """Move constructs the array from another array.

        Args:
            move: The array to move from.

        Notes:
            Moves the elements from the source array into this array.
        """

        comptime if is_trivially_movable[Self.ElementType]():
            self._array = move._array
        else:
            self = Self(uninitialized=True)
            for idx in range(Self.size):
                var other_ptr = move.unsafe_ptr() + idx
                (self.unsafe_ptr() + idx).init_pointee_move_from(other_ptr)

    def __del__(
        deinit self,
    ) where conforms_to(Self.ElementType, ImplicitlyDeletable):
        """Destroys the array's elements."""
        destroy_n(self.unsafe_ptr(), Self.size)

    def destroy_with(
        deinit self, destroy_func: Some[def(var Self.ElementType)], /
    ):
        """Consumes this array and destroys its elements using the provided
        closure.

        This can be used to destroy an `InlineArray` of
        non-`ImplicitlyDeletable` values.

        Args:
            destroy_func: The deinitializing closure called on each array
                element.
        """
        for idx in range(Self.size):
            # TODO(MOCO-4111): `destroy_func` cannot convert to
            # `UnsafePointer.destroy_pointee_with` since `UnsafePointer` is
            # bound on `T: AnyType` but `InlineArray` has `ElementType: Movable`.
            destroy_func(
                __get_address_as_owned_value((self.unsafe_ptr() + idx).address)
            )

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline
    def __getitem__(ref self, idx: Some[Indexer]) -> ref[self] Self.ElementType:
        """Gets a reference to the element at the given index.

        Args:
            idx: The index to access (0 to len-1).

        Returns:
            A reference to the element at the specified index.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        print(arr[0])            # Prints 1 - first element
        print(arr[1])            # Prints 2 - second element
        print(arr[len(arr) - 1]) # Prints 3 - last element
        ```

        Notes:
            This method provides array-style indexing access to elements in the
            InlineArray. The index is bounds-checked at runtime.
        """
        check_bounds(idx, len(self))
        return self._unchecked_get(idx)

    @always_inline
    def __getitem_param__[
        idx: Some[Indexer & ImplicitlyDeletable]
    ](ref self) -> ref[self] Self.ElementType:
        """Gets a reference to the element at the given index with compile-time
        bounds checking.

        Parameters:
            idx: The compile-time constant index to access (0 to len-1).

        Returns:
            A reference to the element at the specified index.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        print(arr[0])            # Prints 1 - first element
        print(arr[1])            # Prints 2 - second element
        print(arr[len(arr) - 1]) # Prints 3 - last element
        ```

        Notes:
            This overload provides array-style indexing with compile-time bounds
            checking. The index must be a compile-time constant value.
        """
        # Can't construct a String here with the index for the error message, as
        # it causes infinite cycles in gpu compilation tests
        comptime assert (
            index(idx) >= 0
        ), "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
        comptime assert index(idx) < Self.size, "index is out of bounds"
        return self._unchecked_get(materialize[idx]())

    @always_inline
    def _unchecked_get(
        ref self, idx: Some[Indexer]
    ) -> ref[self] Self.ElementType:
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer(to=self._array).address,
            index(idx).__mlir_index__(),
        )
        return UnsafePointer[_, origin_of(self)](ptr)[]

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline
    def __len__(self) -> Int:
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

    @always_inline
    def __eq__(
        self, other: Self
    ) -> Bool where conforms_to(Self.ElementType, Equatable):
        """Compares two arrays for equality.

        Args:
            other: The other array to compare against.

        Returns:
            True if all elements are equal, False otherwise.
        """
        comptime for i in range(Self.size):
            if self.unsafe_get(i) != other.unsafe_get(i):
                return False
        return True

    @always_inline
    def __ne__(
        self, other: Self
    ) -> Bool where conforms_to(Self.ElementType, Equatable):
        """Compares two arrays for inequality.

        Args:
            other: The other array to compare against.

        Returns:
            True if any elements are not equal, False otherwise.
        """
        comptime for i in range(Self.size):
            if self.unsafe_get(i) != other.unsafe_get(i):
                return True
        return False

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.ElementType, Hashable):
        """Hashes the elements of the array using the given hasher.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        comptime for i in range(Self.size):
            self.unsafe_get(i).__hash__(hasher)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    def unsafe_get(ref self, idx: Some[Indexer]) -> ref[self] Self.ElementType:
        """Gets a reference to an element without bounds checking.

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
        check_bounds[cpu_default=False](idx, len(self))
        return self._unchecked_get(idx)

    @always_inline
    def unsafe_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Self.ElementType, origin, address_space=address_space
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

    def fill[
        T: Copyable & ImplicitlyDestructible, //
    ](mut self: InlineArray[T, Self.size], value: T):
        """Fills all elements of the array with the given value.

        Parameters:
            T: The element type, must implement `Copyable` and
                `ImplicitlyDestructible`.

        Args:
            value: The value to assign to each element.

        Examples:

        ```mojo
        var arr: InlineArray[Int, 3] = [1, 2, 3]
        arr.fill(0)  # arr is now [0, 0, 0]
        ```
        """
        Span(self).fill(value)

    @always_inline
    def __contains__(
        self, value: Self.ElementType
    ) -> Bool where conforms_to(Self.ElementType, Equatable):
        """Tests if a value is present in the array using the `in` operator.

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
            implement the `Equatable` trait to support equality comparison.
        """
        comptime for i in range(Self.size):
            if self[i] == value:
                return True
        return False

    # ===-------------------------------------------------------------------===#
    # String representation
    # ===-------------------------------------------------------------------===#

    def _write_self_to[
        f: def(Self.ElementType, mut Some[Writer]) thin
    ](self, mut writer: Some[Writer]) where conforms_to(
        Self.ElementType, Writable
    ):
        var index = 0

        @parameter
        def iterate(mut w: Some[Writer]) raises StopIteration:
            if index >= Self.size:
                raise StopIteration()
            f(self.unsafe_get(index), w)
            index += 1

        fmt.write_sequence_to[ElementFn=iterate](writer)
        _ = index

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.ElementType, Writable):
        """Writes the InlineArray representation to a Writer.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[f=fmt.write_to[Self.ElementType]](writer)

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.ElementType, Writable):
        """Writes the repr representation of this InlineArray to a Writer.

        Args:
            writer: The object to write to.
        """

        @parameter
        def write_fields(mut w: Some[Writer]):
            self._write_self_to[f=fmt.write_repr_to[Self.ElementType]](w)

        fmt.FormatStruct(writer, "InlineArray").params(
            fmt.TypeNames[Self.ElementType](),
            Self.size,
        ).fields[FieldsFn=write_fields]()

    def __iter__(
        var self,
    ) -> Self.IteratorOwnedType where conforms_to(
        Self.ElementType, ImplicitlyDeletable
    ):
        """Consume the array and return an iterator over its elements.

        Returns:
            An iterator that owns the array's elements.
        """
        return Self.IteratorOwnedType(
            rebind_var[
                InlineArray[
                    downcast[Self.ElementType, ImplicitlyDeletable],
                    Self.size,
                ]
            ](self^)
        )

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the array, returning immutable references.

        Returns:
            An iterator of immutable references to the array elements.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
        comptime assert conforms_to(
            Self.ElementType, Copyable
        ), "InlineArray iteration requires the element to be `Copyable`."
        return {
            0,
            rebind[
                Pointer[
                    InlineArray[
                        downcast[Self.ElementType, Copyable], Self.size
                    ],
                    origin_of(self),
                ]
            ](Pointer(to=self)),
        }

    def __reversed__(
        ref self,
    ) -> _InlineArrayIter[
        downcast[Self.ElementType, Copyable],
        Self.size,
        origin_of(self),
        False,
    ]:
        """Iterate over elements of the array in reverse order, returning
        immutable references.

        Returns:
            An iterator of immutable references to the array elements in reverse
            order.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
        comptime assert conforms_to(
            Self.ElementType, Copyable
        ), "InlineArray iteration requires the element to be `Copyable`."
        return _InlineArrayIter[forward=False](
            Self.size,
            rebind[
                Pointer[
                    InlineArray[
                        downcast[Self.ElementType, Copyable], Self.size
                    ],
                    origin_of(self),
                ]
            ](Pointer(to=self)),
        )
