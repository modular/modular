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
"""Provides a fixed-size array implementation with compile-time size checking.

The `InlineArray` type represents a fixed-size sequence of homogeneous elements where the size
is determined at compile time. It provides efficient memory layout and bounds checking while
maintaining type safety.  The `InlineArray` type is part of the `prelude` module and therefore
does not need to be imported in order to use it.

Example:
```mojo
    # Create an array of 3 integers
    var arr = InlineArray[Int, 3](1, 2, 3)

    # Access elements
    print(arr[0])  # Prints 1

    # Fill with a value
    var filled = InlineArray[Int, 5](fill=42)
```

Note:
- For historical reasons, destructors are not run by default on the elements of an `InlineArray`.
  This can be controlled with the `run_destructors` parameter. In the future, this will
  default to `True` and the `run_destructors` parameter will be removed.
"""

from collections._index_normalization import normalize_index
from sys.intrinsics import _type_is_eq

from memory import UnsafePointer
from memory.maybe_uninitialized import UnsafeMaybeUninitialized

# ===-----------------------------------------------------------------------===#
# Array
# ===-----------------------------------------------------------------------===#


fn _inline_array_construction_checks[size: Int]():
    """Checks if the properties in `InlineArray` are valid.

    Validity right now is just ensuring the number of elements is > 0.

    Parameters:
        size: The number of elements.
    """
    constrained[size > 0, "number of elements in `InlineArray` must be > 0"]()


@value
struct InlineArray[
    ElementType: CollectionElement,
    size: Int,
    *,
    run_destructors: Bool = False,
](Sized, Movable, Copyable, ExplicitlyCopyable):
    """A fixed-size sequence of homogeneous elements where size is a constant expression.

    InlineArray provides a fixed-size array implementation with compile-time size checking.
    The array size is determined at compile time and cannot be changed. Elements must
    implement the CollectionElement trait.

    Parameters:
        ElementType: The type of the elements in the array. Must implement `CollectionElement`.
        size: The size of the array. Must be a positive integer constant.
        run_destructors: Whether to run destructors on the elements. Defaults to `False` for
            backwards compatibility. Will default to `True` in the future.

    Example:
    ```mojo
        # Create array of 3 integers
        var arr = InlineArray[Int, 3](1, 2, 3)

        # Create array filled with value
        var filled = InlineArray[Int, 5](fill=42)

        # Access elements
        print(arr[0])  # Prints 1
    ```
    """

    # Fields
    alias type = __mlir_type[
        `!pop.array<`, size.value, `, `, Self.ElementType, `>`
    ]
    var _array: Self.type
    """The underlying storage for the array."""

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """This constructor will always cause a compile time error if used.
        It is used to steer users away from uninitialized memory.
        """
        constrained[
            False,
            (
                "Initialize with either a variadic list of arguments, a default"
                " fill element or pass the keyword argument"
                " 'unsafe_uninitialized'."
            ),
        ]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    fn __init__(out self, *, unsafe_uninitialized: Bool):
        """Create an InlineArray with uninitialized memory.

        This constructor is unsafe and should be used with caution. The array elements
        will be uninitialized and accessing them before initialization is undefined behavior.

        Args:
            unsafe_uninitialized: A boolean to indicate if the array should be initialized.
                Always set to `True` (it's not actually used inside the constructor).

        Example:
            ```mojo
                var uninitialized_array = InlineArray[Int, 10](unsafe_uninitialized=True)
            ```
        """
        _inline_array_construction_checks[size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    fn __init__(
        mut self,
        *,
        owned unsafe_assume_initialized: InlineArray[
            UnsafeMaybeUninitialized[Self.ElementType], Self.size
        ],
    ):
        """Constructs an `InlineArray` from an `InlineArray` of `UnsafeMaybeUninitialized`.

        This constructor assumes all elements in the input array are initialized. Using
        uninitialized elements results in undefined behavior, even for types that are
        valid for any bit pattern (e.g. Int or Float).

        Args:
            unsafe_assume_initialized: The array of `UnsafeMaybeUninitialized` elements.
                All elements must be initialized.

        Warning:
            This is an unsafe constructor. Only use it if you are certain all elements
            are properly initialized.
        """

        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        for i in range(Self.size):
            unsafe_assume_initialized[i].unsafe_ptr().move_pointee_into(
                self.unsafe_ptr() + i
            )

    @always_inline
    @implicit
    fn __init__(out self, fill: Self.ElementType):
        """Constructs an array where each element is initialized to the supplied value.

        Args:
            fill: The element value to fill each index with.

        Example:
            ```mojo
            var filled = InlineArray[Int, 5](fill=42)  # [42, 42, 42, 42, 42]
            ```
        """
        _inline_array_construction_checks[size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        @parameter
        for i in range(size):
            var ptr = UnsafePointer.address_of(self.unsafe_get(i))
            ptr.init_pointee_copy(fill)

    @always_inline
    @implicit
    fn __init__(out self, owned *elems: Self.ElementType):
        """Constructs an array from a variadic list of elements.

        Args:
            elems: The elements to initialize the array with. Must match the array size.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)  # [1, 2, 3]
            ```
        """

        self = Self(storage=elems^)

    @always_inline
    fn __init__(
        mut self,
        *,
        owned storage: VariadicListMem[Self.ElementType, _],
    ):
        """Construct an array from a low-level internal representation.

        Args:
            storage: The variadic list storage to construct from. Must match array size.
        """

        debug_assert(len(storage) == size, "Elements must be of length size")
        _inline_array_construction_checks[size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        # Move each element into the array storage.
        @parameter
        for i in range(size):
            var eltptr = UnsafePointer.address_of(self.unsafe_get(i))
            UnsafePointer.address_of(storage[i]).move_pointee_into(eltptr)

        # Do not destroy the elements when their backing storage goes away.
        __disable_del storage

    fn copy(self) -> Self:
        """Creates a deep copy of the array.

        Returns:
            A new array containing copies of all elements.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            var copy = arr.copy()  # Creates new array [1, 2, 3]
            ```
        .
        """

        var copy = Self(unsafe_uninitialized=True)

        for idx in range(size):
            var ptr = copy.unsafe_ptr() + idx
            ptr.init_pointee_copy(self[idx])

        return copy^

    fn __copyinit__(out self, other: Self):
        """Copy constructs the array from another array.

        Creates a deep copy by copying each element individually.

        Args:
            other: The array to copy from.
        """

        self = other.copy()

    fn __del__(owned self):
        """Deallocates the array and destroys its elements.

        This destructor is called automatically when the array goes out of scope.
        If the array's `run_destructors` parameter is `True`, it will call the destructor
        on each element in the array before deallocating the array's memory.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            # arr's destructor is called automatically when it goes out of scope
            ```
        """

        @parameter
        if Self.run_destructors:

            @parameter
            for idx in range(size):
                var ptr = self.unsafe_ptr() + idx
                ptr.destroy_pointee()

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[I: Indexer](ref self, idx: I) -> ref [self] Self.ElementType:
        """Gets a reference to the element at the given index.

        This method provides array-style indexing access to elements in the InlineArray.
        It supports both positive indices starting from 0 and negative indices counting
        backwards from the end of the array. The index is bounds-checked at runtime.

        Parameters:
            I: The type parameter representing the index type, must implement Indexer trait.

        Args:
            idx: The index to access. Can be positive (0 to len-1) or negative (-len to -1).

        Returns:
            A reference to the element at the specified index.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            print(arr[0])   # Prints 1 - first element
            print(arr[1])   # Prints 2 - second element
            print(arr[-1])  # Prints 3 - last element
            print(arr[-2])  # Prints 2 - second to last element
            ```
        .
        """
        var normalized_index = normalize_index["InlineArray"](idx, len(self))
        return self.unsafe_get(normalized_index)

    @always_inline
    fn __getitem__[
        I: Indexer, //, idx: I
    ](ref self) -> ref [self] Self.ElementType:
        """Gets a reference to the element at the given index with compile-time bounds checking.

        This overload provides array-style indexing with compile-time bounds checking. The index
        must be a compile-time constant value. It supports both positive indices starting from 0
        and negative indices counting backwards from the end of the array.

        Parameters:
            I: The type parameter representing the index type, must implement Indexer trait.
            idx: The compile-time constant index to access. Can be positive (0 to len-1)
                or negative (-len to -1).

        Returns:
            A reference to the element at the specified index.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            print(arr[0])   # Prints 1 - first element
            print(arr[-1])  # Prints 3 - last element
            ```
        .
        """
        constrained[-size <= Int(idx) < size, "Index must be within bounds."]()
        alias normalized_index = normalize_index["InlineArray"](idx, size)
        return self.unsafe_get(normalized_index)

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the array.

        The length is a compile-time constant value determined by the size parameter
        used when creating the array.

        Returns:
            The size of the array as an Int.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            print(len(arr))  # Prints 3
            ```
            .
        """
        return size

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn unsafe_get[I: Indexer](ref self, idx: I) -> ref [self] Self.ElementType:
        """Gets a reference to an element without bounds checking.

        This is an unsafe method that skips bounds checking for performance.
        Users should prefer `__getitem__` instead for safety.

        Parameters:
            I: A type parameter representing the index type, must implement Indexer trait.

        Args:
            idx: The index of the element to get. Must be non-negative and in bounds.
                Using an invalid index will cause undefined behavior.

        Returns:
            A reference to the element at the given index.

        Warning:
            This is an unsafe method. No bounds checking is performed.
            Using an invalid index will cause undefined behavior.
            Negative indices are not supported.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            print(arr.unsafe_get(0))  # Prints 1
            ```
            .
        """
        var i = index(idx)
        debug_assert(
            0 <= Int(i) < size,
            " InlineArray.unsafe_get() index out of bounds: ",
            Int(idx),
            " should be less than: ",
            size,
        )
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer.address_of(self._array).address,
            i,
        )
        return UnsafePointer(ptr)[]

    @always_inline
    fn unsafe_ptr(
        ref self,
    ) -> UnsafePointer[
        Self.ElementType,
        mut = Origin(__origin_of(self)).is_mutable,
        origin = __origin_of(self),
    ]:
        """Gets an unsafe pointer to the underlying array storage.

        Returns a raw pointer to the array's memory that can be used for direct
        memory access. The pointer inherits mutability from the array reference.

        Returns:
            An `UnsafePointer` to the underlying array storage. The pointer's mutability
            matches that of the array reference.

        Warning:
            This is an unsafe method. The returned pointer:
            - Becomes invalid if the array is moved
            - Must not be used to access memory outside array bounds
            - Must be refreshed after any operation that could move the array

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            var ptr = arr.unsafe_ptr()
            print(ptr[0])  # Prints 1
            ```
            .
        """
        return UnsafePointer.address_of(self._array).bitcast[Self.ElementType]()

    @always_inline
    fn __contains__[
        T: EqualityComparableCollectionElement, //
    ](self: InlineArray[T, size], value: T) -> Bool:
        """Tests if a value is present in the array using the `in` operator.

        This method enables using the `in` operator to check if a value exists in the array.
        It performs a linear search comparing each element for equality with the given value.
        The element type must implement both `EqualityComparable` and `CollectionElement` traits
        to support equality comparison.

        Parameters:
            T: The element type, must implement both `EqualityComparable` and `CollectionElement`.

        Args:
            value: The value to search for.

        Returns:
            True if the value is found in any position in the array, False otherwise.

        Example:
            ```mojo
            var arr = InlineArray[Int, 3](1, 2, 3)
            print(3 in arr)  # Prints True - value exists
            print(4 in arr)  # Prints False - value not found
            ```
            .
        """

        @parameter
        for i in range(size):
            if self[i] == value:
                return True
        return False
