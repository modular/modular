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
"""Implement a generic unsafe pointer type.

These APIs are imported automatically, just like builtins.
"""

from sys import align_of, is_gpu, is_nvidia_gpu, size_of
from sys.intrinsics import (
    gather,
    scatter,
    strided_load,
    strided_store,
)

from builtin.simd import _simd_construction_checks
from memory.memory import _free, _malloc

from python import PythonObject

# ===----------------------------------------------------------------------=== #
# UnsafePointer
# ===----------------------------------------------------------------------=== #


@always_inline
fn _default_invariant[mut: Bool]() -> Bool:
    return is_gpu() and mut == False


alias _must_be_mut_err = "UnsafePointer must be mutable for this operation"


@register_passable("trivial")
struct UnsafePointer[
    type: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin],
](
    Comparable,
    Defaultable,
    ImplicitlyBoolable,
    ImplicitlyCopyable,
    Intable,
    Movable,
    Stringable,
    Writable,
):
    """`UnsafePointer[T]` represents an indirect reference to one or more values
    of type `T` consecutively in memory, and can refer to uninitialized memory.

    Because it supports referring to uninitialized memory, it provides unsafe
    methods for initializing and destroying instances of `T`, as well as methods
    for accessing the values once they are initialized. You should instead use
    safer pointers when possible.

    Important things to know:

    - This pointer is unsafe and nullable. No bounds checks; reading before
      writing is undefined.
    - It does not own existing memory. When memory is heap-allocated with
      `alloc()`, you must call `free()`.
    - For simple read/write access, use `(ptr + i)[]` or `ptr[i]` where `i`
      is the offset size.
    - For SIMD operations on numeric data, use `UnsafePointer[Scalar[DType.xxx]]`
      with `load[dtype=DType.xxx]()` and `store[dtype=DType.xxx]()`.

    Key APIs:

    - `alloc()`: Allocates heap space for the specified number of
      elements (contiguous). Must be paired with `free()`.
    - `free()`: Frees memory previously allocated by `alloc()`. Do not call on
      pointers that were not allocated by `alloc()`.
    - `offset(i)` / `+ i` / `- i`: Pointer arithmetic. Returns a new pointer
      shifted by `i` elements. No bounds checking.
    - `[]` or `[i]`: Dereference to a reference of the pointee (or at
      offset `i`). Only valid if the memory at that location is initialized.
    - `load()`: Loads `width` elements starting at `offset` (default 0) as
      `SIMD[dtype, width]` from `UnsafePointer[Scalar[dtype]]`. Pass
      `alignment` when data is not naturally aligned.
    - `store()`: Stores `val: SIMD[dtype, width]` at `offset` into
      `UnsafePointer[Scalar[dtype]]`. Requires a mutable pointer.
    - `destroy_pointee()` / `take_pointee()` / `move_pointee_into(dst)`:
      Explicitly end the lifetime of the current pointee or move it out and
      into another pointer without running an extra copy. Use these to manage
      lifecycles when working with uninitialized memory patterns.

    For more information see [Unsafe
    pointers](/mojo/manual/pointers/unsafe-pointers) in the Mojo Manual. For a
    comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/).

    Examples:

    Element-wise store and load (width = 1):

    ```mojo
    var p = UnsafePointer[Scalar[DType.float32]].alloc(4)
    for i in range(4):
        p.store(i, Scalar[DType.float32](Float32(i)))
    var v = p.load(2)
    print(v[0])  # => 2.0
    p.free()
    ```

    Vectorized store and load (width = 4):

    ```mojo
    var p = UnsafePointer[Scalar[DType.int32]].alloc(8)
    var vec = SIMD[DType.int32, 4](1, 2, 3, 4)
    p.store(0, vec)
    var out = p.load[width=4](0)
    print(out)  # => [1, 2, 3, 4]
    p.free()
    ```

    Pointer arithmetic and dereference:

    ```mojo
    var p = UnsafePointer[Int32].alloc(3)
    (p + 0)[] = 10  # offset by 0 elements, then dereference to write
    (p + 1)[] = 20  # offset +1 element, then dereference to write
    p[2] = 30  # equivalent offset/dereference with brackets (via __getitem__)
    var second = p[1]  # reads the element at index 1
    print(second, p[2])  # => 20 30
    p.free()
    ```

    Point to a value on the stack:

    ```mojo
    var foo: Int = 123
    var p = UnsafePointer(to=foo)
    print(p[])  # => 123
    # Don't call `free()` because the value was not heap-allocated
    # Mojo will destroy it when the `foo` lifetime ends
    ```

    Parameters:
        type: The type the pointer points to.
        address_space: The address space associated with the UnsafePointer allocated memory.
        mut: Whether the origin is mutable.
        origin: The origin of the memory being addressed.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    # Fields
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        type,
        `, `,
        address_space._value._mlir_value,
        `>`,
    ]
    """The underlying pointer type."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var address: Self._mlir_type
    """The underlying pointer."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Create a null pointer."""
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @doc_private
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        """Create a pointer from a low-level pointer primitive.

        Args:
            value: The MLIR value of the pointer to construct with.
        """
        self.address = value

    @always_inline("nodebug")
    fn __init__(
        out self, *, ref [origin, address_space._value._mlir_value]to: type
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @always_inline("builtin")
    @implicit
    fn __init__(
        out self, other: UnsafePointer[type, address_space=address_space, **_]
    ):
        """Exclusivity parameter cast a pointer.

        Args:
            other: Pointer to cast.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[_type = Self._mlir_type](
            other.address
        )

    fn __init__(
        out self: UnsafePointer[type, mut=mut, origin=origin],
        *,
        ref [origin]unchecked_downcast_value: PythonObject,
    ):
        """Downcast a `PythonObject` known to contain a Mojo object to a pointer.

        This operation is only valid if the provided Python object contains
        an initialized Mojo object of matching type.

        Args:
            unchecked_downcast_value: The Python object to downcast from.
        """

        self = unchecked_downcast_value.unchecked_downcast_value_ptr[type]()

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn alloc[
        *, alignment: Int = align_of[type]()
    ](
        count: Int,
    ) -> UnsafePointer[
        type,
        address_space = AddressSpace.GENERIC,
        # This is a newly allocated pointer, so should not alias anything
        # already existing.
        origin = MutableOrigin.empty,
    ]:
        """Allocates contiguous storage for `count` elements of `type`
        with compile-time alignment `alignment`.

        - The returned memory is uninitialized; reading before writing is undefined.
        - The returned pointer has an empty mutable origin; you must call `free()`
          to release it.
        - `count` must be positive and `size_of[type]()` must be > 0.

        Example:

        ```mojo
        var p = UnsafePointer[Scalar[DType.int32]].alloc(4)
        p.store(0, SIMD[DType.int32, 1](42))
        p.store(1, SIMD[DType.int32, 1](7))
        p.store(2, SIMD[DType.int32, 1](9))
        var a = p.load(0)
        print(a[0], p.load(1)[0], p.load(2)[0])
        p.free()
        ```

        Parameters:
            alignment: The alignment of the allocation.

        Args:
            count: Number of elements to allocate.

        Returns:
            Pointer to the newly allocated uninitialized array.
        """
        alias size_of_t = size_of[type]()
        constrained[size_of_t > 0, "size must be greater than zero"]()
        return _malloc[type, alignment=alignment](size_of_t * count)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref [origin, address_space] type:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """

        # We're unsafe, so we can have unsafe things.
        alias _ref_type = Pointer[type, origin, address_space]
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type = _ref_type._mlir_type](
                self.address
            )
        )

    @always_inline("nodebug")
    fn offset[I: Indexer, //](self, idx: I) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The offset of the new pointer.

        Returns:
            The new constructed UnsafePointer.
        """
        return __mlir_op.`pop.offset`(self.address, index(idx)._mlir_value)

    @always_inline("nodebug")
    fn __getitem__[
        I: Indexer, //
    ](self, offset: I) -> ref [origin, address_space] type:
        """Return a reference to the underlying data, offset by the given index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset reference.
        """
        return (self + offset)[]

    @always_inline("nodebug")
    fn __add__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self.offset(offset)

    @always_inline
    fn __sub__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self + (-1 * index(offset))

    @always_inline
    fn __iadd__[I: Indexer, //](mut self, offset: I):
        """Add an offset to this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self + offset

    @always_inline
    fn __isub__[I: Indexer, //](mut self, offset: I):
        """Subtract an offset from this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self - offset

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) >= Int(rhs)

    @always_inline("builtin")
    fn __merge_with__[
        other_type: __type_of(
            UnsafePointer[
                type,
                address_space=address_space,
                mut=_,
                origin=_,
            ]
        ),
    ](self) -> UnsafePointer[
        type=type,
        mut = mut & other_type.origin.mut,
        origin = __origin_of(origin, other_type.origin),
        address_space=address_space,
    ]:
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return self.address  # allow kgen.pointer to convert.

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return Int(self) != 0

    @always_inline
    fn __as_bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return self.__bool__()

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return Int(mlir_value=__mlir_op.`pop.pointer_to_index`(self.address))

    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the pointer.

        Returns:
            The string representation of the pointer.
        """
        return hex(Int(self))

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this pointer address to the provided Writer.

        Args:
            writer: The object to write to.
        """

        # TODO: Avoid intermediate String allocation.
        writer.write(String(self))

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn as_noalias_ptr(self) -> Self:
        """Cast the pointer to a new pointer that is known not to locally alias
        any other pointer. In other words, the pointer transitively does not
        alias any other memory value declared in the local function context.

        This information is relayed to the optimizer. If the pointer does
        locally alias another memory value, the behaviour is undefined.

        Returns:
            A noalias pointer.
        """
        return __mlir_op.`pop.noalias_pointer_cast`(self.address)

    @always_inline("nodebug")
    fn load[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointer[Scalar[dtype], **_]) -> SIMD[dtype, width]:
        """Loads `width` elements from the value the pointer points to.

        Use `alignment` to specify minimal known alignment in bytes; pass a
        smaller value (such as 1) if loading from packed/unaligned memory. The
        `volatile`/`invariant` flags control reordering and common-subexpression
        elimination semantics for special cases.

        Example:

        ```mojo
        var p = UnsafePointer[Scalar[DType.int32]].alloc(8)
        p.store(0, SIMD[DType.int32, 4](1, 2, 3, 4))
        var v = p.load[width=4]()
        print(v)  # => [1, 2, 3, 4]
        p.free()
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of the SIMD vector.
            width: The number of elements to load.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.
            invariant: Whether the load is from invariant memory.

        Returns:
            The loaded SIMD vector.
        """
        _simd_construction_checks[dtype, width]()
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()
        constrained[
            not volatile or volatile ^ invariant,
            "both volatile and invariant cannot be set at the same time",
        ]()

        @parameter
        if is_nvidia_gpu() and size_of[dtype]() == 1 and alignment == 1:
            # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
            # regardless of the alignment that is passed. This causes issues if
            # this method is called on an unaligned pointer.
            # TODO #37823 We can make this smarter when we add an `aligned`
            # trait to the pointer class.
            var v = SIMD[dtype, width]()

            # intentionally don't unroll, otherwise the compiler vectorizes
            for i in range(width):
                v[i] = __mlir_op.`pop.load`[
                    alignment = alignment._mlir_value,
                    isVolatile = volatile._mlir_value,
                    isInvariant = invariant._mlir_value,
                ]((self + i).address)
            return v

        var address = self.bitcast[SIMD[dtype, width]]().address

        return __mlir_op.`pop.load`[
            alignment = alignment._mlir_value,
            isVolatile = volatile._mlir_value,
            isInvariant = invariant._mlir_value,
        ](address)

    @always_inline("nodebug")
    fn load[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointer[Scalar[dtype], **_], offset: Scalar) -> SIMD[
        dtype, width
    ]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be integer.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        constrained[offset.dtype.is_integral(), "offset must be integer"]()
        return self.offset(Int(offset)).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ]()

    @always_inline("nodebug")
    fn load[
        I: Indexer,
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointer[Scalar[dtype], **_], offset: I) -> SIMD[dtype, width]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.offset(offset).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ]()

    @always_inline("nodebug")
    fn store[
        I: Indexer,
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: UnsafePointer[Scalar[dtype], **_],
        offset: I,
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be integer.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        constrained[mut, _must_be_mut_err]()
        self.offset(offset).store[alignment=alignment, volatile=volatile](val)

    @always_inline("nodebug")
    fn store[
        dtype: DType,
        offset_type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: UnsafePointer[Scalar[dtype], **_],
        offset: Scalar[offset_type],
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            offset_type: The data type of the offset value.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        constrained[mut, _must_be_mut_err]()
        constrained[offset_type.is_integral(), "offset must be integer"]()
        self.offset(Int(offset))._store[alignment=alignment, volatile=volatile](
            val
        )

    @always_inline("nodebug")
    fn store[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](self: UnsafePointer[Scalar[dtype], **_], val: SIMD[dtype, width]):
        """Stores a single element value `val` at element offset 0.

        Specify `alignment` when writing to packed/unaligned memory. Requires a
        mutable pointer. For writing at an element offset, use the overloads
        that accept an index or scalar offset.

        Example:

        ```mojo
        var p = UnsafePointer[Scalar[DType.float32]].alloc(4)
        var vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
        p.store(vec)
        var out = p.load[width=4]()
        print(out)  # => [1.0, 2.0, 3.0, 4.0]
        p.free()
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The number of elements to store.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.

        Args:
            val: The SIMD value to store.
        """
        constrained[mut, _must_be_mut_err]()
        self._store[alignment=alignment, volatile=volatile](val)

    @always_inline("nodebug")
    fn _store[
        dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](self: UnsafePointer[Scalar[dtype], **_], val: SIMD[dtype, width]):
        constrained[mut, _must_be_mut_err]()
        constrained[width > 0, "width must be a positive integer value"]()
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()

        __mlir_op.`pop.store`[
            alignment = alignment._mlir_value,
            isVolatile = volatile._mlir_value,
        ](val, self.bitcast[SIMD[dtype, width]]().address)

    @always_inline("nodebug")
    fn strided_load[
        dtype: DType, T: Intable, //, width: Int
    ](self: UnsafePointer[Scalar[dtype], **_], stride: T) -> SIMD[dtype, width]:
        """Performs a strided load of the SIMD vector.

        Parameters:
            dtype: DType of returned SIMD value.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            stride: The stride between loads.

        Returns:
            A vector which is stride loaded.
        """
        return strided_load(
            self, Int(stride), SIMD[DType.bool, width](fill=True)
        )

    @always_inline("nodebug")
    fn strided_store[
        dtype: DType,
        T: Intable, //,
        width: Int = 1,
    ](
        self: UnsafePointer[Scalar[dtype], **_],
        val: SIMD[dtype, width],
        stride: T,
    ):
        """Performs a strided store of the SIMD vector.

        Parameters:
            dtype: DType of `val`, the SIMD value to store.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            val: The SIMD value to store.
            stride: The stride between stores.
        """
        constrained[mut, _must_be_mut_err]()
        strided_store(
            val, self, Int(stride), SIMD[DType.bool, width](fill=True)
        )

    @always_inline("nodebug")
    fn gather[
        dtype: DType, //,
        *,
        width: Int = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafePointer[Scalar[dtype], **_],
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
        default: SIMD[dtype, width] = 0,
    ) -> SIMD[dtype, width]:
        """Gathers a SIMD vector from offsets of the current pointer.

        This method loads from memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        or takes from the `default` SIMD vector, depending on the values of
        the `mask` SIMD vector.

        If a mask element is `True`, the respective result element is given
        by the current pointer and the `offset` SIMD vector; otherwise, the
        result element is taken from the `default` SIMD vector.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            dtype: DType of the return SIMD.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to gather from.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to load from memory or to take from the
                `default` SIMD vector.
            default: The SIMD vector providing default values to be taken
                where the `mask` SIMD vector is `False`.

        Returns:
            The SIMD vector containing the gathered values.
        """
        constrained[
            offset.dtype.is_integral(),
            "offset type must be an integral type",
        ]()
        constrained[
            alignment.is_power_of_two(),
            "alignment must be a power of two integer value",
        ]()

        var base = offset.cast[DType.index]().fma(size_of[dtype](), Int(self))
        return gather(base, mask, default, alignment)

    @always_inline("nodebug")
    fn scatter[
        dtype: DType, //,
        *,
        width: Int = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafePointer[Scalar[dtype], **_],
        offset: SIMD[_, width],
        val: SIMD[dtype, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
    ):
        """Scatters a SIMD vector into offsets of the current pointer.

        This method stores at memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        depending on the values of the `mask` SIMD vector.

        If a mask element is `True`, the respective element in the `val` SIMD
        vector is stored at the memory address defined by the current pointer
        and the `offset` SIMD vector; otherwise, no action is taken for that
        element in `val`.

        If the same offset is targeted multiple times, the values are stored
        in the order they appear in the `val` SIMD vector, from the first to
        the last element.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            dtype: DType of `value`, the result SIMD buffer.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to scatter into.
            val: The SIMD vector containing the values to be scattered.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to store at memory or not.
        """
        constrained[mut, _must_be_mut_err]()
        constrained[
            offset.dtype.is_integral(),
            "offset type must be an integral type",
        ]()
        constrained[
            alignment.is_power_of_two(),
            "alignment must be a power of two integer value",
        ]()

        var base = offset.cast[DType.index]().fma(size_of[dtype](), Int(self))
        scatter(val, base, mask, alignment)

    @always_inline
    fn free(self: UnsafePointer[_, address_space = AddressSpace.GENERIC, **_]):
        """Free the memory referenced by the pointer."""
        _free(self)

    @always_inline("builtin")
    fn bitcast[
        T: AnyType = Self.type,
    ](self) -> UnsafePointer[
        T,
        address_space=address_space,
        mut=mut,
        origin=origin,
    ]:
        """Bitcasts a UnsafePointer to a different type.

        Parameters:
            T: The target type.

        Returns:
            A new UnsafePointer object with the specified type and the same address,
            as the original UnsafePointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointer[
                T,
                address_space=address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn origin_cast[
        target_mut: Bool = Self.mut,
        target_origin: Origin[target_mut] = Origin[target_mut].cast_from[
            Self.origin
        ],
    ](self) -> UnsafePointer[
        type,
        address_space=address_space,
        mut=target_mut,
        origin=target_origin,
    ]:
        """Changes the origin or mutability of a pointer.

        Parameters:
            target_mut: Whether the origin is mutable.
            target_origin: Origin of the destination pointer.

        Returns:
            A new UnsafePointer object with the same type and the same address,
            as the original UnsafePointer and the new specified mutability and origin.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointer[
                type,
                address_space=address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> UnsafePointer[
        type,
        address_space=target_address_space,
        mut=mut,
        origin=origin,
    ]:
        """Casts an UnsafePointer to a different address space.

        Parameters:
            target_address_space: The address space of the result.

        Returns:
            A new UnsafePointer object with the same type and the same address,
            as the original UnsafePointer and the new address space.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointer[
                type,
                address_space=target_address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline
    fn destroy_pointee(
        self: UnsafePointer[type, address_space = AddressSpace.GENERIC, **_]
    ):
        """Destroy the pointed-to value.

        The pointer must not be null, and the pointer memory location is assumed
        to contain a valid initialized instance of `type`.  This is equivalent to
        `_ = self.take_pointee()` but doesn't require `Movable` and is
        more efficient because it doesn't invoke `__moveinit__`.

        """
        constrained[mut, _must_be_mut_err]()
        _ = __get_address_as_owned_value(self.address)

    @always_inline
    fn take_pointee[
        T: Movable, //,
    ](self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_]) -> T:
        """Move the value at the pointer out, leaving it uninitialized.

        The pointer must not be null, and the pointer memory location is assumed
        to contain a valid initialized instance of `T`.

        This performs a _consuming_ move, ending the origin of the value stored
        in this pointer memory location. Subsequent reads of this pointer are
        not valid. If a new valid value is stored using `init_pointee_move()`, then
        reading from this pointer becomes valid again.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Returns:
            The value at the pointer.
        """
        constrained[mut, _must_be_mut_err]()
        return __get_address_as_owned_value(self.address)

    # TODO: Allow overloading on more specific traits
    @always_inline
    fn init_pointee_move[
        T: Movable, //,
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
        var value: T,
    ):
        """Emplace a new value into the pointer location, moving from `value`.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_copy`, this avoids an extra copy on
        the caller side when the value is an `owned` rvalue.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            value: The value to emplace.
        """
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(self.address) = value^

    @always_inline
    fn init_pointee_copy[
        T: Copyable, //,
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
        value: T,
    ):
        """Emplace a copy of `value` into the pointer location.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_move`, this avoids an extra move on
        the callee side when the value must be copied.

        Parameters:
            T: The type the pointer points to, which must be `Copyable`.

        Args:
            value: The value to emplace.
        """
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(self.address) = value.copy()

    @always_inline
    fn init_pointee_explicit_copy[
        T: Copyable, //
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
        value: T,
    ):
        """Emplace a copy of `value` into this pointer location.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_move`, this avoids an extra move on
        the callee side when the value must be copied.

        Parameters:
            T: The type the pointer points to, which must be
               `Copyable`.

        Args:
            value: The value to emplace.
        """
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(self.address) = value.copy()

    @always_inline
    fn move_pointee_into[
        T: Movable, //,
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
        dst: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
    ):
        """Moves the value `self` points to into the memory location pointed to by
        `dst`.

        This performs a consuming move (using `__moveinit__()`) out of the
        memory location pointed to by `self`. Subsequent reads of this
        pointer are not valid unless and until a new, valid value has been
        moved into this pointer's memory location using `init_pointee_move()`.

        This transfers the value out of `self` and into `dest` using at most one
        `__moveinit__()` call.

        **Safety:**

        * `self` must be non-null
        * `self` must contain a valid, initialized instance of `T`
        * `dst` must not be null
        * The contents of `dst` should be uninitialized. If `dst` was
            previously written with a valid value, that value will be be
            overwritten and its destructor will NOT be run.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            dst: Destination pointer that the value will be moved into.
        """
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(
            dst.address
        ) = __get_address_as_owned_value(self.address)


alias OpaquePointer = UnsafePointer[NoneType]
"""An opaque pointer, equivalent to the C `void*` type."""
