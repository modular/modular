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
"""Implements unsafe pointer types for manual memory management.

This module provides `UnsafePointer` and related type aliases for direct memory
manipulation with explicit control over mutability, origins, and address spaces.
It includes the `alloc()` function for heap allocation and comprehensive methods
for loading, storing, and managing pointer lifetimes. These types enable
low-level memory operations, interfacing with C code, and building custom data
structures.
"""

from std.sys import align_of, is_apple_gpu, is_gpu, is_nvidia_gpu, size_of
from std.sys.intrinsics import (
    gather,
    scatter,
    strided_load,
    strided_store,
    unlikely,
)

from std.builtin.device_passable import DevicePassable
from std.builtin.rebind import downcast
from std.builtin.format_int import _write_int
from std.builtin.simd import _simd_construction_checks
from std.collections import OptionalReg
from std.format._utils import FormatStruct, Named, TypeNames
from std.reflection import reflect
from std.memory import memcpy
from std.memory.memory import _free, _malloc
from std.memory import UnsafeMaybeUninit
from std.memory._poison import _check_not_poison, _check_not_poison_masked
from std.os import abort
from std.python import PythonObject
from std.utils._nicheable import (
    UnsafeSingleNicheable,
    UnsafeCustomNicheStorage,
    NicheStorageTraits,
)


@always_inline
def _default_invariant[mut: Bool]() -> Bool:
    return is_gpu() and mut == False


# ===----------------------------------------------------------------------=== #
# Nicheable utilities
# ===----------------------------------------------------------------------=== #


struct _Null[
    type: AnyType = NoneType, address_space: AddressSpace = AddressSpace.GENERIC
](Defaultable, Intable, TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]

    var address: Self._mlir_type

    @always_inline("builtin")
    def __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @always_inline("nodebug")
    def __int__(self) -> Int:
        return Int(mlir_value=__mlir_op.`pop.pointer_to_index`(self.address))


struct _UnsafePointerNicheStorage[
    type: AnyType,
    address_space: AddressSpace,
](NicheStorageTraits):
    """Custom niche backing for `UnsafePointer` that lowers directly to
    `kgen.pointer` instead of `pop.array<1, kgen.pointer>`."""

    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]

    var address: Self._mlir_type

    @always_inline
    def __init__(out self):
        self.address = _Null[Self.type, Self.address_space]().address

    @always_inline
    def as_uninit[
        U: AnyType
    ](ref self) -> UnsafePointer[UnsafeMaybeUninit[U], origin_of(self)]:
        return (
            UnsafePointer(to=self.address)
            .bitcast[UnsafeMaybeUninit[U]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


# ===----------------------------------------------------------------------=== #
# unsafe_cast
# ===----------------------------------------------------------------------=== #


@always_inline
@doc_hidden
def unsafe_cast[
    from_mut: Bool,
    from_type: AnyType,
    from_origin: Origin[mut=from_mut],
    from_address_space: AddressSpace,
    mut: Bool = from_mut,
    //,
    *,
    Type: AnyType = from_type,
    origin: Origin[mut=mut] = from_origin,
    address_space: AddressSpace = from_address_space,
](
    pointer: Optional[
        UnsafePointer[from_type, from_origin, address_space=from_address_space]
    ],
    out result: Optional[
        UnsafePointer[Type, origin, address_space=address_space]
    ],
):
    result = UnsafePointer(to=pointer).bitcast[type_of(result)]()[]


@always_inline
@doc_hidden
def unsafe_cast[
    from_mut: Bool,
    from_type: AnyType,
    from_origin: Origin[mut=from_mut],
    from_address_space: AddressSpace,
    mut: Bool = from_mut,
    //,
    *,
    Type: AnyType = from_type,
    origin: Origin[mut=mut] = from_origin,
    address_space: AddressSpace = from_address_space,
](
    pointer: OptionalReg[
        UnsafePointer[from_type, from_origin, address_space=from_address_space]
    ],
    out result: OptionalReg[
        UnsafePointer[Type, origin, address_space=address_space]
    ],
):
    result = UnsafePointer(to=pointer).bitcast[type_of(result)]()[]


@always_inline("nodebug")
@doc_hidden
def pointer_to_int(pointer: OptionalUnsafePointer[...]) -> Int:
    return UnsafePointer(to=pointer).bitcast[Int]()[]


# ===----------------------------------------------------------------------=== #
# alloc
# ===----------------------------------------------------------------------=== #


@always_inline
def alloc[
    type: AnyType, /
](count: Int, *, alignment: Int = align_of[type]()) -> UnsafePointer[
    type, MutExternalOrigin
]:
    """Allocates contiguous storage for `count` elements of `type` with
    alignment `alignment`.

    Parameters:
        type: The type of the elements to allocate storage for.

    Args:
        count: Number of elements to allocate.
        alignment: The alignment of the allocation.

    Returns:
        A pointer to the newly allocated uninitialized array.

    Constraints:
        `count` must be positive and `size_of[type]()` must be > 0.

    Safety:

    - The returned memory is uninitialized; reading before writing is undefined.
    - The returned pointer has an empty mutable origin; you must call `free()`
      to release it.

    Example:

    ```mojo
    var ptr = alloc[Int32](4)
    ptr.store(0, Int32(42))
    ptr.store(1, Int32(7))
    ptr.store(2, Int32(9))
    var a = ptr.load(0)
    print(a[0], ptr.load(1)[0], ptr.load(2)[0])
    ptr.free()
    ```
    """
    comptime size_of_t = size_of[type]()
    comptime type_name = reflect[type].name()
    comptime assert size_of_t > 0, "size must be greater than zero"
    debug_assert(
        count >= 0,
        "alloc[",
        type_name,
        "]() count must be non-negative: ",
        count,
    )
    var pointer = _malloc[type](size_of_t * count, alignment=alignment)
    if unlikely(not pointer):
        abort("alloc failed: returned a null pointer")
    return pointer.unsafe_value()


# ===----------------------------------------------------------------------=== #
# UnsafePointer aliases
# ===----------------------------------------------------------------------=== #


comptime MutUnsafePointer[
    type: AnyType,
    origin: MutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointer[mut=True, type, origin, address_space=address_space]
"""A mutable unsafe pointer.

Parameters:
    type: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime ImmutUnsafePointer[
    type: AnyType,
    origin: ImmutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointer[type, origin, address_space=address_space]
"""An immutable unsafe pointer.

Parameters:
    type: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime OpaquePointer[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointer[NoneType, origin, address_space=address_space]
"""An opaque pointer, equivalent to the C `(const) void*` type.

Parameters:
    mut: Whether the pointer is mutable.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime MutOpaquePointer[
    origin: MutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = OpaquePointer[origin, address_space=address_space]
"""A mutable opaque pointer, equivalent to the C `void*` type.

Parameters:
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime ImmutOpaquePointer[
    origin: ImmutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = OpaquePointer[origin, address_space=address_space]
"""An immutable opaque pointer, equivalent to the C `const void*` type.

Parameters:
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime OptionalUnsafePointer[
    mut: Bool,
    //,
    type: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Optional[UnsafePointer[type, origin, address_space=address_space]]
"""An optional (nullable) `UnsafePointer`.

Parameters:
    mut: The mutability of the pointer.
    type: The type of the pointee.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""


struct UnsafePointer[
    mut: Bool,
    //,
    type: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    Comparable,
    DevicePassable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    UnsafeCustomNicheStorage,
    UnsafeSingleNicheable,
    Writable,
):
    """`UnsafePointer` represents an indirect reference to one or more values
    of type `T` consecutively in memory, and can refer to uninitialized memory.

    Because it supports referring to uninitialized memory, it provides unsafe
    methods for initializing and destroying instances of `T`, as well as methods
    for accessing the values once they are initialized. You should instead use
    safer pointers when possible.

    Important things to know:

    - This pointer is unsafe and non-nullable by design. To model a nullable pointer,
      use `Optional[UnsafePointer[...]]`, which shares the same layout (the null
      address is the `None` niche) so it remains zero-overhead.
    - It does not own existing memory. When memory is heap-allocated with
      `alloc()`, you must call `.free()`.
    - For simple read/write access, use `(ptr + i)[]` or `ptr[i]` where `i`
      is the offset size.
    - For SIMD operations on numeric data, use `UnsafePointer[Scalar[DType.xxx]]`
      with `load[dtype=DType.xxx]()` and `store[dtype=DType.xxx]()`.

    Key APIs:

    - `free()`: Frees memory previously allocated by `alloc()`. Do not call on
      pointers that were not allocated by `alloc()`.
    - `+ i` / `- i`: Pointer arithmetic. Returns a new pointer shifted by `i`
      elements. No bounds checking.
    - `[]` or `[i]`: Dereference to a reference of the pointee (or at
      offset `i`). Only valid if the memory at that location is initialized.
    - `load()`: Loads `width` elements starting at `offset` (default 0) as
      `SIMD[dtype, width]` from `UnsafePointer[Scalar[dtype]]`. Pass
      `alignment` when data is not naturally aligned.
    - `store()`: Stores `val: SIMD[dtype, width]` at `offset` into
      `UnsafePointer[Scalar[dtype]]`. Requires a mutable pointer.
    - `destroy_pointee()` / `take_pointee()`:
      Explicitly end the lifetime of the current pointee, or move it out, taking
      ownership.
    - `init_pointee_move()` / `init_pointee_move_from()` / `init_pointee_copy()`
      Initialize a pointee that is currently uninitialized, by moving an existing
      value, moving from another pointee, or by copying an existing value.
      Use these to manage lifecycles when working with uninitialized memory.

    For more information see [Unsafe
    pointers](/docs/manual/pointers/unsafe-pointers) in the Mojo Manual. For a
    comparison with other pointer types, see [Intro to
    pointers](/docs/manual/pointers/).

    Examples:

    Element-wise store and load (width = 1):

    ```mojo
    var ptr = alloc[Float32](4)
    for i in range(4):
        ptr.store(i, Float32(i))
    var v = ptr.load(2)
    print(v[0])  # => 2.0
    ptr.free()
    ```

    Vectorized store and load (width = 4):

    ```mojo
    var ptr = alloc[Int32](8)
    var vec = SIMD[DType.int32, 4](1, 2, 3, 4)
    ptr.store(0, vec)
    var out = ptr.load[width=4](0)
    print(out)  # => [1, 2, 3, 4]
    ptr.free()
    ```

    Pointer arithmetic and dereference:

    ```mojo
    var ptr = alloc[Int32](3)
    (ptr + 0)[] = 10  # offset by 0 elements, then dereference to write
    (ptr + 1)[] = 20  # offset +1 element, then dereference to write
    ptr[2] = 30  # equivalent offset/dereference with brackets (via __getitem__)
    var second = ptr[1]  # reads the element at index 1
    print(second, ptr[2])  # => 20 30
    ptr.free()
    ```

    Point to a value on the stack:

    ```mojo
    var foo: Int = 123
    var ptr = UnsafePointer(to=foo)
    print(ptr[])  # => 123
    # Don't call `free()` because the value was not heap-allocated
    # Mojo will destroy it when the `foo` lifetime ends
    ```

    Model a nullable pointer with `Optional`:

    `UnsafePointer` is non-nullable by design, so nullability must be modeled
    explicitly with `Optional[UnsafePointer[T, origin]]`. This keeps the same
    layout as `Optional` stores the null address as its `None` niche, so there
    is no overhead compared to a raw pointer.

    ```mojo
    from std.random import random_float64

    # A field that may or may not point to a heap-allocated Int.
    var maybe_ptr: Optional[UnsafePointer[Int, MutExternalOrigin]] = None

    # Maybe populate it later.
    if random_float64() > 0.5:
        maybe_ptr = alloc[Int](1)

    # Check for absence, then unwrap to use the pointer.
    if maybe_ptr:
        var ptr = maybe_ptr.value()
        ptr.init_pointee_move(42)
        print(ptr[])  # => 42
        ptr.free()
    ```

    If you instead need a non-null placeholder for a field that will be
    populated on demand (for example, a buffer that is allocated lazily),
    use `UnsafePointer.unsafe_dangling()`. Note that `unsafe_dangling()` is
    not a null sentinel — it returns an aligned but dangling address, so
    types that lazily allocate must track initialization separately.

    Parameters:
        mut: Whether the origin is mutable.
        type: The type the pointer points to.
        origin: The origin of the memory being addressed.
        address_space: The address space associated with the UnsafePointer allocated memory.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    """The underlying pointer type."""

    comptime _with_origin[
        with_mut: Bool, //, with_origin: Origin[mut=with_mut]
    ] = UnsafePointer[
        mut=with_mut,
        Self.type,
        with_origin,
        address_space=Self.address_space,
    ]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var address: Self._mlir_type
    """The underlying pointer."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: Self._mlir_type):
        """Create a pointer from a low-level pointer primitive.

        Args:
            value: The MLIR value of the pointer to construct with.
        """
        self.address = value

    @always_inline
    def __init__(out self, *, unsafe_from_address: Int):
        """Create a pointer from a raw address.

        Args:
            unsafe_from_address: The raw address to create a pointer from.

        Safety:
            Creating a pointer from a raw address is inherently unsafe as the
            caller must ensure the address is valid before writing to it, and
            that the memory is initialized before reading from it. The caller
            must also ensure the pointer's origin and mutability is valid for
            the address, failure to do may result in undefined behavior.
        """
        comptime assert (
            size_of[type_of(self)]() == size_of[Int]()
        ), "Pointer/Int size mismatch"
        self = UnsafePointer(to=unsafe_from_address).bitcast[type_of(self)]()[]

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value._mlir_value] to: Self.type,
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @always_inline("builtin")
    @implicit
    def __init__[
        disambig2: Int = 0
    ](
        other: UnsafePointer,
        out self: UnsafePointer[
            other.type,
            ImmutOrigin(other.origin),
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a mutable pointer to immutable.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig2: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline("builtin")
    @implicit
    def __init__[
        disambig: Int = 0  # FIXME: Work around name mangling conflict.
    ](
        other: UnsafePointer[mut=True, ...],
        out self: UnsafePointer[
            other.type,
            MutAnyOrigin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a mutable pointer to `MutAnyOrigin`.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline("builtin")
    @implicit
    def __init__(
        other: UnsafePointer[...],
        out self: UnsafePointer[
            other.type,
            ImmutAnyOrigin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a pointer to `ImmutAnyOrigin`.

        Args:
            other: The pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    def __init__[
        T: ImplicitlyDestructible, //
    ](
        out self: UnsafePointer[T, Self.origin],
        *,
        ref[Self.origin] unchecked_downcast_value: PythonObject,
    ):
        """Downcast a `PythonObject` known to contain a Mojo object to a pointer.

        This operation is only valid if the provided Python object contains
        an initialized Mojo object of matching type.

        Parameters:
            T: Pointee type that can be destroyed implicitly (without
              deinitializer arguments).

        Args:
            unchecked_downcast_value: The Python object to downcast from.
        """
        self = unchecked_downcast_value.unchecked_downcast_value_ptr[T]()

    # ===------------------------------------------------------------------===#
    # UnsafeNicheable
    # ===------------------------------------------------------------------===#

    @doc_hidden
    comptime NicheStorage: NicheStorageTraits = _UnsafePointerNicheStorage[
        Self.type, Self.address_space
    ]

    @staticmethod
    @always_inline
    @doc_hidden
    def write_niche(
        memory: UnsafePointer[mut=True, UnsafeMaybeUninit[Self], _]
    ):
        memory.bitcast[
            _Null[Self.type, Self.address_space]
        ]().init_pointee_move({})

    @staticmethod
    @always_inline
    @doc_hidden
    def isa_niche(
        memory: UnsafePointer[mut=False, UnsafeMaybeUninit[Self], _]
    ) -> Bool:
        comptime NullType = _Null[Self.type, Self.address_space]
        comptime null_address = Int(NullType())
        return Int(memory.bitcast[NullType]()[]) == null_address

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __getitem__(self) -> ref[Self.origin, Self.address_space] Self.type:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """

        # We're unsafe, so we can have unsafe things.
        comptime _ref_type = Pointer[Self.type, Self.origin, Self.address_space]
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type=_ref_type._mlir_type](
                self.address
            )
        )

    @always_inline("nodebug")
    def __getitem__[
        I: Indexer, //
    ](self, offset: I) -> ref[Self.origin, Self.address_space] Self.type:
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
    def __add__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return __mlir_op.`pop.offset`(
            self.address, index(offset)._int_mlir_index()
        )

    @always_inline
    def __sub__[I: Indexer, //](self, offset: I) -> Self:
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
    def __iadd__[I: Indexer, //](mut self, offset: I):
        """Add an offset to this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self + offset

    @always_inline
    def __isub__[I: Indexer, //](mut self, offset: I):
        """Subtract an offset from this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self - offset

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __eq__(
        self,
        rhs: UnsafePointer[Self.type, address_space=Self.address_space, ...],
    ) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ne__(
        self,
        rhs: UnsafePointer[Self.type, address_space=Self.address_space, ...],
    ) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __lt__(
        self,
        rhs: UnsafePointer[Self.type, address_space=Self.address_space, ...],
    ) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __le__(
        self,
        rhs: UnsafePointer[Self.type, address_space=Self.address_space, ...],
    ) -> Bool:
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
    def __le__(self, rhs: Self) -> Bool:
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
    def __gt__(
        self,
        rhs: UnsafePointer[Self.type, address_space=Self.address_space, ...],
    ) -> Bool:
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
    def __gt__(self, rhs: Self) -> Bool:
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
    def __ge__(
        self,
        rhs: UnsafePointer[Self.type, address_space=Self.address_space, ...],
    ) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) >= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ge__(self, rhs: Self) -> Bool:
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
    def __merge_with__[
        other_type: type_of(
            UnsafePointer[
                Self.type,
                origin=_,
                address_space=Self.address_space,
            ]
        ),
    ](self) -> UnsafePointer[
        type=Self.type,
        origin=origin_of(Self.origin, other_type.origin),
        address_space=Self.address_space,
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
    @deprecated(
        "UnsafePointer is non-null by design, so Bool(ptr) is no longer"
        " meaningful. To model a null pointer, use"
        " `Optional[UnsafePointer[...]]` and check with `Bool(opt_ptr)` / `!="
        " None`."
    )
    def __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Deprecated: `UnsafePointer` is non-null by design, so `Bool(ptr)` is
        no longer meaningful. To model a nullable pointer, use
        `Optional[UnsafePointer[...]]` and check with `Bool(opt_ptr)` or
        `!= None`.

        Returns:
            True if the pointer address is non-zero, False otherwise.
        """
        return Int(self) != 0

    @always_inline
    def __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return Int(mlir_value=__mlir_op.`pop.pointer_to_index`(self.address))

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Formats this pointer address to the provided Writer.

        Args:
            writer: The object to write to.
        """
        _write_int[radix=16](writer, Scalar[DType.int](Int(self)), prefix="0x")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the UnsafePointer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "UnsafePointer").params(
            Named("mut", Self.mut),
            TypeNames[Self.type](),
            Named("address_space", Self.address_space),
        ).fields(self)

    # ===-------------------------------------------------------------------===#
    # DevicePassable
    # ===-------------------------------------------------------------------===#

    # Implementation of `DevicePassable`
    comptime device_type: AnyType = Self
    """DeviceBuffer dtypes are remapped to UnsafePointer when passed to accelerator devices."""

    @staticmethod
    def _is_convertible_to_device_type[T: AnyType]() -> Bool:
        comptime if Self.mut:
            return TypeList.of[
                Trait=AnyType,
                Self,
                Self._OriginCastType[MutAnyOrigin],
                Self._OriginCastType[MutExternalOrigin],
                Self._OriginCastType[ImmutAnyOrigin],
                Self._OriginCastType[ImmutExternalOrigin],
                Self._UnsafePointerType,
                Self._UnsafePointerType._OriginCastType[MutAnyOrigin],
                Self._UnsafePointerType._OriginCastType[MutExternalOrigin],
                Self._UnsafePointerType._OriginCastType[ImmutAnyOrigin],
                Self._UnsafePointerType._OriginCastType[ImmutExternalOrigin],
            ]().contains[T]()
        else:
            return TypeList.of[
                Trait=AnyType,
                Self,
                Self._OriginCastType[ImmutAnyOrigin],
                Self._OriginCastType[ImmutExternalOrigin],
                Self._UnsafePointerType,
                Self._UnsafePointerType._OriginCastType[ImmutAnyOrigin],
                Self._UnsafePointerType._OriginCastType[ImmutExternalOrigin],
            ]().contains[T]()

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Device dtype mapping from DeviceBuffer to the device's UnsafePointer.
        """
        # TODO: Allow the low-level DeviceContext implementation to intercept
        # these translations.
        target.bitcast[Self.device_type]()[] = self.address

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this type name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This name of the type.
        """
        return String(
            "UnsafePointer[",
            reflect[Self.type].name(),
            ", mut=",
            Self.mut,
            ", address_space=",
            Self.address_space,
            "]",
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    @staticmethod
    def unsafe_dangling() -> Self:
        """Creates a new `UnsafePointer` that is dangling, but well-aligned.

        This is useful for initializing types which lazily allocate.

        Note that the address of the returned pointer may potentially be that
        of a valid pointer, which means this must not be used as a "not yet
        initialized" sentinel value. Types that lazily allocate must track
        initialization by some other means.

        Returns:
            A dangling but well-aligned `UnsafePointer`.

        Example:

        ```mojo
        var ptr = UnsafePointer[Int, MutExternalOrigin].unsafe_dangling()
        # Important: don't try to access the value of `ptr` without
        # initializing it first! The pointer is not null but isn't valid either!
        ```
        """
        comptime alignment = align_of[Self.type]()
        return Self(unsafe_from_address=alignment)

    @always_inline("nodebug")
    def swap_pointees[
        U: Movable
    ](
        self: UnsafePointer[mut=True, U, _],
        other: UnsafePointer[mut=True, U, _],
    ):
        """Swap the values at the pointers.

        This function assumes that `self` and `other` _may_ overlap in memory.
        If that is not the case, or when references are available, you should
        use `builtin.swap` instead.

        Parameters:
            U: The type the pointers point to, which must be `Movable`.

        Args:
            other: The other pointer to swap with.

        Safety:
            - `self` and `other` must both point to valid, initialized instances
              of `T`.
        """

        comptime if U.__move_ctor_is_trivial:
            # If `moveinit` is trivial, we can avoid the branch introduced from
            # checking if the pointers are equal by using temporary stack
            # values.
            #
            # Since `lhs` may overlap with `rhs` we need two temporary stack
            # values since we cannot call `memcpy` with the potentially
            # overlapping pointers.
            #
            # Even if they are not overlapping, this also produces better llvm
            # code with only 2 loads and 2 stores. Whereas with only 1 temporary
            # and a memcpy between the pointers it produces 3 load and 3 stores.

            var self_tmp = UnsafeMaybeUninit[U]()
            var other_tmp = UnsafeMaybeUninit[U]()
            memcpy(dest=self_tmp.unsafe_ptr(), src=self, count=1)
            memcpy(dest=other_tmp.unsafe_ptr(), src=other, count=1)

            memcpy(dest=self, src=other_tmp.unsafe_ptr(), count=1)
            memcpy(dest=other, src=self_tmp.unsafe_ptr(), count=1)
        else:
            # If `moveinit` is NOT trivial, we need to check if the pointers are
            # the same to avoid undefined behavior when moving from rhs to lhs.
            if self == other:
                return
            var tmp = self.take_pointee()
            self.init_pointee_move_from(other)
            other.init_pointee_move(tmp^)

    @always_inline("nodebug")
    def as_noalias_ptr(self) -> Self._UnsafePointerType:
        """Cast the pointer to a new pointer that is known not to locally alias
        any other pointer. In other words, the pointer transitively does not
        comptime any other memory value declared in the local function context.

        This information is relayed to the optimizer. If the pointer does
        locally alias another memory value, the behaviour is undefined.

        Returns:
            A noalias pointer.
        """
        return __mlir_op.`pop.noalias_pointer_cast`(self.address)

    @always_inline("nodebug")
    def load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: UnsafePointer[Scalar[dtype], ...]) -> SIMD[dtype, width]:
        """Loads `width` elements from the value the pointer points to.

        Use `alignment` to specify minimal known alignment in bytes; pass a
        smaller value (such as 1) if loading from packed/unaligned memory. The
        `volatile`/`invariant` flags control reordering and common-subexpression
        elimination semantics for special cases.

        Example:

        ```mojo
        var p = alloc[Int32](8)
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
            non_temporal: Whether the load has no temporal locality (streaming).

        Returns:
            The loaded SIMD vector.
        """
        _simd_construction_checks[dtype, width]()
        comptime assert (
            alignment > 0
        ), "alignment must be a positive integer value"
        comptime assert (
            not volatile or volatile ^ invariant
        ), "both volatile and invariant cannot be set at the same time"

        comptime if is_nvidia_gpu() and size_of[
            dtype
        ]() == 1 and alignment == 1:
            # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
            # regardless of the alignment that is passed. This causes issues if
            # this method is called on an unaligned pointer.
            # TODO #37823 We can make this smarter when we add an `aligned`
            # trait to the pointer class.
            var v = SIMD[dtype, width]()

            # intentionally don't unroll, otherwise the compiler vectorizes
            for i in range(width):
                v[i] = __mlir_op.`pop.load`[
                    alignment=alignment._int_mlir_index(),
                    isVolatile=volatile._mlir_value,
                    isInvariant=invariant._mlir_value,
                    isNonTemporal=non_temporal._mlir_value,
                ]((self + i).address)
            comptime if dtype.is_floating_point():
                _check_not_poison[dtype, width](v)
            return v
        elif dtype == DType.bool and width > 1:
            # Bool (i1) is sub-byte, so a vector load of SIMD[bool, N]
            # packs bits. Load as uint8 and convert to bool so each
            # element occupies its own byte boundary.
            return rebind[SIMD[dtype, width]](
                self.bitcast[Scalar[DType.uint8]]()
                .load[
                    width=width,
                    alignment=alignment,
                    volatile=volatile,
                    invariant=invariant,
                    non_temporal=non_temporal,
                ]()
                .cast[DType.bool]()
            )

        var address = self.bitcast[SIMD[dtype, width]]().address

        var result = __mlir_op.`pop.load`[
            alignment=alignment._int_mlir_index(),
            isVolatile=volatile._mlir_value,
            isInvariant=invariant._mlir_value,
            isNonTemporal=non_temporal._mlir_value,
        ](address)
        comptime if dtype.is_floating_point():
            _check_not_poison[dtype, width](result)
        return result

    @always_inline("nodebug")
    def load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](
        self: UnsafePointer[Scalar[dtype], ...],
        offset: Scalar,
    ) -> SIMD[
        dtype, width
    ]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be an integer.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.
            non_temporal: Whether the load has no temporal locality (streaming).

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        comptime assert offset.dtype.is_integral(), "offset must be an integer"
        return (self + Int(offset)).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @always_inline("nodebug")
    def load[
        I: Indexer,
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](
        self: UnsafePointer[Scalar[dtype], ...],
        offset: I,
    ) -> SIMD[
        dtype, width
    ]:
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
            non_temporal: Whether the load has no temporal locality (streaming).

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return (self + offset).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @always_inline("nodebug")
    def store[
        I: Indexer,
        dtype: DType,
        //,
        width: SIMDSize = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafePointer[mut=True, Scalar[dtype], ...],
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
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        (self + offset).store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @always_inline("nodebug")
    def store[
        dtype: DType,
        offset_type: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafePointer[mut=True, Scalar[dtype], ...],
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
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        comptime assert offset_type.is_integral(), "offset must be integer"
        (self + Int(offset))._store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @always_inline("nodebug")
    def store[
        dtype: DType,
        //,
        width: SIMDSize = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
    ):
        """Stores a single element value `val` at element offset 0.

        Specify `alignment` when writing to packed/unaligned memory. Requires a
        mutable pointer. For writing at an element offset, use the overloads
        that accept an index or scalar offset.

        Example:

        ```mojo
        var p = alloc[Float32](4)
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
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            val: The SIMD value to store.
        """
        self._store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @always_inline("nodebug")
    def _store[
        dtype: DType,
        width: SIMDSize,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
    ):
        comptime assert width > 0, "width must be a positive integer value"
        comptime assert (
            alignment > 0
        ), "alignment must be a positive integer value"

        comptime if dtype == DType.bool and width > 1:
            # Bool (i1) is sub-byte, so a vector store of SIMD[bool, N]
            # packs bits. Cast to uint8 and store so each element
            # occupies its own byte boundary.
            self.bitcast[Scalar[DType.uint8]]()._store[
                alignment=alignment,
                volatile=volatile,
                non_temporal=non_temporal,
            ](val.cast[DType.uint8]())
        else:
            __mlir_op.`pop.store`[
                alignment=alignment._int_mlir_index(),
                isVolatile=volatile._mlir_value,
                isNonTemporal=non_temporal._mlir_value,
            ](val, self.bitcast[SIMD[dtype, width]]().address)

    @always_inline("nodebug")
    def strided_load[
        dtype: DType, T: Intable, //, width: Int
    ](
        self: UnsafePointer[Scalar[dtype], ...],
        stride: T,
    ) -> SIMD[
        dtype, width
    ]:
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
    def strided_store[
        dtype: DType,
        T: Intable,
        //,
        width: SIMDSize = 1,
    ](
        self: UnsafePointer[mut=True, Scalar[dtype], ...],
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
        strided_store(
            val, self, Int(stride), SIMD[DType.bool, width](fill=True)
        )

    @always_inline("nodebug")
    def gather[
        dtype: DType,
        //,
        *,
        width: SIMDSize = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafePointer[Scalar[dtype], ...],
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
        comptime assert (
            offset.dtype.is_integral()
        ), "offset type must be an integral type"
        comptime assert (
            alignment.is_power_of_two()
        ), "alignment must be a power of two integer value"

        comptime if is_apple_gpu():
            # `Int(self)` would erase the address space; on Apple AIR the
            # resulting GENERIC load silently reads zero (MOCO-3762).
            var result = default
            comptime for i in range(width):
                if mask[i]:
                    result[i] = self.load[alignment=alignment](Int(offset[i]))
            return result

        var base = offset.cast[DType.int]().fma(
            SIMD[DType.int, width](size_of[dtype]()),
            SIMD[DType.int, width](Int(self)),
        )
        return gather[alignment=alignment](base, mask, default)

    @always_inline("nodebug")
    def scatter[
        dtype: DType,
        //,
        *,
        width: SIMDSize = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafePointer[mut=True, Scalar[dtype], ...],
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
        comptime assert (
            offset.dtype.is_integral()
        ), "offset type must be an integral type"
        comptime assert (
            alignment.is_power_of_two()
        ), "alignment must be a power of two integer value"

        comptime if is_apple_gpu():
            # See `gather` for the address-space rationale (MOCO-3762).
            comptime for i in range(width):
                if mask[i]:
                    self.store[alignment=alignment](Int(offset[i]), val[i])
            return

        var base = offset.cast[DType.int]().fma(
            SIMD[DType.int, width](size_of[dtype]()),
            SIMD[DType.int, width](Int(self)),
        )
        scatter[alignment=alignment](val, base, mask)

    @always_inline
    def free(self: UnsafePointer[mut=True, Self.type, ...]):
        """Free the memory referenced by the pointer."""
        _free(self)

    @always_inline("builtin")
    def bitcast[
        T: AnyType
    ](self) -> UnsafePointer[T, Self.origin, address_space=Self.address_space,]:
        """Bitcasts an UnsafePointer to a different type.

        Parameters:
            T: The target type.

        Returns:
            A new UnsafePointer object with the specified type and the same address,
            as the original UnsafePointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type=UnsafePointer[
                T,
                Self.origin,
                address_space=Self.address_space,
            ]._mlir_type,
        ](self.address)

    comptime _UnsafePointerType = UnsafePointer[
        Self.type, Self.origin, address_space=Self.address_space
    ]

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = UnsafePointer[
        Self.type,
        target_origin,
        address_space=Self.address_space,
    ]

    @always_inline("nodebug")
    def mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Changes the mutability of a pointer.

        This is a safe way to change the mutability of a pointer with an
        unbounded mutability. This function will emit a compile time error if
        you try to cast an immutable pointer to mutable.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.
        """
        comptime assert (
            target_mut == False or target_mut == Self.mut
        ), "Cannot safely cast an immutable pointer to mutable"
        return self.unsafe_mut_cast[target_mut]()

    @always_inline("builtin")
    def unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Changes the mutability of a pointer.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.

        If you are unconditionally casting the mutability to `False`, use
        `as_immutable` instead.
        If you are casting to mutable or a parameterized mutability, prefer
        using the safe `mut_cast` method instead.

        Safety:
            Casting the mutability of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior. Consider restricting
            types to their proper mutability at the function signature level.
            For example, taking an `UnsafePointer[T, mut=True, ...]` as an
            argument over an unbound `UnsafePointer[T, ...]` is preferred.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type=Self._OriginCastType[
                Self.origin.unsafe_mut_cast[target_mut]()
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        """Changes the origin of a pointer.

        Parameters:
            target_origin: Origin of the destination pointer.

        Returns:
            A pointer with the same type, mutability and address space as the
            original pointer, but with the newly specified origin.

        If you are unconditionally casting the origin to an `AnyOrigin`, use
        `as_any_origin` instead.

        Safety:
            Casting the origin of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior or unexpected variable
            destruction. Considering parameterizing the origin at the function
            level to avoid unnecessary casts.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type=Self._OriginCastType[target_origin]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    def as_immutable(
        self,
    ) -> Self._OriginCastType[ImmutOrigin(Self.origin)]:
        """Changes the mutability of a pointer to immutable.

        Unlike `unsafe_mut_cast`, this function is always safe to use as casting
        from (im)mutable to immutable is always safe.

        Returns:
            A pointer with the mutability set to immutable.
        """
        return self.unsafe_mut_cast[False]()

    @always_inline("builtin")
    def as_any_origin(
        self,
    ) -> UnsafePointer[
        Self.type,
        AnyOrigin[mut=Self.mut],
        address_space=Self.address_space,
    ]:
        """Casts the origin of a pointer to `AnyOrigin`.

        Returns:
            A pointer with the origin set to `AnyOrigin`.

        It is usually preferred to maintain concrete origin values instead of
        using `AnyOrigin`. However, if it is needed, keep in mind that
        `AnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the pointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type=UnsafePointer[
                Self.type,
                AnyOrigin[mut=Self.mut],
                address_space=Self.address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    def address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> UnsafePointer[
        Self.type,
        Self.origin,
        address_space=target_address_space,
    ]:
        """Casts an UnsafePointer to a different address space.

        Parameters:
            target_address_space: The address space of the result.

        Returns:
            A new UnsafePointer object with the same type and the same address,
            as the original UnsafePointer and the new address space.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type=UnsafePointer[
                Self.type,
                Self.origin,
                address_space=target_address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline
    def destroy_pointee[
        T: ImplicitlyDestructible, //
    ](self: UnsafePointer[T, _]) where type_of(self).mut:
        """Destroy the pointed-to value.

        The pointer must point to a valid, initialized instance of `type`.
        This is equivalent to `_ = self.take_pointee()` but doesn't require
        `Movable` and is more efficient because it doesn't invoke a move
        constructor.

        Parameters:
            T: Pointee type that can be destroyed implicitly (without
              deinitializer arguments).

        """
        _ = __get_address_as_owned_value(self.address)

    # TODO(MOCO-2367): Use a `unified` closure parameter here instead.
    @always_inline
    def destroy_pointee_with(
        self: UnsafePointer[
            Self.type,
            _,
            address_space=AddressSpace.GENERIC,
        ],
        destroy_func: def(var Self.type) thin,
    ) where type_of(self).mut:
        """Destroy the pointed-to value using a user-provided destructor function.

        This can be used to destroy non-`ImplicitlyDestructible` values in-place
        without moving.

        Args:
            destroy_func: A function that takes ownership of the pointee value
                for the purpose of deinitializing it.
        """
        destroy_func(__get_address_as_owned_value(self.address))

    @always_inline
    def take_pointee[
        T: Movable,
        //,
    ](self: UnsafePointer[T, _]) -> T where type_of(self).mut:
        """Move the value at the pointer out, leaving it uninitialized.

        The pointer must point to a valid, initialized instance of `T`.

        This performs a _consuming_ move, ending the origin of the value stored
        in this pointer memory location. Subsequent reads of this pointer are
        not valid. If a new valid value is stored using `init_pointee_move()`, then
        reading from this pointer becomes valid again.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Returns:
            The value at the pointer.
        """
        return __get_address_as_owned_value(self.address)

    @always_inline
    def init_pointee_move[
        T: Movable,
        //,
    ](self: UnsafePointer[T, _], var value: T) where type_of(self).mut:
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
        __get_address_as_uninit_lvalue(self.address) = value^

    @always_inline
    def init_pointee_copy[
        T: Copyable,
        //,
    ](self: UnsafePointer[T, _], value: T) where type_of(self).mut:
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
        __get_address_as_uninit_lvalue(self.address) = value.copy()

    @always_inline
    def init_pointee_move_from[
        T: Movable,
        //,
    ](self: UnsafePointer[T, _], src: UnsafePointer[T, _]) where (
        type_of(self).mut
    ) and (type_of(src).mut):
        """Moves the value `src` points to into the memory location pointed to
        by `self`.

        The `self` pointer memory location is assumed to contain uninitialized
        data prior to this assignment, and consequently the current contents of
        this pointer are not destructed before writing the value from the `src`
        pointer.

        Ownership of the value is logically transferred from `src` into `self`'s
        pointer location.

        After this call, the `src` pointee value should be treated as
        uninitialized data. Subsequent reads of or destructor calls on the `src`
        pointee value are invalid, unless and until a new valid value has been
        moved into the `src` pointer's memory location using an
        `init_pointee_*()` operation.

        This transfers the value out of `src` and into `self` using at most one
        move constructor call.

        Example:

        ```mojo
        var a_ptr = alloc[String](1)
        var b_ptr = alloc[String](2)

        # Initialize A pointee
        a_ptr.init_pointee_move("foo")

        # Perform the move
        b_ptr.init_pointee_move_from(a_ptr)

        # Clean up
        b_ptr.destroy_pointee()
        a_ptr.free()
        b_ptr.free()
        ```

        Safety:

        * `src` must point to a valid, initialized instance of `T`.
        * `self` must point to writable memory for `T`. The pointee contents
          of `self` should be uninitialized; if `self` was previously written
          with a valid value, that value will be overwritten and its
          destructor will NOT be run.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            src: Source pointer that the value will be moved from.
        """
        __get_address_as_uninit_lvalue(
            self.address
        ) = __get_address_as_owned_value(src.address)
