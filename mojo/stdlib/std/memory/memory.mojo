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
"""Defines functions for memory manipulations.

You can import these APIs from the `memory` package. For example:

```mojo
from std.memory import memcmp
```
"""


from std.math import iota
from std.memory.unsafe_pointer import unsafe_cast
from std.sys import _libc as libc
from std.ffi import external_call
from std.sys import (
    align_of,
    codegen_unreachable,
    get_defined_string,
    is_gpu,
    llvm_intrinsic,
    simd_bit_width,
    simd_width_of,
    size_of,
)

from std.algorithm import vectorize

# ===-----------------------------------------------------------------------===#
# memcmp
# ===-----------------------------------------------------------------------===#


@always_inline
def _memcmp_impl_unconstrained[
    dtype: DType, //
](
    s1: Pointer[mut=False, Scalar[dtype], ...],
    s2: Pointer[mut=False, Scalar[dtype], ...],
    count: Int,
) -> Int:
    for i in range(count):
        var s1i = s1[unsafe_offset=i]
        var s2i = s2[unsafe_offset=i]
        if s1i != s2i:
            return 1 if s1i > s2i else -1
    return 0


@always_inline
def _memcmp_opt_impl_unconstrained[
    dtype: DType, //
](
    s1: Pointer[mut=False, Scalar[dtype], ...],
    s2: Pointer[mut=False, Scalar[dtype], ...],
    count: Int,
) -> Int:
    comptime simd_width = simd_width_of[dtype]()
    if count < simd_width:
        for i in range(count):
            var s1i = s1[unsafe_offset=i]
            var s2i = s2[unsafe_offset=i]
            if s1i != s2i:
                return 1 if s1i > s2i else -1
        return 0

    var last = count - simd_width

    for i in range(0, last, simd_width):
        var s1i = s1.unsafe_load[width=simd_width](i)
        var s2i = s2.unsafe_load[width=simd_width](i)
        var diff = s1i.ne(s2i)
        if any(diff):
            var index = Int(
                diff.select(
                    iota[DType.uint8, simd_width](),
                    SIMD[DType.uint8, simd_width](255),
                ).reduce_min()
            )
            return -1 if s1i[index] < s2i[index] else 1

    var s1i = s1.unsafe_load[width=simd_width](last)
    var s2i = s2.unsafe_load[width=simd_width](last)
    var diff = s1i.ne(s2i)
    if any(diff):
        var index = Int(
            diff.select(
                iota[DType.uint8, simd_width](),
                SIMD[DType.uint8, simd_width](255),
            ).reduce_min()
        )
        return -1 if s1i[index] < s2i[index] else 1
    return 0


@always_inline
def _memcmp_impl[
    dtype: DType
](
    s1: Pointer[mut=False, Scalar[dtype], ...],
    s2: Pointer[mut=False, Scalar[dtype], ...],
    count: Int,
) -> Int where dtype.is_integral():
    if __is_run_in_comptime_interpreter:
        return _memcmp_impl_unconstrained(s1, s2, count)
    else:
        return _memcmp_opt_impl_unconstrained(s1, s2, count)


@always_inline
def unsafe_memcmp[
    type: AnyType, address_space: AddressSpace
](
    s1: Pointer[mut=False, type, _, address_space=address_space],
    s2: Pointer[mut=False, type, _, address_space=address_space],
    count: Int,
) -> Int:
    """Compares two buffers. Both strings are assumed to be of the same length.

    Parameters:
        type: The element type.
        address_space: The address space of the pointer.

    Args:
        s1: The first buffer address.
        s2: The second buffer address.
        count: The number of elements in the buffers.

    Returns:
        Returns 0 if the bytes strings are identical, 1 if s1 > s2, and -1 if
        s1 < s2. The comparison is performed by the first different byte in the
        byte strings.
    """
    var byte_count = count * size_of[type]()

    comptime if size_of[type]() % size_of[DType.int32]() == 0:
        return _memcmp_impl(
            s1.unsafe_bitcast[Int32](),
            s2.unsafe_bitcast[Int32](),
            byte_count // size_of[DType.int32](),
        )

    return _memcmp_impl(
        s1.unsafe_bitcast[Byte](), s2.unsafe_bitcast[Byte](), byte_count
    )


@always_inline
@deprecated(use=unsafe_memcmp)
def memcmp[
    type: AnyType, address_space: AddressSpace
](
    s1: Pointer[mut=False, type, _, address_space=address_space],
    s2: Pointer[mut=False, type, _, address_space=address_space],
    count: Int,
) -> Int:
    """Compares two buffers. Both strings are assumed to be of the same length.

    Parameters:
        type: The element type.
        address_space: The address space of the pointer.

    Args:
        s1: The first buffer address.
        s2: The second buffer address.
        count: The number of elements in the buffers.

    Returns:
        Returns 0 if the bytes strings are identical, 1 if s1 > s2, and -1 if
        s1 < s2. The comparison is performed by the first different byte in the
        byte strings.
    """
    return unsafe_memcmp(s1, s2, count)


# ===-----------------------------------------------------------------------===#
# memcpy
# ===-----------------------------------------------------------------------===#


@always_inline
def _memcpy_impl(
    dest_data: Pointer[mut=True, Byte, ...],
    src_data: Pointer[mut=False, Byte, ...],
    n: Int,
):
    """Copies a memory area.

    Args:
        dest_data: The destination pointer.
        src_data: The source pointer.
        n: The number of bytes to copy.
    """

    def copy[width: Int](offset: Int) {imm}:
        dest_data.unsafe_store(
            offset, src_data.unsafe_load[width=width](offset)
        )

    comptime if is_gpu():
        vectorize[simd_bit_width()](n, copy)

        return

    if n < 5:
        if n == 0:
            return
        dest_data[unsafe_offset=0] = src_data[unsafe_offset=0]
        dest_data[unsafe_offset=n - 1] = src_data[unsafe_offset=n - 1]
        if n <= 2:
            return
        dest_data[unsafe_offset=1] = src_data[unsafe_offset=1]
        dest_data[unsafe_offset=n - 2] = src_data[unsafe_offset=n - 2]
        return

    if n <= 16:
        if n >= 8:
            var ui64_size = size_of[UInt64]()
            dest_data.unsafe_bitcast[UInt64]().unsafe_store[alignment=1](
                0, src_data.unsafe_bitcast[UInt64]().unsafe_load[alignment=1](0)
            )
            dest_data.unsafe_offset(n - ui64_size).unsafe_bitcast[
                UInt64
            ]().unsafe_store[alignment=1](
                0,
                src_data.unsafe_offset(n - ui64_size)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1](0),
            )
            return

        var ui32_size = size_of[UInt32]()
        dest_data.unsafe_bitcast[UInt32]().unsafe_store[alignment=1](
            0, src_data.unsafe_bitcast[UInt32]().unsafe_load[alignment=1](0)
        )
        dest_data.unsafe_offset(n - ui32_size).unsafe_bitcast[
            UInt32
        ]().unsafe_store[alignment=1](
            0,
            src_data.unsafe_offset(n - ui32_size)
            .unsafe_bitcast[UInt32]()
            .unsafe_load[alignment=1](0),
        )
        return

    # TODO (#10566): This branch appears to cause a 12% regression in BERT by
    # slowing down broadcast ops
    # if n <= 32:
    #    alias simd_16xui8_size = 16 * size_of[Int8]()
    #    dest_data.store[width=16](src_data.load[width=16]())
    #    # note that some of these bytes may have already been written by the
    #    # previous simd_store
    #    dest_data.store[width=16](
    #        n - simd_16xui8_size, src_data.load[width=16](n - simd_16xui8_size)
    #    )
    #    return

    # Copy in 32-byte chunks.
    vectorize[32](n, copy)


@always_inline
def unsafe_memcpy[
    T: AnyType
](*, dest: Pointer[mut=True, T, _], src: Pointer[T, _], count: Int,):
    """Copy `count * size_of[T]()` bytes from src to dest.

    The dest and src memory must **not** overlap. For potentially
    overlapping memory regions, use `unsafe_memmove`.

    Parameters:
        T: The element type.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.

    Safety:
        `dest` and `src` must be valid for at least `count * size_of[T]()`
        bytes.
    """
    if count == 0:
        return

    var n = count * size_of[T]()

    var dest_bytes = dest.unsafe_bitcast[Byte]()
    var src_bytes = src.unsafe_bitcast[Byte]()

    if __is_run_in_comptime_interpreter:
        llvm_intrinsic["llvm.memcpy", NoneType](
            dest_bytes, src_bytes, n.__mlir_index__()
        )
    else:
        _memcpy_impl(dest_bytes, src_bytes, n)


@always_inline
@deprecated(use=unsafe_memcpy)
def memcpy[
    T: AnyType
](
    *,
    dest: OptionalUnsafePointer[mut=True, T, _],
    src: OptionalUnsafePointer[T, _],
    count: Int,
):
    """Copy `count * size_of[T]()` bytes from src to dest.

    The dest and src memory must **not** overlap. For potentially
    overlapping memory regions, use `unsafe_memmove`.

    Parameters:
        T: The element type.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.

    Safety:
        `dest` and `src` must be valid for at least `count * size_of[T]()`
        bytes. `dest` or `src` can only be `None` when `count == 0`.
    """
    if count == 0:
        return

    unsafe_memcpy(dest=dest.unsafe_value(), src=src.unsafe_value(), count=count)


# ===-----------------------------------------------------------------------===#
# memmove
# ===-----------------------------------------------------------------------===#


@always_inline
def unsafe_memmove[
    T: AnyType
](*, dest: Pointer[mut=True, T, _], src: Pointer[mut=False, T, _], count: Int,):
    """Copy `count * size_of[T]()` bytes from src to dest.

    Unlike `unsafe_memcpy`, the memory regions are allowed to overlap.

    Parameters:
        T: The element type.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.
    """
    var n = count * size_of[T]()
    if __is_run_in_comptime_interpreter:
        for i in range(n):
            dest.unsafe_bitcast[Byte]().unsafe_offset(i).unsafe_store(
                src.unsafe_bitcast[Byte]().unsafe_offset(i).unsafe_load()
            )
    else:
        llvm_intrinsic["llvm.memmove", NoneType](
            # <dest>, <src>, <len>, <isvolatile>
            dest.unsafe_bitcast[Byte](),
            src.unsafe_bitcast[Byte](),
            n,
            False,
        )


@always_inline
@deprecated(use=unsafe_memmove)
def memmove[
    T: AnyType
](
    *,
    dest: UnsafePointer[mut=True, T, _],
    src: UnsafePointer[mut=False, T, _],
    count: Int,
):
    """Copy `count * size_of[T]()` bytes from src to dest.

    Unlike `memcpy`, the memory regions are allowed to overlap.

    Parameters:
        T: The element type.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.
    """
    unsafe_memmove(dest=dest, src=src, count=count)


# ===-----------------------------------------------------------------------===#
# memset
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def _memset_impl(ptr: Pointer[mut=True, Byte, ...], value: Byte, count: Int):
    def fill[width: Int](offset: Int) {imm}:
        ptr.unsafe_store(offset, SIMD[DType.uint8, width](value))

    comptime simd_width = simd_width_of[Byte]()
    vectorize[simd_width](count, fill)


@always_inline
def unsafe_memset(ptr: Pointer[mut=True, ...], value: Byte, count: Int):
    """Fills memory with the given value.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        value: The value to fill with.
        count: Number of elements to fill (in elements, not bytes).
    """
    _memset_impl(ptr.unsafe_bitcast[Byte](), value, count * size_of[ptr.T]())


@always_inline
@deprecated(use=unsafe_memset)
def memset(ptr: Pointer[mut=True, ...], value: Byte, count: Int):
    """Fills memory with the given value.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        value: The value to fill with.
        count: Number of elements to fill (in elements, not bytes).
    """
    unsafe_memset(ptr, value, count)


# ===-----------------------------------------------------------------------===#
# memset_zero
# ===-----------------------------------------------------------------------===#


@always_inline
def unsafe_memset_zero(ptr: Pointer[mut=True, ...], count: Int):
    """Fills memory with zeros.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        count: Number of elements to fill (in elements, not bytes).
    """
    unsafe_memset(ptr, 0, count)


@always_inline
def unsafe_memset_zero[
    dtype: DType, //, *, count: Int
](ptr: Pointer[mut=True, Scalar[dtype], ...]):
    """Fills memory with zeros.

    Parameters:
        dtype: The element type.
        count: Number of elements to fill (in elements, not bytes).

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
    """

    comptime if count > 128:
        return unsafe_memset_zero(ptr, count)

    def fill[width: Int](offset: Int) {imm}:
        ptr.unsafe_store(offset, SIMD[dtype, width](0))

    vectorize[simd_width_of[dtype]()](count, fill)


@always_inline
@deprecated(use=unsafe_memset_zero)
def memset_zero(ptr: Pointer[mut=True, ...], count: Int):
    """Fills memory with zeros.

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
        count: Number of elements to fill (in elements, not bytes).
    """
    unsafe_memset_zero(ptr, count)


@always_inline
@deprecated(use=unsafe_memset_zero)
def memset_zero[
    dtype: DType, //, *, count: Int
](ptr: Pointer[mut=True, Scalar[dtype], ...]):
    """Fills memory with zeros.

    Parameters:
        dtype: The element type.
        count: Number of elements to fill (in elements, not bytes).

    Args:
        ptr: Pointer to the beginning of the memory block to fill.
    """
    unsafe_memset_zero[count=count](ptr)


# ===-----------------------------------------------------------------------===#
# malloc
# ===-----------------------------------------------------------------------===#


@always_inline
def _malloc[
    type: AnyType,
    /,
](
    size: Int,
    /,
    *,
    alignment: Int = align_of[type](),
    out result: OptionalPointer[
        type,
        MutUntrackedOrigin,
        address_space=AddressSpace.GENERIC,
    ],
):
    comptime MlirPointerType = type_of(result).T._mlir_type
    var mlir_pointer: MlirPointerType

    comptime if is_gpu():
        comptime enable_gpu_malloc = get_defined_string[
            "ENABLE_GPU_MALLOC", "true"
        ]()
        # no runtime allocation on GPU
        codegen_unreachable[
            enable_gpu_malloc != "true",
            "runtime allocation on GPU not allowed",
        ]()

        mlir_pointer = external_call["malloc", MlirPointerType](size)
    else:
        mlir_pointer = __mlir_op.`pop.aligned_alloc`[_type=MlirPointerType](
            alignment.__mlir_index__(), size.__mlir_index__()
        )

    # SAFETY: Due to the niche optimization, `Optional[Pointer]` is
    # represented exactly as the `MlirPointerType` so we can do a bit-cast.
    result = Pointer(to=mlir_pointer).unsafe_bitcast[type_of(result)]()[]


# ===-----------------------------------------------------------------------===#
# aligned_free
# ===-----------------------------------------------------------------------===#


@always_inline
def _free(ptr: Pointer[mut=True, ...]):
    comptime if is_gpu():
        libc.free(ptr.unsafe_bitcast[NoneType]())
    else:
        __mlir_op.`pop.aligned_free`(ptr._get_kgen_pointer())


@always_inline
def _free(ptr: OptionalPointer[mut=True, ...]):
    comptime if is_gpu():
        libc.free(unsafe_cast[Type=NoneType, origin=MutUntrackedOrigin](ptr))
    else:
        comptime KgenPointerType = type_of(ptr).T._mlir_type
        # SAFETY: Due to the niche optimization, `Optional[Pointer]` is
        # represented exactly as the `KgenPointerType` so we can do a bit-cast.
        var kgen_pointer = Pointer(to=ptr).unsafe_bitcast[KgenPointerType]()[]
        __mlir_op.`pop.aligned_free`(kgen_pointer)


# ===-----------------------------------------------------------------------===#
# is_trivial_* functions
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def is_trivially_movable[T: Movable]() -> Bool:
    """Returns whether `T` has a trivial move constructor.

    A move constructor is trivial when the compiler generates it and all of
    `T`'s fields are themselves trivially movable. In practice this means the
    value can be moved by copying its bits to a new location without any
    additional side effects.

    Parameters:
        T: The type to check.

    Returns:
        `True` if `T` has a trivial move constructor.
    """
    return T.__move_ctor_is_trivial


@always_inline("nodebug")
def is_trivially_copyable[T: Copyable]() -> Bool:
    """Returns whether `T` has a trivial copy constructor.

    A copy constructor is trivial when the compiler generates it and all of
    `T`'s fields are themselves trivially copyable. In practice this means the
    value can be copied by duplicating its bits to a new location without any
    additional side effects.

    Parameters:
        T: The type to check.

    Returns:
        `True` if `T` has a trivial copy constructor.
    """
    return T.__copy_ctor_is_trivial


@always_inline("nodebug")
def is_trivially_deletable[T: AnyType]() -> Bool:
    """Returns whether `T` has a trivial destructor.

    A destructor is trivial when the compiler generates it and all of `T`'s
    fields are themselves trivially destructible. In practice this means
    `__del__` is a no-op. A non-`ImplicitlyDeletable` (linear) type returns `False`

    Parameters:
        T: The type to check.

    Returns:
        `True` if `T` has a trivial destructor.
    """
    comptime if conforms_to(T, ImplicitlyDeletable):
        return T.__del__is_trivial
    else:
        return False


# ===-----------------------------------------------------------------------===#
# Uninitialized Memory Ops
# ===-----------------------------------------------------------------------===#


@always_inline
def unsafe_uninit_move_n[
    T: Movable,
    //,
    *,
    overlapping: Bool,
](*, dest: Pointer[mut=True, T, _], src: Pointer[mut=True, T, _], count: Int,):
    """Move `count` values from `src` into memory at `dest`.

    This function transfers ownership of `count` values from the source memory
    to the destination memory. After this call, the source values should be
    treated as uninitialized, and the destination values are valid and
    initialized.

    For types with trivial move constructors, this is optimized to a single
    `unsafe_memcpy` (or `unsafe_memmove` when `overlapping=True`) operation.
    Otherwise, it manually moves each element.

    The destination memory is treated as a raw span of bits to write to. Any
    existing values at `dest` are silently overwritten without being destroyed.
    For types with non-trivial destructors, this can cause memory leaks. Call
    `unsafe_destroy_n()` on the destination region first if it contains
    initialized values that need cleanup. For trivial types like `Int`, this is
    not a concern.

    Parameters:
        T: The type of values to move, which must be `Movable`.
        overlapping: If False, the function assumes `src` and `dest` do not
            overlap and uses `unsafe_memcpy`. If True, the function assumes
            `src` and `dest` may overlap and uses `unsafe_memmove` to handle
            this safely.

    Args:
        dest: Pointer to the destination memory region.
        src: Pointer to the source memory region. Must point to initialized
            values.
        count: The number of elements to move.

    Safety:

    - `dest` must point to a valid memory region with space for at least
        `count` elements of type `T`.
    - `src` must point to a valid memory region containing at least `count`
        **initialized** elements of type `T`.
    - If `overlapping=False`, the `src` and `dest` memory regions must **not**
        overlap. Overlapping regions with `overlapping=False` is undefined
        behavior.
    """

    comptime if is_trivially_movable[T]():
        comptime if overlapping:
            unsafe_memmove(dest=dest, src=src, count=count)
        else:
            unsafe_memcpy(dest=dest, src=src, count=count)
    else:
        for i in range(count):
            dest.unsafe_offset(i).unsafe_write_move_from(src.unsafe_offset(i))


@always_inline
@deprecated(use=unsafe_uninit_move_n)
def uninit_move_n[
    T: Movable,
    //,
    *,
    overlapping: Bool,
](
    *,
    dest: UnsafePointer[mut=True, T, _],
    src: UnsafePointer[mut=True, T, _],
    count: Int,
):
    """Move `count` values from `src` into memory at `dest`.

    This function transfers ownership of `count` values from the source memory
    to the destination memory. After this call, the source values should be
    treated as uninitialized, and the destination values are valid and
    initialized.

    For types with trivial move constructors, this is optimized to a single
    `unsafe_memcpy` (or `memmove` when `overlapping=True`) operation. Otherwise,
    it manually moves each element.

    The destination memory is treated as a raw span of bits to write to. Any
    existing values at `dest` are silently overwritten without being destroyed.
    For types with non-trivial destructors, this can cause memory leaks. Call
    `unsafe_destroy_n()` on the destination region first if it contains
    initialized values that need cleanup. For trivial types like `Int`, this is
    not a concern.

    Parameters:
        T: The type of values to move, which must be `Movable`.
        overlapping: If False, the function assumes `src` and `dest` do not
            overlap and uses `unsafe_memcpy`. If True, the function assumes
            `src` and `dest` may overlap and uses `memmove` to handle this
            safely.

    Args:
        dest: Pointer to the destination memory region.
        src: Pointer to the source memory region. Must point to initialized
            values.
        count: The number of elements to move.

    Safety:

    - `dest` must point to a valid memory region with space for at least
        `count` elements of type `T`.
    - `src` must point to a valid memory region containing at least `count`
        **initialized** elements of type `T`.
    - If `overlapping=False`, the `src` and `dest` memory regions must **not**
        overlap. Overlapping regions with `overlapping=False` is undefined
        behavior.
    """
    unsafe_uninit_move_n[overlapping=overlapping](
        dest=dest, src=src, count=count
    )


@always_inline
def unsafe_uninit_copy_n[
    T: Copyable,
    //,
    *,
    overlapping: Bool,
](*, dest: Pointer[mut=True, T, _], src: Pointer[mut=False, T, _], count: Int,):
    """Copy `count` values from `src` into memory at `dest`.

    This function creates copies of `count` values from the source memory in the
    destination memory. After this call, both source and destination values are
    valid and initialized.

    For types with trivial copy constructors, this is optimized to a single
    `unsafe_memcpy` (or `unsafe_memmove` when `overlapping=True`) operation.
    Otherwise, it calls `unsafe_write()` on each element.

    The destination memory is treated as a raw span of bits to write to. Any
    existing values at `dest` are silently overwritten without being destroyed.
    For types with non-trivial destructors, this can cause memory leaks. Call
    `unsafe_destroy_n()` on the destination region first if it contains
    initialized values that need cleanup. For trivial types like `Int`, this is
    not a concern.

    Parameters:
        T: The type of values to copy, which must be `Copyable`.
        overlapping: If False, the function assumes `src` and `dest` do not
            overlap and uses `unsafe_memcpy`. If True, the function assumes
            `src` and `dest` may overlap and uses `unsafe_memmove` to handle
            this safely.

    Args:
        dest: Pointer to the destination memory region.
        src: Pointer to the source memory region. Must point to initialized
            values.
        count: The number of elements to copy.

    Safety:

    - `dest` must point to a valid memory region with space for at least
        `count` elements of type `T`.
    - `src` must point to a valid memory region containing at least `count`
        **initialized** elements of type `T`.
    - If `overlapping=False`, the `src` and `dest` memory regions must **not**
        overlap. Overlapping regions with `overlapping=False` is undefined
        behavior.
    """

    comptime if is_trivially_copyable[T]():
        comptime if overlapping:
            unsafe_memmove(dest=dest, src=src, count=count)
        else:
            unsafe_memcpy(dest=dest, src=src, count=count)
    else:
        for i in range(count):
            dest.unsafe_offset(i).unsafe_write(copy=src.unsafe_offset(i)[])


@always_inline
@deprecated(use=unsafe_uninit_copy_n)
def uninit_copy_n[
    T: Copyable,
    //,
    *,
    overlapping: Bool,
](
    *,
    dest: UnsafePointer[mut=True, T, _],
    src: UnsafePointer[mut=False, T, _],
    count: Int,
):
    """Copy `count` values from `src` into memory at `dest`.

    This function creates copies of `count` values from the source memory in the
    destination memory. After this call, both source and destination values are
    valid and initialized.

    For types with trivial copy constructors, this is optimized to a single
    `unsafe_memcpy` (or `memmove` when `overlapping=True`) operation. Otherwise,
    it calls `unsafe_write()` on each element.

    The destination memory is treated as a raw span of bits to write to. Any
    existing values at `dest` are silently overwritten without being destroyed.
    For types with non-trivial destructors, this can cause memory leaks. Call
    `unsafe_destroy_n()` on the destination region first if it contains
    initialized values that need cleanup. For trivial types like `Int`, this is
    not a concern.

    Parameters:
        T: The type of values to copy, which must be `Copyable`.
        overlapping: If False, the function assumes `src` and `dest` do not
            overlap and uses `unsafe_memcpy`. If True, the function assumes
            `src` and `dest` may overlap and uses `memmove` to handle this
            safely.

    Args:
        dest: Pointer to the destination memory region.
        src: Pointer to the source memory region. Must point to initialized
            values.
        count: The number of elements to copy.

    Safety:

    - `dest` must point to a valid memory region with space for at least
        `count` elements of type `T`.
    - `src` must point to a valid memory region containing at least `count`
        **initialized** elements of type `T`.
    - If `overlapping=False`, the `src` and `dest` memory regions must **not**
        overlap. Overlapping regions with `overlapping=False` is undefined
        behavior.
    """
    unsafe_uninit_copy_n[overlapping=overlapping](
        dest=dest, src=src, count=count
    )


@always_inline
def unsafe_destroy_n[
    T: ImplicitlyDeletable
](pointer: Pointer[mut=True, T, _], count: Int):
    """Destroy `count` initialized values at `pointer`.

    This function runs the destructor for each of the `count` values, leaving
    the memory uninitialized.

    For types with trivial destructors, this is a no-op and generates no code.
    Otherwise, it calls `unsafe_deinit_pointee()` on each element.

    Parameters:
        T: The type of values to destroy, which must be `ImplicitlyDeletable`.

    Args:
        pointer: Pointer to the memory region containing values to destroy.
        count: The number of elements to destroy.

    Safety:

    - `pointer` must point to a valid memory region containing at least `count`
        **initialized** elements of type `T`.
    - After this call, the values at `pointer[0:count]` are uninitialized and
        must not be read or destroyed again until re-initialized.
    """

    comptime if is_trivially_deletable[T]():
        # Trivial destructors don't need to be called!
        pass
    else:
        for i in range(count):
            pointer.unsafe_offset(i).unsafe_deinit_pointee()


@always_inline
@deprecated(use=unsafe_destroy_n)
def destroy_n[
    T: ImplicitlyDeletable
](pointer: Pointer[mut=True, T, _], count: Int):
    """Destroy `count` initialized values at `pointer`.

    This function runs the destructor for each of the `count` values, leaving
    the memory uninitialized.

    For types with trivial destructors, this is a no-op and generates no code.
    Otherwise, it calls `unsafe_deinit_pointee()` on each element.

    Parameters:
        T: The type of values to destroy, which must be `ImplicitlyDeletable`.

    Args:
        pointer: Pointer to the memory region containing values to destroy.
        count: The number of elements to destroy.

    Safety:

    - `pointer` must point to a valid memory region containing at least `count`
        **initialized** elements of type `T`.
    - After this call, the values at `pointer[0:count]` are uninitialized and
        must not be read or destroyed again until re-initialized.
    """

    unsafe_destroy_n(pointer, count)


# ===-----------------------------------------------------------------------===#
# Ownership Ops
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def forget_deinit[T: AnyType](var value: T):
    """Takes ownership and skips running `__del__` deinitializers.

    This is a low-level operation, and should not be used unless necessary.
    Consider if refactoring to avoid needing this function would be more
    appropriate.

    This operation is not considered unsafe, as Mojo can not guarantee in
    general that destructors will eventually be run.

    Note: Take care to use `^` to transfer when passing `ImplicitlyCopyable`
    values to `forget_deinit()`, to avoid forgetting a copy instead of the
    original value.

    Parameters:
        T: The type of the value to discard without running a deinitializer.

    Args:
        value: The value to discard without running a deinitializer.

    Example:

    ```mojo
    from std.memory import forget_deinit

    @fieldwise_init
    struct Noisy:
        def __del__(deinit self):
            print("@ Noisy.__del__: Noisy is being deleted!")

    def main():
        var noisy = Noisy()

        # No deletion message is printed
        forget_deinit(noisy^)
    ```

    This will skip the destructor for the "root" `value` object and all of
    it's fields, recursively. Example:

    ```mojo
    from std.memory import forget_deinit

    @fieldwise_init
    struct Parent:
        var child: Child

        def __del__(deinit self):
            print("@ Parent.__del__")

    @fieldwise_init
    struct Child(Movable):
        def __del__(deinit self):
            print("@ Child.__del__")

    def main():
        var parent = Parent(Child())

        # Neither Parent.__del__ nor Child.__del__ is called.
        forget_deinit(parent^)
    ```
    """
    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(value))
