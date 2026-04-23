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
    s1: UnsafePointer[mut=False, Scalar[dtype], ...],
    s2: UnsafePointer[mut=False, Scalar[dtype], ...],
    count: Int,
) -> Int:
    for i in range(count):
        var s1i = s1[i]
        var s2i = s2[i]
        if s1i != s2i:
            return 1 if s1i > s2i else -1
    return 0


@always_inline
def _memcmp_opt_impl_unconstrained[
    dtype: DType, //
](
    s1: UnsafePointer[mut=False, Scalar[dtype], ...],
    s2: UnsafePointer[mut=False, Scalar[dtype], ...],
    count: Int,
) -> Int:
    comptime simd_width = simd_width_of[dtype]()
    if count < simd_width:
        for i in range(count):
            var s1i = s1[i]
            var s2i = s2[i]
            if s1i != s2i:
                return 1 if s1i > s2i else -1
        return 0

    var last = count - simd_width

    for i in range(0, last, simd_width):
        var s1i = s1.load[width=simd_width](i)
        var s2i = s2.load[width=simd_width](i)
        var diff = s1i.ne(s2i)
        if any(diff):
            var index = Int(
                diff.select(
                    iota[DType.uint8, simd_width](),
                    SIMD[DType.uint8, simd_width](255),
                ).reduce_min()
            )
            return -1 if s1i[index] < s2i[index] else 1

    var s1i = s1.load[width=simd_width](last)
    var s2i = s2.load[width=simd_width](last)
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
    s1: UnsafePointer[mut=False, Scalar[dtype], ...],
    s2: UnsafePointer[mut=False, Scalar[dtype], ...],
    count: Int,
) -> Int where dtype.is_integral():
    if __is_run_in_comptime_interpreter:
        return _memcmp_impl_unconstrained(s1, s2, count)
    else:
        return _memcmp_opt_impl_unconstrained(s1, s2, count)


@always_inline
def memcmp[
    type: AnyType, address_space: AddressSpace
](
    s1: UnsafePointer[mut=False, type, _, address_space=address_space],
    s2: UnsafePointer[mut=False, type, _, address_space=address_space],
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
            s1.bitcast[Int32](),
            s2.bitcast[Int32](),
            byte_count // size_of[DType.int32](),
        )

    return _memcmp_impl(s1.bitcast[Byte](), s2.bitcast[Byte](), byte_count)


# ===-----------------------------------------------------------------------===#
# memcpy
# ===-----------------------------------------------------------------------===#


@always_inline
def _memcpy_impl(
    dest_data: UnsafePointer[mut=True, Byte, ...],
    src_data: UnsafePointer[mut=False, Byte, ...],
    n: Int,
):
    """Copies a memory area.

    Args:
        dest_data: The destination pointer.
        src_data: The source pointer.
        n: The number of bytes to copy.
    """

    def copy[width: Int](offset: Int) unified {read}:
        dest_data.store(offset, src_data.load[width=width](offset))

    comptime if is_gpu():
        vectorize[simd_bit_width()](n, copy)

        return

    if n < 5:
        if n == 0:
            return
        dest_data[0] = src_data[0]
        dest_data[n - 1] = src_data[n - 1]
        if n <= 2:
            return
        dest_data[1] = src_data[1]
        dest_data[n - 2] = src_data[n - 2]
        return

    if n <= 16:
        if n >= 8:
            var ui64_size = size_of[UInt64]()
            dest_data.bitcast[UInt64]().store[alignment=1](
                0, src_data.bitcast[UInt64]().load[alignment=1](0)
            )
            (dest_data + n - ui64_size).bitcast[UInt64]().store[alignment=1](
                0,
                (src_data + n - ui64_size)
                .bitcast[UInt64]()
                .load[alignment=1](0),
            )
            return

        var ui32_size = size_of[UInt32]()
        dest_data.bitcast[UInt32]().store[alignment=1](
            0, src_data.bitcast[UInt32]().load[alignment=1](0)
        )
        (dest_data + n - ui32_size).bitcast[UInt32]().store[alignment=1](
            0,
            (src_data + n - ui32_size).bitcast[UInt32]().load[alignment=1](0),
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
    overlapping memory regions, use `memmove`.

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

    var n = count * size_of[dest.T.type]()

    var dest_bytes = dest.unsafe_value().bitcast[Byte]()
    var src_bytes = src.unsafe_value().bitcast[Byte]()

    if __is_run_in_comptime_interpreter:
        # A fast version for the interpreter to evaluate
        # this function during compile time.
        llvm_intrinsic["llvm.memcpy", NoneType](
            dest_bytes, src_bytes, n._int_mlir_index()
        )
    else:
        _memcpy_impl(dest_bytes, src_bytes, n)


# ===-----------------------------------------------------------------------===#
# memmove
# ===-----------------------------------------------------------------------===#


@always_inline
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
    var n = count * size_of[T]()
    if __is_run_in_comptime_interpreter:
        for i in range(n):
            (dest.bitcast[Byte]() + i).store((src.bitcast[Byte]() + i).load())
    else:
        llvm_intrinsic["llvm.memmove", NoneType](
            # <dest>, <src>, <len>, <isvolatile>
            dest.bitcast[Byte](),
            src.bitcast[Byte](),
            n,
            False,
        )


# ===-----------------------------------------------------------------------===#
# memset
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def _memset_impl(
    ptr: UnsafePointer[mut=True, Byte, ...], value: Byte, count: Int
):
    def fill[width: Int](offset: Int) unified {read}:
        ptr.store(offset, SIMD[DType.uint8, width](value))

    comptime simd_width = simd_width_of[Byte]()
    vectorize[simd_width](count, fill)


@always_inline
def memset(ptr: UnsafePointer[mut=True, ...], value: Byte, count: Int):
    """Fills memory with the given value.

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
        value: The value to fill with.
        count: Number of elements to fill (in elements, not bytes).
    """
    _memset_impl(ptr.bitcast[Byte](), value, count * size_of[ptr.type]())


# ===-----------------------------------------------------------------------===#
# memset_zero
# ===-----------------------------------------------------------------------===#


@always_inline
def memset_zero(ptr: UnsafePointer[mut=True, ...], count: Int):
    """Fills memory with zeros.

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
        count: Number of elements to fill (in elements, not bytes).
    """
    memset(ptr, 0, count)


@always_inline
def memset_zero[
    dtype: DType, //, *, count: Int
](ptr: UnsafePointer[mut=True, Scalar[dtype], ...]):
    """Fills memory with zeros.

    Parameters:
        dtype: The element type.
        count: Number of elements to fill (in elements, not bytes).

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
    """

    comptime if count > 128:
        return memset_zero(ptr, count)

    def fill[width: Int](offset: Int) unified {read}:
        ptr.store(offset, SIMD[dtype, width](0))

    vectorize[simd_width_of[dtype]()](count, fill)


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
    out result: Optional[
        UnsafePointer[
            type,
            MutExternalOrigin,
            address_space=AddressSpace.GENERIC,
        ]
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
            alignment._int_mlir_index(), size._int_mlir_index()
        )

    # SAFETY: Due to the niche optimization, `Optional[UnsafePointer]` is
    # represented exactly as the `MlirPointerType` so we can do a bit-cast.
    result = UnsafePointer(to=mlir_pointer).bitcast[type_of(result)]()[]


# ===-----------------------------------------------------------------------===#
# aligned_free
# ===-----------------------------------------------------------------------===#


@always_inline
def _free(ptr: UnsafePointer[mut=True, ...]):
    comptime if is_gpu():
        libc.free(ptr.bitcast[NoneType]())
    else:
        __mlir_op.`pop.aligned_free`(ptr.address)


@always_inline
def _free(ptr: OptionalUnsafePointer[mut=True, ...]):
    comptime if is_gpu():
        libc.free(unsafe_cast[Type=NoneType, origin=MutExternalOrigin](ptr))
    else:
        comptime KgenPointerType = type_of(ptr).T._mlir_type
        # SAFETY: Due to the niche optimization, `Optional[UnsafePointer]` is
        # represented exactly as the `KgenPointerType` so we can do a bit-cast.
        var kgen_pointer = UnsafePointer(to=ptr).bitcast[KgenPointerType]()[]
        __mlir_op.`pop.aligned_free`(kgen_pointer)


# ===-----------------------------------------------------------------------===#
# Uninitialized Memory Ops
# ===-----------------------------------------------------------------------===#


@always_inline
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
    `memcpy` (or `memmove` when `overlapping=True`) operation. Otherwise, it
    manually moves each element.

    The destination memory is treated as a raw span of bits to write to. Any
    existing values at `dest` are silently overwritten without being destroyed.
    For types with non-trivial destructors, this can cause memory leaks. Call
    `destroy_n()` on the destination region first if it contains initialized
    values that need cleanup. For trivial types like `Int`, this is not a
    concern.

    Parameters:
        T: The type of values to move, which must be `Movable`.
        overlapping: If False, the function assumes `src` and `dest` do not
            overlap and uses `memcpy`. If True, the function assumes `src` and
            `dest` may overlap and uses `memmove` to handle this safely.

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

    comptime if T.__move_ctor_is_trivial:
        comptime if overlapping:
            memmove(dest=dest, src=src, count=count)
        else:
            memcpy(dest=dest, src=src, count=count)
    else:
        for i in range(count):
            (dest + i).init_pointee_move_from(src + i)


@always_inline
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
    `memcpy` (or `memmove` when `overlapping=True`) operation. Otherwise, it
    calls `init_pointee_copy()` on each element.

    The destination memory is treated as a raw span of bits to write to. Any
    existing values at `dest` are silently overwritten without being destroyed.
    For types with non-trivial destructors, this can cause memory leaks. Call
    `destroy_n()` on the destination region first if it contains initialized
    values that need cleanup. For trivial types like `Int`, this is not a
    concern.

    Parameters:
        T: The type of values to copy, which must be `Copyable`.
        overlapping: If False, the function assumes `src` and `dest` do not
            overlap and uses `memcpy`. If True, the function assumes `src` and
            `dest` may overlap and uses `memmove` to handle this safely.

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

    comptime if T.__copy_ctor_is_trivial:
        comptime if overlapping:
            memmove(dest=dest, src=src, count=count)
        else:
            memcpy(dest=dest, src=src, count=count)
    else:
        for i in range(count):
            (dest + i).init_pointee_copy((src + i)[])


@always_inline
def destroy_n[
    T: ImplicitlyDestructible
](pointer: UnsafePointer[mut=True, T, _], count: Int):
    """Destroy `count` initialized values at `pointer`.

    This function runs the destructor for each of the `count` values, leaving
    the memory uninitialized.

    For types with trivial destructors, this is a no-op and generates no code.
    Otherwise, it calls `destroy_pointee()` on each element.

    Parameters:
        T: The type of values to destroy, which must be `ImplicitlyDestructible`.

    Args:
        pointer: Pointer to the memory region containing values to destroy.
        count: The number of elements to destroy.

    Safety:

    - `pointer` must point to a valid memory region containing at least `count`
        **initialized** elements of type `T`.
    - After this call, the values at `pointer[0:count]` are uninitialized and
        must not be read or destroyed again until re-initialized.
    """

    comptime if T.__del__is_trivial:
        # Trivial destructors don't need to be called!
        pass
    else:
        for i in range(count):
            (pointer + i).destroy_pointee()


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
