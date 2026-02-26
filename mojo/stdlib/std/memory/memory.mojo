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
from memory import memcmp
```
"""


from collections.string.string_slice import _get_kgen_string
from math import iota
from sys import _libc as libc
from ffi import external_call
from sys import (
    align_of,
    codegen_unreachable,
    env_get_string,
    is_compile_time,
    is_gpu,
    llvm_intrinsic,
    simd_bit_width,
    simd_width_of,
    size_of,
)

from algorithm import vectorize

# ===-----------------------------------------------------------------------===#
# memcmp
# ===-----------------------------------------------------------------------===#


@always_inline
fn _memcmp_impl_unconstrained[
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
fn _memcmp_opt_impl_unconstrained[
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
fn _memcmp_impl[
    dtype: DType
](
    s1: UnsafePointer[mut=False, Scalar[dtype], ...],
    s2: UnsafePointer[mut=False, Scalar[dtype], ...],
    count: Int,
) -> Int where dtype.is_integral():
    if is_compile_time():
        return _memcmp_impl_unconstrained(s1, s2, count)
    else:
        return _memcmp_opt_impl_unconstrained(s1, s2, count)


@always_inline
fn memcmp[
    type: AnyType, address_space: AddressSpace
](
    s1: UnsafePointer[mut=False, type, address_space=address_space],
    s2: UnsafePointer[mut=False, type, address_space=address_space],
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
fn _memcpy_impl(
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

    fn copy[width: Int](offset: Int) unified {mut}:
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
fn memcpy[
    T: AnyType
](
    *,
    dest: UnsafePointer[mut=True, T],
    src: UnsafePointer[mut=False, T],
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
    """
    var n = count * size_of[dest.type]()

    if is_compile_time():
        # A fast version for the interpreter to evaluate
        # this function during compile time.
        llvm_intrinsic["llvm.memcpy", NoneType](
            dest.bitcast[Byte](), src.bitcast[Byte](), n
        )
    else:
        _memcpy_impl(dest.bitcast[Byte](), src.bitcast[Byte](), n)


# ===-----------------------------------------------------------------------===#
# memmove
# ===-----------------------------------------------------------------------===#


@always_inline
fn memmove[
    T: AnyType
](
    *,
    dest: UnsafePointer[mut=True, T],
    src: UnsafePointer[mut=False, T],
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
    if is_compile_time():
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
fn _memset_impl(
    ptr: UnsafePointer[mut=True, Byte, ...], value: Byte, count: Int
):
    fn fill[width: Int](offset: Int) unified {mut}:
        ptr.store(offset, SIMD[DType.uint8, width](value))

    comptime simd_width = simd_width_of[Byte]()
    vectorize[simd_width](count, fill)


@always_inline
fn memset(ptr: UnsafePointer[mut=True, _, ...], value: Byte, count: Int):
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
fn memset_zero(ptr: UnsafePointer[mut=True, _, ...], count: Int):
    """Fills memory with zeros.

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
        count: Number of elements to fill (in elements, not bytes).
    """
    memset(ptr, 0, count)


@always_inline
fn memset_zero[
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

    fn fill[width: Int](offset: Int) unified {mut}:
        ptr.store(offset, SIMD[dtype, width](0))

    vectorize[simd_width_of[dtype]()](count, fill)


# ===-----------------------------------------------------------------------===#
# stack_allocation
# ===-----------------------------------------------------------------------===#


# TODO(MSTDL-2015): ASAN error when updating to use `UnsafePointer`.
@always_inline
fn stack_allocation[
    count: Int,
    dtype: DType,
    /,
    alignment: Int = align_of[dtype](),
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[
    Scalar[dtype],
    MutExternalOrigin,
    address_space=address_space,
]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        dtype: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return stack_allocation[
        count, Scalar[dtype], alignment=alignment, address_space=address_space
    ]()


# TODO(MSTDL-2015): ASAN error when updating to use `UnsafePointer`.
@always_inline
fn stack_allocation[
    count: Int,
    type: AnyType,
    /,
    name: Optional[StaticString] = None,
    alignment: Int = align_of[type](),
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[type, MutExternalOrigin, address_space=address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        name: The name of the global variable (only honored in certain cases).
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    comptime if is_gpu():
        # On NVGPU, SHARED and CONSTANT address spaces lower to global memory.

        comptime global_name = name.value() if name else "_global_alloc"

        comptime if address_space == AddressSpace.SHARED:
            return __mlir_op.`pop.global_alloc`[
                name = _get_kgen_string[global_name](),
                count = count._mlir_value,
                memoryType = __mlir_attr.`#pop<global_alloc_addr_space gpu_shared>`,
                _type = UnsafePointer[
                    type, MutExternalOrigin, address_space=address_space
                ]._mlir_type,
                alignment = alignment._mlir_value,
            ]()
        elif address_space == AddressSpace.CONSTANT:
            # No need to annotation this global_alloc because constants in
            # GPU shared memory won't prevent llvm module splitting to
            # happen since they are immutables.
            return __mlir_op.`pop.global_alloc`[
                name = _get_kgen_string[global_name](),
                count = count._mlir_value,
                _type = UnsafePointer[
                    type, MutExternalOrigin, address_space=address_space
                ]._mlir_type,
                alignment = alignment._mlir_value,
            ]()

        # MSTDL-797: The NVPTX backend requires that `alloca` instructions may
        # only have generic address spaces. When allocating LOCAL memory,
        # addrspacecast the resulting pointer.
        elif address_space == AddressSpace.LOCAL:
            var generic_ptr = __mlir_op.`pop.stack_allocation`[
                count = count._mlir_value,
                _type = UnsafePointer[type, MutExternalOrigin]._mlir_type,
                alignment = alignment._mlir_value,
            ]()
            return __mlir_op.`pop.pointer.bitcast`[
                _type = UnsafePointer[
                    type, MutExternalOrigin, address_space=address_space
                ]._mlir_type
            ](generic_ptr)

    # Perform a stack allocation of the requested size, alignment, and type.
    return __mlir_op.`pop.stack_allocation`[
        count = count._mlir_value,
        _type = UnsafePointer[
            type, MutExternalOrigin, address_space=address_space
        ]._mlir_type,
        alignment = alignment._mlir_value,
    ]()


# ===-----------------------------------------------------------------------===#
# malloc
# ===-----------------------------------------------------------------------===#


@always_inline
fn _malloc[
    type: AnyType,
    /,
](
    size: Int,
    /,
    *,
    alignment: Int = align_of[type](),
    out res: UnsafePointer[
        type,
        MutExternalOrigin,
        address_space = AddressSpace.GENERIC,
    ],
):
    comptime if is_gpu():
        comptime enable_gpu_malloc = env_get_string[
            "ENABLE_GPU_MALLOC", "true"
        ]()
        # no runtime allocation on GPU
        codegen_unreachable[
            enable_gpu_malloc != "true",
            "runtime allocation on GPU not allowed",
        ]()

        comptime U = UnsafePointer[
            NoneType,
            MutExternalOrigin,
            address_space = AddressSpace.GENERIC,
        ]
        var ptr = external_call["malloc", U](size)
        return ptr.bitcast[type]()
    else:
        return __mlir_op.`pop.aligned_alloc`[_type = type_of(res)._mlir_type](
            alignment._mlir_value, size._mlir_value
        )


# ===-----------------------------------------------------------------------===#
# aligned_free
# ===-----------------------------------------------------------------------===#


@always_inline
fn _free(ptr: UnsafePointer[mut=True, ...]):
    comptime if is_gpu():
        libc.free(ptr.bitcast[NoneType]())
    else:
        __mlir_op.`pop.aligned_free`(ptr.address)


# ===-----------------------------------------------------------------------===#
# Uninitialized Memory Ops
# ===-----------------------------------------------------------------------===#


@always_inline
fn uninit_move_n[
    T: Movable,
    //,
    *,
    overlapping: Bool,
](
    *,
    dest: UnsafePointer[mut=True, T],
    src: UnsafePointer[mut=True, T],
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
fn uninit_copy_n[
    T: Copyable,
    //,
    *,
    overlapping: Bool,
](
    *,
    dest: UnsafePointer[mut=True, T],
    src: UnsafePointer[mut=False, T],
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
fn destroy_n[
    T: ImplicitlyDestructible
](pointer: UnsafePointer[mut=True, T], count: Int):
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
