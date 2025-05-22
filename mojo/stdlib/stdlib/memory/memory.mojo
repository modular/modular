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
"""Defines functions for memory manipulations.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import memcmp
```
"""


from collections.string.string_slice import _get_kgen_string
from math import align_down, iota
from sys import _libc as libc
from sys import (
    alignof,
    external_call,
    is_compile_time,
    is_gpu,
    llvm_intrinsic,
    simdbitwidth,
    simdwidthof,
    sizeof,
)

from algorithm import vectorize
from memory.pointer import AddressSpace, _GPUAddressSpace

# ===-----------------------------------------------------------------------===#
# memcmp
# ===-----------------------------------------------------------------------===#


@always_inline
fn _memcmp_impl_unconstrained[
    dtype: DType, //
](
    s1: UnsafePointer[Scalar[dtype], **_],
    s2: UnsafePointer[Scalar[dtype], **_],
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
    s1: UnsafePointer[Scalar[dtype], **_],
    s2: UnsafePointer[Scalar[dtype], **_],
    count: Int,
) -> Int:
    alias simd_width = simdwidthof[dtype]()
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
        var diff = s1i != s2i
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
    var diff = s1i != s2i
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
    s1: UnsafePointer[Scalar[dtype], **_],
    s2: UnsafePointer[Scalar[dtype], **_],
    count: Int,
) -> Int:
    constrained[dtype.is_integral(), "the input dtype must be integral"]()
    if is_compile_time():
        return _memcmp_impl_unconstrained(s1, s2, count)
    else:
        return _memcmp_opt_impl_unconstrained(s1, s2, count)


@always_inline
fn memcmp[
    type: AnyType, address_space: AddressSpace
](
    s1: UnsafePointer[type, address_space=address_space],
    s2: UnsafePointer[type, address_space=address_space],
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
    var byte_count = count * sizeof[type]()

    @parameter
    if sizeof[type]() >= sizeof[DType.int32]():
        return _memcmp_impl(
            s1.bitcast[Int32](),
            s2.bitcast[Int32](),
            byte_count // sizeof[DType.int32](),
        )

    return _memcmp_impl(s1.bitcast[Byte](), s2.bitcast[Byte](), byte_count)


# ===-----------------------------------------------------------------------===#
# memcpy
# ===-----------------------------------------------------------------------===#


@always_inline
fn _memcpy_impl(
    dest_data: UnsafePointer[Byte, mut=True, *_, **_],
    src_data: UnsafePointer[Byte, mut=False, *_, **_],
    n: Int,
):
    """Copies a memory area.

    Args:
        dest_data: The destination pointer.
        src_data: The source pointer.
        n: The number of bytes to copy.
    """

    @parameter
    fn copy[width: Int](offset: Int):
        dest_data.store(offset, src_data.load[width=width](offset))

    @parameter
    if is_gpu():
        vectorize[copy, simdbitwidth()](n)

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
            var ui64_size = sizeof[UInt64]()
            dest_data.bitcast[UInt64]().store[alignment=1](
                0, src_data.bitcast[UInt64]().load[alignment=1](0)
            )
            dest_data.offset(n - ui64_size).bitcast[UInt64]().store[
                alignment=1
            ](
                0,
                src_data.offset(n - ui64_size)
                .bitcast[UInt64]()
                .load[alignment=1](0),
            )
            return

        var ui32_size = sizeof[UInt32]()
        dest_data.bitcast[UInt32]().store[alignment=1](
            0, src_data.bitcast[UInt32]().load[alignment=1](0)
        )
        dest_data.offset(n - ui32_size).bitcast[UInt32]().store[alignment=1](
            0,
            src_data.offset(n - ui32_size)
            .bitcast[UInt32]()
            .load[alignment=1](0),
        )
        return

    # TODO (#10566): This branch appears to cause a 12% regression in BERT by
    # slowing down broadcast ops
    # if n <= 32:
    #    alias simd_16xui8_size = 16 * sizeof[Int8]()
    #    dest_data.store[width=16](src_data.load[width=16]())
    #    # note that some of these bytes may have already been written by the
    #    # previous simd_store
    #    dest_data.store[width=16](
    #        n - simd_16xui8_size, src_data.load[width=16](n - simd_16xui8_size)
    #    )
    #    return

    # Copy in 32-byte chunks.
    vectorize[copy, 32](n)


@always_inline
fn memcpy[
    T: AnyType
](
    dest: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
    src: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
    count: Int,
):
    """Copies a memory area.

    Parameters:
        T: The element type.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.
    """
    var n = count * sizeof[dest.type]()

    if is_compile_time():
        # A fast version for the interpreter to evaluate
        # this function during compile time.
        llvm_intrinsic["llvm.memcpy", NoneType](
            dest.bitcast[Byte]().origin_cast[origin=MutableAnyOrigin](),
            src.bitcast[Byte]().origin_cast[origin=MutableAnyOrigin](),
            n,
        )
    else:
        _memcpy_impl(
            dest.bitcast[Byte]().origin_cast[origin=MutableAnyOrigin](),
            src.bitcast[Byte]().origin_cast[mut=False](),
            n,
        )


# ===-----------------------------------------------------------------------===#
# memset
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _memset_impl[
    address_space: AddressSpace
](
    ptr: UnsafePointer[Byte, address_space=address_space],
    value: Byte,
    count: Int,
):
    @parameter
    fn fill[width: Int](offset: Int):
        ptr.store(offset, SIMD[DType.uint8, width](value))

    alias simd_width = simdwidthof[Byte]()
    vectorize[fill, simd_width](count)


@always_inline
fn memset[
    type: AnyType, address_space: AddressSpace
](
    ptr: UnsafePointer[type, address_space=address_space],
    value: Byte,
    count: Int,
):
    """Fills memory with the given value.

    Parameters:
        type: The element dtype.
        address_space: The address space of the pointer.

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
        value: The value to fill with.
        count: Number of elements to fill (in elements, not bytes).
    """
    _memset_impl(ptr.bitcast[Byte](), value, count * sizeof[type]())


# ===-----------------------------------------------------------------------===#
# memset_zero
# ===-----------------------------------------------------------------------===#


@always_inline
fn memset_zero[
    type: AnyType, address_space: AddressSpace, //
](ptr: UnsafePointer[type, address_space=address_space], count: Int):
    """Fills memory with zeros.

    Parameters:
        type: The element type.
        address_space: The address space of the pointer.

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
        count: Number of elements to fill (in elements, not bytes).
    """
    memset(ptr, 0, count)


@always_inline
fn memset_zero[
    dtype: DType, address_space: AddressSpace, //, *, count: Int
](ptr: UnsafePointer[Scalar[dtype], address_space=address_space]):
    """Fills memory with zeros.

    Parameters:
        dtype: The element type.
        address_space: The address space of the pointer.
        count: Number of elements to fill (in elements, not bytes).

    Args:
        ptr: UnsafePointer to the beginning of the memory block to fill.
    """

    @parameter
    if count > 128:
        return memset_zero(ptr, count)

    @parameter
    fn fill[width: Int](offset: Int):
        ptr.store(offset, SIMD[dtype, width](0))

    vectorize[fill, simdwidthof[dtype](), size=count]()


# ===-----------------------------------------------------------------------===#
# stack_allocation
# ===-----------------------------------------------------------------------===#


@always_inline
fn stack_allocation[
    count: Int,
    dtype: DType,
    /,
    alignment: Int = alignof[dtype]() if is_gpu() else 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[Scalar[dtype], address_space=address_space]:
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


@always_inline
fn stack_allocation[
    count: Int,
    type: AnyType,
    /,
    name: Optional[StaticString] = None,
    alignment: Int = alignof[type]() if is_gpu() else 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[type, address_space=address_space]:
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

    @parameter
    if is_gpu():
        # On NVGPU, SHARED and CONSTANT address spaces lower to global memory.

        alias global_name = name.value() if name else "_global_alloc"

        @parameter
        if address_space == _GPUAddressSpace.SHARED:
            return __mlir_op.`pop.global_alloc`[
                name = _get_kgen_string[global_name](),
                count = count.value,
                memoryType = __mlir_attr.`#pop<global_alloc_addr_space gpu_shared>`,
                _type = UnsafePointer[
                    type, address_space=address_space, alignment=alignment
                ]._mlir_type,
                alignment = alignment.value,
            ]()
        elif address_space == _GPUAddressSpace.CONSTANT:
            # No need to annotation this global_alloc because constants in
            # GPU shared memory won't prevent llvm module splitting to
            # happen since they are immutables.
            return __mlir_op.`pop.global_alloc`[
                name = _get_kgen_string[global_name](),
                count = count.value,
                _type = UnsafePointer[
                    type, address_space=address_space, alignment=alignment
                ]._mlir_type,
                alignment = alignment.value,
            ]()

        # MSTDL-797: The NVPTX backend requires that `alloca` instructions may
        # only have generic address spaces. When allocating LOCAL memory,
        # addrspacecast the resulting pointer.
        elif address_space == _GPUAddressSpace.LOCAL:
            var generic_ptr = __mlir_op.`pop.stack_allocation`[
                count = count.value,
                _type = UnsafePointer[type]._mlir_type,
                alignment = alignment.value,
            ]()
            return __mlir_op.`pop.pointer.bitcast`[
                _type = UnsafePointer[
                    type, address_space=address_space
                ]._mlir_type
            ](generic_ptr)

    # Perofrm a stack allocation of the requested size, alignment, and type.
    return __mlir_op.`pop.stack_allocation`[
        count = count.value,
        _type = UnsafePointer[
            type, address_space=address_space, alignment=alignment
        ]._mlir_type,
        alignment = alignment.value,
    ]()


# ===-----------------------------------------------------------------------===#
# malloc
# ===-----------------------------------------------------------------------===#


@always_inline
fn _malloc[
    type: AnyType,
    /,
    *,
    alignment: Int = alignof[type]() if is_gpu() else 1,
](size: Int, /) -> UnsafePointer[
    type, address_space = AddressSpace.GENERIC, alignment=alignment
]:
    @parameter
    if is_gpu():
        return external_call[
            "malloc",
            UnsafePointer[NoneType, address_space = AddressSpace.GENERIC],
        ](size).bitcast[type]()
    else:
        return __mlir_op.`pop.aligned_alloc`[
            _type = UnsafePointer[
                type, address_space = AddressSpace.GENERIC
            ]._mlir_type
        ](alignment.value, size.value)


# ===-----------------------------------------------------------------------===#
# aligned_free
# ===-----------------------------------------------------------------------===#


@always_inline
fn _free(ptr: UnsafePointer[_, address_space = AddressSpace.GENERIC, *_, **_]):
    @parameter
    if is_gpu():
        libc.free(ptr.bitcast[NoneType]())
    else:
        __mlir_op.`pop.aligned_free`(ptr.address)
