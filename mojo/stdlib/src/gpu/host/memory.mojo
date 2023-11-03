# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA memory operations."""

from sys.info import sizeof

from memory.unsafe import DTypePointer, Pointer, bitcast

from ._utils import _check_error, _get_dylib_function
from .stream import Stream, _StreamImpl

# ===----------------------------------------------------------------------===#
# Memory
# ===----------------------------------------------------------------------===#


fn _malloc[type: AnyType](count: Int) raises -> Pointer[type]:
    var ptr = Pointer[UInt32]()
    _check_error(
        _get_dylib_function[fn (Pointer[Pointer[UInt32]], Int) -> Result](
            "cuMemAlloc_v2"
        )(Pointer.address_of(ptr), count * sizeof[type]())
    )
    return ptr.bitcast[type]()


fn _malloc[type: DType](count: Int) raises -> DTypePointer[type]:
    return _malloc[SIMD[type, 1]](count)


fn _malloc_managed[type: AnyType](count: Int) raises -> Pointer[type]:
    var ptr = Pointer[UInt32]()
    _check_error(
        _get_dylib_function[fn (Pointer[Pointer[UInt32]], Int) -> Result](
            "cuMemAllocManaged"
        )(Pointer.address_of(ptr), count * sizeof[type]())
    )
    return ptr.bitcast[type]()


fn _malloc_managed[type: DType](count: Int) raises -> DTypePointer[type]:
    return _malloc_managed[SIMD[type, 1]](count)


fn _free[type: AnyType](ptr: Pointer[type]) raises:
    _check_error(
        _get_dylib_function[fn (Pointer[UInt32]) -> Result]("cuMemFree_v2")(
            ptr.bitcast[UInt32]()
        )
    )


fn _free[type: DType](ptr: DTypePointer[type]) raises:
    _free(ptr._as_scalar_pointer())


fn _copy_host_to_device[
    type: AnyType
](device_dest: Pointer[type], host_src: Pointer[type], count: Int) raises:
    _check_error(
        _get_dylib_function[
            fn (Pointer[UInt32], Pointer[NoneType], Int) -> Result
        ]("cuMemcpyHtoD_v2")(
            device_dest.bitcast[UInt32](),
            host_src.bitcast[NoneType](),
            count * sizeof[type](),
        )
    )


fn _copy_host_to_device[
    type: DType
](
    device_dest: DTypePointer[type], host_src: DTypePointer[type], count: Int
) raises:
    _copy_host_to_device[SIMD[type, 1]](
        device_dest._as_scalar_pointer(),
        host_src._as_scalar_pointer(),
        count,
    )


fn _copy_device_to_host[
    type: AnyType
](host_dest: Pointer[type], device_src: Pointer[type], count: Int) raises:
    _check_error(
        _get_dylib_function[
            fn (Pointer[NoneType], Pointer[UInt32], Int) -> Result
        ]("cuMemcpyDtoH_v2")(
            host_dest.bitcast[NoneType](),
            device_src.bitcast[UInt32](),
            count * sizeof[type](),
        )
    )


fn _copy_device_to_host[
    type: DType
](
    host_dest: DTypePointer[type], device_src: DTypePointer[type], count: Int
) raises:
    _copy_device_to_host[SIMD[type, 1]](
        host_dest._as_scalar_pointer(),
        device_src._as_scalar_pointer(),
        count,
    )


fn _memset[
    type: AnyType
](device_dest: Pointer[type], val: UInt8, count: Int) raises:
    _check_error(
        _get_dylib_function[fn (Pointer[UInt32], UInt8, Int) -> Result](
            "cuMemsetD8_v2"
        )(
            device_dest.bitcast[UInt32](),
            val,
            count * sizeof[type](),
        )
    )


fn _memset[
    type: DType
](device_dest: DTypePointer[type], val: UInt8, count: Int) raises:
    _memset[SIMD[type, 1]](
        device_dest._as_scalar_pointer(),
        val,
        count,
    )


fn _memset_async[
    type: DType
](
    device_dest: DTypePointer[type],
    val: SIMD[type, 1],
    count: Int,
    stream: Stream,
) raises:
    alias bitwidth = type.bitwidth()
    constrained[
        bitwidth == 8 or bitwidth == 16 or bitwidth == 32,
        "bitwidth of memset type must be one of [8,16,32]",
    ]()

    @parameter
    if bitwidth == 8:
        _check_error(
            _get_dylib_function[
                fn (
                    DTypePointer[DType.uint8], UInt8, Int, _StreamImpl
                ) -> Result
            ]("cuMemsetD8Async")(
                device_dest.bitcast[DType.uint8](),
                bitcast[DType.uint8, 1](val),
                count * sizeof[type](),
                stream.stream,
            )
        )
    elif bitwidth == 16:
        _check_error(
            _get_dylib_function[
                fn (
                    DTypePointer[DType.uint16], UInt16, Int, _StreamImpl
                ) -> Result
            ]("cuMemsetD16Async")(
                device_dest.bitcast[DType.uint16](),
                bitcast[DType.uint16, 1](val),
                count,
                stream.stream,
            )
        )
    elif bitwidth == 32:
        _check_error(
            _get_dylib_function[
                fn (
                    DTypePointer[DType.uint32], UInt32, Int, _StreamImpl
                ) -> Result
            ]("cuMemsetD32Async")(
                device_dest.bitcast[DType.uint32](),
                bitcast[DType.uint32, 1](val),
                count,
                stream.stream,
            )
        )
