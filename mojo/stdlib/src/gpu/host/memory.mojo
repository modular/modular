# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA memory operations."""

from sys.info import sizeof

from memory import UnsafePointer
from memory.unsafe import bitcast

from ._utils import _check_error, _get_dylib_function
from .stream import Stream, _StreamHandle

# ===----------------------------------------------------------------------===#
# Memory
# ===----------------------------------------------------------------------===#


fn _malloc[type: AnyType](count: Int) raises -> UnsafePointer[type]:
    """Allocates GPU device memory."""

    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAlloc_v2",
            fn (UnsafePointer[UnsafePointer[Int]], Int) -> Result,
        ]()(UnsafePointer.address_of(ptr), count * sizeof[type]())
    )
    return ptr.bitcast[type]()


fn _malloc[type: DType](count: Int) raises -> UnsafePointer[Scalar[type]]:
    return _malloc[Scalar[type]](count)


fn _malloc_managed[type: AnyType](count: Int) raises -> UnsafePointer[type]:
    """Allocates memory that will be automatically managed by the Unified Memory system.
    """
    alias CU_MEM_ATTACH_GLOBAL = UInt32(1)
    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAllocManaged",
            fn (UnsafePointer[UnsafePointer[Int]], Int, UInt32) -> Result,
        ]()(
            UnsafePointer.address_of(ptr),
            count * sizeof[type](),
            CU_MEM_ATTACH_GLOBAL,
        )
    )
    return ptr.bitcast[type]()


fn _malloc_managed[
    type: DType
](count: Int) raises -> UnsafePointer[Scalar[type]]:
    return _malloc_managed[Scalar[type]](count)


fn _free[type: AnyType](ptr: UnsafePointer[type]) raises:
    """Frees allocated GPU device memory."""

    _check_error(
        _get_dylib_function[
            "cuMemFree_v2", fn (UnsafePointer[Int]) -> Result
        ]()(ptr.bitcast[Int]())
    )


fn _copy_host_to_device[
    type: AnyType
](
    device_dest: UnsafePointer[type], host_src: UnsafePointer[type], count: Int
) raises:
    """Copies memory from host to device."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyHtoD_v2",
            fn (UnsafePointer[Int], UnsafePointer[NoneType], Int) -> Result,
        ]()(
            device_dest.bitcast[Int](),
            host_src.bitcast[NoneType](),
            count * sizeof[type](),
        )
    )


fn _copy_host_to_device_async[
    type: AnyType
](
    device_dst: UnsafePointer[type],
    host_src: UnsafePointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from host to device asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyHtoDAsync_v2",
            fn (
                UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
            ) -> Result,
        ]()(
            device_dst.bitcast[NoneType](),
            host_src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _copy_device_to_host[
    type: AnyType
](
    host_dest: UnsafePointer[type], device_src: UnsafePointer[type], count: Int
) raises:
    """Copies memory from device to host."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoH_v2",
            fn (UnsafePointer[NoneType], UnsafePointer[Int], Int) -> Result,
        ]()(
            host_dest.bitcast[NoneType](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
        )
    )


fn _copy_device_to_host_async[
    type: AnyType
](
    host_dest: UnsafePointer[type],
    device_src: UnsafePointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from device to host asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoHAsync_v2",
            fn (
                UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
            ) -> Result,
        ]()(
            host_dest.bitcast[NoneType](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _copy_device_to_device_async[
    type: AnyType
](
    dst: UnsafePointer[type],
    src: UnsafePointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from device to device asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoDAsync_v2",
            fn (
                UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
            ) -> Result,
        ]()(
            dst.bitcast[NoneType](),
            src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _memset[
    type: AnyType
](device_dest: UnsafePointer[type], val: UInt8, count: Int) raises:
    """Sets the memory range of N 8-bit values to a specified value."""

    _check_error(
        _get_dylib_function[
            "cuMemsetD8_v2", fn (UnsafePointer[Int], UInt8, Int) -> Result
        ]()(
            device_dest.bitcast[Int](),
            val,
            count * sizeof[type](),
        )
    )


fn _memset_async[
    type: DType
](
    device_dest: UnsafePointer[Scalar[type]],
    val: Scalar[type],
    count: Int,
    stream: Stream,
) raises:
    """Sets the memory range of N 8-bit, 16-bit and 32-bit values to a specified value asynchronously.
    """

    alias bitwidth = bitwidthof[type]()
    constrained[
        bitwidth == 8 or bitwidth == 16 or bitwidth == 32,
        "bitwidth of memset type must be one of [8,16,32]",
    ]()

    @parameter
    if bitwidth == 8:
        _check_error(
            _get_dylib_function[
                "cuMemsetD8Async",
                fn (UnsafePointer[UInt8], UInt8, Int, _StreamHandle) -> Result,
            ]()(
                device_dest.bitcast[DType.uint8](),
                bitcast[DType.uint8, 1](val),
                count * sizeof[type](),
                stream.stream,
            )
        )
    elif bitwidth == 16:
        _check_error(
            _get_dylib_function[
                "cuMemsetD16Async",
                fn (
                    UnsafePointer[UInt16], UInt16, Int, _StreamHandle
                ) -> Result,
            ]()(
                device_dest.bitcast[DType.uint16](),
                bitcast[DType.uint16, 1](val),
                count,
                stream.stream,
            )
        )
    elif bitwidth == 32:
        _check_error(
            _get_dylib_function[
                "cuMemsetD32Async",
                fn (
                    UnsafePointer[UInt32], UInt32, Int, _StreamHandle
                ) -> Result,
            ]()(
                device_dest.bitcast[DType.uint32](),
                bitcast[DType.uint32, 1](val),
                count,
                stream.stream,
            )
        )


@always_inline
fn _copy_device_to_device[
    type: AnyType
](
    device_dest: UnsafePointer[type],
    device_src: UnsafePointer[type],
    count: Int,
) raises:
    """Copies memory from device to device."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoD_v2",
            fn (
                UnsafePointer[Int],
                UnsafePointer[Int],
                Int,
            ) -> Result,
        ]()(
            device_dest.bitcast[Int](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
        )
    )


fn _malloc_async[
    type: AnyType
](count: Int, stream: Stream) raises -> UnsafePointer[type]:
    """Allocates memory with stream ordered semantics."""

    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAllocAsync",
            fn (
                UnsafePointer[UnsafePointer[Int]], Int, _StreamHandle
            ) -> Result,
        ]()(
            UnsafePointer.address_of(ptr), count * sizeof[type](), stream.stream
        )
    )
    return ptr.bitcast[type]()


fn _free_async[type: AnyType](ptr: UnsafePointer[type], stream: Stream) raises:
    """Frees memory with stream ordered semantics."""

    _check_error(
        _get_dylib_function[
            "cuMemFreeAsync", fn (UnsafePointer[Int], _StreamHandle) -> Result
        ]()(ptr.bitcast[Int](), stream.stream)
    )
