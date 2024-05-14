# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA memory operations."""

from sys.info import sizeof

from memory.unsafe import DTypePointer, Pointer, bitcast

from ._utils import _check_error, _get_dylib_function
from .stream import Stream, _StreamHandle

# ===----------------------------------------------------------------------===#
# Memory
# ===----------------------------------------------------------------------===#


fn _malloc[type: AnyRegType](count: Int) raises -> Pointer[type]:
    """Allocates GPU device memory."""

    var ptr = Pointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAlloc_v2", fn (Pointer[Pointer[Int]], Int) -> Result
        ]()(Pointer.address_of(ptr), count * sizeof[type]())
    )
    return ptr.bitcast[type]()


fn _malloc[type: DType](count: Int) raises -> DTypePointer[type]:
    return _malloc[Scalar[type]](count)


fn _malloc_managed[type: AnyRegType](count: Int) raises -> Pointer[type]:
    """Allocates memory that will be automatically managed by the Unified Memory system.
    """
    alias CU_MEM_ATTACH_GLOBAL = UInt32(1)
    var ptr = Pointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAllocManaged",
            fn (Pointer[Pointer[Int]], Int, UInt32) -> Result,
        ]()(
            Pointer.address_of(ptr),
            count * sizeof[type](),
            CU_MEM_ATTACH_GLOBAL,
        )
    )
    return ptr.bitcast[type]()


fn _malloc_managed[type: DType](count: Int) raises -> DTypePointer[type]:
    return _malloc_managed[Scalar[type]](count)


fn _free[type: AnyRegType](ptr: Pointer[type]) raises:
    """Frees allocated GPU device memory."""

    _check_error(
        _get_dylib_function["cuMemFree_v2", fn (Pointer[Int]) -> Result]()(
            ptr.bitcast[Int]()
        )
    )


fn _free[type: DType](ptr: DTypePointer[type]) raises:
    _free(ptr._as_scalar_pointer())


fn _copy_host_to_device[
    type: AnyRegType
](device_dest: Pointer[type], host_src: Pointer[type], count: Int) raises:
    """Copies memory from host to device."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyHtoD_v2",
            fn (Pointer[Int], Pointer[NoneType], Int) -> Result,
        ]()(
            device_dest.bitcast[Int](),
            host_src.bitcast[NoneType](),
            count * sizeof[type](),
        )
    )


fn _copy_host_to_device[
    type: DType
](
    device_dest: DTypePointer[type], host_src: DTypePointer[type], count: Int
) raises:
    _copy_host_to_device[Scalar[type]](
        device_dest._as_scalar_pointer(),
        host_src._as_scalar_pointer(),
        count,
    )


fn _copy_host_to_device_async[
    type: AnyRegType
](
    device_dst: Pointer[type],
    host_src: Pointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from host to device asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyHtoDAsync_v2",
            fn (Pointer[NoneType], Pointer[Int], Int, _StreamHandle) -> Result,
        ]()(
            device_dst.bitcast[NoneType](),
            host_src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _copy_host_to_device_async[
    type: DType
](
    device_dst: DTypePointer[type],
    host_src: DTypePointer[type],
    count: Int,
    stream: Stream,
) raises:
    _copy_host_to_device_async[Scalar[type]](
        device_dst._as_scalar_pointer(),
        host_src._as_scalar_pointer(),
        count,
        stream,
    )


fn _copy_device_to_host[
    type: AnyRegType
](host_dest: Pointer[type], device_src: Pointer[type], count: Int) raises:
    """Copies memory from device to host."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoH_v2",
            fn (Pointer[NoneType], Pointer[Int], Int) -> Result,
        ]()(
            host_dest.bitcast[NoneType](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
        )
    )


fn _copy_device_to_host[
    type: DType
](
    host_dest: DTypePointer[type], device_src: DTypePointer[type], count: Int
) raises:
    _copy_device_to_host[Scalar[type]](
        host_dest._as_scalar_pointer(),
        device_src._as_scalar_pointer(),
        count,
    )


fn _copy_device_to_host_async[
    type: AnyRegType
](
    host_dest: Pointer[type],
    device_src: Pointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from device to host asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoHAsync_v2",
            fn (Pointer[NoneType], Pointer[Int], Int, _StreamHandle) -> Result,
        ]()(
            host_dest.bitcast[NoneType](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _copy_device_to_host_async[
    type: DType
](
    host_dest: DTypePointer[type],
    device_src: DTypePointer[type],
    count: Int,
    stream: Stream,
) raises:
    _copy_device_to_host_async[Scalar[type]](
        host_dest._as_scalar_pointer(),
        device_src._as_scalar_pointer(),
        count,
        stream,
    )


fn _copy_device_to_device_async[
    type: AnyRegType
](dst: Pointer[type], src: Pointer[type], count: Int, stream: Stream) raises:
    """Copies memory from device to device asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoDAsync_v2",
            fn (Pointer[NoneType], Pointer[Int], Int, _StreamHandle) -> Result,
        ]()(
            dst.bitcast[NoneType](),
            src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _copy_device_to_device_async[
    type: DType
](
    dst: DTypePointer[type], src: DTypePointer[type], count: Int, stream: Stream
) raises:
    return _copy_device_to_device_async(
        dst._as_scalar_pointer(), src._as_scalar_pointer(), count, stream
    )


fn _memset[
    type: AnyRegType
](device_dest: Pointer[type], val: UInt8, count: Int) raises:
    """Sets the memory range of N 8-bit values to a specified value."""

    _check_error(
        _get_dylib_function[
            "cuMemsetD8_v2", fn (Pointer[Int], UInt8, Int) -> Result
        ]()(
            device_dest.bitcast[Int](),
            val,
            count * sizeof[type](),
        )
    )


fn _memset[
    type: DType
](device_dest: DTypePointer[type], val: UInt8, count: Int) raises:
    _memset[Scalar[type]](
        device_dest._as_scalar_pointer(),
        val,
        count,
    )


fn _memset_async[
    type: DType
](
    device_dest: DTypePointer[type],
    val: Scalar[type],
    count: Int,
    stream: Stream,
) raises:
    """Sets the memory range of N 8-bit, 16-bit and 32-bit values to a specified value asynchronously.
    """

    alias bitwidth = type.bitwidth()
    constrained[
        bitwidth == 8 or bitwidth == 16 or bitwidth == 32,
        "bitwidth of memset type must be one of [8,16,32]",
    ]()

    @parameter
    if bitwidth == 8:
        _check_error(
            _get_dylib_function[
                "cuMemsetD8Async",
                fn (
                    DTypePointer[DType.uint8], UInt8, Int, _StreamHandle
                ) -> Result,
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
                    DTypePointer[DType.uint16], UInt16, Int, _StreamHandle
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
                    DTypePointer[DType.uint32], UInt32, Int, _StreamHandle
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
    type: AnyRegType
](device_dest: Pointer[type], device_src: Pointer[type], count: Int,) raises:
    """Copies memory from device to device."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoD_v2",
            fn (
                Pointer[Int],
                Pointer[Int],
                Int,
            ) -> Result,
        ]()(
            device_dest.bitcast[Int](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
        )
    )


@always_inline
fn _copy_device_to_device[
    type: DType
](
    device_dest: DTypePointer[type],
    device_src: DTypePointer[type],
    count: Int,
) raises:
    _copy_device_to_device[Scalar[type]](
        device_dest._as_scalar_pointer(),
        device_src._as_scalar_pointer(),
        count,
    )


fn _malloc_async[
    type: AnyRegType
](count: Int, stream: Stream) raises -> Pointer[type]:
    """Allocates memory with stream ordered semantics."""

    var ptr = Pointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAllocAsync",
            fn (Pointer[Pointer[Int]], Int, _StreamHandle) -> Result,
        ]()(Pointer.address_of(ptr), count * sizeof[type](), stream.stream)
    )
    return ptr.bitcast[type]()


fn _free_async[type: AnyRegType](ptr: Pointer[type], stream: Stream) raises:
    """Frees memory with stream ordered semantics."""

    _check_error(
        _get_dylib_function[
            "cuMemFreeAsync", fn (Pointer[Int], _StreamHandle) -> Result
        ]()(ptr.bitcast[Int](), stream.stream)
    )
