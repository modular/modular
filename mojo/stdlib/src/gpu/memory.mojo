# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs memory operations."""

from math import is_power_of_2
from sys.info import alignof, simdwidthof, sizeof, triple_is_nvidia_cuda

from memory import stack_allocation as _generic_stack_allocation
from memory.unsafe import DTypePointer, Pointer, bitcast
from gpu.host.stream import Stream, _StreamImpl
from gpu.host._utils import _check_error, _get_dylib_function
from gpu.host.result import Result


# ===----------------------------------------------------------------------===#
# Address Space
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct AddressSpace:
    var _value: Int

    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    alias GENERIC = AddressSpace(0)
    """Generic address space."""
    alias GLOBAL = AddressSpace(1)
    """Global address space."""
    alias CONSTANT = AddressSpace(2)
    """Constant address space."""
    alias SHARED = AddressSpace(3)
    """Shared address space."""
    alias PARAM = AddressSpace(4)
    """Param address space."""
    alias LOCAL = AddressSpace(5)
    """Local address space."""

    @always_inline("nodebug")
    fn __init__(value: Int) -> Self:
        return Self {_value: value}

    @always_inline("nodebug")
    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("nodebug")
    fn __eq__(self, other: AddressSpace) -> Bool:
        """The True if the two address spaces are equal and False otherwise.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() == other.value()


# ===----------------------------------------------------------------------===#
# cp.async
# ===----------------------------------------------------------------------===#


@always_inline
fn async_copy[
    size: Int, type: DType
](
    src: DTypePointer[type, AddressSpace.GLOBAL],
    dst: DTypePointer[type, AddressSpace.SHARED],
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        size: Number of bytes to copy.
        type: The pointer type.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
    """
    # TODO: Constrained on device capability.
    constrained[size == 4 or size == 8 or size == 16]()

    @parameter
    if size == 4:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.4", NoneType](
            dst, src
        )
    elif size == 8:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.8", NoneType](
            dst, src
        )
    else:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.16", NoneType](
            dst, src
        )


@always_inline
fn async_copy[
    size: Int, type: AnyRegType
](
    src: Pointer[type, AddressSpace.GLOBAL],
    dst: Pointer[type, AddressSpace.SHARED],
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        size: Number of bytes to copy.
        type: The pointer type.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
    """
    # TODO: Constrained on device capability.
    constrained[size == 4 or size == 8 or size == 16]()

    @parameter
    if size == 4:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.4", NoneType](
            dst, src
        )
    elif size == 8:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.8", NoneType](
            dst, src
        )
    else:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.16", NoneType](
            dst, src
        )


@always_inline
fn async_copy_commit_group():
    """Commits all prior initiated but uncommitted cp.async instructions into
    a cp.async-group.
    """
    llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()


@always_inline
fn async_copy_wait_group(n: Int32):
    """Wait for the completion of `n` or asynchronous copy operations.

    Args:
        n: The number of pending cp.async-groups.
    """
    llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)


@always_inline
fn async_copy_wait_all():
    """Wait for the completion of all commited cp.async-groups."""
    llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()


@always_inline
fn _copy_device_to_device[
    type: AnyRegType
](device_dest: Pointer[type], device_src: Pointer[type], count: Int,) raises:
    _check_error(
        _get_dylib_function[
            fn (
                Pointer[UInt32],
                Pointer[UInt32],
                Int,
            ) -> Result
        ]("cuMemcpyDtoD_v2")(
            device_dest.bitcast[UInt32](),
            device_src.bitcast[UInt32](),
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
    _copy_device_to_device[SIMD[type, 1]](
        device_dest._as_scalar_pointer(),
        device_src._as_scalar_pointer(),
        count,
    )


@always_inline
fn _copy_device_to_device_async[
    type: AnyRegType
](
    device_dest: Pointer[type],
    device_src: Pointer[type],
    count: Int,
    stream: Stream,
) raises:
    _check_error(
        _get_dylib_function[
            fn (Pointer[UInt32], Pointer[UInt32], Int, _StreamImpl) -> Result
        ]("cuMemcpyDtoDAsync_v2")(
            device_dest.bitcast[UInt32](),
            device_src.bitcast[UInt32](),
            count * sizeof[type](),
            stream.stream,
        )
    )


@always_inline
fn _copy_device_to_device_async[
    type: DType
](
    device_dest: DTypePointer[type],
    device_src: DTypePointer[type],
    count: Int,
    stream: Stream,
) raises:
    _copy_device_to_device_async[SIMD[type, 1]](
        device_dest._as_scalar_pointer(),
        device_src._as_scalar_pointer(),
        count,
        stream,
    )


fn _malloc_async[
    type: AnyRegType
](count: Int, stream: Stream) raises -> Pointer[type]:
    var ptr = Pointer[UInt32]()
    _check_error(
        _get_dylib_function[
            fn (Pointer[Pointer[UInt32]], Int, _StreamImpl) -> Result
        ]("cuMemAllocAsync")(
            Pointer.address_of(ptr), count * sizeof[type](), stream.stream
        )
    )
    return ptr.bitcast[type]()


fn _free_async[type: AnyRegType](ptr: Pointer[type], stream: Stream) raises:
    _check_error(
        _get_dylib_function[fn (Pointer[UInt32], _StreamImpl) -> Result](
            "cuMemFreeAsync"
        )(ptr.bitcast[UInt32](), stream.stream)
    )
