# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA context operations."""

from os import abort

from memory import UnsafePointer

from ._utils import _check_error, _ContextHandle, _StreamHandle
from .cuda_instance import *
from .device import Device
from gpu.host.function import FunctionCache

# ===----------------------------------------------------------------------===#
# Context
# ===----------------------------------------------------------------------===#


struct Context:
    var device: Device
    var ctx: _ContextHandle
    var cuda_dll: Optional[CudaDLL]
    var cuda_function_cache: UnsafePointer[FunctionCache]
    var owner: Bool

    fn __init__(inout self) raises:
        self.__init__(Device())

    fn __init__(inout self, device: Device, flags: Int = 0) raises:
        self.device = device
        self.cuda_dll = device.cuda_dll
        self.cuda_function_cache = UnsafePointer[FunctionCache]().alloc(1)
        self.cuda_function_cache.init_pointee_move(FunctionCache())
        self.ctx = _ContextHandle()
        self.owner = True

        var cuCtxCreate = self.cuda_dll.value().cuCtxCreate if self.cuda_dll else cuCtxCreate.load()
        _check_error(
            cuCtxCreate(UnsafePointer.address_of(self.ctx), flags, device.id)
        )

    fn __del__(owned self):
        try:
            if self.ctx and self.owner:
                var cuCtxDestroy = self.cuda_dll.value().cuCtxDestroy if self.cuda_dll else cuCtxDestroy.load()
                _check_error(cuCtxDestroy(self.ctx))
                self.ctx = _ContextHandle()
                if self.cuda_function_cache:
                    self.cuda_function_cache.destroy_pointee()
                    self.cuda_function_cache.free()
                self.cuda_dll = None
                self.owner = False
        except e:
            abort(str(e))

    fn __enter__(owned self) -> Self:
        return self^

    fn __moveinit__(inout self, owned existing: Self):
        self.device = existing.device
        self.ctx = existing.ctx
        self.cuda_dll = existing.cuda_dll
        self.cuda_function_cache = existing.cuda_function_cache
        self.owner = True
        existing.ctx = _ContextHandle()
        existing.cuda_dll = None
        existing.owner = False

    fn __copyinit__(inout self, existing: Self):
        self.device = existing.device
        self.ctx = existing.ctx
        self.cuda_dll = existing.cuda_dll
        self.cuda_function_cache = existing.cuda_function_cache
        self.owner = False

    fn synchronize(self) raises:
        """Blocks for a Cuda Context's tasks to complete."""
        var cuCtxSynchronize = self.cuda_dll.value().cuCtxSynchronize if self.cuda_dll else cuCtxSynchronize.load()
        _check_error(cuCtxSynchronize())

    fn malloc[type: AnyType](self, count: Int) raises -> UnsafePointer[type]:
        """Allocates GPU device memory."""

        var ptr = UnsafePointer[Int]()
        var cuMemAlloc = self.cuda_dll.value().cuMemAlloc if self.cuda_dll else cuMemAlloc.load()
        _check_error(
            cuMemAlloc(UnsafePointer.address_of(ptr), count * sizeof[type]())
        )
        return ptr.bitcast[type]()

    fn malloc_managed[
        type: AnyType
    ](self, count: Int) raises -> UnsafePointer[type]:
        """Allocates memory that will be automatically managed by the Unified Memory system.
        """
        alias CU_MEM_ATTACH_GLOBAL = UInt32(1)
        var ptr = UnsafePointer[Int]()
        var cuMemAllocManaged = self.cuda_dll.value().cuMemAllocManaged if self.cuda_dll else cuMemAllocManaged.load()
        _check_error(
            cuMemAllocManaged(
                UnsafePointer.address_of(ptr),
                count * sizeof[type](),
                CU_MEM_ATTACH_GLOBAL,
            )
        )
        return ptr.bitcast[type]()

    fn free[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        """Frees allocated GPU device memory."""

        var cuMemFree = self.cuda_dll.value().cuMemFree if self.cuda_dll else cuMemFree.load()
        _check_error(cuMemFree(ptr.bitcast[Int]()))

    fn copy_host_to_device[
        type: AnyType
    ](
        self,
        device_dest: UnsafePointer[type],
        host_src: UnsafePointer[type],
        count: Int,
    ) raises:
        """Copies memory from host to device."""

        var cuMemcpyHtoD = self.cuda_dll.value().cuMemcpyHtoD if self.cuda_dll else cuMemcpyHtoD.load()
        _check_error(
            cuMemcpyHtoD(
                device_dest.bitcast[Int](),
                host_src.bitcast[NoneType](),
                count * sizeof[type](),
            )
        )

    fn copy_host_to_device_async[
        type: AnyType
    ](
        self,
        device_dst: UnsafePointer[type],
        host_src: UnsafePointer[type],
        count: Int,
        stream: Stream,
    ) raises:
        """Copies memory from host to device asynchronously."""

        var cuMemcpyHtoDAsync = self.cuda_dll.value().cuMemcpyHtoDAsync if self.cuda_dll else cuMemcpyHtoDAsync.load()
        _check_error(
            cuMemcpyHtoDAsync(
                device_dst.bitcast[NoneType](),
                host_src.bitcast[Int](),
                count * sizeof[type](),
                stream.stream,
            )
        )

    fn copy_device_to_host[
        type: AnyType
    ](
        self,
        host_dest: UnsafePointer[type],
        device_src: UnsafePointer[type],
        count: Int,
    ) raises:
        """Copies memory from device to host."""

        var cuMemcpyDtoH = self.cuda_dll.value().cuMemcpyDtoH if self.cuda_dll else cuMemcpyDtoH.load()
        _check_error(
            cuMemcpyDtoH(
                host_dest.bitcast[NoneType](),
                device_src.bitcast[Int](),
                count * sizeof[type](),
            )
        )

    fn copy_device_to_host_async[
        type: AnyType
    ](
        self,
        host_dest: UnsafePointer[type],
        device_src: UnsafePointer[type],
        count: Int,
        stream: Stream,
    ) raises:
        """Copies memory from device to host asynchronously."""

        var cuMemcpyDtoHAsync = self.cuda_dll.value().cuMemcpyDtoHAsync if self.cuda_dll else cuMemcpyDtoHAsync.load()
        _check_error(
            cuMemcpyDtoHAsync(
                host_dest.bitcast[NoneType](),
                device_src.bitcast[Int](),
                count * sizeof[type](),
                stream.stream,
            )
        )

    fn copy_device_to_device_async[
        type: AnyType
    ](
        self,
        dst: UnsafePointer[type],
        src: UnsafePointer[type],
        count: Int,
        stream: Stream,
    ) raises:
        """Copies memory from device to device asynchronously."""

        var cuMemcpyDtoDAsync = self.cuda_dll.value().cuMemcpyDtoDAsync if self.cuda_dll else cuMemcpyDtoDAsync.load()
        _check_error(
            cuMemcpyDtoDAsync(
                dst.bitcast[NoneType](),
                src.bitcast[Int](),
                count * sizeof[type](),
                stream.stream,
            )
        )

    fn memset[
        type: AnyType
    ](self, device_dest: UnsafePointer[type], val: UInt8, count: Int) raises:
        """Sets the memory range of N 8-bit values to a specified value."""

        var cuMemsetD8 = self.cuda_dll.value().cuMemsetD8 if self.cuda_dll else cuMemsetD8.load()
        _check_error(
            cuMemsetD8(
                device_dest.bitcast[Int](),
                val,
                count * sizeof[type](),
            )
        )

    fn memset_async[
        type: DType
    ](
        self,
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
            var cuMemsetD8Async = self.cuda_dll.value().cuMemsetD8Async if self.cuda_dll else cuMemsetD8Async.load()
            _check_error(
                cuMemsetD8Async(
                    device_dest.bitcast[DType.uint8](),
                    bitcast[DType.uint8, 1](val),
                    count * sizeof[type](),
                    stream.stream,
                )
            )
        elif bitwidth == 16:
            var cuMemsetD16Async = self.cuda_dll.value().cuMemsetD16Async if self.cuda_dll else cuMemsetD16Async.load()
            _check_error(
                cuMemsetD16Async(
                    device_dest.bitcast[DType.uint16](),
                    bitcast[DType.uint16, 1](val),
                    count,
                    stream.stream,
                )
            )
        elif bitwidth == 32:
            var cuMemsetD32Async = self.cuda_dll.value().cuMemsetD32Async if self.cuda_dll else cuMemsetD32Async.load()
            _check_error(
                cuMemsetD32Async(
                    device_dest.bitcast[DType.uint32](),
                    bitcast[DType.uint32, 1](val),
                    count,
                    stream.stream,
                )
            )

    @always_inline
    fn copy_device_to_device[
        type: AnyType
    ](
        self,
        device_dest: UnsafePointer[type],
        device_src: UnsafePointer[type],
        count: Int,
    ) raises:
        """Copies memory from device to device."""

        var cuMemcpyDtoD = self.cuda_dll.value().cuMemcpyDtoD if self.cuda_dll else cuMemcpyDtoD.load()
        _check_error(
            cuMemcpyDtoD(
                device_dest.bitcast[Int](),
                device_src.bitcast[Int](),
                count * sizeof[type](),
            )
        )

    fn malloc_async[
        type: AnyType
    ](self, count: Int, stream: Stream) raises -> UnsafePointer[type]:
        """Allocates memory with stream ordered semantics."""

        var ptr = UnsafePointer[Int]()
        var cuMemAllocAsync = self.cuda_dll.value().cuMemAllocAsync if self.cuda_dll else cuMemAllocAsync.load()
        _check_error(
            cuMemAllocAsync(
                UnsafePointer.address_of(ptr),
                count * sizeof[type](),
                stream.stream,
            )
        )
        return ptr.bitcast[type]()

    fn free_async[
        type: AnyType
    ](self, ptr: UnsafePointer[type], stream: Stream) raises:
        """Frees memory with stream ordered semantics."""

        var cuMemFreeAsync = self.cuda_dll.value().cuMemFreeAsync if self.cuda_dll else cuMemFreeAsync.load()
        _check_error(cuMemFreeAsync(ptr.bitcast[Int](), stream.stream))

    fn get_compute_capability(self) raises -> Float64:
        """Returns the device compute capability version."""
        return self.device.compute_capability()
