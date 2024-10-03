# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA context operations."""

from os import abort
from sys import bitwidthof, sizeof
from sys.ffi import c_size_t

from gpu.host.function import FunctionCache
from memory import UnsafePointer, bitcast

from ._utils import _check_error, _ContextHandle, _StreamHandle
from .cuda_instance import *
from .device import Device

# ===----------------------------------------------------------------------===#
# Context
# ===----------------------------------------------------------------------===#


struct Context:
    var device: Device
    var ctx: _ContextHandle
    var cuda_dll: CudaDLL
    var cuda_function_cache: UnsafePointer[FunctionCache]
    var owner: Bool

    alias USE_CTX_RETAIN = False

    fn __init__(inout self) raises:
        self.__init__(Device())

    fn __init__(inout self, device: Device, flags: Int = 0) raises:
        self.device = device
        self.cuda_dll = device.cuda_dll
        self.cuda_function_cache = UnsafePointer[FunctionCache]().alloc(1)
        self.cuda_function_cache.init_pointee_move(FunctionCache())
        self.ctx = _ContextHandle()
        self.owner = True

        @parameter
        if self.USE_CTX_RETAIN:
            # FIXME: This tends to fail. KERN-841.
            # _check_error(
            #     self.cuda_dll.cuDevicePrimaryCtxSetFlags(device.id, flags)
            # )
            _check_error(
                self.cuda_dll.cuDevicePrimaryCtxRetain(
                    UnsafePointer.address_of(self.ctx), device.id
                )
            )
            _check_error(self.cuda_dll.cuCtxSetCurrent(self.ctx))
        else:
            _check_error(
                self.cuda_dll.cuCtxCreate(
                    UnsafePointer.address_of(self.ctx), flags, device.id
                )
            )

        _check_error(
            self.cuda_dll.cuCtxSetCacheConfig(CacheConfig.PREFER_SHARED)
        )

    fn __del__(owned self):
        try:
            if self.ctx and self.owner:

                @parameter
                if self.USE_CTX_RETAIN:
                    _check_error(
                        self.cuda_dll.cuDevicePrimaryCtxRelease(self.device.id)
                    )
                else:
                    _check_error(self.cuda_dll.cuCtxDestroy(self.ctx))

                self.ctx = _ContextHandle()
                if self.cuda_function_cache:
                    self.cuda_function_cache.destroy_pointee()
                    self.cuda_function_cache.free()
                self.cuda_dll = CudaDLL()
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
        self.owner = existing.owner
        existing.ctx = _ContextHandle()
        existing.cuda_dll = CudaDLL()
        existing.owner = False

    fn __copyinit__(inout self, existing: Self):
        self.device = existing.device
        self.ctx = existing.ctx
        self.cuda_dll = existing.cuda_dll
        self.cuda_function_cache = existing.cuda_function_cache
        self.owner = False

    fn set_current(self) raises:
        """Set the current context for this thread."""
        _check_error(self.cuda_dll.cuCtxSetCurrent(self.ctx))
        pass

    fn synchronize(self) raises:
        """Blocks for a Cuda Context's tasks to complete."""

        _check_error(self.cuda_dll.cuCtxSynchronize())

    fn malloc_host[
        type: AnyType
    ](self, count: Int) raises -> UnsafePointer[type]:
        """Allocates pinned memory on the host registered with the GPU device.
        """

        alias sizeof_t = sizeof[type]()
        var ptr = UnsafePointer[Int]()
        _check_error(
            self.cuda_dll.cuMemAllocHost(
                UnsafePointer.address_of(ptr), count * sizeof_t
            )
        )
        return ptr.bitcast[type]()

    fn malloc[type: AnyType](self, count: Int) raises -> UnsafePointer[type]:
        """Allocates GPU device memory."""

        var ptr = UnsafePointer[Int]()
        _check_error(
            self.cuda_dll.cuMemAlloc(
                UnsafePointer.address_of(ptr), count * sizeof[type]()
            )
        )
        return ptr.bitcast[type]()

    fn malloc_managed[
        type: AnyType
    ](self, count: Int) raises -> UnsafePointer[type]:
        """Allocates memory that will be automatically managed by the Unified Memory system.
        """

        alias CU_MEM_ATTACH_GLOBAL = UInt32(1)
        var ptr = UnsafePointer[Int]()
        _check_error(
            self.cuda_dll.cuMemAllocManaged(
                UnsafePointer.address_of(ptr),
                count * sizeof[type](),
                CU_MEM_ATTACH_GLOBAL,
            )
        )
        return ptr.bitcast[type]()

    fn free_host[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        """Frees memory allocated with malloc_host()."""
        _check_error(self.cuda_dll.cuMemFreeHost(ptr.bitcast[Int]()))

    fn free[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        """Frees allocated GPU device memory."""
        _check_error(self.cuda_dll.cuMemFree(ptr.bitcast[Int]()))

    fn copy_host_to_device[
        type: AnyType
    ](
        self,
        device_dest: UnsafePointer[type],
        host_src: UnsafePointer[type],
        count: Int,
    ) raises:
        """Copies memory from host to device."""

        _check_error(
            self.cuda_dll.cuMemcpyHtoD(
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

        _check_error(
            self.cuda_dll.cuMemcpyHtoDAsync(
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

        _check_error(
            self.cuda_dll.cuMemcpyDtoH(
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

        _check_error(
            self.cuda_dll.cuMemcpyDtoHAsync(
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

        _check_error(
            self.cuda_dll.cuMemcpyDtoDAsync(
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

        _check_error(
            self.cuda_dll.cuMemsetD8(
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
            bitwidth in (8, 16, 32),
            "bitwidth of memset type must be one of [8,16,32]",
        ]()

        @parameter
        if bitwidth == 8:
            _check_error(
                self.cuda_dll.cuMemsetD8Async(
                    device_dest.bitcast[DType.uint8](),
                    bitcast[DType.uint8, 1](val),
                    count * sizeof[type](),
                    stream.stream,
                )
            )
        elif bitwidth == 16:
            _check_error(
                self.cuda_dll.cuMemsetD16Async(
                    device_dest.bitcast[DType.uint16](),
                    bitcast[DType.uint16, 1](val),
                    count,
                    stream.stream,
                )
            )
        elif bitwidth == 32:
            _check_error(
                self.cuda_dll.cuMemsetD32Async(
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

        _check_error(
            self.cuda_dll.cuMemcpyDtoD(
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
        _check_error(
            self.cuda_dll.cuMemAllocAsync(
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

        _check_error(
            self.cuda_dll.cuMemFreeAsync(ptr.bitcast[Int](), stream.stream)
        )

    fn get_compute_capability(self) raises -> Float64:
        """Returns the device compute capability version."""

        return self.device.compute_capability()

    fn get_version(self) raises -> Float64:
        """Returns the cuda version."""
        var cuda_version = self.device.cuda_version()
        return cuda_version[0] + Float64(cuda_version[1]) / 1000

    fn get_memory_info(self) raises -> (c_size_t, c_size_t):
        var free = c_size_t(0)
        var total = c_size_t(0)

        _check_error(
            self.cuda_dll.cuMemGetInfo(
                UnsafePointer.address_of(free),
                UnsafePointer.address_of(total),
            )
        )

        return (free, total)
