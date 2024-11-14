# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import c_char, c_size_t

from utils.static_tuple import StaticTuple

from gpu.host.device_attribute import DeviceAttribute
from gpu.host.event_v1 import Flag
from gpu.host.function_v1 import CacheConfig, _FunctionHandle
from gpu.host.module_v1 import JitOptions
from gpu.host.result_v1 import Result
from gpu.host._utils_v1 import (
    _check_error,
    _ContextHandle,
    _EventHandle,
    _get_dylib_function,
    _ModuleHandle,
    _StreamHandle,
)


@register_passable("trivial")
struct _dylib_function[fn_name: StringLiteral, type: AnyTrivialRegType]:
    @staticmethod
    fn load() -> type:
        return _get_dylib_function[fn_name, type]()


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

alias _DeviceHandle = Int32

alias cuDeviceGetCount = _dylib_function[
    "cuDeviceGetCount", fn (UnsafePointer[Int32]) -> Result
]

alias cuDeviceGetAttribute = _dylib_function[
    "cuDeviceGetAttribute",
    fn (UnsafePointer[Int32], DeviceAttribute, _DeviceHandle) -> Result,
]

alias cuDriverGetVersion = _dylib_function[
    "cuDriverGetVersion", fn (UnsafePointer[Int32]) -> Result
]

alias cuDeviceGetName = _dylib_function[
    "cuDeviceGetName",
    fn (UnsafePointer[Int8], Int32, _DeviceHandle) -> Result,
]

alias cuDeviceTotalMem = _dylib_function[
    "cuDeviceTotalMem_v2", fn (UnsafePointer[Int], _DeviceHandle) -> Result
]

alias cuDevicePrimaryCtxRetain = _dylib_function[
    "cuDevicePrimaryCtxRetain",
    fn (UnsafePointer[_ContextHandle], _DeviceHandle) -> Result,
]

alias cuDevicePrimaryCtxSetFlags = _dylib_function[
    "cuDevicePrimaryCtxSetFlags",
    fn (_DeviceHandle, Int32) -> Result,
]

alias cuDevicePrimaryCtxRelease = _dylib_function[
    "cuDevicePrimaryCtxRelease", fn (_DeviceHandle) -> Result
]

alias cuDevicePrimaryCtxReset = _dylib_function[
    "cuDevicePrimaryCtxReset", fn (_DeviceHandle) -> Result
]

alias cuCtxCreate = _dylib_function[
    "cuCtxCreate_v2",
    fn (UnsafePointer[_ContextHandle], Int32, _DeviceHandle) -> Result,
]

alias cuCtxPushCurrent = _dylib_function[
    "cuCtxPushCurrent_v2",
    fn (_ContextHandle) -> Result,
]

alias cuCtxGetCurrent = _dylib_function[
    "cuCtxGetCurrent",
    fn (UnsafePointer[_ContextHandle]) -> Result,
]

alias cuCtxDestroy = _dylib_function[
    "cuCtxDestroy_v2", fn (_ContextHandle) -> Result
]

alias cuCtxSetCurrent = _dylib_function[
    "cuCtxSetCurrent", fn (_ContextHandle) -> Result
]

alias cuCtxSynchronize = _dylib_function["cuCtxSynchronize", fn () -> Result]

alias cuEventCreate = _dylib_function[
    "cuEventCreate", fn (UnsafePointer[_EventHandle], Flag) -> Result
]

alias cuEventDestroy = _dylib_function[
    "cuEventDestroy_v2", fn (_EventHandle) -> Result
]

alias cuEventSynchronize = _dylib_function[
    "cuEventSynchronize", fn (_EventHandle) -> Result
]

alias cuEventRecord = _dylib_function[
    "cuEventRecord", fn (_EventHandle, _StreamHandle) -> Result
]

alias cuEventElapsedTime = _dylib_function[
    "cuEventElapsedTime",
    fn (UnsafePointer[Float32], _EventHandle, _EventHandle) -> Result,
]

alias cuStreamCreate = _dylib_function[
    "cuStreamCreate", fn (UnsafePointer[_StreamHandle], Int32) -> Result
]

alias cuStreamDestroy = _dylib_function[
    "cuStreamDestroy", fn (_StreamHandle) -> Result
]

alias cuStreamSynchronize = _dylib_function[
    "cuStreamSynchronize", fn (_StreamHandle) -> Result
]

alias cuMemAllocHost = _dylib_function[
    "cuMemAllocHost_v2", fn (UnsafePointer[UnsafePointer[Int]], Int) -> Result
]

alias cuMemAlloc = _dylib_function[
    "cuMemAlloc_v2", fn (UnsafePointer[UnsafePointer[Int]], Int) -> Result
]

alias cuMemAllocAsync = _dylib_function[
    "cuMemAllocAsync",
    fn (UnsafePointer[UnsafePointer[Int]], Int, _StreamHandle) -> Result,
]

alias cuMemAllocManaged = _dylib_function[
    "cuMemAllocManaged",
    fn (UnsafePointer[UnsafePointer[Int]], Int, UInt32) -> Result,
]

alias cuMemFreeHost = _dylib_function[
    "cuMemFreeHost", fn (UnsafePointer[Int]) -> Result
]

alias cuMemFree = _dylib_function[
    "cuMemFree_v2", fn (UnsafePointer[Int]) -> Result
]

alias cuMemFreeAsync = _dylib_function[
    "cuMemFreeAsync", fn (UnsafePointer[Int], _StreamHandle) -> Result
]

alias cuMemcpyHtoD = _dylib_function[
    "cuMemcpyHtoD_v2",
    fn (UnsafePointer[Int], UnsafePointer[NoneType], Int) -> Result,
]

alias cuMemcpyHtoDAsync = _dylib_function[
    "cuMemcpyHtoDAsync_v2",
    fn (
        UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
    ) -> Result,
]

alias cuMemcpyDtoH = _dylib_function[
    "cuMemcpyDtoH_v2",
    fn (UnsafePointer[NoneType], UnsafePointer[Int], Int) -> Result,
]

alias cuMemcpyDtoHAsync = _dylib_function[
    "cuMemcpyDtoHAsync_v2",
    fn (
        UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
    ) -> Result,
]

alias cuMemcpyDtoDAsync = _dylib_function[
    "cuMemcpyDtoDAsync_v2",
    fn (
        UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
    ) -> Result,
]

alias cuMemcpyDtoD = _dylib_function[
    "cuMemcpyDtoD_v2",
    fn (UnsafePointer[Int], UnsafePointer[Int], Int) -> Result,
]

alias cuMemsetD8 = _dylib_function[
    "cuMemsetD8_v2", fn (UnsafePointer[Int], UInt8, Int) -> Result
]

alias cuMemsetD8Async = _dylib_function[
    "cuMemsetD8Async",
    fn (UnsafePointer[UInt8], UInt8, Int, _StreamHandle) -> Result,
]

alias cuMemsetD16Async = _dylib_function[
    "cuMemsetD16Async",
    fn (UnsafePointer[UInt16], UInt16, Int, _StreamHandle) -> Result,
]

alias cuMemsetD32Async = _dylib_function[
    "cuMemsetD32Async",
    fn (UnsafePointer[UInt32], UInt32, Int, _StreamHandle) -> Result,
]

alias cuLaunchKernelEx = _dylib_function[
    "cuLaunchKernelEx",
    fn (
        UnsafePointer[LaunchConfig],
        _FunctionHandle,
        UnsafePointer[UnsafePointer[NoneType]],  # Args
        UnsafePointer[NoneType],  # Extra
    ) -> Result,
]

alias cuFuncSetCacheConfig = _dylib_function[
    "cuFuncSetCacheConfig",
    fn (_FunctionHandle, Int32) -> Result,
]

alias cuFuncSetAttribute = _dylib_function[
    "cuFuncSetAttribute",
    fn (_FunctionHandle, Int32, Int32) -> Result,
]

alias cuFuncGetAttribute = _dylib_function[
    "cuFuncGetAttribute",
    fn (UnsafePointer[Int32], Int32, _FunctionHandle) -> Result,
]

alias cuModuleLoad = _dylib_function[
    "cuModuleLoad",
    fn (UnsafePointer[_ModuleHandle], UnsafePointer[c_char]) -> Result,
]

alias cuModuleLoadData = _dylib_function[
    "cuModuleLoadData",
    fn (UnsafePointer[_ModuleHandle], UnsafePointer[UInt8]) -> Result,
]

alias cuModuleLoadDataEx = _dylib_function[
    "cuModuleLoadDataEx",
    fn (
        UnsafePointer[_ModuleHandle],
        UnsafePointer[UInt8],
        UInt32,
        UnsafePointer[JitOptions],
        UnsafePointer[Int],
    ) -> Result,
]

alias cuModuleUnload = _dylib_function[
    "cuModuleUnload", fn (_ModuleHandle) -> Result
]

alias cuModuleGetFunction = _dylib_function[
    "cuModuleGetFunction",
    fn (
        UnsafePointer[_FunctionHandle],
        _ModuleHandle,
        UnsafePointer[c_char],
    ) -> Result,
]

alias cuCtxGetLimit = _dylib_function[
    "cuCtxGetLimit",
    fn (UnsafePointer[Int], LimitProperty) -> Result,
]

alias cuCtxSetLimit = _dylib_function[
    "cuCtxSetLimit",
    fn (LimitProperty, Int) -> Result,
]

alias cuCtxSetCacheConfig = _dylib_function[
    "cuCtxSetCacheConfig",
    fn (CacheConfig) -> Result,
]

alias cuCtxResetPersistingL2Cache = _dylib_function[
    "cuCtxResetPersistingL2Cache",
    fn () -> Result,
]

alias cuModuleGetGlobal = _dylib_function[
    "cuModuleGetGlobal_v2",
    fn (
        UnsafePointer[UnsafePointer[NoneType]],
        UnsafePointer[Int],
        _ModuleHandle,
        UnsafePointer[c_char],
    ) -> Result,
]

alias cuMemGetInfo = _dylib_function[
    "cuMemGetInfo_v2",
    fn (UnsafePointer[c_size_t], UnsafePointer[c_size_t]) -> Result,
]

# ===----------------------------------------------------------------------===#
# LimitProperty
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct LimitProperty:
    var _value: Int32

    alias STACK_SIZE = 0x00
    """Controls the stack size in bytes of each GPU thread. The driver
    automatically increases the per-thread stack size for each kernel launch as
    needed. This size isn't reset back to the original value after each launch.
    Setting this value will take effect immediately, and if necessary, the device
  w ill block until all preceding requested tasks are complete."""

    alias PRINTF_FIFO_SIZE = 0x01
    """Controls the size in bytes of the FIFO used by the printf() device
    system call. Setting CU_LIMIT_PRINTF_FIFO_SIZE must be performed before
    launching any kernel that uses the printf() device system call, otherwise
    CUDA_ERROR_INVALID_VALUE will be returned."""

    alias MALLOC_HEAP_SIZE = 0x02
    """Controls the size in bytes of the heap used by the malloc() and free()
    device system calls. Setting CU_LIMIT_MALLOC_HEAP_SIZE must be performed
    before launching any kernel that uses the malloc() or free() device system
    calls, otherwise CUDA_ERROR_INVALID_VALUE will be returned."""

    alias DEV_RUNTIME_SYNC_DEPTH = 0x03
    """GPU device runtime launch synchronize depth."""

    alias DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04
    """GPU device runtime pending launch count."""

    alias MAX_L2_FETCH_GRANULARITY = 0x05
    """A value between 0 and 128 that indicates the maximum fetch granularity
    of L2 (in Bytes). This is a hint."""

    alias PERSISTING_L2_CACHE_SIZE = 0x06
    """A size in bytes for L2 persisting lines cache size."""

    alias SHMEM_SIZE = 0x07
    """A maximum size in bytes of shared memory available to CUDA kernels on a
    CIG context. Can only be queried, cannot be set."""

    alias CIG_ENABLED = 0x08
    """A non-zero value indicates this CUDA context is a CIG-enabled context.
    Can only be queried, cannot be set."""

    alias CIG_SHMEM_FALLBACK_ENABLED = 0x09
    """When set to a non-zero value, CUDA will fail to launch a kernel on a
    CIG context, instead of using the fallback path, if the kernel uses more
    shared memory than available."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, inout writer: W):
        return writer.write(self._value)


# ===----------------------------------------------------------------------===#
# Launch Config
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct LaunchConfig:
    var grid_dim_x: UInt32
    """Width of grid in blocks."""
    var grid_dim_y: UInt32
    """Height of grid in blocks."""
    var grid_dim_z: UInt32
    """Depth of grid in blocks."""
    var block_dim_x: UInt32
    """X dimension of each thread block."""
    var block_dim_y: UInt32
    """Y dimension of each thread block."""
    var block_dim_z: UInt32
    """Z dimension of each thread block."""
    var shared_mem_bytes: UInt32
    """Dynamic shared-memory size per thread block in bytes."""
    var stream: _StreamHandle
    """Stream identifier."""
    var attrs: UnsafePointer[LaunchAttribute]
    """List of attributes; nullable if num_attrs == 0."""
    var num_attrs: UInt32
    """Number of attributes populated in attrs."""

    @always_inline
    fn __init__(
        inout self,
        *,
        grid_dim_x: UInt32,
        block_dim_x: UInt32,
        block_dim_y: UInt32 = 1,
        block_dim_z: UInt32 = 1,
        grid_dim_y: UInt32 = 1,
        grid_dim_z: UInt32 = 1,
        shared_mem_bytes: UInt32 = 0,
        stream: _StreamHandle = _StreamHandle(),
        attrs: UnsafePointer[LaunchAttribute] = UnsafePointer[
            LaunchAttribute
        ](),
        num_attrs: UInt32 = 0,
    ):
        self.grid_dim_x = grid_dim_x
        self.grid_dim_y = grid_dim_y
        self.grid_dim_z = grid_dim_z
        self.block_dim_x = block_dim_x
        self.block_dim_y = block_dim_y
        self.block_dim_z = block_dim_z
        self.shared_mem_bytes = shared_mem_bytes
        self.stream = stream
        self.attrs = attrs
        self.num_attrs = num_attrs


# ===----------------------------------------------------------------------=== #
# CudaDLL
# ===----------------------------------------------------------------------=== #


@value
struct CudaDLL:
    # cuDevice
    var cuDeviceGetCount: cuDeviceGetCount.type
    var cuDeviceGetAttribute: cuDeviceGetAttribute.type
    var cuDriverGetVersion: cuDriverGetVersion.type
    var cuDeviceGetName: cuDeviceGetName.type
    var cuDeviceTotalMem: cuDeviceTotalMem.type

    # cuDevicePrimaryCtx
    var cuDevicePrimaryCtxRetain: cuDevicePrimaryCtxRetain.type
    var cuDevicePrimaryCtxSetFlags: cuDevicePrimaryCtxSetFlags.type
    var cuDevicePrimaryCtxRelease: cuDevicePrimaryCtxRelease.type
    var cuDevicePrimaryCtxReset: cuDevicePrimaryCtxReset.type

    # cuCtx
    var cuCtxCreate: cuCtxCreate.type
    var cuCtxPushCurrent: cuCtxPushCurrent.type
    var cuCtxGetCurrent: cuCtxGetCurrent.type
    var cuCtxDestroy: cuCtxDestroy.type
    var cuCtxSynchronize: cuCtxSynchronize.type
    var cuCtxSetCurrent: cuCtxSetCurrent.type
    var cuCtxSetCacheConfig: cuCtxSetCacheConfig.type
    var cuCtxResetPersistingL2Cache: cuCtxResetPersistingL2Cache.type

    # cuEvent
    var cuEventCreate: cuEventCreate.type
    var cuEventDestroy: cuEventDestroy.type
    var cuEventSynchronize: cuEventSynchronize.type
    var cuEventRecord: cuEventRecord.type
    var cuEventElapsedTime: cuEventElapsedTime.type

    # cuStream
    var cuStreamCreate: cuStreamCreate.type
    var cuStreamDestroy: cuStreamDestroy.type
    var cuStreamSynchronize: cuStreamSynchronize.type

    # cuMalloc
    var cuMemAllocHost: cuMemAllocHost.type
    var cuMemAlloc: cuMemAlloc.type
    var cuMemAllocAsync: cuMemAllocAsync.type
    var cuMemAllocManaged: cuMemAllocManaged.type
    var cuMemFreeHost: cuMemFreeHost.type
    var cuMemFree: cuMemFree.type
    var cuMemFreeAsync: cuMemFreeAsync.type

    # cuMemcpy
    var cuMemcpyHtoD: cuMemcpyHtoD.type
    var cuMemcpyHtoDAsync: cuMemcpyHtoDAsync.type
    var cuMemcpyDtoH: cuMemcpyDtoH.type
    var cuMemcpyDtoHAsync: cuMemcpyDtoHAsync.type
    var cuMemcpyDtoD: cuMemcpyDtoD.type
    var cuMemcpyDtoDAsync: cuMemcpyDtoDAsync.type

    # cuMemSet
    var cuMemsetD8: cuMemsetD8.type
    var cuMemsetD8Async: cuMemsetD8Async.type
    var cuMemsetD16Async: cuMemsetD16Async.type
    var cuMemsetD32Async: cuMemsetD32Async.type

    # cuFunc
    var cuLaunchKernelEx: cuLaunchKernelEx.type
    var cuFuncSetCacheConfig: cuFuncSetCacheConfig.type
    var cuFuncSetAttribute: cuFuncSetAttribute.type
    var cuFuncGetAttribute: cuFuncGetAttribute.type

    # cuModule
    var cuModuleLoad: cuModuleLoad.type
    var cuModuleLoadData: cuModuleLoadData.type
    var cuModuleLoadDataEx: cuModuleLoadDataEx.type
    var cuModuleUnload: cuModuleUnload.type
    var cuModuleGetFunction: cuModuleGetFunction.type
    var cuModuleGetGlobal: cuModuleGetGlobal.type

    # cuMem
    var cuMemGetInfo: cuMemGetInfo.type

    fn __init__(out self):
        self.cuDeviceGetCount = cuDeviceGetCount.load()
        self.cuDeviceGetAttribute = cuDeviceGetAttribute.load()
        self.cuDriverGetVersion = cuDriverGetVersion.load()
        self.cuDeviceGetName = cuDeviceGetName.load()
        self.cuDeviceTotalMem = cuDeviceTotalMem.load()
        self.cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease.load()
        self.cuDevicePrimaryCtxReset = cuDevicePrimaryCtxReset.load()
        self.cuDevicePrimaryCtxRetain = cuDevicePrimaryCtxRetain.load()
        self.cuDevicePrimaryCtxSetFlags = cuDevicePrimaryCtxSetFlags.load()
        self.cuCtxCreate = cuCtxCreate.load()
        self.cuCtxResetPersistingL2Cache = cuCtxResetPersistingL2Cache.load()
        self.cuCtxPushCurrent = cuCtxPushCurrent.load()
        self.cuCtxGetCurrent = cuCtxGetCurrent.load()
        self.cuCtxSetCacheConfig = cuCtxSetCacheConfig.load()
        self.cuCtxDestroy = cuCtxDestroy.load()
        self.cuCtxSynchronize = cuCtxSynchronize.load()
        self.cuCtxSetCurrent = cuCtxSetCurrent.load()
        self.cuEventCreate = cuEventCreate.load()
        self.cuEventDestroy = cuEventDestroy.load()
        self.cuEventSynchronize = cuEventSynchronize.load()
        self.cuEventRecord = cuEventRecord.load()
        self.cuEventElapsedTime = cuEventElapsedTime.load()
        self.cuStreamCreate = cuStreamCreate.load()
        self.cuStreamDestroy = cuStreamDestroy.load()
        self.cuStreamSynchronize = cuStreamSynchronize.load()
        self.cuMemAllocHost = cuMemAllocHost.load()
        self.cuMemAlloc = cuMemAlloc.load()
        self.cuMemAllocAsync = cuMemAllocAsync.load()
        self.cuMemAllocManaged = cuMemAllocManaged.load()
        self.cuMemFreeHost = cuMemFreeHost.load()
        self.cuMemFree = cuMemFree.load()
        self.cuMemFreeAsync = cuMemFreeAsync.load()
        self.cuMemcpyHtoD = cuMemcpyHtoD.load()
        self.cuMemcpyHtoDAsync = cuMemcpyHtoDAsync.load()
        self.cuMemcpyDtoH = cuMemcpyDtoH.load()
        self.cuMemcpyDtoHAsync = cuMemcpyDtoHAsync.load()
        self.cuMemcpyDtoDAsync = cuMemcpyDtoDAsync.load()
        self.cuMemcpyDtoD = cuMemcpyDtoD.load()
        self.cuMemsetD8 = cuMemsetD8.load()
        self.cuMemsetD8Async = cuMemsetD8Async.load()
        self.cuMemsetD16Async = cuMemsetD16Async.load()
        self.cuMemsetD32Async = cuMemsetD32Async.load()
        self.cuLaunchKernelEx = cuLaunchKernelEx.load()
        self.cuFuncSetCacheConfig = cuFuncSetCacheConfig.load()
        self.cuFuncSetAttribute = cuFuncSetAttribute.load()
        self.cuFuncGetAttribute = cuFuncGetAttribute.load()
        self.cuModuleLoad = cuModuleLoad.load()
        self.cuModuleLoadData = cuModuleLoadData.load()
        self.cuModuleLoadDataEx = cuModuleLoadDataEx.load()
        self.cuModuleUnload = cuModuleUnload.load()
        self.cuModuleGetFunction = cuModuleGetFunction.load()
        self.cuModuleGetGlobal = cuModuleGetGlobal.load()
        self.cuMemGetInfo = cuMemGetInfo.load()

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other


struct CudaInstance:
    var cuda_dll: CudaDLL

    fn __init__(out self) raises:
        self.cuda_dll = CudaDLL()

    fn __copyinit__(out self, existing: Self):
        self.cuda_dll = existing.cuda_dll

    fn num_devices(self) raises -> Int:
        var res: Int32 = 0
        _check_error(
            self.cuda_dll.cuDeviceGetCount(UnsafePointer.address_of(res))
        )
        return int(res)
