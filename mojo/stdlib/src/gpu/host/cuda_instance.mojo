# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import C_char

from ._utils import (
    _check_error,
    _ContextHandle,
    _EventHandle,
    _get_dylib_function,
    _human_memory,
    _ModuleHandle,
    _StreamHandle,
)
from .device import Device
from .event import Flag
from .function import _FunctionHandle


@register_passable("trivial")
struct _dylib_function[fn_name: StringLiteral, type: AnyTrivialRegType]:
    @staticmethod
    fn load() -> type:
        return _get_dylib_function[fn_name, type]()


alias _DeviceHandle = Int32

alias cuDeviceGetCount = _dylib_function[
    "cuDeviceGetCount", fn (UnsafePointer[Int32]) -> Result
]

alias cuDeviceGetAttribute = _dylib_function[
    "cuDeviceGetAttribute",
    fn (UnsafePointer[Int32], DeviceAttribute, _DeviceHandle) -> Result,
]

alias cuDeviceGetName = _dylib_function[
    "cuDeviceGetName",
    fn (UnsafePointer[Int8], Int32, _DeviceHandle) -> Result,
]

alias cuDeviceTotalMem = _dylib_function[
    "cuDeviceTotalMem_v2", fn (UnsafePointer[Int], _DeviceHandle) -> Result
]

alias cuCtxCreate = _dylib_function[
    "cuCtxCreate_v2",
    fn (UnsafePointer[_ContextHandle], Int32, _DeviceHandle) -> Result,
]

alias cuCtxDestroy = _dylib_function[
    "cuCtxDestroy_v2", fn (_ContextHandle) -> Result
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

alias cuLaunchKernel = _dylib_function[
    "cuLaunchKernel",
    fn (
        _FunctionHandle,
        UInt32,  # GridDimZ
        UInt32,  # GridDimY
        UInt32,  # GridDimX
        UInt32,  # BlockDimZ
        UInt32,  # BlockDimY
        UInt32,  # BlockDimX
        UInt32,  # SharedMemSize
        _StreamHandle,
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

alias cuModuleLoad = _dylib_function[
    "cuModuleLoad",
    fn (UnsafePointer[_ModuleHandle], UnsafePointer[C_char]) -> Result,
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
        UnsafePointer[C_char],
    ) -> Result,
]


@value
struct CudaDLL:
    # cuDevice
    var cuDeviceGetCount: cuDeviceGetCount.type
    var cuDeviceGetAttribute: cuDeviceGetAttribute.type
    var cuDeviceGetName: cuDeviceGetName.type
    var cuDeviceTotalMem: cuDeviceTotalMem.type

    # cuCtx
    var cuCtxCreate: cuCtxCreate.type
    var cuCtxDestroy: cuCtxDestroy.type
    var cuCtxSynchronize: cuCtxSynchronize.type

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
    var cuMemAlloc: cuMemAlloc.type
    var cuMemAllocAsync: cuMemAllocAsync.type
    var cuMemAllocManaged: cuMemAllocManaged.type
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
    var cuLaunchKernel: cuLaunchKernel.type
    var cuFuncSetCacheConfig: cuFuncSetCacheConfig.type
    var cuFuncSetAttribute: cuFuncSetAttribute.type

    # cuModule
    var cuModuleLoad: cuModuleLoad.type
    var cuModuleLoadData: cuModuleLoadData.type
    var cuModuleLoadDataEx: cuModuleLoadDataEx.type
    var cuModuleUnload: cuModuleUnload.type
    var cuModuleGetFunction: cuModuleGetFunction.type

    fn __init__(inout self):
        self.cuDeviceGetCount = cuDeviceGetCount.load()
        self.cuDeviceGetAttribute = cuDeviceGetAttribute.load()
        self.cuDeviceGetName = cuDeviceGetName.load()
        self.cuDeviceTotalMem = cuDeviceTotalMem.load()
        self.cuCtxCreate = cuCtxCreate.load()
        self.cuCtxDestroy = cuCtxDestroy.load()
        self.cuCtxSynchronize = cuCtxSynchronize.load()
        self.cuEventCreate = cuEventCreate.load()
        self.cuEventDestroy = cuEventDestroy.load()
        self.cuEventSynchronize = cuEventSynchronize.load()
        self.cuEventRecord = cuEventRecord.load()
        self.cuEventElapsedTime = cuEventElapsedTime.load()
        self.cuStreamCreate = cuStreamCreate.load()
        self.cuStreamDestroy = cuStreamDestroy.load()
        self.cuStreamSynchronize = cuStreamSynchronize.load()
        self.cuMemAlloc = cuMemAlloc.load()
        self.cuMemAllocAsync = cuMemAllocAsync.load()
        self.cuMemAllocManaged = cuMemAllocManaged.load()
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
        self.cuLaunchKernel = cuLaunchKernel.load()
        self.cuFuncSetCacheConfig = cuFuncSetCacheConfig.load()
        self.cuFuncSetAttribute = cuFuncSetAttribute.load()
        self.cuModuleLoad = cuModuleLoad.load()
        self.cuModuleLoadData = cuModuleLoadData.load()
        self.cuModuleLoadDataEx = cuModuleLoadDataEx.load()
        self.cuModuleUnload = cuModuleUnload.load()
        self.cuModuleGetFunction = cuModuleGetFunction.load()

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other


struct CudaInstance:
    var cuda_dll: CudaDLL

    fn __init__(inout self) raises:
        self.cuda_dll = CudaDLL()

    fn __enter__(owned self) -> Self:
        return self^

    fn __moveinit__(inout self, owned existing: Self):
        self.cuda_dll = existing.cuda_dll
        existing.cuda_dll = CudaDLL()

    fn __copyinit__(inout self, existing: Self):
        self.cuda_dll = existing.cuda_dll

    fn num_devices(self) raises -> Int:
        var res: Int32 = 0
        _check_error(
            self.cuda_dll.cuDeviceGetCount(UnsafePointer.address_of(res))
        )
        return int(res)
