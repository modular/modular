# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements some utilties."""

from collections import List
from os import abort
from pathlib import Path
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function

from .result import Result as DriverResult

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _check_error(err: DriverResult) raises:
    if err != DriverResult.SUCCESS:
        raise Error(err.__str__())


fn _human_memory(size: Int) -> String:
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

    if size >= GB:
        return str(Float32(size) / GB) + "GB"

    if size >= MB:
        return str(Float32(size) / MB) + "MB"

    if size >= KB:
        return str(Float32(size) / KB) + "KB"

    return str(size) + "B"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _get_cuda_driver_path() raises -> Path:
    alias _CUDA_DRIVER_LIB_NAME = "libcuda.so"
    var _DEFAULT_CUDA_DRIVER_BASE_PATHS = List[Path](
        Path("/usr/lib/x86_64-linux-gnu"),  # Ubuntu like
        Path("/usr/lib64/nvidia"),  # Redhat like
        Path("/usr/lib/wsl/lib"),  # WSL
    )

    # Quick lookup for libcuda.so assuming it's symlinked.
    for loc in _DEFAULT_CUDA_DRIVER_BASE_PATHS:
        var lib_path = loc[] / _CUDA_DRIVER_LIB_NAME
        if lib_path.exists():
            return lib_path

    # If we cannot find libcuda.so, then search harder.
    for loc in _DEFAULT_CUDA_DRIVER_BASE_PATHS:
        for file in loc[].listdir():
            var lib_path = loc[] / file[]
            if not lib_path.is_file():
                continue
            if _CUDA_DRIVER_LIB_NAME in str(file[]):
                return lib_path

    raise "the CUDA library was not found"


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    try:
        var driver_path = _get_cuda_driver_path()
        var ptr = UnsafePointer[DLHandle].alloc(1)
        var handle = DLHandle(str(driver_path))
        _ = handle.get_function[fn (UInt32) -> Result]("cuInit")(0)
        ptr[] = handle
        return ptr.bitcast[NoneType]()
    except e:
        return abort[UnsafePointer[NoneType]](e)


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyTrivialRegType
]() raises -> result_type:
    return _ffi_get_dylib_function[
        "CUDA_DRIVER_LIBRARY",
        func_name,
        _init_dylib,
        _destroy_dylib,
        result_type,
    ]()


@value
@register_passable("trivial")
struct CudaHandle(Boolable, Stringable):
    var handle: DTypePointer[DType.invalid]

    fn __init__(inout self):
        self.handle = DTypePointer[DType.invalid]()

    fn __init__(inout self, handle: DTypePointer[DType.invalid]):
        self.handle = handle

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    fn __str__(self) -> String:
        return str(self.handle)


alias _ContextHandle = CudaHandle
alias _EventHandle = CudaHandle
alias _StreamHandle = CudaHandle
alias _ModuleHandle = CudaHandle
alias _FunctionHandle = CudaHandle
alias _DeviceHandle = Int32
