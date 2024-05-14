# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements some utilties."""

from os import abort
from pathlib import Path
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function

from ._constants import CUDA_DRIVER_PATH
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
        return String(Float32(size) / GB) + "GB"

    if size >= MB:
        return String(Float32(size) / MB) + "MB"

    if size >= KB:
        return String(Float32(size) / KB) + "KB"

    return String(size) + "B"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    if not Path(CUDA_DRIVER_PATH).exists():
        print("the CUDA library was not found at", CUDA_DRIVER_PATH)
        abort()

    var ptr = UnsafePointer[DLHandle].alloc(1)
    var handle = DLHandle(CUDA_DRIVER_PATH)
    _ = handle.get_function[fn (UInt32) -> Result]("cuInit")(0)
    ptr[] = handle
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyRegType
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
