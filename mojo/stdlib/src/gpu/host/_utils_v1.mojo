# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements some utilties."""

from collections import List
from math import floor
from os import abort
from os.path import isdir
from pathlib import Path
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function

from builtin._location import __call_location, _SourceLocation

from .result_v1 import Result as DriverResult

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _check_error(
    err: DriverResult,
    *,
    msg: StringLiteral = "",
    location: OptionalReg[_SourceLocation] = None,
) raises:
    _check_error_impl(err, msg, location.or_else(__call_location()))


@no_inline
fn _check_error_impl(
    err: DriverResult,
    msg: StringLiteral,
    location: _SourceLocation,
) raises:
    """We do not want to inline this code since we want to make sure that the
    stringification of the error is not duplicated many times."""
    if err != DriverResult.SUCCESS:
        raise Error(location.prefix(str(err) + " " + msg))


fn _pretty_print_float(val: Float64) -> String:
    """This converts the float value to a string, but omits the fractional part
    if not needed (e.g. prints 2 instead of 2.0).
    """
    if Float64(floor(val)) == val:
        return str(int(val))
    return str(val)


fn _human_memory(size: Int) -> String:
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

    if size >= GB:
        return _pretty_print_float(Float64(size) / GB) + "GB"

    if size >= MB:
        return _pretty_print_float(Float64(size) / MB) + "MB"

    if size >= KB:
        return _pretty_print_float(Float64(size) / KB) + "KB"

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
        if isdir(loc[]):
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
        ptr.init_pointee_move(DLHandle(str(driver_path)))
        _ = ptr[].get_function[fn (UInt32) -> Result]("cuInit")(0)
        return ptr.bitcast[NoneType]()
    except e:
        return abort[UnsafePointer[NoneType]](e)


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyTrivialRegType
]() -> result_type:
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
    var handle: UnsafePointer[NoneType]

    fn __init__(out self):
        self.handle = UnsafePointer[NoneType]()

    fn __init__(out self, handle: UnsafePointer[NoneType]):
        self.handle = handle

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    @no_inline
    fn __str__(self) -> String:
        return str(self.handle)


alias _ContextHandle = CudaHandle
alias _EventHandle = CudaHandle
alias _StreamHandle = CudaHandle
alias _ModuleHandle = CudaHandle
alias _FunctionHandle = CudaHandle
alias _DeviceHandle = Int32
