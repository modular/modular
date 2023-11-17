# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements some utilties."""

from sys.ffi import DLHandle

from ._constants import CUDA_DRIVER_PATH
from .result import Result as DriverResult
from .nvml import Result as NVMLResult
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from pathlib import Path
from debug import trap

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _check_error(err: DriverResult) raises:
    if err != DriverResult.SUCCESS:
        raise Error(err.__str__())


@always_inline
fn _check_error(err: NVMLResult) raises:
    if err != NVMLResult.SUCCESS:
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


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    if not Path(CUDA_DRIVER_PATH).exists():
        print("the CUDA library was not found at", CUDA_DRIVER_PATH)
        trap()

    let ptr = Pointer[DLHandle].alloc(1)
    let handle = DLHandle(CUDA_DRIVER_PATH)
    _ = handle.get_function[fn (UInt32) -> Result]("cuInit")(0)
    __get_address_as_lvalue(ptr.address) = handle
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    __get_address_as_lvalue(ptr.bitcast[DLHandle]().address)._del_old()
    ptr.free()


@always_inline
fn _get_dylib_function[
    result_type: AnyRegType
](name: StringRef) raises -> result_type:
    return _ffi_get_dylib_function[
        "CUDA_DRIVER_LIBRARY", _init_dylib, _destroy_dylib, result_type
    ](name)
