# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List
from os import abort
from pathlib import Path
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from .types import Status

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias ROCM_ROCBLAS_LIBRARY_PATH = "/opt/rocm/lib/librocblas.so"


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    if not Path(ROCM_ROCBLAS_LIBRARY_PATH).exists():
        return abort[UnsafePointer[NoneType]](
            "the ROCM rocBLAS library was not found at "
            + ROCM_ROCBLAS_LIBRARY_PATH
        )
    var ptr = UnsafePointer[DLHandle].alloc(1)
    ptr.init_pointee_move(DLHandle(ROCM_ROCBLAS_LIBRARY_PATH))
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        "ROCM_ROCBLAS_LIBRARY",
        func_name,
        _init_dylib,
        _destroy_dylib,
        result_type,
    ]()


@always_inline
fn check_error(stat: Status) raises:
    if stat != Status.SUCCESS:
        raise Error("ROCBLAS ERROR:" + str(stat))
