# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List
from os import abort
from pathlib import Path
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle
from collections.string import StaticString

from .types import Status

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias ROCM_ROCBLAS_LIBRARY_PATH = "/opt/rocm/lib/librocblas.so.4"

alias ROCM_ROCBLAS_LIBRARY = _Global[
    "ROCM_ROCBLAS_LIBRARY", _OwnedDLHandle, _init_dylib
]()


fn _init_dylib() -> _OwnedDLHandle:
    if not Path(ROCM_ROCBLAS_LIBRARY_PATH).exists():
        return abort[_OwnedDLHandle](
            "the ROCM rocBLAS library was not found at "
            + ROCM_ROCBLAS_LIBRARY_PATH
        )
    var dylib = _OwnedDLHandle(ROCM_ROCBLAS_LIBRARY_PATH)
    return dylib^


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        ROCM_ROCBLAS_LIBRARY,
        func_name,
        result_type,
    ]()


@always_inline
fn check_error(stat: Status) raises:
    if stat != Status.SUCCESS:
        raise String("ROCBLAS ERROR:", stat)
