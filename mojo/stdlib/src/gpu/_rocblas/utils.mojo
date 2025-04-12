# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List
from collections.string import StaticString
from os import abort
from pathlib import Path
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle

from .types import Status

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias ROCM_ROCBLAS_LIBRARY_PATHS = List[String](
    "/opt/rocm/lib/librocblas.so",
    "/opt/rocm/lib/librocblas.so.4",
)

alias ROCM_ROCBLAS_LIBRARY = _Global[
    "ROCM_ROCBLAS_LIBRARY", _OwnedDLHandle, _init_dylib
]()


fn _get_library_path() -> String:
    for path in ROCM_ROCBLAS_LIBRARY_PATHS:
        if Path(path[]).exists():
            return path[]
    return abort[String](
        String(
            (
                "the ROCM rocBLAS library was not found in any of the"
                " following paths: "
            ),
            String(", ").join(ROCM_ROCBLAS_LIBRARY_PATHS),
        )
    )


fn _init_dylib() -> _OwnedDLHandle:
    return _OwnedDLHandle(_get_library_path())


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
