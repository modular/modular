# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA memory operations."""

from os.path import isdir
from pathlib import Path
from sys import sizeof
from sys.ffi import _OwnedDLHandle, _Global, _get_dylib_function

from builtin._location import __call_location, _SourceLocation
from gpu.host.nvidia_cuda import CUcontext
from gpu.host.result_v1 import Result
from memory import UnsafePointer, stack_allocation
from memory.unsafe import bitcast

from utils import StaticTuple

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _check_error(
    err: Result,
    *,
    msg: StringLiteral = "",
    location: OptionalReg[_SourceLocation] = None,
) raises:
    _check_error_impl(err, msg, location.or_else(__call_location()))


@no_inline
fn _check_error_impl(
    err: Result,
    msg: StringLiteral,
    location: _SourceLocation,
) raises:
    """We do not want to inline this code since we want to make sure that the
    stringification of the error is not duplicated many times."""
    if err != Result.SUCCESS:
        raise Error(location.prefix(str(err) + " " + msg))


# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


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


alias CUDA_DRIVER_LIBRARY = _Global[
    "CUDA_DRIVER_LIBRARY", _OwnedDLHandle, _init_dylib
]()


fn _init_dylib() -> _OwnedDLHandle:
    try:
        var driver_path = _get_cuda_driver_path()
        var dylib = _OwnedDLHandle(str(driver_path))
        _ = dylib._handle.get_function[fn (UInt32) -> Result]("cuInit")(0)
        return dylib^
    except e:
        return abort[_OwnedDLHandle](e)


@always_inline
fn _get_cuda_dylib_function[
    func_name: StringLiteral, result_type: AnyTrivialRegType
]() -> result_type:
    return _get_dylib_function[
        CUDA_DRIVER_LIBRARY,
        func_name,
        result_type,
    ]()


fn _make_ctx_current(ctx: CUcontext) raises -> CUcontext:
    var prev_ctx = CUcontext()
    _check_error(
        _get_cuda_dylib_function[
            "cuCtxGetCurrent", fn (UnsafePointer[CUcontext]) -> Result
        ]()(UnsafePointer.address_of(prev_ctx))
    )
    _check_error(
        _get_cuda_dylib_function["cuCtxSetCurrent", fn (CUcontext) -> Result]()(
            ctx
        )
    )
    return prev_ctx


# ===-----------------------------------------------------------------------===#
# Memory
# ===-----------------------------------------------------------------------===#


fn _malloc[type: AnyType](count: Int) raises -> UnsafePointer[type]:
    """Allocates GPU device memory."""

    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_cuda_dylib_function[
            "cuMemAlloc_v2",
            fn (UnsafePointer[UnsafePointer[Int]], Int) -> Result,
        ]()(UnsafePointer.address_of(ptr), count * sizeof[type]())
    )
    return ptr.bitcast[type]()


fn _malloc[type: DType](count: Int) raises -> UnsafePointer[Scalar[type]]:
    return _malloc[Scalar[type]](count)


fn _malloc_managed[type: AnyType](count: Int) raises -> UnsafePointer[type]:
    """Allocates memory that will be automatically managed by the Unified Memory system.
    """
    alias CU_MEM_ATTACH_GLOBAL = UInt32(1)
    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_cuda_dylib_function[
            "cuMemAllocManaged",
            fn (UnsafePointer[UnsafePointer[Int]], Int, UInt32) -> Result,
        ]()(
            UnsafePointer.address_of(ptr),
            count * sizeof[type](),
            CU_MEM_ATTACH_GLOBAL,
        )
    )
    return ptr.bitcast[type]()


fn _malloc_managed[
    type: DType
](count: Int) raises -> UnsafePointer[Scalar[type]]:
    return _malloc_managed[Scalar[type]](count)


fn _free[type: AnyType](ptr: UnsafePointer[type]) raises:
    """Frees allocated GPU device memory."""

    _check_error(
        _get_cuda_dylib_function[
            "cuMemFree_v2", fn (UnsafePointer[Int]) -> Result
        ]()(ptr.bitcast[Int]())
    )


fn _memset[
    type: AnyType
](device_dest: UnsafePointer[type], val: UInt8, count: Int) raises:
    """Sets the memory range of N 8-bit values to a specified value."""

    _check_error(
        _get_cuda_dylib_function[
            "cuMemsetD8_v2", fn (UnsafePointer[Int], UInt8, Int) -> Result
        ]()(
            device_dest.bitcast[Int](),
            val,
            count * sizeof[type](),
        )
    )
