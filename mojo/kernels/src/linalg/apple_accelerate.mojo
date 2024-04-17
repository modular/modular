# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort
from pathlib import Path
from sys.info import os_is_macos
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from collections import OptionalReg as Optional
from buffer.buffer import NDBuffer

# ===----------------------------------------------------------------------===#
# Constants
# ===----------------------------------------------------------------------===#

alias LIB_ACC_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
alias LIB_ACC_PLIST = "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Resources/Info.plist"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    if not Path(LIB_ACC_PATH).exists():
        abort("the accelerate library was not found at" + LIB_ACC_PATH)

    var ptr = Pointer[DLHandle].alloc(1)
    ptr[] = DLHandle(LIB_ACC_PATH)
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyRegType
]() -> result_type:
    constrained[os_is_macos(), "operating system must be macOS"]()
    return _ffi_get_dylib_function[
        "ACCELERATE_PATH",
        func_name,
        _init_dylib,
        _destroy_dylib,
        result_type,
    ]()


# ===----------------------------------------------------------------------===#
# CBLAS
# ===----------------------------------------------------------------------===#


@value
struct _CBLASOrder:
    var value: Int32
    alias ROW_MAJOR = _CBLASOrder(101)
    alias COL_MAJOR = _CBLASOrder(102)


@value
struct _CBLASTranspose:
    var value: Int32
    alias NO_TRANSPOSE = _CBLASTranspose(111)
    alias TRANSPOSE = _CBLASTranspose(112)
    alias CONJ_TRANSPOSE = _CBLASTranspose(113)


@always_inline
fn _cblas_f32[
    transpose_b: Bool = False,
](c: NDBuffer[_, 2], a: NDBuffer[_, 2], b: NDBuffer[_, 2]):
    constrained[
        a.type == b.type == c.type == DType.float32,
        "input and output types must be float32",
    ]()

    var M = a.dim[0]()
    var N = b.dim[0]() if transpose_b else b.dim[1]()
    var K = a.dim[1]()

    # void cblas_sgemm(const enum CBLAS_ORDER ORDER,
    #                  const enum CBLAS_TRANSPOSE TRANSA,
    #                  const enum CBLAS_TRANSPOSE TRANSB,
    #                  const int M,
    #                  const int N,
    #                  const int K,
    #                  const float ALPHA,
    #                  const float *A,
    #                  const int LDA,
    #                  const float *B,
    #                  const int LDB,
    #                  const float BETA,
    #                  float *C,
    #                  const int LDC);
    _get_dylib_function[
        "cblas_sgemm",
        fn (
            _CBLASOrder,
            _CBLASTranspose,
            _CBLASTranspose,
            Int32,
            Int32,
            Int32,
            Float32,
            DTypePointer[DType.float32],
            Int32,
            DTypePointer[DType.float32],
            Int32,
            Float32,
            DTypePointer[DType.float32],
            Int32,
        ) -> NoneType,
    ]()(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.TRANSPOSE if transpose_b else _CBLASTranspose.NO_TRANSPOSE,
        Int32(M),
        Int32(N),
        Int32(K),
        Float32(1.0),
        rebind[DTypePointer[DType.float32]](a.data),
        Int32(K),
        rebind[DTypePointer[DType.float32]](b.data),
        Int32(b.dim[1]()),
        Float32(0.0),
        rebind[DTypePointer[DType.float32]](c.data),
        Int32(N),
    )
