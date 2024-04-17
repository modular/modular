# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort
from sys.info import os_is_macos
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from collections import OptionalReg as Optional
from buffer.buffer import NDBuffer
from .BatchedMatmul import _reshape_nd_buffer_with_batch_to_3d
from utils.index import Index

# ===----------------------------------------------------------------------===#
# Constants
# ===----------------------------------------------------------------------===#

alias LIB_ACC_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
alias LIB_ACC_PLIST = "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Resources/Info.plist"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    var handle = DLHandle(LIB_ACC_PATH)
    if not handle:
        abort("the accelerate library was not found at " + LIB_ACC_PATH)
    var ptr = Pointer[DLHandle].alloc(1)
    ptr[] = handle
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
@register_passable("trivial")
struct _CBLASOrder:
    var value: Int32
    alias ROW_MAJOR = _CBLASOrder(101)
    alias COL_MAJOR = _CBLASOrder(102)


@value
@register_passable("trivial")
struct _CBLASTranspose:
    var value: Int32
    alias NO_TRANSPOSE = _CBLASTranspose(111)
    alias TRANSPOSE = _CBLASTranspose(112)
    alias CONJ_TRANSPOSE = _CBLASTranspose(113)


@always_inline
fn _cblas_f32[
    *,
    transpose_b: Bool = False,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    constrained[a.rank == b.rank == c.rank == 2, "rank must be 2"]()
    constrained[
        a.type == b.type == c.type == DType.float32,
        "input and output types must be float32",
    ]()

    var M = Int32(a.dim[0]())
    var N = Int32(b.dim[0]() if transpose_b else b.dim[1]())
    var K = Int32(a.dim[1]())

    var lda = K
    var ldb = N
    var ldc = N

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
    var cblas_gemm = _get_dylib_function[
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
    ]()
    cblas_gemm(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.TRANSPOSE if transpose_b else _CBLASTranspose.NO_TRANSPOSE,
        M,
        N,
        K,
        Float32(1.0),
        rebind[DTypePointer[DType.float32]](a.data),
        lda,
        rebind[DTypePointer[DType.float32]](b.data),
        ldb,
        Float32(0.0),
        rebind[DTypePointer[DType.float32]](c.data),
        ldc,
    )


# ===----------------------------------------------------------------------===#
# Matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn matmul[
    *,
    transpose_b: Bool = False,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    @parameter
    if a.type == b.type == c.type == DType.float32:
        return _cblas_f32[transpose_b=transpose_b](c, a, b)

    constrained[False, "unsupported type in apple accelerate"]()


# ===----------------------------------------------------------------------===#
# Batched Matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn batched_matmul[
    *,
    transpose_b: Bool = False,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    var c3 = _reshape_nd_buffer_with_batch_to_3d(c)
    var a3 = _reshape_nd_buffer_with_batch_to_3d(a)
    var b3 = _reshape_nd_buffer_with_batch_to_3d(b)
    var batch_size = c.dim[0]()

    var m = c3.dim[1]()
    var n = c3.dim[2]()
    var k = a3.dim[2]()

    var c_shape = Index(c3.dim[1](), c3.dim[2]())
    var a_shape = Index(a3.dim[1](), a3.dim[2]())
    var b_shape = Index(b3.dim[1](), b3.dim[2]())

    for batch in range(batch_size):
        var c2 = NDBuffer[c.type, 2, address_space = c.address_space](
            c3.data + (c_shape[0] * c_shape[1]) * batch, c_shape
        )
        var a2 = NDBuffer[a.type, 2, address_space = a.address_space](
            a3.data + (a_shape[0] * a_shape[1]) * batch, a_shape
        )
        var b2 = NDBuffer[b.type, 2, address_space = b.address_space](
            b3.data + (b_shape[0] * b_shape[1]) * batch, b_shape
        )

        matmul[transpose_b=transpose_b](c2, a2, b2)
