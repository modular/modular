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

from gpu.host._nvidia_cuda import CUstream

from .dtype import DataType, Property
from .result import Result

alias cublasContext = NoneType

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_CUBLAS_LIBRARY_PATH = "/usr/local/cuda/lib64/libcublas.so.12"


alias CUDA_CUBLAS_LIBRARY = _Global[
    "CUDA_CUBLAS_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    if not Path(CUDA_CUBLAS_LIBRARY_PATH).exists():
        return abort[_OwnedDLHandle](
            "the CUDA cuBLAS library was not found at "
            + CUDA_CUBLAS_LIBRARY_PATH
        )
    return _OwnedDLHandle(CUDA_CUBLAS_LIBRARY_PATH)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUBLAS_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline
fn check_cublas_error(stat: Result) raises:
    if stat != Result.SUCCESS:
        raise Error(String("CUBLAS ERROR:", stat))


@always_inline
fn _convert_to_cublas_datatype[mojo_type: DType]() -> DataType:
    @parameter
    if mojo_type is DType.float32:
        return DataType.R_32F
    elif mojo_type is DType.float16:
        return DataType.R_16F
    elif mojo_type is DType.float8_e4m3fn:
        return DataType.R_8F_E4M3
    elif mojo_type is DType.float8_e5m2:
        return DataType.R_8F_E5M2
    else:
        constrained[
            mojo_type is DType.bfloat16,
            (
                "Only support FP32, FP16, BF16, E4M3, and E5M2. Please extend"
                " it if more types are needed."
            ),
        ]()
        return DataType.R_16BF


@always_inline
fn _convert_to_cublas_transpose(transpose: Bool) -> cublasOperation_t:
    return (
        cublasOperation_t.CUBLAS_OP_T if transpose else cublasOperation_t.CUBLAS_OP_N
    )


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


fn cublasScopy(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasScopy_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDgemv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDgemv_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasStpsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStpsv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDgbmv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    kl: Int16,
    ku: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDgbmv_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDgemmStridedBatched(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    stride_a: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int64,
    stride_b: Int64,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDgemmStridedBatched_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            Int64,
            UnsafePointer[Float64],
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            Int64,
            Int64,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


fn cublasDsyrkx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int64,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsyrkx_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasUint8gemmBias(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    transc: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Int8],
    _a_bias: Int16,
    lda: Int16,
    _b: UnsafePointer[Int8],
    _b_bias: Int16,
    ldb: Int16,
    _c: UnsafePointer[Int8],
    _c_bias: Int16,
    ldc: Int16,
    _c_mult: Int16,
    _c_shift: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasUint8gemmBias",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Int8],
            Int16,
            Int16,
            UnsafePointer[Int8],
            Int16,
            Int16,
            UnsafePointer[Int8],
            Int16,
            Int16,
            Int16,
            Int16,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        transc,
        m,
        n,
        k,
        _a,
        _a_bias,
        lda,
        _b,
        _b_bias,
        ldb,
        _c,
        _c_bias,
        ldc,
        _c_mult,
        _c_shift,
    )


fn cublasGetProperty(type: Property, value: UnsafePointer[Int16]) -> Result:
    return _get_dylib_function[
        "cublasGetProperty",
        fn (Property, UnsafePointer[Int16]) -> Result,
    ]()(type, value)


fn cublasSsyr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsyr_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasIdamax(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    result: UnsafePointer[Int16],
) -> Result:
    return _get_dylib_function[
        "cublasIdamax_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Int16],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasGetMatrix(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: UnsafePointer[NoneType],
    lda: Int16,
    _b: UnsafePointer[NoneType],
    ldb: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasGetMatrix",
        fn (
            Int16,
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasSgemvStridedBatched(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    stride_a: Int64,
    x: UnsafePointer[Float32],
    incx: Int16,
    stridex: Int64,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int16,
    stridey: Int64,
    batch_count: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSgemvStridedBatched",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            Int64,
            UnsafePointer[Float32],
            Int16,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            Int64,
            Int16,
        ) -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


fn cublasStrsm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    _b: UnsafePointer[Float32],
    ldb: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStrsm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasRotmEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int16,
    param: UnsafePointer[NoneType],
    param_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasRotmEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        param,
        param_type,
        executiontype,
    )


fn cublasSgemm(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSgemm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSgeam(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    beta: UnsafePointer[Float32],
    _b: UnsafePointer[Float32],
    ldb: Int64,
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSgeam_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasStrttp(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
    _ap: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasStrttp",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, uplo, n, _a, lda, _ap)


fn cublasRotmgEx(
    handle: UnsafePointer[cublasContext],
    d1: UnsafePointer[NoneType],
    d1_type: DataType,
    d2: UnsafePointer[NoneType],
    d2_type: DataType,
    x1: UnsafePointer[NoneType],
    x1_type: DataType,
    y1: UnsafePointer[NoneType],
    y1_type: DataType,
    param: UnsafePointer[NoneType],
    param_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasRotmgEx",
        fn (
            UnsafePointer[cublasContext],
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        d1,
        d1_type,
        d2,
        d2_type,
        x1,
        x1_type,
        y1,
        y1_type,
        param,
        param_type,
        executiontype,
    )


fn cublasStrmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStrmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


@value
@register_passable("trivial")
struct cublasPointerMode_t:
    var _value: Int32
    alias CUBLAS_POINTER_MODE_HOST = cublasPointerMode_t(0)
    alias CUBLAS_POINTER_MODE_DEVICE = cublasPointerMode_t(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.CUBLAS_POINTER_MODE_HOST:
            return "CUBLAS_POINTER_MODE_HOST"
        if self == Self.CUBLAS_POINTER_MODE_DEVICE:
            return "CUBLAS_POINTER_MODE_DEVICE"
        return abort[String]("invalid cublasPointerMode_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasDnrm2(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    result: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDnrm2_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasIaminEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[Int16],
) -> Result:
    return _get_dylib_function[
        "cublasIaminEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[Int16],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasDger(
    handle: UnsafePointer[cublasContext],
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDger_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDgemmStridedBatched(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    stride_a: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int16,
    stride_b: Int64,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDgemmStridedBatched",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            Int64,
            UnsafePointer[Float64],
            Int16,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            Int64,
            Int16,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


@value
@register_passable("trivial")
struct cublasMath_t:
    var _value: Int32
    alias CUBLAS_DEFAULT_MATH = cublasMath_t(0)
    alias CUBLAS_TENSOR_OP_MATH = cublasMath_t(1)
    alias CUBLAS_PEDANTIC_MATH = cublasMath_t(2)
    alias CUBLAS_TF32_TENSOR_OP_MATH = cublasMath_t(3)
    alias CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = cublasMath_t(4)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.CUBLAS_DEFAULT_MATH:
            return "CUBLAS_DEFAULT_MATH"
        if self == Self.CUBLAS_TENSOR_OP_MATH:
            return "CUBLAS_TENSOR_OP_MATH"
        if self == Self.CUBLAS_PEDANTIC_MATH:
            return "CUBLAS_PEDANTIC_MATH"
        if self == Self.CUBLAS_TF32_TENSOR_OP_MATH:
            return "CUBLAS_TF32_TENSOR_OP_MATH"
        if self == Self.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
            return "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION"
        return abort[String]("invalid cublasMath_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSdot(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
    result: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSdot_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasGetMatrixAsync(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: UnsafePointer[NoneType],
    lda: Int16,
    _b: UnsafePointer[NoneType],
    ldb: Int16,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasGetMatrixAsync",
        fn (
            Int16,
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


fn cublasGetVector(
    n: Int64,
    elem_size: Int64,
    x: UnsafePointer[NoneType],
    incx: Int64,
    y: UnsafePointer[NoneType],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasGetVector_64",
        fn (
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
        ) -> Result,
    ]()(n, elem_size, x, incx, y, incy)


fn cublasStrsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStrsv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasSgemv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSgemv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasXerbla(sr_name: UnsafePointer[Int8], info: Int16) -> None:
    return _get_dylib_function[
        "cublasXerbla", fn (UnsafePointer[Int8], Int16) -> None
    ]()(sr_name, info)


fn cublasGetMatrixAsync(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: UnsafePointer[NoneType],
    lda: Int64,
    _b: UnsafePointer[NoneType],
    ldb: Int64,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasGetMatrixAsync_64",
        fn (
            Int64,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


fn cublasStbsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStbsv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasGetSmCountTarget(
    handle: UnsafePointer[cublasContext], sm_count_target: UnsafePointer[Int16]
) -> Result:
    return _get_dylib_function[
        "cublasGetSmCountTarget",
        fn (UnsafePointer[cublasContext], UnsafePointer[Int16]) -> Result,
    ]()(handle, sm_count_target)


fn cublasSetMathMode(
    handle: UnsafePointer[cublasContext], mode: cublasMath_t
) -> Result:
    return _get_dylib_function[
        "cublasSetMathMode",
        fn (UnsafePointer[cublasContext], cublasMath_t) -> Result,
    ]()(handle, mode)


fn cublasDsbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsbmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSdot(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
    result: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSdot_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasSsbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsbmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasIsamax(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    result: UnsafePointer[Int64],
) -> Result:
    return _get_dylib_function[
        "cublasIsamax_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Int64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSdgmm(
    handle: UnsafePointer[cublasContext],
    mode: cublasSideMode_t,
    m: Int64,
    n: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSdgmm_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasSwapEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSwapEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasDotcEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int16,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasDotcEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


fn cublasRotEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int16,
    c: UnsafePointer[NoneType],
    s: UnsafePointer[NoneType],
    cs_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasRotEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        c,
        s,
        cs_type,
        executiontype,
    )


fn cublasSsymv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsymv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSsyr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsyr2_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasGetStream(
    handle: UnsafePointer[cublasContext],
    stream_id: UnsafePointer[CUstream],
) -> Result:
    return _get_dylib_function[
        "cublasGetStream_v2",
        fn (UnsafePointer[cublasContext], UnsafePointer[CUstream]) -> Result,
    ]()(handle, stream_id)


fn cublasIsamin(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    result: UnsafePointer[Int16],
) -> Result:
    return _get_dylib_function[
        "cublasIsamin_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Int16],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasStbsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStbsv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasSetMatrixAsync(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: UnsafePointer[NoneType],
    lda: Int16,
    _b: UnsafePointer[NoneType],
    ldb: Int16,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasSetMatrixAsync",
        fn (
            Int16,
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


fn cublasSaxpy(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSaxpy_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasDgeam(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    beta: UnsafePointer[Float64],
    _b: UnsafePointer[Float64],
    ldb: Int16,
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDgeam",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasCopyEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasCopyEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasGetCudartVersion() -> Int:
    return _get_dylib_function["cublasGetCudartVersion", fn () -> Int]()()


fn cublasIdamax(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    result: UnsafePointer[Int64],
) -> Result:
    return _get_dylib_function[
        "cublasIdamax_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Int64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSsyr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsyr2_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDaxpy(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDaxpy_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasDsyr2k(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int64,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsyr2k_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSetLoggerCallback(
    user_callback: fn (UnsafePointer[Int8]) -> None,
) -> Result:
    return _get_dylib_function[
        "cublasSetLoggerCallback",
        fn (fn (UnsafePointer[Int8]) -> None) -> Result,
    ]()(user_callback)


fn cublasSgeam(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    beta: UnsafePointer[Float32],
    _b: UnsafePointer[Float32],
    ldb: Int16,
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSgeam",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasDtpttr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    _ap: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtpttr",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, _ap, _a, lda)


fn cublasIamaxEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[Int16],
) -> Result:
    return _get_dylib_function[
        "cublasIamaxEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[Int16],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasSspmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _ap: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSspmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSsymv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsymv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasGemmStridedBatchedEx(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[NoneType],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int64,
    stride_a: Int64,
    _b: UnsafePointer[NoneType],
    _btype: DataType,
    ldb: Int64,
    stride_b: Int64,
    beta: UnsafePointer[NoneType],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
    compute_type: ComputeType,
    algo: Algorithm,
) -> Result:
    return _get_dylib_function[
        "cublasGemmStridedBatchedEx_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int64,
            Int64,
            Int64,
            ComputeType,
            Algorithm,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        stride_a,
        _b,
        _btype,
        ldb,
        stride_b,
        beta,
        _c,
        _ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    )


fn cublasNrm2Ex(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasNrm2Ex_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, execution_type)


fn cublasGetPointerMode(
    handle: UnsafePointer[cublasContext],
    mode: UnsafePointer[cublasPointerMode_t],
) -> Result:
    return _get_dylib_function[
        "cublasGetPointerMode_v2",
        fn (
            UnsafePointer[cublasContext], UnsafePointer[cublasPointerMode_t]
        ) -> Result,
    ]()(handle, mode)


fn cublasSrotm(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
    param: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSrotm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


@value
@register_passable("trivial")
struct Algorithm:
    var _value: Int32

    # According to https://docs.nvidia.com/cuda/cublas/#cublasgemmalgo-t, the
    # only useful algorithm options are default and algo0 - algo23.
    # We never specify 0-23 in pratice.

    alias DEFAULT = Self(-1)
    alias ALGO0 = Self(0)
    alias ALGO1 = Self(1)
    alias ALGO2 = Self(2)
    alias ALGO3 = Self(3)
    alias ALGO4 = Self(4)
    alias ALGO5 = Self(5)
    alias ALGO6 = Self(6)
    alias ALGO7 = Self(7)
    alias ALGO8 = Self(8)
    alias ALGO9 = Self(9)
    alias ALGO10 = Self(10)
    alias ALGO11 = Self(11)
    alias ALGO12 = Self(12)
    alias ALGO13 = Self(13)
    alias ALGO14 = Self(14)
    alias ALGO15 = Self(15)
    alias ALGO16 = Self(16)
    alias ALGO17 = Self(17)
    alias ALGO18 = Self(18)
    alias ALGO19 = Self(19)
    alias ALGO20 = Self(20)
    alias ALGO21 = Self(21)
    alias ALGO22 = Self(22)
    alias ALGO23 = Self(23)
    alias DEFAULT_TENSOR_OP = Self(99)
    alias ALGO0_TENSOR_OP = Self(100)
    alias ALGO1_TENSOR_OP = Self(101)
    alias ALGO2_TENSOR_OP = Self(102)
    alias ALGO3_TENSOR_OP = Self(103)
    alias ALGO4_TENSOR_OP = Self(104)
    alias ALGO5_TENSOR_OP = Self(105)
    alias ALGO6_TENSOR_OP = Self(106)
    alias ALGO7_TENSOR_OP = Self(107)
    alias ALGO8_TENSOR_OP = Self(108)
    alias ALGO9_TENSOR_OP = Self(109)
    alias ALGO10_TENSOR_OP = Self(110)
    alias ALGO11_TENSOR_OP = Self(111)
    alias ALGO12_TENSOR_OP = Self(112)
    alias ALGO13_TENSOR_OP = Self(113)
    alias ALGO14_TENSOR_OP = Self(114)
    alias ALGO15_TENSOR_OP = Self(115)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.DEFAULT:
            return "DEFAULT"
        if self == Self.ALGO0:
            return "ALGO0"
        if self == Self.ALGO1:
            return "ALGO1"
        if self == Self.ALGO2:
            return "ALGO2"
        if self == Self.ALGO3:
            return "ALGO3"
        if self == Self.ALGO4:
            return "ALGO4"
        if self == Self.ALGO5:
            return "ALGO5"
        if self == Self.ALGO6:
            return "ALGO6"
        if self == Self.ALGO7:
            return "ALGO7"
        if self == Self.ALGO8:
            return "ALGO8"
        if self == Self.ALGO9:
            return "ALGO9"
        if self == Self.ALGO10:
            return "ALGO10"
        if self == Self.ALGO11:
            return "ALGO11"
        if self == Self.ALGO12:
            return "ALGO12"
        if self == Self.ALGO13:
            return "ALGO13"
        if self == Self.ALGO14:
            return "ALGO14"
        if self == Self.ALGO15:
            return "ALGO15"
        if self == Self.ALGO16:
            return "ALGO16"
        if self == Self.ALGO17:
            return "ALGO17"
        if self == Self.ALGO18:
            return "ALGO18"
        if self == Self.ALGO19:
            return "ALGO19"
        if self == Self.ALGO20:
            return "ALGO20"
        if self == Self.ALGO21:
            return "ALGO21"
        if self == Self.ALGO22:
            return "ALGO22"
        if self == Self.ALGO23:
            return "ALGO23"
        if self == Self.DEFAULT_TENSOR_OP:
            return "DEFAULT_TENSOR_OP"
        if self == Self.ALGO0_TENSOR_OP:
            return "ALGO0_TENSOR_OP"
        if self == Self.ALGO1_TENSOR_OP:
            return "ALGO1_TENSOR_OP"
        if self == Self.ALGO2_TENSOR_OP:
            return "ALGO2_TENSOR_OP"
        if self == Self.ALGO3_TENSOR_OP:
            return "ALGO3_TENSOR_OP"
        if self == Self.ALGO4_TENSOR_OP:
            return "ALGO4_TENSOR_OP"
        if self == Self.ALGO5_TENSOR_OP:
            return "ALGO5_TENSOR_OP"
        if self == Self.ALGO6_TENSOR_OP:
            return "ALGO6_TENSOR_OP"
        if self == Self.ALGO7_TENSOR_OP:
            return "ALGO7_TENSOR_OP"
        if self == Self.ALGO8_TENSOR_OP:
            return "ALGO8_TENSOR_OP"
        if self == Self.ALGO9_TENSOR_OP:
            return "ALGO9_TENSOR_OP"
        if self == Self.ALGO10_TENSOR_OP:
            return "ALGO10_TENSOR_OP"
        if self == Self.ALGO11_TENSOR_OP:
            return "ALGO11_TENSOR_OP"
        if self == Self.ALGO12_TENSOR_OP:
            return "ALGO12_TENSOR_OP"
        if self == Self.ALGO13_TENSOR_OP:
            return "ALGO13_TENSOR_OP"
        if self == Self.ALGO14_TENSOR_OP:
            return "ALGO14_TENSOR_OP"
        if self == Self.ALGO15_TENSOR_OP:
            return "ALGO15_TENSOR_OP"
        return abort[String]("invalid Algorithm entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSsyrk(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsyrk_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDsyr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsyr_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasStrmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStrmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDcopy(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDcopy_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDtrmm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int64,
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtrmm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


fn cublasDdot(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
    result: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDdot_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasSscal(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSscal_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasSgemmStridedBatched(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    stride_a: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int64,
    stride_b: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSgemmStridedBatched_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            Int64,
            UnsafePointer[Float32],
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            Int64,
            Int64,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


fn cublasDdgmm(
    handle: UnsafePointer[cublasContext],
    mode: cublasSideMode_t,
    m: Int64,
    n: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDdgmm_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasStpttr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    _ap: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStpttr",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, _ap, _a, lda)


fn cublasDsyr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsyr_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasSetVector(
    n: Int16,
    elem_size: Int16,
    x: UnsafePointer[NoneType],
    incx: Int16,
    device_ptr: UnsafePointer[NoneType],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSetVector",
        fn (
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
        ) -> Result,
    ]()(n, elem_size, x, incx, device_ptr, incy)


fn cublasSetMatrixAsync(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: UnsafePointer[NoneType],
    lda: Int64,
    _b: UnsafePointer[NoneType],
    ldb: Int64,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasSetMatrixAsync_64",
        fn (
            Int64,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


# fn cublasGetLoggerCallback(user_callback: UNKNOWN) -> Result:
#     return _get_dylib_function[
#         "cublasGetLoggerCallback", fn (UNKNOWN) -> Result
#     ]()(user_callback)


fn cublasSasum(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    result: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSasum_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasRotgEx(
    handle: UnsafePointer[cublasContext],
    a: UnsafePointer[NoneType],
    b: UnsafePointer[NoneType],
    ab_type: DataType,
    c: UnsafePointer[NoneType],
    s: UnsafePointer[NoneType],
    cs_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasRotgEx",
        fn (
            UnsafePointer[cublasContext],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, a, b, ab_type, c, s, cs_type, executiontype)


@value
@register_passable("trivial")
struct cublasDiagType_t:
    var _value: Int32
    alias CUBLAS_DIAG_NON_UNIT = cublasDiagType_t(0)
    alias CUBLAS_DIAG_UNIT = cublasDiagType_t(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.CUBLAS_DIAG_NON_UNIT:
            return "CUBLAS_DIAG_NON_UNIT"
        if self == Self.CUBLAS_DIAG_UNIT:
            return "CUBLAS_DIAG_UNIT"
        return abort[String]("invalid cublasDiagType_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct ComputeType:
    var _value: Int32
    alias COMPUTE_16F = Self(64)
    alias COMPUTE_16F_PEDANTIC = Self(65)
    alias COMPUTE_32F = Self(68)
    alias COMPUTE_32F_PEDANTIC = Self(69)
    alias COMPUTE_32F_FAST_16F = Self(74)
    alias COMPUTE_32F_FAST_16BF = Self(75)
    alias COMPUTE_32F_FAST_TF32 = Self(77)
    alias COMPUTE_64F = Self(70)
    alias COMPUTE_64F_PEDANTIC = Self(71)
    alias COMPUTE_32I = Self(72)
    alias COMPUTE_32I_PEDANTIC = Self(73)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.COMPUTE_16F:
            return "COMPUTE_16F"
        if self == Self.COMPUTE_16F_PEDANTIC:
            return "COMPUTE_16F_PEDANTIC"
        if self == Self.COMPUTE_32F:
            return "COMPUTE_32F"
        if self == Self.COMPUTE_32F_PEDANTIC:
            return "COMPUTE_32F_PEDANTIC"
        if self == Self.COMPUTE_32F_FAST_16F:
            return "COMPUTE_32F_FAST_16F"
        if self == Self.COMPUTE_32F_FAST_16BF:
            return "COMPUTE_32F_FAST_16BF"
        if self == Self.COMPUTE_32F_FAST_TF32:
            return "COMPUTE_32F_FAST_TF32"
        if self == Self.COMPUTE_64F:
            return "COMPUTE_64F"
        if self == Self.COMPUTE_64F_PEDANTIC:
            return "COMPUTE_64F_PEDANTIC"
        if self == Self.COMPUTE_32I:
            return "COMPUTE_32I"
        if self == Self.COMPUTE_32I_PEDANTIC:
            return "COMPUTE_32I_PEDANTIC"
        return abort[String]("invalid ComputeType entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasDsymm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int64,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsymm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSspr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
    _ap: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSspr_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasIdamin(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    result: UnsafePointer[Int64],
) -> Result:
    return _get_dylib_function[
        "cublasIdamin_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Int64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasGetVectorAsync(
    n: Int16,
    elem_size: Int16,
    device_ptr: UnsafePointer[NoneType],
    incx: Int16,
    host_ptr: UnsafePointer[NoneType],
    incy: Int16,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasGetVectorAsync",
        fn (
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, device_ptr, incx, host_ptr, incy, stream)


fn cublasGetMatrix(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: UnsafePointer[NoneType],
    lda: Int64,
    _b: UnsafePointer[NoneType],
    ldb: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasGetMatrix_64",
        fn (
            Int64,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasDaxpy(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDaxpy_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasDsyr2k(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    _b: UnsafePointer[Float64],
    ldb: Int16,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsyr2k_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSger(
    handle: UnsafePointer[cublasContext],
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSger_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasSdgmm(
    handle: UnsafePointer[cublasContext],
    mode: cublasSideMode_t,
    m: Int16,
    n: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSdgmm",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasDtbsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtbsv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasDtrsm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    _b: UnsafePointer[Float64],
    ldb: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtrsm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasStbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStbmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasDspmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _ap: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDspmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSswap(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSswap_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDspmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _ap: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDspmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSrotmg(
    handle: UnsafePointer[cublasContext],
    d1: UnsafePointer[Float32],
    d2: UnsafePointer[Float32],
    x1: UnsafePointer[Float32],
    y1: UnsafePointer[Float32],
    param: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSrotmg_v2",
        fn (
            UnsafePointer[cublasContext],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, d1, d2, x1, y1, param)


fn cublasDtpmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtpmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDasum(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    result: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDasum_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasRotEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int64,
    c: UnsafePointer[NoneType],
    s: UnsafePointer[NoneType],
    cs_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasRotEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        c,
        s,
        cs_type,
        executiontype,
    )


fn cublasDrotm(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
    param: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDrotm_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


fn cublasAxpyEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    alpha: UnsafePointer[NoneType],
    alpha_type: DataType,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int16,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasAxpyEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        executiontype,
    )


fn cublasSgemm(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    _b: UnsafePointer[Float32],
    ldb: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSgemm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSsymm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsymm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasCopyEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasCopyEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasSwapEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSwapEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasSrot(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
    c: UnsafePointer[Float32],
    s: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSrot_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasGetVector(
    n: Int16,
    elem_size: Int16,
    x: UnsafePointer[NoneType],
    incx: Int16,
    y: UnsafePointer[NoneType],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasGetVector",
        fn (
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
        ) -> Result,
    ]()(n, elem_size, x, incx, y, incy)


fn cublasDtrsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtrsv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasSsymm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    _b: UnsafePointer[Float32],
    ldb: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsymm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasDtrmm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    _b: UnsafePointer[Float64],
    ldb: Int16,
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtrmm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


fn cublasCherk3mEx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasCherk3mEx_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int64,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


alias cublasLogCallback = fn (UnsafePointer[Int8]) -> None


fn cublasDtrmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtrmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDdgmm(
    handle: UnsafePointer[cublasContext],
    mode: cublasSideMode_t,
    m: Int16,
    n: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDdgmm",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasDtbsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtbsv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasSsyr2k(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    _b: UnsafePointer[Float32],
    ldb: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsyr2k_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasDgemm(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    _b: UnsafePointer[Float64],
    ldb: Int16,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDgemm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasGetMathMode(
    handle: UnsafePointer[cublasContext], mode: UnsafePointer[cublasMath_t]
) -> Result:
    return _get_dylib_function[
        "cublasGetMathMode",
        fn (
            UnsafePointer[cublasContext], UnsafePointer[cublasMath_t]
        ) -> Result,
    ]()(handle, mode)


fn cublasDrot(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
    c: UnsafePointer[Float64],
    s: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDrot_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasSspr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
    _ap: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSspr_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasGemmEx64(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[NoneType],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int64,
    _b: UnsafePointer[NoneType],
    _btype: DataType,
    ldb: Int64,
    beta: UnsafePointer[NoneType],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int64,
    compute_type: ComputeType,
    algo: Algorithm,
) -> Result:
    return _get_dylib_function[
        "cublasGemmEx_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int64,
            ComputeType,
            Algorithm,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
        compute_type,
        algo,
    )


fn cublasDotEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int16,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasDotEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


fn cublasSswap(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSswap_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDrotm(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
    param: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDrotm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


fn cublasSgemmEx(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int64,
    _b: UnsafePointer[NoneType],
    _btype: DataType,
    ldb: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSgemmEx_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int64,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
    )


fn cublasDgemm(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int64,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDgemm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSsyrk(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsyrk_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDnrm2(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    result: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDnrm2_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDasum(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    result: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDasum_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDsyrkx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    _b: UnsafePointer[Float64],
    ldb: Int16,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsyrkx",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasRotmEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int64,
    param: UnsafePointer[NoneType],
    param_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasRotmEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        param,
        param_type,
        executiontype,
    )


fn cublasDtpsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtpsv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSspr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
    _ap: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSspr2_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasSetMatrix(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: UnsafePointer[NoneType],
    lda: Int64,
    _b: UnsafePointer[NoneType],
    ldb: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSetMatrix_64",
        fn (
            Int64,
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasDrotg(
    handle: UnsafePointer[cublasContext],
    a: UnsafePointer[Float64],
    b: UnsafePointer[Float64],
    c: UnsafePointer[Float64],
    s: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDrotg_v2",
        fn (
            UnsafePointer[cublasContext],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, a, b, c, s)


fn cublasGetAtomicsMode(
    handle: UnsafePointer[cublasContext],
    mode: UnsafePointer[cublasAtomicsMode_t],
) -> Result:
    return _get_dylib_function[
        "cublasGetAtomicsMode",
        fn (
            UnsafePointer[cublasContext], UnsafePointer[cublasAtomicsMode_t]
        ) -> Result,
    ]()(handle, mode)


fn cublasStbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStbmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasAxpyEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    alpha: UnsafePointer[NoneType],
    alpha_type: DataType,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int64,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasAxpyEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        executiontype,
    )


fn cublasIaminEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[Int64],
) -> Result:
    return _get_dylib_function[
        "cublasIaminEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[Int64],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasDspr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
    _ap: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDspr2_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasDotEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int64,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasDotEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


fn cublasScopy(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasScopy_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDsyrk(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsyrk_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDestroy(handle: UnsafePointer[cublasContext]) -> Result:
    return _get_dylib_function[
        "cublasDestroy_v2", fn (UnsafePointer[cublasContext]) -> Result
    ]()(handle)


fn cublasSetVectorAsync(
    n: Int16,
    elem_size: Int16,
    host_ptr: UnsafePointer[NoneType],
    incx: Int16,
    device_ptr: UnsafePointer[NoneType],
    incy: Int16,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasSetVectorAsync",
        fn (
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, host_ptr, incx, device_ptr, incy, stream)


fn cublasIamaxEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[Int64],
) -> Result:
    return _get_dylib_function[
        "cublasIamaxEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[Int64],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasSsyrkx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsyrkx_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasDswap(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDswap_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasAsumEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasAsumEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, executiontype)


@value
@register_passable("trivial")
struct FillMode:
    var _value: Int32
    alias LOWER = Self(0)
    alias UPPER = Self(1)
    alias FULL = Self(2)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == Self.LOWER:
            return "LOWER"
        if self == Self.UPPER:
            return "UPPER"
        if self == Self.FULL:
            return "FULL"
        return abort[String]("invalid FillMode entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSspr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
    y: UnsafePointer[Float32],
    incy: Int64,
    _ap: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSspr2_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasSgbmv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    kl: Int64,
    ku: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSgbmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasAsumEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    executiontype: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasAsumEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, executiontype)


fn cublasGetVersion(
    handle: UnsafePointer[cublasContext], version: UnsafePointer[Int16]
) -> Result:
    return _get_dylib_function[
        "cublasGetVersion_v2",
        fn (UnsafePointer[cublasContext], UnsafePointer[Int16]) -> Result,
    ]()(handle, version)


fn cublasScalEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    alpha: UnsafePointer[NoneType],
    alpha_type: DataType,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasScalEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            DataType,
        ) -> Result,
    ]()(handle, n, alpha, alpha_type, x, x_type, incx, execution_type)


fn cublasSetPointerMode(
    handle: UnsafePointer[cublasContext], mode: cublasPointerMode_t
) -> Result:
    return _get_dylib_function[
        "cublasSetPointerMode_v2",
        fn (UnsafePointer[cublasContext], cublasPointerMode_t) -> Result,
    ]()(handle, mode)


fn cublasDgemv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDgemv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasGetStatusString(status: Result) -> UnsafePointer[Int8]:
    return _get_dylib_function[
        "cublasGetStatusString", fn (Result) -> UnsafePointer[Int8]
    ]()(status)


fn cublasSnrm2(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    result: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSnrm2_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDgbmv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    kl: Int64,
    ku: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDgbmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDsyr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsyr2_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDtpsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtpsv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSetVector(
    n: Int64,
    elem_size: Int64,
    x: UnsafePointer[NoneType],
    incx: Int64,
    device_ptr: UnsafePointer[NoneType],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSetVector_64",
        fn (
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
        ) -> Result,
    ]()(n, elem_size, x, incx, device_ptr, incy)


fn cublasDgemvStridedBatched(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    stride_a: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    stridex: Int64,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int64,
    stridey: Int64,
    batch_count: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDgemvStridedBatched_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            Int64,
            UnsafePointer[Float64],
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            Int64,
            Int64,
        ) -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


fn cublasSsyrkx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    _b: UnsafePointer[Float32],
    ldb: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsyrkx",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasGetStatusName(status: Result) -> UnsafePointer[Int8]:
    return _get_dylib_function[
        "cublasGetStatusName", fn (Result) -> UnsafePointer[Int8]
    ]()(status)


fn cublasDtbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtbmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasSrotg(
    handle: UnsafePointer[cublasContext],
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    s: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSrotg_v2",
        fn (
            UnsafePointer[cublasContext],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, a, b, c, s)


fn cublasCherkEx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasCherkEx",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int16,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


fn cublasDrotmg(
    handle: UnsafePointer[cublasContext],
    d1: UnsafePointer[Float64],
    d2: UnsafePointer[Float64],
    x1: UnsafePointer[Float64],
    y1: UnsafePointer[Float64],
    param: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDrotmg_v2",
        fn (
            UnsafePointer[cublasContext],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, d1, d2, x1, y1, param)


fn cublasDger(
    handle: UnsafePointer[cublasContext],
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDger_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasSscal(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSscal_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasSetWorkspace(
    handle: UnsafePointer[cublasContext],
    workspace: UnsafePointer[NoneType],
    workspace_size_in_bytes: Int,
) -> Result:
    return _get_dylib_function[
        "cublasSetWorkspace_v2",
        fn (
            UnsafePointer[cublasContext], UnsafePointer[NoneType], Int
        ) -> Result,
    ]()(handle, workspace, workspace_size_in_bytes)


fn cublasStpsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStpsv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDspr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
    _ap: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDspr_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasGemmEx(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int32,
    n: Int32,
    k: Int32,
    alpha: UnsafePointer[NoneType],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int32,
    _b: UnsafePointer[NoneType],
    _btype: DataType,
    ldb: Int32,
    beta: UnsafePointer[NoneType],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int32,
    compute_type: ComputeType,
    algo: Algorithm,
) -> Result:
    return _get_dylib_function[
        "cublasGemmEx",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int32,
            Int32,
            Int32,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int32,
            UnsafePointer[NoneType],
            DataType,
            Int32,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int32,
            ComputeType,
            Algorithm,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
        compute_type,
        algo,
    )


fn cublasSsbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSsbmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDgemvStridedBatched(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    stride_a: Int64,
    x: UnsafePointer[Float64],
    incx: Int16,
    stridex: Int64,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int16,
    stridey: Int64,
    batch_count: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDgemvStridedBatched",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            Int64,
            UnsafePointer[Float64],
            Int16,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            Int64,
            Int16,
        ) -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


fn cublasDsymv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsymv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasLoggerConfigure(
    log_is_on: Int16,
    log_to_std_out: Int16,
    log_to_std_err: Int16,
    log_file_name: UnsafePointer[Int8],
) -> Result:
    return _get_dylib_function[
        "cublasLoggerConfigure",
        fn (Int16, Int16, Int16, UnsafePointer[Int8]) -> Result,
    ]()(log_is_on, log_to_std_out, log_to_std_err, log_file_name)


fn cublasStpmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStpmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSgemvStridedBatched(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    stride_a: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    stridex: Int64,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int64,
    stridey: Int64,
    batch_count: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSgemvStridedBatched_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            Int64,
            UnsafePointer[Float32],
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            Int64,
            Int64,
        ) -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


fn cublasIsamin(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    result: UnsafePointer[Int64],
) -> Result:
    return _get_dylib_function[
        "cublasIsamin_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Int64],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDrot(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
    c: UnsafePointer[Float64],
    s: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDrot_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasDgeam(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    beta: UnsafePointer[Float64],
    _b: UnsafePointer[Float64],
    ldb: Int64,
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDgeam_64",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasGetVectorAsync(
    n: Int64,
    elem_size: Int64,
    device_ptr: UnsafePointer[NoneType],
    incx: Int64,
    host_ptr: UnsafePointer[NoneType],
    incy: Int64,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasGetVectorAsync_64",
        fn (
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, device_ptr, incx, host_ptr, incy, stream)


fn cublasStrsm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStrsm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasSgemmEx(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int16,
    _b: UnsafePointer[NoneType],
    _btype: DataType,
    ldb: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSgemmEx",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int16,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
    )


fn cublasStpmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStpmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDtrmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtrmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDtrsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtrsv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDsyr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
    _a: UnsafePointer[Float64],
    lda: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsyr2_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasSrot(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
    c: UnsafePointer[Float32],
    s: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSrot_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasDscal(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDscal_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasCreate(handle: UnsafePointer[UnsafePointer[cublasContext]]) -> Result:
    return _get_dylib_function[
        "cublasCreate_v2",
        fn (UnsafePointer[UnsafePointer[cublasContext]]) -> Result,
    ]()(handle)


fn cublasSetSmCountTarget(
    handle: UnsafePointer[cublasContext], sm_count_target: Int16
) -> Result:
    return _get_dylib_function[
        "cublasSetSmCountTarget",
        fn (UnsafePointer[cublasContext], Int16) -> Result,
    ]()(handle, sm_count_target)


fn cublasDswap(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDswap_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasStrsv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStrsv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDspr2(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
    _ap: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDspr2_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasSsyr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int64,
    _a: UnsafePointer[Float32],
    lda: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsyr_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasNrm2Ex(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasNrm2Ex",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, execution_type)


fn cublasDtbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDtbmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


@value
@register_passable("trivial")
struct cublasAtomicsMode_t:
    var _value: Int32
    alias CUBLAS_ATOMICS_NOT_ALLOWED = cublasAtomicsMode_t(0)
    alias CUBLAS_ATOMICS_ALLOWED = cublasAtomicsMode_t(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.CUBLAS_ATOMICS_NOT_ALLOWED:
            return "CUBLAS_ATOMICS_NOT_ALLOWED"
        if self == Self.CUBLAS_ATOMICS_ALLOWED:
            return "CUBLAS_ATOMICS_ALLOWED"
        return abort[String]("invalid cublasAtomicsMode_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSsyr2k(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasSsyr2k_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasCherk3mEx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int16,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasCherk3mEx",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int16,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


fn cublasScalEx(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    alpha: UnsafePointer[NoneType],
    alpha_type: DataType,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int16,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasScalEx",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[NoneType],
            DataType,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            DataType,
        ) -> Result,
    ]()(handle, n, alpha, alpha_type, x, x_type, incx, execution_type)


fn cublasDotcEx(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[NoneType],
    x_type: DataType,
    incx: Int64,
    y: UnsafePointer[NoneType],
    y_type: DataType,
    incy: Int64,
    result: UnsafePointer[NoneType],
    result_type: DataType,
    execution_type: DataType,
) -> Result:
    return _get_dylib_function[
        "cublasDotcEx_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            DataType,
        ) -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


fn cublasDsymm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    _b: UnsafePointer[Float64],
    ldb: Int16,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsymm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasIsamax(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    result: UnsafePointer[Int16],
) -> Result:
    return _get_dylib_function[
        "cublasIsamax_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Int16],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSaxpy(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSaxpy_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasSnrm2(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    result: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSnrm2_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasCherkEx(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasCherkEx_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[NoneType],
            DataType,
            Int64,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


@value
@register_passable("trivial")
struct cublasSideMode_t:
    var _value: Int32
    alias CUBLAS_SIDE_LEFT = cublasSideMode_t(0)
    alias CUBLAS_SIDE_RIGHT = cublasSideMode_t(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.CUBLAS_SIDE_LEFT:
            return "CUBLAS_SIDE_LEFT"
        if self == Self.CUBLAS_SIDE_RIGHT:
            return "CUBLAS_SIDE_RIGHT"
        return abort[String]("invalid cublasSideMode_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSetMatrix(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: UnsafePointer[NoneType],
    lda: Int16,
    _b: UnsafePointer[NoneType],
    ldb: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSetMatrix",
        fn (
            Int16,
            Int16,
            Int16,
            UnsafePointer[NoneType],
            Int16,
            UnsafePointer[NoneType],
            Int16,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasDtrsm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    _b: UnsafePointer[Float64],
    ldb: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtrsm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasDcopy(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDcopy_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasSetVectorAsync(
    n: Int64,
    elem_size: Int64,
    host_ptr: UnsafePointer[NoneType],
    incx: Int64,
    device_ptr: UnsafePointer[NoneType],
    incy: Int64,
    stream: CUstream,
) -> Result:
    return _get_dylib_function[
        "cublasSetVectorAsync_64",
        fn (
            Int64,
            Int64,
            UnsafePointer[NoneType],
            Int64,
            UnsafePointer[NoneType],
            Int64,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, host_ptr, incx, device_ptr, incy, stream)


fn cublasDspr(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int16,
    _ap: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDspr_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasSgemv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSgemv_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDtrttp(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    _a: UnsafePointer[Float64],
    lda: Int16,
    _ap: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDtrttp",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, uplo, n, _a, lda, _ap)


fn cublasDdot(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    y: UnsafePointer[Float64],
    incy: Int64,
    result: UnsafePointer[Float64],
) -> Result:
    return _get_dylib_function[
        "cublasDdot_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasGemmStridedBatchedEx(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[NoneType],
    _a: UnsafePointer[NoneType],
    _atype: DataType,
    lda: Int16,
    stride_a: Int64,
    _b: UnsafePointer[NoneType],
    _btype: DataType,
    ldb: Int16,
    stride_b: Int64,
    beta: UnsafePointer[NoneType],
    _c: UnsafePointer[NoneType],
    _ctype: DataType,
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
    compute_type: ComputeType,
    algo: Algorithm,
) -> Result:
    return _get_dylib_function[
        "cublasGemmStridedBatchedEx",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int16,
            Int64,
            UnsafePointer[NoneType],
            DataType,
            Int16,
            Int64,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            DataType,
            Int16,
            Int64,
            Int16,
            ComputeType,
            Algorithm,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        stride_a,
        _b,
        _btype,
        ldb,
        stride_b,
        beta,
        _c,
        _ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    )


fn cublasStrmm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int64,
    _c: UnsafePointer[Float32],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasStrmm_v2_64",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


fn cublasDsyrk(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    beta: UnsafePointer[Float64],
    _c: UnsafePointer[Float64],
    ldc: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsyrk_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDscal(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    alpha: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDscal_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasDtpmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float64],
    x: UnsafePointer[Float64],
    incx: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDtpmv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSgbmv(
    handle: UnsafePointer[cublasContext],
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    kl: Int16,
    ku: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSgbmv_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSrotm(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
    param: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSrotm_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


fn cublasSetAtomicsMode(
    handle: UnsafePointer[cublasContext], mode: cublasAtomicsMode_t
) -> Result:
    return _get_dylib_function[
        "cublasSetAtomicsMode",
        fn (UnsafePointer[cublasContext], cublasAtomicsMode_t) -> Result,
    ]()(handle, mode)


fn cublasDsbmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasDsbmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSger(
    handle: UnsafePointer[cublasContext],
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
    y: UnsafePointer[Float32],
    incy: Int16,
    _a: UnsafePointer[Float32],
    lda: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSger_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDsymv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64],
    _a: UnsafePointer[Float64],
    lda: Int64,
    x: UnsafePointer[Float64],
    incx: Int64,
    beta: UnsafePointer[Float64],
    y: UnsafePointer[Float64],
    incy: Int64,
) -> Result:
    return _get_dylib_function[
        "cublasDsymv_v2_64",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            Int64,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSetStream(
    handle: UnsafePointer[cublasContext], stream_id: CUstream
) -> Result:
    return _get_dylib_function[
        "cublasSetStream_v2",
        fn (UnsafePointer[cublasContext], CUstream) -> Result,
    ]()(handle, stream_id)


fn cublasStrmm(
    handle: UnsafePointer[cublasContext],
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    _b: UnsafePointer[Float32],
    ldb: Int16,
    _c: UnsafePointer[Float32],
    ldc: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasStrmm_v2",
        fn (
            UnsafePointer[cublasContext],
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


@value
@register_passable("trivial")
struct cublasOperation_t:
    var _value: Int32
    alias CUBLAS_OP_N = cublasOperation_t(0)
    alias CUBLAS_OP_T = cublasOperation_t(1)
    alias CUBLAS_OP_C = cublasOperation_t(2)
    alias CUBLAS_OP_HERMITAN = cublasOperation_t(2)
    alias CUBLAS_OP_CONJG = cublasOperation_t(3)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.CUBLAS_OP_N:
            return "CUBLAS_OP_N"
        if self == Self.CUBLAS_OP_T:
            return "CUBLAS_OP_T"
        if self == Self.CUBLAS_OP_C:
            return "CUBLAS_OP_C"
        if self == Self.CUBLAS_OP_HERMITAN:
            return "CUBLAS_OP_HERMITAN"
        if self == Self.CUBLAS_OP_CONJG:
            return "CUBLAS_OP_CONJG"
        return abort[String]("invalid cublasOperation_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasIdamin(
    handle: UnsafePointer[cublasContext],
    n: Int16,
    x: UnsafePointer[Float64],
    incx: Int16,
    result: UnsafePointer[Int16],
) -> Result:
    return _get_dylib_function[
        "cublasIdamin_v2",
        fn (
            UnsafePointer[cublasContext],
            Int16,
            UnsafePointer[Float64],
            Int16,
            UnsafePointer[Int16],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSspmv(
    handle: UnsafePointer[cublasContext],
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32],
    _ap: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    incx: Int16,
    beta: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    incy: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSspmv_v2",
        fn (
            UnsafePointer[cublasContext],
            FillMode,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSgemmStridedBatched(
    handle: UnsafePointer[cublasContext],
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32],
    _a: UnsafePointer[Float32],
    lda: Int16,
    stride_a: Int64,
    _b: UnsafePointer[Float32],
    ldb: Int16,
    stride_b: Int64,
    beta: UnsafePointer[Float32],
    _c: UnsafePointer[Float32],
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
) -> Result:
    return _get_dylib_function[
        "cublasSgemmStridedBatched",
        fn (
            UnsafePointer[cublasContext],
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            Int64,
            UnsafePointer[Float32],
            Int16,
            Int64,
            UnsafePointer[Float32],
            UnsafePointer[Float32],
            Int16,
            Int64,
            Int16,
        ) -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


fn cublasSasum(
    handle: UnsafePointer[cublasContext],
    n: Int64,
    x: UnsafePointer[Float32],
    incx: Int64,
    result: UnsafePointer[Float32],
) -> Result:
    return _get_dylib_function[
        "cublasSasum_v2_64",
        fn (
            UnsafePointer[cublasContext],
            Int64,
            UnsafePointer[Float32],
            Int64,
            UnsafePointer[Float32],
        ) -> Result,
    ]()(handle, n, x, incx, result)
