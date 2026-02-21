# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from os import abort
from pathlib import Path
from ffi import _find_dylib
from ffi import _get_dylib_function as _ffi_get_dylib_function
from ffi import _Global, OwnedDLHandle

from gpu.host._nvidia_cuda import CUstream

from .dtype import DataType, Property
from .result import Result


comptime cublasContext = NoneType
comptime cublasHandle_t = UnsafePointer[cublasContext, MutAnyOrigin]

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUBLAS_LIBRARY_PATHS: List[Path] = [
    "libcublas.so.13",
    "/usr/local/cuda-13.1/lib64/libcublas.so.13",
    "/usr/local/cuda-13.0/lib64/libcublas.so.13",
    "/usr/local/cuda/lib64/libcublas.so.13",
    "libcublas.so.12",
    "/usr/local/cuda-12.8/lib64/libcublas.so.12",
    "/usr/local/cuda/lib64/libcublas.so.12",
]


fn _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the cuBLAS libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUBLAS_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUBLAS_LIBRARY = _Global[
    "CUDA_CUBLAS_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


fn _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUBLAS_LIBRARY_PATHS]()
    )


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: __TypeOfAllTypes
]() raises -> result_type:
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
        raise Error(String("failed to operate on CUBLAS due to error: ", stat))


@always_inline
fn check_cublas_error(stat: Result, msg: StringSlice) raises:
    if stat != Result.SUCCESS:
        raise Error(String(msg, ". Got a CUBLAS error: ", stat))


@always_inline
fn _convert_to_cublas_datatype[mojo_type: DType]() -> DataType:
    comptime if mojo_type == DType.float32:
        return DataType.R_32F
    elif mojo_type == DType.float16:
        return DataType.R_16F
    elif mojo_type == DType.float8_e4m3fn:
        return DataType.R_8F_E4M3
    elif mojo_type == DType.float8_e5m2:
        return DataType.R_8F_E5M2
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    elif mojo_type == DType.uint8:
        return DataType.R_4F_E2M1
    else:
        comptime assert mojo_type == DType.bfloat16, (
            "Only support FP32, FP16, BF16, E4M3, E5M2, and E2M1x2 (UInt8)."
            " Please extend it if more types are needed."
        )
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
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasScopy_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemv_v2",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasStpsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpsv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    kl: Int16,
    ku: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgbmv_v2",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    stride_a: Int64,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int64,
    stride_b: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemmStridedBatched_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrkx_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasUint8gemmBias(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    transc: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Int8, ImmutAnyOrigin],
    _a_bias: Int16,
    lda: Int16,
    _b: UnsafePointer[Int8, ImmutAnyOrigin],
    _b_bias: Int16,
    ldb: Int16,
    _c: UnsafePointer[Int8, MutAnyOrigin],
    _c_bias: Int16,
    ldc: Int16,
    _c_mult: Int16,
    _c_shift: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasUint8gemmBias",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Int8, ImmutAnyOrigin],
            Int16,
            Int16,
            UnsafePointer[Int8, ImmutAnyOrigin],
            Int16,
            Int16,
            UnsafePointer[Int8, MutAnyOrigin],
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


fn cublasGetProperty(
    type: Property, value: UnsafePointer[Int16, MutAnyOrigin]
) raises -> Result:
    return _get_dylib_function[
        "cublasGetProperty",
        fn(Property, UnsafePointer[Int16, MutAnyOrigin]) -> Result,
    ]()(type, value)


fn cublasSsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    _a: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasIdamax(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamax_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Int16, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasGetMatrix(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrix",
        fn(
            Int16,
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasSgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    stride_a: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    stridex: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
    stridey: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemvStridedBatched",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
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
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float32, MutAnyOrigin],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsm_v2",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasRotmEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    param: OpaquePointer[ImmutAnyOrigin],
    param_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotmEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
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
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemm_v2_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int64,
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgeam_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasStrttp(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    _ap: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasStrttp",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, _a, lda, _ap)


fn cublasRotmgEx(
    handle: cublasHandle_t,
    d1: OpaquePointer[MutAnyOrigin],
    d1_type: DataType,
    d2: OpaquePointer[MutAnyOrigin],
    d2_type: DataType,
    x1: OpaquePointer[MutAnyOrigin],
    x1_type: DataType,
    y1: OpaquePointer[ImmutAnyOrigin],
    y1_type: DataType,
    param: OpaquePointer[MutAnyOrigin],
    param_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotmgEx",
        fn(
            cublasHandle_t,
            OpaquePointer[MutAnyOrigin],
            DataType,
            OpaquePointer[MutAnyOrigin],
            DataType,
            OpaquePointer[MutAnyOrigin],
            DataType,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


@fieldwise_init
struct cublasPointerMode_t(TrivialRegisterPassable):
    var _value: Int32
    comptime CUBLAS_POINTER_MODE_HOST = cublasPointerMode_t(0)
    comptime CUBLAS_POINTER_MODE_DEVICE = cublasPointerMode_t(1)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid cublasPointerMode_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasDnrm2(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDnrm2_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasIaminEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIaminEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            UnsafePointer[Int16, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasDger(
    handle: cublasHandle_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int64,
    _a: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDger_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    stride_a: Int64,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int16,
    stride_b: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemmStridedBatched",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
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


@fieldwise_init
struct cublasMath_t(TrivialRegisterPassable):
    var _value: Int32
    comptime CUBLAS_DEFAULT_MATH = cublasMath_t(0)
    comptime CUBLAS_TENSOR_OP_MATH = cublasMath_t(1)
    comptime CUBLAS_PEDANTIC_MATH = cublasMath_t(2)
    comptime CUBLAS_TF32_TENSOR_OP_MATH = cublasMath_t(3)
    comptime CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = cublasMath_t(4)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid cublasMath_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSdot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int64,
    result: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSdot_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasGetMatrixAsync(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrixAsync",
        fn(
            Int16,
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


fn cublasGetVector(
    n: Int64,
    elem_size: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVector_64",
        fn(
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(n, elem_size, x, incx, y, incy)


fn cublasStrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasSgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemv_v2_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasXerbla(
    sr_name: UnsafePointer[Int8, ImmutAnyOrigin], info: Int16
) raises:
    return _get_dylib_function[
        "cublasXerbla", fn(UnsafePointer[Int8, ImmutAnyOrigin], Int16) -> None
    ]()(sr_name, info)


fn cublasGetMatrixAsync(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrixAsync_64",
        fn(
            Int64,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


fn cublasStbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbsv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasGetSmCountTarget(
    handle: cublasHandle_t,
    sm_count_target: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetSmCountTarget",
        fn(cublasHandle_t, UnsafePointer[Int16, MutAnyOrigin]) -> Result,
    ]()(handle, sm_count_target)


fn cublasSetMathMode(
    handle: cublasHandle_t, mode: cublasMath_t
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMathMode",
        fn(cublasHandle_t, cublasMath_t) -> Result,
    ]()(handle, mode)


fn cublasDsbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsbmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSdot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int16,
    result: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSdot_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasSsbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsbmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasIsamax(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Int64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamax_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Int64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int64,
    n: Int64,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSdgmm_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasSwapEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSwapEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasDotcEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotcEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    c: OpaquePointer[ImmutAnyOrigin],
    s: OpaquePointer[ImmutAnyOrigin],
    cs_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int16,
    _a: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasGetStream(
    handle: cublasHandle_t,
    stream_id: UnsafePointer[CUstream, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetStream_v2",
        fn(cublasHandle_t, UnsafePointer[CUstream, MutAnyOrigin]) -> Result,
    ]()(handle, stream_id)


fn cublasIsamin(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamin_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Int16, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasStbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbsv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasSetMatrixAsync(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrixAsync",
        fn(
            Int16,
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


fn cublasSaxpy(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSaxpy_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasDgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int16,
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgeam",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasCopyEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasCopyEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasGetCudartVersion() raises -> Int:
    return _get_dylib_function["cublasGetCudartVersion", fn() -> Int]()()


fn cublasIdamax(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Int64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamax_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Int64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int64,
    _a: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDaxpy(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDaxpy_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasDsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2k_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSetLoggerCallback(
    user_callback: fn(UnsafePointer[Int8, ImmutAnyOrigin]) -> None,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetLoggerCallback",
        fn(fn(UnsafePointer[Int8, ImmutAnyOrigin]) -> None) -> Result,
    ]()(user_callback)


fn cublasSgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int16,
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgeam",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasDtpttr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _ap: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpttr",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, _ap, _a, lda)


fn cublasIamaxEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIamaxEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            UnsafePointer[Int16, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasSspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _ap: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSspmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSsymv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    stride_a: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int64,
    stride_b: Int64,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmStridedBatchedEx_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasNrm2Ex_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, execution_type)


fn cublasGetPointerMode(
    handle: cublasHandle_t,
    mode: UnsafePointer[cublasPointerMode_t, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetPointerMode_v2",
        fn(
            cublasHandle_t,
            UnsafePointer[cublasPointerMode_t, MutAnyOrigin],
        ) -> Result,
    ]()(handle, mode)


fn cublasSrotm(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
    param: UnsafePointer[Float32, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotm_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


@fieldwise_init
struct Algorithm(TrivialRegisterPassable):
    var _value: Int32

    # According to https://docs.nvidia.com/cuda/cublas/#cublasgemmalgo-t, the
    # only useful algorithm options are default and algo0 - algo23.
    # We never specify 0-23 in practice.

    comptime DEFAULT = Self(-1)
    comptime ALGO0 = Self(0)
    comptime ALGO1 = Self(1)
    comptime ALGO2 = Self(2)
    comptime ALGO3 = Self(3)
    comptime ALGO4 = Self(4)
    comptime ALGO5 = Self(5)
    comptime ALGO6 = Self(6)
    comptime ALGO7 = Self(7)
    comptime ALGO8 = Self(8)
    comptime ALGO9 = Self(9)
    comptime ALGO10 = Self(10)
    comptime ALGO11 = Self(11)
    comptime ALGO12 = Self(12)
    comptime ALGO13 = Self(13)
    comptime ALGO14 = Self(14)
    comptime ALGO15 = Self(15)
    comptime ALGO16 = Self(16)
    comptime ALGO17 = Self(17)
    comptime ALGO18 = Self(18)
    comptime ALGO19 = Self(19)
    comptime ALGO20 = Self(20)
    comptime ALGO21 = Self(21)
    comptime ALGO22 = Self(22)
    comptime ALGO23 = Self(23)
    comptime DEFAULT_TENSOR_OP = Self(99)
    comptime ALGO0_TENSOR_OP = Self(100)
    comptime ALGO1_TENSOR_OP = Self(101)
    comptime ALGO2_TENSOR_OP = Self(102)
    comptime ALGO3_TENSOR_OP = Self(103)
    comptime ALGO4_TENSOR_OP = Self(104)
    comptime ALGO5_TENSOR_OP = Self(105)
    comptime ALGO6_TENSOR_OP = Self(106)
    comptime ALGO7_TENSOR_OP = Self(107)
    comptime ALGO8_TENSOR_OP = Self(108)
    comptime ALGO9_TENSOR_OP = Self(109)
    comptime ALGO10_TENSOR_OP = Self(110)
    comptime ALGO11_TENSOR_OP = Self(111)
    comptime ALGO12_TENSOR_OP = Self(112)
    comptime ALGO13_TENSOR_OP = Self(113)
    comptime ALGO14_TENSOR_OP = Self(114)
    comptime ALGO15_TENSOR_OP = Self(115)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid Algorithm entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrk_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    _a: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasStrmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDcopy(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDcopy_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDtrmm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int64,
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmm_v2_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


fn cublasDdot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int16,
    result: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDdot_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasSscal(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSscal_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    stride_a: Int64,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int64,
    stride_b: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmStridedBatched_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
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
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int64,
    n: Int64,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDdgmm_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasStpttr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _ap: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpttr",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, _ap, _a, lda)


fn cublasDsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    _a: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasSetVector(
    n: Int16,
    elem_size: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVector",
        fn(
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(n, elem_size, x, incx, device_ptr, incy)


fn cublasSetMatrixAsync(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrixAsync_64",
        fn(
            Int64,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
            CUstream,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


# fn cublasGetLoggerCallback(user_callback: UNKNOWN) raises -> Result:
#     return _get_dylib_function[
#         "cublasGetLoggerCallback", fn (UNKNOWN) -> Result
#     ]()(user_callback)


fn cublasSasum(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSasum_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasRotgEx(
    handle: cublasHandle_t,
    a: OpaquePointer[MutAnyOrigin],
    b: OpaquePointer[MutAnyOrigin],
    ab_type: DataType,
    c: OpaquePointer[MutAnyOrigin],
    s: OpaquePointer[MutAnyOrigin],
    cs_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotgEx",
        fn(
            cublasHandle_t,
            OpaquePointer[MutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
            DataType,
            OpaquePointer[MutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, a, b, ab_type, c, s, cs_type, executiontype)


@fieldwise_init
struct cublasDiagType_t(TrivialRegisterPassable):
    var _value: Int32
    comptime CUBLAS_DIAG_NON_UNIT = cublasDiagType_t(0)
    comptime CUBLAS_DIAG_UNIT = cublasDiagType_t(1)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid cublasDiagType_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct ComputeType(TrivialRegisterPassable):
    var _value: Int32
    comptime COMPUTE_16F = Self(64)
    comptime COMPUTE_16F_PEDANTIC = Self(65)
    comptime COMPUTE_32F = Self(68)
    comptime COMPUTE_32F_PEDANTIC = Self(69)
    comptime COMPUTE_32F_FAST_16F = Self(74)
    comptime COMPUTE_32F_FAST_16BF = Self(75)
    comptime COMPUTE_32F_FAST_TF32 = Self(77)
    comptime COMPUTE_64F = Self(70)
    comptime COMPUTE_64F_PEDANTIC = Self(71)
    comptime COMPUTE_32I = Self(72)
    comptime COMPUTE_32I_PEDANTIC = Self(73)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid ComputeType entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasDsymm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymm_v2_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    _ap: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasIdamin(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Int64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamin_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Int64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasGetVectorAsync(
    n: Int16,
    elem_size: Int16,
    device_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    host_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVectorAsync",
        fn(
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, device_ptr, incx, host_ptr, incy, stream)


fn cublasGetMatrix(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrix_64",
        fn(
            Int64,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasDaxpy(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDaxpy_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasDsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2k_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSger(
    handle: cublasHandle_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int64,
    _a: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSger_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasSdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int16,
    n: Int16,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSdgmm",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasDtbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbsv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasDtrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float64, MutAnyOrigin],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsm_v2",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasStbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasDspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _ap: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDspmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSswap(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSswap_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _ap: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDspmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSrotmg(
    handle: cublasHandle_t,
    d1: UnsafePointer[Float32, MutAnyOrigin],
    d2: UnsafePointer[Float32, MutAnyOrigin],
    x1: UnsafePointer[Float32, MutAnyOrigin],
    y1: UnsafePointer[Float32, ImmutAnyOrigin],
    param: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotmg_v2",
        fn(
            cublasHandle_t,
            UnsafePointer[Float32, MutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, d1, d2, x1, y1, param)


fn cublasDtpmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDasum(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDasum_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasRotEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    c: OpaquePointer[ImmutAnyOrigin],
    s: OpaquePointer[ImmutAnyOrigin],
    cs_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
    param: UnsafePointer[Float64, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotm_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


fn cublasAxpyEx(
    handle: cublasHandle_t,
    n: Int16,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAxpyEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemm_v2",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSsymm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymm_v2_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasCopyEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasCopyEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasSwapEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSwapEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


fn cublasSrot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
    c: UnsafePointer[Float32, ImmutAnyOrigin],
    s: UnsafePointer[Float32, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrot_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasGetVector(
    n: Int16,
    elem_size: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVector",
        fn(
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(n, elem_size, x, incx, y, incy)


fn cublasDtrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasSsymm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymm_v2",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasDtrmm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int16,
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmm_v2",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


fn cublasCherk3mEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherk3mEx_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


comptime cublasLogCallback = fn(UnsafePointer[Int8, ImmutAnyOrigin]) -> None


fn cublasDtrmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int16,
    n: Int16,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDdgmm",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


fn cublasDtbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbsv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasSsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2k_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasDgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemm_v2",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasGetMathMode(
    handle: cublasHandle_t,
    mode: UnsafePointer[cublasMath_t, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMathMode",
        fn(
            cublasHandle_t,
            UnsafePointer[cublasMath_t, MutAnyOrigin],
        ) -> Result,
    ]()(handle, mode)


fn cublasDrot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
    c: UnsafePointer[Float64, ImmutAnyOrigin],
    s: UnsafePointer[Float64, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrot_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasSspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    _ap: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasGemmEx64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int64,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmEx_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSswap_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDrotm(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
    param: UnsafePointer[Float64, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotm_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


fn cublasSgemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmEx_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemm_v2_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasSsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrk_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDnrm2(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDnrm2_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDasum(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDasum_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDsyrkx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrkx",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasRotmEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    param: OpaquePointer[ImmutAnyOrigin],
    param_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotmEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpsv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int16,
    _ap: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr2_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasSetMatrix(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrix_64",
        fn(
            Int64,
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasDrotg(
    handle: cublasHandle_t,
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    c: UnsafePointer[Float64, MutAnyOrigin],
    s: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotg_v2",
        fn(
            cublasHandle_t,
            UnsafePointer[Float64, MutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, a, b, c, s)


fn cublasGetAtomicsMode(
    handle: cublasHandle_t,
    mode: UnsafePointer[cublasAtomicsMode_t, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetAtomicsMode",
        fn(
            cublasHandle_t,
            UnsafePointer[cublasAtomicsMode_t, MutAnyOrigin],
        ) -> Result,
    ]()(handle, mode)


fn cublasStbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasAxpyEx(
    handle: cublasHandle_t,
    n: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAxpyEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[Int64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIaminEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            UnsafePointer[Int64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasDspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int16,
    _ap: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr2_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasDotEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasScopy_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasDsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrk_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDestroy(handle: cublasHandle_t) raises -> Result:
    return _get_dylib_function[
        "cublasDestroy_v2",
        fn(cublasHandle_t) -> Result,
    ]()(handle)


fn cublasSetVectorAsync(
    n: Int16,
    elem_size: Int16,
    host_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVectorAsync",
        fn(
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, host_ptr, incx, device_ptr, incy, stream)


fn cublasIamaxEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[Int64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIamaxEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            UnsafePointer[Int64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result)


fn cublasSsyrkx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrkx_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasDswap(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDswap_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasAsumEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAsumEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, executiontype)


@fieldwise_init
struct FillMode(TrivialRegisterPassable):
    var _value: Int32
    comptime LOWER = Self(0)
    comptime UPPER = Self(1)
    comptime FULL = Self(2)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid FillMode entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int64,
    _ap: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr2_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasSgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    kl: Int64,
    ku: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgbmv_v2_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasAsumEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAsumEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, executiontype)


fn cublasGetVersion(
    handle: cublasHandle_t,
    version: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVersion_v2",
        fn(cublasHandle_t, UnsafePointer[Int16, MutAnyOrigin]) -> Result,
    ]()(handle, version)


fn cublasScalEx(
    handle: cublasHandle_t,
    n: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasScalEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
            DataType,
        ) -> Result,
    ]()(handle, n, alpha, alpha_type, x, x_type, incx, execution_type)


fn cublasSetPointerMode(
    handle: cublasHandle_t,
    mode: cublasPointerMode_t,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetPointerMode_v2",
        fn(cublasHandle_t, cublasPointerMode_t) -> Result,
    ]()(handle, mode)


fn cublasDgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemv_v2_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasGetStatusString(
    status: Result,
) raises -> UnsafePointer[Int8, ImmutAnyOrigin]:
    return _get_dylib_function[
        "cublasGetStatusString",
        fn(Result) -> UnsafePointer[Int8, ImmutAnyOrigin],
    ]()(status)


fn cublasSnrm2(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSnrm2_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    kl: Int64,
    ku: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgbmv_v2_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int16,
    _a: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDtpsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpsv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSetVector(
    n: Int64,
    elem_size: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVector_64",
        fn(
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(n, elem_size, x, incx, device_ptr, incy)


fn cublasDgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    stride_a: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    stridex: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
    stridey: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemvStridedBatched_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrkx",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasGetStatusName(
    status: Result,
) raises -> UnsafePointer[Int8, ImmutAnyOrigin]:
    return _get_dylib_function[
        "cublasGetStatusName",
        fn(Result) -> UnsafePointer[Int8, ImmutAnyOrigin],
    ]()(status)


fn cublasDtbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


fn cublasSrotg(
    handle: cublasHandle_t,
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    s: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotg_v2",
        fn(
            cublasHandle_t,
            UnsafePointer[Float32, MutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, a, b, c, s)


fn cublasCherkEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherkEx",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


fn cublasDrotmg(
    handle: cublasHandle_t,
    d1: UnsafePointer[Float64, MutAnyOrigin],
    d2: UnsafePointer[Float64, MutAnyOrigin],
    x1: UnsafePointer[Float64, MutAnyOrigin],
    y1: UnsafePointer[Float64, ImmutAnyOrigin],
    param: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotmg_v2",
        fn(
            cublasHandle_t,
            UnsafePointer[Float64, MutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, d1, d2, x1, y1, param)


fn cublasDger(
    handle: cublasHandle_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int16,
    _a: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDger_v2",
        fn(
            cublasHandle_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasSscal(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSscal_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasSetWorkspace(
    handle: cublasHandle_t,
    workspace: OpaquePointer[MutAnyOrigin],
    workspace_size_in_bytes: Int,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetWorkspace_v2",
        fn(cublasHandle_t, OpaquePointer[MutAnyOrigin], Int) -> Result,
    ]()(handle, workspace, workspace_size_in_bytes)


fn cublasStpsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpsv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    _ap: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int32,
    n: Int32,
    k: Int32,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int32,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int32,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int32,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmEx",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int32,
            Int32,
            Int32,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int32,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int32,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsbmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    stride_a: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    stridex: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
    stridey: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemvStridedBatched",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasLoggerConfigure(
    log_is_on: Int16,
    log_to_std_out: Int16,
    log_to_std_err: Int16,
    log_file_name: UnsafePointer[Int8, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasLoggerConfigure",
        fn(Int16, Int16, Int16, UnsafePointer[Int8, ImmutAnyOrigin]) -> Result,
    ]()(log_is_on, log_to_std_out, log_to_std_err, log_file_name)


fn cublasStpmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    stride_a: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    stridex: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int64,
    stridey: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemvStridedBatched_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Int64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamin_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Int64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasDrot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
    c: UnsafePointer[Float64, ImmutAnyOrigin],
    s: UnsafePointer[Float64, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrot_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasDgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int64,
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgeam_64",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


fn cublasGetVectorAsync(
    n: Int64,
    elem_size: Int64,
    device_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    host_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVectorAsync_64",
        fn(
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, device_ptr, incx, host_ptr, incy, stream)


fn cublasStrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float32, MutAnyOrigin],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsm_v2_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasSgemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmEx",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasDtrmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDtrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int64,
    _a: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


fn cublasSrot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
    c: UnsafePointer[Float32, ImmutAnyOrigin],
    s: UnsafePointer[Float32, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrot_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


fn cublasDscal(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDscal_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasCreate(
    handle: UnsafePointer[cublasHandle_t, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasCreate_v2",
        fn(UnsafePointer[cublasHandle_t, MutAnyOrigin]) -> Result,
    ]()(handle)


fn cublasSetSmCountTarget(
    handle: cublasHandle_t, sm_count_target: Int16
) raises -> Result:
    return _get_dylib_function[
        "cublasSetSmCountTarget",
        fn(cublasHandle_t, Int16) -> Result,
    ]()(handle, sm_count_target)


fn cublasDswap(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDswap_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasStrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


fn cublasDspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int64,
    _ap: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr2_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


fn cublasSsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    _a: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


fn cublasNrm2Ex(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasNrm2Ex",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            OpaquePointer[MutAnyOrigin],
            DataType,
            DataType,
        ) -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, execution_type)


fn cublasDtbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


@fieldwise_init
struct cublasAtomicsMode_t(TrivialRegisterPassable):
    var _value: Int32
    comptime CUBLAS_ATOMICS_NOT_ALLOWED = cublasAtomicsMode_t(0)
    comptime CUBLAS_ATOMICS_ALLOWED = cublasAtomicsMode_t(1)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid cublasAtomicsMode_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2k_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasCherk3mEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherk3mEx",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


fn cublasScalEx(
    handle: cublasHandle_t,
    n: Int16,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasScalEx",
        fn(
            cublasHandle_t,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int16,
            DataType,
        ) -> Result,
    ]()(handle, n, alpha, alpha_type, x, x_type, incx, execution_type)


fn cublasDotcEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotcEx_64",
        fn(
            cublasHandle_t,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymm_v2",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


fn cublasIsamax(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamax_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Int16, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSaxpy(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSaxpy_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


fn cublasSnrm2(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSnrm2_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasCherkEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherkEx_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
            DataType,
            Int64,
        ) -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


@fieldwise_init
struct cublasSideMode_t(TrivialRegisterPassable):
    var _value: Int32
    comptime CUBLAS_SIDE_LEFT = cublasSideMode_t(0)
    comptime CUBLAS_SIDE_RIGHT = cublasSideMode_t(1)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid cublasSideMode_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasSetMatrix(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrix",
        fn(
            Int16,
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            Int16,
            OpaquePointer[MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


fn cublasDtrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float64, MutAnyOrigin],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsm_v2_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


fn cublasDcopy(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDcopy_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, n, x, incx, y, incy)


fn cublasSetVectorAsync(
    n: Int64,
    elem_size: Int64,
    host_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVectorAsync_64",
        fn(
            Int64,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            Int64,
            OpaquePointer[MutAnyOrigin],
            Int64,
            CUstream,
        ) -> Result,
    ]()(n, elem_size, host_ptr, incx, device_ptr, incy, stream)


fn cublasDspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    _ap: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


fn cublasSgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemv_v2",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasDtrttp(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    _ap: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrttp",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, uplo, n, _a, lda, _ap)


fn cublasDdot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int64,
    result: UnsafePointer[Float64, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasDdot_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, result)


fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    stride_a: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int16,
    stride_b: Int64,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmStridedBatchedEx",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            DataType,
            Int16,
            Int64,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[MutAnyOrigin],
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
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int64,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int64,
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmm_v2_64",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


fn cublasDsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    _c: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrk_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


fn cublasDscal(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDscal_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, n, alpha, x, incx)


fn cublasDtpmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpmv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


fn cublasSgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    kl: Int16,
    ku: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgbmv_v2",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSrotm(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
    param: UnsafePointer[Float32, ImmutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotm_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, y, incy, param)


fn cublasSetAtomicsMode(
    handle: cublasHandle_t,
    mode: cublasAtomicsMode_t,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetAtomicsMode",
        fn(cublasHandle_t, cublasAtomicsMode_t) -> Result,
    ]()(handle, mode)


fn cublasDsbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsbmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSger(
    handle: cublasHandle_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int16,
    _a: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSger_v2",
        fn(
            cublasHandle_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


fn cublasDsymv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[Float64, ImmutAnyOrigin],
    _a: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int64,
    beta: UnsafePointer[Float64, ImmutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymv_v2_64",
        fn(
            cublasHandle_t,
            FillMode,
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float64, ImmutAnyOrigin],
            UnsafePointer[Float64, MutAnyOrigin],
            Int64,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


fn cublasSetStream(
    handle: cublasHandle_t, stream_id: CUstream
) raises -> Result:
    return _get_dylib_function[
        "cublasSetStream_v2",
        fn(cublasHandle_t, CUstream) -> Result,
    ]()(handle, stream_id)


fn cublasStrmm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int16,
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmm_v2",
        fn(
            cublasHandle_t,
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


@fieldwise_init
struct cublasOperation_t(TrivialRegisterPassable):
    var _value: Int32
    comptime CUBLAS_OP_N = cublasOperation_t(0)
    comptime CUBLAS_OP_T = cublasOperation_t(1)
    comptime CUBLAS_OP_C = cublasOperation_t(2)
    comptime CUBLAS_OP_HERMITAN = cublasOperation_t(2)
    comptime CUBLAS_OP_CONJG = cublasOperation_t(3)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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
        abort("invalid cublasOperation_t entry")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cublasIdamin(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int16,
    result: UnsafePointer[Int16, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamin_v2",
        fn(
            cublasHandle_t,
            Int16,
            UnsafePointer[Float64, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Int16, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)


fn cublasSspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _ap: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int16,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSspmv_v2",
        fn(
            cublasHandle_t,
            FillMode,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int16,
        ) -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    _a: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int16,
    stride_a: Int64,
    _b: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int16,
    stride_b: Int64,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    _c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmStridedBatched",
        fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int16,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
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
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int64,
    result: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasSasum_v2_64",
        fn(
            cublasHandle_t,
            Int64,
            UnsafePointer[Float32, ImmutAnyOrigin],
            Int64,
            UnsafePointer[Float32, MutAnyOrigin],
        ) -> Result,
    ]()(handle, n, x, incx, result)
