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

from std.os import abort
from std.pathlib import Path
from std.ffi import _find_dylib
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle

from max.gpu.host import CUstream

from .dtype import DataType
from .result import Result

comptime cublasContext = NoneType
comptime cublasHandle_t = OptionalPointer[cublasContext, _]

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


def _on_error_msg() -> Error:
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


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUBLAS_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUBLAS_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@inline(.always)
def check_cublas_error(stat: Result) raises:
    if stat != Result.SUCCESS:
        raise Error(t"failed to operate on CUBLAS due to error: {stat}")


@inline(.always)
def check_cublas_error(stat: Result, msg: StringSlice) raises:
    if stat != Result.SUCCESS:
        raise Error(t"{msg}. Got a CUBLAS error: {stat}")


@inline(.always)
def _convert_to_cublas_datatype[mojo_type: DType]() -> DataType:
    comptime match mojo_type:
        case .float32:
            return DataType.R_32F
        case .float16:
            return DataType.R_16F
        case .float8_e4m3fn:
            return DataType.R_8F_E4M3
        case .float8_e5m2:
            return DataType.R_8F_E5M2
        # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
        # Replace this with float4-e2m1fn when GENAI-337 is fixed.
        case .uint8:
            return DataType.R_4F_E2M1
        case .bfloat16:
            return DataType.R_16BF
        case _:
            comptime assert False, (
                "Only support FP32, FP16, BF16, E4M3, E5M2, and E2M1x2 (UInt8)."
                " Please extend it if more types are needed."
            )


@inline(.always)
def _convert_to_cublas_transpose(transpose: Bool) -> cublasOperation_t:
    return (
        cublasOperation_t.CUBLAS_OP_T if transpose else cublasOperation_t.CUBLAS_OP_N
    )


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct cublasMath_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_DEFAULT_MATH = cublasMath_t(0)
    comptime CUBLAS_TENSOR_OP_MATH = cublasMath_t(1)
    comptime CUBLAS_PEDANTIC_MATH = cublasMath_t(2)
    comptime CUBLAS_TF32_TENSOR_OP_MATH = cublasMath_t(3)
    comptime CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = cublasMath_t(4)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_DEFAULT_MATH:
            return writer.write_string("CUBLAS_DEFAULT_MATH")
        if self == Self.CUBLAS_TENSOR_OP_MATH:
            return writer.write_string("CUBLAS_TENSOR_OP_MATH")
        if self == Self.CUBLAS_PEDANTIC_MATH:
            return writer.write_string("CUBLAS_PEDANTIC_MATH")
        if self == Self.CUBLAS_TF32_TENSOR_OP_MATH:
            return writer.write_string("CUBLAS_TF32_TENSOR_OP_MATH")
        if self == Self.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
            return writer.write_string(
                "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION"
            )
        abort("invalid cublasMath_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasSetMathMode(
    handle: cublasHandle_t, mode: cublasMath_t
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMathMode",
        def(type_of(handle), cublasMath_t) thin -> Result,
    ]()(handle, mode)


@fieldwise_init
struct Algorithm(TrivialRegisterPassable, Writable):
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

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.DEFAULT:
            return writer.write_string("DEFAULT")
        if self == Self.ALGO0:
            return writer.write_string("ALGO0")
        if self == Self.ALGO1:
            return writer.write_string("ALGO1")
        if self == Self.ALGO2:
            return writer.write_string("ALGO2")
        if self == Self.ALGO3:
            return writer.write_string("ALGO3")
        if self == Self.ALGO4:
            return writer.write_string("ALGO4")
        if self == Self.ALGO5:
            return writer.write_string("ALGO5")
        if self == Self.ALGO6:
            return writer.write_string("ALGO6")
        if self == Self.ALGO7:
            return writer.write_string("ALGO7")
        if self == Self.ALGO8:
            return writer.write_string("ALGO8")
        if self == Self.ALGO9:
            return writer.write_string("ALGO9")
        if self == Self.ALGO10:
            return writer.write_string("ALGO10")
        if self == Self.ALGO11:
            return writer.write_string("ALGO11")
        if self == Self.ALGO12:
            return writer.write_string("ALGO12")
        if self == Self.ALGO13:
            return writer.write_string("ALGO13")
        if self == Self.ALGO14:
            return writer.write_string("ALGO14")
        if self == Self.ALGO15:
            return writer.write_string("ALGO15")
        if self == Self.ALGO16:
            return writer.write_string("ALGO16")
        if self == Self.ALGO17:
            return writer.write_string("ALGO17")
        if self == Self.ALGO18:
            return writer.write_string("ALGO18")
        if self == Self.ALGO19:
            return writer.write_string("ALGO19")
        if self == Self.ALGO20:
            return writer.write_string("ALGO20")
        if self == Self.ALGO21:
            return writer.write_string("ALGO21")
        if self == Self.ALGO22:
            return writer.write_string("ALGO22")
        if self == Self.ALGO23:
            return writer.write_string("ALGO23")
        if self == Self.DEFAULT_TENSOR_OP:
            return writer.write_string("DEFAULT_TENSOR_OP")
        if self == Self.ALGO0_TENSOR_OP:
            return writer.write_string("ALGO0_TENSOR_OP")
        if self == Self.ALGO1_TENSOR_OP:
            return writer.write_string("ALGO1_TENSOR_OP")
        if self == Self.ALGO2_TENSOR_OP:
            return writer.write_string("ALGO2_TENSOR_OP")
        if self == Self.ALGO3_TENSOR_OP:
            return writer.write_string("ALGO3_TENSOR_OP")
        if self == Self.ALGO4_TENSOR_OP:
            return writer.write_string("ALGO4_TENSOR_OP")
        if self == Self.ALGO5_TENSOR_OP:
            return writer.write_string("ALGO5_TENSOR_OP")
        if self == Self.ALGO6_TENSOR_OP:
            return writer.write_string("ALGO6_TENSOR_OP")
        if self == Self.ALGO7_TENSOR_OP:
            return writer.write_string("ALGO7_TENSOR_OP")
        if self == Self.ALGO8_TENSOR_OP:
            return writer.write_string("ALGO8_TENSOR_OP")
        if self == Self.ALGO9_TENSOR_OP:
            return writer.write_string("ALGO9_TENSOR_OP")
        if self == Self.ALGO10_TENSOR_OP:
            return writer.write_string("ALGO10_TENSOR_OP")
        if self == Self.ALGO11_TENSOR_OP:
            return writer.write_string("ALGO11_TENSOR_OP")
        if self == Self.ALGO12_TENSOR_OP:
            return writer.write_string("ALGO12_TENSOR_OP")
        if self == Self.ALGO13_TENSOR_OP:
            return writer.write_string("ALGO13_TENSOR_OP")
        if self == Self.ALGO14_TENSOR_OP:
            return writer.write_string("ALGO14_TENSOR_OP")
        if self == Self.ALGO15_TENSOR_OP:
            return writer.write_string("ALGO15_TENSOR_OP")
        abort("invalid Algorithm entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct ComputeType(TrivialRegisterPassable, Writable):
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

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.COMPUTE_16F:
            return writer.write_string("COMPUTE_16F")
        if self == Self.COMPUTE_16F_PEDANTIC:
            return writer.write_string("COMPUTE_16F_PEDANTIC")
        if self == Self.COMPUTE_32F:
            return writer.write_string("COMPUTE_32F")
        if self == Self.COMPUTE_32F_PEDANTIC:
            return writer.write_string("COMPUTE_32F_PEDANTIC")
        if self == Self.COMPUTE_32F_FAST_16F:
            return writer.write_string("COMPUTE_32F_FAST_16F")
        if self == Self.COMPUTE_32F_FAST_16BF:
            return writer.write_string("COMPUTE_32F_FAST_16BF")
        if self == Self.COMPUTE_32F_FAST_TF32:
            return writer.write_string("COMPUTE_32F_FAST_TF32")
        if self == Self.COMPUTE_64F:
            return writer.write_string("COMPUTE_64F")
        if self == Self.COMPUTE_64F_PEDANTIC:
            return writer.write_string("COMPUTE_64F_PEDANTIC")
        if self == Self.COMPUTE_32I:
            return writer.write_string("COMPUTE_32I")
        if self == Self.COMPUTE_32I_PEDANTIC:
            return writer.write_string("COMPUTE_32I_PEDANTIC")
        abort("invalid ComputeType entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasDestroy(handle: cublasHandle_t) raises -> Result:
    return _get_dylib_function[
        "cublasDestroy_v2",
        def(type_of(handle)) thin -> Result,
    ]()(handle)


def cublasSetWorkspace(
    handle: cublasHandle_t,
    workspace: OpaquePointer[MutAnyOrigin],
    workspace_size_in_bytes: Int,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetWorkspace_v2",
        def(type_of(handle), type_of(workspace), Int) thin -> Result,
    ]()(handle, workspace, workspace_size_in_bytes)


def cublasGemmEx(
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
    _c: OptionalPointer[NoneType, MutAnyOrigin],
    _ctype: DataType,
    ldc: Int32,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmEx",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int32,
            Int32,
            Int32,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int32,
            type_of(_b),
            DataType,
            Int32,
            type_of(beta),
            type_of(_c),
            DataType,
            Int32,
            ComputeType,
            Algorithm,
        ) thin -> Result,
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


def cublasLoggerConfigure(
    log_is_on: Int16,
    log_to_std_out: Int16,
    log_to_std_err: Int16,
    log_file_name: OptionalPointer[Int8, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasLoggerConfigure",
        def(Int16, Int16, Int16, type_of(log_file_name)) thin -> Result,
    ]()(log_is_on, log_to_std_out, log_to_std_err, log_file_name)


def cublasCreate(
    handle: Pointer[cublasHandle_t[MutAnyOrigin], _],
) raises -> Result:
    return _get_dylib_function[
        "cublasCreate_v2",
        def(type_of(handle)) thin -> Result,
    ]()(handle)


def cublasSetStream(
    handle: cublasHandle_t, stream_id: CUstream
) raises -> Result:
    return _get_dylib_function[
        "cublasSetStream_v2",
        def(type_of(handle), CUstream) thin -> Result,
    ]()(handle, stream_id)


@fieldwise_init
struct cublasOperation_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_OP_N = cublasOperation_t(0)
    comptime CUBLAS_OP_T = cublasOperation_t(1)
    comptime CUBLAS_OP_C = cublasOperation_t(2)
    comptime CUBLAS_OP_HERMITAN = cublasOperation_t(2)
    comptime CUBLAS_OP_CONJG = cublasOperation_t(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_OP_N:
            return writer.write_string("CUBLAS_OP_N")
        if self == Self.CUBLAS_OP_T:
            return writer.write_string("CUBLAS_OP_T")
        if self == Self.CUBLAS_OP_C:
            return writer.write_string("CUBLAS_OP_C")
        if self == Self.CUBLAS_OP_HERMITAN:
            return writer.write_string("CUBLAS_OP_HERMITAN")
        if self == Self.CUBLAS_OP_CONJG:
            return writer.write_string("CUBLAS_OP_CONJG")
        abort("invalid cublasOperation_t entry")

    def __int__(self) -> Int:
        return Int(self._value)
