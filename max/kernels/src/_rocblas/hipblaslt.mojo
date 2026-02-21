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

from gpu.host._amdgpu_hip import hipStream_t

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]
from utils import StaticTuple

comptime hipblasLtHandle_t = OpaquePointer
comptime hipblasLtMatmulDesc_t = OpaquePointer
comptime hipblasLtMatrixLayout_t = OpaquePointer
comptime hipblasLtMatmulPreference_t = OpaquePointer


@fieldwise_init
struct Status(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime SUCCESS = Self(0)
    comptime NOT_INITIALIZED = Self(1)
    comptime ALLOC_FAILED = Self(2)
    comptime INVALID_VALUE = Self(3)
    comptime MAPPING_ERROR = Self(4)
    comptime EXECUTION_FAILED = Self(5)
    comptime INTERNAL_ERROR = Self(6)
    comptime NOT_SUPPORTED = Self(7)
    comptime ARCH_MISMATCH = Self(8)
    comptime HANDLE_IS_NULLPTR = Self(9)
    comptime INVALID_ENUM = Self(10)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        if self == Self.SUCCESS:
            return writer.write("SUCCESS")
        if self == Self.NOT_INITIALIZED:
            return writer.write("NOT_INITIALIZED")
        if self == Self.ALLOC_FAILED:
            return writer.write("ALLOC_FAILED")
        if self == Self.INVALID_VALUE:
            return writer.write("INVALID_VALUE")
        if self == Self.MAPPING_ERROR:
            return writer.write("MAPPING_ERROR")
        if self == Self.EXECUTION_FAILED:
            return writer.write("EXECUTION_FAILED")
        if self == Self.INTERNAL_ERROR:
            return writer.write("INTERNAL_ERROR")
        if self == Self.NOT_SUPPORTED:
            return writer.write("NOT_SUPPORTED")
        if self == Self.ARCH_MISMATCH:
            return writer.write("ARCH_MISMATCH")
        if self == Self.HANDLE_IS_NULLPTR:
            return writer.write("HANDLE_IS_NULLPTR")
        if self == Self.INVALID_ENUM:
            return writer.write("INVALID_ENUM")

        abort("unreachable: invalid Status entry")

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct hipDataType_t(TrivialRegisterPassable):
    var _value: Int32
    comptime R_32F = Self(0)
    comptime R_64F = Self(1)
    comptime R_16F = Self(2)
    comptime R_8I = Self(3)
    comptime R_16BF = Self(14)
    comptime R_8F_E4M3 = Self(28)
    comptime R_8F_E5M2 = Self(29)
    comptime R_8F_E4M3_FNUZ = Self(1000)
    comptime R_8F_E5M2_FNUZ = Self(1001)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasComputeType_t(TrivialRegisterPassable):
    var _value: Int32
    comptime COMPUTE_16F = Self(0)
    comptime COMPUTE_16F_PEDANTIC = Self(1)
    comptime COMPUTE_32F = Self(2)
    comptime COMPUTE_32F_PEDANTIC = Self(3)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasOperation_t(TrivialRegisterPassable):
    var _value: Int32
    comptime OP_N = Self(111)
    comptime OP_T = Self(112)
    comptime OP_C = Self(113)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasLtOrder_t(TrivialRegisterPassable):
    var _value: Int32
    comptime COL = Self(0)
    comptime ROW = Self(1)
    comptime COL16_4R16 = Self(100)
    comptime COL16_4R8 = Self(101)
    comptime COL16_4R4 = Self(102)
    comptime COL16_4R2 = Self(103)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasLtMatmulDescAttributes_t(TrivialRegisterPassable):
    var _value: Int32
    comptime TRANSA = Self(0)
    comptime TRANSB = Self(1)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasLtMatmulLayoutAttribute_t(TrivialRegisterPassable):
    var _value: Int32
    comptime BATCH_COUNT = Self(0)
    comptime STRIDED_BATCH_OFFSET = Self(1)
    comptime TYPE = Self(2)
    comptime ORDER = Self(3)
    comptime ROWS = Self(4)
    comptime COLS = Self(5)
    comptime LD = Self(6)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


struct hipblasLtMatmulAlgo_t(Defaultable, TrivialRegisterPassable):
    var data: StaticTuple[UInt8, 16]
    var maxWorkspaceBytes: Int

    fn __init__(out self):
        self.data = StaticTuple[UInt8, 16](0)
        self.maxWorkspaceBytes = 0


struct hipblasLtMatmulHeuristicResult_t(Defaultable, TrivialRegisterPassable):
    var algo: hipblasLtMatmulAlgo_t
    var workspaceSize: Int
    var state: Status
    var wavesCount: Float32
    var reserved: StaticTuple[Int32, 4]

    fn __init__(out self):
        self.algo = hipblasLtMatmulAlgo_t()
        self.workspaceSize = 0
        self.state = Status.SUCCESS
        self.wavesCount = 1.0
        self.reserved = StaticTuple[Int32, 4](0)


# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime HIPBLASLT_LIBRARY_PATHS: List[Path] = [
    "libhipblaslt.so.0",
    "libhipblaslt.so.1",
    "/opt/rocm/lib/libhipblaslt.so.0",
    "/opt/rocm/lib/libhipblaslt.so.1",
]

comptime HIPBLASLT_LIBRARY = _Global["HIPBLASLT_LIBRARY", _init_dylib]


fn _init_dylib() -> OwnedDLHandle:
    return _find_dylib["HIP BLAS LT"](materialize[HIPBLASLT_LIBRARY_PATHS]())


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: __TypeOfAllTypes
]() raises -> result_type:
    return _ffi_get_dylib_function[
        HIPBLASLT_LIBRARY(), func_name, result_type
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


fn hipblasLtCreate(
    light_handle: UnsafePointer[hipblasLtHandle_t],
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtCreate",
        fn(UnsafePointer[hipblasLtHandle_t]) -> Status,
    ]()(light_handle)


fn hipblasLtDestroy(light_handle: hipblasLtHandle_t) raises -> Status:
    return _get_dylib_function[
        "hipblasLtDestroy", fn(hipblasLtHandle_t) -> Status
    ]()(light_handle)


fn hipblasLtMatmulDescCreate(
    matmul_desc: UnsafePointer[hipblasLtMatmulDesc_t],
    compute_type: hipblasComputeType_t,
    scale_type: hipDataType_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescCreate",
        fn(
            UnsafePointer[hipblasLtMatmulDesc_t],
            hipblasComputeType_t,
            hipDataType_t,
        ) -> Status,
    ]()(matmul_desc, compute_type, scale_type)


fn hipblasLtMatmulDescSetAttribute(
    matmul_desc: hipblasLtMatmulDesc_t,
    attr: hipblasLtMatmulDescAttributes_t,
    buf: OpaquePointer,
    size_in_bytes: Int,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescSetAttribute",
        fn(
            hipblasLtMatmulDesc_t,
            hipblasLtMatmulDescAttributes_t,
            OpaquePointer,
            Int,
        ) -> Status,
    ]()(matmul_desc, attr, buf, size_in_bytes)


fn hipblasLtMatmulDescDestroy(
    matmul_desc: hipblasLtMatmulDesc_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescDestroy", fn(hipblasLtMatmulDesc_t) -> Status
    ]()(matmul_desc)


fn hipblasLtMatrixLayoutCreate(
    mat_layout: UnsafePointer[hipblasLtMatrixLayout_t],
    type: hipDataType_t,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatrixLayoutCreate",
        fn(
            UnsafePointer[hipblasLtMatrixLayout_t],
            hipDataType_t,
            UInt64,
            UInt64,
            Int64,
        ) -> Status,
    ]()(mat_layout, type, rows, cols, ld)


fn hipblasLtMatrixLayoutSetAttribute(
    mat_layout: hipblasLtMatrixLayout_t,
    attr: hipblasLtMatmulLayoutAttribute_t,
    buf: OpaquePointer,
    size_in_bytes: Int,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatrixLayoutSetAttribute",
        fn(
            hipblasLtMatrixLayout_t,
            hipblasLtMatmulLayoutAttribute_t,
            OpaquePointer,
            Int,
        ) -> Status,
    ]()(mat_layout, attr, buf, size_in_bytes)


fn hipblasLtMatrixLayoutDestroy(
    mat_layout: hipblasLtMatrixLayout_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatrixLayoutDestroy", fn(hipblasLtMatrixLayout_t) -> Status
    ]()(mat_layout)


fn hipblasLtMatmulPreferenceCreate(
    pref: UnsafePointer[hipblasLtMatmulPreference_t],
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulPreferenceCreate",
        fn(UnsafePointer[hipblasLtMatmulPreference_t]) -> Status,
    ]()(pref)


fn hipblasLtMatmulAlgoGetHeuristic(
    light_handle: hipblasLtHandle_t,
    operation_desc: hipblasLtMatmulDesc_t,
    _adesc: hipblasLtMatrixLayout_t,
    _bdesc: hipblasLtMatrixLayout_t,
    _cdesc: hipblasLtMatrixLayout_t,
    _ddesc: hipblasLtMatrixLayout_t,
    preference: hipblasLtMatmulPreference_t,
    requested_algo_count: Int,
    heuristic_results_array: UnsafePointer[hipblasLtMatmulHeuristicResult_t],
    return_algo_count: UnsafePointer[Int],
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulAlgoGetHeuristic",
        fn(
            hipblasLtHandle_t,
            hipblasLtMatmulDesc_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatmulPreference_t,
            Int,
            UnsafePointer[hipblasLtMatmulHeuristicResult_t],
            UnsafePointer[Int],
        ) -> Status,
    ]()(
        light_handle,
        operation_desc,
        _adesc,
        _bdesc,
        _cdesc,
        _ddesc,
        preference,
        requested_algo_count,
        heuristic_results_array,
        return_algo_count,
    )


fn hipblasLtMatmulPreferenceDestroy(
    pref: hipblasLtMatmulPreference_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulPreferenceDestroy",
        fn(hipblasLtMatmulPreference_t) -> Status,
    ]()(pref)


fn hipblasLtMatmul(
    light_handle: hipblasLtHandle_t,
    compute_desc: hipblasLtMatmulDesc_t,
    alpha: OpaquePointer,
    _a: OpaquePointer,
    _adesc: hipblasLtMatrixLayout_t,
    _b: OpaquePointer,
    _bdesc: hipblasLtMatrixLayout_t,
    beta: OpaquePointer,
    _c: OpaquePointer,
    _cdesc: hipblasLtMatrixLayout_t,
    _d: OpaquePointer,
    _ddesc: hipblasLtMatrixLayout_t,
    algo: UnsafePointer[hipblasLtMatmulAlgo_t],
    workspace: OpaquePointer,
    workspace_size_in_bytes: Int,
    stream: hipStream_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmul",
        fn(
            hipblasLtHandle_t,
            hipblasLtMatmulDesc_t,
            OpaquePointer,
            OpaquePointer,
            hipblasLtMatrixLayout_t,
            OpaquePointer,
            hipblasLtMatrixLayout_t,
            OpaquePointer,
            OpaquePointer,
            hipblasLtMatrixLayout_t,
            OpaquePointer,
            hipblasLtMatrixLayout_t,
            UnsafePointer[hipblasLtMatmulAlgo_t],
            OpaquePointer,
            Int,
            hipStream_t,
        ) -> Status,
    ]()(
        light_handle,
        compute_desc,
        alpha,
        _a,
        _adesc,
        _b,
        _bdesc,
        beta,
        _c,
        _cdesc,
        _d,
        _ddesc,
        algo,
        workspace,
        workspace_size_in_bytes,
        stream,
    )


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline
fn _check_hipblas_error(status: Status) raises:
    if status != Status.SUCCESS:
        raise Error(String("HIPBLASLT ERROR:", status))


@always_inline
fn _convert_to_hip_datatype[dtype: DType]() -> hipDataType_t:
    comptime if dtype == DType.float32:
        return hipDataType_t.R_32F
    elif dtype == DType.float16:
        return hipDataType_t.R_16F
    elif dtype == DType.float8_e4m3fn:
        return hipDataType_t.R_8F_E4M3
    elif dtype == DType.float8_e5m2:
        return hipDataType_t.R_8F_E5M2
    elif dtype == DType.float8_e4m3fnuz:
        return hipDataType_t.R_8F_E4M3_FNUZ
    elif dtype == DType.float8_e5m2fnuz:
        return hipDataType_t.R_8F_E5M2_FNUZ
    else:
        comptime assert dtype == DType.bfloat16, (
            "Only support FP32, FP16, BF16, E4M3(FNUZ), and E5M2(FNUZ)."
            " Please extend it if more dtypes are needed."
        )
        return hipDataType_t.R_16BF
