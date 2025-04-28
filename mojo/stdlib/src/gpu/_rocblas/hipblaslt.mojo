# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.string import StaticString
from sys.ffi import (
    _get_dylib_function as _ffi_get_dylib_function,
    _Global,
    _OwnedDLHandle,
    _find_dylib,
)

from gpu.host._amdgpu_hip import hipStream_t

alias hipblasLtHandle_t = UnsafePointer[NoneType]
alias hipblasLtMatmulDesc_t = UnsafePointer[NoneType]
alias hipblasLtMatrixLayout_t = UnsafePointer[NoneType]
alias hipblasLtMatmulPreference_t = UnsafePointer[NoneType]


@value
@register_passable("trivial")
struct Status(Writable):
    var _value: Int32
    alias SUCCESS = Self(0)
    alias NOT_INITIALIZED = Self(1)
    alias ALLOC_FAILED = Self(2)
    alias INVALID_VALUE = Self(3)
    alias MAPPING_ERROR = Self(4)
    alias EXECUTION_FAILED = Self(5)
    alias INTERNAL_ERROR = Self(6)
    alias NOT_SUPPORTED = Self(7)
    alias ARCH_MISMATCH = Self(8)
    alias HANDLE_IS_NULLPTR = Self(9)
    alias INVALID_ENUM = Self(10)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
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

        return abort("unreachable: invalid Status entry")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct hipDataType_t:
    var _value: Int32
    alias R_32F = Self(0)
    alias R_64F = Self(1)
    alias R_16F = Self(2)
    alias R_8I = Self(3)
    alias R_16BF = Self(14)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@value
@register_passable("trivial")
struct hipblasComputeType_t:
    var _value: Int32
    alias COMPUTE_16F = Self(0)
    alias COMPUTE_16F_PEDANTIC = Self(1)
    alias COMPUTE_32F = Self(2)
    alias COMPUTE_32F_PEDANTIC = Self(3)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@value
@register_passable("trivial")
struct hipblasOperation_t:
    var _value: Int32
    alias OP_N = Self(111)
    alias OP_T = Self(112)
    alias OP_C = Self(113)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@value
@register_passable("trivial")
struct hipblasLtMatmulDescAttributes_t:
    var _value: Int32
    alias TRANSA = Self(0)
    alias TRANSB = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


@register_passable("trivial")
struct hipblasLtMatmulAlgo_t:
    var data: StaticTuple[UInt8, 16]
    var maxWorkspaceBytes: Int

    fn __init__(out self):
        self.data = StaticTuple[UInt8, 16](0)
        self.maxWorkspaceBytes = 0


@register_passable("trivial")
struct hipblasLtMatmulHeuristicResult_t:
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

alias HIPBLASLT_LIBRARY_PATH = "/opt/rocm/lib/libhipblaslt.so.0"

alias HIPBLASLT_LIBRARY = _Global[
    "HIPBLASLT_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    return _find_dylib["HIP BLAS LT library"](HIPBLASLT_LIBRARY_PATH)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
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
        fn (UnsafePointer[hipblasLtHandle_t]) -> Status,
    ]()(light_handle)


fn hipblasLtDestroy(light_handle: hipblasLtHandle_t) raises -> Status:
    return _get_dylib_function[
        "hipblasLtDestroy", fn (hipblasLtHandle_t) -> Status
    ]()(light_handle)


fn hipblasLtMatmulDescCreate(
    matmul_desc: UnsafePointer[hipblasLtMatmulDesc_t],
    compute_type: hipblasComputeType_t,
    scale_type: hipDataType_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescCreate",
        fn (
            UnsafePointer[hipblasLtMatmulDesc_t],
            hipblasComputeType_t,
            hipDataType_t,
        ) -> Status,
    ]()(matmul_desc, compute_type, scale_type)


fn hipblasLtMatmulDescSetAttribute(
    matmul_desc: hipblasLtMatmulDesc_t,
    attr: hipblasLtMatmulDescAttributes_t,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescSetAttribute",
        fn (
            hipblasLtMatmulDesc_t,
            hipblasLtMatmulDescAttributes_t,
            UnsafePointer[NoneType],
            Int,
        ) -> Status,
    ]()(matmul_desc, attr, buf, size_in_bytes)


fn hipblasLtMatmulDescDestroy(
    matmul_desc: hipblasLtMatmulDesc_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescDestroy", fn (hipblasLtMatmulDesc_t) -> Status
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
        fn (
            UnsafePointer[hipblasLtMatrixLayout_t],
            hipDataType_t,
            UInt64,
            UInt64,
            Int64,
        ) -> Status,
    ]()(mat_layout, type, rows, cols, ld)


fn hipblasLtMatrixLayoutDestroy(
    mat_layout: hipblasLtMatrixLayout_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatrixLayoutDestroy", fn (hipblasLtMatrixLayout_t) -> Status
    ]()(mat_layout)


fn hipblasLtMatmulPreferenceCreate(
    pref: UnsafePointer[hipblasLtMatmulPreference_t],
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulPreferenceCreate",
        fn (UnsafePointer[hipblasLtMatmulPreference_t]) -> Status,
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
        fn (
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
        fn (hipblasLtMatmulPreference_t) -> Status,
    ]()(pref)


fn hipblasLtMatmul(
    light_handle: hipblasLtHandle_t,
    compute_desc: hipblasLtMatmulDesc_t,
    alpha: UnsafePointer[NoneType],
    _a: UnsafePointer[NoneType],
    _adesc: hipblasLtMatrixLayout_t,
    _b: UnsafePointer[NoneType],
    _bdesc: hipblasLtMatrixLayout_t,
    beta: UnsafePointer[NoneType],
    _c: UnsafePointer[NoneType],
    _cdesc: hipblasLtMatrixLayout_t,
    _d: UnsafePointer[NoneType],
    _ddesc: hipblasLtMatrixLayout_t,
    algo: UnsafePointer[hipblasLtMatmulAlgo_t],
    workspace: UnsafePointer[NoneType],
    workspace_size_in_bytes: Int,
    stream: hipStream_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmul",
        fn (
            hipblasLtHandle_t,
            hipblasLtMatmulDesc_t,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            hipblasLtMatrixLayout_t,
            UnsafePointer[NoneType],
            hipblasLtMatrixLayout_t,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            hipblasLtMatrixLayout_t,
            UnsafePointer[NoneType],
            hipblasLtMatrixLayout_t,
            UnsafePointer[hipblasLtMatmulAlgo_t],
            UnsafePointer[NoneType],
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
fn _convert_to_hip_datatype[type: DType]() -> hipDataType_t:
    @parameter
    if type is DType.float32:
        return hipDataType_t.R_32F
    elif type is DType.float16:
        return hipDataType_t.R_16F
    else:
        constrained[
            type is DType.bfloat16,
            (
                "Only support FP32, FP16, BF16. Please extend"
                " it if more types are needed."
            ),
        ]()
        return hipDataType_t.R_16BF
