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

from max.gpu.host import _CUstream_st

from std.utils import StaticTuple

from .cublas import ComputeType
from .dtype import DataType
from .result import Result


comptime cublasLtHandle_t = OptionalPointer[NoneType, _]

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUBLASLT_LIBRARY_PATHS: List[Path] = [
    "libcublasLt.so.13",
    "/usr/local/cuda-13.1/lib64/libcublasLt.so.13",
    "/usr/local/cuda-13.0/lib64/libcublasLt.so.13",
    "/usr/local/cuda/lib64/libcublasLt.so.13",
    "libcublasLt.so.12",
    "/usr/local/cuda-12.8/lib64/libcublasLt.so.12",
    "/usr/local/cuda/lib64/libcublasLt.so.12",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the cuBLASLT libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUBLASLT_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUBLASLT_LIBRARY = _Global[
    "CUDA_CUBLASLT_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUBLASLT_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUBLASLT_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct PointerMode(TrivialRegisterPassable, Writable):
    """UnsafePointer mode to use for alpha/beta ."""

    var _value: Int32
    comptime HOST = PointerMode(0)
    """Pointer targets a single value in host memory.
    """
    comptime DEVICE = PointerMode(1)
    """Pointer targets a single value in device memory.
    """
    comptime DEVICE_VECTOR = PointerMode(2)
    """Pointer targets an array in device memory.
    """
    comptime ALPHA_DEVICE_VECTOR_BETA_ZERO = PointerMode(3)
    """Alpha pointer targets an array in device memory, beta is zero.

    Note: CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE is not supported, must be 0.
    """
    comptime ALPHA_DEVICE_VECTOR_BETA_HOST = PointerMode(4)
    """Alpha pointer targets an array in device memory, beta is a single value in host memory.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case .HOST:
                writer.write_string("HOST")
            case .DEVICE:
                writer.write_string("DEVICE")
            case .DEVICE_VECTOR:
                writer.write_string("DEVICE_VECTOR")
            case .ALPHA_DEVICE_VECTOR_BETA_ZERO:
                writer.write_string("ALPHA_DEVICE_VECTOR_BETA_ZERO")
            case .ALPHA_DEVICE_VECTOR_BETA_HOST:
                writer.write_string("ALPHA_DEVICE_VECTOR_BETA_HOST")
            case _:
                abort("invalid PointerMode entry")

    def __int__(self) -> Int:
        return Int(self._value)


# Opaque descriptor for matrix memory layout
# .
comptime cublasLtMatrixLayout_t = OptionalPointer[MatrixLayout, MutAnyOrigin]

# Opaque descriptor for cublasLtMatrixTransform() operation details
# .
comptime cublasLtMatrixTransformDesc_t = MutPointer[Transform, MutAnyOrigin]


@fieldwise_init
struct Search(TrivialRegisterPassable, Writable):
    """Matmul heuristic search mode
    ."""

    var _value: Int32
    comptime BEST_FIT = Self(0)
    """Ask heuristics for best algo for given usecase.
    """
    comptime LIMITED_BY_ALGO_ID = Self(1)
    """Only try to find best config for preconfigured algo id.
    """
    comptime RESERVED_02 = Self(2)
    """Reserved for future use.
    """
    comptime RESERVED_03 = Self(3)
    """Reserved for future use.
    """
    comptime RESERVED_04 = Self(4)
    """Reserved for future use.
    """
    comptime RESERVED_05 = Self(5)
    """Reserved for future use.
    """
    comptime RESERVED_06 = Self(6)
    """Reserved for future use.
    """
    comptime RESERVED_07 = Self(7)
    """Reserved for future use.
    """
    comptime RESERVED_08 = Self(8)
    """Reserved for future use.
    """
    comptime RESERVED_09 = Self(9)
    """Reserved for future use.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case .BEST_FIT:
                writer.write_string("BEST_FIT")
            case .LIMITED_BY_ALGO_ID:
                writer.write_string("LIMITED_BY_ALGO_ID")
            case .RESERVED_02:
                writer.write_string("RESERVED_02")
            case .RESERVED_03:
                writer.write_string("RESERVED_03")
            case .RESERVED_04:
                writer.write_string("RESERVED_04")
            case .RESERVED_05:
                writer.write_string("RESERVED_05")
            case .RESERVED_06:
                writer.write_string("RESERVED_06")
            case .RESERVED_07:
                writer.write_string("RESERVED_07")
            case .RESERVED_08:
                writer.write_string("RESERVED_08")
            case .RESERVED_09:
                writer.write_string("RESERVED_09")
            case _:
                abort("invalid Search entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct ReductionScheme(TrivialRegisterPassable, Writable):
    """Reduction scheme for portions of the dot-product calculated in parallel (a. k. a. "split - K").
    ."""

    var _value: Int32
    comptime NONE = ReductionScheme(0)
    """No reduction scheme, dot-product shall be performed in one sequence.
    """
    comptime INPLACE = ReductionScheme(1)
    """Reduction is performed "in place" - using the output buffer (and output data type) and counters (in workspace) to
    guarantee the sequentiality.
    """
    comptime COMPUTE_TYPE = ReductionScheme(2)
    """Intermediate results are stored in compute type in the workspace and reduced in a separate step.
    """
    comptime OUTPUT_TYPE = ReductionScheme(4)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """
    comptime MASK = ReductionScheme(0x7)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case .NONE:
                writer.write_string("NONE")
            case .INPLACE:
                writer.write_string("INPLACE")
            case .COMPUTE_TYPE:
                writer.write_string("COMPUTE_TYPE")
            case .OUTPUT_TYPE:
                writer.write_string("OUTPUT_TYPE")
            case .MASK:
                writer.write_string("MASK")
            case _:
                abort("invalid ReductionScheme entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtGetVersion() raises -> Int:
    return _get_dylib_function["cublasLtGetVersion", def() thin -> Int]()()


struct PreferenceOpaque(TrivialRegisterPassable):
    """Semi-opaque descriptor for cublasLtMatmulSelf() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


@fieldwise_init
struct cublasLtMatmulDescAttributes_t(TrivialRegisterPassable, Writable):
    """Matmul descriptor attributes to define details of the operation. ."""

    var _value: Int32
    comptime CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = Self(0)
    """Compute type, see cudaDataType. Defines data type used for multiply and accumulate operations and the
    accumulator during matrix multiplication.

    int32_t.
    """
    comptime CUBLASLT_MATMUL_DESC_SCALE_TYPE = Self(1)
    """Scale type, see cudaDataType. Defines data type of alpha and beta. Accumulator and value from matrix C are
    typically converted to scale type before final scaling. Value is then converted from scale type to type of matrix
    D before being stored in memory.

    int32_t, default: same as CUBLASLT_MATMUL_DESC_COMPUTE_TYPE.
    """
    comptime CUBLASLT_MATMUL_DESC_POINTER_MODE = Self(2)
    """UnsafePointer mode of alpha and beta, see PointerMode. When DEVICE_VECTOR is in use,
    alpha/beta vector lengths must match number of output matrix rows.

    int32_t, default: HOST.
    """
    comptime CUBLASLT_MATMUL_DESC_TRANSA = Self(3)
    """Transform of matrix A, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    comptime CUBLASLT_MATMUL_DESC_TRANSB = Self(4)
    """Transform of matrix B, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    comptime CUBLASLT_MATMUL_DESC_TRANSC = Self(5)
    """Transform of matrix C, see cublasOperation_t.

    Currently only CUBLAS_OP_N is supported.

    int32_t, default: CUBLAS_OP_N.
    """
    comptime CUBLASLT_MATMUL_DESC_FILL_MODE = Self(6)
    """Matrix fill mode, see cublasFillMode_t.

    int32_t, default: CUBLAS_FILL_MODE_FULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE = Self(7)
    """Epilogue function, see Epilogue.

    uint32_t, default: DEFAULT.
    """
    comptime CUBLASLT_MATMUL_DESC_BIAS_POINTER = Self(8)
    """Bias or bias gradient vector pointer in the device memory.

    Bias case. See BIAS.
    For bias data type see CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE.

    Bias vector length must match matrix D rows count.

    Bias gradient case. See DRELU_BGRAD and DGELU_BGRAD.
    Bias gradient vector elements are the same type as the output elements
    (Ctype) with the exception of IMMA kernels (see above).

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Bias case: const void *, default: NULL
    Bias gradient case: void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = Self(10)
    """Batch stride for bias or bias gradient vector.

    Used together with CUBLASLT_MATMUL_DESC_BIAS_POINTER when matrix D's BATCH_COUNT > 1.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = Self(11)
    """UnsafePointer for epilogue auxiliary buffer.

    - Output vector for ReLu bit-mask in forward pass when RELU_AUX
     or RELU_AUX_BIAS epilogue is used.
    - Input vector for ReLu bit-mask in backward pass when
     DRELU_BGRAD epilogue is used.

    - Output of GELU input matrix in forward pass when
     GELU_AUX_BIAS epilogue is used.
    - Input of GELU input matrix for backward pass when
     DGELU_BGRAD epilogue is used.

    For aux data type see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE.

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Requires setting CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD attribute.

    Forward pass: void *, default: NULL
    Backward pass: const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = Self(12)
    """Leading dimension for epilogue auxiliary buffer.

    - ReLu bit-mask matrix leading dimension in elements (i.e. bits)
     when RELU_AUX, RELU_AUX_BIAS or DRELU_BGRAD epilogue is
    used. Must be divisible by 128 and be no less than the number of rows in the output matrix.

    - GELU input matrix leading dimension in elements
     when GELU_AUX_BIAS or DGELU_BGRAD epilogue used.
     Must be divisible by 8 and be no less than the number of rows in the output matrix.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = Self(13)
    """Batch stride for epilogue auxiliary buffer.

    - ReLu bit-mask matrix batch stride in elements (i.e. bits)
     when RELU_AUX, RELU_AUX_BIAS or DRELU_BGRAD epilogue is
    used. Must be divisible by 128.

    - GELU input matrix batch stride in elements
     when GELU_AUX_BIAS or DGELU_BGRAD epilogue used.
     Must be divisible by 8.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = Self(14)
    """Batch stride for alpha vector.

    Used together with ALPHA_DEVICE_VECTOR_BETA_HOST when matrix D's
    BATCH_COUNT > 1. If ALPHA_DEVICE_VECTOR_BETA_ZERO is set then
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE must be set to 0 as this mode doesn't support batched alpha vector.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = Self(15)
    """Number of SMs to target for parallel execution. Optimizes heuristics for execution on a different number of SMs
    when user expects a concurrent stream to be using some of the device resources.

    int32_t, default: 0 - use the number reported by the device.
    """
    comptime CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = Self(17)
    """Device pointer to the scale factor value that converts data in matrix A to the compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = Self(18)
    """Device pointer to the scale factor value to convert data in matrix B to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = Self(19)
    """Device pointer to the scale factor value to convert data in matrix C to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = Self(20)
    """Device pointer to the scale factor value to convert data in matrix D to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = Self(21)
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    output matrix.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = Self(22)
    """Type of the data to be stored to the memory pointed to by CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    If unset, the data type defaults to the type of elements of the output matrix with some exceptions, see details
    below.

    ReLu uses a bit-mask.

    GELU input matrix elements type is the same as the type of elements of
    the output matrix with some exceptions, see details below.

    For fp8 kernels with output type CUDA_R_8F_E4M3 the aux data type can be CUDA_R_8F_E4M3 or CUDA_R_16F with some
    restrictions.  See https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulDescAttributes_t for more details.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    int32_t based on cudaDataType, default: -1.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = Self(23)
    """Device pointer to the scaling factor value to convert results from compute type data range to storage
    data range in the auxiliary matrix that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data,
    scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = Self(24)
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    buffer that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_FAST_ACCUM = Self(25)
    """Flag for managing fp8 fast accumulation mode.
    When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results
    will not periodically be promoted to a higher precision.

    int8_t, default: 0 - fast accumulation mode is disabled.
    """
    comptime CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = Self(26)
    """Type of bias or bias gradient vector in the device memory.

    Bias case: see BIAS.

    Bias vector elements are the same type as the elements of output matrix (Dtype) with the following exceptions:
    - IMMA kernels with computeType=CUDA_R_32I and Ctype=CUDA_R_8I where the bias vector elements
     are the same type as alpha, beta (CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F)
    - fp8 kernels with an output type of CUDA_R_32F, CUDA_R_8F_E4M3 or CUDA_R_8F_E5M2, See
     https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul for details.

    int32_t based on cudaDataType, default: -1.
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = Self(27)
    """EXPERIMENTAL: Number of atomic synchronization chunks in the row dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = Self(28)
    """EXPERIMENTAL: Number of atomic synchronization chunks in the column dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = Self(29)
    """EXPERIMENTAL: UnsafePointer to a device array of input atomic counters consumed by a matmul.

    int32_t *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = Self(30)
    """EXPERIMENTAL: UnsafePointer to a device array of output atomic counters produced by a matmul.

    int32_t *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_A_SCALE_MODE = Self(31)
    """Scaling mode that defines how the matrix scaling factor for matrix A is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_B_SCALE_MODE = Self(32)
    """Scaling mode that defines how the matrix scaling factor for matrix B is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_C_SCALE_MODE = Self(33)
    """Scaling mode that defines how the matrix scaling factor for matrix C is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_D_SCALE_MODE = Self(34)
    """Scaling mode that defines how the matrix scaling factor for matrix D is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE = Self(35)
    """Scaling mode that defines how the matrix scaling factor for the auxiliary matrix is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER = Self(36)
    """Device pointer to the scale factors that are used to convert data in matrix D to the compute data type range.

    The scaling factor value type is defined by the scaling mode (see CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE)

    If set for an unsupported matrix data, scale, scale mode, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    void *, default: NULL
    """

    comptime CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE = Self(37)
    """Scaling mode that defines how the output matrix scaling factor for matrix D is interpreted.

    int32_t, default: 0.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case .CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
                writer.write_string("CUBLASLT_MATMUL_DESC_COMPUTE_TYPE")
            case .CUBLASLT_MATMUL_DESC_SCALE_TYPE:
                writer.write_string("CUBLASLT_MATMUL_DESC_SCALE_TYPE")
            case .CUBLASLT_MATMUL_DESC_POINTER_MODE:
                writer.write_string("CUBLASLT_MATMUL_DESC_POINTER_MODE")
            case .CUBLASLT_MATMUL_DESC_TRANSA:
                writer.write_string("CUBLASLT_MATMUL_DESC_TRANSA")
            case .CUBLASLT_MATMUL_DESC_TRANSB:
                writer.write_string("CUBLASLT_MATMUL_DESC_TRANSB")
            case .CUBLASLT_MATMUL_DESC_TRANSC:
                writer.write_string("CUBLASLT_MATMUL_DESC_TRANSC")
            case .CUBLASLT_MATMUL_DESC_FILL_MODE:
                writer.write_string("CUBLASLT_MATMUL_DESC_FILL_MODE")
            case .CUBLASLT_MATMUL_DESC_EPILOGUE:
                writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE")
            case .CUBLASLT_MATMUL_DESC_BIAS_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_BIAS_POINTER")
            case .CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE:
                writer.write_string("CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE")
            case .CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER")
            case .CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
                writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD")
            case .CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE"
                )
            case .CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE"
                )
            case .CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET:
                writer.write_string("CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET")
            case .CUBLASLT_MATMUL_DESC_A_SCALE_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_A_SCALE_POINTER")
            case .CUBLASLT_MATMUL_DESC_B_SCALE_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_B_SCALE_POINTER")
            case .CUBLASLT_MATMUL_DESC_C_SCALE_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_C_SCALE_POINTER")
            case .CUBLASLT_MATMUL_DESC_D_SCALE_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_D_SCALE_POINTER")
            case .CUBLASLT_MATMUL_DESC_AMAX_D_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_AMAX_D_POINTER")
            case .CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE"
                )
            case .CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER"
                )
            case .CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER"
                )
            case .CUBLASLT_MATMUL_DESC_FAST_ACCUM:
                writer.write_string("CUBLASLT_MATMUL_DESC_FAST_ACCUM")
            case .CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
                writer.write_string("CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE")
            case .CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS"
                )
            case .CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS"
                )
            case .CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER"
                )
            case .CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER"
                )
            case .CUBLASLT_MATMUL_DESC_A_SCALE_MODE:
                writer.write_string("CUBLASLT_MATMUL_DESC_A_SCALE_MODE")
            case .CUBLASLT_MATMUL_DESC_B_SCALE_MODE:
                writer.write_string("CUBLASLT_MATMUL_DESC_B_SCALE_MODE")
            case .CUBLASLT_MATMUL_DESC_C_SCALE_MODE:
                writer.write_string("CUBLASLT_MATMUL_DESC_C_SCALE_MODE")
            case .CUBLASLT_MATMUL_DESC_D_SCALE_MODE:
                writer.write_string("CUBLASLT_MATMUL_DESC_D_SCALE_MODE")
            case .CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE:
                writer.write_string(
                    "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE"
                )
            case .CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER:
                writer.write_string("CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER")
            case .CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE:
                writer.write_string("CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE")
            case _:
                abort("invalid cublasLtMatmulDescAttributes_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtMatrixLayoutDestroy(
    mat_layout: cublasLtMatrixLayout_t,
) raises -> Result:
    """Destroy matrix layout descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutDestroy",
        def(cublasLtMatrixLayout_t) thin -> Result,
    ]()(mat_layout)


# Opaque descriptor for cublasLtMatmul() operation details
# .
comptime cublasLtMatmulDesc_t = OptionalPointer[Descriptor, MutAnyOrigin]

# Opaque descriptor for cublasLtMatmulAlgoGetHeuristic() configuration
# .
comptime cublasLtMatmulPreference_t = OptionalPointer[
    PreferenceOpaque, MutAnyOrigin
]


def cublasLtMatmul(
    light_handle: cublasLtHandle_t,
    compute_desc: cublasLtMatmulDesc_t,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _adesc: cublasLtMatrixLayout_t,
    _b: OptionalPointer[NoneType, ImmutAnyOrigin],
    _bdesc: cublasLtMatrixLayout_t,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OptionalPointer[NoneType, ImmutAnyOrigin],
    _cdesc: cublasLtMatrixLayout_t,
    _d: OptionalPointer[NoneType, MutAnyOrigin],
    _ddesc: cublasLtMatrixLayout_t,
    algo: ImmPointer[MatmulAlgorithm, ImmutAnyOrigin],
    workspace: OpaquePointer[MutAnyOrigin],
    workspace_size_in_bytes: Int,
    stream: _CUstream_st,
) raises -> Result:
    """Execute matrix multiplication (D = alpha * op(A) * op(B) + beta * C).

    \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
    \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
                                               when workspaceSizeInBytes is less than workspace required by configured
                                               algo
    \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
                                               operation
    \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
    \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
    \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmul",
        def(
            type_of(light_handle),
            type_of(compute_desc),
            type_of(alpha),
            type_of(_a),
            type_of(_adesc),
            type_of(_b),
            type_of(_bdesc),
            type_of(beta),
            type_of(_c),
            type_of(_cdesc),
            type_of(_d),
            type_of(_ddesc),
            type_of(algo),
            type_of(workspace),
            type_of(workspace_size_in_bytes),
            type_of(stream),
        ) thin -> Result,
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


def cublasLtMatmulDescSetAttribute(
    matmul_desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: OpaquePointer[ImmutAnyOrigin],
    size_in_bytes: Int,
) raises -> Result:
    """Set matmul operation descriptor attribute.

    matmulDesc   The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescSetAttribute",
        def(
            cublasLtMatmulDesc_t,
            cublasLtMatmulDescAttributes_t,
            OpaquePointer[ImmutAnyOrigin],
            Int,
        ) thin -> Result,
    ]()(matmul_desc, attr, buf, size_in_bytes)


def cublasLtMatmulPreferenceSetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: Preference,
    buf: OpaquePointer[ImmutAnyOrigin],
    size_in_bytes: Int,
) raises -> Result:
    """Set matmul heuristic search preference descriptor attribute.

    pref         The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceSetAttribute",
        def(
            cublasLtMatmulPreference_t,
            Preference,
            OpaquePointer[ImmutAnyOrigin],
            Int,
        ) thin -> Result,
    ]()(pref, attr, buf, size_in_bytes)


@fieldwise_init
struct Preference(TrivialRegisterPassable, Writable):
    """Algo search preference to fine tune the heuristic function. ."""

    var _value: Int32
    comptime SEARCH_MODE = Self(0)
    """Search mode, see Search.

    uint32_t, default: BEST_FIT.
    """
    comptime MAX_WORKSPACE_BYTES = Self(1)
    """Maximum allowed workspace size in bytes.

    uint64_t, default: 0 - no workspace allowed.
    """
    comptime REDUCTION_SCHEME_MASK = Self(3)
    """Reduction scheme mask, see ReductionScheme. Filters heuristic result to only include algo configs that
    use one of the required modes.

    E.g. mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE reduction schemes.

    uint32_t, default: MASK (allows all reduction schemes).
    """
    comptime MIN_ALIGNMENT_A_BYTES = Self(5)
    """Minimum buffer alignment for matrix A (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix A that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MIN_ALIGNMENT_B_BYTES = Self(6)
    """Minimum buffer alignment for matrix B (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix B that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MIN_ALIGNMENT_C_BYTES = Self(7)
    """Minimum buffer alignment for matrix C (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix C that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MIN_ALIGNMENT_D_BYTES = Self(8)
    """Minimum buffer alignment for matrix D (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix D that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MAX_WAVES_COUNT = Self(9)
    """Maximum wave count.

    See cublasLtMatmulHeuristicResult_t::wavesCount.

    Selecting a non-zero value will exclude algorithms that report device utilization higher than specified.

    float, default: 0.0f.
    """
    comptime IMPL_MASK = Self(12)
    """Numerical implementation details mask, see cublasLtNumericalImplFlags_t. Filters heuristic result to only include
    algorithms that use the allowed implementations.

    uint64_t, default: uint64_t(-1) (allow everything).
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case .SEARCH_MODE:
                writer.write_string("SEARCH_MODE")
            case .MAX_WORKSPACE_BYTES:
                writer.write_string("MAX_WORKSPACE_BYTES")
            case .REDUCTION_SCHEME_MASK:
                writer.write_string("REDUCTION_SCHEME_MASK")
            case .MIN_ALIGNMENT_A_BYTES:
                writer.write_string("MIN_ALIGNMENT_A_BYTES")
            case .MIN_ALIGNMENT_B_BYTES:
                writer.write_string("MIN_ALIGNMENT_B_BYTES")
            case .MIN_ALIGNMENT_C_BYTES:
                writer.write_string("MIN_ALIGNMENT_C_BYTES")
            case .MIN_ALIGNMENT_D_BYTES:
                writer.write_string("MIN_ALIGNMENT_D_BYTES")
            case .MAX_WAVES_COUNT:
                writer.write_string("MAX_WAVES_COUNT")
            case .IMPL_MASK:
                writer.write_string("IMPL_MASK")
            case _:
                abort("invalid Preference entry")

    def __int__(self) -> Int:
        return Int(self._value)


struct MatmulAlgorithm(Defaultable, TrivialRegisterPassable):
    """Semi-opaque algorithm descriptor (to avoid complicated alloc/free schemes).

    This structure can be trivially serialized and later restored for use with the same version of cuBLAS library to save
    on selecting the right configuration again.
    """

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]

    def __init__(out self):
        self.data = StaticTuple[UInt64, 8](0)


comptime cublasLtNumericalImplFlags_t = UInt64


def cublasLtMatmulPreferenceDestroy(
    pref: cublasLtMatmulPreference_t,
) raises -> Result:
    """Destroy matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceDestroy",
        def(cublasLtMatmulPreference_t) thin -> Result,
    ]()(pref)


def cublasLtMatmulAlgoGetHeuristic(
    light_handle: cublasLtHandle_t,
    operation_desc: cublasLtMatmulDesc_t,
    _adesc: cublasLtMatrixLayout_t,
    _bdesc: cublasLtMatrixLayout_t,
    _cdesc: cublasLtMatrixLayout_t,
    _ddesc: cublasLtMatrixLayout_t,
    preference: cublasLtMatmulPreference_t,
    requested_algo_count: Int,
    heuristic_results_array: MutPointer[
        cublasLtMatmulHeuristicResult_t, MutAnyOrigin
    ],
    return_algo_count: MutPointer[Int, MutAnyOrigin],
) raises -> Result:
    """Query cublasLt heuristic for algorithm appropriate for given use case.

        lightHandle            UnsafePointer to the allocated cuBLASLt handle for the cuBLASLt
                                          context. See cublasLtHandle_t.
        operationDesc          Handle to the matrix multiplication descriptor.
        Adesc                  Handle to the layout descriptors for matrix A.
        Bdesc                  Handle to the layout descriptors for matrix B.
        Cdesc                  Handle to the layout descriptors for matrix C.
        Ddesc                  Handle to the layout descriptors for matrix D.
        preference             UnsafePointer to the structure holding the heuristic search
                                          preferences descriptor. See cublasLtMatrixLayout_t.
        requestedAlgoCount     Size of heuristicResultsArray (in elements) and requested
                                          maximum number of algorithms to return.
        returnAlgoCount        The number of heuristicResultsArray elements written.

    \retval  CUBLAS_STATUS_INVALID_VALUE   if requestedAlgoCount is less or equal to zero
    \retval  CUBLAS_STATUS_NOT_SUPPORTED   if no heuristic function available for current configuration
    \retval  CUBLAS_STATUS_SUCCESS         if query was successful, inspect
                                          heuristicResultsArray[0 to (returnAlgoCount - 1)].state
                                          for detail status of results
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoGetHeuristic",
        def(
            type_of(light_handle),
            cublasLtMatmulDesc_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatmulPreference_t,
            Int16,
            MutPointer[cublasLtMatmulHeuristicResult_t, MutAnyOrigin],
            MutPointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(
        light_handle,
        operation_desc,
        _adesc,
        _bdesc,
        _cdesc,
        _ddesc,
        preference,
        Int16(requested_algo_count),
        heuristic_results_array,
        return_algo_count,
    )


@fieldwise_init
struct cublasLtMatmulMatrixScale_t(TrivialRegisterPassable, Writable):
    """Scaling mode for per-matrix scaling."""

    var _value: Int32
    comptime MATRIX_SCALE_SCALAR_32F = Self(0)
    """Scaling factors are single precision scalars applied to the whole tensor.
    """
    comptime MATRIX_SCALE_VEC16_UE4M3 = Self(1)
    """Scaling factors are tensors with a dedicated 8-bit CUDA_R_8F_UE4M3 value per 16-element block.

    The scaling factor is stored for each 16-element block in the innermost dimension of the
    corresponding data tensor.
    """
    comptime MATRIX_SCALE_VEC32_UE8M0 = Self(2)
    """Same as VEC16_UE4M3, but with CUDA_R_8F_UE8M0 type and block size of 32 elements.
    """
    comptime MATRIX_SCALE_OUTER_VEC_32F = Self(3)
    """Scaling factors are single-precision vectors.

    This mode is only applicable to matrices A and B, in which case the vectors are expected to
    have M and N elements respectively, and each (i, j)-th element of product of A and B is
    multiplied by i-th element of A scale and j-th element of B scale.
    """
    comptime MATRIX_SCALE_VEC128_32F = Self(4)
    """Scaling factors are tensors with a dedicated FP32 scaling factor per 128-element block.

    The scaling factor is stored for each 128-element block in the innermost dimension of the
    corresponding data tensor.
    """
    comptime MATRIX_SCALE_BLK128x128_32F = Self(5)
    """Scaling factors are tensors with a dedicated FP32 scaling factor per 128x128-element block.
    """
    comptime MATRIX_SCALE_END = Self(6)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case .MATRIX_SCALE_SCALAR_32F:
                writer.write_string("MATRIX_SCALE_SCALAR_32F")
            case .MATRIX_SCALE_VEC16_UE4M3:
                writer.write_string("MATRIX_SCALE_VEC16_UE4M3")
            case .MATRIX_SCALE_VEC32_UE8M0:
                writer.write_string("MATRIX_SCALE_VEC32_UE8M0")
            case .MATRIX_SCALE_OUTER_VEC_32F:
                writer.write_string("MATRIX_SCALE_OUTER_VEC_32F")
            case .MATRIX_SCALE_VEC128_32F:
                writer.write_string("MATRIX_SCALE_VEC128_32F")
            case .MATRIX_SCALE_BLK128x128_32F:
                writer.write_string("MATRIX_SCALE_BLK128x128_32F")
            case .MATRIX_SCALE_END:
                writer.write_string("MATRIX_SCALE_END")
            case _:
                abort("invalid MatmulMatrixScale entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtLoggerSetLevel(level: Int16) raises -> Result:
    """Experimental: Log level setter.

    level                        log level, should be one of the following:
                                            0. Off
                                            1. Errors
                                            2. Performance Trace
                                            3. Performance Hints
                                            4. Heuristics Trace
                                            5. API Trace

    \retval     CUBLAS_STATUS_INVALID_VALUE  if log level is not one of the above levels

    \retval     CUBLAS_STATUS_SUCCESS        if log level was set successfully

    Raises:
        If the dynamic library cannot be found.
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetLevel", def(Int16) thin -> Result
    ]()(level)


def cublasLtMatmulDescDestroy(
    matmul_desc: cublasLtMatmulDesc_t,
) raises -> Result:
    """Destroy matmul operation descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescDestroy",
        def(cublasLtMatmulDesc_t) thin -> Result,
    ]()(matmul_desc)


def cublasLtMatmulAlgoInit(
    light_handle: cublasLtHandle_t,
    compute_type: ComputeType,
    scale_type: DataType,
    _atype: DataType,
    _btype: DataType,
    _ctype: DataType,
    _dtype: DataType,
    algo_id: Int16,
    algo: MutPointer[MatmulAlgorithm, MutAnyOrigin],
) raises -> Result:
    """Initialize algo structure.

    \retval     CUBLAS_STATUS_INVALID_VALUE  if algo is NULL or algoId is outside of recognized range
    \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algoId is not supported for given combination of data types
    \retval     CUBLAS_STATUS_SUCCESS        if the structure was successfully initialized
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoInit",
        def(
            type_of(light_handle),
            ComputeType,
            DataType,
            DataType,
            DataType,
            DataType,
            DataType,
            Int16,
            MutPointer[MatmulAlgorithm, MutAnyOrigin],
        ) thin -> Result,
    ]()(
        light_handle,
        compute_type,
        scale_type,
        _atype,
        _btype,
        _ctype,
        _dtype,
        algo_id,
        algo,
    )


@fieldwise_init
struct Epilogue(TrivialRegisterPassable, Writable):
    """Postprocessing options for the epilogue
    ."""

    var _value: Int32
    comptime DEFAULT = Self(1)
    """No special postprocessing, just scale and quantize results if necessary.
    """
    comptime RELU = Self(2)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).
    """
    comptime RELU_AUX = Self(Self.RELU._value | 128)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime BIAS = Self(4)
    """Bias, apply (broadcasted) Bias from bias vector. Bias vector length must match matrix D rows, it must be packed
    (stride between vector elements is 1). Bias vector is broadcasted to all columns and added before applying final
    postprocessing.
    """
    comptime RELU_BIAS = Self(Self.RELU._value | Self.BIAS._value)
    """ReLu and Bias, apply Bias and then ReLu transform.
    """
    comptime RELU_AUX_BIAS = Self(Self.RELU_AUX._value | Self.BIAS._value)
    """ReLu and Bias, apply Bias and then ReLu transform.

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DRELU = Self(8 | 128)
    """DReLu, apply derivative of ReLu transform.

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DRELU_BGRAD = Self(Self.DRELU._value | 16)
    """DReLu with bias gradient.

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime GELU = Self(32)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
    """
    comptime GELU_AUX = Self(Self.GELU._value | 128)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime GELU_BIAS = Self(Self.GELU._value | Self.BIAS._value)
    """GELU and Bias, apply Bias and then GELU transform.
    """
    comptime GELU_AUX_BIAS = Self(Self.GELU_AUX._value | Self.BIAS._value)
    """GELU and Bias, apply Bias and then GELU transform.

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DGELU = Self(64 | 128)
    """DGELU, apply derivative of GELU transform.

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DGELU_BGRAD = Self(Self.DGELU._value | 16)
    """DGELU with bias gradient.

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime BGRADA = Self(256)
    """Bias gradient based on the input matrix A.

    The bias size corresponds to the number of rows of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """
    comptime BGRADB = Self(512)
    """Bias gradient based on the input matrix B.

    The bias size corresponds to the number of columns of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        match self:
            case .DEFAULT:
                writer.write_string("DEFAULT")
            case .RELU:
                writer.write_string("RELU")
            case .RELU_AUX:
                writer.write_string("RELU_AUX")
            case .BIAS:
                writer.write_string("BIAS")
            case .RELU_BIAS:
                writer.write_string("RELU_BIAS")
            case .RELU_AUX_BIAS:
                writer.write_string("RELU_AUX_BIAS")
            case .DRELU:
                writer.write_string("DRELU")
            case .DRELU_BGRAD:
                writer.write_string("DRELU_BGRAD")
            case .GELU:
                writer.write_string("GELU")
            case .GELU_AUX:
                writer.write_string("GELU_AUX")
            case .GELU_BIAS:
                writer.write_string("GELU_BIAS")
            case .GELU_AUX_BIAS:
                writer.write_string("GELU_AUX_BIAS")
            case .DGELU:
                writer.write_string("DGELU")
            case .DGELU_BGRAD:
                writer.write_string("DGELU_BGRAD")
            case .BGRADA:
                writer.write_string("BGRADA")
            case .BGRADB:
                writer.write_string("BGRADB")
            case _:
                abort("invalid Epilogue entry")

    def __int__(self) -> Int:
        return Int(self._value)


struct Descriptor(TrivialRegisterPassable):
    """Semi-opaque descriptor for cublasLtMatmul() operation details
    ."""

    var data: StaticTuple[UInt64, 32]


def cublasLtMatrixLayoutCreate(
    mat_layout: MutPointer[cublasLtMatrixLayout_t, MutAnyOrigin],
    type: DataType,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) raises -> Result:
    """Create new matrix layout descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if descriptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutCreate",
        def(
            MutPointer[cublasLtMatrixLayout_t, MutAnyOrigin],
            DataType,
            UInt64,
            UInt64,
            Int64,
        ) thin -> Result,
    ]()(mat_layout, type, rows, cols, ld)


struct MatrixLayout(TrivialRegisterPassable):
    """Semi-opaque descriptor for matrix memory layout
    ."""

    var data: StaticTuple[UInt64, 8]


def cublasLtMatmulDescCreate(
    matmul_desc: MutPointer[cublasLtMatmulDesc_t, MutAnyOrigin],
    compute_type: ComputeType,
    scale_type: DataType,
) raises -> Result:
    """Create new matmul operation descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if descriptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescCreate",
        def(
            MutPointer[cublasLtMatmulDesc_t, MutAnyOrigin],
            ComputeType,
            DataType,
        ) thin -> Result,
    ]()(matmul_desc, compute_type, scale_type)


def cublasLtMatmulPreferenceCreate(
    pref: MutPointer[cublasLtMatmulPreference_t, MutAnyOrigin],
) raises -> Result:
    """Create new matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if descriptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceCreate",
        def(
            MutPointer[cublasLtMatmulPreference_t, MutAnyOrigin]
        ) thin -> Result,
    ]()(pref)


struct cublasLtMatmulHeuristicResult_t(Defaultable, TrivialRegisterPassable):
    """Results structure used by cublasLtMatmulGetAlgo.

    Holds returned configured algo descriptor and its runtime properties.
    ."""

    # Matmul algorithm descriptor.
    #
    # Must be initialized with cublasLtMatmulAlgoInit() if preferences' CUBLASLT_MATMUL_PERF_SEARCH_MODE is set to
    # LIMITED_BY_ALGO_ID
    # .
    var algo: MatmulAlgorithm
    # Actual size of workspace memory required.
    # .
    var workspaceSize: Int
    # Result status, other fields are only valid if after call to cublasLtMatmulAlgoGetHeuristic() this member is set to
    # CUBLAS_STATUS_SUCCESS.
    # .
    var state: Result
    # Waves count - a device utilization metric.
    #
    # wavesCount value of 1.0f suggests that when kernel is launched it will fully occupy the GPU.
    # .
    var wavesCount: Float32
    var reserved: StaticTuple[Int32, 4]

    def __init__(out self):
        self.algo = MatmulAlgorithm()
        self.workspaceSize = 0
        self.state = Result.NOT_INITIALIZED
        self.wavesCount = 0.0
        self.reserved = StaticTuple[Int32, 4](0)


def cublasLtMatrixTransform(
    light_handle: cublasLtHandle_t,
    transform_desc: cublasLtMatrixTransformDesc_t,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _adesc: cublasLtMatrixLayout_t,
    beta: OpaquePointer[ImmutAnyOrigin],
    _b: OpaquePointer[ImmutAnyOrigin],
    _bdesc: cublasLtMatrixLayout_t,
    _c: OpaquePointer[MutAnyOrigin],
    _cdesc: cublasLtMatrixLayout_t,
    stream: _CUstream_st,
) raises -> Result:
    """Matrix layout conversion helper (C = alpha * op(A) + beta * op(B)).

    Can be used to change memory order of data or to scale and shift the values.

    \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
    \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
                                               when A is not NULL, but Adesc is NULL
    \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
                                               operation
    \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
    \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
    \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransform",
        def(
            type_of(light_handle),
            cublasLtMatrixTransformDesc_t,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            cublasLtMatrixLayout_t,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            cublasLtMatrixLayout_t,
            OpaquePointer[MutAnyOrigin],
            cublasLtMatrixLayout_t,
            _CUstream_st,
        ) thin -> Result,
    ]()(
        light_handle,
        transform_desc,
        alpha,
        _a,
        _adesc,
        beta,
        _b,
        _bdesc,
        _c,
        _cdesc,
        stream,
    )


struct Transform(TrivialRegisterPassable):
    """Semi-opaque descriptor for cublasLtMatrixTransform() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]
