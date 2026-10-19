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

from max.gpu.host._nvidia_cuda import CUstream

from std.utils import StaticTuple

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUDNN_LIBRARY_PATHS: List[Path] = [
    "libcudnn.so.9",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.9",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the CUDNN libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUDNN_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUDNN_INFER_LIBRARY = _Global[
    "CUDA_CUDNN_INFER_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUDNN_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_INFER_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

# Parameters cannot be inferred from nested types, that's why we use use
# AnyOrigin here, to avoid having to explicitly pass 10+ parameters in large
# function signatures. AnyOrigin will still extend the lifetime of any structs
# that pass their pointers into the functions below. It turns off the check for
# aliasing mutable pointers, but the pointers are passed directly to an external
# library, the data isn't mutated from Mojo. Do not use UntrackedOrigin here, as
# that turns off extending the lifetime of Mojo objects e.g. CuDNNConvMeta in
# conv.mojo destroys the tensor descriptors on last use.
comptime AnyOpaquePointer = OpaquePointer[AnyOrigin[mut=True]]

# This is for calls that have two levels of nesting i.e. 3 UnsafePointers.
# `type` doesn't do anything here, it's just for descriptive purposes, allowing
# the ability to pass the descriptive pointer names below
comptime DoubleNestedPointer[type: AnyType] = Pointer[
    Optional[MutPointer[AnyOpaquePointer, AnyOrigin[mut=True]]], _
]

comptime cudnnContext = AnyOpaquePointer
comptime cudnnTensorStruct = AnyOpaquePointer
comptime cudnnAlgorithmStruct = AnyOpaquePointer
comptime cudnnTensorTransformStruct = AnyOpaquePointer
comptime cudnnSpatialTransformerStruct = AnyOpaquePointer
comptime cudnnDropoutStruct = AnyOpaquePointer
comptime cudnnPoolingStruct = AnyOpaquePointer
comptime cudnnFilterStruct = AnyOpaquePointer
comptime cudnnOpTensorStruct = AnyOpaquePointer
comptime cudnnReduceTensorStruct = AnyOpaquePointer
comptime cudnnLRNStruct = AnyOpaquePointer
comptime cudnnActivationStruct = AnyOpaquePointer
comptime cudnnAlgorithmPerformanceStruct = AnyOpaquePointer
comptime cudnnCTCLossStruct = AnyOpaquePointer
comptime cudnnRuntimeTag_t = NoneType


@fieldwise_init
struct cudnnSoftmaxMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_SOFTMAX_MODE_INSTANCE = Self(0)
    comptime CUDNN_SOFTMAX_MODE_CHANNEL = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_SOFTMAX_MODE_INSTANCE:
            return writer.write_string("CUDNN_SOFTMAX_MODE_INSTANCE")
        if self is Self.CUDNN_SOFTMAX_MODE_CHANNEL:
            return writer.write_string("CUDNN_SOFTMAX_MODE_CHANNEL")
        abort("invalid cudnnSoftmaxMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnSoftmaxMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnDestroyAlgorithmPerformance(
    algo_perf: DoubleNestedPointer[cudnnAlgorithmPerformanceStruct],
    number_to_destroy: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyAlgorithmPerformance",
        def(
            type_of(algo_perf), type_of(number_to_destroy)
        ) thin -> cudnnStatus_t,
    ]()(algo_perf, number_to_destroy)


def cudnnCreate(
    handle: DoubleNestedPointer[cudnnContext],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreate",
        def(type_of(handle)) thin -> cudnnStatus_t,
    ]()(handle)


@fieldwise_init
struct cudnnReduceTensorIndices_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_REDUCE_TENSOR_NO_INDICES = Self(0)
    comptime CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_REDUCE_TENSOR_NO_INDICES:
            return writer.write_string("CUDNN_REDUCE_TENSOR_NO_INDICES")
        if self is Self.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES:
            return writer.write_string("CUDNN_REDUCE_TENSOR_FLATTENED_INDICES")
        abort("invalid cudnnReduceTensorIndices_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnReduceTensorIndices_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnReduceTensor(
    handle: Pointer[cudnnContext, _],
    reduce_tensor_desc: Pointer[cudnnReduceTensorStruct, _],
    indices: OpaquePointer,
    indices_size_in_bytes: Int,
    workspace: OpaquePointer,
    workspace_size_in_bytes: Int,
    alpha: OpaquePointer,
    a_desc: Pointer[cudnnTensorStruct, _],
    _a: OpaquePointer,
    beta: OpaquePointer,
    c_desc: Pointer[cudnnTensorStruct, _],
    _c: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnReduceTensor",
        def(
            type_of(handle),
            type_of(reduce_tensor_desc),
            type_of(indices),
            type_of(indices_size_in_bytes),
            type_of(workspace),
            type_of(workspace_size_in_bytes),
            type_of(alpha),
            type_of(a_desc),
            type_of(_a),
            type_of(beta),
            type_of(c_desc),
            type_of(_c),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        reduce_tensor_desc,
        indices,
        indices_size_in_bytes,
        workspace,
        workspace_size_in_bytes,
        alpha,
        a_desc,
        _a,
        beta,
        c_desc,
        _c,
    )


def cudnnGetActivationDescriptorSwishBeta(
    activation_desc: Pointer[cudnnActivationStruct, _],
    swish_beta: Pointer[Float64, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetActivationDescriptorSwishBeta",
        def(
            type_of(activation_desc), type_of(swish_beta)
        ) thin -> cudnnStatus_t,
    ]()(activation_desc, swish_beta)


def cudnnDestroyAlgorithmDescriptor(
    algo_desc: Pointer[cudnnAlgorithmStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyAlgorithmDescriptor",
        def(type_of(algo_desc)) thin -> cudnnStatus_t,
    ]()(algo_desc)


comptime cudnnTensorTransformDescriptor_t = Pointer[
    cudnnTensorTransformStruct, _
]

comptime cudnnTensorDescriptor_t = Pointer[cudnnTensorStruct, _]


def cudnnDropoutGetReserveSpaceSize(
    xdesc: Pointer[cudnnTensorStruct, _],
    size_in_bytes: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDropoutGetReserveSpaceSize",
        def(type_of(xdesc), type_of(size_in_bytes)) thin -> cudnnStatus_t,
    ]()(xdesc, size_in_bytes)


def cudnnGetReduceTensorDescriptor(
    reduce_tensor_desc: Pointer[cudnnReduceTensorStruct, _],
    reduce_tensor_op: Pointer[cudnnReduceTensorOp_t, _],
    reduce_tensor_comp_type: Pointer[cudnnDataType_t, _],
    reduce_tensor_nan_opt: Pointer[cudnnNanPropagation_t, _],
    reduce_tensor_indices: Pointer[cudnnReduceTensorIndices_t, _],
    reduce_tensor_indices_type: Pointer[cudnnIndicesType_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetReduceTensorDescriptor",
        def(
            type_of(reduce_tensor_desc),
            type_of(reduce_tensor_op),
            type_of(reduce_tensor_comp_type),
            type_of(reduce_tensor_nan_opt),
            type_of(reduce_tensor_indices),
            type_of(reduce_tensor_indices_type),
        ) thin -> cudnnStatus_t,
    ]()(
        reduce_tensor_desc,
        reduce_tensor_op,
        reduce_tensor_comp_type,
        reduce_tensor_nan_opt,
        reduce_tensor_indices,
        reduce_tensor_indices_type,
    )


def cudnnSetPoolingNdDescriptor(
    pooling_desc: Pointer[cudnnPoolingStruct, _],
    mode: cudnnPoolingMode_t,
    maxpooling_nan_opt: cudnnNanPropagation_t,
    nb_dims: Int32,
    window_dim_a: OpaquePointer,
    padding_a: OpaquePointer,
    stride_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetPoolingNdDescriptor",
        def(
            type_of(pooling_desc),
            type_of(mode),
            type_of(maxpooling_nan_opt),
            type_of(nb_dims),
            type_of(window_dim_a),
            type_of(padding_a),
            type_of(stride_a),
        ) thin -> cudnnStatus_t,
    ]()(
        pooling_desc,
        mode,
        maxpooling_nan_opt,
        nb_dims,
        window_dim_a,
        padding_a,
        stride_a,
    )


@fieldwise_init
struct cudnnReduceTensorOp_t(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUDNN_REDUCE_TENSOR_ADD = Self(0)
    comptime CUDNN_REDUCE_TENSOR_MUL = Self(1)
    comptime CUDNN_REDUCE_TENSOR_MIN = Self(2)
    comptime CUDNN_REDUCE_TENSOR_MAX = Self(3)
    comptime CUDNN_REDUCE_TENSOR_AMAX = Self(4)
    comptime CUDNN_REDUCE_TENSOR_AVG = Self(5)
    comptime CUDNN_REDUCE_TENSOR_NORM1 = Self(6)
    comptime CUDNN_REDUCE_TENSOR_NORM2 = Self(7)
    comptime CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = Self(8)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_REDUCE_TENSOR_ADD:
            return writer.write_string("CUDNN_REDUCE_TENSOR_ADD")
        if self is Self.CUDNN_REDUCE_TENSOR_MUL:
            return writer.write_string("CUDNN_REDUCE_TENSOR_MUL")
        if self is Self.CUDNN_REDUCE_TENSOR_MIN:
            return writer.write_string("CUDNN_REDUCE_TENSOR_MIN")
        if self is Self.CUDNN_REDUCE_TENSOR_MAX:
            return writer.write_string("CUDNN_REDUCE_TENSOR_MAX")
        if self is Self.CUDNN_REDUCE_TENSOR_AMAX:
            return writer.write_string("CUDNN_REDUCE_TENSOR_AMAX")
        if self is Self.CUDNN_REDUCE_TENSOR_AVG:
            return writer.write_string("CUDNN_REDUCE_TENSOR_AVG")
        if self is Self.CUDNN_REDUCE_TENSOR_NORM1:
            return writer.write_string("CUDNN_REDUCE_TENSOR_NORM1")
        if self is Self.CUDNN_REDUCE_TENSOR_NORM2:
            return writer.write_string("CUDNN_REDUCE_TENSOR_NORM2")
        if self is Self.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
            return writer.write_string("CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS")
        abort("invalid cudnnReduceTensorOp_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnReduceTensorOp_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSetTensor4dDescriptor(
    tensor_desc: Pointer[cudnnTensorStruct, _],
    format: cudnnTensorFormat_t,
    data_type: cudnnDataType_t,
    n: Int32,
    c: Int32,
    h: Int32,
    w: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensor4dDescriptor",
        def(
            type_of(tensor_desc),
            type_of(format),
            type_of(data_type),
            type_of(n),
            type_of(c),
            type_of(h),
            type_of(w),
        ) thin -> cudnnStatus_t,
    ]()(tensor_desc, format, data_type, n, c, h, w)


def cudnnLRNCrossChannelForward(
    handle: Pointer[cudnnContext, _],
    norm_desc: Pointer[cudnnLRNStruct, _],
    lrn_mode: cudnnLRNMode_t,
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnLRNCrossChannelForward",
        def(
            type_of(handle),
            type_of(norm_desc),
            type_of(lrn_mode),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(handle, norm_desc, lrn_mode, alpha, x_desc, x, beta, y_desc, y)


@fieldwise_init
struct cudnnDeterminism_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_NON_DETERMINISTIC = Self(0)
    comptime CUDNN_DETERMINISTIC = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_NON_DETERMINISTIC:
            return writer.write_string("CUDNN_NON_DETERMINISTIC")
        if self is Self.CUDNN_DETERMINISTIC:
            return writer.write_string("CUDNN_DETERMINISTIC")
        abort("invalid cudnnDeterminism_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnDeterminism_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnAlgorithmDescriptor_t = Pointer[cudnnAlgorithmStruct, _]

comptime cudnnActivationDescriptor_t = Pointer[cudnnActivationStruct, _]


@fieldwise_init
struct cudnnStatus_t(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUDNN_STATUS_SUCCESS = Self(0)
    comptime CUDNN_STATUS_NOT_INITIALIZED = Self(1001)
    comptime CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH = Self(1002)
    comptime CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH = Self(1003)
    comptime CUDNN_STATUS_DEPRECATED = Self(1004)
    comptime CUDNN_STATUS_LICENSE_ERROR = Self(1005)
    comptime CUDNN_STATUS_RUNTIME_IN_PROGRESS = Self(1006)
    comptime CUDNN_STATUS_RUNTIME_FP_OVERFLOW = Self(1007)
    comptime CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED = Self(1008)
    comptime CUDNN_STATUS_BAD_PARAM = Self(2000)
    comptime CUDNN_STATUS_BAD_PARAM_NULL_POINTER = Self(2002)
    comptime CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER = Self(2003)
    comptime CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED = Self(2004)
    comptime CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND = Self(2005)
    comptime CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT = Self(2006)
    comptime CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH = Self(2007)
    comptime CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH = Self(2008)
    comptime CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES = Self(2009)
    comptime CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE = Self(2010)
    comptime CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH = Self(2011)
    comptime CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE = Self(2012)
    comptime CUDNN_STATUS_NOT_SUPPORTED = Self(3000)
    comptime CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN = Self(3001)
    comptime CUDNN_STATUS_NOT_SUPPORTED_SHAPE = Self(3002)
    comptime CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE = Self(3003)
    comptime CUDNN_STATUS_NOT_SUPPORTED_LAYOUT = Self(3004)
    comptime CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER = Self(3005)
    comptime CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART = Self(3006)
    comptime CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH = Self(3007)
    comptime CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING = Self(
        3008
    )
    comptime CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE = Self(3009)
    comptime CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT = Self(3010)
    comptime CUDNN_STATUS_NOT_SUPPORTED_PADDING = Self(3011)
    comptime CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM = Self(3012)
    comptime CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API = Self(3013)
    comptime CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE = Self(3014)
    comptime CUDNN_STATUS_INTERNAL_ERROR = Self(4000)
    comptime CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED = Self(4001)
    comptime CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE = Self(4002)
    comptime CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED = Self(4003)
    comptime CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED = Self(4004)
    comptime CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM = Self(4005)
    comptime CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED = Self(4006)
    comptime CUDNN_STATUS_EXECUTION_FAILED = Self(5000)
    comptime CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER = Self(5001)
    comptime CUDNN_STATUS_EXECUTION_FAILED_CUBLAS = Self(5002)
    comptime CUDNN_STATUS_EXECUTION_FAILED_CUDART = Self(5003)
    comptime CUDNN_STATUS_EXECUTION_FAILED_CURAND = Self(5004)
    comptime CUDNN_STATUS_ALLOC_FAILED = Self(4003)
    comptime CUDNN_STATUS_INVALID_VALUE = Self(2001)
    comptime CUDNN_STATUS_ARCH_MISMATCH = Self(3007)
    comptime CUDNN_STATUS_MAPPING_ERROR = Self(4006)
    comptime CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = Self(3008)
    comptime CUDNN_STATUS_VERSION_MISMATCH = Self(1002)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_STATUS_SUCCESS:
            return writer.write_string("CUDNN_STATUS_SUCCESS")
        if self is Self.CUDNN_STATUS_NOT_INITIALIZED:
            return writer.write_string("CUDNN_STATUS_NOT_INITIALIZED")
        if self is Self.CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH:
            return writer.write_string(
                "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"
            )
        if self is Self.CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH:
            return writer.write_string(
                "CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH"
            )
        if self is Self.CUDNN_STATUS_DEPRECATED:
            return writer.write_string("CUDNN_STATUS_DEPRECATED")
        if self is Self.CUDNN_STATUS_LICENSE_ERROR:
            return writer.write_string("CUDNN_STATUS_LICENSE_ERROR")
        if self is Self.CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return writer.write_string("CUDNN_STATUS_RUNTIME_IN_PROGRESS")
        if self is Self.CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return writer.write_string("CUDNN_STATUS_RUNTIME_FP_OVERFLOW")
        if self is Self.CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED:
            return writer.write_string("CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED")
        if self is Self.CUDNN_STATUS_BAD_PARAM:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM")
        if self is Self.CUDNN_STATUS_BAD_PARAM_NULL_POINTER:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM_NULL_POINTER")
        if self is Self.CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER:
            return writer.write_string(
                "CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER"
            )
        if self is Self.CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED")
        if self is Self.CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND")
        if self is Self.CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT:
            return writer.write_string(
                "CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT"
            )
        if self is Self.CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH")
        if self is Self.CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH")
        if self is Self.CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES:
            return writer.write_string(
                "CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES"
            )
        if self is Self.CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE")
        if self is Self.CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH:
            return writer.write_string(
                "CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH"
            )
        if self is Self.CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE:
            return writer.write_string("CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE")
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED:
            return writer.write_string("CUDNN_STATUS_NOT_SUPPORTED")
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_SHAPE:
            return writer.write_string("CUDNN_STATUS_NOT_SUPPORTED_SHAPE")
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE:
            return writer.write_string("CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE")
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_LAYOUT:
            return writer.write_string("CUDNN_STATUS_NOT_SUPPORTED_LAYOUT")
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_PADDING:
            return writer.write_string("CUDNN_STATUS_NOT_SUPPORTED_PADDING")
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API"
            )
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE:
            return writer.write_string(
                "CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE"
            )
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR:
            return writer.write_string("CUDNN_STATUS_INTERNAL_ERROR")
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED:
            return writer.write_string(
                "CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED"
            )
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE:
            return writer.write_string(
                "CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE"
            )
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED:
            return writer.write_string(
                "CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED"
            )
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED:
            return writer.write_string(
                "CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED"
            )
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM:
            return writer.write_string(
                "CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM"
            )
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED:
            return writer.write_string(
                "CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED"
            )
        if self is Self.CUDNN_STATUS_EXECUTION_FAILED:
            return writer.write_string("CUDNN_STATUS_EXECUTION_FAILED")
        if self is Self.CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER:
            return writer.write_string(
                "CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER"
            )
        if self is Self.CUDNN_STATUS_EXECUTION_FAILED_CUBLAS:
            return writer.write_string("CUDNN_STATUS_EXECUTION_FAILED_CUBLAS")
        if self is Self.CUDNN_STATUS_EXECUTION_FAILED_CUDART:
            return writer.write_string("CUDNN_STATUS_EXECUTION_FAILED_CUDART")
        if self is Self.CUDNN_STATUS_EXECUTION_FAILED_CURAND:
            return writer.write_string("CUDNN_STATUS_EXECUTION_FAILED_CURAND")
        if self is Self.CUDNN_STATUS_ALLOC_FAILED:
            return writer.write_string("CUDNN_STATUS_ALLOC_FAILED")
        if self is Self.CUDNN_STATUS_INVALID_VALUE:
            return writer.write_string("CUDNN_STATUS_INVALID_VALUE")
        if self is Self.CUDNN_STATUS_ARCH_MISMATCH:
            return writer.write_string("CUDNN_STATUS_ARCH_MISMATCH")
        if self is Self.CUDNN_STATUS_MAPPING_ERROR:
            return writer.write_string("CUDNN_STATUS_MAPPING_ERROR")
        if self is Self.CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return writer.write_string(
                "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING"
            )
        if self is Self.CUDNN_STATUS_VERSION_MISMATCH:
            return writer.write_string("CUDNN_STATUS_VERSION_MISMATCH")
        t"cudnnStatus_t(unknown={self._value})".write_to(writer)

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnStatus_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnCTCLossAlgo_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = Self(0)
    comptime CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC:
            return writer.write_string("CUDNN_CTC_LOSS_ALGO_DETERMINISTIC")
        if self is Self.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC:
            return writer.write_string("CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC")
        abort("invalid cudnnCTCLossAlgo_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnCTCLossAlgo_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetFilter4dDescriptor(
    filter_desc: Pointer[cudnnFilterStruct, _],
    data_type: Pointer[cudnnDataType_t, _],
    format: Pointer[cudnnTensorFormat_t, _],
    k: Pointer[Int32, _],
    c: Pointer[Int32, _],
    h: Pointer[Int32, _],
    w: Pointer[Int32, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFilter4dDescriptor",
        def(
            type_of(filter_desc),
            type_of(data_type),
            type_of(format),
            type_of(k),
            type_of(c),
            type_of(h),
            type_of(w),
        ) thin -> cudnnStatus_t,
    ]()(filter_desc, data_type, format, k, c, h, w)


@fieldwise_init
struct cudnnTensorFormat_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_TENSOR_NCHW = Self(0)
    comptime CUDNN_TENSOR_NHWC = Self(1)
    comptime CUDNN_TENSOR_NCHW_VECT_C = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_TENSOR_NCHW:
            return writer.write_string("CUDNN_TENSOR_NCHW")
        if self is Self.CUDNN_TENSOR_NHWC:
            return writer.write_string("CUDNN_TENSOR_NHWC")
        if self is Self.CUDNN_TENSOR_NCHW_VECT_C:
            return writer.write_string("CUDNN_TENSOR_NCHW_VECT_C")
        abort("invalid cudnnTensorFormat_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnTensorFormat_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnAddTensor(
    handle: Pointer[cudnnContext, _],
    alpha: OpaquePointer,
    a_desc: Pointer[cudnnTensorStruct, _],
    _a: OpaquePointer,
    beta: OpaquePointer,
    c_desc: Pointer[cudnnTensorStruct, _],
    _c: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnAddTensor",
        def(
            type_of(handle),
            type_of(alpha),
            type_of(a_desc),
            type_of(_a),
            type_of(beta),
            type_of(c_desc),
            type_of(_c),
        ) thin -> cudnnStatus_t,
    ]()(handle, alpha, a_desc, _a, beta, c_desc, _c)


def cudnnDestroyFilterDescriptor(
    filter_desc: Pointer[cudnnFilterStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyFilterDescriptor",
        def(type_of(filter_desc)) thin -> cudnnStatus_t,
    ]()(filter_desc)


def cudnnGetTensorTransformDescriptor(
    transform_desc: Pointer[cudnnTensorTransformStruct, _],
    nb_dims_requested: UInt32,
    dest_format: Pointer[cudnnTensorFormat_t, _],
    pad_before_a: OpaquePointer,
    pad_after_a: OpaquePointer,
    fold_a: OpaquePointer,
    direction: Pointer[cudnnFoldingDirection_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensorTransformDescriptor",
        def(
            type_of(transform_desc),
            type_of(nb_dims_requested),
            type_of(dest_format),
            type_of(pad_before_a),
            type_of(pad_after_a),
            type_of(fold_a),
            type_of(direction),
        ) thin -> cudnnStatus_t,
    ]()(
        transform_desc,
        nb_dims_requested,
        dest_format,
        pad_before_a,
        pad_after_a,
        fold_a,
        direction,
    )


def cudnnGetVersion() raises -> Int:
    return _get_dylib_function["cudnnGetVersion", def() thin -> Int]()()


def cudnnGetCudartVersion() raises -> Int:
    return _get_dylib_function["cudnnGetCudartVersion", def() thin -> Int]()()


def cudnnGetCallback(
    mask: Pointer[Int32, _],
    udata: Pointer[OpaquePointer[UntrackedOrigin[mut=True]], _],
    fptr: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetCallback",
        def(
            type_of(mask),
            type_of(udata),
            type_of(fptr),
        ) thin -> cudnnStatus_t,
    ]()(mask, udata, fptr)


def cudnnCreateTensorTransformDescriptor(
    transform_desc: DoubleNestedPointer[cudnnTensorTransformStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateTensorTransformDescriptor",
        def(type_of(transform_desc)) thin -> cudnnStatus_t,
    ]()(transform_desc)


def cudnnCreateLRNDescriptor(
    norm_desc: DoubleNestedPointer[cudnnLRNStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateLRNDescriptor",
        def(type_of(norm_desc)) thin -> cudnnStatus_t,
    ]()(norm_desc)


def cudnnSetActivationDescriptor(
    activation_desc: Pointer[cudnnActivationStruct, _],
    mode: cudnnActivationMode_t,
    relu_nan_opt: cudnnNanPropagation_t,
    coef: Float64,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetActivationDescriptor",
        def(
            type_of(activation_desc),
            type_of(mode),
            type_of(relu_nan_opt),
            type_of(coef),
        ) thin -> cudnnStatus_t,
    ]()(activation_desc, mode, relu_nan_opt, coef)


@fieldwise_init
struct cudnnNormAlgo_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_NORM_ALGO_STANDARD = Self(0)
    comptime CUDNN_NORM_ALGO_PERSIST = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_NORM_ALGO_STANDARD:
            return writer.write_string("CUDNN_NORM_ALGO_STANDARD")
        if self is Self.CUDNN_NORM_ALGO_PERSIST:
            return writer.write_string("CUDNN_NORM_ALGO_PERSIST")
        abort("invalid cudnnNormAlgo_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnNormAlgo_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnOpTensorOp_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_OP_TENSOR_ADD = Self(0)
    comptime CUDNN_OP_TENSOR_MUL = Self(1)
    comptime CUDNN_OP_TENSOR_MIN = Self(2)
    comptime CUDNN_OP_TENSOR_MAX = Self(3)
    comptime CUDNN_OP_TENSOR_SQRT = Self(4)
    comptime CUDNN_OP_TENSOR_NOT = Self(5)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_OP_TENSOR_ADD:
            return writer.write_string("CUDNN_OP_TENSOR_ADD")
        if self is Self.CUDNN_OP_TENSOR_MUL:
            return writer.write_string("CUDNN_OP_TENSOR_MUL")
        if self is Self.CUDNN_OP_TENSOR_MIN:
            return writer.write_string("CUDNN_OP_TENSOR_MIN")
        if self is Self.CUDNN_OP_TENSOR_MAX:
            return writer.write_string("CUDNN_OP_TENSOR_MAX")
        if self is Self.CUDNN_OP_TENSOR_SQRT:
            return writer.write_string("CUDNN_OP_TENSOR_SQRT")
        if self is Self.CUDNN_OP_TENSOR_NOT:
            return writer.write_string("CUDNN_OP_TENSOR_NOT")
        abort("invalid cudnnOpTensorOp_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnOpTensorOp_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnCreateReduceTensorDescriptor(
    reduce_tensor_desc: DoubleNestedPointer[cudnnReduceTensorStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateReduceTensorDescriptor",
        def(type_of(reduce_tensor_desc)) thin -> cudnnStatus_t,
    ]()(reduce_tensor_desc)


def cudnnGetPoolingNdForwardOutputDim(
    pooling_desc: Pointer[cudnnPoolingStruct, _],
    input_tensor_desc: Pointer[cudnnTensorStruct, _],
    nb_dims: Int32,
    output_tensor_dim_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPoolingNdForwardOutputDim",
        def(
            type_of(pooling_desc),
            type_of(input_tensor_desc),
            type_of(nb_dims),
            type_of(output_tensor_dim_a),
        ) thin -> cudnnStatus_t,
    ]()(pooling_desc, input_tensor_desc, nb_dims, output_tensor_dim_a)


def cudnnDestroySpatialTransformerDescriptor(
    st_desc: Pointer[cudnnSpatialTransformerStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroySpatialTransformerDescriptor",
        def(type_of(st_desc)) thin -> cudnnStatus_t,
    ]()(st_desc)


comptime cudnnReduceTensorDescriptor_t = Pointer[cudnnReduceTensorStruct, _]


def cudnnCreateTensorDescriptor(
    tensor_desc: DoubleNestedPointer[cudnnTensorStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateTensorDescriptor",
        def(type_of(tensor_desc)) thin -> cudnnStatus_t,
    ]()(tensor_desc)


def cudnnSetOpTensorDescriptor(
    op_tensor_desc: Pointer[cudnnOpTensorStruct, _],
    op_tensor_op: cudnnOpTensorOp_t,
    op_tensor_comp_type: cudnnDataType_t,
    op_tensor_nan_opt: cudnnNanPropagation_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetOpTensorDescriptor",
        def(
            type_of(op_tensor_desc),
            type_of(op_tensor_op),
            type_of(op_tensor_comp_type),
            type_of(op_tensor_nan_opt),
        ) thin -> cudnnStatus_t,
    ]()(op_tensor_desc, op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt)


def cudnnBatchNormalizationForwardInference(
    handle: Pointer[cudnnContext, _],
    mode: cudnnBatchNormMode_t,
    alpha: OpaquePointer,
    beta: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
    bn_scale_bias_mean_var_desc: Pointer[cudnnTensorStruct, _],
    bn_scale: OpaquePointer,
    bn_bias: OpaquePointer,
    estimated_mean: OpaquePointer,
    estimated_variance: OpaquePointer,
    epsilon: Float64,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBatchNormalizationForwardInference",
        def(
            type_of(handle),
            type_of(mode),
            type_of(alpha),
            type_of(beta),
            type_of(x_desc),
            type_of(x),
            type_of(y_desc),
            type_of(y),
            type_of(bn_scale_bias_mean_var_desc),
            type_of(bn_scale),
            type_of(bn_bias),
            type_of(estimated_mean),
            type_of(estimated_variance),
            type_of(epsilon),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        mode,
        alpha,
        beta,
        x_desc,
        x,
        y_desc,
        y,
        bn_scale_bias_mean_var_desc,
        bn_scale,
        bn_bias,
        estimated_mean,
        estimated_variance,
        epsilon,
    )


def cudnnCreateAlgorithmPerformance(
    algo_perf: DoubleNestedPointer[cudnnAlgorithmPerformanceStruct],
    number_to_create: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateAlgorithmPerformance",
        def(
            type_of(algo_perf), type_of(number_to_create)
        ) thin -> cudnnStatus_t,
    ]()(algo_perf, number_to_create)


def cudnnDropoutForward(
    handle: Pointer[cudnnContext, _],
    dropout_desc: Pointer[cudnnDropoutStruct, _],
    xdesc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    ydesc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
    reserve_space: OpaquePointer,
    reserve_space_size_in_bytes: Int,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDropoutForward",
        def(
            type_of(handle),
            type_of(dropout_desc),
            type_of(xdesc),
            type_of(x),
            type_of(ydesc),
            type_of(y),
            type_of(reserve_space),
            type_of(reserve_space_size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        dropout_desc,
        xdesc,
        x,
        ydesc,
        y,
        reserve_space,
        reserve_space_size_in_bytes,
    )


def cudnnDestroy(handle: Pointer[cudnnContext, _]) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroy", def(type_of(handle)) thin -> cudnnStatus_t
    ]()(handle)


def cudnnGetActivationDescriptor(
    activation_desc: Pointer[cudnnActivationStruct, _],
    mode: Pointer[cudnnActivationMode_t, _],
    relu_nan_opt: Pointer[cudnnNanPropagation_t, _],
    coef: Pointer[Float64, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetActivationDescriptor",
        def(
            type_of(activation_desc),
            type_of(mode),
            type_of(relu_nan_opt),
            type_of(coef),
        ) thin -> cudnnStatus_t,
    ]()(activation_desc, mode, relu_nan_opt, coef)


def cudnnOpTensor(
    handle: Pointer[cudnnContext, _],
    op_tensor_desc: Pointer[cudnnOpTensorStruct, _],
    alpha1: OpaquePointer,
    a_desc: Pointer[cudnnTensorStruct, _],
    _a: OpaquePointer,
    alpha2: OpaquePointer,
    b_desc: Pointer[cudnnTensorStruct, _],
    _b: OpaquePointer,
    beta: OpaquePointer,
    c_desc: Pointer[cudnnTensorStruct, _],
    _c: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnOpTensor",
        def(
            type_of(handle),
            type_of(op_tensor_desc),
            type_of(alpha1),
            type_of(a_desc),
            type_of(_a),
            type_of(alpha2),
            type_of(b_desc),
            type_of(_b),
            type_of(beta),
            type_of(c_desc),
            type_of(_c),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        op_tensor_desc,
        alpha1,
        a_desc,
        _a,
        alpha2,
        b_desc,
        _b,
        beta,
        c_desc,
        _c,
    )


def cudnnDeriveBNTensorDescriptor(
    derived_bn_desc: Pointer[cudnnTensorStruct, _],
    x_desc: Pointer[cudnnTensorStruct, _],
    mode: cudnnBatchNormMode_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDeriveBNTensorDescriptor",
        def(
            type_of(derived_bn_desc),
            type_of(x_desc),
            type_of(mode),
        ) thin -> cudnnStatus_t,
    ]()(derived_bn_desc, x_desc, mode)


@fieldwise_init
struct cudnnActivationMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_ACTIVATION_SIGMOID = Self(0)
    comptime CUDNN_ACTIVATION_RELU = Self(1)
    comptime CUDNN_ACTIVATION_TANH = Self(2)
    comptime CUDNN_ACTIVATION_CLIPPED_RELU = Self(3)
    comptime CUDNN_ACTIVATION_ELU = Self(4)
    comptime CUDNN_ACTIVATION_IDENTITY = Self(5)
    comptime CUDNN_ACTIVATION_SWISH = Self(6)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_ACTIVATION_SIGMOID:
            return writer.write_string("CUDNN_ACTIVATION_SIGMOID")
        if self is Self.CUDNN_ACTIVATION_RELU:
            return writer.write_string("CUDNN_ACTIVATION_RELU")
        if self is Self.CUDNN_ACTIVATION_TANH:
            return writer.write_string("CUDNN_ACTIVATION_TANH")
        if self is Self.CUDNN_ACTIVATION_CLIPPED_RELU:
            return writer.write_string("CUDNN_ACTIVATION_CLIPPED_RELU")
        if self is Self.CUDNN_ACTIVATION_ELU:
            return writer.write_string("CUDNN_ACTIVATION_ELU")
        if self is Self.CUDNN_ACTIVATION_IDENTITY:
            return writer.write_string("CUDNN_ACTIVATION_IDENTITY")
        if self is Self.CUDNN_ACTIVATION_SWISH:
            return writer.write_string("CUDNN_ACTIVATION_SWISH")
        abort("invalid cudnnActivationMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnActivationMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSpatialTfGridGeneratorForward(
    handle: Pointer[cudnnContext, _],
    st_desc: Pointer[cudnnSpatialTransformerStruct, _],
    theta: OpaquePointer,
    grid: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSpatialTfGridGeneratorForward",
        def(
            type_of(handle),
            type_of(st_desc),
            type_of(theta),
            type_of(grid),
        ) thin -> cudnnStatus_t,
    ]()(handle, st_desc, theta, grid)


def cudnnGetTensorSizeInBytes(
    tensor_desc: Pointer[cudnnTensorStruct, _],
    size: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensorSizeInBytes",
        def(type_of(tensor_desc), type_of(size)) thin -> cudnnStatus_t,
    ]()(tensor_desc, size)


@fieldwise_init
struct cudnnConvolutionBwdDataAlgo_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = Self(0)
    comptime CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = Self(1)
    comptime CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = Self(2)
    comptime CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = Self(3)
    comptime CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = Self(4)
    comptime CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = Self(5)
    comptime CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = Self(6)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_DATA_ALGO_0")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_DATA_ALGO_1")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
            return writer.write_string(
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
            return writer.write_string(
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
            return writer.write(
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT")
        abort("invalid cudnnConvolutionBwdDataAlgo_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnConvolutionBwdDataAlgo_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetFilterNdDescriptor(
    filter_desc: Pointer[cudnnFilterStruct, _],
    nb_dims_requested: Int32,
    data_type: Pointer[cudnnDataType_t, _],
    format: Pointer[cudnnTensorFormat_t, _],
    nb_dims: Pointer[Int32, _],
    filter_dim_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFilterNdDescriptor",
        def(
            type_of(filter_desc),
            type_of(nb_dims_requested),
            type_of(data_type),
            type_of(format),
            type_of(nb_dims),
            type_of(filter_dim_a),
        ) thin -> cudnnStatus_t,
    ]()(
        filter_desc, nb_dims_requested, data_type, format, nb_dims, filter_dim_a
    )


def cudnnGetPooling2dForwardOutputDim(
    pooling_desc: Pointer[cudnnPoolingStruct, _],
    input_tensor_desc: Pointer[cudnnTensorStruct, _],
    n: Pointer[Int32, _],
    c: Pointer[Int32, _],
    h: Pointer[Int32, _],
    w: Pointer[Int32, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPooling2dForwardOutputDim",
        def(
            type_of(pooling_desc),
            type_of(input_tensor_desc),
            type_of(n),
            type_of(c),
            type_of(h),
            type_of(w),
        ) thin -> cudnnStatus_t,
    ]()(pooling_desc, input_tensor_desc, n, c, h, w)


comptime cudnnLRNDescriptor_t = Pointer[cudnnLRNStruct, _]


@fieldwise_init
struct cudnnSamplerType_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_SAMPLER_BILINEAR = Self(0)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_SAMPLER_BILINEAR:
            return writer.write_string("CUDNN_SAMPLER_BILINEAR")
        abort("invalid cudnnSamplerType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnSamplerType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSpatialTfSamplerForward(
    handle: Pointer[cudnnContext, _],
    st_desc: Pointer[cudnnSpatialTransformerStruct, _],
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    grid: OpaquePointer,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSpatialTfSamplerForward",
        def(
            type_of(handle),
            type_of(st_desc),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(grid),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(handle, st_desc, alpha, x_desc, x, grid, beta, y_desc, y)


@fieldwise_init
struct cudnnNormMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_NORM_PER_ACTIVATION = Self(0)
    comptime CUDNN_NORM_PER_CHANNEL = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_NORM_PER_ACTIVATION:
            return writer.write_string("CUDNN_NORM_PER_ACTIVATION")
        if self is Self.CUDNN_NORM_PER_CHANNEL:
            return writer.write_string("CUDNN_NORM_PER_CHANNEL")
        abort("invalid cudnnNormMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnNormMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSetPooling2dDescriptor(
    pooling_desc: Pointer[cudnnPoolingStruct, _],
    mode: cudnnPoolingMode_t,
    maxpooling_nan_opt: cudnnNanPropagation_t,
    window_height: Int32,
    window_width: Int32,
    vertical_padding: Int32,
    horizontal_padding: Int32,
    vertical_stride: Int32,
    horizontal_stride: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetPooling2dDescriptor",
        def(
            type_of(pooling_desc),
            type_of(mode),
            type_of(maxpooling_nan_opt),
            type_of(window_height),
            type_of(window_width),
            type_of(vertical_padding),
            type_of(horizontal_padding),
            type_of(vertical_stride),
            type_of(horizontal_stride),
        ) thin -> cudnnStatus_t,
    ]()(
        pooling_desc,
        mode,
        maxpooling_nan_opt,
        window_height,
        window_width,
        vertical_padding,
        horizontal_padding,
        vertical_stride,
        horizontal_stride,
    )


def cudnnGetPooling2dDescriptor(
    pooling_desc: Pointer[cudnnPoolingStruct, _],
    mode: Pointer[cudnnPoolingMode_t, _],
    maxpooling_nan_opt: Pointer[cudnnNanPropagation_t, _],
    window_height: Pointer[Int32, _],
    window_width: Pointer[Int32, _],
    vertical_padding: Pointer[Int32, _],
    horizontal_padding: Pointer[Int32, _],
    vertical_stride: Pointer[Int32, _],
    horizontal_stride: Pointer[Int32, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPooling2dDescriptor",
        def(
            type_of(pooling_desc),
            type_of(mode),
            type_of(maxpooling_nan_opt),
            type_of(window_height),
            type_of(window_width),
            type_of(vertical_padding),
            type_of(horizontal_padding),
            type_of(vertical_stride),
            type_of(horizontal_stride),
        ) thin -> cudnnStatus_t,
    ]()(
        pooling_desc,
        mode,
        maxpooling_nan_opt,
        window_height,
        window_width,
        vertical_padding,
        horizontal_padding,
        vertical_stride,
        horizontal_stride,
    )


@fieldwise_init
struct cudnnNormOps_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_NORM_OPS_NORM = Self(0)
    comptime CUDNN_NORM_OPS_NORM_ACTIVATION = Self(1)
    comptime CUDNN_NORM_OPS_NORM_ADD_ACTIVATION = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_NORM_OPS_NORM:
            return writer.write_string("CUDNN_NORM_OPS_NORM")
        if self is Self.CUDNN_NORM_OPS_NORM_ACTIVATION:
            return writer.write_string("CUDNN_NORM_OPS_NORM_ACTIVATION")
        if self is Self.CUDNN_NORM_OPS_NORM_ADD_ACTIVATION:
            return writer.write_string("CUDNN_NORM_OPS_NORM_ADD_ACTIVATION")
        abort("invalid cudnnNormOps_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnNormOps_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSoftmaxForward(
    handle: Pointer[cudnnContext, _],
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSoftmaxForward",
        def(
            type_of(handle),
            type_of(algo),
            type_of(mode),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(handle, algo, mode, alpha, x_desc, x, beta, y_desc, y)


comptime cudnnSpatialTransformerDescriptor_t = Pointer[
    cudnnSpatialTransformerStruct, _
]


@fieldwise_init
struct cudnnSoftmaxAlgorithm_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_SOFTMAX_FAST = Self(0)
    comptime CUDNN_SOFTMAX_ACCURATE = Self(1)
    comptime CUDNN_SOFTMAX_LOG = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_SOFTMAX_FAST:
            return writer.write_string("CUDNN_SOFTMAX_FAST")
        if self is Self.CUDNN_SOFTMAX_ACCURATE:
            return writer.write_string("CUDNN_SOFTMAX_ACCURATE")
        if self is Self.CUDNN_SOFTMAX_LOG:
            return writer.write_string("CUDNN_SOFTMAX_LOG")
        abort("invalid cudnnSoftmaxAlgorithm_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnSoftmaxAlgorithm_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetErrorString(
    status: cudnnStatus_t,
) raises -> ImmPointer[Int8, ImmUntrackedOrigin]:
    return _get_dylib_function[
        "cudnnGetErrorString",
        def(type_of(status),) thin -> ImmPointer[Int8, ImmUntrackedOrigin],
    ]()(status)


def cudnnPoolingForward(
    handle: Pointer[cudnnContext, _],
    pooling_desc: Pointer[cudnnPoolingStruct, _],
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnPoolingForward",
        def(
            type_of(handle),
            type_of(pooling_desc),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(handle, pooling_desc, alpha, x_desc, x, beta, y_desc, y)


def cudnnGetStream(
    handle: Pointer[cudnnContext, _],
    stream_id: Pointer[CUstream, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetStream",
        def(
            type_of(handle),
            type_of(stream_id),
        ) thin -> cudnnStatus_t,
    ]()(handle, stream_id)


@fieldwise_init
struct cudnnBatchNormOps_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_BATCHNORM_OPS_BN = Self(0)
    comptime CUDNN_BATCHNORM_OPS_BN_ACTIVATION = Self(1)
    comptime CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_BATCHNORM_OPS_BN:
            return writer.write_string("CUDNN_BATCHNORM_OPS_BN")
        if self is Self.CUDNN_BATCHNORM_OPS_BN_ACTIVATION:
            return writer.write_string("CUDNN_BATCHNORM_OPS_BN_ACTIVATION")
        if self is Self.CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION:
            return writer.write_string("CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION")
        abort("invalid cudnnBatchNormOps_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBatchNormOps_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnConvolutionFwdAlgo_t(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = Self(0)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = Self(1)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_GEMM = Self(2)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = Self(3)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_FFT = Self(4)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = Self(5)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = Self(6)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = Self(7)
    comptime CUDNN_CONVOLUTION_FWD_ALGO_COUNT = Self(8)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return writer.write_string(
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"
            )
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return writer.write(
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
            )
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return writer.write_string("CUDNN_CONVOLUTION_FWD_ALGO_GEMM")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return writer.write_string("CUDNN_CONVOLUTION_FWD_ALGO_DIRECT")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return writer.write_string("CUDNN_CONVOLUTION_FWD_ALGO_FFT")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return writer.write_string("CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return writer.write_string("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return writer.write_string(
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
            )
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_COUNT:
            return writer.write_string("CUDNN_CONVOLUTION_FWD_ALGO_COUNT")
        abort("invalid cudnnConvolutionFwdAlgo_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnConvolutionFwdAlgo_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSaveAlgorithm(
    handle: Pointer[cudnnContext, _],
    algo_desc: Pointer[cudnnAlgorithmStruct, _],
    algo_space: OpaquePointer,
    algo_space_size_in_bytes: Int,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSaveAlgorithm",
        def(
            type_of(handle),
            type_of(algo_desc),
            type_of(algo_space),
            type_of(algo_space_size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, algo_desc, algo_space, algo_space_size_in_bytes)


def cudnnCopyAlgorithmDescriptor(
    src: Pointer[cudnnAlgorithmStruct, _],
    dest: Pointer[cudnnAlgorithmStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCopyAlgorithmDescriptor",
        def(
            type_of(src),
            type_of(dest),
        ) thin -> cudnnStatus_t,
    ]()(src, dest)


def cudnnDeriveNormTensorDescriptor(
    derived_norm_scale_bias_desc: Pointer[cudnnTensorStruct, _],
    derived_norm_mean_var_desc: Pointer[cudnnTensorStruct, _],
    x_desc: Pointer[cudnnTensorStruct, _],
    mode: cudnnNormMode_t,
    group_cnt: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDeriveNormTensorDescriptor",
        def(
            type_of(derived_norm_scale_bias_desc),
            type_of(derived_norm_mean_var_desc),
            type_of(x_desc),
            type_of(mode),
            type_of(group_cnt),
        ) thin -> cudnnStatus_t,
    ]()(
        derived_norm_scale_bias_desc,
        derived_norm_mean_var_desc,
        x_desc,
        mode,
        group_cnt,
    )


def cudnnTransformFilter(
    handle: Pointer[cudnnContext, _],
    trans_desc: Pointer[cudnnTensorTransformStruct, _],
    alpha: OpaquePointer,
    src_desc: Pointer[cudnnFilterStruct, _],
    src_data: OpaquePointer,
    beta: OpaquePointer,
    dest_desc: Pointer[cudnnFilterStruct, _],
    dest_data: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnTransformFilter",
        def(
            type_of(handle),
            type_of(trans_desc),
            type_of(alpha),
            type_of(src_desc),
            type_of(src_data),
            type_of(beta),
            type_of(dest_desc),
            type_of(dest_data),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        trans_desc,
        alpha,
        src_desc,
        src_data,
        beta,
        dest_desc,
        dest_data,
    )


def cudnnOpsInferVersionCheck() raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnOpsInferVersionCheck", def() thin -> cudnnStatus_t
    ]()()


def cudnnActivationForward(
    handle: Pointer[cudnnContext, _],
    activation_desc: Pointer[cudnnActivationStruct, _],
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnActivationForward",
        def(
            type_of(handle),
            type_of(activation_desc),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y)


def cudnnSetAlgorithmPerformance(
    algo_perf: Pointer[cudnnAlgorithmPerformanceStruct, _],
    algo_desc: Pointer[cudnnAlgorithmStruct, _],
    status: cudnnStatus_t,
    time: Float32,
    memory: Int,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetAlgorithmPerformance",
        def(
            type_of(algo_perf),
            type_of(algo_desc),
            type_of(status),
            type_of(time),
            type_of(memory),
        ) thin -> cudnnStatus_t,
    ]()(algo_perf, algo_desc, status, time, memory)


def cudnnCreateActivationDescriptor(
    activation_desc: DoubleNestedPointer[cudnnActivationStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateActivationDescriptor",
        def(type_of(activation_desc)) thin -> cudnnStatus_t,
    ]()(activation_desc)


@fieldwise_init
struct libraryPropertyType_t(TrivialRegisterPassable):
    var _value: Int32
    comptime MAJOR_VERSION = Self(0)
    comptime MINOR_VERSION = Self(1)
    comptime PATCH_LEVEL = Self(2)


def cudnnGetProperty(
    type: libraryPropertyType_t, value: Pointer[Int32, _]
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetProperty",
        def(type_of(type), type_of(value)) thin -> cudnnStatus_t,
    ]()(type, value)


def cudnnDestroyPoolingDescriptor(
    pooling_desc: Pointer[cudnnPoolingStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyPoolingDescriptor",
        def(type_of(pooling_desc)) thin -> cudnnStatus_t,
    ]()(pooling_desc)


def cudnnGetFilterSizeInBytes(
    filter_desc: Pointer[cudnnFilterStruct, _],
    size: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFilterSizeInBytes",
        def(type_of(filter_desc), type_of(size)) thin -> cudnnStatus_t,
    ]()(filter_desc, size)


@fieldwise_init
struct cudnnLRNMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_LRN_CROSS_CHANNEL_DIM1 = Self(0)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_LRN_CROSS_CHANNEL_DIM1:
            return writer.write_string("CUDNN_LRN_CROSS_CHANNEL_DIM1")
        abort("invalid cudnnLRNMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnLRNMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSetTensorNdDescriptorEx(
    tensor_desc: Pointer[cudnnTensorStruct, _],
    format: cudnnTensorFormat_t,
    data_type: cudnnDataType_t,
    nb_dims: Int32,
    dim_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensorNdDescriptorEx",
        def(
            type_of(tensor_desc),
            type_of(format),
            type_of(data_type),
            type_of(nb_dims),
            type_of(dim_a),
        ) thin -> cudnnStatus_t,
    ]()(tensor_desc, format, data_type, nb_dims, dim_a)


def cudnnSetTensorNdDescriptor(
    tensor_desc: Pointer[cudnnTensorStruct, _],
    data_type: cudnnDataType_t,
    nb_dims: Int32,
    dim_a: OpaquePointer,
    stride_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensorNdDescriptor",
        def(
            type_of(tensor_desc),
            type_of(data_type),
            type_of(nb_dims),
            type_of(dim_a),
            type_of(stride_a),
        ) thin -> cudnnStatus_t,
    ]()(tensor_desc, data_type, nb_dims, dim_a, stride_a)


def cudnnTransformTensorEx(
    handle: Pointer[cudnnContext, _],
    trans_desc: Pointer[cudnnTensorTransformStruct, _],
    alpha: OpaquePointer,
    src_desc: Pointer[cudnnTensorStruct, _],
    src_data: OpaquePointer,
    beta: OpaquePointer,
    dest_desc: Pointer[cudnnTensorStruct, _],
    dest_data: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnTransformTensorEx",
        def(
            type_of(handle),
            type_of(trans_desc),
            type_of(alpha),
            type_of(src_desc),
            type_of(src_data),
            type_of(beta),
            type_of(dest_desc),
            type_of(dest_data),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        trans_desc,
        alpha,
        src_desc,
        src_data,
        beta,
        dest_desc,
        dest_data,
    )


def cudnnGetAlgorithmDescriptor(
    algo_desc: Pointer[cudnnAlgorithmStruct, _],
    algorithm: Pointer[cudnnAlgorithmUnionStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAlgorithmDescriptor",
        def(
            type_of(algo_desc),
            type_of(algorithm),
        ) thin -> cudnnStatus_t,
    ]()(algo_desc, algorithm)


@fieldwise_init
struct cudnnFoldingDirection_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_TRANSFORM_FOLD = Self(0)
    comptime CUDNN_TRANSFORM_UNFOLD = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_TRANSFORM_FOLD:
            return writer.write_string("CUDNN_TRANSFORM_FOLD")
        if self is Self.CUDNN_TRANSFORM_UNFOLD:
            return writer.write_string("CUDNN_TRANSFORM_UNFOLD")
        abort("invalid cudnnFoldingDirection_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnFoldingDirection_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetTensorNdDescriptor(
    tensor_desc: Pointer[cudnnTensorStruct, _],
    nb_dims_requested: Int32,
    data_type: Pointer[cudnnDataType_t, _],
    nb_dims: Pointer[Int32, _],
    dim_a: OpaquePointer,
    stride_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensorNdDescriptor",
        def(
            type_of(tensor_desc),
            type_of(nb_dims_requested),
            type_of(data_type),
            type_of(nb_dims),
            type_of(dim_a),
            type_of(stride_a),
        ) thin -> cudnnStatus_t,
    ]()(tensor_desc, nb_dims_requested, data_type, nb_dims, dim_a, stride_a)


@fieldwise_init
struct cudnnErrQueryMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_ERRQUERY_RAWCODE = Self(0)
    comptime CUDNN_ERRQUERY_NONBLOCKING = Self(1)
    comptime CUDNN_ERRQUERY_BLOCKING = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_ERRQUERY_RAWCODE:
            return writer.write_string("CUDNN_ERRQUERY_RAWCODE")
        if self is Self.CUDNN_ERRQUERY_NONBLOCKING:
            return writer.write_string("CUDNN_ERRQUERY_NONBLOCKING")
        if self is Self.CUDNN_ERRQUERY_BLOCKING:
            return writer.write_string("CUDNN_ERRQUERY_BLOCKING")
        abort("invalid cudnnErrQueryMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnErrQueryMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetOpTensorDescriptor(
    op_tensor_desc: Pointer[cudnnOpTensorStruct, _],
    op_tensor_op: Pointer[cudnnOpTensorOp_t, _],
    op_tensor_comp_type: Pointer[cudnnDataType_t, _],
    op_tensor_nan_opt: Pointer[cudnnNanPropagation_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetOpTensorDescriptor",
        def(
            type_of(op_tensor_desc),
            type_of(op_tensor_op),
            type_of(op_tensor_comp_type),
            type_of(op_tensor_nan_opt),
        ) thin -> cudnnStatus_t,
    ]()(op_tensor_desc, op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt)


def cudnnGetReductionIndicesSize(
    handle: Pointer[cudnnContext, _],
    reduce_tensor_desc: Pointer[cudnnReduceTensorStruct, _],
    a_desc: Pointer[cudnnTensorStruct, _],
    c_desc: Pointer[cudnnTensorStruct, _],
    size_in_bytes: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetReductionIndicesSize",
        def(
            type_of(handle),
            type_of(reduce_tensor_desc),
            type_of(a_desc),
            type_of(c_desc),
            type_of(size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, reduce_tensor_desc, a_desc, c_desc, size_in_bytes)


def cudnnTransformTensor(
    handle: Pointer[cudnnContext, _],
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnTransformTensor",
        def(
            type_of(handle),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(handle, alpha, x_desc, x, beta, y_desc, y)


comptime cudnnCallback_t = def(
    cudnnSeverity_t,
    OpaquePointer,
    Pointer[cudnnDebugStruct, _],
    Pointer[Int8, _],
) thin -> NoneType


struct cudnnAlgorithmUnionStruct(TrivialRegisterPassable):
    @__allow_legacy_any_origin_fields
    var algo: OpaquePointer[MutAnyOrigin]


comptime cudnnDropoutDescriptor_t = Pointer[cudnnDropoutStruct, _]


def cudnnSetTensor4dDescriptorEx(
    tensor_desc: Pointer[cudnnTensorStruct, _],
    data_type: cudnnDataType_t,
    n: Int32,
    c: Int32,
    h: Int32,
    w: Int32,
    n_stride: Int32,
    c_stride: Int32,
    h_stride: Int32,
    w_stride: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensor4dDescriptorEx",
        def(
            type_of(tensor_desc),
            type_of(data_type),
            type_of(n),
            type_of(c),
            type_of(h),
            type_of(w),
            type_of(n_stride),
            type_of(c_stride),
            type_of(h_stride),
            type_of(w_stride),
        ) thin -> cudnnStatus_t,
    ]()(
        tensor_desc,
        data_type,
        n,
        c,
        h,
        w,
        n_stride,
        c_stride,
        h_stride,
        w_stride,
    )


@fieldwise_init
struct cudnnBatchNormMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_BATCHNORM_PER_ACTIVATION = Self(0)
    comptime CUDNN_BATCHNORM_SPATIAL = Self(1)
    comptime CUDNN_BATCHNORM_SPATIAL_PERSISTENT = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_BATCHNORM_PER_ACTIVATION:
            return writer.write_string("CUDNN_BATCHNORM_PER_ACTIVATION")
        if self is Self.CUDNN_BATCHNORM_SPATIAL:
            return writer.write_string("CUDNN_BATCHNORM_SPATIAL")
        if self is Self.CUDNN_BATCHNORM_SPATIAL_PERSISTENT:
            return writer.write_string("CUDNN_BATCHNORM_SPATIAL_PERSISTENT")
        abort("invalid cudnnBatchNormMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBatchNormMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnCTCLossDescriptor_t = Pointer[cudnnCTCLossStruct, _]


def cudnnGetLRNDescriptor(
    norm_desc: Pointer[cudnnLRNStruct, _],
    lrn_n: Pointer[Int32, _],
    lrn_alpha: Pointer[Float64, _],
    lrn_beta: Pointer[Float64, _],
    lrn_k: Pointer[Float64, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetLRNDescriptor",
        def(
            type_of(norm_desc),
            type_of(lrn_n),
            type_of(lrn_alpha),
            type_of(lrn_beta),
            type_of(lrn_k),
        ) thin -> cudnnStatus_t,
    ]()(norm_desc, lrn_n, lrn_alpha, lrn_beta, lrn_k)


comptime cudnnAlgorithmPerformance_t = Pointer[
    cudnnAlgorithmPerformanceStruct, _
]


def cudnnScaleTensor(
    handle: Pointer[cudnnContext, _],
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
    alpha: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnScaleTensor",
        def(
            type_of(handle),
            type_of(y_desc),
            type_of(y),
            type_of(alpha),
        ) thin -> cudnnStatus_t,
    ]()(handle, y_desc, y, alpha)


@fieldwise_init
struct cudnnSeverity_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_SEV_FATAL = Self(0)
    comptime CUDNN_SEV_ERROR = Self(1)
    comptime CUDNN_SEV_WARNING = Self(2)
    comptime CUDNN_SEV_INFO = Self(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_SEV_FATAL:
            return writer.write_string("CUDNN_SEV_FATAL")
        if self is Self.CUDNN_SEV_ERROR:
            return writer.write_string("CUDNN_SEV_ERROR")
        if self is Self.CUDNN_SEV_WARNING:
            return writer.write_string("CUDNN_SEV_WARNING")
        if self is Self.CUDNN_SEV_INFO:
            return writer.write_string("CUDNN_SEV_INFO")
        abort("invalid cudnnSeverity_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnSeverity_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnDebug_t = cudnnDebugStruct


@fieldwise_init
struct cudnnMathType_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_DEFAULT_MATH = Self(0)
    comptime CUDNN_TENSOR_OP_MATH = Self(1)
    comptime CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = Self(2)
    comptime CUDNN_FMA_MATH = Self(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_DEFAULT_MATH:
            return writer.write_string("CUDNN_DEFAULT_MATH")
        if self is Self.CUDNN_TENSOR_OP_MATH:
            return writer.write_string("CUDNN_TENSOR_OP_MATH")
        if self is Self.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
            return writer.write_string("CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION")
        if self is Self.CUDNN_FMA_MATH:
            return writer.write_string("CUDNN_FMA_MATH")
        abort("invalid cudnnMathType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnMathType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnNanPropagation_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_NOT_PROPAGATE_NAN = Self(0)
    comptime CUDNN_PROPAGATE_NAN = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_NOT_PROPAGATE_NAN:
            return writer.write_string("CUDNN_NOT_PROPAGATE_NAN")
        if self is Self.CUDNN_PROPAGATE_NAN:
            return writer.write_string("CUDNN_PROPAGATE_NAN")
        abort("invalid cudnnNanPropagation_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnNanPropagation_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnFilterDescriptor_t = Pointer[cudnnFilterStruct, _]


@fieldwise_init
struct cudnnRNNAlgo_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_RNN_ALGO_STANDARD = Self(0)
    comptime CUDNN_RNN_ALGO_PERSIST_STATIC = Self(1)
    comptime CUDNN_RNN_ALGO_PERSIST_DYNAMIC = Self(2)
    comptime CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H = Self(3)
    comptime CUDNN_RNN_ALGO_COUNT = Self(4)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_RNN_ALGO_STANDARD:
            return writer.write_string("CUDNN_RNN_ALGO_STANDARD")
        if self is Self.CUDNN_RNN_ALGO_PERSIST_STATIC:
            return writer.write_string("CUDNN_RNN_ALGO_PERSIST_STATIC")
        if self is Self.CUDNN_RNN_ALGO_PERSIST_DYNAMIC:
            return writer.write_string("CUDNN_RNN_ALGO_PERSIST_DYNAMIC")
        if self is Self.CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H:
            return writer.write_string("CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H")
        if self is Self.CUDNN_RNN_ALGO_COUNT:
            return writer.write_string("CUDNN_RNN_ALGO_COUNT")
        abort("invalid cudnnRNNAlgo_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnRNNAlgo_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnOpTensorDescriptor_t = Pointer[cudnnOpTensorStruct, _]


struct Algorithm(TrivialRegisterPassable):
    var convFwdAlgo: cudnnConvolutionFwdAlgo_t
    var convBwdFilterAlgo: cudnnConvolutionBwdFilterAlgo_t
    var convBwdDataAlgo: cudnnConvolutionBwdDataAlgo_t
    var RNNAlgo: cudnnRNNAlgo_t
    var CTCLossAlgo: cudnnCTCLossAlgo_t


def cudnnGetReductionWorkspaceSize(
    handle: Pointer[cudnnContext, _],
    reduce_tensor_desc: Pointer[cudnnReduceTensorStruct, _],
    a_desc: Pointer[cudnnTensorStruct, _],
    c_desc: Pointer[cudnnTensorStruct, _],
    size_in_bytes: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetReductionWorkspaceSize",
        def(
            type_of(handle),
            type_of(reduce_tensor_desc),
            type_of(a_desc),
            type_of(c_desc),
            type_of(size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, reduce_tensor_desc, a_desc, c_desc, size_in_bytes)


def cudnnSetFilter4dDescriptor(
    filter_desc: Pointer[cudnnFilterStruct, _],
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: Int32,
    c: Int32,
    h: Int32,
    w: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetFilter4dDescriptor",
        def(
            type_of(filter_desc),
            type_of(data_type),
            type_of(format),
            type_of(k),
            type_of(c),
            type_of(h),
            type_of(w),
        ) thin -> cudnnStatus_t,
    ]()(filter_desc, data_type, format, k, c, h, w)


def cudnnDestroyActivationDescriptor(
    activation_desc: Pointer[cudnnActivationStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyActivationDescriptor",
        def(type_of(activation_desc)) thin -> cudnnStatus_t,
    ]()(activation_desc)


def cudnnGetAlgorithmSpaceSize(
    handle: Pointer[cudnnContext, _],
    algo_desc: Pointer[cudnnAlgorithmStruct, _],
    algo_space_size_in_bytes: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAlgorithmSpaceSize",
        def(
            type_of(handle),
            type_of(algo_desc),
            type_of(algo_space_size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, algo_desc, algo_space_size_in_bytes)


@fieldwise_init
struct cudnnDataType_t(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUDNN_DATA_FLOAT = Self(0)
    comptime CUDNN_DATA_DOUBLE = Self(1)
    comptime CUDNN_DATA_HALF = Self(2)
    comptime CUDNN_DATA_INT8 = Self(3)
    comptime CUDNN_DATA_INT32 = Self(4)
    comptime CUDNN_DATA_INT8x4 = Self(5)
    comptime CUDNN_DATA_UINT8 = Self(6)
    comptime CUDNN_DATA_UINT8x4 = Self(7)
    comptime CUDNN_DATA_INT8x32 = Self(8)
    comptime CUDNN_DATA_BFLOAT16 = Self(9)
    comptime CUDNN_DATA_INT64 = Self(10)
    comptime CUDNN_DATA_BOOLEAN = Self(11)
    comptime CUDNN_DATA_FP8_E4M3 = Self(12)
    comptime CUDNN_DATA_FP8_E5M2 = Self(13)
    comptime CUDNN_DATA_FAST_FLOAT_FOR_FP8 = Self(14)
    comptime CUDNN_DATA_FP8_E8M0 = Self(15)
    comptime CUDNN_DATA_FP4_E2M1 = Self(16)
    comptime CUDNN_DATA_INT4 = Self(17)
    comptime CUDNN_DATA_UINT4 = Self(18)
    comptime CUDNN_DATA_UINT32 = Self(19)
    comptime CUDNN_DATA_COMPLEX_FP32 = Self(20)
    comptime CUDNN_DATA_COMPLEX_FP64 = Self(21)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_DATA_FLOAT:
            return writer.write_string("CUDNN_DATA_FLOAT")
        if self is Self.CUDNN_DATA_DOUBLE:
            return writer.write_string("CUDNN_DATA_DOUBLE")
        if self is Self.CUDNN_DATA_HALF:
            return writer.write_string("CUDNN_DATA_HALF")
        if self is Self.CUDNN_DATA_INT8:
            return writer.write_string("CUDNN_DATA_INT8")
        if self is Self.CUDNN_DATA_INT32:
            return writer.write_string("CUDNN_DATA_INT32")
        if self is Self.CUDNN_DATA_INT8x4:
            return writer.write_string("CUDNN_DATA_INT8x4")
        if self is Self.CUDNN_DATA_UINT8:
            return writer.write_string("CUDNN_DATA_UINT8")
        if self is Self.CUDNN_DATA_UINT8x4:
            return writer.write_string("CUDNN_DATA_UINT8x4")
        if self is Self.CUDNN_DATA_INT8x32:
            return writer.write_string("CUDNN_DATA_INT8x32")
        if self is Self.CUDNN_DATA_BFLOAT16:
            return writer.write_string("CUDNN_DATA_BFLOAT16")
        if self is Self.CUDNN_DATA_INT64:
            return writer.write_string("CUDNN_DATA_INT64")
        if self is Self.CUDNN_DATA_BOOLEAN:
            return writer.write_string("CUDNN_DATA_BOOLEAN")
        if self is Self.CUDNN_DATA_FP8_E4M3:
            return writer.write_string("CUDNN_DATA_FP8_E4M3")
        if self is Self.CUDNN_DATA_FP8_E5M2:
            return writer.write_string("CUDNN_DATA_FP8_E5M2")
        if self is Self.CUDNN_DATA_FAST_FLOAT_FOR_FP8:
            return writer.write_string("CUDNN_DATA_FAST_FLOAT_FOR_FP8")
        if self is Self.CUDNN_DATA_FP8_E8M0:
            return writer.write_string("CUDNN_DATA_FP8_E8M0")
        if self is Self.CUDNN_DATA_FP4_E2M1:
            return writer.write_string("CUDNN_DATA_FP4_E2M1")
        if self is Self.CUDNN_DATA_INT4:
            return writer.write_string("CUDNN_DATA_INT4")
        if self is Self.CUDNN_DATA_UINT4:
            return writer.write_string("CUDNN_DATA_UINT4")
        if self is Self.CUDNN_DATA_UINT32:
            return writer.write_string("CUDNN_DATA_UINT32")
        if self is Self.CUDNN_DATA_COMPLEX_FP32:
            return writer.write_string("CUDNN_DATA_COMPLEX_FP32")
        if self is Self.CUDNN_DATA_COMPLEX_FP64:
            return writer.write_string("CUDNN_DATA_COMPLEX_FP64")
        abort("invalid cudnnDataType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnDataType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSetLRNDescriptor(
    norm_desc: Pointer[cudnnLRNStruct, _],
    lrn_n: Int32,
    lrn_alpha: Float64,
    lrn_beta: Float64,
    lrn_k: Float64,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetLRNDescriptor",
        def(
            type_of(norm_desc),
            type_of(lrn_n),
            type_of(lrn_alpha),
            type_of(lrn_beta),
            type_of(lrn_k),
        ) thin -> cudnnStatus_t,
    ]()(norm_desc, lrn_n, lrn_alpha, lrn_beta, lrn_k)


def cudnnDestroyDropoutDescriptor(
    dropout_desc: Pointer[cudnnDropoutStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyDropoutDescriptor",
        def(type_of(dropout_desc)) thin -> cudnnStatus_t,
    ]()(dropout_desc)


def cudnnGetTensor4dDescriptor(
    tensor_desc: Pointer[cudnnTensorStruct, _],
    data_type: Pointer[cudnnDataType_t, _],
    n: Pointer[Int32, _],
    c: Pointer[Int32, _],
    h: Pointer[Int32, _],
    w: Pointer[Int32, _],
    n_stride: Pointer[Int32, _],
    c_stride: Pointer[Int32, _],
    h_stride: Pointer[Int32, _],
    w_stride: Pointer[Int32, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensor4dDescriptor",
        def(
            type_of(tensor_desc),
            type_of(data_type),
            type_of(n),
            type_of(c),
            type_of(h),
            type_of(w),
            type_of(n_stride),
            type_of(c_stride),
            type_of(h_stride),
            type_of(w_stride),
        ) thin -> cudnnStatus_t,
    ]()(
        tensor_desc,
        data_type,
        n,
        c,
        h,
        w,
        n_stride,
        c_stride,
        h_stride,
        w_stride,
    )


def cudnnGetAlgorithmPerformance(
    algo_perf: Pointer[cudnnAlgorithmPerformanceStruct, _],
    algo_desc: DoubleNestedPointer[cudnnAlgorithmStruct],
    status: Pointer[cudnnStatus_t, _],
    time: Pointer[Float32, _],
    memory: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAlgorithmPerformance",
        def(
            type_of(algo_perf),
            type_of(algo_desc),
            type_of(status),
            type_of(time),
            type_of(memory),
        ) thin -> cudnnStatus_t,
    ]()(algo_perf, algo_desc, status, time, memory)


struct cudnnDebugStruct(ImplicitlyCopyable, RegisterPassable):
    var cudnn_version: Int32
    var cudnnStatus: cudnnStatus_t
    var time_sec: Int32
    var time_usec: Int32
    var time_delta: Int32

    @__allow_legacy_any_origin_fields
    var handle: MutPointer[cudnnContext, UntrackedOrigin[mut=True]]
    var stream: CUstream
    var pid: Int64
    var tid: Int64
    var cudaDeviceId: Int32
    var reserved: StaticTuple[Int32, 15]


def cudnnSetSpatialTransformerNdDescriptor(
    st_desc: Pointer[cudnnSpatialTransformerStruct, _],
    sampler_type: cudnnSamplerType_t,
    data_type: cudnnDataType_t,
    nb_dims: Int32,
    dim_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetSpatialTransformerNdDescriptor",
        def(
            type_of(st_desc),
            type_of(sampler_type),
            type_of(data_type),
            type_of(nb_dims),
            type_of(dim_a),
        ) thin -> cudnnStatus_t,
    ]()(st_desc, sampler_type, data_type, nb_dims, dim_a)


comptime cudnnAlgorithm_t = cudnnAlgorithmUnionStruct


@fieldwise_init
struct cudnnIndicesType_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_32BIT_INDICES = Self(0)
    comptime CUDNN_64BIT_INDICES = Self(1)
    comptime CUDNN_16BIT_INDICES = Self(2)
    comptime CUDNN_8BIT_INDICES = Self(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_32BIT_INDICES:
            return writer.write_string("CUDNN_32BIT_INDICES")
        if self is Self.CUDNN_64BIT_INDICES:
            return writer.write_string("CUDNN_64BIT_INDICES")
        if self is Self.CUDNN_16BIT_INDICES:
            return writer.write_string("CUDNN_16BIT_INDICES")
        if self is Self.CUDNN_8BIT_INDICES:
            return writer.write_string("CUDNN_8BIT_INDICES")
        abort("invalid cudnnIndicesType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnIndicesType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSetTensorTransformDescriptor(
    transform_desc: Pointer[cudnnTensorTransformStruct, _],
    nb_dims: UInt32,
    dest_format: cudnnTensorFormat_t,
    pad_before_a: OpaquePointer,
    pad_after_a: OpaquePointer,
    fold_a: OpaquePointer,
    direction: cudnnFoldingDirection_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensorTransformDescriptor",
        def(
            type_of(transform_desc),
            type_of(nb_dims),
            type_of(dest_format),
            type_of(pad_before_a),
            type_of(pad_after_a),
            type_of(fold_a),
            type_of(direction),
        ) thin -> cudnnStatus_t,
    ]()(
        transform_desc,
        nb_dims,
        dest_format,
        pad_before_a,
        pad_after_a,
        fold_a,
        direction,
    )


def cudnnSetStream(
    handle: Pointer[cudnnContext, _], stream_id: CUstream
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetStream",
        def(type_of(handle), type_of(stream_id)) thin -> cudnnStatus_t,
    ]()(handle, stream_id)


def cudnnDestroyReduceTensorDescriptor(
    reduce_tensor_desc: Pointer[cudnnReduceTensorStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyReduceTensorDescriptor",
        def(type_of(reduce_tensor_desc)) thin -> cudnnStatus_t,
    ]()(reduce_tensor_desc)


def cudnnSetTensor(
    handle: Pointer[cudnnContext, _],
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
    value_ptr: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensor",
        def(
            type_of(handle),
            type_of(y_desc),
            type_of(y),
            type_of(value_ptr),
        ) thin -> cudnnStatus_t,
    ]()(handle, y_desc, y, value_ptr)


def cudnnDivisiveNormalizationForward(
    handle: Pointer[cudnnContext, _],
    norm_desc: Pointer[cudnnLRNStruct, _],
    mode: cudnnDivNormMode_t,
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    means: OpaquePointer,
    temp: OpaquePointer,
    temp2: OpaquePointer,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDivisiveNormalizationForward",
        def(
            type_of(handle),
            type_of(norm_desc),
            type_of(mode),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(means),
            type_of(temp),
            type_of(temp2),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        norm_desc,
        mode,
        alpha,
        x_desc,
        x,
        means,
        temp,
        temp2,
        beta,
        y_desc,
        y,
    )


def cudnnSetActivationDescriptorSwishBeta(
    activation_desc: Pointer[cudnnActivationStruct, _],
    swish_beta: Float64,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetActivationDescriptorSwishBeta",
        def(
            type_of(activation_desc), type_of(swish_beta)
        ) thin -> cudnnStatus_t,
    ]()(activation_desc, swish_beta)


def cudnnSetCallback(
    mask: Int32,
    udata: OpaquePointer,
    fptr: def(
        cudnnSeverity_t,
        OpaquePointer[UntrackedOrigin[mut=True]],
        MutPointer[cudnnDebugStruct, UntrackedOrigin[mut=True]],
        MutPointer[Int8, AnyOrigin[mut=True]],
    ) thin -> NoneType,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetCallback",
        def(
            type_of(mask),
            type_of(udata),
            type_of(fptr),
        ) thin -> cudnnStatus_t,
    ]()(mask, udata, fptr)


def cudnnDropoutGetStatesSize(
    handle: Pointer[cudnnContext, _], size_in_bytes: Pointer[Int, _]
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDropoutGetStatesSize",
        def(type_of(handle), type_of(size_in_bytes)) thin -> cudnnStatus_t,
    ]()(handle, size_in_bytes)


def cudnnCreateDropoutDescriptor(
    dropout_desc: DoubleNestedPointer[cudnnDropoutStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateDropoutDescriptor",
        def(type_of(dropout_desc)) thin -> cudnnStatus_t,
    ]()(dropout_desc)


def cudnnNormalizationForwardInference(
    handle: Pointer[cudnnContext, _],
    mode: cudnnNormMode_t,
    norm_ops: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: OpaquePointer,
    beta: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    norm_scale_bias_desc: Pointer[cudnnTensorStruct, _],
    norm_scale: OpaquePointer,
    norm_bias: OpaquePointer,
    norm_mean_var_desc: Pointer[cudnnTensorStruct, _],
    estimated_mean: OpaquePointer,
    estimated_variance: OpaquePointer,
    z_desc: Pointer[cudnnTensorStruct, _],
    z: OpaquePointer,
    activation_desc: Pointer[cudnnActivationStruct, _],
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
    epsilon: Float64,
    group_cnt: Int32,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnNormalizationForwardInference",
        def(
            type_of(handle),
            type_of(mode),
            type_of(norm_ops),
            type_of(algo),
            type_of(alpha),
            type_of(beta),
            type_of(x_desc),
            type_of(x),
            type_of(norm_scale_bias_desc),
            type_of(norm_scale),
            type_of(norm_bias),
            type_of(norm_mean_var_desc),
            type_of(estimated_mean),
            type_of(estimated_variance),
            type_of(z_desc),
            type_of(z),
            type_of(activation_desc),
            type_of(y_desc),
            type_of(y),
            type_of(epsilon),
            type_of(group_cnt),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        mode,
        norm_ops,
        algo,
        alpha,
        beta,
        x_desc,
        x,
        norm_scale_bias_desc,
        norm_scale,
        norm_bias,
        norm_mean_var_desc,
        estimated_mean,
        estimated_variance,
        z_desc,
        z,
        activation_desc,
        y_desc,
        y,
        epsilon,
        group_cnt,
    )


@fieldwise_init
struct cudnnConvolutionBwdFilterAlgo_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = Self(0)
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = Self(1)
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = Self(2)
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = Self(3)
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = Self(4)
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = Self(5)
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = Self(6)
    comptime CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = Self(7)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
            return writer.write_string("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
            return writer.write_string(
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
            return writer.write(
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
            return writer.write_string(
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
            return writer.write_string(
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT"
            )
        abort("invalid cudnnConvolutionBwdFilterAlgo_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnConvolutionBwdFilterAlgo_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnQueryRuntimeError(
    handle: Pointer[cudnnContext, _],
    rstatus: Pointer[cudnnStatus_t, _],
    mode: cudnnErrQueryMode_t,
    tag: Pointer[cudnnRuntimeTag_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnQueryRuntimeError",
        def(
            type_of(handle),
            type_of(rstatus),
            type_of(mode),
            type_of(tag),
        ) thin -> cudnnStatus_t,
    ]()(handle, rstatus, mode, tag)


def cudnnDestroyLRNDescriptor(
    lrn_desc: Pointer[cudnnLRNStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyLRNDescriptor",
        def(type_of(lrn_desc)) thin -> cudnnStatus_t,
    ]()(lrn_desc)


def cudnnDestroyTensorTransformDescriptor(
    transform_desc: Pointer[cudnnTensorTransformStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyTensorTransformDescriptor",
        def(type_of(transform_desc)) thin -> cudnnStatus_t,
    ]()(transform_desc)


def cudnnSetReduceTensorDescriptor(
    reduce_tensor_desc: Pointer[cudnnReduceTensorStruct, _],
    reduce_tensor_op: cudnnReduceTensorOp_t,
    reduce_tensor_comp_type: cudnnDataType_t,
    reduce_tensor_nan_opt: cudnnNanPropagation_t,
    reduce_tensor_indices: cudnnReduceTensorIndices_t,
    reduce_tensor_indices_type: cudnnIndicesType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetReduceTensorDescriptor",
        def(
            type_of(reduce_tensor_desc),
            type_of(reduce_tensor_op),
            type_of(reduce_tensor_comp_type),
            type_of(reduce_tensor_nan_opt),
            type_of(reduce_tensor_indices),
            type_of(reduce_tensor_indices_type),
        ) thin -> cudnnStatus_t,
    ]()(
        reduce_tensor_desc,
        reduce_tensor_op,
        reduce_tensor_comp_type,
        reduce_tensor_nan_opt,
        reduce_tensor_indices,
        reduce_tensor_indices_type,
    )


def cudnnSetAlgorithmDescriptor(
    algo_desc: Pointer[cudnnAlgorithmStruct, _],
    algorithm: cudnnAlgorithmUnionStruct,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetAlgorithmDescriptor",
        def(type_of(algo_desc), type_of(algorithm)) thin -> cudnnStatus_t,
    ]()(algo_desc, algorithm)


def cudnnCreateFilterDescriptor(
    filter_desc: DoubleNestedPointer[cudnnFilterStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateFilterDescriptor",
        def(type_of(filter_desc)) thin -> cudnnStatus_t,
    ]()(filter_desc)


comptime cudnnHandle_t = Pointer[cudnnContext, _]

comptime cudnnPoolingDescriptor_t = Pointer[cudnnPoolingStruct, _]


def cudnnDestroyOpTensorDescriptor(
    op_tensor_desc: Pointer[cudnnOpTensorStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyOpTensorDescriptor",
        def(type_of(op_tensor_desc)) thin -> cudnnStatus_t,
    ]()(op_tensor_desc)


@fieldwise_init
struct cudnnPoolingMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_POOLING_MAX = Self(0)
    comptime CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = Self(1)
    comptime CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = Self(2)
    comptime CUDNN_POOLING_MAX_DETERMINISTIC = Self(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_POOLING_MAX:
            return writer.write_string("CUDNN_POOLING_MAX")
        if self is Self.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
            return writer.write_string(
                "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING"
            )
        if self is Self.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
            return writer.write_string(
                "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING"
            )
        if self is Self.CUDNN_POOLING_MAX_DETERMINISTIC:
            return writer.write_string("CUDNN_POOLING_MAX_DETERMINISTIC")
        abort("invalid cudnnPoolingMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnPoolingMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetMaxDeviceVersion() raises -> Int:
    return _get_dylib_function[
        "cudnnGetMaxDeviceVersion", def() thin -> Int
    ]()()


def cudnnCreatePoolingDescriptor(
    pooling_desc: DoubleNestedPointer[cudnnPoolingStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreatePoolingDescriptor",
        def(type_of(pooling_desc)) thin -> cudnnStatus_t,
    ]()(pooling_desc)


def cudnnRestoreDropoutDescriptor(
    dropout_desc: Pointer[cudnnDropoutStruct, _],
    handle: Pointer[cudnnContext, _],
    dropout: Float32,
    states: OpaquePointer,
    state_size_in_bytes: Int,
    seed: Int64,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRestoreDropoutDescriptor",
        def(
            type_of(dropout_desc),
            type_of(handle),
            type_of(dropout),
            type_of(states),
            type_of(state_size_in_bytes),
            type_of(seed),
        ) thin -> cudnnStatus_t,
    ]()(dropout_desc, handle, dropout, states, state_size_in_bytes, seed)


def cudnnGetDropoutDescriptor(
    dropout_desc: Pointer[cudnnDropoutStruct, _],
    handle: Pointer[cudnnContext, _],
    dropout: Pointer[Float32, _],
    states: AnyOpaquePointer,
    seed: Pointer[Int64, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetDropoutDescriptor",
        def(
            type_of(dropout_desc),
            type_of(handle),
            type_of(dropout),
            type_of(states),
            type_of(seed),
        ) thin -> cudnnStatus_t,
    ]()(dropout_desc, handle, dropout, states, seed)


@fieldwise_init
struct cudnnDivNormMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_DIVNORM_PRECOMPUTED_MEANS = Self(0)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_DIVNORM_PRECOMPUTED_MEANS:
            return writer.write_string("CUDNN_DIVNORM_PRECOMPUTED_MEANS")
        abort("invalid cudnnDivNormMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnDivNormMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnCreateOpTensorDescriptor(
    op_tensor_desc: DoubleNestedPointer[cudnnOpTensorStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateOpTensorDescriptor",
        def(type_of(op_tensor_desc)) thin -> cudnnStatus_t,
    ]()(op_tensor_desc)


def cudnnSetFilterNdDescriptor(
    filter_desc: Pointer[cudnnFilterStruct, _],
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nb_dims: Int32,
    filter_dim_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetFilterNdDescriptor",
        def(
            type_of(filter_desc),
            type_of(data_type),
            type_of(format),
            type_of(nb_dims),
            type_of(filter_dim_a),
        ) thin -> cudnnStatus_t,
    ]()(filter_desc, data_type, format, nb_dims, filter_dim_a)


def cudnnRestoreAlgorithm(
    handle: Pointer[cudnnContext, _],
    algo_space: OpaquePointer,
    algo_space_size_in_bytes: Int,
    algo_desc: Pointer[cudnnAlgorithmStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRestoreAlgorithm",
        def(
            type_of(handle),
            type_of(algo_space),
            type_of(algo_space_size_in_bytes),
            type_of(algo_desc),
        ) thin -> cudnnStatus_t,
    ]()(handle, algo_space, algo_space_size_in_bytes, algo_desc)


def cudnnGetPoolingNdDescriptor(
    pooling_desc: Pointer[cudnnPoolingStruct, _],
    nb_dims_requested: Int32,
    mode: Pointer[cudnnPoolingMode_t, _],
    maxpooling_nan_opt: Pointer[cudnnNanPropagation_t, _],
    nb_dims: Pointer[Int32, _],
    window_dim_a: OpaquePointer,
    padding_a: OpaquePointer,
    stride_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPoolingNdDescriptor",
        def(
            type_of(pooling_desc),
            type_of(nb_dims_requested),
            type_of(mode),
            type_of(maxpooling_nan_opt),
            type_of(nb_dims),
            type_of(window_dim_a),
            type_of(padding_a),
            type_of(stride_a),
        ) thin -> cudnnStatus_t,
    ]()(
        pooling_desc,
        nb_dims_requested,
        mode,
        maxpooling_nan_opt,
        nb_dims,
        window_dim_a,
        padding_a,
        stride_a,
    )


def cudnnSetDropoutDescriptor(
    dropout_desc: Pointer[cudnnDropoutStruct, _],
    handle: Pointer[cudnnContext, _],
    dropout: Float32,
    states: OpaquePointer,
    state_size_in_bytes: Int,
    seed: Int64,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetDropoutDescriptor",
        def(
            type_of(dropout_desc),
            type_of(handle),
            type_of(dropout),
            type_of(states),
            type_of(state_size_in_bytes),
            type_of(seed),
        ) thin -> cudnnStatus_t,
    ]()(dropout_desc, handle, dropout, states, state_size_in_bytes, seed)


def cudnnCreateSpatialTransformerDescriptor(
    st_desc: DoubleNestedPointer[cudnnSpatialTransformerStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateSpatialTransformerDescriptor",
        def(type_of(st_desc)) thin -> cudnnStatus_t,
    ]()(st_desc)


def cudnnInitTransformDest(
    transform_desc: Pointer[cudnnTensorTransformStruct, _],
    src_desc: Pointer[cudnnTensorStruct, _],
    dest_desc: Pointer[cudnnTensorStruct, _],
    dest_size_in_bytes: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnInitTransformDest",
        def(
            type_of(transform_desc),
            type_of(src_desc),
            type_of(dest_desc),
            type_of(dest_size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(transform_desc, src_desc, dest_desc, dest_size_in_bytes)


def cudnnCreateAlgorithmDescriptor(
    algo_desc: DoubleNestedPointer[cudnnAlgorithmStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateAlgorithmDescriptor",
        def(type_of(algo_desc)) thin -> cudnnStatus_t,
    ]()(algo_desc)


def cudnnDestroyTensorDescriptor(
    tensor_desc: Pointer[cudnnTensorStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyTensorDescriptor",
        def(type_of(tensor_desc)) thin -> cudnnStatus_t,
    ]()(tensor_desc)
