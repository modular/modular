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
comptime cudnnFilterStruct = AnyOpaquePointer


def cudnnCreate(
    handle: DoubleNestedPointer[cudnnContext],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreate",
        def(type_of(handle)) thin -> cudnnStatus_t,
    ]()(handle)


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


def cudnnDestroyFilterDescriptor(
    filter_desc: Pointer[cudnnFilterStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyFilterDescriptor",
        def(type_of(filter_desc)) thin -> cudnnStatus_t,
    ]()(filter_desc)


def cudnnCreateTensorDescriptor(
    tensor_desc: DoubleNestedPointer[cudnnTensorStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateTensorDescriptor",
        def(type_of(tensor_desc)) thin -> cudnnStatus_t,
    ]()(tensor_desc)


def cudnnDestroy(handle: Pointer[cudnnContext, _]) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroy", def(type_of(handle)) thin -> cudnnStatus_t
    ]()(handle)


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


def cudnnGetErrorString(
    status: cudnnStatus_t,
) raises -> ImmPointer[Int8, ImmUntrackedOrigin]:
    return _get_dylib_function[
        "cudnnGetErrorString",
        def(type_of(status),) thin -> ImmPointer[Int8, ImmUntrackedOrigin],
    ]()(status)


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


def cudnnSetStream(
    handle: Pointer[cudnnContext, _], stream_id: CUstream
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetStream",
        def(type_of(handle), type_of(stream_id)) thin -> cudnnStatus_t,
    ]()(handle, stream_id)


def cudnnCreateFilterDescriptor(
    filter_desc: DoubleNestedPointer[cudnnFilterStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateFilterDescriptor",
        def(type_of(filter_desc)) thin -> cudnnStatus_t,
    ]()(filter_desc)


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


def cudnnDestroyTensorDescriptor(
    tensor_desc: Pointer[cudnnTensorStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyTensorDescriptor",
        def(type_of(tensor_desc)) thin -> cudnnStatus_t,
    ]()(tensor_desc)
