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
from os import abort
from pathlib import Path
from sys.ffi import (
    _get_dylib_function as _ffi_get_dylib_function,
    _Global,
    _OwnedDLHandle,
    _find_dylib,
)

from gpu.host._nvidia_cuda import CUstream
from memory import UnsafePointer

from utils import StaticTuple

from .backend import *

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_CUDNN_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu/libcudnn.so.8"

alias CUDA_CUDNN_INFER_LIBRARY = _Global[
    "CUDA_CUDNN_INFER_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    return _find_dylib["CUDA CUDNN library"](CUDA_CUDNN_LIBRARY_PATH)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_INFER_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

alias cudnnContext = UnsafePointer[NoneType]
alias cudnnTensorStruct = UnsafePointer[NoneType]
alias cudnnAlgorithmStruct = UnsafePointer[NoneType]
alias cudnnTensorTransformStruct = UnsafePointer[NoneType]
alias cudnnSpatialTransformerStruct = UnsafePointer[NoneType]
alias cudnnDropoutStruct = UnsafePointer[NoneType]
alias cudnnPoolingStruct = UnsafePointer[NoneType]
alias cudnnFilterStruct = UnsafePointer[NoneType]
alias cudnnOpTensorStruct = UnsafePointer[NoneType]
alias cudnnReduceTensorStruct = UnsafePointer[NoneType]
alias cudnnLRNStruct = UnsafePointer[NoneType]
alias cudnnActivationStruct = UnsafePointer[NoneType]
alias cudnnAlgorithmPerformanceStruct = UnsafePointer[NoneType]
alias cudnnCTCLossStruct = UnsafePointer[NoneType]
alias cudnnRuntimeTag_t = NoneType


@value
@register_passable("trivial")
struct cudnnSoftmaxMode_t(Writable):
    var _value: Int8
    alias CUDNN_SOFTMAX_MODE_INSTANCE = Self(0)
    alias CUDNN_SOFTMAX_MODE_CHANNEL = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_SOFTMAX_MODE_INSTANCE:
            return writer.write("CUDNN_SOFTMAX_MODE_INSTANCE")
        if self is Self.CUDNN_SOFTMAX_MODE_CHANNEL:
            return writer.write("CUDNN_SOFTMAX_MODE_CHANNEL")
        abort("invalid cudnnSoftmaxMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnSoftmaxMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnDestroyAlgorithmPerformance(
    algo_perf: UnsafePointer[UnsafePointer[cudnnAlgorithmPerformanceStruct]],
    number_to_destroy: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyAlgorithmPerformance",
        fn (
            UnsafePointer[UnsafePointer[cudnnAlgorithmPerformanceStruct]], Int16
        ) -> cudnnStatus_t,
    ]()(algo_perf, number_to_destroy)


fn cudnnCreate(
    handle: UnsafePointer[UnsafePointer[cudnnContext]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreate",
        fn (UnsafePointer[UnsafePointer[cudnnContext]]) -> cudnnStatus_t,
    ]()(handle)


@value
@register_passable("trivial")
struct cudnnReduceTensorIndices_t(Writable):
    var _value: Int8
    alias CUDNN_REDUCE_TENSOR_NO_INDICES = Self(0)
    alias CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_REDUCE_TENSOR_NO_INDICES:
            return writer.write("CUDNN_REDUCE_TENSOR_NO_INDICES")
        if self is Self.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES:
            return writer.write("CUDNN_REDUCE_TENSOR_FLATTENED_INDICES")
        abort("invalid cudnnReduceTensorIndices_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnReduceTensorIndices_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnReduceTensor(
    handle: UnsafePointer[cudnnContext],
    reduce_tensor_desc: UnsafePointer[cudnnReduceTensorStruct],
    indices: UnsafePointer[NoneType],
    indices_size_in_bytes: Int,
    workspace: UnsafePointer[NoneType],
    workspace_size_in_bytes: Int,
    alpha: UnsafePointer[NoneType],
    a_desc: UnsafePointer[cudnnTensorStruct],
    _a: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    c_desc: UnsafePointer[cudnnTensorStruct],
    _c: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnReduceTensor",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnReduceTensorStruct],
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
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


fn cudnnGetActivationDescriptorSwishBeta(
    activation_desc: UnsafePointer[cudnnActivationStruct],
    swish_beta: UnsafePointer[Float64],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetActivationDescriptorSwishBeta",
        fn (
            UnsafePointer[cudnnActivationStruct], UnsafePointer[Float64]
        ) -> cudnnStatus_t,
    ]()(activation_desc, swish_beta)


fn cudnnDestroyAlgorithmDescriptor(
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyAlgorithmDescriptor",
        fn (UnsafePointer[cudnnAlgorithmStruct]) -> cudnnStatus_t,
    ]()(algo_desc)


alias cudnnTensorTransformDescriptor_t = UnsafePointer[
    cudnnTensorTransformStruct
]

alias cudnnTensorDescriptor_t = UnsafePointer[cudnnTensorStruct]


fn cudnnDropoutGetReserveSpaceSize(
    xdesc: UnsafePointer[cudnnTensorStruct], size_in_bytes: UnsafePointer[Int]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDropoutGetReserveSpaceSize",
        fn (
            UnsafePointer[cudnnTensorStruct], UnsafePointer[Int]
        ) -> cudnnStatus_t,
    ]()(xdesc, size_in_bytes)


fn cudnnGetReduceTensorDescriptor(
    reduce_tensor_desc: UnsafePointer[cudnnReduceTensorStruct],
    reduce_tensor_op: UnsafePointer[cudnnReduceTensorOp_t],
    reduce_tensor_comp_type: UnsafePointer[cudnnDataType_t],
    reduce_tensor_nan_opt: UnsafePointer[cudnnNanPropagation_t],
    reduce_tensor_indices: UnsafePointer[cudnnReduceTensorIndices_t],
    reduce_tensor_indices_type: UnsafePointer[cudnnIndicesType_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetReduceTensorDescriptor",
        fn (
            UnsafePointer[cudnnReduceTensorStruct],
            UnsafePointer[cudnnReduceTensorOp_t],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnNanPropagation_t],
            UnsafePointer[cudnnReduceTensorIndices_t],
            UnsafePointer[cudnnIndicesType_t],
        ) -> cudnnStatus_t,
    ]()(
        reduce_tensor_desc,
        reduce_tensor_op,
        reduce_tensor_comp_type,
        reduce_tensor_nan_opt,
        reduce_tensor_indices,
        reduce_tensor_indices_type,
    )


fn cudnnSetPoolingNdDescriptor(
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
    mode: cudnnPoolingMode_t,
    maxpooling_nan_opt: cudnnNanPropagation_t,
    nb_dims: Int16,
    window_dim_a: UnsafePointer[NoneType],
    padding_a: UnsafePointer[NoneType],
    stride_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetPoolingNdDescriptor",
        fn (
            UnsafePointer[cudnnPoolingStruct],
            cudnnPoolingMode_t,
            cudnnNanPropagation_t,
            Int16,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(
        pooling_desc,
        mode,
        maxpooling_nan_opt,
        nb_dims,
        window_dim_a,
        padding_a,
        stride_a,
    )


@value
@register_passable("trivial")
struct cudnnReduceTensorOp_t(Writable):
    var _value: Int8
    alias CUDNN_REDUCE_TENSOR_ADD = Self(0)
    alias CUDNN_REDUCE_TENSOR_MUL = Self(1)
    alias CUDNN_REDUCE_TENSOR_MIN = Self(2)
    alias CUDNN_REDUCE_TENSOR_MAX = Self(3)
    alias CUDNN_REDUCE_TENSOR_AMAX = Self(4)
    alias CUDNN_REDUCE_TENSOR_AVG = Self(5)
    alias CUDNN_REDUCE_TENSOR_NORM1 = Self(6)
    alias CUDNN_REDUCE_TENSOR_NORM2 = Self(7)
    alias CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = Self(8)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_REDUCE_TENSOR_ADD:
            return writer.write("CUDNN_REDUCE_TENSOR_ADD")
        if self is Self.CUDNN_REDUCE_TENSOR_MUL:
            return writer.write("CUDNN_REDUCE_TENSOR_MUL")
        if self is Self.CUDNN_REDUCE_TENSOR_MIN:
            return writer.write("CUDNN_REDUCE_TENSOR_MIN")
        if self is Self.CUDNN_REDUCE_TENSOR_MAX:
            return writer.write("CUDNN_REDUCE_TENSOR_MAX")
        if self is Self.CUDNN_REDUCE_TENSOR_AMAX:
            return writer.write("CUDNN_REDUCE_TENSOR_AMAX")
        if self is Self.CUDNN_REDUCE_TENSOR_AVG:
            return writer.write("CUDNN_REDUCE_TENSOR_AVG")
        if self is Self.CUDNN_REDUCE_TENSOR_NORM1:
            return writer.write("CUDNN_REDUCE_TENSOR_NORM1")
        if self is Self.CUDNN_REDUCE_TENSOR_NORM2:
            return writer.write("CUDNN_REDUCE_TENSOR_NORM2")
        if self is Self.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
            return writer.write("CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS")
        abort("invalid cudnnReduceTensorOp_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnReduceTensorOp_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSetTensor4dDescriptor(
    tensor_desc: UnsafePointer[cudnnTensorStruct],
    format: cudnnTensorFormat_t,
    data_type: cudnnDataType_t,
    n: Int16,
    c: Int16,
    h: Int16,
    w: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensor4dDescriptor",
        fn (
            UnsafePointer[cudnnTensorStruct],
            cudnnTensorFormat_t,
            cudnnDataType_t,
            Int16,
            Int16,
            Int16,
            Int16,
        ) -> cudnnStatus_t,
    ]()(tensor_desc, format, data_type, n, c, h, w)


fn cudnnLRNCrossChannelForward(
    handle: UnsafePointer[cudnnContext],
    norm_desc: UnsafePointer[cudnnLRNStruct],
    lrn_mode: cudnnLRNMode_t,
    alpha: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnLRNCrossChannelForward",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnLRNStruct],
            cudnnLRNMode_t,
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, norm_desc, lrn_mode, alpha, x_desc, x, beta, y_desc, y)


@value
@register_passable("trivial")
struct cudnnDeterminism_t(Writable):
    var _value: Int8
    alias CUDNN_NON_DETERMINISTIC = Self(0)
    alias CUDNN_DETERMINISTIC = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_NON_DETERMINISTIC:
            return writer.write("CUDNN_NON_DETERMINISTIC")
        if self is Self.CUDNN_DETERMINISTIC:
            return writer.write("CUDNN_DETERMINISTIC")
        abort("invalid cudnnDeterminism_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnDeterminism_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias cudnnAlgorithmDescriptor_t = UnsafePointer[cudnnAlgorithmStruct]

alias cudnnActivationDescriptor_t = UnsafePointer[cudnnActivationStruct]


@value
@register_passable("trivial")
struct cudnnStatus_t(Writable):
    var _value: Int8
    alias CUDNN_STATUS_SUCCESS = Self(0)
    alias CUDNN_STATUS_NOT_INITIALIZED = Self(1)
    alias CUDNN_STATUS_ALLOC_FAILED = Self(2)
    alias CUDNN_STATUS_BAD_PARAM = Self(3)
    alias CUDNN_STATUS_INTERNAL_ERROR = Self(4)
    alias CUDNN_STATUS_INVALID_VALUE = Self(5)
    alias CUDNN_STATUS_ARCH_MISMATCH = Self(6)
    alias CUDNN_STATUS_MAPPING_ERROR = Self(7)
    alias CUDNN_STATUS_EXECUTION_FAILED = Self(8)
    alias CUDNN_STATUS_NOT_SUPPORTED = Self(9)
    alias CUDNN_STATUS_LICENSE_ERROR = Self(10)
    alias CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = Self(11)
    alias CUDNN_STATUS_RUNTIME_IN_PROGRESS = Self(12)
    alias CUDNN_STATUS_RUNTIME_FP_OVERFLOW = Self(13)
    alias CUDNN_STATUS_VERSION_MISMATCH = Self(14)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_STATUS_SUCCESS:
            return writer.write("CUDNN_STATUS_SUCCESS")
        if self is Self.CUDNN_STATUS_NOT_INITIALIZED:
            return writer.write("CUDNN_STATUS_NOT_INITIALIZED")
        if self is Self.CUDNN_STATUS_ALLOC_FAILED:
            return writer.write("CUDNN_STATUS_ALLOC_FAILED")
        if self is Self.CUDNN_STATUS_BAD_PARAM:
            return writer.write("CUDNN_STATUS_BAD_PARAM")
        if self is Self.CUDNN_STATUS_INTERNAL_ERROR:
            return writer.write("CUDNN_STATUS_INTERNAL_ERROR")
        if self is Self.CUDNN_STATUS_INVALID_VALUE:
            return writer.write("CUDNN_STATUS_INVALID_VALUE")
        if self is Self.CUDNN_STATUS_ARCH_MISMATCH:
            return writer.write("CUDNN_STATUS_ARCH_MISMATCH")
        if self is Self.CUDNN_STATUS_MAPPING_ERROR:
            return writer.write("CUDNN_STATUS_MAPPING_ERROR")
        if self is Self.CUDNN_STATUS_EXECUTION_FAILED:
            return writer.write("CUDNN_STATUS_EXECUTION_FAILED")
        if self is Self.CUDNN_STATUS_NOT_SUPPORTED:
            return writer.write("CUDNN_STATUS_NOT_SUPPORTED")
        if self is Self.CUDNN_STATUS_LICENSE_ERROR:
            return writer.write("CUDNN_STATUS_LICENSE_ERROR")
        if self is Self.CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return writer.write("CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING")
        if self is Self.CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return writer.write("CUDNN_STATUS_RUNTIME_IN_PROGRESS")
        if self is Self.CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return writer.write("CUDNN_STATUS_RUNTIME_FP_OVERFLOW")
        if self is Self.CUDNN_STATUS_VERSION_MISMATCH:
            return writer.write("CUDNN_STATUS_VERSION_MISMATCH")
        abort("invalid cudnnStatus_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnStatus_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct cudnnCTCLossAlgo_t(Writable):
    var _value: Int8
    alias CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = Self(0)
    alias CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC:
            return writer.write("CUDNN_CTC_LOSS_ALGO_DETERMINISTIC")
        if self is Self.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC:
            return writer.write("CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC")
        abort("invalid cudnnCTCLossAlgo_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnCTCLossAlgo_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetFilter4dDescriptor(
    filter_desc: UnsafePointer[cudnnFilterStruct],
    data_type: UnsafePointer[cudnnDataType_t],
    format: UnsafePointer[cudnnTensorFormat_t],
    k: UnsafePointer[Int16],
    c: UnsafePointer[Int16],
    h: UnsafePointer[Int16],
    w: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFilter4dDescriptor",
        fn (
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnTensorFormat_t],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
        ) -> cudnnStatus_t,
    ]()(filter_desc, data_type, format, k, c, h, w)


@value
@register_passable("trivial")
struct cudnnTensorFormat_t(Writable):
    var _value: Int8
    alias CUDNN_TENSOR_NCHW = Self(0)
    alias CUDNN_TENSOR_NHWC = Self(1)
    alias CUDNN_TENSOR_NCHW_VECT_C = Self(2)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_TENSOR_NCHW:
            return writer.write("CUDNN_TENSOR_NCHW")
        if self is Self.CUDNN_TENSOR_NHWC:
            return writer.write("CUDNN_TENSOR_NHWC")
        if self is Self.CUDNN_TENSOR_NCHW_VECT_C:
            return writer.write("CUDNN_TENSOR_NCHW_VECT_C")
        abort("invalid cudnnTensorFormat_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnTensorFormat_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnAddTensor(
    handle: UnsafePointer[cudnnContext],
    alpha: UnsafePointer[NoneType],
    a_desc: UnsafePointer[cudnnTensorStruct],
    _a: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    c_desc: UnsafePointer[cudnnTensorStruct],
    _c: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnAddTensor",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, alpha, a_desc, _a, beta, c_desc, _c)


fn cudnnDestroyFilterDescriptor(
    filter_desc: UnsafePointer[cudnnFilterStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyFilterDescriptor",
        fn (UnsafePointer[cudnnFilterStruct]) -> cudnnStatus_t,
    ]()(filter_desc)


fn cudnnGetTensorTransformDescriptor(
    transform_desc: UnsafePointer[cudnnTensorTransformStruct],
    nb_dims_requested: UInt32,
    dest_format: UnsafePointer[cudnnTensorFormat_t],
    pad_before_a: UnsafePointer[NoneType],
    pad_after_a: UnsafePointer[NoneType],
    fold_a: UnsafePointer[NoneType],
    direction: UnsafePointer[cudnnFoldingDirection_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensorTransformDescriptor",
        fn (
            UnsafePointer[cudnnTensorTransformStruct],
            UInt32,
            UnsafePointer[cudnnTensorFormat_t],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnFoldingDirection_t],
        ) -> cudnnStatus_t,
    ]()(
        transform_desc,
        nb_dims_requested,
        dest_format,
        pad_before_a,
        pad_after_a,
        fold_a,
        direction,
    )


fn cudnnGetVersion() -> Int:
    return _get_dylib_function["cudnnGetVersion", fn () -> Int]()()


fn cudnnGetCudartVersion() -> Int:
    return _get_dylib_function["cudnnGetCudartVersion", fn () -> Int]()()


fn cudnnGetCallback(
    mask: UnsafePointer[Int16],
    udata: UnsafePointer[UnsafePointer[NoneType]],
    fptr: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetCallback",
        fn (
            UnsafePointer[Int16],
            UnsafePointer[UnsafePointer[NoneType]],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(mask, udata, fptr)


fn cudnnCreateTensorTransformDescriptor(
    transform_desc: UnsafePointer[UnsafePointer[cudnnTensorTransformStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateTensorTransformDescriptor",
        fn (
            UnsafePointer[UnsafePointer[cudnnTensorTransformStruct]],
        ) -> cudnnStatus_t,
    ]()(transform_desc)


fn cudnnCreateLRNDescriptor(
    norm_desc: UnsafePointer[UnsafePointer[cudnnLRNStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateLRNDescriptor",
        fn (UnsafePointer[UnsafePointer[cudnnLRNStruct]]) -> cudnnStatus_t,
    ]()(norm_desc)


fn cudnnSetActivationDescriptor(
    activation_desc: UnsafePointer[cudnnActivationStruct],
    mode: cudnnActivationMode_t,
    relu_nan_opt: cudnnNanPropagation_t,
    coef: Float64,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetActivationDescriptor",
        fn (
            UnsafePointer[cudnnActivationStruct],
            cudnnActivationMode_t,
            cudnnNanPropagation_t,
            Float64,
        ) -> cudnnStatus_t,
    ]()(activation_desc, mode, relu_nan_opt, coef)


@value
@register_passable("trivial")
struct cudnnNormAlgo_t(Writable):
    var _value: Int8
    alias CUDNN_NORM_ALGO_STANDARD = Self(0)
    alias CUDNN_NORM_ALGO_PERSIST = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_NORM_ALGO_STANDARD:
            return writer.write("CUDNN_NORM_ALGO_STANDARD")
        if self is Self.CUDNN_NORM_ALGO_PERSIST:
            return writer.write("CUDNN_NORM_ALGO_PERSIST")
        abort("invalid cudnnNormAlgo_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnNormAlgo_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct cudnnOpTensorOp_t(Writable):
    var _value: Int8
    alias CUDNN_OP_TENSOR_ADD = Self(0)
    alias CUDNN_OP_TENSOR_MUL = Self(1)
    alias CUDNN_OP_TENSOR_MIN = Self(2)
    alias CUDNN_OP_TENSOR_MAX = Self(3)
    alias CUDNN_OP_TENSOR_SQRT = Self(4)
    alias CUDNN_OP_TENSOR_NOT = Self(5)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_OP_TENSOR_ADD:
            return writer.write("CUDNN_OP_TENSOR_ADD")
        if self is Self.CUDNN_OP_TENSOR_MUL:
            return writer.write("CUDNN_OP_TENSOR_MUL")
        if self is Self.CUDNN_OP_TENSOR_MIN:
            return writer.write("CUDNN_OP_TENSOR_MIN")
        if self is Self.CUDNN_OP_TENSOR_MAX:
            return writer.write("CUDNN_OP_TENSOR_MAX")
        if self is Self.CUDNN_OP_TENSOR_SQRT:
            return writer.write("CUDNN_OP_TENSOR_SQRT")
        if self is Self.CUDNN_OP_TENSOR_NOT:
            return writer.write("CUDNN_OP_TENSOR_NOT")
        abort("invalid cudnnOpTensorOp_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnOpTensorOp_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnCreateReduceTensorDescriptor(
    reduce_tensor_desc: UnsafePointer[UnsafePointer[cudnnReduceTensorStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateReduceTensorDescriptor",
        fn (
            UnsafePointer[UnsafePointer[cudnnReduceTensorStruct]],
        ) -> cudnnStatus_t,
    ]()(reduce_tensor_desc)


fn cudnnGetPoolingNdForwardOutputDim(
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
    input_tensor_desc: UnsafePointer[cudnnTensorStruct],
    nb_dims: Int16,
    output_tensor_dim_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPoolingNdForwardOutputDim",
        fn (
            UnsafePointer[cudnnPoolingStruct],
            UnsafePointer[cudnnTensorStruct],
            Int16,
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(pooling_desc, input_tensor_desc, nb_dims, output_tensor_dim_a)


fn cudnnDestroySpatialTransformerDescriptor(
    st_desc: UnsafePointer[cudnnSpatialTransformerStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroySpatialTransformerDescriptor",
        fn (UnsafePointer[cudnnSpatialTransformerStruct]) -> cudnnStatus_t,
    ]()(st_desc)


alias cudnnReduceTensorDescriptor_t = UnsafePointer[cudnnReduceTensorStruct]


fn cudnnCreateTensorDescriptor(
    tensor_desc: UnsafePointer[UnsafePointer[cudnnTensorStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateTensorDescriptor",
        fn (UnsafePointer[UnsafePointer[cudnnTensorStruct]]) -> cudnnStatus_t,
    ]()(tensor_desc)


fn cudnnSetOpTensorDescriptor(
    op_tensor_desc: UnsafePointer[cudnnOpTensorStruct],
    op_tensor_op: cudnnOpTensorOp_t,
    op_tensor_comp_type: cudnnDataType_t,
    op_tensor_nan_opt: cudnnNanPropagation_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetOpTensorDescriptor",
        fn (
            UnsafePointer[cudnnOpTensorStruct],
            cudnnOpTensorOp_t,
            cudnnDataType_t,
            cudnnNanPropagation_t,
        ) -> cudnnStatus_t,
    ]()(op_tensor_desc, op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt)


fn cudnnBatchNormalizationForwardInference(
    handle: UnsafePointer[cudnnContext],
    mode: cudnnBatchNormMode_t,
    alpha: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
    bn_scale_bias_mean_var_desc: UnsafePointer[cudnnTensorStruct],
    bn_scale: UnsafePointer[NoneType],
    bn_bias: UnsafePointer[NoneType],
    estimated_mean: UnsafePointer[NoneType],
    estimated_variance: UnsafePointer[NoneType],
    epsilon: Float64,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBatchNormalizationForwardInference",
        fn (
            UnsafePointer[cudnnContext],
            cudnnBatchNormMode_t,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            Float64,
        ) -> cudnnStatus_t,
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


fn cudnnCreateAlgorithmPerformance(
    algo_perf: UnsafePointer[UnsafePointer[cudnnAlgorithmPerformanceStruct]],
    number_to_create: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateAlgorithmPerformance",
        fn (
            UnsafePointer[UnsafePointer[cudnnAlgorithmPerformanceStruct]], Int16
        ) -> cudnnStatus_t,
    ]()(algo_perf, number_to_create)


fn cudnnDropoutForward(
    handle: UnsafePointer[cudnnContext],
    dropout_desc: UnsafePointer[cudnnDropoutStruct],
    xdesc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    ydesc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
    reserve_space: UnsafePointer[NoneType],
    reserve_space_size_in_bytes: Int,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDropoutForward",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnDropoutStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            Int,
        ) -> cudnnStatus_t,
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


fn cudnnDestroy(handle: UnsafePointer[cudnnContext]) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroy", fn (UnsafePointer[cudnnContext]) -> cudnnStatus_t
    ]()(handle)


fn cudnnGetActivationDescriptor(
    activation_desc: UnsafePointer[cudnnActivationStruct],
    mode: UnsafePointer[cudnnActivationMode_t],
    relu_nan_opt: UnsafePointer[cudnnNanPropagation_t],
    coef: UnsafePointer[Float64],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetActivationDescriptor",
        fn (
            UnsafePointer[cudnnActivationStruct],
            UnsafePointer[cudnnActivationMode_t],
            UnsafePointer[cudnnNanPropagation_t],
            UnsafePointer[Float64],
        ) -> cudnnStatus_t,
    ]()(activation_desc, mode, relu_nan_opt, coef)


fn cudnnOpTensor(
    handle: UnsafePointer[cudnnContext],
    op_tensor_desc: UnsafePointer[cudnnOpTensorStruct],
    alpha1: UnsafePointer[NoneType],
    a_desc: UnsafePointer[cudnnTensorStruct],
    _a: UnsafePointer[NoneType],
    alpha2: UnsafePointer[NoneType],
    b_desc: UnsafePointer[cudnnTensorStruct],
    _b: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    c_desc: UnsafePointer[cudnnTensorStruct],
    _c: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnOpTensor",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnOpTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
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


fn cudnnDeriveBNTensorDescriptor(
    derived_bn_desc: UnsafePointer[cudnnTensorStruct],
    x_desc: UnsafePointer[cudnnTensorStruct],
    mode: cudnnBatchNormMode_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDeriveBNTensorDescriptor",
        fn (
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            cudnnBatchNormMode_t,
        ) -> cudnnStatus_t,
    ]()(derived_bn_desc, x_desc, mode)


@value
@register_passable("trivial")
struct cudnnActivationMode_t(Writable):
    var _value: Int8
    alias CUDNN_ACTIVATION_SIGMOID = Self(0)
    alias CUDNN_ACTIVATION_RELU = Self(1)
    alias CUDNN_ACTIVATION_TANH = Self(2)
    alias CUDNN_ACTIVATION_CLIPPED_RELU = Self(3)
    alias CUDNN_ACTIVATION_ELU = Self(4)
    alias CUDNN_ACTIVATION_IDENTITY = Self(5)
    alias CUDNN_ACTIVATION_SWISH = Self(6)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_ACTIVATION_SIGMOID:
            return writer.write("CUDNN_ACTIVATION_SIGMOID")
        if self is Self.CUDNN_ACTIVATION_RELU:
            return writer.write("CUDNN_ACTIVATION_RELU")
        if self is Self.CUDNN_ACTIVATION_TANH:
            return writer.write("CUDNN_ACTIVATION_TANH")
        if self is Self.CUDNN_ACTIVATION_CLIPPED_RELU:
            return writer.write("CUDNN_ACTIVATION_CLIPPED_RELU")
        if self is Self.CUDNN_ACTIVATION_ELU:
            return writer.write("CUDNN_ACTIVATION_ELU")
        if self is Self.CUDNN_ACTIVATION_IDENTITY:
            return writer.write("CUDNN_ACTIVATION_IDENTITY")
        if self is Self.CUDNN_ACTIVATION_SWISH:
            return writer.write("CUDNN_ACTIVATION_SWISH")
        abort("invalid cudnnActivationMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnActivationMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSpatialTfGridGeneratorForward(
    handle: UnsafePointer[cudnnContext],
    st_desc: UnsafePointer[cudnnSpatialTransformerStruct],
    theta: UnsafePointer[NoneType],
    grid: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSpatialTfGridGeneratorForward",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnSpatialTransformerStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, st_desc, theta, grid)


fn cudnnGetTensorSizeInBytes(
    tensor_desc: UnsafePointer[cudnnTensorStruct], size: UnsafePointer[Int]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensorSizeInBytes",
        fn (
            UnsafePointer[cudnnTensorStruct], UnsafePointer[Int]
        ) -> cudnnStatus_t,
    ]()(tensor_desc, size)


@value
@register_passable("trivial")
struct cudnnConvolutionBwdDataAlgo_t(Writable):
    var _value: Int8
    alias CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = Self(0)
    alias CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = Self(1)
    alias CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = Self(2)
    alias CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = Self(3)
    alias CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = Self(4)
    alias CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = Self(5)
    alias CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = Self(6)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
            return writer.write("CUDNN_CONVOLUTION_BWD_DATA_ALGO_0")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
            return writer.write("CUDNN_CONVOLUTION_BWD_DATA_ALGO_1")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
            return writer.write("CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
            return writer.write("CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
            return writer.write("CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD")
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
            return writer.write(
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
            return writer.write("CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT")
        abort("invalid cudnnConvolutionBwdDataAlgo_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnConvolutionBwdDataAlgo_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetFilterNdDescriptor(
    filter_desc: UnsafePointer[cudnnFilterStruct],
    nb_dims_requested: Int16,
    data_type: UnsafePointer[cudnnDataType_t],
    format: UnsafePointer[cudnnTensorFormat_t],
    nb_dims: UnsafePointer[Int16],
    filter_dim_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFilterNdDescriptor",
        fn (
            UnsafePointer[cudnnFilterStruct],
            Int16,
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnTensorFormat_t],
            UnsafePointer[Int16],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(
        filter_desc, nb_dims_requested, data_type, format, nb_dims, filter_dim_a
    )


fn cudnnGetPooling2dForwardOutputDim(
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
    input_tensor_desc: UnsafePointer[cudnnTensorStruct],
    n: UnsafePointer[Int16],
    c: UnsafePointer[Int16],
    h: UnsafePointer[Int16],
    w: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPooling2dForwardOutputDim",
        fn (
            UnsafePointer[cudnnPoolingStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
        ) -> cudnnStatus_t,
    ]()(pooling_desc, input_tensor_desc, n, c, h, w)


alias cudnnLRNDescriptor_t = UnsafePointer[cudnnLRNStruct]


@value
@register_passable("trivial")
struct cudnnSamplerType_t(Writable):
    var _value: Int8
    alias CUDNN_SAMPLER_BILINEAR = Self(0)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_SAMPLER_BILINEAR:
            return writer.write("CUDNN_SAMPLER_BILINEAR")
        abort("invalid cudnnSamplerType_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnSamplerType_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSpatialTfSamplerForward(
    handle: UnsafePointer[cudnnContext],
    st_desc: UnsafePointer[cudnnSpatialTransformerStruct],
    alpha: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    grid: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSpatialTfSamplerForward",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnSpatialTransformerStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, st_desc, alpha, x_desc, x, grid, beta, y_desc, y)


@value
@register_passable("trivial")
struct cudnnNormMode_t(Writable):
    var _value: Int8
    alias CUDNN_NORM_PER_ACTIVATION = Self(0)
    alias CUDNN_NORM_PER_CHANNEL = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_NORM_PER_ACTIVATION:
            return writer.write("CUDNN_NORM_PER_ACTIVATION")
        if self is Self.CUDNN_NORM_PER_CHANNEL:
            return writer.write("CUDNN_NORM_PER_CHANNEL")
        abort("invalid cudnnNormMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnNormMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSetPooling2dDescriptor(
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
    mode: cudnnPoolingMode_t,
    maxpooling_nan_opt: cudnnNanPropagation_t,
    window_height: Int16,
    window_width: Int16,
    vertical_padding: Int16,
    horizontal_padding: Int16,
    vertical_stride: Int16,
    horizontal_stride: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetPooling2dDescriptor",
        fn (
            UnsafePointer[cudnnPoolingStruct],
            cudnnPoolingMode_t,
            cudnnNanPropagation_t,
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
        ) -> cudnnStatus_t,
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


fn cudnnGetPooling2dDescriptor(
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
    mode: UnsafePointer[cudnnPoolingMode_t],
    maxpooling_nan_opt: UnsafePointer[cudnnNanPropagation_t],
    window_height: UnsafePointer[Int16],
    window_width: UnsafePointer[Int16],
    vertical_padding: UnsafePointer[Int16],
    horizontal_padding: UnsafePointer[Int16],
    vertical_stride: UnsafePointer[Int16],
    horizontal_stride: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPooling2dDescriptor",
        fn (
            UnsafePointer[cudnnPoolingStruct],
            UnsafePointer[cudnnPoolingMode_t],
            UnsafePointer[cudnnNanPropagation_t],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
        ) -> cudnnStatus_t,
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


@value
@register_passable("trivial")
struct cudnnNormOps_t(Writable):
    var _value: Int8
    alias CUDNN_NORM_OPS_NORM = Self(0)
    alias CUDNN_NORM_OPS_NORM_ACTIVATION = Self(1)
    alias CUDNN_NORM_OPS_NORM_ADD_ACTIVATION = Self(2)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_NORM_OPS_NORM:
            return writer.write("CUDNN_NORM_OPS_NORM")
        if self is Self.CUDNN_NORM_OPS_NORM_ACTIVATION:
            return writer.write("CUDNN_NORM_OPS_NORM_ACTIVATION")
        if self is Self.CUDNN_NORM_OPS_NORM_ADD_ACTIVATION:
            return writer.write("CUDNN_NORM_OPS_NORM_ADD_ACTIVATION")
        abort("invalid cudnnNormOps_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnNormOps_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSoftmaxForward(
    handle: UnsafePointer[cudnnContext],
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSoftmaxForward",
        fn (
            UnsafePointer[cudnnContext],
            cudnnSoftmaxAlgorithm_t,
            cudnnSoftmaxMode_t,
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, algo, mode, alpha, x_desc, x, beta, y_desc, y)


alias cudnnSpatialTransformerDescriptor_t = UnsafePointer[
    cudnnSpatialTransformerStruct
]


@value
@register_passable("trivial")
struct cudnnSoftmaxAlgorithm_t(Writable):
    var _value: Int8
    alias CUDNN_SOFTMAX_FAST = Self(0)
    alias CUDNN_SOFTMAX_ACCURATE = Self(1)
    alias CUDNN_SOFTMAX_LOG = Self(2)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_SOFTMAX_FAST:
            return writer.write("CUDNN_SOFTMAX_FAST")
        if self is Self.CUDNN_SOFTMAX_ACCURATE:
            return writer.write("CUDNN_SOFTMAX_ACCURATE")
        if self is Self.CUDNN_SOFTMAX_LOG:
            return writer.write("CUDNN_SOFTMAX_LOG")
        abort("invalid cudnnSoftmaxAlgorithm_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnSoftmaxAlgorithm_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetErrorString(status: cudnnStatus_t) -> UnsafePointer[Int8]:
    return _get_dylib_function[
        "cudnnGetErrorString", fn (cudnnStatus_t) -> UnsafePointer[Int8]
    ]()(status)


fn cudnnPoolingForward(
    handle: UnsafePointer[cudnnContext],
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
    alpha: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnPoolingForward",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnPoolingStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, pooling_desc, alpha, x_desc, x, beta, y_desc, y)


fn cudnnGetStream(
    handle: UnsafePointer[cudnnContext],
    stream_id: UnsafePointer[CUstream],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetStream",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[CUstream],
        ) -> cudnnStatus_t,
    ]()(handle, stream_id)


@value
@register_passable("trivial")
struct cudnnBatchNormOps_t(Writable):
    var _value: Int8
    alias CUDNN_BATCHNORM_OPS_BN = Self(0)
    alias CUDNN_BATCHNORM_OPS_BN_ACTIVATION = Self(1)
    alias CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = Self(2)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_BATCHNORM_OPS_BN:
            return writer.write("CUDNN_BATCHNORM_OPS_BN")
        if self is Self.CUDNN_BATCHNORM_OPS_BN_ACTIVATION:
            return writer.write("CUDNN_BATCHNORM_OPS_BN_ACTIVATION")
        if self is Self.CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION:
            return writer.write("CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION")
        abort("invalid cudnnBatchNormOps_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnBatchNormOps_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct cudnnConvolutionFwdAlgo_t(Writable):
    var _value: Int8
    alias CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = Self(0)
    alias CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = Self(1)
    alias CUDNN_CONVOLUTION_FWD_ALGO_GEMM = Self(2)
    alias CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = Self(3)
    alias CUDNN_CONVOLUTION_FWD_ALGO_FFT = Self(4)
    alias CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = Self(5)
    alias CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = Self(6)
    alias CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = Self(7)
    alias CUDNN_CONVOLUTION_FWD_ALGO_COUNT = Self(8)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return writer.write(
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
            )
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_GEMM")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_DIRECT")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_FFT")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED")
        if self is Self.CUDNN_CONVOLUTION_FWD_ALGO_COUNT:
            return writer.write("CUDNN_CONVOLUTION_FWD_ALGO_COUNT")
        abort("invalid cudnnConvolutionFwdAlgo_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnConvolutionFwdAlgo_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSaveAlgorithm(
    handle: UnsafePointer[cudnnContext],
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
    algo_space: UnsafePointer[NoneType],
    algo_space_size_in_bytes: Int,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSaveAlgorithm",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnAlgorithmStruct],
            UnsafePointer[NoneType],
            Int,
        ) -> cudnnStatus_t,
    ]()(handle, algo_desc, algo_space, algo_space_size_in_bytes)


fn cudnnCopyAlgorithmDescriptor(
    src: UnsafePointer[cudnnAlgorithmStruct],
    dest: UnsafePointer[cudnnAlgorithmStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCopyAlgorithmDescriptor",
        fn (
            UnsafePointer[cudnnAlgorithmStruct],
            UnsafePointer[cudnnAlgorithmStruct],
        ) -> cudnnStatus_t,
    ]()(src, dest)


fn cudnnDeriveNormTensorDescriptor(
    derived_norm_scale_bias_desc: UnsafePointer[cudnnTensorStruct],
    derived_norm_mean_var_desc: UnsafePointer[cudnnTensorStruct],
    x_desc: UnsafePointer[cudnnTensorStruct],
    mode: cudnnNormMode_t,
    group_cnt: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDeriveNormTensorDescriptor",
        fn (
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            cudnnNormMode_t,
            Int16,
        ) -> cudnnStatus_t,
    ]()(
        derived_norm_scale_bias_desc,
        derived_norm_mean_var_desc,
        x_desc,
        mode,
        group_cnt,
    )


fn cudnnTransformFilter(
    handle: UnsafePointer[cudnnContext],
    trans_desc: UnsafePointer[cudnnTensorTransformStruct],
    alpha: UnsafePointer[NoneType],
    src_desc: UnsafePointer[cudnnFilterStruct],
    src_data: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    dest_desc: UnsafePointer[cudnnFilterStruct],
    dest_data: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnTransformFilter",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorTransformStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
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


fn cudnnOpsInferVersionCheck() -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnOpsInferVersionCheck", fn () -> cudnnStatus_t
    ]()()


fn cudnnActivationForward(
    handle: UnsafePointer[cudnnContext],
    activation_desc: UnsafePointer[cudnnActivationStruct],
    alpha: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnActivationForward",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnActivationStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y)


fn cudnnSetAlgorithmPerformance(
    algo_perf: UnsafePointer[cudnnAlgorithmPerformanceStruct],
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
    status: cudnnStatus_t,
    time: Float32,
    memory: Int,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetAlgorithmPerformance",
        fn (
            UnsafePointer[cudnnAlgorithmPerformanceStruct],
            UnsafePointer[cudnnAlgorithmStruct],
            cudnnStatus_t,
            Float32,
            Int,
        ) -> cudnnStatus_t,
    ]()(algo_perf, algo_desc, status, time, memory)


fn cudnnCreateActivationDescriptor(
    activation_desc: UnsafePointer[UnsafePointer[cudnnActivationStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateActivationDescriptor",
        fn (
            UnsafePointer[UnsafePointer[cudnnActivationStruct]],
        ) -> cudnnStatus_t,
    ]()(activation_desc)


@value
struct libraryPropertyType_t:
    var _value: Int32
    alias MAJOR_VERSION = Self(0)
    alias MINOR_VERSION = Self(1)
    alias PATCH_LEVEL = Self(2)


fn cudnnGetProperty(
    type: libraryPropertyType_t, value: UnsafePointer[Int16]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetProperty",
        fn (libraryPropertyType_t, UnsafePointer[Int16]) -> cudnnStatus_t,
    ]()(type, value)


fn cudnnDestroyPoolingDescriptor(
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyPoolingDescriptor",
        fn (UnsafePointer[cudnnPoolingStruct]) -> cudnnStatus_t,
    ]()(pooling_desc)


fn cudnnGetFilterSizeInBytes(
    filter_desc: UnsafePointer[cudnnFilterStruct], size: UnsafePointer[Int]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFilterSizeInBytes",
        fn (
            UnsafePointer[cudnnFilterStruct], UnsafePointer[Int]
        ) -> cudnnStatus_t,
    ]()(filter_desc, size)


@value
@register_passable("trivial")
struct cudnnLRNMode_t(Writable):
    var _value: Int8
    alias CUDNN_LRN_CROSS_CHANNEL_DIM1 = Self(0)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_LRN_CROSS_CHANNEL_DIM1:
            return writer.write("CUDNN_LRN_CROSS_CHANNEL_DIM1")
        abort("invalid cudnnLRNMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnLRNMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSetTensorNdDescriptorEx(
    tensor_desc: UnsafePointer[cudnnTensorStruct],
    format: cudnnTensorFormat_t,
    data_type: cudnnDataType_t,
    nb_dims: Int16,
    dim_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensorNdDescriptorEx",
        fn (
            UnsafePointer[cudnnTensorStruct],
            cudnnTensorFormat_t,
            cudnnDataType_t,
            Int16,
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(tensor_desc, format, data_type, nb_dims, dim_a)


fn cudnnSetTensorNdDescriptor(
    tensor_desc: UnsafePointer[cudnnTensorStruct],
    data_type: cudnnDataType_t,
    nb_dims: Int16,
    dim_a: UnsafePointer[NoneType],
    stride_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensorNdDescriptor",
        fn (
            UnsafePointer[cudnnTensorStruct],
            cudnnDataType_t,
            Int16,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(tensor_desc, data_type, nb_dims, dim_a, stride_a)


fn cudnnTransformTensorEx(
    handle: UnsafePointer[cudnnContext],
    trans_desc: UnsafePointer[cudnnTensorTransformStruct],
    alpha: UnsafePointer[NoneType],
    src_desc: UnsafePointer[cudnnTensorStruct],
    src_data: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    dest_desc: UnsafePointer[cudnnTensorStruct],
    dest_data: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnTransformTensorEx",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorTransformStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
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


fn cudnnGetAlgorithmDescriptor(
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
    algorithm: UnsafePointer[cudnnAlgorithmUnionStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAlgorithmDescriptor",
        fn (
            UnsafePointer[cudnnAlgorithmStruct],
            UnsafePointer[cudnnAlgorithmUnionStruct],
        ) -> cudnnStatus_t,
    ]()(algo_desc, algorithm)


@value
@register_passable("trivial")
struct cudnnFoldingDirection_t(Writable):
    var _value: Int8
    alias CUDNN_TRANSFORM_FOLD = Self(0)
    alias CUDNN_TRANSFORM_UNFOLD = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_TRANSFORM_FOLD:
            return writer.write("CUDNN_TRANSFORM_FOLD")
        if self is Self.CUDNN_TRANSFORM_UNFOLD:
            return writer.write("CUDNN_TRANSFORM_UNFOLD")
        abort("invalid cudnnFoldingDirection_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnFoldingDirection_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetTensorNdDescriptor(
    tensor_desc: UnsafePointer[cudnnTensorStruct],
    nb_dims_requested: Int16,
    data_type: UnsafePointer[cudnnDataType_t],
    nb_dims: UnsafePointer[Int16],
    dim_a: UnsafePointer[NoneType],
    stride_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensorNdDescriptor",
        fn (
            UnsafePointer[cudnnTensorStruct],
            Int16,
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[Int16],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(tensor_desc, nb_dims_requested, data_type, nb_dims, dim_a, stride_a)


@value
@register_passable("trivial")
struct cudnnErrQueryMode_t(Writable):
    var _value: Int8
    alias CUDNN_ERRQUERY_RAWCODE = Self(0)
    alias CUDNN_ERRQUERY_NONBLOCKING = Self(1)
    alias CUDNN_ERRQUERY_BLOCKING = Self(2)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_ERRQUERY_RAWCODE:
            return writer.write("CUDNN_ERRQUERY_RAWCODE")
        if self is Self.CUDNN_ERRQUERY_NONBLOCKING:
            return writer.write("CUDNN_ERRQUERY_NONBLOCKING")
        if self is Self.CUDNN_ERRQUERY_BLOCKING:
            return writer.write("CUDNN_ERRQUERY_BLOCKING")
        abort("invalid cudnnErrQueryMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnErrQueryMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetOpTensorDescriptor(
    op_tensor_desc: UnsafePointer[cudnnOpTensorStruct],
    op_tensor_op: UnsafePointer[cudnnOpTensorOp_t],
    op_tensor_comp_type: UnsafePointer[cudnnDataType_t],
    op_tensor_nan_opt: UnsafePointer[cudnnNanPropagation_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetOpTensorDescriptor",
        fn (
            UnsafePointer[cudnnOpTensorStruct],
            UnsafePointer[cudnnOpTensorOp_t],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnNanPropagation_t],
        ) -> cudnnStatus_t,
    ]()(op_tensor_desc, op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt)


fn cudnnGetReductionIndicesSize(
    handle: UnsafePointer[cudnnContext],
    reduce_tensor_desc: UnsafePointer[cudnnReduceTensorStruct],
    a_desc: UnsafePointer[cudnnTensorStruct],
    c_desc: UnsafePointer[cudnnTensorStruct],
    size_in_bytes: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetReductionIndicesSize",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnReduceTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(handle, reduce_tensor_desc, a_desc, c_desc, size_in_bytes)


fn cudnnTransformTensor(
    handle: UnsafePointer[cudnnContext],
    alpha: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnTransformTensor",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, alpha, x_desc, x, beta, y_desc, y)


alias cudnnCallback_t = fn (
    cudnnSeverity_t,
    UnsafePointer[NoneType],
    UnsafePointer[cudnnDebugStruct],
    UnsafePointer[Int8],
) -> NoneType


@register_passable("trivial")
struct cudnnAlgorithmUnionStruct:
    var algo: UnsafePointer[NoneType]


alias cudnnDropoutDescriptor_t = UnsafePointer[cudnnDropoutStruct]


fn cudnnSetTensor4dDescriptorEx(
    tensor_desc: UnsafePointer[cudnnTensorStruct],
    data_type: cudnnDataType_t,
    n: Int16,
    c: Int16,
    h: Int16,
    w: Int16,
    n_stride: Int16,
    c_stride: Int16,
    h_stride: Int16,
    w_stride: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensor4dDescriptorEx",
        fn (
            UnsafePointer[cudnnTensorStruct],
            cudnnDataType_t,
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
        ) -> cudnnStatus_t,
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


@value
@register_passable("trivial")
struct cudnnBatchNormMode_t(Writable):
    var _value: Int8
    alias CUDNN_BATCHNORM_PER_ACTIVATION = Self(0)
    alias CUDNN_BATCHNORM_SPATIAL = Self(1)
    alias CUDNN_BATCHNORM_SPATIAL_PERSISTENT = Self(2)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_BATCHNORM_PER_ACTIVATION:
            return writer.write("CUDNN_BATCHNORM_PER_ACTIVATION")
        if self is Self.CUDNN_BATCHNORM_SPATIAL:
            return writer.write("CUDNN_BATCHNORM_SPATIAL")
        if self is Self.CUDNN_BATCHNORM_SPATIAL_PERSISTENT:
            return writer.write("CUDNN_BATCHNORM_SPATIAL_PERSISTENT")
        abort("invalid cudnnBatchNormMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnBatchNormMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias cudnnCTCLossDescriptor_t = UnsafePointer[cudnnCTCLossStruct]


fn cudnnGetLRNDescriptor(
    norm_desc: UnsafePointer[cudnnLRNStruct],
    lrn_n: UnsafePointer[Int16],
    lrn_alpha: UnsafePointer[Float64],
    lrn_beta: UnsafePointer[Float64],
    lrn_k: UnsafePointer[Float64],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetLRNDescriptor",
        fn (
            UnsafePointer[cudnnLRNStruct],
            UnsafePointer[Int16],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
        ) -> cudnnStatus_t,
    ]()(norm_desc, lrn_n, lrn_alpha, lrn_beta, lrn_k)


alias cudnnAlgorithmPerformance_t = UnsafePointer[
    cudnnAlgorithmPerformanceStruct
]


fn cudnnScaleTensor(
    handle: UnsafePointer[cudnnContext],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
    alpha: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnScaleTensor",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, y_desc, y, alpha)


@value
@register_passable("trivial")
struct cudnnSeverity_t(Writable):
    var _value: Int8
    alias CUDNN_SEV_FATAL = Self(0)
    alias CUDNN_SEV_ERROR = Self(1)
    alias CUDNN_SEV_WARNING = Self(2)
    alias CUDNN_SEV_INFO = Self(3)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_SEV_FATAL:
            return writer.write("CUDNN_SEV_FATAL")
        if self is Self.CUDNN_SEV_ERROR:
            return writer.write("CUDNN_SEV_ERROR")
        if self is Self.CUDNN_SEV_WARNING:
            return writer.write("CUDNN_SEV_WARNING")
        if self is Self.CUDNN_SEV_INFO:
            return writer.write("CUDNN_SEV_INFO")
        abort("invalid cudnnSeverity_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnSeverity_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias cudnnDebug_t = cudnnDebugStruct


@value
@register_passable("trivial")
struct cudnnMathType_t(Writable):
    var _value: Int8
    alias CUDNN_DEFAULT_MATH = Self(0)
    alias CUDNN_TENSOR_OP_MATH = Self(1)
    alias CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = Self(2)
    alias CUDNN_FMA_MATH = Self(3)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_DEFAULT_MATH:
            return writer.write("CUDNN_DEFAULT_MATH")
        if self is Self.CUDNN_TENSOR_OP_MATH:
            return writer.write("CUDNN_TENSOR_OP_MATH")
        if self is Self.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
            return writer.write("CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION")
        if self is Self.CUDNN_FMA_MATH:
            return writer.write("CUDNN_FMA_MATH")
        abort("invalid cudnnMathType_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnMathType_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct cudnnNanPropagation_t(Writable):
    var _value: Int8
    alias CUDNN_NOT_PROPAGATE_NAN = Self(0)
    alias CUDNN_PROPAGATE_NAN = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_NOT_PROPAGATE_NAN:
            return writer.write("CUDNN_NOT_PROPAGATE_NAN")
        if self is Self.CUDNN_PROPAGATE_NAN:
            return writer.write("CUDNN_PROPAGATE_NAN")
        abort("invalid cudnnNanPropagation_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnNanPropagation_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias cudnnFilterDescriptor_t = UnsafePointer[cudnnFilterStruct]


@value
@register_passable("trivial")
struct cudnnRNNAlgo_t(Writable):
    var _value: Int8
    alias CUDNN_RNN_ALGO_STANDARD = Self(0)
    alias CUDNN_RNN_ALGO_PERSIST_STATIC = Self(1)
    alias CUDNN_RNN_ALGO_PERSIST_DYNAMIC = Self(2)
    alias CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H = Self(3)
    alias CUDNN_RNN_ALGO_COUNT = Self(4)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_RNN_ALGO_STANDARD:
            return writer.write("CUDNN_RNN_ALGO_STANDARD")
        if self is Self.CUDNN_RNN_ALGO_PERSIST_STATIC:
            return writer.write("CUDNN_RNN_ALGO_PERSIST_STATIC")
        if self is Self.CUDNN_RNN_ALGO_PERSIST_DYNAMIC:
            return writer.write("CUDNN_RNN_ALGO_PERSIST_DYNAMIC")
        if self is Self.CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H:
            return writer.write("CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H")
        if self is Self.CUDNN_RNN_ALGO_COUNT:
            return writer.write("CUDNN_RNN_ALGO_COUNT")
        abort("invalid cudnnRNNAlgo_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnRNNAlgo_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias cudnnOpTensorDescriptor_t = UnsafePointer[cudnnOpTensorStruct]


@register_passable("trivial")
struct Algorithm:
    var convFwdAlgo: cudnnConvolutionFwdAlgo_t
    var convBwdFilterAlgo: cudnnConvolutionBwdFilterAlgo_t
    var convBwdDataAlgo: cudnnConvolutionBwdDataAlgo_t
    var RNNAlgo: cudnnRNNAlgo_t
    var CTCLossAlgo: cudnnCTCLossAlgo_t


fn cudnnGetReductionWorkspaceSize(
    handle: UnsafePointer[cudnnContext],
    reduce_tensor_desc: UnsafePointer[cudnnReduceTensorStruct],
    a_desc: UnsafePointer[cudnnTensorStruct],
    c_desc: UnsafePointer[cudnnTensorStruct],
    size_in_bytes: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetReductionWorkspaceSize",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnReduceTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(handle, reduce_tensor_desc, a_desc, c_desc, size_in_bytes)


fn cudnnSetFilter4dDescriptor(
    filter_desc: UnsafePointer[cudnnFilterStruct],
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: Int16,
    c: Int16,
    h: Int16,
    w: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetFilter4dDescriptor",
        fn (
            UnsafePointer[cudnnFilterStruct],
            cudnnDataType_t,
            cudnnTensorFormat_t,
            Int16,
            Int16,
            Int16,
            Int16,
        ) -> cudnnStatus_t,
    ]()(filter_desc, data_type, format, k, c, h, w)


fn cudnnDestroyActivationDescriptor(
    activation_desc: UnsafePointer[cudnnActivationStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyActivationDescriptor",
        fn (UnsafePointer[cudnnActivationStruct]) -> cudnnStatus_t,
    ]()(activation_desc)


fn cudnnGetAlgorithmSpaceSize(
    handle: UnsafePointer[cudnnContext],
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
    algo_space_size_in_bytes: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAlgorithmSpaceSize",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnAlgorithmStruct],
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(handle, algo_desc, algo_space_size_in_bytes)


@value
@register_passable("trivial")
struct cudnnDataType_t(Writable):
    var _value: Int8
    alias CUDNN_DATA_FLOAT = Self(0)
    alias CUDNN_DATA_DOUBLE = Self(1)
    alias CUDNN_DATA_HALF = Self(2)
    alias CUDNN_DATA_INT8 = Self(3)
    alias CUDNN_DATA_INT32 = Self(4)
    alias CUDNN_DATA_INT8x4 = Self(5)
    alias CUDNN_DATA_UINT8 = Self(6)
    alias CUDNN_DATA_UINT8x4 = Self(7)
    alias CUDNN_DATA_INT8x32 = Self(8)
    alias CUDNN_DATA_BFLOAT16 = Self(9)
    alias CUDNN_DATA_INT64 = Self(10)
    alias CUDNN_DATA_BOOLEAN = Self(11)
    alias CUDNN_DATA_FP8_E4M3 = Self(12)
    alias CUDNN_DATA_FP8_E5M2 = Self(13)
    alias CUDNN_DATA_FAST_FLOAT_FOR_FP8 = Self(14)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_DATA_FLOAT:
            return writer.write("CUDNN_DATA_FLOAT")
        if self is Self.CUDNN_DATA_DOUBLE:
            return writer.write("CUDNN_DATA_DOUBLE")
        if self is Self.CUDNN_DATA_HALF:
            return writer.write("CUDNN_DATA_HALF")
        if self is Self.CUDNN_DATA_INT8:
            return writer.write("CUDNN_DATA_INT8")
        if self is Self.CUDNN_DATA_INT32:
            return writer.write("CUDNN_DATA_INT32")
        if self is Self.CUDNN_DATA_INT8x4:
            return writer.write("CUDNN_DATA_INT8x4")
        if self is Self.CUDNN_DATA_UINT8:
            return writer.write("CUDNN_DATA_UINT8")
        if self is Self.CUDNN_DATA_UINT8x4:
            return writer.write("CUDNN_DATA_UINT8x4")
        if self is Self.CUDNN_DATA_INT8x32:
            return writer.write("CUDNN_DATA_INT8x32")
        if self is Self.CUDNN_DATA_BFLOAT16:
            return writer.write("CUDNN_DATA_BFLOAT16")
        if self is Self.CUDNN_DATA_INT64:
            return writer.write("CUDNN_DATA_INT64")
        if self is Self.CUDNN_DATA_BOOLEAN:
            return writer.write("CUDNN_DATA_BOOLEAN")
        if self is Self.CUDNN_DATA_FP8_E4M3:
            return writer.write("CUDNN_DATA_FP8_E4M3")
        if self is Self.CUDNN_DATA_FP8_E5M2:
            return writer.write("CUDNN_DATA_FP8_E5M2")
        if self is Self.CUDNN_DATA_FAST_FLOAT_FOR_FP8:
            return writer.write("CUDNN_DATA_FAST_FLOAT_FOR_FP8")
        abort("invalid cudnnDataType_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnDataType_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSetLRNDescriptor(
    norm_desc: UnsafePointer[cudnnLRNStruct],
    lrn_n: Int16,
    lrn_alpha: Float64,
    lrn_beta: Float64,
    lrn_k: Float64,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetLRNDescriptor",
        fn (
            UnsafePointer[cudnnLRNStruct], Int16, Float64, Float64, Float64
        ) -> cudnnStatus_t,
    ]()(norm_desc, lrn_n, lrn_alpha, lrn_beta, lrn_k)


fn cudnnDestroyDropoutDescriptor(
    dropout_desc: UnsafePointer[cudnnDropoutStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyDropoutDescriptor",
        fn (UnsafePointer[cudnnDropoutStruct]) -> cudnnStatus_t,
    ]()(dropout_desc)


fn cudnnGetTensor4dDescriptor(
    tensor_desc: UnsafePointer[cudnnTensorStruct],
    data_type: UnsafePointer[cudnnDataType_t],
    n: UnsafePointer[Int16],
    c: UnsafePointer[Int16],
    h: UnsafePointer[Int16],
    w: UnsafePointer[Int16],
    n_stride: UnsafePointer[Int16],
    c_stride: UnsafePointer[Int16],
    h_stride: UnsafePointer[Int16],
    w_stride: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetTensor4dDescriptor",
        fn (
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
        ) -> cudnnStatus_t,
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


fn cudnnGetAlgorithmPerformance(
    algo_perf: UnsafePointer[cudnnAlgorithmPerformanceStruct],
    algo_desc: UnsafePointer[UnsafePointer[cudnnAlgorithmStruct]],
    status: UnsafePointer[cudnnStatus_t],
    time: UnsafePointer[Float32],
    memory: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAlgorithmPerformance",
        fn (
            UnsafePointer[cudnnAlgorithmPerformanceStruct],
            UnsafePointer[UnsafePointer[cudnnAlgorithmStruct]],
            UnsafePointer[cudnnStatus_t],
            UnsafePointer[Float32],
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(algo_perf, algo_desc, status, time, memory)


@register_passable("trivial")
struct cudnnDebugStruct:
    var cudnn_version: Int16
    var cudnnStatus: cudnnStatus_t
    var time_sec: Int16
    var time_usec: Int16
    var time_delta: Int16
    var handle: UnsafePointer[cudnnContext]
    var stream: CUstream
    var pid: Int64
    var tid: Int64
    var cudaDeviceId: Int16
    var reserved: StaticTuple[Int32, 15]


fn cudnnSetSpatialTransformerNdDescriptor(
    st_desc: UnsafePointer[cudnnSpatialTransformerStruct],
    sampler_type: cudnnSamplerType_t,
    data_type: cudnnDataType_t,
    nb_dims: Int16,
    dim_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetSpatialTransformerNdDescriptor",
        fn (
            UnsafePointer[cudnnSpatialTransformerStruct],
            cudnnSamplerType_t,
            cudnnDataType_t,
            Int16,
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(st_desc, sampler_type, data_type, nb_dims, dim_a)


alias cudnnAlgorithm_t = cudnnAlgorithmUnionStruct


@value
@register_passable("trivial")
struct cudnnIndicesType_t(Writable):
    var _value: Int8
    alias CUDNN_32BIT_INDICES = Self(0)
    alias CUDNN_64BIT_INDICES = Self(1)
    alias CUDNN_16BIT_INDICES = Self(2)
    alias CUDNN_8BIT_INDICES = Self(3)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_32BIT_INDICES:
            return writer.write("CUDNN_32BIT_INDICES")
        if self is Self.CUDNN_64BIT_INDICES:
            return writer.write("CUDNN_64BIT_INDICES")
        if self is Self.CUDNN_16BIT_INDICES:
            return writer.write("CUDNN_16BIT_INDICES")
        if self is Self.CUDNN_8BIT_INDICES:
            return writer.write("CUDNN_8BIT_INDICES")
        abort("invalid cudnnIndicesType_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnIndicesType_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSetTensorTransformDescriptor(
    transform_desc: UnsafePointer[cudnnTensorTransformStruct],
    nb_dims: UInt32,
    dest_format: cudnnTensorFormat_t,
    pad_before_a: UnsafePointer[NoneType],
    pad_after_a: UnsafePointer[NoneType],
    fold_a: UnsafePointer[NoneType],
    direction: cudnnFoldingDirection_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensorTransformDescriptor",
        fn (
            UnsafePointer[cudnnTensorTransformStruct],
            UInt32,
            cudnnTensorFormat_t,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            cudnnFoldingDirection_t,
        ) -> cudnnStatus_t,
    ]()(
        transform_desc,
        nb_dims,
        dest_format,
        pad_before_a,
        pad_after_a,
        fold_a,
        direction,
    )


fn cudnnSetStream(
    handle: UnsafePointer[cudnnContext], stream_id: CUstream
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetStream",
        fn (UnsafePointer[cudnnContext], CUstream) -> cudnnStatus_t,
    ]()(handle, stream_id)


fn cudnnDestroyReduceTensorDescriptor(
    reduce_tensor_desc: UnsafePointer[cudnnReduceTensorStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyReduceTensorDescriptor",
        fn (UnsafePointer[cudnnReduceTensorStruct]) -> cudnnStatus_t,
    ]()(reduce_tensor_desc)


fn cudnnSetTensor(
    handle: UnsafePointer[cudnnContext],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
    value_ptr: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetTensor",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(handle, y_desc, y, value_ptr)


fn cudnnDivisiveNormalizationForward(
    handle: UnsafePointer[cudnnContext],
    norm_desc: UnsafePointer[cudnnLRNStruct],
    mode: cudnnDivNormMode_t,
    alpha: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    means: UnsafePointer[NoneType],
    temp: UnsafePointer[NoneType],
    temp2: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDivisiveNormalizationForward",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnLRNStruct],
            cudnnDivNormMode_t,
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
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


fn cudnnSetActivationDescriptorSwishBeta(
    activation_desc: UnsafePointer[cudnnActivationStruct], swish_beta: Float64
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetActivationDescriptorSwishBeta",
        fn (UnsafePointer[cudnnActivationStruct], Float64) -> cudnnStatus_t,
    ]()(activation_desc, swish_beta)


fn cudnnSetCallback(
    mask: Int16,
    udata: UnsafePointer[NoneType],
    fptr: fn (
        cudnnSeverity_t,
        UnsafePointer[NoneType],
        UnsafePointer[cudnnDebugStruct],
        UnsafePointer[Int8],
    ) -> NoneType,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetCallback",
        fn (
            Int16,
            UnsafePointer[NoneType],
            fn (
                cudnnSeverity_t,
                UnsafePointer[NoneType],
                UnsafePointer[cudnnDebugStruct],
                UnsafePointer[Int8],
            ) -> NoneType,
        ) -> cudnnStatus_t,
    ]()(mask, udata, fptr)


fn cudnnDropoutGetStatesSize(
    handle: UnsafePointer[cudnnContext], size_in_bytes: UnsafePointer[Int]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDropoutGetStatesSize",
        fn (UnsafePointer[cudnnContext], UnsafePointer[Int]) -> cudnnStatus_t,
    ]()(handle, size_in_bytes)


fn cudnnCreateDropoutDescriptor(
    dropout_desc: UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateDropoutDescriptor",
        fn (UnsafePointer[UnsafePointer[cudnnDropoutStruct]]) -> cudnnStatus_t,
    ]()(dropout_desc)


fn cudnnNormalizationForwardInference(
    handle: UnsafePointer[cudnnContext],
    mode: cudnnNormMode_t,
    norm_ops: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: UnsafePointer[NoneType],
    beta: UnsafePointer[NoneType],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: UnsafePointer[NoneType],
    norm_scale_bias_desc: UnsafePointer[cudnnTensorStruct],
    norm_scale: UnsafePointer[NoneType],
    norm_bias: UnsafePointer[NoneType],
    norm_mean_var_desc: UnsafePointer[cudnnTensorStruct],
    estimated_mean: UnsafePointer[NoneType],
    estimated_variance: UnsafePointer[NoneType],
    z_desc: UnsafePointer[cudnnTensorStruct],
    z: UnsafePointer[NoneType],
    activation_desc: UnsafePointer[cudnnActivationStruct],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: UnsafePointer[NoneType],
    epsilon: Float64,
    group_cnt: Int16,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnNormalizationForwardInference",
        fn (
            UnsafePointer[cudnnContext],
            cudnnNormMode_t,
            cudnnNormOps_t,
            cudnnNormAlgo_t,
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            UnsafePointer[cudnnActivationStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[NoneType],
            Float64,
            Int16,
        ) -> cudnnStatus_t,
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


@value
@register_passable("trivial")
struct cudnnConvolutionBwdFilterAlgo_t(Writable):
    var _value: Int8
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = Self(0)
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = Self(1)
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = Self(2)
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = Self(3)
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = Self(4)
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = Self(5)
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = Self(6)
    alias CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = Self(7)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
            return writer.write("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
            return writer.write("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
            return writer.write("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
            return writer.write("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
            return writer.write("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
            return writer.write(
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED"
            )
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
            return writer.write("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING")
        if self is Self.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
            return writer.write("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT")
        abort("invalid cudnnConvolutionBwdFilterAlgo_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnConvolutionBwdFilterAlgo_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnQueryRuntimeError(
    handle: UnsafePointer[cudnnContext],
    rstatus: UnsafePointer[cudnnStatus_t],
    mode: cudnnErrQueryMode_t,
    tag: UnsafePointer[cudnnRuntimeTag_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnQueryRuntimeError",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnStatus_t],
            cudnnErrQueryMode_t,
            UnsafePointer[cudnnRuntimeTag_t],
        ) -> cudnnStatus_t,
    ]()(handle, rstatus, mode, tag)


fn cudnnDestroyLRNDescriptor(
    lrn_desc: UnsafePointer[cudnnLRNStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyLRNDescriptor",
        fn (UnsafePointer[cudnnLRNStruct]) -> cudnnStatus_t,
    ]()(lrn_desc)


fn cudnnDestroyTensorTransformDescriptor(
    transform_desc: UnsafePointer[cudnnTensorTransformStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyTensorTransformDescriptor",
        fn (UnsafePointer[cudnnTensorTransformStruct]) -> cudnnStatus_t,
    ]()(transform_desc)


fn cudnnSetReduceTensorDescriptor(
    reduce_tensor_desc: UnsafePointer[cudnnReduceTensorStruct],
    reduce_tensor_op: cudnnReduceTensorOp_t,
    reduce_tensor_comp_type: cudnnDataType_t,
    reduce_tensor_nan_opt: cudnnNanPropagation_t,
    reduce_tensor_indices: cudnnReduceTensorIndices_t,
    reduce_tensor_indices_type: cudnnIndicesType_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetReduceTensorDescriptor",
        fn (
            UnsafePointer[cudnnReduceTensorStruct],
            cudnnReduceTensorOp_t,
            cudnnDataType_t,
            cudnnNanPropagation_t,
            cudnnReduceTensorIndices_t,
            cudnnIndicesType_t,
        ) -> cudnnStatus_t,
    ]()(
        reduce_tensor_desc,
        reduce_tensor_op,
        reduce_tensor_comp_type,
        reduce_tensor_nan_opt,
        reduce_tensor_indices,
        reduce_tensor_indices_type,
    )


fn cudnnSetAlgorithmDescriptor(
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
    algorithm: cudnnAlgorithmUnionStruct,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetAlgorithmDescriptor",
        fn (
            UnsafePointer[cudnnAlgorithmStruct], cudnnAlgorithmUnionStruct
        ) -> cudnnStatus_t,
    ]()(algo_desc, algorithm)


fn cudnnCreateFilterDescriptor(
    filter_desc: UnsafePointer[UnsafePointer[cudnnFilterStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateFilterDescriptor",
        fn (UnsafePointer[UnsafePointer[cudnnFilterStruct]]) -> cudnnStatus_t,
    ]()(filter_desc)


alias cudnnHandle_t = UnsafePointer[cudnnContext]

alias cudnnPoolingDescriptor_t = UnsafePointer[cudnnPoolingStruct]


fn cudnnDestroyOpTensorDescriptor(
    op_tensor_desc: UnsafePointer[cudnnOpTensorStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyOpTensorDescriptor",
        fn (UnsafePointer[cudnnOpTensorStruct]) -> cudnnStatus_t,
    ]()(op_tensor_desc)


@value
@register_passable("trivial")
struct cudnnPoolingMode_t(Writable):
    var _value: Int8
    alias CUDNN_POOLING_MAX = Self(0)
    alias CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = Self(1)
    alias CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = Self(2)
    alias CUDNN_POOLING_MAX_DETERMINISTIC = Self(3)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_POOLING_MAX:
            return writer.write("CUDNN_POOLING_MAX")
        if self is Self.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
            return writer.write("CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING")
        if self is Self.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
            return writer.write("CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING")
        if self is Self.CUDNN_POOLING_MAX_DETERMINISTIC:
            return writer.write("CUDNN_POOLING_MAX_DETERMINISTIC")
        abort("invalid cudnnPoolingMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnPoolingMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetMaxDeviceVersion() -> Int:
    return _get_dylib_function["cudnnGetMaxDeviceVersion", fn () -> Int]()()


fn cudnnCreatePoolingDescriptor(
    pooling_desc: UnsafePointer[UnsafePointer[cudnnPoolingStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreatePoolingDescriptor",
        fn (UnsafePointer[UnsafePointer[cudnnPoolingStruct]]) -> cudnnStatus_t,
    ]()(pooling_desc)


fn cudnnRestoreDropoutDescriptor(
    dropout_desc: UnsafePointer[cudnnDropoutStruct],
    handle: UnsafePointer[cudnnContext],
    dropout: Float32,
    states: UnsafePointer[NoneType],
    state_size_in_bytes: Int,
    seed: Int64,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRestoreDropoutDescriptor",
        fn (
            UnsafePointer[cudnnDropoutStruct],
            UnsafePointer[cudnnContext],
            Float32,
            UnsafePointer[NoneType],
            Int,
            Int64,
        ) -> cudnnStatus_t,
    ]()(dropout_desc, handle, dropout, states, state_size_in_bytes, seed)


fn cudnnGetDropoutDescriptor(
    dropout_desc: UnsafePointer[cudnnDropoutStruct],
    handle: UnsafePointer[cudnnContext],
    dropout: UnsafePointer[Float32],
    states: UnsafePointer[UnsafePointer[NoneType]],
    seed: UnsafePointer[Int64],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetDropoutDescriptor",
        fn (
            UnsafePointer[cudnnDropoutStruct],
            UnsafePointer[cudnnContext],
            UnsafePointer[Float32],
            UnsafePointer[UnsafePointer[NoneType]],
            UnsafePointer[Int64],
        ) -> cudnnStatus_t,
    ]()(dropout_desc, handle, dropout, states, seed)


@value
@register_passable("trivial")
struct cudnnDivNormMode_t(Writable):
    var _value: Int8
    alias CUDNN_DIVNORM_PRECOMPUTED_MEANS = Self(0)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUDNN_DIVNORM_PRECOMPUTED_MEANS:
            return writer.write("CUDNN_DIVNORM_PRECOMPUTED_MEANS")
        abort("invalid cudnnDivNormMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnDivNormMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnCreateOpTensorDescriptor(
    op_tensor_desc: UnsafePointer[UnsafePointer[cudnnOpTensorStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateOpTensorDescriptor",
        fn (UnsafePointer[UnsafePointer[cudnnOpTensorStruct]]) -> cudnnStatus_t,
    ]()(op_tensor_desc)


fn cudnnSetFilterNdDescriptor(
    filter_desc: UnsafePointer[cudnnFilterStruct],
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nb_dims: Int16,
    filter_dim_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetFilterNdDescriptor",
        fn (
            UnsafePointer[cudnnFilterStruct],
            cudnnDataType_t,
            cudnnTensorFormat_t,
            Int16,
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
    ]()(filter_desc, data_type, format, nb_dims, filter_dim_a)


fn cudnnRestoreAlgorithm(
    handle: UnsafePointer[cudnnContext],
    algo_space: UnsafePointer[NoneType],
    algo_space_size_in_bytes: Int,
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRestoreAlgorithm",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[cudnnAlgorithmStruct],
        ) -> cudnnStatus_t,
    ]()(handle, algo_space, algo_space_size_in_bytes, algo_desc)


fn cudnnGetPoolingNdDescriptor(
    pooling_desc: UnsafePointer[cudnnPoolingStruct],
    nb_dims_requested: Int16,
    mode: UnsafePointer[cudnnPoolingMode_t],
    maxpooling_nan_opt: UnsafePointer[cudnnNanPropagation_t],
    nb_dims: UnsafePointer[Int16],
    window_dim_a: UnsafePointer[NoneType],
    padding_a: UnsafePointer[NoneType],
    stride_a: UnsafePointer[NoneType],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetPoolingNdDescriptor",
        fn (
            UnsafePointer[cudnnPoolingStruct],
            Int16,
            UnsafePointer[cudnnPoolingMode_t],
            UnsafePointer[cudnnNanPropagation_t],
            UnsafePointer[Int16],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
        ) -> cudnnStatus_t,
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


fn cudnnSetDropoutDescriptor(
    dropout_desc: UnsafePointer[cudnnDropoutStruct],
    handle: UnsafePointer[cudnnContext],
    dropout: Float32,
    states: UnsafePointer[NoneType],
    state_size_in_bytes: Int,
    seed: Int64,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetDropoutDescriptor",
        fn (
            UnsafePointer[cudnnDropoutStruct],
            UnsafePointer[cudnnContext],
            Float32,
            UnsafePointer[NoneType],
            Int,
            Int64,
        ) -> cudnnStatus_t,
    ]()(dropout_desc, handle, dropout, states, state_size_in_bytes, seed)


fn cudnnCreateSpatialTransformerDescriptor(
    st_desc: UnsafePointer[UnsafePointer[cudnnSpatialTransformerStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateSpatialTransformerDescriptor",
        fn (
            UnsafePointer[UnsafePointer[cudnnSpatialTransformerStruct]],
        ) -> cudnnStatus_t,
    ]()(st_desc)


fn cudnnInitTransformDest(
    transform_desc: UnsafePointer[cudnnTensorTransformStruct],
    src_desc: UnsafePointer[cudnnTensorStruct],
    dest_desc: UnsafePointer[cudnnTensorStruct],
    dest_size_in_bytes: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnInitTransformDest",
        fn (
            UnsafePointer[cudnnTensorTransformStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(transform_desc, src_desc, dest_desc, dest_size_in_bytes)


fn cudnnCreateAlgorithmDescriptor(
    algo_desc: UnsafePointer[UnsafePointer[cudnnAlgorithmStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateAlgorithmDescriptor",
        fn (
            UnsafePointer[UnsafePointer[cudnnAlgorithmStruct]],
        ) -> cudnnStatus_t,
    ]()(algo_desc)


fn cudnnDestroyTensorDescriptor(
    tensor_desc: UnsafePointer[cudnnTensorStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyTensorDescriptor",
        fn (UnsafePointer[cudnnTensorStruct]) -> cudnnStatus_t,
    ]()(tensor_desc)
