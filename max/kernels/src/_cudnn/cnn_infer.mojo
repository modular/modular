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

from std.utils import StaticTuple

from .infer import (
    cudnnContext,
    cudnnConvolutionBwdDataAlgo_t,
    cudnnConvolutionFwdAlgo_t,
    cudnnDataType_t,
    cudnnDeterminism_t,
    cudnnMathType_t,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    AnyOpaquePointer,
    DoubleNestedPointer,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS: List[Path] = [
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
        ", ".join(materialize[CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUDNN_CNN_INFER_LIBRARY = _Global[
    "CUDA_CUDNN_CNN_INFER_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_CNN_INFER_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

comptime cudnnTensorStruct = AnyOpaquePointer
comptime cudnnConvolutionStruct = AnyOpaquePointer
comptime cudnnFilterStruct = AnyOpaquePointer


struct cudnnConvolutionFwdAlgoPerfStruct(TrivialRegisterPassable):
    var algo: cudnnConvolutionFwdAlgo_t
    var status: cudnnStatus_t
    var time: Float32
    var memory: Int
    var determinism: cudnnDeterminism_t
    var mathType: cudnnMathType_t
    var reserved: StaticTuple[Int32, 3]


def cudnnSetConvolution2dDescriptor(
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    pad_h: Int32,
    pad_w: Int32,
    u: Int32,
    v: Int32,
    dilation_h: Int32,
    dilation_w: Int32,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolution2dDescriptor",
        def(
            type_of(conv_desc),
            type_of(pad_h),
            type_of(pad_w),
            type_of(u),
            type_of(v),
            type_of(dilation_h),
            type_of(dilation_w),
            type_of(mode),
            type_of(compute_type),
        ) thin -> cudnnStatus_t,
    ]()(
        conv_desc,
        pad_h,
        pad_w,
        u,
        v,
        dilation_h,
        dilation_w,
        mode,
        compute_type,
    )


def cudnnCreateConvolutionDescriptor(
    conv_desc: DoubleNestedPointer[cudnnConvolutionStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateConvolutionDescriptor",
        def(type_of(conv_desc)) thin -> cudnnStatus_t,
    ]()(conv_desc)


def cudnnSetConvolutionGroupCount(
    conv_desc: Pointer[cudnnConvolutionStruct, _], group_count: Int32
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionGroupCount",
        def(type_of(conv_desc), type_of(group_count)) thin -> cudnnStatus_t,
    ]()(conv_desc, group_count)


comptime cudnnConvolutionFwdAlgoPerf_t = cudnnConvolutionFwdAlgoPerfStruct


def cudnnGetConvolutionForwardWorkspaceSize(
    handle: Pointer[cudnnContext, _],
    x_desc: Pointer[cudnnTensorStruct, _],
    w_desc: Pointer[cudnnFilterStruct, _],
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    y_desc: Pointer[cudnnTensorStruct, _],
    algo: cudnnConvolutionFwdAlgo_t,
    size_in_bytes: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardWorkspaceSize",
        def(
            type_of(handle),
            type_of(x_desc),
            type_of(w_desc),
            type_of(conv_desc),
            type_of(y_desc),
            type_of(algo),
            type_of(size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, x_desc, w_desc, conv_desc, y_desc, algo, size_in_bytes)


def cudnnGetConvolutionForwardAlgorithm_v7(
    handle: Pointer[cudnnContext, _],
    src_desc: Pointer[cudnnTensorStruct, _],
    filter_desc: Pointer[cudnnFilterStruct, _],
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    dest_desc: Pointer[cudnnTensorStruct, _],
    requested_algo_count: Int32,
    returned_algo_count: Pointer[Int32, _],
    perf_results: Pointer[cudnnConvolutionFwdAlgoPerfStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardAlgorithm_v7",
        def(
            type_of(handle),
            type_of(src_desc),
            type_of(filter_desc),
            type_of(conv_desc),
            type_of(dest_desc),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        src_desc,
        filter_desc,
        conv_desc,
        dest_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


def cudnnDestroyConvolutionDescriptor(
    conv_desc: Pointer[cudnnConvolutionStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyConvolutionDescriptor",
        def(type_of(conv_desc)) thin -> cudnnStatus_t,
    ]()(conv_desc)


def cudnnConvolutionForward(
    handle: Pointer[cudnnContext, _],
    alpha: OpaquePointer,
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    w_desc: Pointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    algo: cudnnConvolutionFwdAlgo_t,
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
    beta: OpaquePointer,
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionForward",
        def(
            type_of(handle),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(w_desc),
            type_of(w),
            type_of(conv_desc),
            type_of(algo),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        y_desc,
        y,
    )


def cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle: Pointer[cudnnContext, _],
    w_desc: Pointer[cudnnFilterStruct, _],
    dy_desc: Pointer[cudnnTensorStruct, _],
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    dx_desc: Pointer[cudnnTensorStruct, _],
    algo: cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: Pointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionBackwardDataWorkspaceSize",
        def(
            type_of(handle),
            type_of(w_desc),
            type_of(dy_desc),
            type_of(conv_desc),
            type_of(dx_desc),
            type_of(algo),
            type_of(size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, w_desc, dy_desc, conv_desc, dx_desc, algo, size_in_bytes)


def cudnnConvolutionBackwardData(
    handle: Pointer[cudnnContext, _],
    alpha: OpaquePointer,
    w_desc: Pointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    dy_desc: Pointer[cudnnTensorStruct, _],
    dy: OpaquePointer,
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    algo: cudnnConvolutionBwdDataAlgo_t,
    work_space: OptionalPointer[NoneType, MutAnyOrigin],
    work_space_size_in_bytes: Int,
    beta: OpaquePointer,
    dx_desc: Pointer[cudnnTensorStruct, _],
    dx: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionBackwardData",
        def(
            type_of(handle),
            type_of(alpha),
            type_of(w_desc),
            type_of(w),
            type_of(dy_desc),
            type_of(dy),
            type_of(conv_desc),
            type_of(algo),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
            type_of(beta),
            type_of(dx_desc),
            type_of(dx),
        ) thin abi("C") -> cudnnStatus_t,
    ]()(
        handle,
        alpha,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        dx_desc,
        dx,
    )


def cudnnSetConvolutionNdDescriptor(
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    array_length: Int32,
    pad_a: OpaquePointer,
    filter_stride_a: OpaquePointer,
    dilation_a: OpaquePointer,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionNdDescriptor",
        def(
            type_of(conv_desc),
            type_of(array_length),
            type_of(pad_a),
            type_of(filter_stride_a),
            type_of(dilation_a),
            type_of(mode),
            type_of(compute_type),
        ) thin -> cudnnStatus_t,
    ]()(
        conv_desc,
        array_length,
        pad_a,
        filter_stride_a,
        dilation_a,
        mode,
        compute_type,
    )


def cudnnSetConvolutionMathType(
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    math_type: cudnnMathType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionMathType",
        def(type_of(conv_desc), type_of(math_type)) thin -> cudnnStatus_t,
    ]()(conv_desc, math_type)


def cudnnFindConvolutionForwardAlgorithmEx(
    handle: Pointer[cudnnContext, _],
    x_desc: Pointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    w_desc: Pointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    conv_desc: Pointer[cudnnConvolutionStruct, _],
    y_desc: Pointer[cudnnTensorStruct, _],
    y: OpaquePointer,
    requested_algo_count: Int32,
    returned_algo_count: Pointer[Int32, _],
    perf_results: Pointer[cudnnConvolutionFwdAlgoPerfStruct, _],
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionForwardAlgorithmEx",
        def(
            type_of(handle),
            type_of(x_desc),
            type_of(x),
            type_of(w_desc),
            type_of(w),
            type_of(conv_desc),
            type_of(y_desc),
            type_of(y),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        y_desc,
        y,
        requested_algo_count,
        returned_algo_count,
        perf_results,
        work_space,
        work_space_size_in_bytes,
    )


@fieldwise_init
struct cudnnConvolutionMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int32
    comptime CUDNN_CONVOLUTION = Self(0)
    comptime CUDNN_CROSS_CORRELATION = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_CONVOLUTION:
            return writer.write_string("CUDNN_CONVOLUTION")
        if self is Self.CUDNN_CROSS_CORRELATION:
            return writer.write_string("CUDNN_CROSS_CORRELATION")
        abort("invalid cudnnConvolutionMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnConvolutionMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)
