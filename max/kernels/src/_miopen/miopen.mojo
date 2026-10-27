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

from max.gpu.host import hipStream_t

from .types import (
    Handle,
    TensorDescriptor,
    ConvolutionDescriptor,
    Status,
    DataType,
    ConvolutionMode,
    ConvFwdAlgorithm,
    ConvAlgoPerf,
)
from .utils import _get_dylib_function

# ===-----------------------------------------------------------------------===#
# Handle Management
# ===-----------------------------------------------------------------------===#


def miopenCreate(handle: OpaquePointer) raises -> Status:
    return _get_dylib_function[
        "miopenCreate",
        def(type_of(handle)) thin -> Status,
    ]()(handle)


def miopenSetStream(handle: Handle, stream: hipStream_t) raises -> Status:
    return _get_dylib_function[
        "miopenSetStream",
        def(Handle, hipStream_t) thin -> Status,
    ]()(handle, stream)


# ===-----------------------------------------------------------------------===#
# Tensor Descriptor Management
# ===-----------------------------------------------------------------------===#


def miopenCreateTensorDescriptor(
    desc: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenCreateTensorDescriptor",
        def(type_of(desc)) thin -> Status,
    ]()(desc)


def miopenSetTensorDescriptorV2(
    desc: TensorDescriptor,
    dtype: DataType,
    num_dims: Int32,
    dims: ImmPointer[UInt64, ImmutAnyOrigin],
    strides: ImmPointer[UInt64, ImmutAnyOrigin],
) raises -> Status:
    """Set an N-D tensor descriptor with explicit dimensions and strides.
    Dims are always in NCHW order; strides describe the physical layout."""
    return _get_dylib_function[
        "miopenSetTensorDescriptorV2",
        def(
            TensorDescriptor,
            DataType,
            Int32,
            type_of(dims),
            type_of(strides),
        ) thin -> Status,
    ]()(desc, dtype, num_dims, dims, strides)


# ===-----------------------------------------------------------------------===#
# Convolution Descriptor Management
# ===-----------------------------------------------------------------------===#


def miopenCreateConvolutionDescriptor(
    desc: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenCreateConvolutionDescriptor",
        def(type_of(desc)) thin -> Status,
    ]()(desc)


def miopenInitConvolutionNdDescriptor(
    desc: ConvolutionDescriptor,
    spatial_dim: Int32,
    pad: ImmPointer[Int32, ImmutAnyOrigin],
    stride: ImmPointer[Int32, ImmutAnyOrigin],
    dilation: ImmPointer[Int32, ImmutAnyOrigin],
    mode: ConvolutionMode,
) raises -> Status:
    return _get_dylib_function[
        "miopenInitConvolutionNdDescriptor",
        def(
            ConvolutionDescriptor,
            Int32,
            type_of(pad),
            type_of(stride),
            type_of(dilation),
            ConvolutionMode,
        ) thin -> Status,
    ]()(desc, spatial_dim, pad, stride, dilation, mode)


def miopenSetConvolutionGroupCount(
    desc: ConvolutionDescriptor,
    group_count: Int32,
) raises -> Status:
    return _get_dylib_function[
        "miopenSetConvolutionGroupCount",
        def(ConvolutionDescriptor, Int32) thin -> Status,
    ]()(desc, group_count)


# ===-----------------------------------------------------------------------===#
# Convolution Forward Operations
# ===-----------------------------------------------------------------------===#


def miopenConvolutionForwardGetWorkSpaceSize(
    handle: Handle,
    w_desc: TensorDescriptor,
    x_desc: TensorDescriptor,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    workspace_size: MutPointer[UInt64, MutAnyOrigin],
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForwardGetWorkSpaceSize",
        def(
            Handle,
            TensorDescriptor,
            TensorDescriptor,
            ConvolutionDescriptor,
            TensorDescriptor,
            type_of(workspace_size),
        ) thin -> Status,
    ]()(handle, w_desc, x_desc, conv_desc, y_desc, workspace_size)


def miopenFindConvolutionForwardAlgorithm(
    handle: Handle,
    x_desc: TensorDescriptor,
    x: OpaquePointer,
    w_desc: TensorDescriptor,
    w: OpaquePointer,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    y: OpaquePointer,
    requested_algo_count: Int32,
    returned_algo_count: MutPointer[Int32, MutAnyOrigin],
    perf_results: MutPointer[ConvAlgoPerf, MutAnyOrigin],
    workspace: OpaquePointer,
    workspace_size: UInt64,
    exhaustive_search: Bool,
) raises -> Status:
    return _get_dylib_function[
        "miopenFindConvolutionForwardAlgorithm",
        def(
            Handle,
            TensorDescriptor,
            type_of(x),
            TensorDescriptor,
            type_of(w),
            ConvolutionDescriptor,
            TensorDescriptor,
            type_of(y),
            Int32,
            type_of(returned_algo_count),
            type_of(perf_results),
            type_of(workspace),
            UInt64,
            Bool,
        ) thin -> Status,
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
        workspace,
        workspace_size,
        exhaustive_search,
    )


def miopenConvolutionForward(
    handle: Handle,
    alpha: ImmPointer[Float32, ImmutAnyOrigin],
    x_desc: TensorDescriptor,
    x: OpaquePointer,
    w_desc: TensorDescriptor,
    w: OpaquePointer,
    conv_desc: ConvolutionDescriptor,
    algo: ConvFwdAlgorithm,
    beta: ImmPointer[Float32, ImmutAnyOrigin],
    y_desc: TensorDescriptor,
    y: OpaquePointer,
    workspace: OpaquePointer,
    workspace_size: UInt64,
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForward",
        def(
            Handle,
            type_of(alpha),
            TensorDescriptor,
            type_of(x),
            TensorDescriptor,
            type_of(w),
            ConvolutionDescriptor,
            ConvFwdAlgorithm,
            type_of(beta),
            TensorDescriptor,
            type_of(y),
            type_of(workspace),
            UInt64,
        ) thin -> Status,
    ]()(
        handle,
        alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        beta,
        y_desc,
        y,
        workspace,
        workspace_size,
    )
