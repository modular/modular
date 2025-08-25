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

from os import abort
from pathlib import Path
from sys.ffi import _find_dylib
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle


from utils import StaticTuple

from .infer import (
    cudnnMathType_t,
    cudnnDeterminism_t,
    cudnnTensorTransformStruct,
    cudnnConvolutionFwdAlgo_t,
    cudnnTensorFormat_t,
    cudnnStatus_t,
    cudnnConvolutionBwdDataAlgo_t,
    cudnnDataType_t,
    cudnnContext,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS = List[Path](
    "libcudnn.so",
    "libcudnn.so.9",
    "libcudnn.so.8",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.9",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
)

alias CUDA_CUDNN_CNN_INFER_LIBRARY = _Global[
    "CUDA_CUDNN_CNN_INFER_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    return _find_dylib["CUDA cuDNN CNN Infer"](
        CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS
    )


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_CNN_INFER_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

alias cudnnTensorStruct = OpaquePointer
alias cudnnConvolutionStruct = OpaquePointer
alias cudnnFilterStruct = OpaquePointer
alias cudnnActivationStruct = OpaquePointer
alias cudnnFusedOpsPlanStruct = OpaquePointer
alias cudnnFusedOpsVariantParamStruct = NoneType
alias cudnnFusedOpsConstParamStruct = NoneType


fn cudnnGetConvolutionMathType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    math_type: UnsafePointer[cudnnMathType_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionMathType",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnMathType_t],
        ) -> cudnnStatus_t,
    ]()(conv_desc, math_type)


fn cudnnIm2Col(
    handle: UnsafePointer[cudnnContext],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    col_buffer: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnIm2Col",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnConvolutionStruct],
            OpaquePointer,
        ) -> cudnnStatus_t,
    ]()(handle, x_desc, x, w_desc, conv_desc, col_buffer)


fn cudnnConvolutionBiasActivationForward(
    handle: UnsafePointer[cudnnContext],
    alpha1: OpaquePointer,
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct],
    w: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    algo: cudnnConvolutionFwdAlgo_t,
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
    alpha2: OpaquePointer,
    z_desc: UnsafePointer[cudnnTensorStruct],
    z: OpaquePointer,
    bias_desc: UnsafePointer[cudnnTensorStruct],
    bias: OpaquePointer,
    activation_desc: UnsafePointer[cudnnActivationStruct],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionBiasActivationForward",
        fn (
            UnsafePointer[cudnnContext],
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnFilterStruct],
            OpaquePointer,
            UnsafePointer[cudnnConvolutionStruct],
            cudnnConvolutionFwdAlgo_t,
            OpaquePointer,
            Int,
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnActivationStruct],
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
        ) -> cudnnStatus_t,
    ]()(
        handle,
        alpha1,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        alpha2,
        z_desc,
        z,
        bias_desc,
        bias,
        activation_desc,
        y_desc,
        y,
    )


@register_passable("trivial")
struct cudnnConvolutionFwdAlgoPerfStruct:
    var algo: cudnnConvolutionFwdAlgo_t
    var status: cudnnStatus_t
    var time: Float32
    var memory: Int
    var determinism: cudnnDeterminism_t
    var mathType: cudnnMathType_t
    var reserved: StaticTuple[Int32, 3]


fn cudnnSetConvolution2dDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    pad_h: Int16,
    pad_w: Int16,
    u: Int16,
    v: Int16,
    dilation_h: Int16,
    dilation_w: Int16,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolution2dDescriptor",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
            Int16,
            cudnnConvolutionMode_t,
            cudnnDataType_t,
        ) -> cudnnStatus_t,
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


fn cudnnCreateConvolutionDescriptor(
    conv_desc: UnsafePointer[UnsafePointer[cudnnConvolutionStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateConvolutionDescriptor",
        fn (
            UnsafePointer[UnsafePointer[cudnnConvolutionStruct]],
        ) -> cudnnStatus_t,
    ]()(conv_desc)


fn cudnnSetConvolutionGroupCount(
    conv_desc: UnsafePointer[cudnnConvolutionStruct], group_count: Int16
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionGroupCount",
        fn (UnsafePointer[cudnnConvolutionStruct], Int16) -> cudnnStatus_t,
    ]()(conv_desc, group_count)


alias cudnnFusedOpsVariantParamPack_t = UnsafePointer[
    cudnnFusedOpsVariantParamStruct
]

alias cudnnConvolutionFwdAlgoPerf_t = cudnnConvolutionFwdAlgoPerfStruct


@register_passable("trivial")
struct cudnnConvolutionBwdDataAlgoPerfStruct:
    var algo: cudnnConvolutionBwdDataAlgo_t
    var status: cudnnStatus_t
    var time: Float32
    var memory: Int
    var determinism: cudnnDeterminism_t
    var mathType: cudnnMathType_t
    var reserved: StaticTuple[Int32, 3]


fn cudnnGetConvolutionForwardWorkspaceSize(
    handle: UnsafePointer[cudnnContext],
    x_desc: UnsafePointer[cudnnTensorStruct],
    w_desc: UnsafePointer[cudnnFilterStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    y_desc: UnsafePointer[cudnnTensorStruct],
    algo: cudnnConvolutionFwdAlgo_t,
    size_in_bytes: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardWorkspaceSize",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            cudnnConvolutionFwdAlgo_t,
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(handle, x_desc, w_desc, conv_desc, y_desc, algo, size_in_bytes)


fn cudnnGetConvolution2dDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    pad_h: UnsafePointer[Int16],
    pad_w: UnsafePointer[Int16],
    u: UnsafePointer[Int16],
    v: UnsafePointer[Int16],
    dilation_h: UnsafePointer[Int16],
    dilation_w: UnsafePointer[Int16],
    mode: UnsafePointer[cudnnConvolutionMode_t],
    compute_type: UnsafePointer[cudnnDataType_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolution2dDescriptor",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[cudnnConvolutionMode_t],
            UnsafePointer[cudnnDataType_t],
        ) -> cudnnStatus_t,
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


@fieldwise_init
@register_passable("trivial")
struct cudnnFusedOpsConstParamLabel_t(Writable):
    var _value: Int8
    alias CUDNN_PARAM_XDESC = Self(0)
    alias CUDNN_PARAM_XDATA_PLACEHOLDER = Self(1)
    alias CUDNN_PARAM_BN_MODE = Self(2)
    alias CUDNN_PARAM_BN_EQSCALEBIAS_DESC = Self(3)
    alias CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = Self(4)
    alias CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = Self(5)
    alias CUDNN_PARAM_ACTIVATION_DESC = Self(6)
    alias CUDNN_PARAM_CONV_DESC = Self(7)
    alias CUDNN_PARAM_WDESC = Self(8)
    alias CUDNN_PARAM_WDATA_PLACEHOLDER = Self(9)
    alias CUDNN_PARAM_DWDESC = Self(10)
    alias CUDNN_PARAM_DWDATA_PLACEHOLDER = Self(11)
    alias CUDNN_PARAM_YDESC = Self(12)
    alias CUDNN_PARAM_YDATA_PLACEHOLDER = Self(13)
    alias CUDNN_PARAM_DYDESC = Self(14)
    alias CUDNN_PARAM_DYDATA_PLACEHOLDER = Self(15)
    alias CUDNN_PARAM_YSTATS_DESC = Self(16)
    alias CUDNN_PARAM_YSUM_PLACEHOLDER = Self(17)
    alias CUDNN_PARAM_YSQSUM_PLACEHOLDER = Self(18)
    alias CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = Self(19)
    alias CUDNN_PARAM_BN_SCALE_PLACEHOLDER = Self(20)
    alias CUDNN_PARAM_BN_BIAS_PLACEHOLDER = Self(21)
    alias CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = Self(22)
    alias CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = Self(23)
    alias CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = Self(24)
    alias CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = Self(25)
    alias CUDNN_PARAM_ZDESC = Self(26)
    alias CUDNN_PARAM_ZDATA_PLACEHOLDER = Self(27)
    alias CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = Self(28)
    alias CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = Self(29)
    alias CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = Self(30)
    alias CUDNN_PARAM_ACTIVATION_BITMASK_DESC = Self(31)
    alias CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = Self(32)
    alias CUDNN_PARAM_DXDESC = Self(33)
    alias CUDNN_PARAM_DXDATA_PLACEHOLDER = Self(34)
    alias CUDNN_PARAM_DZDESC = Self(35)
    alias CUDNN_PARAM_DZDATA_PLACEHOLDER = Self(36)
    alias CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = Self(37)
    alias CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = Self(38)

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
    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_PARAM_XDESC:
            return writer.write("CUDNN_PARAM_XDESC")
        if self is Self.CUDNN_PARAM_XDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_XDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_MODE:
            return writer.write("CUDNN_PARAM_BN_MODE")
        if self is Self.CUDNN_PARAM_BN_EQSCALEBIAS_DESC:
            return writer.write("CUDNN_PARAM_BN_EQSCALEBIAS_DESC")
        if self is Self.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_ACTIVATION_DESC:
            return writer.write("CUDNN_PARAM_ACTIVATION_DESC")
        if self is Self.CUDNN_PARAM_CONV_DESC:
            return writer.write("CUDNN_PARAM_CONV_DESC")
        if self is Self.CUDNN_PARAM_WDESC:
            return writer.write("CUDNN_PARAM_WDESC")
        if self is Self.CUDNN_PARAM_WDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_WDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_DWDESC:
            return writer.write("CUDNN_PARAM_DWDESC")
        if self is Self.CUDNN_PARAM_DWDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_DWDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_YDESC:
            return writer.write("CUDNN_PARAM_YDESC")
        if self is Self.CUDNN_PARAM_YDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_YDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_DYDESC:
            return writer.write("CUDNN_PARAM_DYDESC")
        if self is Self.CUDNN_PARAM_DYDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_DYDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_YSTATS_DESC:
            return writer.write("CUDNN_PARAM_YSTATS_DESC")
        if self is Self.CUDNN_PARAM_YSUM_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_YSUM_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_YSQSUM_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_YSQSUM_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC:
            return writer.write("CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC")
        if self is Self.CUDNN_PARAM_BN_SCALE_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_SCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_BIAS_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_BIAS_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_ZDESC:
            return writer.write("CUDNN_PARAM_ZDESC")
        if self is Self.CUDNN_PARAM_ZDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_ZDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC:
            return writer.write("CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC")
        if self is Self.CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_ACTIVATION_BITMASK_DESC:
            return writer.write("CUDNN_PARAM_ACTIVATION_BITMASK_DESC")
        if self is Self.CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_DXDESC:
            return writer.write("CUDNN_PARAM_DXDESC")
        if self is Self.CUDNN_PARAM_DXDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_DXDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_DZDESC:
            return writer.write("CUDNN_PARAM_DZDESC")
        if self is Self.CUDNN_PARAM_DZDATA_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_DZDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_DSCALE_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_DSCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_DBIAS_PLACEHOLDER:
            return writer.write("CUDNN_PARAM_BN_DBIAS_PLACEHOLDER")
        abort("invalid cudnnFusedOpsConstParamLabel_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnFusedOpsConstParamLabel_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSetConvolutionReorderType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    reorder_type: cudnnReorderType_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionReorderType",
        fn (
            UnsafePointer[cudnnConvolutionStruct], cudnnReorderType_t
        ) -> cudnnStatus_t,
    ]()(conv_desc, reorder_type)


@fieldwise_init
@register_passable("trivial")
struct cudnnReorderType_t(Writable):
    var _value: Int8
    alias CUDNN_DEFAULT_REORDER = Self(0)
    alias CUDNN_NO_REORDER = Self(1)

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
    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_DEFAULT_REORDER:
            return writer.write("CUDNN_DEFAULT_REORDER")
        if self is Self.CUDNN_NO_REORDER:
            return writer.write("CUDNN_NO_REORDER")
        abort("invalid cudnnReorderType_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnReorderType_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias cudnnConvolutionBwdDataAlgoPerf_t = cudnnConvolutionBwdDataAlgoPerfStruct


fn cudnnGetConvolution2dForwardOutputDim(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    input_tensor_desc: UnsafePointer[cudnnTensorStruct],
    filter_desc: UnsafePointer[cudnnFilterStruct],
    n: UnsafePointer[Int16],
    c: UnsafePointer[Int16],
    h: UnsafePointer[Int16],
    w: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolution2dForwardOutputDim",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
        ) -> cudnnStatus_t,
    ]()(conv_desc, input_tensor_desc, filter_desc, n, c, h, w)


fn cudnnFindConvolutionForwardAlgorithm(
    handle: UnsafePointer[cudnnContext],
    x_desc: UnsafePointer[cudnnTensorStruct],
    w_desc: UnsafePointer[cudnnFilterStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    y_desc: UnsafePointer[cudnnTensorStruct],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16],
    perf_results: UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionForwardAlgorithm",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct],
        ) -> cudnnStatus_t,
    ]()(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


alias cudnnFusedOpsConstParamPack_t = UnsafePointer[
    cudnnFusedOpsConstParamStruct
]


fn cudnnGetConvolutionForwardAlgorithm_v7(
    handle: UnsafePointer[cudnnContext],
    src_desc: UnsafePointer[cudnnTensorStruct],
    filter_desc: UnsafePointer[cudnnFilterStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    dest_desc: UnsafePointer[cudnnTensorStruct],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16],
    perf_results: UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardAlgorithm_v7",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct],
        ) -> cudnnStatus_t,
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


@fieldwise_init
@register_passable("trivial")
struct cudnnFusedOps_t(Writable):
    var _value: Int8
    alias CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = Self(0)
    alias CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = Self(1)
    alias CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = Self(2)
    alias CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = Self(3)
    alias CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = Self(4)
    alias CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = Self(5)
    alias CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = Self(6)

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
    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS:
            return writer.write(
                "CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS"
            )
        if self is Self.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD:
            return writer.write("CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD")
        if self is Self.CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING:
            return writer.write("CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING")
        if self is Self.CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE:
            return writer.write("CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE")
        if self is Self.CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION:
            return writer.write("CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION")
        if self is Self.CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK:
            return writer.write(
                "CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK"
            )
        if self is Self.CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM:
            return writer.write("CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM")
        abort("invalid cudnnFusedOps_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnFusedOps_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    handle: UnsafePointer[cudnnContext], count: UnsafePointer[Int16]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionBackwardDataAlgorithmMaxCount",
        fn (UnsafePointer[cudnnContext], UnsafePointer[Int16]) -> cudnnStatus_t,
    ]()(handle, count)


fn cudnnDestroyConvolutionDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyConvolutionDescriptor",
        fn (UnsafePointer[cudnnConvolutionStruct]) -> cudnnStatus_t,
    ]()(conv_desc)


@fieldwise_init
@register_passable("trivial")
struct cudnnFusedOpsPointerPlaceHolder_t(Writable):
    var _value: Int8
    alias CUDNN_PTR_NULL = Self(0)
    alias CUDNN_PTR_ELEM_ALIGNED = Self(1)
    alias CUDNN_PTR_16B_ALIGNED = Self(2)

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
    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_PTR_NULL:
            return writer.write("CUDNN_PTR_NULL")
        if self is Self.CUDNN_PTR_ELEM_ALIGNED:
            return writer.write("CUDNN_PTR_ELEM_ALIGNED")
        if self is Self.CUDNN_PTR_16B_ALIGNED:
            return writer.write("CUDNN_PTR_16B_ALIGNED")
        abort("invalid cudnnFusedOpsPointerPlaceHolder_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnFusedOpsPointerPlaceHolder_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


alias cudnnFusedOpsPlan_t = UnsafePointer[cudnnFusedOpsPlanStruct]

alias cudnnConvolutionDescriptor_t = UnsafePointer[cudnnConvolutionStruct]


fn cudnnConvolutionForward(
    handle: UnsafePointer[cudnnContext],
    alpha: OpaquePointer,
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct],
    w: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    algo: cudnnConvolutionFwdAlgo_t,
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
    beta: OpaquePointer,
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionForward",
        fn (
            UnsafePointer[cudnnContext],
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnFilterStruct],
            OpaquePointer,
            UnsafePointer[cudnnConvolutionStruct],
            cudnnConvolutionFwdAlgo_t,
            OpaquePointer,
            Int,
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
        ) -> cudnnStatus_t,
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


fn cudnnGetConvolutionReorderType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    reorder_type: UnsafePointer[cudnnReorderType_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionReorderType",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnReorderType_t],
        ) -> cudnnStatus_t,
    ]()(conv_desc, reorder_type)


@fieldwise_init
@register_passable("trivial")
struct cudnnFusedOpsVariantParamLabel_t(Writable):
    var _value: Int8
    alias CUDNN_PTR_XDATA = Self(0)
    alias CUDNN_PTR_BN_EQSCALE = Self(1)
    alias CUDNN_PTR_BN_EQBIAS = Self(2)
    alias CUDNN_PTR_WDATA = Self(3)
    alias CUDNN_PTR_DWDATA = Self(4)
    alias CUDNN_PTR_YDATA = Self(5)
    alias CUDNN_PTR_DYDATA = Self(6)
    alias CUDNN_PTR_YSUM = Self(7)
    alias CUDNN_PTR_YSQSUM = Self(8)
    alias CUDNN_PTR_WORKSPACE = Self(9)
    alias CUDNN_PTR_BN_SCALE = Self(10)
    alias CUDNN_PTR_BN_BIAS = Self(11)
    alias CUDNN_PTR_BN_SAVED_MEAN = Self(12)
    alias CUDNN_PTR_BN_SAVED_INVSTD = Self(13)
    alias CUDNN_PTR_BN_RUNNING_MEAN = Self(14)
    alias CUDNN_PTR_BN_RUNNING_VAR = Self(15)
    alias CUDNN_PTR_ZDATA = Self(16)
    alias CUDNN_PTR_BN_Z_EQSCALE = Self(17)
    alias CUDNN_PTR_BN_Z_EQBIAS = Self(18)
    alias CUDNN_PTR_ACTIVATION_BITMASK = Self(19)
    alias CUDNN_PTR_DXDATA = Self(20)
    alias CUDNN_PTR_DZDATA = Self(21)
    alias CUDNN_PTR_BN_DSCALE = Self(22)
    alias CUDNN_PTR_BN_DBIAS = Self(23)
    alias CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = Self(24)
    alias CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = Self(25)
    alias CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = Self(26)
    alias CUDNN_SCALAR_DOUBLE_BN_EPSILON = Self(27)

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
    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_PTR_XDATA:
            return writer.write("CUDNN_PTR_XDATA")
        if self is Self.CUDNN_PTR_BN_EQSCALE:
            return writer.write("CUDNN_PTR_BN_EQSCALE")
        if self is Self.CUDNN_PTR_BN_EQBIAS:
            return writer.write("CUDNN_PTR_BN_EQBIAS")
        if self is Self.CUDNN_PTR_WDATA:
            return writer.write("CUDNN_PTR_WDATA")
        if self is Self.CUDNN_PTR_DWDATA:
            return writer.write("CUDNN_PTR_DWDATA")
        if self is Self.CUDNN_PTR_YDATA:
            return writer.write("CUDNN_PTR_YDATA")
        if self is Self.CUDNN_PTR_DYDATA:
            return writer.write("CUDNN_PTR_DYDATA")
        if self is Self.CUDNN_PTR_YSUM:
            return writer.write("CUDNN_PTR_YSUM")
        if self is Self.CUDNN_PTR_YSQSUM:
            return writer.write("CUDNN_PTR_YSQSUM")
        if self is Self.CUDNN_PTR_WORKSPACE:
            return writer.write("CUDNN_PTR_WORKSPACE")
        if self is Self.CUDNN_PTR_BN_SCALE:
            return writer.write("CUDNN_PTR_BN_SCALE")
        if self is Self.CUDNN_PTR_BN_BIAS:
            return writer.write("CUDNN_PTR_BN_BIAS")
        if self is Self.CUDNN_PTR_BN_SAVED_MEAN:
            return writer.write("CUDNN_PTR_BN_SAVED_MEAN")
        if self is Self.CUDNN_PTR_BN_SAVED_INVSTD:
            return writer.write("CUDNN_PTR_BN_SAVED_INVSTD")
        if self is Self.CUDNN_PTR_BN_RUNNING_MEAN:
            return writer.write("CUDNN_PTR_BN_RUNNING_MEAN")
        if self is Self.CUDNN_PTR_BN_RUNNING_VAR:
            return writer.write("CUDNN_PTR_BN_RUNNING_VAR")
        if self is Self.CUDNN_PTR_ZDATA:
            return writer.write("CUDNN_PTR_ZDATA")
        if self is Self.CUDNN_PTR_BN_Z_EQSCALE:
            return writer.write("CUDNN_PTR_BN_Z_EQSCALE")
        if self is Self.CUDNN_PTR_BN_Z_EQBIAS:
            return writer.write("CUDNN_PTR_BN_Z_EQBIAS")
        if self is Self.CUDNN_PTR_ACTIVATION_BITMASK:
            return writer.write("CUDNN_PTR_ACTIVATION_BITMASK")
        if self is Self.CUDNN_PTR_DXDATA:
            return writer.write("CUDNN_PTR_DXDATA")
        if self is Self.CUDNN_PTR_DZDATA:
            return writer.write("CUDNN_PTR_DZDATA")
        if self is Self.CUDNN_PTR_BN_DSCALE:
            return writer.write("CUDNN_PTR_BN_DSCALE")
        if self is Self.CUDNN_PTR_BN_DBIAS:
            return writer.write("CUDNN_PTR_BN_DBIAS")
        if self is Self.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES:
            return writer.write("CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES")
        if self is Self.CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT:
            return writer.write("CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT")
        if self is Self.CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR:
            return writer.write("CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR")
        if self is Self.CUDNN_SCALAR_DOUBLE_BN_EPSILON:
            return writer.write("CUDNN_SCALAR_DOUBLE_BN_EPSILON")
        abort("invalid cudnnFusedOpsVariantParamLabel_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnFusedOpsVariantParamLabel_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: UnsafePointer[cudnnContext],
    filter_desc: UnsafePointer[cudnnFilterStruct],
    diff_desc: UnsafePointer[cudnnTensorStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    grad_desc: UnsafePointer[cudnnTensorStruct],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16],
    perf_results: UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionBackwardDataAlgorithm_v7",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct],
        ) -> cudnnStatus_t,
    ]()(
        handle,
        filter_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


fn cudnnGetConvolutionGroupCount(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    group_count: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionGroupCount",
        fn (
            UnsafePointer[cudnnConvolutionStruct], UnsafePointer[Int16]
        ) -> cudnnStatus_t,
    ]()(conv_desc, group_count)


fn cudnnGetConvolutionNdDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    array_length_requested: Int16,
    array_length: UnsafePointer[Int16],
    pad_a: OpaquePointer,
    stride_a: OpaquePointer,
    dilation_a: OpaquePointer,
    mode: UnsafePointer[cudnnConvolutionMode_t],
    compute_type: UnsafePointer[cudnnDataType_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionNdDescriptor",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            Int16,
            UnsafePointer[Int16],
            OpaquePointer,
            OpaquePointer,
            OpaquePointer,
            UnsafePointer[cudnnConvolutionMode_t],
            UnsafePointer[cudnnDataType_t],
        ) -> cudnnStatus_t,
    ]()(
        conv_desc,
        array_length_requested,
        array_length,
        pad_a,
        stride_a,
        dilation_a,
        mode,
        compute_type,
    )


fn cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle: UnsafePointer[cudnnContext],
    w_desc: UnsafePointer[cudnnFilterStruct],
    dy_desc: UnsafePointer[cudnnTensorStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    dx_desc: UnsafePointer[cudnnTensorStruct],
    algo: cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionBackwardDataWorkspaceSize",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            cudnnConvolutionBwdDataAlgo_t,
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(handle, w_desc, dy_desc, conv_desc, dx_desc, algo, size_in_bytes)


fn cudnnReorderFilterAndBias(
    handle: UnsafePointer[cudnnContext],
    filter_desc: UnsafePointer[cudnnFilterStruct],
    reorder_type: cudnnReorderType_t,
    filter_data: OpaquePointer,
    reordered_filter_data: OpaquePointer,
    reorder_bias: Int16,
    bias_data: OpaquePointer,
    reordered_bias_data: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnReorderFilterAndBias",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnFilterStruct],
            cudnnReorderType_t,
            OpaquePointer,
            OpaquePointer,
            Int16,
            OpaquePointer,
            OpaquePointer,
        ) -> cudnnStatus_t,
    ]()(
        handle,
        filter_desc,
        reorder_type,
        filter_data,
        reordered_filter_data,
        reorder_bias,
        bias_data,
        reordered_bias_data,
    )


fn cudnnFindConvolutionBackwardDataAlgorithm(
    handle: UnsafePointer[cudnnContext],
    w_desc: UnsafePointer[cudnnFilterStruct],
    dy_desc: UnsafePointer[cudnnTensorStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    dx_desc: UnsafePointer[cudnnTensorStruct],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16],
    perf_results: UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionBackwardDataAlgorithm",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct],
        ) -> cudnnStatus_t,
    ]()(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


fn cudnnConvolutionBackwardData(
    handle: UnsafePointer[cudnnContext],
    alpha: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct],
    w: OpaquePointer,
    dy_desc: UnsafePointer[cudnnTensorStruct],
    dy: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    algo: cudnnConvolutionBwdDataAlgo_t,
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
    beta: OpaquePointer,
    dx_desc: UnsafePointer[cudnnTensorStruct],
    dx: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionBackwardData",
        fn (
            UnsafePointer[cudnnContext],
            OpaquePointer,
            UnsafePointer[cudnnFilterStruct],
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnConvolutionStruct],
            cudnnConvolutionBwdDataAlgo_t,
            OpaquePointer,
            Int,
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
        ) -> cudnnStatus_t,
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


fn cudnnSetConvolutionNdDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    array_length: Int16,
    pad_a: OpaquePointer,
    filter_stride_a: OpaquePointer,
    dilation_a: OpaquePointer,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionNdDescriptor",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            Int16,
            OpaquePointer,
            OpaquePointer,
            OpaquePointer,
            cudnnConvolutionMode_t,
            cudnnDataType_t,
        ) -> cudnnStatus_t,
    ]()(
        conv_desc,
        array_length,
        pad_a,
        filter_stride_a,
        dilation_a,
        mode,
        compute_type,
    )


fn cudnnCnnInferVersionCheck() -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCnnInferVersionCheck", fn () -> cudnnStatus_t
    ]()()


fn cudnnSetConvolutionMathType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct], math_type: cudnnMathType_t
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionMathType",
        fn (
            UnsafePointer[cudnnConvolutionStruct], cudnnMathType_t
        ) -> cudnnStatus_t,
    ]()(conv_desc, math_type)


fn cudnnFindConvolutionBackwardDataAlgorithmEx(
    handle: UnsafePointer[cudnnContext],
    w_desc: UnsafePointer[cudnnFilterStruct],
    w: OpaquePointer,
    dy_desc: UnsafePointer[cudnnTensorStruct],
    dy: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    dx_desc: UnsafePointer[cudnnTensorStruct],
    dx: OpaquePointer,
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16],
    perf_results: UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct],
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionBackwardDataAlgorithmEx",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnFilterStruct],
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct],
            OpaquePointer,
            Int,
        ) -> cudnnStatus_t,
    ]()(
        handle,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        dx_desc,
        dx,
        requested_algo_count,
        returned_algo_count,
        perf_results,
        work_space,
        work_space_size_in_bytes,
    )


fn cudnnFindConvolutionForwardAlgorithmEx(
    handle: UnsafePointer[cudnnContext],
    x_desc: UnsafePointer[cudnnTensorStruct],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct],
    w: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    y_desc: UnsafePointer[cudnnTensorStruct],
    y: OpaquePointer,
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16],
    perf_results: UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct],
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionForwardAlgorithmEx",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            UnsafePointer[cudnnFilterStruct],
            OpaquePointer,
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            OpaquePointer,
            Int16,
            UnsafePointer[Int16],
            UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct],
            OpaquePointer,
            Int,
        ) -> cudnnStatus_t,
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


fn cudnnGetConvolutionNdForwardOutputDim(
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    input_tensor_desc: UnsafePointer[cudnnTensorStruct],
    filter_desc: UnsafePointer[cudnnFilterStruct],
    nb_dims: Int16,
    tensor_output_dim_a: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionNdForwardOutputDim",
        fn (
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnFilterStruct],
            Int16,
            OpaquePointer,
        ) -> cudnnStatus_t,
    ]()(conv_desc, input_tensor_desc, filter_desc, nb_dims, tensor_output_dim_a)


fn cudnnGetConvolutionForwardAlgorithmMaxCount(
    handle: UnsafePointer[cudnnContext], count: UnsafePointer[Int16]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardAlgorithmMaxCount",
        fn (UnsafePointer[cudnnContext], UnsafePointer[Int16]) -> cudnnStatus_t,
    ]()(handle, count)


@fieldwise_init
@register_passable("trivial")
struct cudnnConvolutionMode_t(Writable):
    var _value: Int8
    alias CUDNN_CONVOLUTION = Self(0)
    alias CUDNN_CROSS_CORRELATION = Self(1)

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
    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_CONVOLUTION:
            return writer.write("CUDNN_CONVOLUTION")
        if self is Self.CUDNN_CROSS_CORRELATION:
            return writer.write("CUDNN_CROSS_CORRELATION")
        abort("invalid cudnnConvolutionMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnConvolutionMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnGetFoldedConvBackwardDataDescriptors(
    handle: UnsafePointer[cudnnContext],
    filter_desc: UnsafePointer[cudnnFilterStruct],
    diff_desc: UnsafePointer[cudnnTensorStruct],
    conv_desc: UnsafePointer[cudnnConvolutionStruct],
    grad_desc: UnsafePointer[cudnnTensorStruct],
    transform_format: cudnnTensorFormat_t,
    folded_filter_desc: UnsafePointer[cudnnFilterStruct],
    padded_diff_desc: UnsafePointer[cudnnTensorStruct],
    folded_conv_desc: UnsafePointer[cudnnConvolutionStruct],
    folded_grad_desc: UnsafePointer[cudnnTensorStruct],
    filter_fold_trans_desc: UnsafePointer[cudnnTensorTransformStruct],
    diff_pad_trans_desc: UnsafePointer[cudnnTensorTransformStruct],
    grad_fold_trans_desc: UnsafePointer[cudnnTensorTransformStruct],
    grad_unfold_trans_desc: UnsafePointer[cudnnTensorTransformStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFoldedConvBackwardDataDescriptors",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            cudnnTensorFormat_t,
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnConvolutionStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnTensorTransformStruct],
            UnsafePointer[cudnnTensorTransformStruct],
            UnsafePointer[cudnnTensorTransformStruct],
            UnsafePointer[cudnnTensorTransformStruct],
        ) -> cudnnStatus_t,
    ]()(
        handle,
        filter_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        transform_format,
        folded_filter_desc,
        padded_diff_desc,
        folded_conv_desc,
        folded_grad_desc,
        filter_fold_trans_desc,
        diff_pad_trans_desc,
        grad_fold_trans_desc,
        grad_unfold_trans_desc,
    )
