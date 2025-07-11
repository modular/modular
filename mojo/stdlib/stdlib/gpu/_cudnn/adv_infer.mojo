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


from .infer import (
    cudnnFilterStruct,
    cudnnMathType_t,
    cudnnNanPropagation_t,
    cudnnRNNAlgo_t,
    cudnnStatus_t,
    cudnnContext,
    cudnnDataType_t,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_CUDNN_ADV_INFER_LIBRARY_PATHS = List[Path](
    "libcudnn_adv_infer.so",
    "libcudnn_adv_infer.so.9",
    "libcudnn_adv_infer.so.8",
    "/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.9",
    "/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8",
)

alias CUDA_CUDNN_ADV_INFER_LIBRARY = _Global[
    "CUDA_CUDNN_ADV_INFER_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    return _find_dylib["CUDA cuDNN Adv Infer"](
        CUDA_CUDNN_ADV_INFER_LIBRARY_PATHS
    )


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_ADV_INFER_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

alias cudnnRNNStruct = OpaquePointer
alias cudnnDropoutStruct = OpaquePointer
alias cudnnAlgorithmStruct = OpaquePointer
alias cudnnRNNDataStruct = OpaquePointer
alias cudnnAttnStruct = OpaquePointer
alias cudnnTensorStruct = OpaquePointer
alias cudnnSeqDataStruct = OpaquePointer
alias cudnnPersistentRNNPlan = NoneType


@fieldwise_init
@register_passable("trivial")
struct cudnnRNNInputMode_t:
    var _value: Int32

    alias LINEAR_INPUT = Self(0)
    """Adjustable weight matrix in first layer input GEMM."""
    alias SKIP_INPUT = Self(1)
    """Fixed identity matrix in the first layer input GEMM."""


@fieldwise_init
@register_passable("trivial")
struct cudnnDirectionMode_t:
    var _value: Int32

    alias UNIDIRECTIONAL = Self(0)
    """Single direction network."""
    alias BIDIRECTIONAL = Self(1)
    """Output concatenation at each layer."""


@fieldwise_init
@register_passable("trivial")
struct cudnnRNNClipMode_t:
    var _value: Int32

    alias NONE = Self(0)
    """Disables LSTM cell clipping."""
    alias MINMAX = Self(1)
    """Enables LSTM cell clipping."""


@fieldwise_init
@register_passable("trivial")
struct cudnnRNNMode_t:
    var _value: Int32
    alias RNN_RELU = Self(0)
    """Basic RNN cell type with ReLu activation."""
    alias RNN_TANH = Self(1)
    """Basic RNN cell type with tanh activation."""
    alias LTSM = Self(2)
    """LSTM with optional recurrent projection and clipping."""
    alias GRU = Self(3)
    """Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1)."""


@fieldwise_init
@register_passable("trivial")
struct cudnnMultiHeadAttnWeightKind_t:
    var _value: Int32

    alias ATTN_Q_WEIGHTS = Self(0)
    "Input projection weights for 'queries'."

    alias ATTN_K_WEIGHTS = Self(1)
    "Input projection weights for 'keys'."

    alias ATTN_V_WEIGHTS = Self(2)
    "Input projection weights for 'values'."

    alias ATTN_O_WEIGHTS = Self(3)
    "Output projection weights."

    alias ATTN_Q_BIASES = Self(4)
    "Input projection bias for 'queries'."

    alias ATTN_K_BIASES = Self(5)
    "Input projection bias for 'keys'."

    alias ATTN_V_BIASES = Self(6)
    "Input projection bias for 'values'."

    alias ATTN_O_BIASES = Self(6)
    "Output projection bias."


@fieldwise_init
@register_passable("trivial")
struct cudnnRNNBiasMode_t:
    var _value: Int32

    alias NO_BIAS = Self(0)
    """Rnn cell formulas do not use biases."""
    alias SINGLE_INP_BIAS = Self(1)
    """Rnn cell formulas use one input bias in input GEMM."""
    alias DOUBLE_BIAS = Self(2)
    """Default, rnn cell formulas use two bias vectors."""
    alias SINGLE_REC_BIAS = Self(3)
    """Rrnn cell formulas use one recurrent bias in recurrent GEMMs."""


@fieldwise_init
@register_passable("trivial")
struct cudnnRNNDataLayout_t:
    var _value: Int32
    alias SEQ_MAJOR_UNPACKED = Self(0)
    """Padded, outer stride from one time-step to the next."""
    alias SEQ_MAJOR_PACKED = Self(1)
    """Sequence length sorted and packed as in basic RNN api."""
    alias BATCH_MAJOR_UNPACKED = Self(2)
    """Padded, outer stride from one batch to the next."""


fn cudnnGetRNNDescriptor_v6(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    hidden_size: UnsafePointer[Int16],
    num_layers: UnsafePointer[Int16],
    dropout_desc: UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
    input_mode: UnsafePointer[cudnnRNNInputMode_t],
    direction: UnsafePointer[cudnnDirectionMode_t],
    cell_mode: UnsafePointer[cudnnRNNMode_t],
    algo: UnsafePointer[cudnnRNNAlgo_t],
    math_prec: UnsafePointer[cudnnDataType_t],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNDescriptor_v6",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
            UnsafePointer[cudnnRNNInputMode_t],
            UnsafePointer[cudnnDirectionMode_t],
            UnsafePointer[cudnnRNNMode_t],
            UnsafePointer[cudnnRNNAlgo_t],
            UnsafePointer[cudnnDataType_t],
        ) -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        hidden_size,
        num_layers,
        dropout_desc,
        input_mode,
        direction,
        cell_mode,
        algo,
        math_prec,
    )


@fieldwise_init
@register_passable("trivial")
struct cudnnForwardMode_t(Writable):
    var _value: Int8
    alias CUDNN_FWD_MODE_INFERENCE = Self(0)
    alias CUDNN_FWD_MODE_TRAINING = Self(1)

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
        if self is Self.CUDNN_FWD_MODE_INFERENCE:
            return writer.write("CUDNN_FWD_MODE_INFERENCE")
        if self is Self.CUDNN_FWD_MODE_TRAINING:
            return writer.write("CUDNN_FWD_MODE_TRAINING")
        abort("invalid cudnnForwardMode_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnForwardMode_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnDestroyAttnDescriptor(
    attn_desc: UnsafePointer[cudnnAttnStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyAttnDescriptor",
        fn (UnsafePointer[cudnnAttnStruct]) -> cudnnStatus_t,
    ]()(attn_desc)


fn cudnnGetRNNTempSpaceSizes(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    fwd_mode: cudnnForwardMode_t,
    x_desc: UnsafePointer[cudnnRNNDataStruct],
    work_space_size: UnsafePointer[Int],
    reserve_space_size: UnsafePointer[Int],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNTempSpaceSizes",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            cudnnForwardMode_t,
            UnsafePointer[cudnnRNNDataStruct],
            UnsafePointer[Int],
            UnsafePointer[Int],
        ) -> cudnnStatus_t,
    ]()(handle, rnn_desc, fwd_mode, x_desc, work_space_size, reserve_space_size)


fn cudnnSetRNNDescriptor_v6(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    hidden_size: Int16,
    num_layers: Int16,
    dropout_desc: UnsafePointer[cudnnDropoutStruct],
    input_mode: cudnnRNNInputMode_t,
    direction: cudnnDirectionMode_t,
    cell_mode: cudnnRNNMode_t,
    algo: cudnnRNNAlgo_t,
    math_prec: cudnnDataType_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNDescriptor_v6",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            Int16,
            Int16,
            UnsafePointer[cudnnDropoutStruct],
            cudnnRNNInputMode_t,
            cudnnDirectionMode_t,
            cudnnRNNMode_t,
            cudnnRNNAlgo_t,
            cudnnDataType_t,
        ) -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        hidden_size,
        num_layers,
        dropout_desc,
        input_mode,
        direction,
        cell_mode,
        algo,
        math_prec,
    )


fn cudnnCreatePersistentRNNPlan(
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    minibatch: Int16,
    data_type: cudnnDataType_t,
    plan: UnsafePointer[UnsafePointer[cudnnPersistentRNNPlan]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreatePersistentRNNPlan",
        fn (
            UnsafePointer[cudnnRNNStruct],
            Int16,
            cudnnDataType_t,
            UnsafePointer[UnsafePointer[cudnnPersistentRNNPlan]],
        ) -> cudnnStatus_t,
    ]()(rnn_desc, minibatch, data_type, plan)


fn cudnnGetSeqDataDescriptor(
    seq_data_desc: UnsafePointer[cudnnSeqDataStruct],
    data_type: UnsafePointer[cudnnDataType_t],
    nb_dims: UnsafePointer[Int16],
    nb_dims_requested: Int16,
    dim_a: OpaquePointer,
    axes: OpaquePointer,
    seq_length_array_size: UnsafePointer[Int],
    seq_length_size_requested: Int,
    seq_length_array: OpaquePointer,
    padding_fill: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetSeqDataDescriptor",
        fn (
            UnsafePointer[cudnnSeqDataStruct],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[Int16],
            Int16,
            OpaquePointer,
            OpaquePointer,
            UnsafePointer[Int],
            Int,
            OpaquePointer,
            OpaquePointer,
        ) -> cudnnStatus_t,
    ]()(
        seq_data_desc,
        data_type,
        nb_dims,
        nb_dims_requested,
        dim_a,
        axes,
        seq_length_array_size,
        seq_length_size_requested,
        seq_length_array,
        padding_fill,
    )


fn cudnnRNNGetClip_v8(
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    clip_mode: UnsafePointer[cudnnRNNClipMode_t],
    clip_nan_opt: UnsafePointer[cudnnNanPropagation_t],
    lclip: UnsafePointer[Float64],
    rclip: UnsafePointer[Float64],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRNNGetClip_v8",
        fn (
            UnsafePointer[cudnnRNNStruct],
            UnsafePointer[cudnnRNNClipMode_t],
            UnsafePointer[cudnnNanPropagation_t],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
        ) -> cudnnStatus_t,
    ]()(rnn_desc, clip_mode, clip_nan_opt, lclip, rclip)


fn cudnnSetRNNAlgorithmDescriptor(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    algo_desc: UnsafePointer[cudnnAlgorithmStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNAlgorithmDescriptor",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            UnsafePointer[cudnnAlgorithmStruct],
        ) -> cudnnStatus_t,
    ]()(handle, rnn_desc, algo_desc)


fn cudnnGetRNNParamsSize(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    x_desc: UnsafePointer[cudnnTensorStruct],
    size_in_bytes: UnsafePointer[Int],
    data_type: cudnnDataType_t,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNParamsSize",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[Int],
            cudnnDataType_t,
        ) -> cudnnStatus_t,
    ]()(handle, rnn_desc, x_desc, size_in_bytes, data_type)


fn cudnnSetRNNMatrixMathType(
    rnn_desc: UnsafePointer[cudnnRNNStruct], m_type: cudnnMathType_t
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNMatrixMathType",
        fn (UnsafePointer[cudnnRNNStruct], cudnnMathType_t) -> cudnnStatus_t,
    ]()(rnn_desc, m_type)


fn cudnnGetAttnDescriptor(
    attn_desc: UnsafePointer[cudnnAttnStruct],
    attn_mode: UnsafePointer[Int16],
    n_heads: UnsafePointer[Int16],
    sm_scaler: UnsafePointer[Float64],
    data_type: UnsafePointer[cudnnDataType_t],
    compute_prec: UnsafePointer[cudnnDataType_t],
    math_type: UnsafePointer[cudnnMathType_t],
    attn_dropout_desc: UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
    post_dropout_desc: UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
    q_size: UnsafePointer[Int16],
    k_size: UnsafePointer[Int16],
    v_size: UnsafePointer[Int16],
    q_proj_size: UnsafePointer[Int16],
    k_proj_size: UnsafePointer[Int16],
    v_proj_size: UnsafePointer[Int16],
    o_proj_size: UnsafePointer[Int16],
    qo_max_seq_length: UnsafePointer[Int16],
    kv_max_seq_length: UnsafePointer[Int16],
    max_batch_size: UnsafePointer[Int16],
    max_beam_size: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAttnDescriptor",
        fn (
            UnsafePointer[cudnnAttnStruct],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Float64],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnMathType_t],
            UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
            UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
            UnsafePointer[Int16],
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
        attn_desc,
        attn_mode,
        n_heads,
        sm_scaler,
        data_type,
        compute_prec,
        math_type,
        attn_dropout_desc,
        post_dropout_desc,
        q_size,
        k_size,
        v_size,
        q_proj_size,
        k_proj_size,
        v_proj_size,
        o_proj_size,
        qo_max_seq_length,
        kv_max_seq_length,
        max_batch_size,
        max_beam_size,
    )


alias cudnnRNNDescriptor_t = UnsafePointer[cudnnRNNStruct]

alias cudnnRNNDataDescriptor_t = UnsafePointer[cudnnRNNDataStruct]

alias cudnnPersistentRNNPlan_t = UnsafePointer[cudnnPersistentRNNPlan]


fn cudnnRNNSetClip(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    clip_mode: cudnnRNNClipMode_t,
    clip_nan_opt: cudnnNanPropagation_t,
    lclip: Float64,
    rclip: Float64,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRNNSetClip",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            cudnnRNNClipMode_t,
            cudnnNanPropagation_t,
            Float64,
            Float64,
        ) -> cudnnStatus_t,
    ]()(handle, rnn_desc, clip_mode, clip_nan_opt, lclip, rclip)


fn cudnnGetMultiHeadAttnWeights(
    handle: UnsafePointer[cudnnContext],
    attn_desc: UnsafePointer[cudnnAttnStruct],
    w_kind: cudnnMultiHeadAttnWeightKind_t,
    weight_size_in_bytes: Int,
    weights: OpaquePointer,
    w_desc: UnsafePointer[cudnnTensorStruct],
    w_addr: UnsafePointer[OpaquePointer],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetMultiHeadAttnWeights",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnAttnStruct],
            cudnnMultiHeadAttnWeightKind_t,
            Int,
            OpaquePointer,
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[OpaquePointer],
        ) -> cudnnStatus_t,
    ]()(
        handle, attn_desc, w_kind, weight_size_in_bytes, weights, w_desc, w_addr
    )


fn cudnnSetSeqDataDescriptor(
    seq_data_desc: UnsafePointer[cudnnSeqDataStruct],
    data_type: cudnnDataType_t,
    nb_dims: Int16,
    dim_a: OpaquePointer,
    axes: OpaquePointer,
    seq_length_array_size: Int,
    seq_length_array: OpaquePointer,
    padding_fill: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetSeqDataDescriptor",
        fn (
            UnsafePointer[cudnnSeqDataStruct],
            cudnnDataType_t,
            Int16,
            OpaquePointer,
            OpaquePointer,
            Int,
            OpaquePointer,
            OpaquePointer,
        ) -> cudnnStatus_t,
    ]()(
        seq_data_desc,
        data_type,
        nb_dims,
        dim_a,
        axes,
        seq_length_array_size,
        seq_length_array,
        padding_fill,
    )


fn cudnnCreateSeqDataDescriptor(
    seq_data_desc: UnsafePointer[UnsafePointer[cudnnSeqDataStruct]],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateSeqDataDescriptor",
        fn (UnsafePointer[UnsafePointer[cudnnSeqDataStruct]]) -> cudnnStatus_t,
    ]()(seq_data_desc)


fn cudnnGetRNNPaddingMode(
    rnn_desc: UnsafePointer[cudnnRNNStruct], padding_mode: UnsafePointer[Int16]
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNPaddingMode",
        fn (
            UnsafePointer[cudnnRNNStruct], UnsafePointer[Int16]
        ) -> cudnnStatus_t,
    ]()(rnn_desc, padding_mode)


alias cudnnAttnDescriptor_t = UnsafePointer[cudnnAttnStruct]

alias cudnnAttnQueryMap_t = Int16


fn cudnnGetRNNLinLayerBiasParams(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    pseudo_layer: Int16,
    x_desc: UnsafePointer[cudnnTensorStruct],
    w_desc: UnsafePointer[cudnnFilterStruct],
    w: OpaquePointer,
    lin_layer_id: Int16,
    lin_layer_bias_desc: UnsafePointer[cudnnFilterStruct],
    lin_layer_bias: UnsafePointer[OpaquePointer],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNLinLayerBiasParams",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            Int16,
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[cudnnFilterStruct],
            OpaquePointer,
            Int16,
            UnsafePointer[cudnnFilterStruct],
            UnsafePointer[OpaquePointer],
        ) -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        pseudo_layer,
        x_desc,
        w_desc,
        w,
        lin_layer_id,
        lin_layer_bias_desc,
        lin_layer_bias,
    )


fn cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    count: UnsafePointer[Int16],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNForwardInferenceAlgorithmMaxCount",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            UnsafePointer[Int16],
        ) -> cudnnStatus_t,
    ]()(handle, rnn_desc, count)


fn cudnnGetRNNWeightParams(
    handle: UnsafePointer[cudnnContext],
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    pseudo_layer: Int32,
    weight_space_size: Int,
    weight_space: OpaquePointer,
    lin_layer_id: Int32,
    m_desc: UnsafePointer[cudnnTensorStruct],
    m_addr: UnsafePointer[OpaquePointer],
    b_desc: UnsafePointer[cudnnTensorStruct],
    b_addr: UnsafePointer[OpaquePointer],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNWeightParams",
        fn (
            UnsafePointer[cudnnContext],
            UnsafePointer[cudnnRNNStruct],
            Int32,
            Int,
            OpaquePointer,
            Int32,
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[OpaquePointer],
            UnsafePointer[cudnnTensorStruct],
            UnsafePointer[OpaquePointer],
        ) -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        pseudo_layer,
        weight_space_size,
        weight_space,
        lin_layer_id,
        m_desc,
        m_addr,
        b_desc,
        b_addr,
    )


fn cudnnGetRNNDescriptor_v8(
    rnn_desc: UnsafePointer[cudnnRNNStruct],
    algo: UnsafePointer[cudnnRNNAlgo_t],
    cell_mode: UnsafePointer[cudnnRNNMode_t],
    bias_mode: UnsafePointer[cudnnRNNBiasMode_t],
    dir_mode: UnsafePointer[cudnnDirectionMode_t],
    input_mode: UnsafePointer[cudnnRNNInputMode_t],
    data_type: UnsafePointer[cudnnDataType_t],
    math_prec: UnsafePointer[cudnnDataType_t],
    math_type: UnsafePointer[cudnnMathType_t],
    input_size: UnsafePointer[Int32],
    hidden_size: UnsafePointer[Int32],
    proj_size: UnsafePointer[Int32],
    num_layers: UnsafePointer[Int32],
    dropout_desc: UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
    aux_flags: UnsafePointer[UInt32],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNDescriptor_v8",
        fn (
            UnsafePointer[cudnnRNNStruct],
            UnsafePointer[cudnnRNNAlgo_t],
            UnsafePointer[cudnnRNNMode_t],
            UnsafePointer[cudnnRNNBiasMode_t],
            UnsafePointer[cudnnDirectionMode_t],
            UnsafePointer[cudnnRNNInputMode_t],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnDataType_t],
            UnsafePointer[cudnnMathType_t],
            UnsafePointer[Int32],
            UnsafePointer[Int32],
            UnsafePointer[Int32],
            UnsafePointer[Int32],
            UnsafePointer[UnsafePointer[cudnnDropoutStruct]],
            UnsafePointer[UInt32],
        ) -> cudnnStatus_t,
    ]()(
        rnn_desc,
        algo,
        cell_mode,
        bias_mode,
        dir_mode,
        input_mode,
        data_type,
        math_prec,
        math_type,
        input_size,
        hidden_size,
        proj_size,
        num_layers,
        dropout_desc,
        aux_flags,
    )


@fieldwise_init
@register_passable("trivial")
struct cudnnSeqDataAxis_t(Writable):
    var _value: Int8
    alias CUDNN_SEQDATA_TIME_DIM = Self(0)
    alias CUDNN_SEQDATA_BATCH_DIM = Self(1)
    alias CUDNN_SEQDATA_BEAM_DIM = Self(2)
    alias CUDNN_SEQDATA_VECT_DIM = Self(3)

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
        if self is Self.CUDNN_SEQDATA_TIME_DIM:
            return writer.write("CUDNN_SEQDATA_TIME_DIM")
        if self is Self.CUDNN_SEQDATA_BATCH_DIM:
            return writer.write("CUDNN_SEQDATA_BATCH_DIM")
        if self is Self.CUDNN_SEQDATA_BEAM_DIM:
            return writer.write("CUDNN_SEQDATA_BEAM_DIM")
        if self is Self.CUDNN_SEQDATA_VECT_DIM:
            return writer.write("CUDNN_SEQDATA_VECT_DIM")
        abort("invalid cudnnSeqDataAxis_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("cudnnSeqDataAxis_t(", self, ")")

    fn __int__(self) -> Int:
        return Int(self._value)


fn cudnnSetRNNPaddingMode(
    rnn_desc: UnsafePointer[cudnnRNNStruct], padding_mode: Int16
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNPaddingMode",
        fn (UnsafePointer[cudnnRNNStruct], Int16) -> cudnnStatus_t,
    ]()(rnn_desc, padding_mode)


fn cudnnDestroyRNNDescriptor(
    rnn_desc: UnsafePointer[cudnnRNNStruct],
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyRNNDescriptor",
        fn (UnsafePointer[cudnnRNNStruct]) -> cudnnStatus_t,
    ]()(rnn_desc)


fn cudnnSetRNNDataDescriptor(
    rnn_data_desc: UnsafePointer[cudnnRNNDataStruct],
    data_type: cudnnDataType_t,
    layout: cudnnRNNDataLayout_t,
    max_seq_length: Int16,
    batch_size: Int16,
    vector_size: Int16,
    seq_length_array: OpaquePointer,
    padding_fill: OpaquePointer,
) -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNDataDescriptor",
        fn (
            UnsafePointer[cudnnRNNDataStruct],
            cudnnDataType_t,
            cudnnRNNDataLayout_t,
            Int16,
            Int16,
            Int16,
            OpaquePointer,
            OpaquePointer,
        ) -> cudnnStatus_t,
    ]()(
        rnn_data_desc,
        data_type,
        layout,
        max_seq_length,
        batch_size,
        vector_size,
        seq_length_array,
        padding_fill,
    )
