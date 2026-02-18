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
"""Causal Conv1D operation registrations for state space models.

Provides compiler-registered operations for causal 1D convolution:
- CausalConv1D: Forward pass convolution with optional SiLU activation
- CausalConv1DUpdate: Incremental update for autoregressive decoding
"""

from math import ceildiv

import compiler_internal as compiler
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from runtime.asyncrt import DeviceContextPtr

from state_space.causal_conv1d import (
    causal_conv1d_channel_first_fwd_cpu,
    causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_update_cpu,
    causal_conv1d_update_gpu,
)

from utils.index import IndexList
from tensor import InputTensor, OutputTensor


# ============================================================================
# Causal Conv1D Registration
# ============================================================================


@compiler.register("causal_conv1d")
struct CausalConv1D[activation: StaticString]:
    """Causal 1D convolution operation with bias.

    Performs causal (autoregressive) 1D convolution where each output position
    depends only on current and past input positions. Supports optional SiLU
    activation with SIMD-vectorized implementations for widths 1, 2, 3, 4.

    Parameters:
        activation: Activation function to apply after convolution.
            - "none": No activation (identity).
            - "silu": SiLU/Swish activation (x * sigmoid(x)).

    Tensor Shapes:
        - input: (batch, channels, seqlen) - Input sequence tensor.
        - weight: (channels, width) - Convolution weights per channel.
        - bias: (channels,) - Per-channel bias to add.
        - output: (batch, channels, seqlen) - Output tensor (same shape as input).
    """

    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if rank != 3:
            raise Error("Input tensor must be rank 3 (batch, channels, seqlen)")
        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")

        var X = input.to_layout_tensor()
        var W = weight.to_layout_tensor()
        var O = output.to_layout_tensor()
        var B = bias.to_layout_tensor()

        var batch_size: Int = input.dim_size(0)
        var dim: Int = input.dim_size(1)
        var seqlen: Int = input.dim_size(2)
        var width: Int = weight.dim_size(1)

        var x_batch_stride: UInt32 = UInt32(input.strides()[0])
        var x_c_stride: UInt32 = UInt32(input.strides()[1])
        var x_l_stride: UInt32 = UInt32(input.strides()[2])

        var weight_c_stride: UInt32 = UInt32(weight.strides()[0])
        var weight_width_stride: UInt32 = UInt32(weight.strides()[1])

        var out_batch_stride: UInt32 = UInt32(output.strides()[0])
        var out_c_stride: UInt32 = UInt32(output.strides()[1])
        var out_l_stride: UInt32 = UInt32(output.strides()[2])

        var bias_stride: UInt32 = UInt32(bias.strides()[0])

        var silu_activation = Self.activation == "silu"

        @parameter
        if is_cpu[target]():
            causal_conv1d_channel_first_fwd_cpu[
                X.dtype,
                X.layout,
                W.dtype,
                W.layout,
                O.dtype,
                O.layout,
                B.dtype,
                B.layout,
            ](
                batch_size,
                dim,
                seqlen,
                width,
                X,
                W,
                O,
                B,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                bias_stride,
                silu_activation,
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()
            comptime kNThreads = 128
            comptime kNElts = 4
            if width == 1:
                comptime kWidth = 1
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(X.dim(2), kNThreads * kNElts),
                        X.dim(1),
                        X.dim(0),
                    ),
                    block_dim=(kNThreads),
                )
            elif width == 2:
                comptime kWidth = 2
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(X.dim(2), kNThreads * kNElts),
                        X.dim(1),
                        X.dim(0),
                    ),
                    block_dim=(kNThreads),
                )
            elif width == 3:
                comptime kWidth = 3
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(X.dim(2), kNThreads * kNElts),
                        X.dim(1),
                        X.dim(0),
                    ),
                    block_dim=(kNThreads),
                )
            elif width == 4:
                comptime kWidth = 4
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(X.dim(2), kNThreads * kNElts),
                        X.dim(1),
                        X.dim(0),
                    ),
                    block_dim=(kNThreads),
                )
            else:
                raise Error(
                    "Unsupported kernel width: only widths 1, 2, 3, 4 are"
                    " supported"
                )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# Causal Conv1D Update Operation (Autoregressive)
# ===----------------------------------------------------------------------=== #


@compiler.register("causal_conv1d_update")
struct CausalConv1DUpdate[activation: StaticString]:
    """Incremental causal conv1d update for autoregressive decoding.

    This operation is designed for token-by-token generation where:
        1. A sliding window of recent inputs is maintained in conv_state.
        2. Each new input updates the state and produces an output.
        3. The conv_state is modified in-place for efficiency.

    Use this for:
        - Language model inference with autoregressive generation.
        - Real-time streaming applications.
        - Efficient incremental convolution without full sequence recomputation.

    Parameters:
        activation: "none" or "silu" - activation function to apply.

    Tensor Shapes:
        - input: (batch, channels, seqlen) - New input tokens (typically seqlen=1).
        - weight: (channels, width) - Convolution weights.
        - bias: (channels,) - Per-channel bias.
        - conv_state: (batch, channels, state_len) - Sliding window state (modified in-place).
        - output: (batch, channels, seqlen) - Convolution output for new tokens.

    State Management:
        The conv_state maintains the last (width-1) inputs for each channel.
        After update, the oldest values are shifted out and new inputs are appended.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        conv_state: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if rank != 3:
            raise Error("Input tensor must be rank 3 (batch, channels, seqlen)")
        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")
        if conv_state.dim_size(0) != input.dim_size(0) or conv_state.dim_size(
            1
        ) != input.dim_size(1):
            raise Error(
                "conv_state batch and channel dimensions must match input"
            )

        var X = input.to_layout_tensor()
        var CS = conv_state.to_layout_tensor()
        var W = weight.to_layout_tensor()
        var O = output.to_layout_tensor()
        var B = bias.to_layout_tensor()

        var batch_size: Int = input.dim_size(0)
        var dim: Int = input.dim_size(1)
        var seqlen: Int = input.dim_size(2)
        var width: Int = weight.dim_size(1)
        var state_len: Int = conv_state.dim_size(2)

        var x_batch_stride: UInt32 = UInt32(input.strides()[0])
        var x_c_stride: UInt32 = UInt32(input.strides()[1])
        var x_l_stride: UInt32 = UInt32(input.strides()[2])

        var conv_state_batch_stride: UInt32 = UInt32(conv_state.strides()[0])
        var conv_state_c_stride: UInt32 = UInt32(conv_state.strides()[1])
        var conv_state_l_stride: UInt32 = UInt32(conv_state.strides()[2])

        var weight_c_stride: UInt32 = UInt32(weight.strides()[0])
        var weight_width_stride: UInt32 = UInt32(weight.strides()[1])

        var out_batch_stride: UInt32 = UInt32(output.strides()[0])
        var out_c_stride: UInt32 = UInt32(output.strides()[1])
        var out_l_stride: UInt32 = UInt32(output.strides()[2])

        var bias_stride: UInt32 = UInt32(bias.strides()[0])

        var silu_activation = Self.activation == "silu"

        @parameter
        if is_cpu[target]():
            causal_conv1d_update_cpu[
                X.dtype,
                X.layout,
                CS.dtype,
                CS.layout,
                W.dtype,
                W.layout,
                O.dtype,
                O.layout,
                B.dtype,
                B.layout,
            ](
                batch_size,
                dim,
                seqlen,
                width,
                state_len,
                X,
                CS,
                W,
                O,
                B,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                conv_state_batch_stride,
                conv_state_c_stride,
                conv_state_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation,
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()
            comptime kNThreads = 128
            var compiled_func = gpu_ctx.compile_function[
                causal_conv1d_update_gpu[
                    X.dtype,
                    X.layout,
                    CS.dtype,
                    CS.layout,
                    W.dtype,
                    W.layout,
                    O.dtype,
                    O.layout,
                    B.dtype,
                    B.layout,
                    kNThreads,
                ],
                causal_conv1d_update_gpu[
                    X.dtype,
                    X.layout,
                    CS.dtype,
                    CS.layout,
                    W.dtype,
                    W.layout,
                    O.dtype,
                    O.layout,
                    B.dtype,
                    B.layout,
                    kNThreads,
                ],
            ]()
            var silu_activation_int8 = Int8(silu_activation)
            gpu_ctx.enqueue_function(
                compiled_func,
                batch_size,
                dim,
                seqlen,
                width,
                state_len,
                X,
                CS,
                W,
                O,
                B,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                conv_state_batch_stride,
                conv_state_c_stride,
                conv_state_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                bias_stride,
                silu_activation_int8,
                grid_dim=(batch_size, ceildiv(dim, kNThreads)),
                block_dim=(kNThreads),
            )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
    ) -> IndexList[rank]:
        return input.shape()
