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
"""Varlen Causal Conv1D operation registrations for Mamba SSM.

This module registers operations for variable-length causal 1D convolution:
- causal_conv1d_varlen_fwd: Forward pass for varlen sequences
- causal_conv1d_varlen_update: Update for decode/autoregressive inference
- causal_conv1d_varlen_states: Extract states from varlen sequences
"""

from math import ceildiv

import compiler_internal as compiler
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from utils.index import IndexList

from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_fwd_gpu,
    causal_conv1d_varlen_update_cpu,
    causal_conv1d_varlen_update_gpu,
    causal_conv1d_varlen_states_cpu,
    causal_conv1d_varlen_states_gpu,
)


# ============================================================================
# Varlen Causal Conv1D Forward Registration
# ============================================================================


@compiler.register("causal_conv1d_varlen_fwd")
struct CausalConv1DVarlenFwd[activation: StaticString]:
    """Varlen causal 1D convolution forward pass.

    Performs causal 1D convolution on variable-length sequences that are
    concatenated together. Uses cumulative sequence lengths to identify
    sequence boundaries.

    Parameters:
        activation: Activation function - "none" or "silu".

    Tensor Shapes:
        - output: (dim, total_seqlen) - Output tensor
        - x: (dim, total_seqlen) - Input tensor (concatenated sequences)
        - weight: (dim, width) - Convolution weights per channel
        - bias: (dim,) - Per-channel bias
        - query_start_loc: (batch + 1,) - Cumulative sequence lengths
        - cache_indices: (batch,) - Indices into conv_states (optional)
        - has_initial_state: (batch,) - Whether to use initial state (optional)
        - conv_states: (batch, dim, width - 1) - Conv states (optional, in/out)
    """

    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        conv_states: OutputTensor[dtype=dtype, rank=3],
        x: InputTensor[dtype=dtype, rank=2],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        query_start_loc: InputTensor[dtype = DType.int32, rank=1],
        cache_indices: InputTensor[dtype = DType.int32, rank=1],
        has_initial_state: InputTensor[dtype = DType.bool, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var dim = x.dim_size(0)
        var total_seqlen = x.dim_size(1)
        var width = weight.dim_size(1)
        var batch = query_start_loc.dim_size(0) - 1

        var output_lt = output.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var weight_lt = weight.to_layout_tensor()
        var bias_lt = bias.to_layout_tensor()
        var query_start_loc_lt = query_start_loc.to_layout_tensor()
        var cache_indices_lt = cache_indices.to_layout_tensor()
        var has_initial_state_lt = has_initial_state.to_layout_tensor()
        var conv_states_lt = conv_states.to_layout_tensor()

        # Get strides as UInt32
        var x_strides = x.strides()
        var weight_strides = weight.strides()
        var output_strides = output.strides()
        var conv_states_strides = conv_states.strides()

        var x_dim_stride = UInt32(x_strides[0])
        var x_seqlen_stride = UInt32(x_strides[1])
        var weight_dim_stride = UInt32(weight_strides[0])
        var weight_width_stride = UInt32(weight_strides[1])
        var out_dim_stride = UInt32(output_strides[0])
        var out_seqlen_stride = UInt32(output_strides[1])

        var has_conv_states = conv_states.dim_size(0) > 0
        var conv_states_batch_stride = UInt32(
            conv_states_strides[0] if has_conv_states else 0
        )
        var conv_states_dim_stride = UInt32(
            conv_states_strides[1] if has_conv_states else 0
        )
        var conv_states_width_stride = UInt32(
            conv_states_strides[2] if has_conv_states else 0
        )

        var has_cache_indices = cache_indices.dim_size(0) > 0
        var has_initial_state_flag = has_initial_state.dim_size(0) > 0
        var has_bias = bias.dim_size(0) > 0

        var silu_activation = Self.activation == "silu"
        comptime PAD_SLOT_ID: Int32 = -1

        comptime if is_cpu[target]():
            causal_conv1d_varlen_fwd_cpu[
                x_lt.dtype,
                x_lt.layout,
                weight_lt.dtype,
                weight_lt.layout,
                bias_lt.dtype,
                bias_lt.layout,
                output_lt.dtype,
                output_lt.layout,
                query_start_loc_lt.dtype,
                query_start_loc_lt.layout,
                cache_indices_lt.dtype,
                cache_indices_lt.layout,
                has_initial_state_lt.dtype,
                has_initial_state_lt.layout,
                conv_states_lt.dtype,
                conv_states_lt.layout,
            ](
                dim,
                total_seqlen,
                width,
                batch,
                x_lt,
                weight_lt,
                bias_lt,
                query_start_loc_lt,
                cache_indices_lt,
                has_initial_state_lt,
                conv_states_lt,
                output_lt,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                out_dim_stride,
                out_seqlen_stride,
                conv_states_batch_stride,
                conv_states_dim_stride,
                conv_states_width_stride,
                silu_activation,
                PAD_SLOT_ID,
                has_cache_indices,
                has_initial_state_flag,
                has_conv_states,
                has_bias,
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            comptime BLOCK_DIM = 128
            comptime BLOCK_SEQ = 1
            var silu_activation_int8 = Int8(silu_activation)

            if width == 1:
                comptime kWidth = 1
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    dim,
                    total_seqlen,
                    batch,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    query_start_loc_lt,
                    cache_indices_lt,
                    has_initial_state_lt,
                    conv_states_lt,
                    output_lt,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    conv_states_batch_stride,
                    conv_states_dim_stride,
                    conv_states_width_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_cache_indices),
                    Int8(has_initial_state_flag),
                    Int8(has_conv_states),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_SEQ),
                )
            elif width == 2:
                comptime kWidth = 2
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    dim,
                    total_seqlen,
                    batch,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    query_start_loc_lt,
                    cache_indices_lt,
                    has_initial_state_lt,
                    conv_states_lt,
                    output_lt,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    conv_states_batch_stride,
                    conv_states_dim_stride,
                    conv_states_width_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_cache_indices),
                    Int8(has_initial_state_flag),
                    Int8(has_conv_states),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_SEQ),
                )
            elif width == 3:
                comptime kWidth = 3
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    dim,
                    total_seqlen,
                    batch,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    query_start_loc_lt,
                    cache_indices_lt,
                    has_initial_state_lt,
                    conv_states_lt,
                    output_lt,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    conv_states_batch_stride,
                    conv_states_dim_stride,
                    conv_states_width_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_cache_indices),
                    Int8(has_initial_state_flag),
                    Int8(has_conv_states),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_SEQ),
                )
            elif width == 4:
                comptime kWidth = 4
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                    causal_conv1d_varlen_fwd_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        query_start_loc_lt.dtype,
                        query_start_loc_lt.layout,
                        cache_indices_lt.dtype,
                        cache_indices_lt.layout,
                        has_initial_state_lt.dtype,
                        has_initial_state_lt.layout,
                        conv_states_lt.dtype,
                        conv_states_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    dim,
                    total_seqlen,
                    batch,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    query_start_loc_lt,
                    cache_indices_lt,
                    has_initial_state_lt,
                    conv_states_lt,
                    output_lt,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    conv_states_batch_stride,
                    conv_states_dim_stride,
                    conv_states_width_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_cache_indices),
                    Int8(has_initial_state_flag),
                    Int8(has_conv_states),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_SEQ),
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
    ](
        x: InputTensor[dtype=dtype, rank=2],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        query_start_loc: InputTensor[dtype = DType.int32, rank=1],
        cache_indices: InputTensor[dtype = DType.int32, rank=1],
        has_initial_state: InputTensor[dtype = DType.bool, rank=1],
    ) -> IndexList[2]:
        return x.shape()


# ============================================================================
# Varlen Causal Conv1D Update Registration
# ============================================================================


@compiler.register("causal_conv1d_varlen_update")
struct CausalConv1DVarlenUpdate[activation: StaticString]:
    """Varlen causal conv1d update for autoregressive decoding.

    Performs incremental convolution update for token-by-token generation.
    Updates the conv_state in-place with new input values.

    Parameters:
        activation: Activation function - "none" or "silu".

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, seqlen) - Input tensor
        - weight: (dim, width) - Convolution weights
        - bias: (dim,) - Per-channel bias
        - conv_state: (batch, dim, state_len) - Conv state (in/out)
        - cache_seqlens: (batch,) - Current sequence lengths (optional)
        - conv_state_indices: (batch,) - Indices into conv_state (optional)
    """

    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        conv_state: OutputTensor[dtype=dtype, rank=3],
        x: InputTensor[dtype=dtype, rank=3],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        cache_seqlens: InputTensor[dtype = DType.int32, rank=1],
        conv_state_indices: InputTensor[dtype = DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var batch = x.dim_size(0)
        var dim = x.dim_size(1)
        var seqlen = x.dim_size(2)
        var width = weight.dim_size(1)
        var state_len = conv_state.dim_size(2)

        var output_lt = output.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var weight_lt = weight.to_layout_tensor()
        var bias_lt = bias.to_layout_tensor()
        var conv_state_lt = conv_state.to_layout_tensor()
        var cache_seqlens_lt = cache_seqlens.to_layout_tensor()
        var conv_state_indices_lt = conv_state_indices.to_layout_tensor()

        var x_strides = x.strides()
        var weight_strides = weight.strides()
        var output_strides = output.strides()
        var conv_state_strides = conv_state.strides()

        var x_batch_stride = UInt32(x_strides[0])
        var x_dim_stride = UInt32(x_strides[1])
        var x_seqlen_stride = UInt32(x_strides[2])
        var weight_dim_stride = UInt32(weight_strides[0])
        var weight_width_stride = UInt32(weight_strides[1])
        var conv_state_batch_stride = UInt32(conv_state_strides[0])
        var conv_state_dim_stride = UInt32(conv_state_strides[1])
        var conv_state_seqlen_stride = UInt32(conv_state_strides[2])
        var out_batch_stride = UInt32(output_strides[0])
        var out_dim_stride = UInt32(output_strides[1])
        var out_seqlen_stride = UInt32(output_strides[2])

        var has_conv_state_indices = conv_state_indices.dim_size(0) > 0
        var has_cache_seqlens = cache_seqlens.dim_size(0) > 0
        var has_bias = bias.dim_size(0) > 0

        var silu_activation = Self.activation == "silu"
        comptime PAD_SLOT_ID: Int32 = -1

        comptime if is_cpu[target]():
            causal_conv1d_varlen_update_cpu[
                x_lt.dtype,
                x_lt.layout,
                weight_lt.dtype,
                weight_lt.layout,
                bias_lt.dtype,
                bias_lt.layout,
                output_lt.dtype,
                output_lt.layout,
                conv_state_lt.dtype,
                conv_state_lt.layout,
                cache_seqlens_lt.dtype,
                cache_seqlens_lt.layout,
                conv_state_indices_lt.dtype,
                conv_state_indices_lt.layout,
            ](
                batch,
                dim,
                seqlen,
                width,
                state_len,
                x_lt,
                weight_lt,
                bias_lt,
                conv_state_lt,
                cache_seqlens_lt,
                conv_state_indices_lt,
                output_lt,
                x_batch_stride,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                conv_state_batch_stride,
                conv_state_dim_stride,
                conv_state_seqlen_stride,
                out_batch_stride,
                out_dim_stride,
                out_seqlen_stride,
                silu_activation,
                PAD_SLOT_ID,
                has_conv_state_indices,
                has_cache_seqlens,
                has_bias,
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            comptime BLOCK_DIM = 128
            var silu_activation_int8 = Int8(silu_activation)

            if width == 1:
                comptime kWidth = 1
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch,
                    dim,
                    seqlen,
                    state_len,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    conv_state_lt,
                    cache_seqlens_lt,
                    conv_state_indices_lt,
                    output_lt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
                )
            elif width == 2:
                comptime kWidth = 2
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch,
                    dim,
                    seqlen,
                    state_len,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    conv_state_lt,
                    cache_seqlens_lt,
                    conv_state_indices_lt,
                    output_lt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
                )
            elif width == 3:
                comptime kWidth = 3
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch,
                    dim,
                    seqlen,
                    state_len,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    conv_state_lt,
                    cache_seqlens_lt,
                    conv_state_indices_lt,
                    output_lt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
                )
            elif width == 4:
                comptime kWidth = 4
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                    causal_conv1d_varlen_update_gpu[
                        x_lt.dtype,
                        x_lt.layout,
                        weight_lt.dtype,
                        weight_lt.layout,
                        bias_lt.dtype,
                        bias_lt.layout,
                        output_lt.dtype,
                        output_lt.layout,
                        conv_state_lt.dtype,
                        conv_state_lt.layout,
                        cache_seqlens_lt.dtype,
                        cache_seqlens_lt.layout,
                        conv_state_indices_lt.dtype,
                        conv_state_indices_lt.layout,
                        kWidth,
                        BLOCK_DIM,
                    ],
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch,
                    dim,
                    seqlen,
                    state_len,
                    x_lt,
                    weight_lt,
                    bias_lt,
                    conv_state_lt,
                    cache_seqlens_lt,
                    conv_state_indices_lt,
                    output_lt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
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
    ](
        x: InputTensor[dtype=dtype, rank=3],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        cache_seqlens: InputTensor[dtype = DType.int32, rank=1],
        conv_state_indices: InputTensor[dtype = DType.int32, rank=1],
    ) -> IndexList[3]:
        return x.shape()


# ============================================================================
# Varlen Causal Conv1D States Registration
# ============================================================================


@compiler.register("causal_conv1d_varlen_states")
struct CausalConv1DVarlenStates:
    """Extract conv states from variable-length sequences.

    Extracts the last state_len elements from each sequence to initialize
    conv_state for subsequent autoregressive generation.

    Tensor Shapes:
        - states: (batch, dim, state_len) - Output states tensor
        - x: (total_tokens, dim) - Input tensor (concatenated sequences)
        - cu_seqlens: (batch + 1,) - Cumulative sequence lengths
    """

    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        states: OutputTensor[dtype=dtype, rank=3],
        x: InputTensor[dtype=dtype, rank=2],
        cu_seqlens: InputTensor[dtype = DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var total_tokens = x.dim_size(0)
        var dim = x.dim_size(1)
        var batch = cu_seqlens.dim_size(0) - 1
        var state_len = states.dim_size(2)

        var states_lt = states.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var cu_seqlens_lt = cu_seqlens.to_layout_tensor()

        var x_strides = x.strides()
        var states_strides = states.strides()

        var x_seqlen_stride = UInt32(x_strides[0])
        var x_dim_stride = UInt32(x_strides[1])
        var states_batch_stride = UInt32(states_strides[0])
        var states_dim_stride = UInt32(states_strides[1])
        var states_seqlen_stride = UInt32(states_strides[2])

        comptime if is_cpu[target]():
            causal_conv1d_varlen_states_cpu[
                x_lt.dtype,
                x_lt.layout,
                cu_seqlens_lt.dtype,
                cu_seqlens_lt.layout,
                states_lt.dtype,
                states_lt.layout,
            ](
                total_tokens,
                dim,
                batch,
                state_len,
                x_lt,
                cu_seqlens_lt,
                states_lt,
                x_seqlen_stride,
                x_dim_stride,
                states_batch_stride,
                states_dim_stride,
                states_seqlen_stride,
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            comptime BLOCK_DIM = 128
            var compiled_func = gpu_ctx.compile_function[
                causal_conv1d_varlen_states_gpu[
                    x_lt.dtype,
                    x_lt.layout,
                    cu_seqlens_lt.dtype,
                    cu_seqlens_lt.layout,
                    states_lt.dtype,
                    states_lt.layout,
                    BLOCK_DIM,
                    BLOCK_DIM,
                ],
                causal_conv1d_varlen_states_gpu[
                    x_lt.dtype,
                    x_lt.layout,
                    cu_seqlens_lt.dtype,
                    cu_seqlens_lt.layout,
                    states_lt.dtype,
                    states_lt.layout,
                    BLOCK_DIM,
                    BLOCK_DIM,
                ],
            ]()
            gpu_ctx.enqueue_function(
                compiled_func,
                total_tokens,
                dim,
                batch,
                state_len,
                x_lt,
                cu_seqlens_lt,
                states_lt,
                x_seqlen_stride,
                x_dim_stride,
                states_batch_stride,
                states_dim_stride,
                states_seqlen_stride,
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM,),
            )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    fn shape[
        dtype: DType,
    ](
        x: InputTensor[dtype=dtype, rank=2],
        cu_seqlens: InputTensor[dtype = DType.int32, rank=1],
    ) -> IndexList[3]:
        var batch = cu_seqlens.dim_size(0) - 1
        var dim = x.dim_size(1)
        # state_len is derived from the output tensor shape at runtime
        # Return a placeholder shape; actual shape determined by output allocation
        return IndexList[3](batch, dim, 0)
