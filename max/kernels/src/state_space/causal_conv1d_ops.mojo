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
- CausalConv1D: Forward channel-first (B, C, L) convolution with optional SiLU.
- CausalConv1DChannelLast: Forward channel-last (B, L, C) convolution.
- CausalConv1DUpdate: Incremental update for autoregressive decoding.

All forward kernels are parameterized by `has_bias` / `has_seq_idx` rather than
carrying a hand-copied variant per combination. These ops always supply a bias
and never a seq_idx mask, so they pass `has_bias = true`, `has_seq_idx = false`.
"""

from std.math import ceildiv

import extensibility as compiler
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.memory import memcpy


from state_space.causal_conv1d import (
    causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_channel_last_fwd_gpu,
    causal_conv1d_fwd_cpu,
    causal_conv1d_update_cpu,
    causal_conv1d_update_gpu,
)

from std.utils.index import IndexList
from extensibility import InputTensor, OutputTensor


# ============================================================================
# Causal Conv1D Registration (channel-first, B, C, L)
# ============================================================================


@compiler.register("causal_conv1d")
struct CausalConv1D[activation: StaticString]:
    """Causal 1D convolution operation with bias (channel-first layout).

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
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if rank != 3:
            raise Error("Input tensor must be rank 3 (batch, channels, seqlen)")
        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")

        var X = input.to_tile_tensor[DType.int32]()
        var W = weight.to_tile_tensor[DType.int32]()
        var O = output.to_tile_tensor[DType.int32]()
        var B = bias.to_tile_tensor[DType.int32]()

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

        comptime if is_cpu[target]():
            # seq_idx is unused (has_seq_idx=False); B stands in as a valid
            # tensor argument and is never dereferenced.
            causal_conv1d_fwd_cpu[
                X.dtype, W.dtype, O.dtype, B.dtype, B.dtype, True, False
            ](
                batch_size,
                dim,
                seqlen,
                width,
                X.as_immut(),
                W.as_immut(),
                O,
                B.as_immut(),
                B.as_immut(),
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                bias_stride,
                UInt32(0),
                UInt32(0),
                silu_activation,
                Optional[DeviceContext](ctx),
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx
            # 64 threads x kNElts=4 (float4) => a 256-position tile per block with
            # every thread busy. The channel-first kernel's vectorized fast path
            # (one float4 load + halo) beats Dao-AILab/causal-conv1d at mamba
            # prefill dimensions; kNThreads=64 avoids the half-idle block that
            # kNThreads=128 (tile 512) leaves at the common L=256.
            comptime kNThreads = 64
            comptime kNElts = 4
            var silu_activation_int8 = Int8(silu_activation)
            var grid = (
                ceildiv(Int(X.dim[2]()), kNThreads * kNElts),
                Int(X.dim[1]()),
                Int(X.dim[0]()),
            )

            @parameter
            @always_inline
            def launch[kWidth: Int]() raises:
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        W.dtype,
                        O.dtype,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.dtype,
                        X.LayoutType,
                        W.LayoutType,
                        O.LayoutType,
                        B.LayoutType,
                        B.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X.as_immut(),
                    W.as_immut(),
                    O,
                    B.as_immut(),
                    B.as_immut(),
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    UInt32(0),
                    UInt32(0),
                    Int8(True),
                    Int8(False),
                    silu_activation_int8,
                    grid_dim=grid,
                    block_dim=(kNThreads),
                )

            if width == 1:
                launch[1]()
            elif width == 2:
                launch[2]()
            elif width == 3:
                launch[3]()
            elif width == 4:
                launch[4]()
            else:
                raise Error(
                    "Unsupported kernel width: only widths 1, 2, 3, 4 are"
                    " supported"
                )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
    ) -> IndexList[rank]:
        return input.shape()


# ============================================================================
# Causal Conv1D Registration (channel-last, B, L, C)
# ============================================================================


@compiler.register("causal_conv1d_channel_last")
struct CausalConv1DChannelLast[activation: StaticString]:
    """Causal 1D convolution operation with bias (channel-last layout).

    Identical math to `CausalConv1D`, but for inputs/outputs laid out as
    (batch, seqlen, channels). The CPU path reuses the shared stride-driven
    core; the GPU path uses the channel-last kernel that coalesces along the
    contiguous channel axis.

    Parameters:
        activation: "none" or "silu" - activation function to apply.

    Tensor Shapes:
        - input: (batch, seqlen, channels) - Input sequence tensor.
        - weight: (channels, width) - Convolution weights per channel.
        - bias: (channels,) - Per-channel bias to add.
        - output: (batch, seqlen, channels) - Output tensor (same shape as input).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if rank != 3:
            raise Error("Input tensor must be rank 3 (batch, seqlen, channels)")
        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")

        var X = input.to_tile_tensor[DType.int32]()
        var W = weight.to_tile_tensor[DType.int32]()
        var O = output.to_tile_tensor[DType.int32]()
        var B = bias.to_tile_tensor[DType.int32]()

        var batch_size: Int = input.dim_size(0)
        var seqlen: Int = input.dim_size(1)
        var dim: Int = input.dim_size(2)
        var width: Int = weight.dim_size(1)

        # Channel-last strides: (B, L, C).
        var x_batch_stride: UInt32 = UInt32(input.strides()[0])
        var x_l_stride: UInt32 = UInt32(input.strides()[1])
        var x_c_stride: UInt32 = UInt32(input.strides()[2])

        var weight_c_stride: UInt32 = UInt32(weight.strides()[0])
        var weight_width_stride: UInt32 = UInt32(weight.strides()[1])

        var out_batch_stride: UInt32 = UInt32(output.strides()[0])
        var out_l_stride: UInt32 = UInt32(output.strides()[1])
        var out_c_stride: UInt32 = UInt32(output.strides()[2])

        var bias_stride: UInt32 = UInt32(bias.strides()[0])

        var silu_activation = Self.activation == "silu"

        comptime if is_cpu[target]():
            causal_conv1d_fwd_cpu[
                X.dtype, W.dtype, O.dtype, B.dtype, B.dtype, True, False
            ](
                batch_size,
                dim,
                seqlen,
                width,
                X.as_immut(),
                W.as_immut(),
                O,
                B.as_immut(),
                B.as_immut(),
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                bias_stride,
                UInt32(0),
                UInt32(0),
                silu_activation,
                Optional[DeviceContext](ctx),
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx
            # Channel-last maps threads to channels for coalesced loads: a block
            # owns kNThreads contiguous channels (full warps along the contiguous
            # C axis) and each thread scans kNElts sequence positions with a
            # sliding window. 64 channels x 8 positions measured fastest (and
            # beats Dao-AILab's channel-last kernel). grid.x scales with L,
            # grid.y with dim.
            comptime kNThreads = 64
            comptime kNElts = 8
            var silu_activation_int8 = Int8(silu_activation)
            var grid = (
                ceildiv(seqlen, kNElts),
                ceildiv(dim, kNThreads),
                batch_size,
            )

            @parameter
            @always_inline
            def launch[kWidth: Int]() raises:
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_last_fwd_gpu[
                        X.dtype,
                        W.dtype,
                        O.dtype,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.dtype,
                        X.LayoutType,
                        W.LayoutType,
                        O.LayoutType,
                        B.LayoutType,
                        B.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X.as_immut(),
                    W.as_immut(),
                    O,
                    B.as_immut(),
                    B.as_immut(),
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    UInt32(0),
                    UInt32(0),
                    Int8(True),
                    Int8(False),
                    silu_activation_int8,
                    grid_dim=grid,
                    block_dim=(kNThreads),
                )

            if width == 1:
                launch[1]()
            elif width == 2:
                launch[2]()
            elif width == 3:
                launch[3]()
            elif width == 4:
                launch[4]()
            else:
                raise Error(
                    "Unsupported kernel width: only widths 1, 2, 3, 4 are"
                    " supported"
                )
        else:
            raise Error("Unsupported target device")


@compiler.register_shape_function("causal_conv1d")
def causal_conv1d_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    weight: InputTensor[dtype=dtype, rank=2, ...],
    bias: InputTensor[dtype=dtype, rank=1, ...],
) -> IndexList[rank]:
    return input.shape()


# ===----------------------------------------------------------------------=== #
# Causal Conv1D Update Operation (Autoregressive)
# ===----------------------------------------------------------------------=== #


@compiler.register("causal_conv1d_update")
struct CausalConv1DUpdate[activation: StaticString]:
    """Incremental causal conv1d update for autoregressive decoding.

    This operation accepts the previous conv_state as an input and produces
    the updated conv_state as a separate output, compatible with functional
    graph semantics (no in-place mutation).

    Parameters:
        activation: "none" or "silu" - activation function to apply.

    Tensor Shapes:
        Outputs:
            - output: (batch, channels, seqlen) - Convolution output.
            - conv_state_out: (batch, channels, state_len) - Updated state.
        Inputs:
            - input: (batch, channels, seqlen) - New input tokens.
            - conv_state_in: (batch, channels, state_len) - Previous state.
            - weight: (channels, width) - Convolution weights.
            - bias: (channels,) - Per-channel bias.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        conv_state: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        conv_state_in: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
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

        var X = input.to_tile_tensor[DType.int32]()
        var CS = conv_state.to_tile_tensor[DType.int32]()
        var CS_IN = conv_state_in.to_tile_tensor[DType.int32]()
        var W = weight.to_tile_tensor[DType.int32]()
        var O = output.to_tile_tensor[DType.int32]()
        var B = bias.to_tile_tensor[DType.int32]()

        var batch_size: Int = input.dim_size(0)
        var dim: Int = input.dim_size(1)
        var seqlen: Int = input.dim_size(2)
        var width: Int = weight.dim_size(1)
        var state_len: Int = conv_state.dim_size(2)

        # Copy previous state into the output buffer so the kernel can
        # read old values and write updates into the same allocation.
        var total_state_elements = batch_size * dim * state_len

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

        var silu_activation = Self.activation == "silu"

        comptime if is_cpu[target]():
            memcpy(dest=CS.ptr, src=CS_IN.ptr, count=total_state_elements)
            causal_conv1d_update_cpu[
                X.dtype,
                CS.dtype,
                W.dtype,
                O.dtype,
                B.dtype,
                True,
            ](
                batch_size,
                dim,
                seqlen,
                width,
                state_len,
                X.as_immut(),
                CS,
                W.as_immut(),
                O,
                B.as_immut(),
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
            var gpu_ctx: DeviceContext = ctx
            gpu_ctx.enqueue_copy(CS.ptr, CS_IN.ptr, total_state_elements)
            comptime kNThreads = 128
            var compiled_func = gpu_ctx.compile_function[
                causal_conv1d_update_gpu[
                    X.dtype,
                    CS.dtype,
                    W.dtype,
                    O.dtype,
                    B.dtype,
                    kNThreads,
                    X.LayoutType,
                    CS.LayoutType,
                    W.LayoutType,
                    O.LayoutType,
                    B.LayoutType,
                ]
            ]()
            var silu_activation_int8 = Int8(silu_activation)
            gpu_ctx.enqueue_function(
                compiled_func,
                batch_size,
                dim,
                seqlen,
                width,
                state_len,
                X.as_immut(),
                CS,
                W.as_immut(),
                O,
                B.as_immut(),
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
                Int8(True),
                silu_activation_int8,
                grid_dim=(batch_size, ceildiv(dim, kNThreads)),
                block_dim=(kNThreads),
            )
        else:
            raise Error("Unsupported target device")


@compiler.register_shape_function("causal_conv1d_update")
def causal_conv1d_update_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    conv_state_in: InputTensor[dtype=dtype, rank=rank, ...],
    weight: InputTensor[dtype=dtype, rank=2, ...],
    bias: InputTensor[dtype=dtype, rank=1, ...],
) -> Tuple[IndexList[rank], IndexList[rank]]:
    return (input.shape(), conv_state_in.shape())
