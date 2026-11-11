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

from std.math import ceildiv

import extensibility
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from std.memory import unsafe_memcpy


from state_space.causal_conv1d import (
    causal_conv1d_channel_first_fwd_cpu,
    causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_update_cpu,
    enqueue_causal_conv1d_update_gpu,
)

from std.utils.coord import coord_to_index_list
from std.utils.index import IndexList
from extensibility import Input, Output, Tensor, TileTensorable
from layout import ImmTileTensor, MutTileTensor


# ============================================================================
# Causal Conv1D Registration
# ============================================================================


@extensibility.register("causal_conv1d")
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
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime dtype = type_of(input).dtype
        comptime assert (
            type_of(input).rank == 3
        ), "Input tensor must be rank 3 (batch, channels, seqlen)"
        comptime assert type_of(output).rank == 3
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(bias).rank == 1

        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")

        var X = input.to_tile_tensor[dtype]()
        var W = weight.to_tile_tensor[dtype]()
        var O = output.to_tile_tensor[dtype]()
        var B = bias.to_tile_tensor[dtype]()

        causal_conv1d(O, X, W, B, Self.activation == "silu", ctx)


def causal_conv1d[
    dtype: DType,
    //,
](
    output: MutTileTensor[dtype, address_space=.GENERIC, ...],
    input: ImmTileTensor[dtype, address_space=.GENERIC, ...],
    weight: ImmTileTensor[dtype, address_space=.GENERIC, ...],
    bias: ImmTileTensor[dtype, address_space=.GENERIC, ...],
    silu_activation: Bool,
    ctx: DeviceContext,
) raises:
    """Runs the `causal_conv1d` op on `(batch, channels, seqlen)` views.

    Parameters:
        dtype: Element type of every tensor.

    Args:
        output: Output view with the shape of `input`.
        input: Input view.
        weight: Weights with shape `(channels, width)`.
        bias: Per-channel bias with shape `(channels,)`.
        silu_activation: Whether to apply SiLU after the convolution.
        ctx: Device context to run on.
    """
    var batch_size = Int(input.dim[0]())
    var dim = Int(input.dim[1]())
    var seqlen = Int(input.dim[2]())
    var width = Int(weight.dim[1]())

    comptime if ctx.target.is_cpu():
        causal_conv1d_channel_first_fwd_cpu[
            input.dtype,
            weight.dtype,
            output.dtype,
            bias.dtype,
        ](
            batch_size,
            dim,
            seqlen,
            width,
            input,
            weight,
            output,
            bias,
            silu_activation,
            Optional(ctx.assert_cpu_target()),
        )
    elif ctx.target.is_gpu():
        var gpu_ctx = ctx
        comptime kNThreads = 128
        comptime kNElts = 4
        match width:
            case 1:
                comptime kWidth = 1
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        input.dtype,
                        weight.dtype,
                        output.dtype,
                        kNThreads,
                        kWidth,
                        kNElts,
                        bias.dtype,
                        input.LayoutType,
                        weight.LayoutType,
                        output.LayoutType,
                        bias.LayoutType,
                        input.Engine,
                        weight.Engine,
                        output.Engine,
                        bias.Engine,
                    ]
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch_size),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(width),
                    input,
                    weight,
                    output,
                    bias,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(Int(input.dim[2]()), kNThreads * kNElts),
                        Int(input.dim[1]()),
                        Int(input.dim[0]()),
                    ),
                    block_dim=(kNThreads),
                )
            case 2:
                comptime kWidth = 2
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        input.dtype,
                        weight.dtype,
                        output.dtype,
                        kNThreads,
                        kWidth,
                        kNElts,
                        bias.dtype,
                        input.LayoutType,
                        weight.LayoutType,
                        output.LayoutType,
                        bias.LayoutType,
                        input.Engine,
                        weight.Engine,
                        output.Engine,
                        bias.Engine,
                    ]
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch_size),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(width),
                    input,
                    weight,
                    output,
                    bias,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(Int(input.dim[2]()), kNThreads * kNElts),
                        Int(input.dim[1]()),
                        Int(input.dim[0]()),
                    ),
                    block_dim=(kNThreads),
                )
            case 3:
                comptime kWidth = 3
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        input.dtype,
                        weight.dtype,
                        output.dtype,
                        kNThreads,
                        kWidth,
                        kNElts,
                        bias.dtype,
                        input.LayoutType,
                        weight.LayoutType,
                        output.LayoutType,
                        bias.LayoutType,
                        input.Engine,
                        weight.Engine,
                        output.Engine,
                        bias.Engine,
                    ]
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch_size),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(width),
                    input,
                    weight,
                    output,
                    bias,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(Int(input.dim[2]()), kNThreads * kNElts),
                        Int(input.dim[1]()),
                        Int(input.dim[0]()),
                    ),
                    block_dim=(kNThreads),
                )
            case 4:
                comptime kWidth = 4
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_channel_first_fwd_gpu[
                        input.dtype,
                        weight.dtype,
                        output.dtype,
                        kNThreads,
                        kWidth,
                        kNElts,
                        bias.dtype,
                        input.LayoutType,
                        weight.LayoutType,
                        output.LayoutType,
                        bias.LayoutType,
                        input.Engine,
                        weight.Engine,
                        output.Engine,
                        bias.Engine,
                    ]
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch_size),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(width),
                    input,
                    weight,
                    output,
                    bias,
                    silu_activation_int8,
                    grid_dim=(
                        ceildiv(Int(input.dim[2]()), kNThreads * kNElts),
                        Int(input.dim[1]()),
                        Int(input.dim[0]()),
                    ),
                    block_dim=(kNThreads),
                )
            case _:
                raise Error(
                    "Unsupported kernel width: only widths 1, 2, 3, 4 are"
                    " supported"
                )
    else:
        raise Error("Unsupported target device")


@extensibility.register_shape_function("causal_conv1d")
def causal_conv1d_shape(
    input: Some[Tensor],
    weight: Some[Tensor],
    bias: Some[Tensor],
) -> IndexList[type_of(input).rank]:
    """Returns the output shape for the `causal_conv1d` op.

    Causal 1D convolution preserves the input shape: output has the same
    `(batch, channels, seqlen)` as `input`.

    Args:
        input: Input tensor with shape `(batch, channels, seqlen)`.
        weight: Convolution weights with shape `(channels, width)`.
        bias: Per-channel bias with shape `(channels,)`.

    Returns:
        The output tensor shape, equal to `input.shape()`.
    """
    comptime assert type_of(weight).rank == 2, "weight must be rank 2"
    comptime assert (
        type_of(weight).dtype == type_of(input).dtype
    ), "weight dtype must match input dtype"
    comptime assert type_of(bias).rank == 1, "bias must be rank 1"
    comptime assert (
        type_of(bias).dtype == type_of(input).dtype
    ), "bias dtype must match input dtype"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape())
    )


# ===----------------------------------------------------------------------=== #
# Causal Conv1D Update Operation (Autoregressive)
# ===----------------------------------------------------------------------=== #


@extensibility.register("causal_conv1d_update")
struct CausalConv1DUpdate[activation: StaticString]:
    """Incremental causal conv1d update for autoregressive decoding.

    This operation accepts the previous conv_state as an input and produces
    the updated conv_state as a separate output, compatible with functional
    graph semantics (no in-place mutation).

    Parameters:
        activation: Activation function to apply: `"none"` or `"silu"`.

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
    def execute(
        output: Some[Output & Tensor],
        conv_state: Some[Output & Tensor],
        input: Some[Input & Tensor],
        conv_state_in: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(conv_state), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(conv_state_in), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime dtype = type_of(input).dtype
        comptime assert (
            type_of(input).rank == 3
        ), "Input tensor must be rank 3 (batch, channels, seqlen)"
        comptime assert type_of(output).rank == 3
        comptime assert type_of(conv_state).rank == 3
        comptime assert type_of(conv_state_in).rank == 3
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(bias).rank == 1

        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")
        if Int(conv_state.shape()[0].value()) != Int(
            input.shape()[0].value()
        ) or Int(conv_state.shape()[1].value()) != Int(
            input.shape()[1].value()
        ):
            raise Error(
                "conv_state batch and channel dimensions must match input"
            )

        var X = input.to_tile_tensor[dtype]()
        var CS = conv_state.to_tile_tensor[dtype]()
        var CS_IN = conv_state_in.to_tile_tensor[dtype]()
        var W = weight.to_tile_tensor[dtype]()
        var O = output.to_tile_tensor[dtype]()
        var B = bias.to_tile_tensor[dtype]()

        causal_conv1d_update(
            O, CS, X, CS_IN, W, B, Self.activation == "silu", ctx
        )


def causal_conv1d_update[
    dtype: DType,
    //,
](
    output: MutTileTensor[dtype, address_space=.GENERIC, ...],
    conv_state: MutTileTensor[dtype, address_space=.GENERIC, ...],
    input: ImmTileTensor[dtype, address_space=.GENERIC, ...],
    conv_state_in: ImmTileTensor[dtype, address_space=.GENERIC, ...],
    weight: ImmTileTensor[dtype, address_space=.GENERIC, ...],
    bias: ImmTileTensor[dtype, address_space=.GENERIC, ...],
    silu_activation: Bool,
    ctx: DeviceContext,
) raises:
    """Runs the `causal_conv1d_update` op, writing the new state separately.

    Parameters:
        dtype: Element type of every tensor.

    Args:
        output: Convolution output with the shape of `input`.
        conv_state: Updated state with shape `(batch, channels, state_len)`.
        input: New input tokens with shape `(batch, channels, seqlen)`.
        conv_state_in: Previous state with the shape of `conv_state`.
        weight: Weights with shape `(channels, width)`.
        bias: Per-channel bias with shape `(channels,)`.
        silu_activation: Whether to apply SiLU after the convolution.
        ctx: Device context to run on.
    """
    var batch_size = Int(input.dim[0]())
    var dim = Int(input.dim[1]())
    var seqlen = Int(input.dim[2]())
    var width = Int(weight.dim[1]())
    var state_len = Int(conv_state.dim[2]())

    # Copy previous state into the output buffer so the kernel can
    # read old values and write updates into the same allocation.
    var total_state_elements = batch_size * dim * state_len

    comptime if ctx.target.is_cpu():
        unsafe_memcpy(
            dest=conv_state.unsafe_ptr(),
            src=conv_state_in.unsafe_ptr(),
            count=total_state_elements,
        )
        causal_conv1d_update_cpu[
            input.dtype,
            conv_state.dtype,
            weight.dtype,
            output.dtype,
            bias.dtype,
        ](
            batch_size,
            dim,
            seqlen,
            width,
            state_len,
            input,
            conv_state,
            weight,
            output,
            bias,
            silu_activation,
        )
    elif ctx.target.is_gpu():
        var gpu_ctx = ctx
        enqueue_causal_conv1d_update_gpu[separate_state=True](
            gpu_ctx,
            batch_size,
            dim,
            seqlen,
            width,
            state_len,
            input,
            conv_state_in,
            conv_state,
            weight,
            output,
            bias,
            silu_activation,
        )
    else:
        raise Error("Unsupported target device")


@extensibility.register_shape_function("causal_conv1d_update")
def causal_conv1d_update_shape(
    input: Some[Tensor],
    conv_state_in: Some[Tensor],
    weight: Some[Tensor],
    bias: Some[Tensor],
) -> Tuple[IndexList[type_of(input).rank], IndexList[type_of(input).rank]]:
    """Returns the output shapes for the `causal_conv1d_update` op.

    The update produces two tensors: the convolution output for the new
    token(s) and the updated convolution state.

    Args:
        input: New input tokens with shape `(batch, channels, seqlen)`.
        conv_state_in: Previous convolution state with shape
            `(batch, channels, state_len)`.
        weight: Convolution weights with shape `(channels, width)`.
        bias: Per-channel bias with shape `(channels,)`.

    Returns:
        A tuple `(output_shape, conv_state_shape)` where `output_shape`
        matches `input.shape()` and `conv_state_shape` matches
        `conv_state_in.shape()`.
    """
    comptime assert (
        type_of(conv_state_in).rank == type_of(input).rank
    ), "conv_state_in rank must match input rank"
    comptime assert (
        type_of(conv_state_in).dtype == type_of(input).dtype
    ), "conv_state_in dtype must match input dtype"
    comptime assert type_of(weight).rank == 2, "weight must be rank 2"
    comptime assert (
        type_of(weight).dtype == type_of(input).dtype
    ), "weight dtype must match input dtype"
    comptime assert type_of(bias).rank == 1, "bias must be rank 1"
    comptime assert (
        type_of(bias).dtype == type_of(input).dtype
    ), "bias dtype must match input dtype"
    return (
        rebind[IndexList[type_of(input).rank]](
            coord_to_index_list(input.shape())
        ),
        rebind[IndexList[type_of(input).rank]](
            coord_to_index_list(conv_state_in.shape())
        ),
    )
