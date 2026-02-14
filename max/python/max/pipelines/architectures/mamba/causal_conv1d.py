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
"""Causal Conv1D functional API for Mamba models.

Extracted from max.nn.conv for use in the Mamba SSM pipeline.
"""

from __future__ import annotations

from max.graph import DeviceRef, TensorValue, ops
from max.graph.type import TensorType


def causal_conv1d_fn(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None = None,
    algorithm: str = "optimized",
    activation: str = "none",
) -> TensorValue:
    """Causal 1D convolution function matching Mamba API.

    Performs causal (autoregressive) 1D convolution where each output position
    depends only on current and past input positions. This is used in Mamba
    models for sequence processing.

    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/causal_conv1d.py

    Args:
        x: Input tensor of shape (batch, channels, seqlen).
        weight: Weight tensor of shape (channels, width).
        bias: Optional bias tensor of shape (channels,).
        algorithm: Convolution algorithm to use.
            - "naive": Simple loop-based implementation, works for all widths.
            - "optimized": SIMD-vectorized, supports widths 1, 2, 3, 4 on GPU.
        activation: Activation function to apply after convolution.
            - "none": No activation (identity).
            - "silu": SiLU/Swish activation (x * sigmoid(x)).

    Returns:
        Output tensor of shape (batch, channels, seqlen), same as input.
    """
    # Normalize parameters
    algorithm_param = algorithm.lower() if algorithm else "optimized"
    if algorithm_param not in ["naive", "optimized"]:
        algorithm_param = "optimized"

    activation_param = activation.lower() if activation else "none"
    if activation_param not in ["none", "silu", "swish"]:
        activation_param = "none"
    if activation_param == "swish":
        activation_param = "silu"

    # Prepare tensors - ensure they're on the same device
    weight_cast = weight.cast(x.dtype)
    if x.device:
        weight_cast = weight_cast.to(x.device)

    # Create bias tensor (required by kernel, use zeros if not provided)
    if bias is None:
        # Create zero bias tensor
        bias_tensor = ops.broadcast_to(
            ops.constant(0.0, x.dtype, device=x.device or DeviceRef.CPU()),
            shape=(x.shape[-2],),  # channels dimension
        )
    else:
        bias_tensor = bias.cast(x.dtype)
        if x.device:
            bias_tensor = bias_tensor.to(x.device)

    # Call causal_conv1d kernel
    result = ops.custom(
        "causal_conv1d",
        x.device,
        [x, weight_cast, bias_tensor],
        [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        parameters={
            "activation": activation_param,
        },
    )

    # Ensure output tensor has a contiguous memory layout for compatibility
    # with GPU kernels that may have strict layout requirements
    output_tensor = result[0].tensor
    return ops.reshape(output_tensor, output_tensor.shape)


def causal_conv1d_update_fn(
    x: TensorValue,
    conv_state: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None = None,
    activation: str = "none",
) -> TensorValue:
    """Causal 1D convolution update function for autoregressive generation.

    Performs incremental causal convolution update for token-by-token generation.
    This maintains a sliding window state for efficient autoregressive inference.

    Args:
        x: Input tensor of shape (batch, channels, seqlen) - typically seqlen=1.
        conv_state: Convolution state tensor of shape (batch, channels, state_len).
            Modified in-place. Should be initialized with zeros for first call.
        weight: Weight tensor of shape (channels, width).
        bias: Optional bias tensor of shape (channels,).
        activation: Activation function to apply after convolution.
            - "none": No activation (identity).
            - "silu": SiLU/Swish activation (x * sigmoid(x)).

    Returns:
        Output tensor of shape (batch, channels, seqlen), same as input.
    """
    activation_param = activation.lower() if activation else "none"
    if activation_param not in ["none", "silu", "swish"]:
        activation_param = "none"
    if activation_param == "swish":
        activation_param = "silu"

    # Prepare tensors - ensure they're on the same device
    weight_cast = weight.cast(x.dtype)
    if x.device:
        weight_cast = weight_cast.to(x.device)

    # Create bias tensor (required by kernel, use zeros if not provided)
    if bias is None:
        bias_tensor = ops.broadcast_to(
            ops.constant(0.0, x.dtype, device=x.device or DeviceRef.CPU()),
            shape=(x.shape[-2],),  # channels dimension
        )
    else:
        bias_tensor = bias.cast(x.dtype)
        if x.device:
            bias_tensor = bias_tensor.to(x.device)

    # Ensure conv_state is on the same device
    conv_state_cast = conv_state.cast(x.dtype)
    if x.device:
        conv_state_cast = conv_state_cast.to(x.device)

    # Call causal_conv1d_update kernel
    result = ops.custom(
        "causal_conv1d_update",
        x.device,
        [x, conv_state_cast, weight_cast, bias_tensor],
        [
            TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            TensorType(
                dtype=conv_state.dtype, shape=conv_state.shape, device=x.device
            ),
        ],
        parameters={
            "activation": activation_param,
        },
    )

    # Ensure output tensor has a contiguous memory layout for compatibility
    # with GPU kernels that may have strict layout requirements
    output_tensor = result[0].tensor
    return ops.reshape(output_tensor, output_tensor.shape)
