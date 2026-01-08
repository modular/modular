# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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

"""Fused normalization layers with residual connections."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Weight,
    ops,
)

from ..layer import Module, Shardable


def layer_norm_fn(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None = None,
    residual: TensorValue | None = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
) -> TensorValue | tuple[TensorValue, TensorValue]:
    """Layer normalization function matching Mamba API.
    
    This function provides a functional interface for layer normalization
    with optional residual connections, matching the Mamba implementation.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py
    
    Args:
        x: Input tensor to normalize.
        weight: Weight tensor (gamma) for normalization.
        bias: Optional bias tensor (beta) for normalization.
        residual: Optional residual tensor to add to x before normalization.
        eps: Epsilon value for numerical stability.
        dropout_p: Dropout probability. If > 0.0, applies dropout to x.
        prenorm: If True, returns both (normalized_output, pre_normalized_input).
            The pre_normalized_input is x + residual (if residual provided).
        residual_in_fp32: If True and residual is provided, keeps the
            residual addition in float32 precision.
        is_rms_norm: If True, uses RMSNorm instead of LayerNorm.
            
    Returns:
        If prenorm=False: Normalized tensor.
        If prenorm=True: Tuple of (normalized_output, pre_normalized_input).
    """
    # Handle residual connection and normalization
    if residual is not None:
        # Validate shapes match
        if residual.shape != x.shape:
            raise ValueError(
                f"Residual shape {residual.shape} must match input shape {x.shape}"
            )
        
        # Handle residual_in_fp32
        if residual_in_fp32:
            x_fp32 = x.cast(DType.float32)
            residual_fp32 = residual.cast(DType.float32)
            x_input = x_fp32
            residual_input = residual_fp32
        else:
            # Ensure types match
            if residual.dtype != x.dtype:
                residual = residual.cast(x.dtype)
            x_input = x
            residual_input = residual
        
        # Use fused kernel if available (for RMSNorm with residual)
        if is_rms_norm:
            # Use fused RMSNorm kernel with residual
            weight_cast = weight.cast(x_input.dtype)
            if x_input.device:
                weight_cast = weight_cast.to(x_input.device)
            
            # Prepare constants
            eps_constant = ops.constant(eps, dtype=x_input.dtype, device=DeviceRef.CPU())
            weight_offset_constant = ops.constant(0.0, dtype=x_input.dtype, device=DeviceRef.CPU())
            dropout_p_constant = ops.constant(dropout_p, dtype=x_input.dtype, device=DeviceRef.CPU())
            seed = 0  # TODO: Make this configurable or generate from context
            seed_constant = ops.constant(seed, dtype=DType.uint64, device=DeviceRef.CPU())
            
            # Call fused kernel - returns (normalized_output, residual_output)
            results = ops.custom(
                "rms_norm_fused_residual",
                x_input.device,
                [
                    x_input,
                    residual_input,
                    weight_cast,
                    eps_constant,
                    weight_offset_constant,
                    dropout_p_constant,
                    seed_constant,
                ],
                [
                    TensorType(dtype=x_input.dtype, shape=x_input.shape, device=x_input.device),
                    TensorType(dtype=x_input.dtype, shape=x_input.shape, device=x_input.device),
                ],
                parameters={"multiply_before_cast": True},
            )
            
            normalized = results[0].tensor
            pre_normalized = results[1].tensor
            
            # Cast back to original dtype if we used fp32
            if residual_in_fp32:
                normalized = normalized.cast(x.dtype)
                pre_normalized = pre_normalized.cast(x.dtype)
        else:
            # For LayerNorm, add residual manually (no fused kernel yet)
            pre_normalized = x_input + residual_input
            
            # Cast back to original dtype if we used fp32
            if residual_in_fp32:
                pre_normalized = pre_normalized.cast(x.dtype)
            
            # Prepare bias (beta)
            if bias is None:
                # Create zero bias
                bias = ops.broadcast_to(
                    ops.constant(0.0, pre_normalized.dtype, device=pre_normalized.device or DeviceRef.CPU()),
                    shape=(pre_normalized.shape[-1],),
                )
            else:
                bias = bias.cast(pre_normalized.dtype)
                if pre_normalized.device:
                    bias = bias.to(pre_normalized.device)
            
            # Prepare weight (gamma)
            weight_cast = weight.cast(pre_normalized.dtype)
            if pre_normalized.device:
                weight_cast = weight_cast.to(pre_normalized.device)
            
            # Apply layer normalization
            normalized = ops.layer_norm(
                pre_normalized,
                gamma=weight_cast,
                beta=bias,
                epsilon=eps,
            )
    else:
        # No residual, use regular normalization
        pre_normalized = x
        
        if is_rms_norm:
            # Use regular RMSNorm kernel
            weight_cast = weight.cast(pre_normalized.dtype)
            if pre_normalized.device:
                weight_cast = weight_cast.to(pre_normalized.device)
            
            normalized = ops.custom(
                "rms_norm",
                pre_normalized.device,
                [
                    pre_normalized,
                    weight_cast,
                    ops.constant(eps, dtype=pre_normalized.dtype, device=DeviceRef.CPU()),
                    ops.constant(0.0, dtype=pre_normalized.dtype, device=DeviceRef.CPU()),
                ],
                [TensorType(dtype=pre_normalized.dtype, shape=pre_normalized.shape, device=pre_normalized.device)],
                parameters={"multiply_before_cast": True},
            )[0].tensor
        else:
            # Use LayerNorm (no residual)
            # Prepare bias (beta)
            if bias is None:
                # Create zero bias
                bias = ops.broadcast_to(
                    ops.constant(0.0, pre_normalized.dtype, device=pre_normalized.device or DeviceRef.CPU()),
                    shape=(pre_normalized.shape[-1],),
                )
            else:
                bias = bias.cast(pre_normalized.dtype)
                if pre_normalized.device:
                    bias = bias.to(pre_normalized.device)
            
            # Prepare weight (gamma)
            weight_cast = weight.cast(pre_normalized.dtype)
            if pre_normalized.device:
                weight_cast = weight_cast.to(pre_normalized.device)
            
            # Apply layer normalization
            normalized = ops.layer_norm(
                pre_normalized,
                gamma=weight_cast,
                beta=bias,
                epsilon=eps,
            )
    
    # Return based on prenorm mode
    if prenorm:
        return (normalized, pre_normalized)
    return normalized


def rms_norm_fn(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None = None,
    residual: TensorValue | None = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
) -> TensorValue | tuple[TensorValue, TensorValue]:
    """RMS normalization function matching Mamba API.
    
    This is a convenience wrapper around layer_norm_fn with is_rms_norm=True.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py
    
    Args:
        x: Input tensor to normalize.
        weight: Weight tensor (gamma) for normalization.
        bias: Optional bias tensor (beta) for normalization. Note: RMSNorm typically
            doesn't use bias, but this is included for API compatibility.
        residual: Optional residual tensor to add to x before normalization.
        eps: Epsilon value for numerical stability.
        dropout_p: Dropout probability. If > 0.0, applies dropout to x.
        prenorm: If True, returns both (normalized_output, pre_normalized_input).
        residual_in_fp32: If True and residual is provided, keeps the
            residual addition in float32 precision.
            
    Returns:
        If prenorm=False: Normalized tensor.
        If prenorm=True: Tuple of (normalized_output, pre_normalized_input).
    """
    return layer_norm_fn(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        eps=eps,
        dropout_p=dropout_p,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True,
    )


class RMSNorm(Module, Shardable):
    """RMSNorm with fused residual connection support.

    This class uses the `rms_norm_fused_residual` kernel which performs
    RMSNorm(x + residual) in a single fused operation for better performance.

    Args:
        dim: Size of last dimension of the expected input.
        dtype: Data type for weights.
        eps: Value added to denominator for numerical stability.
        weight_offset: Constant offset added to the learned weights at runtime.
            For Gemma-style RMSNorm, this should be set to 1.0.
        multiply_before_cast: True if we multiply the inputs by the learned
            weights before casting to the input type (Gemma3-style). False if we
            cast the inputs to the input type first, then multiply by the learned
            weights (Llama-style).
    """

    def __init__(
        self,
        dim: int,
        dtype: DType,
        eps: float = 1e-6,
        weight_offset: float = 0.0,
        multiply_before_cast: bool = True,
    ) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [dim], device=DeviceRef.CPU())
        self.dim = dim
        self.dtype = dtype
        self.eps = eps
        self.weight_offset = weight_offset
        self.multiply_before_cast = multiply_before_cast
        self._sharding_strategy: ShardingStrategy | None = None

    def __call__(
        self,
        x: TensorValue,
        residual: TensorValue | None = None,
        dropout_p: float = 0.0,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        """Apply RMSNorm with optional fused residual connection.

        Args:
            x: Input tensor to normalize.
            residual: Optional residual tensor to add to x before normalization.
                If provided, performs fused add + norm operation using the
                rms_norm_fused_residual kernel.
            dropout_p: Dropout probability. If > 0.0, applies dropout to x
                before normalization.
            prenorm: If True, returns both (normalized_output, pre_normalized_input).
                The pre_normalized_input is x + residual (if residual provided).
            residual_in_fp32: If True and residual is provided, keeps the
                residual addition in float32 precision.

        Returns:
            If prenorm=False: Normalized tensor.
            If prenorm=True: Tuple of (normalized_output, pre_normalized_input).
        """
        # Validate that weight dimension matches input's last dimension if
        # statically known.
        input_last_dim = x.shape[-1]
        weight_dim = self.weight.shape[0]

        if input_last_dim != weight_dim:
            raise ValueError(
                f"RMSNorm weight dimension ({weight_dim}) must match the input's "
                f"last dimension ({input_last_dim})"
            )

        # Prepare weight
        weight: TensorValue = self.weight.cast(x.dtype)
        if x.device:
            weight = weight.to(x.device)

        # If no residual, fall back to regular rms_norm
        if residual is None:
            # Use regular rms_norm kernel
            normalized = ops.custom(
                "rms_norm",
                x.device,
                [
                    x,
                    weight,
                    ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                    ops.constant(
                        self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
                    ),
                ],
                [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
                parameters={"multiply_before_cast": self.multiply_before_cast},
            )[0].tensor

            if prenorm:
                return (normalized, x)
            return normalized

        # Validate residual shape matches input shape
        if residual.shape != x.shape:
            raise ValueError(
                f"Residual shape {residual.shape} must match input shape {x.shape}"
            )

        # Handle residual_in_fp32: cast inputs to float32 for addition
        if residual_in_fp32:
            x_fp32 = x.cast(DType.float32)
            residual_fp32 = residual.cast(DType.float32)
            # The kernel will handle the addition, but we need to ensure types match
            # For now, we'll pass them as-is and let the kernel handle it
            # Note: The kernel currently doesn't support residual_in_fp32 directly,
            # so we'll do the cast manually if needed
            x_input = x_fp32
            residual_input = residual_fp32
        else:
            # Ensure types match for addition
            if residual.dtype != x.dtype:
                residual = residual.cast(x.dtype)
            x_input = x
            residual_input = residual

        # Prepare constants
        eps_constant = ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU())
        weight_offset_constant = ops.constant(
            self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
        )
        dropout_p_constant = ops.constant(
            dropout_p, dtype=x.dtype, device=DeviceRef.CPU()
        )
        seed = 0  # TODO: Make this configurable or generate from context
        seed_constant = ops.constant(seed, dtype=DType.uint64, device=DeviceRef.CPU())

        # Call fused kernel - returns (normalized_output, residual_output)
        results = ops.custom(
            "rms_norm_fused_residual",
            x.device,
            [
                x_input,
                residual_input,
                weight,
                eps_constant,
                weight_offset_constant,
                dropout_p_constant,
                seed_constant,
            ],
            [
                TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
                TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            ],
            parameters={"multiply_before_cast": self.multiply_before_cast},
        )

        normalized = results[0].tensor
        residual_output = results[1].tensor

        # Cast back to original dtype if we used fp32
        if residual_in_fp32:
            normalized = normalized.cast(x.dtype)
            residual_output = residual_output.cast(x.dtype)

        # Return based on prenorm mode
        if prenorm:
            return (normalized, residual_output)
        return normalized

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the RMSNorm sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the RMSNorm layer.

        Args:
            strategy: The sharding strategy to apply.
        """
        # RMSNorm always uses replicate strategy
        if not strategy.is_replicate:
            raise ValueError("RMSNorm only supports replicate strategy")

        self._sharding_strategy = strategy
        self.weight.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[RMSNorm]:
        """Creates sharded views of this RMSNorm across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded RMSNorm instances, one for each device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        # Get sharded weights
        weight_shards = self.weight.shard(devices)

        shards = []
        for weight_shard in weight_shards:
            # Create new RMSNorm instance with the same configuration
            sharded = RMSNorm(
                dim=self.dim,
                dtype=self.dtype,
                eps=self.eps,
                weight_offset=self.weight_offset,
                multiply_before_cast=self.multiply_before_cast,
            )

            # Assign the sharded weight
            sharded.weight = weight_shard

            shards.append(sharded)

        return shards

