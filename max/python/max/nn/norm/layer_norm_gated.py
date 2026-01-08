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

"""Gated LayerNorm and RMSNorm implementations matching Mamba API.

Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py
"""

from __future__ import annotations

from collections.abc import Sequence

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


def layernorm_fn(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None = None,
    z: TensorValue | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
) -> TensorValue:
    """Layer normalization function with optional gating, matching Mamba API.
    
    If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z)).
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py
    
    Args:
        x: Input tensor to normalize.
        weight: Weight tensor (gamma) for normalization.
        bias: Optional bias tensor (beta) for normalization.
        z: Optional gating tensor. If provided, applies SiLU gating.
        eps: Epsilon value for numerical stability.
        group_size: Group size for group normalization (not yet supported).
        norm_before_gate: If True, applies gating after normalization.
            If False, applies gating before normalization.
        is_rms_norm: If True, uses RMSNorm instead of LayerNorm.
            
    Returns:
        Normalized tensor with optional gating applied.
    """
    if group_size is not None:
        raise NotImplementedError("group_size is not yet supported")
    
    # Validate shapes
    if weight.shape != (x.shape[-1],):
        raise ValueError(
            f"Weight shape {weight.shape} must match last dimension of input {x.shape[-1]}"
        )
    
    if bias is not None and bias.shape != (x.shape[-1],):
        raise ValueError(
            f"Bias shape {bias.shape} must match last dimension of input {x.shape[-1]}"
        )
    
    if z is not None and z.shape != x.shape:
        raise ValueError(
            f"Z tensor shape {z.shape} must match input shape {x.shape}"
        )
    
    # Prepare tensors
    weight_cast = weight.cast(x.dtype)
    if x.device:
        weight_cast = weight_cast.to(x.device)
    
    # Track if z was actually provided (before creating dummy)
    has_z_provided = z is not None
    
    # Create dummy z tensor if not provided (required by kernel)
    if z is None:
        z = ops.broadcast_to(
            ops.constant(0.0, x.dtype, device=x.device or DeviceRef.CPU()),
            shape=x.shape,
        )
    else:
        z = z.cast(x.dtype)
        if x.device:
            z = z.to(x.device)
    
    # Track if bias was actually provided (before creating dummy)
    has_bias_provided = bias is not None
    
    # Create dummy beta tensor if not provided (required by kernel)
    if bias is None:
        beta = ops.broadcast_to(
            ops.constant(0.0, x.dtype, device=x.device or DeviceRef.CPU()),
            shape=(x.shape[-1],),
        )
    else:
        beta = bias.cast(x.dtype)
        if x.device:
            beta = beta.to(x.device)
    
    # Prepare epsilon constant
    eps_constant = ops.constant(eps, dtype=x.dtype, device=DeviceRef.CPU())
    
    # Call gated layernorm kernel
    result = ops.custom(
        "layer_norm_gated",
        x.device,
        [
            x,
            z,
            weight_cast,
            beta,
            eps_constant,
        ],
        [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        parameters={
            "has_z": has_z_provided,
            "has_bias": has_bias_provided,
            "is_rms_norm": is_rms_norm,
            "norm_before_gate": norm_before_gate,
        },
    )
    
    return result[0].tensor


def rmsnorm_fn(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None = None,
    z: TensorValue | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
) -> TensorValue:
    """RMS normalization function with optional gating, matching Mamba API.
    
    If z is not None, we do rms_norm(x) * silu(z) if norm_before_gate, else rms_norm(x * silu(z)).
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py
    
    Args:
        x: Input tensor to normalize.
        weight: Weight tensor (gamma) for normalization.
        bias: Optional bias tensor (beta) for normalization. Note: RMSNorm typically
            doesn't use bias, but this is included for API compatibility.
        z: Optional gating tensor. If provided, applies SiLU gating.
        eps: Epsilon value for numerical stability.
        group_size: Group size for group normalization (not yet supported).
        norm_before_gate: If True, applies gating after normalization.
            If False, applies gating before normalization.
            
    Returns:
        Normalized tensor with optional gating applied.
    """
    return layernorm_fn(
        x,
        weight,
        bias=bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )


class LayerNorm(Module, Shardable):
    """Layer normalization module with optional gating, matching Mamba API.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize LayerNorm module.
        
        Args:
            hidden_size: Size of the hidden dimension.
            eps: Epsilon value for numerical stability.
            group_size: Group size for group normalization (not yet supported).
            norm_before_gate: If True, applies gating after normalization.
                If False, applies gating before normalization.
            devices: Device(s) to place weights on. If None, uses CPU.
            dtype: Data type for weights. If None, uses float32.
        """
        super().__init__()
        if group_size is not None:
            raise NotImplementedError("group_size is not yet supported")
        
        self.hidden_size = hidden_size
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        
        # Determine device and dtype
        if devices is None:
            devices = [DeviceRef.CPU()]
        if dtype is None:
            dtype = DType.float32
        
        self.devices = devices
        self.dtype = dtype
        
        # Create weights
        self.weight = Weight("weight", dtype, (hidden_size,), device=devices[0])
        self.bias = Weight("bias", dtype, (hidden_size,), device=devices[0])
        
        # Initialize weights
        self.reset_parameters()
        
        self._sharding_strategy: ShardingStrategy | None = None

    def reset_parameters(self) -> None:
        """Reset parameters to default values."""
        # Initialize weight to ones and bias to zeros
        # Note: This is a placeholder - actual initialization should be done
        # through the Weight system's initialization mechanism
        pass

    def __call__(self, x: TensorValue, z: TensorValue | None = None) -> TensorValue:
        """Forward pass.
        
        If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z)).
        
        Args:
            x: Input tensor.
            z: Optional gating tensor.
            
        Returns:
            Normalized tensor with optional gating applied.
        """
        return layernorm_fn(
            x,
            self.weight,
            bias=self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=False,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the LayerNorm sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the LayerNorm sharding strategy."""
        self._sharding_strategy = strategy


class RMSNorm(Module, Shardable):
    """RMS normalization module with optional gating, matching Mamba API.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize RMSNorm module.
        
        Args:
            hidden_size: Size of the hidden dimension.
            eps: Epsilon value for numerical stability.
            group_size: Group size for group normalization (not yet supported).
            norm_before_gate: If True, applies gating after normalization.
                If False, applies gating before normalization.
            devices: Device(s) to place weights on. If None, uses CPU.
            dtype: Data type for weights. If None, uses float32.
        """
        super().__init__()
        if group_size is not None:
            raise NotImplementedError("group_size is not yet supported")
        
        self.hidden_size = hidden_size
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        
        # Determine device and dtype
        if devices is None:
            devices = [DeviceRef.CPU()]
        if dtype is None:
            dtype = DType.float32
        
        self.devices = devices
        self.dtype = dtype
        
        # Create weight (RMSNorm doesn't use bias)
        self.weight = Weight("weight", dtype, (hidden_size,), device=devices[0])
        self.register_parameter("bias", None)
        
        # Initialize weights
        self.reset_parameters()
        
        self._sharding_strategy: ShardingStrategy | None = None

    def reset_parameters(self) -> None:
        """Reset parameters to default values."""
        # Initialize weight to ones
        # Note: This is a placeholder - actual initialization should be done
        # through the Weight system's initialization mechanism
        pass

    def __call__(self, x: TensorValue, z: TensorValue | None = None) -> TensorValue:
        """Forward pass.
        
        If z is not None, we do rms_norm(x) * silu(z) if norm_before_gate, else rms_norm(x * silu(z)).
        
        Args:
            x: Input tensor.
            z: Optional gating tensor.
            
        Returns:
            Normalized tensor with optional gating applied.
        """
        return rmsnorm_fn(
            x,
            self.weight,
            bias=None,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the RMSNorm sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the RMSNorm sharding strategy."""
        self._sharding_strategy = strategy

