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

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import overload

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.kernels import flash_attention_gpu

from ..comm import Allreduce
from ..layer import Module
from ..linear import Linear
from .mask_config import MHAMaskVariant


class MultiheadAttention(Module):
    """Multihead attention that handles both single and distributed computation."""

    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        scale: float | None = None,
        has_bias: bool = False,
        stacked_qkv: bool = False,
    ) -> None:
        """Initializes the attention layer.

        Args:
            num_attention_heads: The number of attention heads.
            hidden_size: The dimension of the hidden states (embed_dim).
            devices: Device(s) to place the weights and run the computation.
                If multiple devices provided, uses distributed computation.
            dtype: DType of the QKV and output projection weights.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            stacked_qkv: Whether to use a single stacked QKV weight matrix.
        """
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        if devices is not None and len(devices) == 0:
            raise ValueError("Devices cannot be empty")

        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.embed_dim = hidden_size
        self.devices = devices if devices is not None else [DeviceRef.CPU()]
        self.is_distributed = len(self.devices) > 1
        self.scale = (
            scale if scale is not None else 1.0 / math.sqrt(self.head_dim)
        )
        self.has_bias = has_bias
        self.stacked_qkv = stacked_qkv

        if stacked_qkv and has_bias:
            raise ValueError("Bias is not supported with stacked_qkv=True.")

        # Initialize weights
        self._init_weights(dtype)

        # Initialize distributed components if needed
        if self.is_distributed:
            self._init_distributed()

    def _init_weights(self, dtype: DType) -> None:
        """Initialize the attention weights."""
        if self.stacked_qkv:
            self.qkv_proj = Weight(
                name="qkv_proj.weight",
                dtype=dtype,
                shape=(3 * self.embed_dim, self.embed_dim),
                device=self.devices[0],
            )
        else:
            self.q_proj = Linear(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                has_bias=self.has_bias,
                dtype=dtype,
                device=self.devices[0],
            )
            self.k_proj = Linear(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                has_bias=self.has_bias,
                dtype=dtype,
                device=self.devices[0],
            )
            self.v_proj = Linear(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                has_bias=self.has_bias,
                dtype=dtype,
                device=self.devices[0],
            )

        self.o_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            has_bias=self.has_bias,
            dtype=dtype,
            device=self.devices[0],
        )

    def _init_distributed(self) -> None:
        """Initialize distributed components."""
        if len(self.devices) < 2:
            raise ValueError(
                f"Must provide at least 2 devices for distributed attention, got {self.devices}"
            )

        num_devices = len(self.devices)
        self.allreduce = Allreduce(num_devices)

        # Set up sharding strategies
        if self.stacked_qkv:
            self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
        else:
            self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )

        self.o_proj.sharding_strategy = ShardingStrategy.columnwise(num_devices)

        # Create sharded attention modules for each device
        self._create_device_modules()

    def _create_device_modules(self) -> None:
        """Create per-device attention modules for distributed computation."""
        self.device_modules = []
        sharded_num_heads = self.num_heads // len(self.devices)

        for n, device in enumerate(self.devices):
            # Create a module instance for this device
            module = self._create_device_module(
                num_attention_heads=sharded_num_heads,
                hidden_size=self.embed_dim,
                device=device,
                dtype=self.q_proj.weight.dtype
                if hasattr(self, "q_proj")
                else self.qkv_proj.dtype,
                scale=self.scale,
                has_bias=self.has_bias,
                stacked_qkv=self.stacked_qkv,
            )

            # Shard weights to this device
            if self.stacked_qkv:
                module.qkv_proj = self.qkv_proj.shard(n, device)
            else:
                module.q_proj = self.q_proj.shard(n, device)
                module.k_proj = self.k_proj.shard(n, device)
                module.v_proj = self.v_proj.shard(n, device)

            module.o_proj = self.o_proj.shard(n, device)
            self.device_modules.append(module)

    def _create_device_module(self, **kwargs) -> MultiheadAttention:
        """Create a single-device module instance.

        Override this method in subclasses to use a different module type
        for per-device computation.
        """
        # Create instance of the same class for consistency
        return type(self)(devices=[kwargs.pop("device")], **kwargs)

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj
        else:
            wq: TensorValue = self.q_proj.weight
            wk: TensorValue = self.k_proj.weight
            wv: TensorValue = self.v_proj.weight
            return ops.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        if self.stacked_qkv:
            raise ValueError(
                "Cannot access wqkv_bias when stacked_qkv=True. "
                "Bias is not supported with stacked QKV configuration."
            )

        if (
            self.q_proj.bias is None
            or self.k_proj.bias is None
            or self.v_proj.bias is None
        ):
            raise ValueError(
                "Projection bias is None, but has_bias=True was specified."
            )

        return ops.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def _compute_qkv(
        self, x: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Compute Q, K, V projections.

        Override this method to customize QKV computation (e.g., for quantization).
        """
        # Fused in-projection for Q, K, V
        wqkv = self.wqkv
        qkv = x @ wqkv.T

        if self.wqkv_bias is not None:
            qkv += self.wqkv_bias

        q, k, v = ops.split(
            qkv, [self.embed_dim, self.embed_dim, self.embed_dim], axis=-1
        )

        # Reshape for multihead attention
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))

        return q, k, v

    def _apply_attention(
        self, q: TensorValue, k: TensorValue, v: TensorValue, **kwargs
    ) -> TensorValue:
        """Apply attention mechanism to Q, K, V.

        Override this method to customize attention computation (e.g., add RoPE).
        """
        mask_variant = kwargs.get("mask_variant", MHAMaskVariant.NULL_MASK)

        attn_out = flash_attention_gpu(
            q, k, v, mask_variant=mask_variant, scale=self.scale
        )

        # Reshape back
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        attn_out = attn_out.reshape(
            (batch_size, seq_len, self.num_heads * self.head_dim)
        )

        return attn_out

    def _forward_single(self, x: TensorValue, **kwargs) -> TensorValue:
        """Single-device forward pass.

        Override this method to customize the single-device forward logic.
        """
        # Compute QKV
        q, k, v = self._compute_qkv(x)

        # Apply attention
        attn_out = self._apply_attention(q, k, v, **kwargs)

        # Output projection
        return self.o_proj(attn_out)

    def _forward_distributed(
        self, x: list[TensorValue], signal_buffers: list[BufferValue], **kwargs
    ) -> list[TensorValue]:
        """Distributed forward pass.

        Override this method to customize the distributed forward logic.
        """
        if len(x) != len(self.devices):
            raise ValueError(
                f"Expected {len(self.devices)} inputs, got {len(x)}"
            )

        if len(signal_buffers) != len(self.devices):
            raise ValueError(
                f"Expected {len(self.devices)} signal buffers, got {len(signal_buffers)}"
            )

        # Compute attention on each device
        outputs = [
            self.device_modules[i]._forward_single(x[i], **kwargs)
            for i in range(len(self.devices))
        ]

        # Allreduce across devices
        return self.allreduce(inputs=outputs, signal_buffers=signal_buffers)

    @overload
    def __call__(
        self, x: TensorValue, signal_buffers: None = None, **kwargs
    ) -> TensorValue: ...

    @overload
    def __call__(
        self, x: list[TensorValue], signal_buffers: list[BufferValue], **kwargs
    ) -> list[TensorValue]: ...

    def __call__(
        self,
        x: TensorValue | list[TensorValue],
        signal_buffers: list[BufferValue] | None = None,
        **kwargs,
    ) -> TensorValue | list[TensorValue]:
        """Forward pass for attention computation.

        Args:
            x: Input tensor(s). Single tensor for single-device, list for distributed.
            signal_buffers: Required for distributed mode.
            **kwargs: Additional arguments passed to attention computation.

        Returns:
            Output tensor(s). Type matches input type.
        """
        if self.is_distributed:
            if not isinstance(x, list):
                raise ValueError(
                    "Distributed attention requires list of input tensors, "
                    f"got {type(x)}"
                )
            if signal_buffers is None:
                raise ValueError(
                    "Distributed attention requires signal_buffers to be provided."
                )
            return self._forward_distributed(x, signal_buffers, **kwargs)
        else:
            if isinstance(x, list):
                if len(x) != 1:
                    raise ValueError(
                        f"Single-device attention expects one input tensor, got {len(x)}"
                    )
                x = x[0]
            return self._forward_single(x, **kwargs)
