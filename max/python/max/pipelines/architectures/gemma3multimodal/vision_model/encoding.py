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

from collections.abc import Iterable, Sequence

from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue
from max.nn import (
    LayerList,
    LayerNorm,
    Module,
)
from max.nn.layer import Shardable

from .attention import Gemma3VisionAttention
from ..model_config import Gemma3ForConditionalGenerationConfig
from .projection import Gemma3VisionMLP


# ✅ based on HF and MLX-VLM
class Gemma3VisionEncoderLayer(Module):
    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ):
        self.config = config
        vision_config = config.vision_config
        self.device = device if device is not None else config.devices[0]
        self.embed_dim = vision_config.hidden_size
        self.layer_idx = layer_idx

        # Pre-attention layer norm
        self.layer_norm1 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=self.device,
            dtype=config.dtype,
        )

        # Self-attention
        self.self_attn = Gemma3VisionAttention(
            config=config,
            layer_idx=layer_idx,
        )

        # MLP (Feed-Forward Network) - simple GELUTanh/fc1/fc2 style
        self.mlp = Gemma3VisionMLP(config, device=self.device)

        # post-attention layer norm
        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=self.device,
            dtype=config.dtype,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.self_attn.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.layer_norm1.weight.sharding_strategy = strategy
        if self.layer_norm1.bias is not None:
            self.layer_norm1.bias.sharding_strategy = strategy

        self.self_attn.sharding_strategy = strategy
        self.mlp.sharding_strategy = strategy

        self.layer_norm2.weight.sharding_strategy = strategy
        if self.layer_norm2.bias is not None:
            self.layer_norm2.bias.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3VisionEncoderLayer]:
        assert self.sharding_strategy

        norm1_weight_shards = self.layer_norm1.weight.shard(devices)
        norm1_bias_shards = (
            self.layer_norm1.bias.shard(devices)
            if self.layer_norm1.bias is not None
            else [None] * len(list(devices))
        )
        norm2_weight_shards = self.layer_norm2.weight.shard(devices)
        norm2_bias_shards = (
            self.layer_norm2.bias.shard(devices)
            if self.layer_norm2.bias is not None
            else [None] * len(list(devices))
        )
        attn_shards = self.self_attn.shard(devices)
        mlp_shards = self.mlp.shard(devices)

        shards = []
        for (
            device,
            norm1_w_shard,
            norm1_b_shard,
            norm2_w_shard,
            norm2_b_shard,
            attn_shard,
            mlp_shard,
        ) in zip(
            devices,
            norm1_weight_shards,
            norm1_bias_shards,
            norm2_weight_shards,
            norm2_bias_shards,
            attn_shards,
            mlp_shards,
            strict=True,
        ):
            # Create the new sharded encoder layer.
            sharded = Gemma3VisionEncoderLayer(
                self.config, self.layer_idx, device
            )

            # Assign the sharded components.
            sharded.layer_norm1.weight = norm1_w_shard
            if norm1_b_shard is not None:
                sharded.layer_norm1.bias = norm1_b_shard
            sharded.layer_norm2.weight = norm2_w_shard
            if norm2_b_shard is not None:
                sharded.layer_norm2.bias = norm2_b_shard
            sharded.self_attn = attn_shard
            sharded.mlp = mlp_shard

            shards.append(sharded)

        return shards
    
    def __call__(
        self,
        hidden_states: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue]
    ) -> list[TensorValue]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ✅ based on HF and MLX-VLM
class Gemma3VisionEncoder(Module):
    """SigLIP vision encoder with 27 transformer layers."""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        super().__init__()
        self.config = config
        self.devices = config.devices

        # Create encoder layers
        encoder_layers = [
            Gemma3VisionEncoderLayer(config, layer_idx)
            for layer_idx in range(config.vision_config.num_hidden_layers)
        ]

        # Set sharding strategy for each layer
        for layer in encoder_layers:
            layer.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )

        self.layers = LayerList(encoder_layers)

        # Create per-device instances of each layer
        self.layers_per_device = [
            [layer.shard(config.devices)[i] for layer in encoder_layers]
            for i in range(len(config.devices))
        ]

    def __call__(
        self,
        hidden_states: TensorValue | Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> TensorValue | Sequence[TensorValue]:
        # Handle both single tensor and list of tensors (multi-device case)
        if isinstance(hidden_states, list):
            # Multi-device: process each device's tensor through its device-specific layers
            # Each device processes its data independently with replicated weights
            outputs = []
            for device_idx, state in enumerate(hidden_states):
                device_output = state
                for layer in self.layers_per_device[device_idx]:
                    device_output = layer(device_output, signal_buffers)
                outputs.append(device_output)
            return outputs
        else:
            # Single device case - use first device's layers
            for layer in self.layers_per_device[0]:
                hidden_states = layer(hidden_states, signal_buffers)
            return hidden_states
