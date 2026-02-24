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
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
)
from max.nn.layer import LayerList, Module
from max.nn.norm import LayerNorm

from ..model_config import Gemma3ForConditionalGenerationConfig
from .attention import Gemma3VisionAttention
from .projection import Gemma3VisionMLP


class Gemma3VisionEncoderLayer(Module):
    """An individual layer of encoding within a stack of encoding layers"""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ):
        """prepare the two normalisation layers, the self attention, and the
        multi-layer perceptrion"""
        self.config = config
        vision_config = config.vision_config
        vision_dtype = DType.bfloat16

        self.device = device if device is not None else config.devices[0]
        self.embed_dim = vision_config.hidden_size
        self.layer_idx = layer_idx

        self.layer_norm1 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            devices=[self.device],
            dtype=vision_dtype,
        )

        self.self_attn = Gemma3VisionAttention(
            config=config,
            layer_idx=layer_idx,
        )

        self.mlp = Gemma3VisionMLP(config, device=self.device)

        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            devices=[self.device],
            dtype=vision_dtype,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        signal_buffers: BufferValue,
    ) -> TensorValue:
        """process the input hidden states through each of the sub-layers"""
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
        norm1_bias_shards: Sequence[Weight | None] = (
            self.layer_norm1.bias.shard(devices)
            if self.layer_norm1.bias is not None
            else cast(Sequence[Weight | None], [None] * len(list(devices)))
        )
        norm2_weight_shards = self.layer_norm2.weight.shard(devices)
        norm2_bias_shards: Sequence[Weight | None] = (
            self.layer_norm2.bias.shard(devices)
            if self.layer_norm2.bias is not None
            else cast(Sequence[Weight | None], [None] * len(list(devices)))
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


class Gemma3VisionEncoder(Module):
    """Wrapper class for a stack of vision encoder layers"""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        """Intialise the stack of encoder layers based on config, and prepare
        sharding strategy"""
        super().__init__()
        self.config = config
        self.devices = config.devices

        encoder_layers = [
            Gemma3VisionEncoderLayer(config, layer_idx)
            for layer_idx in range(config.vision_config.num_hidden_layers)
        ]

        for layer in encoder_layers:
            layer.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )

        self.layers = LayerList(encoder_layers)

        self.layers_per_device = [
            [layer.shard(config.devices)[i] for layer in encoder_layers]
            for i in range(len(config.devices))
        ]

    def __call__(
        self,
        hidden_states: TensorValue | Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> TensorValue | Sequence[TensorValue]:
        """Process hidden states through the stack of encoder layers,
        applying multi-device functionality if required"""
        # if hidden_states is a list, we are sharding across devices.  each device has a replication of the weights
        if isinstance(hidden_states, Sequence):
            outputs = []
            for device_idx, state in enumerate(hidden_states):
                device_output = state
                for layer in self.layers_per_device[device_idx]:
                    device_output = layer(
                        device_output, signal_buffers[device_idx]
                    )
                outputs.append(device_output)
            return outputs
        else:
            # Single device case - use first device's layers
            for layer in self.layers_per_device[0]:
                hidden_states = layer(hidden_states, signal_buffers[0])
            return hidden_states
