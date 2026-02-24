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

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import Module
from max.nn.linear import Linear

from ..model_config import Gemma3ForConditionalGenerationConfig


class Gemma3VisionAttention(Module):
    """Standard self-attention for SigLIP vision encoder."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ) -> None:
        """Initialise the vision attention layers for projection and attention"""
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        vision_dtype = DType.bfloat16

        self.layer_idx = layer_idx
        self.device = device if device is not None else config.devices[0]
        self.head_dim = (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        self.num_heads = vision_config.num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=vision_config.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )
        self.k_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=vision_config.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )
        self.v_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=vision_config.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )
        self.out_proj = Linear(
            self.num_heads * self.head_dim,
            vision_config.hidden_size,
            has_bias=vision_config.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Process a tensor through the self attention layers and apply scaling"""
        batch_size, n_patches = x.shape[0], x.shape[1]

        # Project to Q, K, V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # Reshape to multi-head format [batch, n_patches, n_heads, head_dim]
        xq = ops.reshape(
            xq, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xk = ops.reshape(
            xk, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xv = ops.reshape(
            xv, [batch_size, n_patches, self.num_heads, self.head_dim]
        )

        output = flash_attention_gpu(
            xq,
            xk,
            xv,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scaling,
        )

        output = output.reshape([batch_size, n_patches, -1])

        return self.out_proj(output)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.q_proj.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_replicate:
            raise ValueError(
                "only replicate is currently supported for Gemma3VisionAttention"
            )

        self.q_proj.sharding_strategy = strategy
        self.k_proj.sharding_strategy = strategy
        self.v_proj.sharding_strategy = strategy
        self.out_proj.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3VisionAttention]:
        assert self.sharding_strategy

        q_proj_shards = self.q_proj.shard(devices)
        k_proj_shards = self.k_proj.shard(devices)
        v_proj_shards = self.v_proj.shard(devices)
        out_proj_shards = self.out_proj.shard(devices)

        shards = []
        for device, q_shard, k_shard, v_shard, out_shard in zip(
            devices,
            q_proj_shards,
            k_proj_shards,
            v_proj_shards,
            out_proj_shards,
            strict=True,
        ):
            sharded = Gemma3VisionAttention(self.config, self.layer_idx, device)

            sharded.q_proj = q_shard
            sharded.k_proj = k_shard
            sharded.v_proj = v_shard
            sharded.out_proj = out_shard

            shards.append(sharded)

        return shards
