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

from collections.abc import Iterable

from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn import Linear
from max.nn.layer import Module

from ..model_config import Gemma3ForConditionalGenerationConfig


# ⚠️ from Huggingface Transformers not sure if complete
class Gemma3VisionAttention(Module):
    """Standard self-attention for SigLIP vision encoder.

    Unlike Pixtral, SigLIP uses:
    - Standard self-attention (no rotary embeddings)
    - No attention masking
    - Absolute position embeddings (added in embedding layer)
    """

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        vision_config = config.vision_config

        self.layer_type = (
            config.layer_types[layer_idx]
            if hasattr(config, "layer_types")
            else None
        )
        self.config = config
        self.layer_idx = layer_idx
        self.device = device if device is not None else config.devices[0]
        # Vision encoder uses its own head_dim, not the text model's
        self.head_dim = (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        self.num_heads = vision_config.num_attention_heads
        self.attention_dropout = vision_config.attention_dropout
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=vision_config.attention_bias,
            dtype=config.dtype,
            device=self.device,
        )
        self.k_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=vision_config.attention_bias,
            dtype=config.dtype,
            device=self.device,
        )
        self.v_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=vision_config.attention_bias,
            dtype=config.dtype,
            device=self.device,
        )
        self.out_proj = Linear(
            self.num_heads * self.head_dim,
            vision_config.hidden_size,
            has_bias=vision_config.attention_bias,
            dtype=config.dtype,
            device=self.device,
        )

        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = (
            config.sliding_window
            if self.layer_type == "sliding_attention"
            else None
        )
        self.is_sliding = self.layer_type == "sliding_attention"

    @property
    def sharding_strategy(self) -> ShardingStrategy:
        return self.q_proj.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_replicate:
            raise ValueError(
                "only replicate is supported for Gemma3VisionAttention, "
                "currently"
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

    def __call__(self, x: TensorValue) -> TensorValue:
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

        # Transpose to [batch, n_heads, n_patches, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scores = (xq @ ops.transpose(xk, 2, 3)) * self.scaling
        scores = ops.softmax(scores)

        # Apply attention to values
        output = scores @ xv  # [batch, n_heads, n_patches, head_dim]

        # Transpose back and reshape
        output = output.transpose(1, 2).reshape([batch_size, n_patches, -1])

        # Output projection
        return self.out_proj(output)
