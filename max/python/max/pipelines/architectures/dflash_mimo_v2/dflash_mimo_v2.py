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
"""The MiMo-V2 DFlash drafter: a non-causal block over windowed target context.

The block is ``[anchor, mask x (block_size - 1)]``. Each block query attends
to the context keys its window reaches and to every block key, with a learned
sink per head. The layers are pre-norm, with per-head ``q_norm`` and
``k_norm`` and NeoX RoPE on the leading ``rotary_dim`` of each head.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.weights import WeightData
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged
from max.nn.kv_cache import PagedCacheValues
from max.nn.linear import MLP, Linear
from max.nn.transformer.transformer import forward_sharded_layers
from typing_extensions import Self

from .context_writer import (
    DFlashContextAttention,
    DFlashContextLayer,
    DFlashContextWriter,
    _heads_per_shard,
    _replicated_norm,
    _rms_norm,
)
from .model_config import DFlashMiMoV2Config


class DFlashMiMoV2Attention(DFlashContextAttention):
    """Block self-attention over the context and block keys, with sinks."""

    def __init__(
        self,
        config: DFlashMiMoV2Config,
        *,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> None:
        super().__init__(config, shard_index=shard_index, num_shards=num_shards)
        self.num_heads = _heads_per_shard(
            config.num_attention_heads, num_shards, "query"
        )
        q_dim = self.num_heads * config.head_dim
        self.q_proj = Linear(
            config.hidden_size, q_dim, config.dtype, self.device
        )
        self.o_proj = Linear(
            q_dim, config.hidden_size, config.dtype, self.device
        )
        self.q_norm = _rms_norm(config.head_dim, config)
        self.attention_sink_bias = Weight(
            "attention_sink_bias", config.dtype, [self.num_heads], self.device
        )

    def _shard_by_head(self, num_devices: int) -> None:
        super()._shard_by_head(num_devices)
        self.q_proj.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.o_proj.sharding_strategy = ShardingStrategy.columnwise(num_devices)
        self.q_norm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.attention_sink_bias.sharding_strategy = ShardingStrategy.rowwise(
            num_devices
        )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Self]:
        """Splits the heads over ``devices``, one shard per device."""
        devices = list(devices)
        shards = super().shard(devices)
        q_proj = self.q_proj.shard(devices)
        o_proj = self.o_proj.shard(devices)
        q_norm = self.q_norm.shard(devices)
        sinks = self.attention_sink_bias.shard(devices)
        narrowed: list[Self] = []
        for i, shard in enumerate(shards):
            assert isinstance(shard, DFlashMiMoV2Attention)
            shard.q_proj, shard.o_proj, shard.q_norm = (
                q_proj[i],
                o_proj[i],
                q_norm[i],
            )
            shard.attention_sink_bias = sinks[i]
            narrowed.append(shard)
        return narrowed

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        config = self.config
        head_dim = config.head_dim
        q = self.q_norm(self.q_proj(x).reshape((-1, head_dim))).reshape(
            (-1, self.num_heads, head_dim)
        )
        q = self.store_kv(
            layer_idx, x, kv_collection, freqs_cis, input_row_offsets, q=q
        )
        attn = flash_attention_ragged(
            config.kv_params,
            input=q,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK,
            scale=head_dim**-0.5,
            local_window_size=config.sliding_window,
            sink_weights=self.attention_sink_bias,
        )
        return self.o_proj(attn.reshape((-1, self.num_heads * head_dim)))


class DFlashMiMoV2DecoderLayer(DFlashContextLayer):
    """A pre-norm drafter layer: sink attention, then a SiLU MLP.

    Calling it writes context K/V, as for any context layer; the block runs
    through :meth:`forward_block`.
    """

    attention_cls = DFlashMiMoV2Attention

    def __init__(self, config: DFlashMiMoV2Config) -> None:
        super().__init__(config)
        devices = config.devices
        self.input_layernorm, self.input_layernorm_shards = _replicated_norm(
            config.hidden_size, config
        )
        self.post_attention_layernorm, self.post_attention_layernorm_shards = (
            _replicated_norm(config.hidden_size, config)
        )
        self.mlp = MLP(
            config.dtype,
            None,
            config.hidden_size,
            config.intermediate_size,
            devices,
        )
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = list(self.mlp.shard(devices))

    def forward_block(
        self,
        layer_idx: TensorValue,
        h: list[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue] | None,
    ) -> list[TensorValue]:
        def reduce(xs: list[TensorValue]) -> list[TensorValue]:
            if len(xs) == 1:
                return xs
            assert signal_buffers is not None, "TP drafter needs signal buffers"
            return ops.allreduce.sum(xs, signal_buffers)

        normed = forward_sharded_layers(self.input_layernorm_shards, h)
        attn = reduce(
            [
                attn(layer_idx, x, kv, freqs, offsets)
                for attn, x, kv, freqs, offsets in zip(
                    self.self_attn_shards,
                    normed,
                    kv_collections,
                    freqs_cis,
                    input_row_offsets,
                    strict=True,
                )
            ]
        )
        h = [x + a for x, a in zip(h, attn, strict=True)]
        normed = forward_sharded_layers(self.post_attention_layernorm_shards, h)
        mlp = reduce(forward_sharded_layers(self.mlp_shards, normed))
        return [x + m for x, m in zip(h, mlp, strict=True)]


class DFlashMiMoV2(DFlashContextWriter):
    """The drafter: the context writer plus the block forward.

    The target's ``embed_tokens`` embeds the anchor and its ``lm_head`` reads
    the drafts; the mask slots take the checkpoint's trained
    ``mask_embedding``, not the target's embedding of ``mask_token_id``, which
    is an untrained padding row.
    """

    def __init__(
        self,
        config: DFlashMiMoV2Config,
        mask_embedding: WeightData | None = None,
    ) -> None:
        """Builds the drafter.

        Args:
            config: The drafter configuration.
            mask_embedding: The trained mask embedding, to bake into the graph
                as a constant, so an exported graph carries it rather than
                naming a weight the engine must supply. ``None`` declares the
                ``mask_embedding`` weight instead.
        """
        super().__init__(
            config,
            layers=[
                DFlashMiMoV2DecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ],
        )
        devices = config.devices
        self.norm, self.norm_shards = _replicated_norm(
            config.hidden_size, config
        )
        self._mask_values: npt.NDArray[np.float32] | None = None
        if mask_embedding is not None:
            data = mask_embedding.data
            assert isinstance(data, Buffer) and data.dtype == DType.bfloat16
            bits = np.from_dlpack(data.view(DType.uint16)).astype(np.uint32)
            # Widened exactly; the graph casts it back to BFloat16.
            self._mask_values = (bits << 16).view(np.float32)
        else:
            self.mask_embedding = Weight(
                "mask_embedding",
                config.dtype,
                [config.hidden_size],
                DeviceRef.CPU(),
            )
            self.mask_embedding.sharding_strategy = ShardingStrategy.replicate(
                len(devices)
            )
            self.mask_embedding_shards = self.mask_embedding.shard(devices)

    def block_embeddings(
        self, anchor_embeds: Sequence[TensorValue]
    ) -> list[TensorValue]:
        """Returns the block rows, ``[batch * block_size, hidden]`` per device.

        Args:
            anchor_embeds: Per device, the ``[batch, hidden]`` target
                embeddings of each request's anchor token.
        """
        hidden = self.config.hidden_size
        masks_per_device = (
            self.mask_embedding_shards
            if self._mask_values is None
            else [
                ops.constant(self._mask_values, DType.float32, a.device)
                for a in anchor_embeds
            ]
        )
        out = []
        for anchor, mask in zip(anchor_embeds, masks_per_device, strict=True):
            masks = ops.broadcast_to(
                ops.cast(mask, anchor.dtype).reshape((1, 1, hidden)),
                (anchor.shape[0], self.config.block_size - 1, hidden),
            )
            block = ops.concat([ops.unsqueeze(anchor, 1), masks], axis=1)
            out.append(block.reshape((-1, hidden)))
        return out

    def forward_block(
        self,
        block_embeds: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue] | None = None,
    ) -> list[TensorValue]:
        """Runs the block, whose K/V lands at the caches' lengths.

        Args:
            block_embeds: Per device, :meth:`block_embeddings`'s rows.
            kv_collections: Per device, the drafter cache, whose lengths are
                the anchors' positions and whose dispatch metadata is sized
                for the block.
            input_row_offsets: Per device, the block's ragged offsets.
            signal_buffers: The allreduce buffers; needed on more than one
                device.

        Returns:
            Per device, the final-normed ``[batch * block_size, hidden]``
            hidden states.
        """
        freqs_cis = self._freqs_cis()
        h = list(block_embeds)
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, DFlashMiMoV2DecoderLayer)
            h = layer.forward_block(
                ops.constant(layer_idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collections,
                freqs_cis,
                input_row_offsets,
                signal_buffers,
            )
        return forward_sharded_layers(self.norm_shards, h)
