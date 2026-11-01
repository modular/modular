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
"""GLM-5.3-Flash's MTP draft layer.

The checkpoint's layer ``num_hidden_layers`` is the target's own sparse-MLA
block and MoE with a plain residual in place of the hyper-connections, so it is
built from this package's sublayers rather than from DeepSeek-V3.2's NextN. The
draft owns the last slot of the target's latent, indexer and tail ring caches,
which the target's config already declares when speculating.
"""

from __future__ import annotations

from max.graph import BufferValue, ShardingStrategy, TensorValue, ops
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.embedding import VocabParallelEmbedding
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import ColumnParallelLinear, Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import RotaryEmbedding
from max.nn.legacy.transformer.transformer import forward_sharded_layers

from .glm5_next import Glm5Next
from .layers.decoder import Glm5NextDecoderLayer
from .layers.mlp import (
    Glm5NextMlpSublayer,
    Glm5NextMlpSublayerInputs,
    Glm5NextMoE,
)
from .layers.sparse_mla import (
    Glm5NextSparseMLASublayer,
    SparseMLASublayerInputs,
    nope_pad_rotary,
)
from .model_config import Glm5NextConfig

__all__ = ["Glm5NextMTP"]


def _replicated_norm(config: Glm5NextConfig) -> RMSNorm:
    norm = RMSNorm(
        config.hidden_size,
        config.compute_dtype,
        config.rms_norm_eps,
        multiply_before_cast=False,
    )
    norm.sharding_strategy = ShardingStrategy.replicate(len(config.devices))
    return norm


class Glm5NextMTP(Module):
    """The MTP draft: ``eh_proj(enorm(embed(t)) ++ hnorm(h))``, one decoder layer.

    Returns the hidden state after ``shared_head.norm``, which both the
    target's ``lm_head`` and the next draft step read. The draft ships no
    embedding or head of its own: spec decoding shares the target's onto it.

    Args:
        config: The target's config, with the MTP layer in its precision map.

    Raises:
        NotImplementedError: If the checkpoint does not share the draft's
            top-k selection across steps, the only mode implemented.
    """

    embed_tokens: VocabParallelEmbedding
    lm_head: ColumnParallelLinear

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        if not config.index_share_for_mtp_iteration:
            raise NotImplementedError(
                "GLM-5.3-Flash MTP reuses its first step's top-k selection at"
                " every later step; index_share_for_mtp_iteration=False is not"
                " implemented"
            )
        devices = config.devices
        layer_idx = config.num_hidden_layers
        self.cache_layer_idx = len(config.sparse_attention_layers)
        """The draft's slot in the latent, indexer and tail ring caches, one
        past the target's sparse layers."""

        self.enorm = _replicated_norm(config)
        self.hnorm = _replicated_norm(config)
        self.shared_head_norm = _replicated_norm(config)
        self.enorm_shards = self.enorm.shard(devices)
        self.hnorm_shards = self.hnorm.shard(devices)
        self.shared_head_norm_shards = self.shared_head_norm.shard(devices)

        self.eh_proj = Linear(
            in_dim=2 * config.hidden_size,
            out_dim=config.hidden_size,
            dtype=config.compute_dtype,
            device=devices[0],
            has_bias=False,
        )
        self.eh_proj.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.eh_proj_shards = self.eh_proj.shard(devices)

        # Built before its target, so it dispatches its experts through a
        # manager of its own, over the target's expert-parallel buffers.
        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)
        self.decoder_layer = Glm5NextDecoderLayer(
            config,
            layer_idx,
            self_attn=Glm5NextSparseMLASublayer(
                config, layer_idx, rope=nope_pad_rotary(config)
            ),
            mlp=Glm5Next.build_mlp(config, layer_idx, self.ep_manager),
            mhc=False,
        )

    def share_rope(self, rope: RotaryEmbedding) -> None:
        """Reads the target's padding rotary table instead of its own.

        The table's leading dim is ``max_seq_len``, so one copy is staged
        rather than two.
        """
        self.self_attn.rope = rope

    @property
    def self_attn(self) -> Glm5NextSparseMLASublayer:
        """The draft layer's sparse-MLA sublayer."""
        sublayer = self.decoder_layer.self_attn
        assert isinstance(sublayer, Glm5NextSparseMLASublayer)
        return sublayer

    @property
    def mlp(self) -> Glm5NextMlpSublayer:
        """The draft layer's MoE sublayer."""
        sublayer = self.decoder_layer.mlp
        assert isinstance(sublayer, Glm5NextMlpSublayer)
        return sublayer

    @property
    def selection_width(self) -> int:
        """Width of the token selection the draft's later steps reuse."""
        return self.self_attn.indexer.selection_width

    def __call__(
        self,
        tokens: TensorValue,
        hidden: list[TensorValue],
        signal_buffers: list[BufferValue],
        attn_inputs: SparseMLASublayerInputs,
        mlp_inputs: Glm5NextMlpSublayerInputs,
    ) -> list[TensorValue]:
        """Runs one draft step.

        Args:
            tokens: ``[rows]`` tokens, one per hidden state.
            hidden: ``[rows, hidden_size]`` per device, the target's
                normalized hidden state or the previous step's output.
            signal_buffers: One per device.
            attn_inputs: The sparse-MLA sublayer's bundle.
            mlp_inputs: The feed-forward sublayer's bundle.

        Returns:
            ``[rows, hidden_size]`` per device, after ``shared_head.norm``.
        """
        self._place_shared_expert_weights()
        embeds = self.embed_tokens(tokens, signal_buffers)
        xs = [
            ops.concat(
                [
                    self.enorm_shards[i](embed),
                    # The embedding and the carried hidden state come off
                    # different row dims that are equal at runtime.
                    self.hnorm_shards[i](ops.rebind(h, embed.shape)),
                ],
                axis=-1,
            )
            for i, (embed, h) in enumerate(zip(embeds, hidden, strict=True))
        ]
        xs = forward_sharded_layers(self.eh_proj_shards, xs)
        xs = self.decoder_layer(xs, attn_inputs, mlp_inputs)
        return forward_sharded_layers(self.shared_head_norm_shards, xs)

    def _place_shared_expert_weights(self) -> None:
        """Builds each shared expert's per-device weights in this block.

        A sharded weight's value is built at its first use and cached. The
        shared expert runs inside a side-stream region, so a value first
        built there would be out of reach of the next draft step's region.
        """
        for shard in self.mlp.shards:
            assert isinstance(shard, Glm5NextMoE)
            for weight in shard.shared_experts.raw_state_dict().values():
                TensorValue(weight)
