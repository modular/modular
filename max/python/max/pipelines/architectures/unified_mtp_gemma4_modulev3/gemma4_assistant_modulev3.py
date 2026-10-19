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

"""The Gemma4 Assistant (MTP draft) model, on the ModuleV3 API.

A ModuleV3 port of
``max.pipelines.architectures.gemma4_assistant.gemma4_assistant``: a small
stack of Q-only cross-attention decoder layers that draft against the
target's KV caches (``DraftCache.TARGET`` — the draft keeps no cache of its
own).

The graph-API draft shards itself per device and threads per-device lists;
this one holds single modules and lets the sharding machinery split them:
``ColumnParallelLinear`` for ``q_proj``, ``RowParallelLinear`` for
``o_proj``, and the fused MLP projections. The draft's plain ``Linear``
projections (``pre_projection`` / ``post_projection``) run replicated over
the full weight, as the DeepSeek-V3.2 NextN draft's ``eh_proj`` does.
"""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import rope_ragged
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import DeviceMapping, DeviceMesh, Replicated
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import cross_attention_ragged as _cross_attention_ragged
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gemma3_modulev3.layers.scaled_word_embedding import (
    ScaledEmbedding,
)
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.pipelines.architectures.gemma4_modulev3.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)
from max.pipelines.speculative.driver import gather_variable_rows

from ..gemma4_assistant.model_config import Gemma4AssistantConfig

# `cross_attention_ragged` has no pre-wrapped ModuleV3 binding, so it gets
# one here: without a sharding rule the op runs per device, which is what
# this Q-only attention wants — each device's shard of Q against its own
# view of the target's cache.
cross_attention_ragged = F.functional(_cross_attention_ragged)

# Map from layer type string to the index in the kv_by_type list.
_LAYER_TYPE_TO_KV_INDEX = {
    "sliding_attention": 0,
    "full_attention": 1,
}


def _replicated(x: Tensor) -> Tensor:
    """Reduces a Partial tensor to Replicated (RMSNorm is not linear)."""
    return F.transfer_to(
        x, DeviceMapping(x.mesh, (Replicated(),) * x.mesh.ndim)
    )


class Gemma4AssistantAttention(Module[..., Tensor]):
    """Q-only cross-attention against the target model's KV cache.

    The draft has no K/V projections: it projects Q from its own hidden
    states, applies the per-head Q norm and RoPE, then cross-attends into
    the target's cache at ``target_layer_idx_in_cache``.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding | ProportionalRotaryEmbedding,
        num_attention_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        head_dim: int,
        target_layer_idx_in_cache: int,
        is_sliding: bool,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.head_dim = head_dim
        self.kv_params = kv_params
        self.target_layer_idx_in_cache = target_layer_idx_in_cache
        self.is_sliding = is_sliding
        # Attention scaling is absorbed by the q-norm (the target does the
        # same); do not apply 1/sqrt(head_dim).
        self.scale = 1.0
        self.local_window_size = local_window_size

        self.q_weight_dim = head_dim * num_attention_heads

        self.q_proj = ColumnParallelLinear(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            bias=False,
        )
        self.q_norm = Gemma4RMSNorm(head_dim, eps=qk_norm_eps)
        self.o_proj = RowParallelLinear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        *,
        input_row_offsets: Tensor,
        kv_input_row_offsets: Tensor,
        q_max_seq_len: Tensor,
        rope_cache_lengths: Tensor | None = None,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(
            self.target_layer_idx_in_cache, DType.uint32, device=CPU()
        )

        # Q projection, then per-head Q norm.
        x_q = self.q_proj(x)
        x_q = self.q_norm(
            x_q.reshape((-1, self.n_heads, self.head_dim))
        ).reshape((-1, self.q_weight_dim))

        # RoPE over Q. `rope_ragged` expects [total, n_heads, head_dim] and
        # positions from the cache lengths — the driver advances these by
        # the accepted count and hands them back as
        # `rope_cache_lengths` on draft steps.
        freqs_cis = F.cast(self.rope.freqs_cis, x_q.dtype).to(x_q.mesh)
        xq = x_q.reshape((-1, self.n_heads, self.head_dim))
        rope_positions = (
            rope_cache_lengths
            if rope_cache_lengths is not None
            else kv_collection.cache_lengths
        )
        # The op has no sharding rule, so it runs per device and returns an
        # Unknown placement; claim the head-sharded one the input had.
        xq = rope_ragged(
            xq,
            input_row_offsets,
            rope_positions,
            freqs_cis,
            interleaved=self.rope.interleaved,
            output_dtype=self.kv_params.dtype,
        ).rebind_mapping(xq.mapping)

        # Cross-attention against the target's KV cache.
        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.is_sliding
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = cross_attention_ragged(
            self.kv_params,
            input=xq,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=mask_variant,
            kv_input_row_offsets=kv_input_row_offsets,
            q_max_seq_len=q_max_seq_len,
            scale=self.scale,
            local_window_size=self.local_window_size if self.is_sliding else -1,
            output_dtype=DType.bfloat16,
        ).rebind_mapping(xq.mapping)

        # The width is spelled out because a skippable draft step sizes the
        # rows as ``num_draft_offsets - 1``, and shape inference cannot
        # cancel that expression out of a ``-1``.
        attn_out = attn_out.reshape((total_seq_len, self.q_weight_dim))
        return self.o_proj(attn_out)


class Gemma4AssistantDecoderLayer(Module[..., Tensor]):
    """Assistant decoder layer: Q-only cross-attention + dense MLP."""

    def __init__(
        self,
        attention: Gemma4AssistantAttention,
        mlp: Module[[Tensor], Tensor],
        hidden_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.layer_scalar = Tensor.ones([1])

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        *,
        input_row_offsets: Tensor,
        kv_input_row_offsets: Tensor,
        q_max_seq_len: Tensor,
        rope_cache_lengths: Tensor | None = None,
    ) -> Tensor:
        residual = x
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            kv_input_row_offsets=kv_input_row_offsets,
            q_max_seq_len=q_max_seq_len,
            rope_cache_lengths=rope_cache_lengths,
        )
        h = residual + self.post_attention_layernorm(_replicated(attn_out))

        residual = h
        mlp_out = self.mlp(self.pre_feedforward_layernorm(h))
        h = residual + self.post_feedforward_layernorm(_replicated(mlp_out))
        return h * self.layer_scalar


class Gemma4AssistantModuleV3(Module[..., tuple[Tensor, ...]]):
    """The Gemma4 Assistant (MTP draft) model, on ModuleV3.

    Takes token IDs and the target's hidden states, projects them into the
    assistant's 1024-dim hidden space, runs the cross-attention decoder
    stack, then projects back to the backbone dimension for the next MTP
    step's hidden states.

    ``embed_tokens`` is aliased to the target's ``ScaledEmbedding`` before
    ``compile()``; the shared instance means both checkpoint paths
    (``target.embed_tokens.weight`` / ``draft.embed_tokens.weight``) load
    the same weight. ``draft_embed_tokens`` is the assistant's own
    1024-dim embedding, also serving as its tied lm_head.
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        target_layer_types: list[str],
        target_sliding_kv_params: KVCacheParams,
        target_global_kv_params: KVCacheParams,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        device = mesh.devices[0]

        # Per-layer-type rotary embeddings, built as the target builds
        # its own.
        self.rope_sliding = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.sliding_window_rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=device,
            head_dim=config.head_dim,
            interleaved=False,
        )
        self.rope_global = ProportionalRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.global_rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=device,
            head_dim=config.global_head_dim,
            interleaved=False,
            scaling_params=config.global_rope_scaling,
        )

        # Target embedding (backbone dim) for the concat input. Aliased to
        # the target's embed_tokens by the model before compile.
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size,
            dim=config.backbone_hidden_size,
            embed_scale=config.backbone_hidden_size**0.5,
        )

        # Assistant's own embedding (1024-dim) for the tied lm_head.
        self.draft_embed_tokens = ScaledEmbedding(
            config.vocab_size,
            dim=config.hidden_size,
            embed_scale=config.hidden_size**0.5,
        )

        # Pre-projection: concat(embed, hidden) -> assistant dim. A plain
        # Linear over the full weight: its input is replicated, so the
        # matmul runs per device — the GLM NextN draft's eh_proj precedent.
        self.pre_projection = Linear(
            config.backbone_hidden_size * 2, config.hidden_size, bias=False
        )

        # Post-projection: assistant dim -> backbone dim, likewise plain.
        self.post_projection = Linear(
            config.hidden_size, config.backbone_hidden_size, bias=False
        )

        # Per-layer-type mapping from the assistant's layers to the
        # target's cache: every assistant layer of one type cross-attends
        # into the LAST target layer of that type.
        target_layer_type_counts: dict[str, int] = {
            "sliding_attention": 0,
            "full_attention": 0,
        }
        for lt in target_layer_types:
            target_layer_type_counts[lt] += 1
        self._target_kv_layer_idx: dict[str, int] = {
            lt: count - 1 for lt, count in target_layer_type_counts.items()
        }

        kv_params_by_type: dict[str, KVCacheParams] = {
            "sliding_attention": target_sliding_kv_params,
            "full_attention": target_global_kv_params,
        }

        layers = []
        self._layer_kv_index: list[int] = []
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            is_sliding = layer_type == "sliding_attention"
            kv_params = kv_params_by_type[layer_type]
            head_dim = config.head_dim if is_sliding else config.global_head_dim

            layers.append(
                Gemma4AssistantDecoderLayer(
                    attention=Gemma4AssistantAttention(
                        rope=(
                            self.rope_sliding
                            if is_sliding
                            else self.rope_global
                        ),
                        num_attention_heads=config.num_attention_heads,
                        hidden_size=config.hidden_size,
                        kv_params=kv_params,
                        head_dim=head_dim,
                        target_layer_idx_in_cache=(
                            self._target_kv_layer_idx[layer_type]
                        ),
                        is_sliding=is_sliding,
                        qk_norm_eps=config.rms_norm_eps,
                        local_window_size=config.sliding_window,
                    ),
                    mlp=MLP(
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                        activation_function=config.hidden_activation,
                    ),
                    hidden_size=config.hidden_size,
                    rms_norm_eps=config.rms_norm_eps,
                )
            )
            self._layer_kv_index.append(_LAYER_TYPE_TO_KV_INDEX[layer_type])

        self.layers = ModuleList(layers)

        # Final norm on the assistant dim, applied before post_projection.
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _tied_logits(self, h: Tensor) -> Tensor:
        """Tied lm_head: the assistant's own 1024-dim embedding, no cap.

        Mirrors the graph arch's ColumnParallelLinear tied to
        ``draft_embed_tokens.weight``: a plain matmul plus the vocab
        allgather, cast to float32. The draft applies no softcapping (the
        graph arch passes none either).
        """
        outputs = h @ self.draft_embed_tokens.weight.T
        outputs = F.allgather(outputs, tensor_axis=-1)
        return outputs.cast(DType.float32)

    def forward(
        self,
        tokens: Tensor,
        hidden_states: list[Tensor],
        target_sliding_kv: PagedCacheValues,
        target_global_kv: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: list[Tensor],
        kv_input_row_offsets: list[Tensor],
        q_max_seq_len: Tensor,
        rope_cache_lengths: list[Tensor] | None = None,
        *,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates,
    ) -> tuple[Tensor, ...]:
        """One draft pass over the query window.

        Args:
            tokens: The rows to draft.
            hidden_states: The target's per-device hidden states at the same
                rows, one single-device tensor per device.
            target_sliding_kv: The target's sliding cache.
            target_global_kv: The target's full-attention cache.
            return_n_logits: Per-request VARIABLE logit count (prefill).
            input_row_offsets: Per-device query row offsets.
            kv_input_row_offsets: Per-device merged-window row offsets —
                the cache spans that window, not the query.
            q_max_seq_len: Longest query, for the cross-attention kernel.
            rope_cache_lengths: Per-device positions to advance RoPE by on
                draft steps; None in prefill, where the cache lengths hold
                the positions.
            return_logits: ``VARIABLE`` in prefill, ``LAST_TOKEN`` in step.
            return_hidden_states: ``ALL`` in prefill (the full window's
                post-projection output), ``LAST_PER_DEVICE`` in step.

        Returns:
            ``(last_logits,)`` in step mode; in prefill mode
            ``(last_logits, logits, offsets)`` followed by one per-device
            hidden-state tensor, matching the graph arch's output layout
            the adapters index positionally.
        """
        kv_by_type = [target_sliding_kv, target_global_kv]
        replicated = DeviceMapping.replicated(self.mesh)

        # The per-device hidden states and offsets arrive as single-device
        # tensors; rebuild each as one replicated distributed tensor.
        hidden = Tensor.from_shard_values(hidden_states, replicated)
        offsets = Tensor.from_shard_values(input_row_offsets, replicated)
        kv_offsets = Tensor.from_shard_values(kv_input_row_offsets, replicated)
        rope_positions: Tensor | None = None
        if rope_cache_lengths is not None:
            rope_positions = Tensor.from_shard_values(
                rope_cache_lengths, replicated
            )

        # The driver hands the tokens over on the first device. Stage them on
        # every device with one collective: embedding a single-device tensor
        # against the replicated table makes the compiler insert a gpu0 ->
        # gpu1 transfer wrapped in a host-enqueued cross-stream wait, which
        # is illegal inside device-graph capture.
        tokens = F.distributed_broadcast(tokens, self.mesh)
        h_embed = self.embed_tokens(tokens)
        # Align the embed's symbolic batch dim with the hidden states'
        # per-step dim so the concat's shape stays per-device symbolic.
        h_embed = F.rebind(h_embed, hidden.shape)
        h = F.concat([h_embed, hidden], axis=-1)

        # Project to the assistant dim. The concat is replicated, so the
        # plain Linear runs per device over the full weight.
        h = self.pre_projection(h)

        for idx, layer in enumerate(self.layers):
            h = layer(
                h,
                kv_by_type[self._layer_kv_index[idx]],
                input_row_offsets=offsets,
                kv_input_row_offsets=kv_offsets,
                q_max_seq_len=q_max_seq_len,
                rope_cache_lengths=rope_positions,
            )

        # Assistant-dim final norm: HF applies lm_head to this 1024-dim
        # output; the 5376-dim post-projection output is the next step's
        # hidden state, not the logits' source.
        norm_h = self.norm(h)

        # Last-token logits, always computed as the graph arch does.
        last_h = F.gather(norm_h, offsets[1:] - 1, axis=0)
        last_logits = self._tied_logits(last_h)

        ret_val: tuple[Tensor, ...] = (last_logits,)

        if return_logits == ReturnLogits.VARIABLE:
            var_h, logit_offsets = gather_variable_rows(
                norm_h, offsets, return_n_logits, self.mesh
            )
            logits = self._tied_logits(var_h)
            ret_val += (logits, logit_offsets)

        # Hidden states: the 5376-dim post-projection output, feeding the
        # next MTP step.
        h_out = self.post_projection(norm_h)
        if return_hidden_states == ReturnHiddenStates.ALL:
            ret_val += tuple(h_out.local_shards)
        elif return_hidden_states == ReturnHiddenStates.LAST_PER_DEVICE:
            # h_out is replicated, so the gather runs on each device's own
            # rows.
            last_h_out = F.gather(h_out, offsets[1:] - 1, axis=0)
            ret_val += tuple(last_h_out.local_shards)

        return ret_val
