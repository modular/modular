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
"""The Beam DFlash2 drafter on ModuleV3.

A non-causal block over windowed target context, as the MiMo-V2 DFlash
drafter on ModuleV3 is, plus DFlash2's two additions: a dynamic grouped
convolution around each sublayer and a candidate-path selector over the
borrowed head's top-k.

The block is ``[anchor, mask x K]``, every slot embedded through the
target's embedding. Each block query attends to the context keys its window
reaches and to every block key. The layers are pre-norm, with per-head
``q_norm`` and ``k_norm`` and NeoX RoPE over the whole head. The context the
target hands over is ``hidden_norm(fc(taps))``, projected to each layer's
K/V with no ``input_layernorm`` and no convolution.

Weight names are the checkpoint's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, DimLike, TensorValue, ops
from max.nn import Module, ModuleList, RMSNorm
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import NamedMapping

from .layers import DFlash2BeamCandidateSelector, DFlash2BeamConv
from .model_config import DFlash2BeamConfig

__all__ = ["DFlash2Beam", "DFlash2BeamAttention", "DFlash2BeamDecoderLayer"]


def _heads_per_shard(heads: int, num_shards: int, what: str) -> int:
    if heads % num_shards:
        raise ValueError(
            f"Beam DFlash2: {heads} {what} heads do not split over"
            f" {num_shards} devices."
        )
    return heads // num_shards


class DFlash2BeamAttention(Module[..., Tensor]):
    """A drafter layer's attention, column parallel over the heads.

    The K/V a block query attends to must be computed the same way whether it
    came from a target tap or from the block itself, so both paths go through
    :meth:`store_kv`. Each device stores its heads' K/V; ``o_proj`` returns
    the shards' partial sums.
    """

    def __init__(self, config: DFlash2BeamConfig, layer_idx: int) -> None:
        self.config = config
        self.layer_idx = layer_idx
        shards = len(config.devices)
        self.num_heads = config.num_attention_heads
        self.local_heads = _heads_per_shard(self.num_heads, shards, "query")
        self.num_key_value_heads = config.num_key_value_heads
        _heads_per_shard(self.num_key_value_heads, shards, "KV")
        head_dim = config.head_dim
        q_dim = self.num_heads * head_dim
        kv_dim = self.num_key_value_heads * head_dim
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, q_dim, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, kv_dim, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, kv_dim, bias=False
        )
        self.o_proj = RowParallelLinear(q_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def store_kv(
        self,
        x: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        q: Tensor | None = None,
    ) -> list[TensorValue]:
        """Writes the K/V of ``x``'s rows at their positions and ropes ``q``.

        The kernel that ropes the cached K ropes a query beside it, so
        without ``q`` the K rows stand in for it and the result is
        discarded.

        Args:
            x: ``[rows, hidden]``, ragged by ``input_row_offsets`` and
                replicated over the mesh.
            kv_collections: The drafter cache, one per device, whose lengths
                place the rows.
            freqs_cis: The ``[positions, head_dim]`` RoPE table, replicated.
            input_row_offsets: Ragged offsets of ``x``, replicated.
            q: ``[rows, heads, head_dim]`` queries, sharded over the heads.

        Returns:
            ``q``, or the K rows without it, roped at the rows' positions;
            one entry per device.
        """
        config = self.config
        head_dim = config.head_dim
        heads = self.num_key_value_heads
        k = self.k_norm(self.k_proj(x).reshape([-1, heads, head_dim]))
        v = self.v_proj(x).reshape([-1, heads, head_dim])
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        roped: list[TensorValue] = []
        with ensure_context():
            for d in range(x.mesh.num_devices):
                k_d = TensorValue(k.local_shards[d])
                offsets_d = TensorValue(input_row_offsets.local_shards[d])
                store_k_cache_ragged(
                    kv_collections[d], k_d, offsets_d, layer_idx
                )
                store_v_cache_ragged(
                    kv_collections[d],
                    TensorValue(v.local_shards[d]),
                    offsets_d,
                    layer_idx,
                )
                roped.append(
                    fused_qk_ragged_rope(
                        config.kv_params,
                        k_d if q is None else TensorValue(q.local_shards[d]),
                        offsets_d,
                        kv_collections[d],
                        TensorValue(freqs_cis.local_shards[d]),
                        layer_idx,
                        interleaved=False,
                    )
                )
        return roped

    def forward(
        self,
        x: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block's attention on each device's heads.

        Returns:
            This layer's ``o_proj`` output, a partial sum per device.
        """
        head_dim = self.config.head_dim
        q = self.q_norm(self.q_proj(x).reshape([-1, self.num_heads, head_dim]))
        roped = self.store_kv(
            x, kv_collections, freqs_cis, input_row_offsets, q=q
        )
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        attns: list[TensorValue] = []
        with ensure_context():
            for d in range(x.mesh.num_devices):
                rows = x.local_shards[d].shape[0]
                attn = flash_attention_ragged(
                    self.config.kv_params,
                    input=roped[d],
                    input_row_offsets=TensorValue(
                        input_row_offsets.local_shards[d]
                    ),
                    kv_collection=kv_collections[d],
                    layer_idx=layer_idx,
                    mask_variant=MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK,
                    scale=head_dim**-0.5,
                    local_window_size=self.config.sliding_window,
                )
                attns.append(
                    ops.reshape(attn, [rows, self.local_heads * head_dim])
                )
        if x.mesh.num_devices == 1:
            heads = Tensor.from_shard_values(attns, x.mapping)
        else:
            heads = Tensor.from_shard_values(
                attns, NamedMapping(x.mesh, (None, TP))
            )
        return self.o_proj(heads)


class DFlash2BeamDecoderLayer(Module[..., Tensor]):
    """A pre-norm drafter layer with a convolution around each sublayer.

    ``r = x; h = conv_a.prepare(norm(r)); h = attn(h); r += conv_a.finish(h)``
    and the same again for the MLP.
    """

    def __init__(self, config: DFlash2BeamConfig, layer_idx: int) -> None:
        dflash = config.dflash

        def conv() -> DFlash2BeamConv:
            return DFlash2BeamConv(
                config.hidden_size,
                taps=dflash.conv_kernel_size,
                group_size=dflash.conv_group_size,
                block_size=config.block_size,
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = DFlash2BeamAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.attention_conv = conv()
        self.mlp_conv = conv()

    def forward(
        self,
        h: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the layer over the replicated block rows."""
        x, coefficients = self.attention_conv.prepare(self.input_layernorm(h))
        # The heads' partial sums reduce before the convolution reads them.
        x = self.self_attn(x, kv_collections, freqs_cis, input_row_offsets).to(
            h.mapping
        )
        h = h + self.attention_conv.finish(x, coefficients)
        x, coefficients = self.mlp_conv.prepare(
            self.post_attention_layernorm(h)
        )
        x = self.mlp(x).to(h.mapping)
        return h + self.mlp_conv.finish(x, coefficients)


class DFlash2Beam(Module[[], None]):
    """The drafter: the tap projection, the layer stack and the selector.

    The target's ``embed_tokens`` embeds the block and its ``lm_head`` scores
    the drafts, so the drafter holds neither.

    No ``forward``: the fused graph drives the drafter through its named
    entries, never ``__call__``.
    """

    def __init__(self, config: DFlash2BeamConfig) -> None:
        self.config = config
        self.rope = RotaryEmbedding(
            dim=config.head_dim * config.num_attention_heads,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
        )
        # Row parallel: each device projects its slice of the concatenated
        # taps, and the mesh sums the partial products ahead of hidden_norm.
        self.fc = RowParallelLinear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = ModuleList(
            [
                DFlash2BeamDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        dflash = config.dflash
        self.candidate_selector = DFlash2BeamCandidateSelector(
            config.hidden_size,
            vocab_size=config.vocab_size,
            rank=dflash.selector_rank,
            top_k=dflash.selector_top_k,
        )

    def _freqs_cis(self, h: Tensor) -> Tensor:
        """The RoPE table, replicated, in its stored float32.

        The kernel rounds once per rotation; a bf16 table would pre-round
        cos/sin and shift every cached key.
        """
        return self.rope.freqs_cis.to(h.mesh)

    def write_context(
        self,
        taps: Tensor,
        input_row_offsets: Tensor,
        kv_collections: Sequence[PagedCacheValues],
    ) -> None:
        """Writes every layer's context K/V from the target's taps.

        Args:
            taps: The target's residual stream after each
                ``target_layer_ids`` layer, concatenated in that order,
                ``[rows, taps * hidden]``, replicated.
            input_row_offsets: Ragged offsets of the rows, replicated.
            kv_collections: The drafter cache, one per device, whose lengths
                are the rows' first positions.
        """
        mapping = taps.mapping
        if taps.mesh.num_devices > 1:
            taps = taps.to(NamedMapping(taps.mesh, (None, TP)))
        ctx = self.hidden_norm(self.fc(taps).to(mapping))
        freqs_cis = self._freqs_cis(ctx)
        for layer in self.layers:
            layer.self_attn.store_kv(
                ctx, kv_collections, freqs_cis, input_row_offsets
            )

    def forward_block(
        self,
        block_embeds: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block, whose K/V lands at the caches' lengths.

        Args:
            block_embeds: The ``[batch * block_size, hidden]`` block rows,
                replicated.
            kv_collections: Per device, the drafter cache, whose lengths are
                the anchors' positions and whose dispatch metadata is sized
                for the block.
            input_row_offsets: The block's ragged offsets, replicated.

        Returns:
            The final-normed hidden states, replicated.
        """
        h = block_embeds
        freqs_cis = self._freqs_cis(h)
        for layer in self.layers:
            h = layer(h, kv_collections, freqs_cis, input_row_offsets)
        return self.norm(h)

    def draft_tokens(
        self,
        logits: TensorValue,
        drafted: Tensor,
        anchor_token_ids: TensorValue,
        sample: Callable[[TensorValue, int, TensorValue], TensorValue],
    ) -> TensorValue:
        """Picks each block's drafts through the candidate selector.

        Args:
            logits: ``[batch * steps, vocab]`` float32 head logits on device
                0, softcapped and with the padding rows at negative infinity.
            drafted: ``[batch * steps, hidden]`` drafter output at the
                drafted slots, replicated.
            anchor_token_ids: ``[batch]`` anchors on device 0.
            sample: Draws step ``l``'s ``[batch]`` candidate index from its
                ``[batch, top_k]`` successor scores and the candidates' ids.

        Returns:
            ``[batch, steps]`` drafted token ids on device 0.
        """
        selector = self.candidate_selector
        steps, top_k = self.config.block_size - 1, selector.top_k
        projected = selector.hidden_projection(drafted)
        with ensure_context():
            unary, ids = ops.top_k(logits, top_k, axis=-1)
            shape: list[DimLike] = ["batch_size", steps, top_k]
            ids = ids.reshape([-1, steps, top_k]).rebind(shape)
            unary = unary.reshape([-1, steps, top_k]).rebind(shape)
            scores = selector.score_edges(
                ids,
                unary,
                TensorValue(projected.local_shards[0])
                .reshape([-1, steps, int(projected.shape[-1])])
                .rebind(["batch_size", steps, int(projected.shape[-1])]),
                anchor_token_ids,
            )
            return selector.select_path(
                scores,
                ids,
                lambda row, step: sample(row, step, ids[:, step]),
            )
