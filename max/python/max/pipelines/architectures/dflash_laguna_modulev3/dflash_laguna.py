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
"""The Laguna DFlash drafter on ModuleV3.

A causal block over windowed target context. The block is
``[anchor, mask x K]``, every slot embedded through the target's embedding;
each block query attends to the keys of the last ``sliding_window``
positions, itself included. The layers are Laguna's: pre-norm, a fused
``qkv_proj``, per-head ``q_norm`` and ``k_norm``, NeoX RoPE over the whole
head, a per-head softplus gate on the attention output, and a dense SiLU MLP.

The context the target hands over is
``hidden_norm(fc(cat(aux_hidden_norms[i](tap_i))))``. Each layer projects it
to K/V through its own ``input_layernorm``, as vLLM's drafter does.

Weight names are the checkpoint's.
"""

from __future__ import annotations

from collections.abc import Sequence

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module, ModuleList, RMSNorm
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues as PagedCacheGraphValues
from max.tensor import Tensor

from .model_config import DFlashLagunaConfig

__all__ = ["DFlashLaguna", "DFlashLagunaAttention", "DFlashLagunaDecoderLayer"]


def _softplus(x: Tensor) -> Tensor:
    return F.max(x, 0.0) + F.log1p(F.exp(-F.abs(x)))


class DFlashLagunaAttention(Module[..., Tensor]):
    """A drafter layer's gated attention.

    The K/V a block query attends to must be computed the same way whether it
    came from a target tap or from the block itself, so both paths store it
    through :meth:`_store_kv`.
    """

    def __init__(self, config: DFlashLagunaConfig, layer_idx: int) -> None:
        self.config = config
        self.layer_idx = layer_idx
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.q_dim = heads * config.head_dim
        kv_dim = kv_heads * config.head_dim
        self.qkv_proj = Linear(
            config.hidden_size, self.q_dim + 2 * kv_dim, bias=False
        )
        self.g_proj = Linear(config.hidden_size, heads, bias=False)
        self.o_proj = Linear(self.q_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def _store_kv(
        self,
        kv: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        q: Tensor | None = None,
    ) -> Tensor:
        """Writes the K and V heads of ``kv`` at their rows' positions.

        The fused RoPE kernel rotates the cached keys and requires a query
        input. For a context-only write, the key rows stand in as that query
        and the caller discards the result.

        Returns:
            The rotated query rows, or the rotated key rows when ``q`` is
            None.
        """
        kv_heads = self.config.num_key_value_heads
        k = self.k_norm(kv[:, :kv_heads])
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())
        store_k_cache_ragged(kv_collection, k, input_row_offsets, layer_idx)
        store_v_cache_ragged(
            kv_collection, kv[:, kv_heads:], input_row_offsets, layer_idx
        )
        return fused_qk_ragged_rope(
            self.config.kv_params,
            k if q is None else q,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=False,
        )

    def write_context(
        self,
        ctx: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes the K/V of the normed context rows, with no query."""
        config = self.config
        kv = (ctx @ self.qkv_proj.weight[self.q_dim :].T).reshape(
            (-1, 2 * config.num_key_value_heads, config.head_dim)
        )
        self._store_kv(kv, kv_collection, freqs_cis, input_row_offsets)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block's attention over its normed rows."""
        config = self.config
        heads, head_dim = config.num_attention_heads, config.head_dim
        rows = x.shape[0]
        qkv = self.qkv_proj(x).reshape(
            (-1, heads + 2 * config.num_key_value_heads, head_dim)
        )
        q = self._store_kv(
            qkv[:, heads:],
            kv_collection,
            freqs_cis,
            input_row_offsets,
            q=self.q_norm(qkv[:, :heads]),
        )
        attn = flash_attention_ragged(
            config.kv_params,
            input=q,
            kv_collection=kv_collection,
            layer_idx=F.constant(self.layer_idx, DType.uint32, device=CPU()),
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK,
            local_window_size=config.sliding_window,
            scale=head_dim**-0.5,
        )
        # The reference computes the gate in float32 and casts it back.
        gate = _softplus(self.g_proj(x).cast(DType.float32)).cast(attn.dtype)
        gated = attn.reshape((rows, heads, head_dim)) * F.unsqueeze(gate, -1)
        return self.o_proj(gated.reshape((rows, -1)))


class DFlashLagunaDecoderLayer(Module[..., Tensor]):
    """A pre-norm drafter layer: gated attention, then a SiLU MLP."""

    def __init__(self, config: DFlashLagunaConfig, layer_idx: int) -> None:
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = DFlashLagunaAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        h: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the layer over the block's residual stream."""
        h = h + self.self_attn(
            self.input_layernorm(h), kv_collection, freqs_cis, input_row_offsets
        )
        return h + self.mlp(self.post_attention_layernorm(h)).to(h.mapping)

    def write_context(
        self,
        ctx: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes this layer's context K/V."""
        self.self_attn.write_context(
            self.input_layernorm(ctx),
            kv_collection,
            freqs_cis,
            input_row_offsets,
        )


class DFlashLaguna(Module[[], None]):
    """The drafter: the tap projection, the layer stack and the final norm.

    The target's ``embed_tokens`` embeds the block and its ``lm_head`` scores
    the drafts, so the drafter holds neither.

    No ``forward``: the fused graph drives the drafter through its named
    entries, never ``__call__``.
    """

    def __init__(self, config: DFlashLagunaConfig) -> None:
        self.config = config
        eps = config.rms_norm_eps
        self.rope = RotaryEmbedding(
            dim=config.head_dim * config.num_attention_heads,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
        )
        self.aux_hidden_norms = ModuleList(
            [
                RMSNorm(config.hidden_size, eps=eps)
                for _ in config.target_layer_ids
            ]
        )
        self.fc = Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=eps)
        self.layers = ModuleList(
            [
                DFlashLagunaDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=eps)

    def _cache(
        self, kv_collections: Sequence[PagedCacheGraphValues], x: Tensor
    ) -> PagedCacheValues:
        if len(kv_collections) != 1:
            raise ValueError(
                "The Laguna DFlash drafter runs on one device; got"
                f" {len(kv_collections)} caches."
            )
        return PagedCacheValues.from_upstream(kv_collections, x.mapping)

    def _freqs_cis(self, x: Tensor) -> Tensor:
        # vLLM rotates both the context and the block with a table in the
        # activation dtype.
        return self.rope.freqs_cis.cast(x.dtype).to(x.device)

    def write_context(
        self,
        target_hidden: Tensor,
        input_row_offsets: Tensor,
        kv_collections: Sequence[PagedCacheGraphValues],
    ) -> None:
        """Writes every layer's context K/V from the target's taps.

        Args:
            target_hidden: The target's residual stream after each
                ``target_layer_ids`` layer, concatenated in that order,
                ``[rows, taps * hidden]``.
            input_row_offsets: Ragged offsets of the rows.
            kv_collections: The drafter cache, whose lengths are the rows'
                first positions.
        """
        hidden = self.config.hidden_size
        taps = [
            norm(target_hidden[:, i * hidden : (i + 1) * hidden])
            for i, norm in enumerate(self.aux_hidden_norms)
        ]
        ctx = self.hidden_norm(self.fc(F.concat(taps, axis=-1)))
        kv_collection = self._cache(kv_collections, ctx)
        freqs_cis = self._freqs_cis(ctx)
        for layer in self.layers:
            layer.write_context(
                ctx, kv_collection, freqs_cis, input_row_offsets
            )

    def forward_block(
        self,
        block_embeds: Tensor,
        kv_collections: Sequence[PagedCacheGraphValues],
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block, whose K/V lands at the cache's lengths.

        Args:
            block_embeds: The ``[batch * block_size, hidden]`` block rows.
            kv_collections: The drafter cache, whose lengths are the anchors'
                positions and whose dispatch metadata is sized for the block.
            input_row_offsets: The block's ragged offsets.

        Returns:
            The final-normed ``[batch * block_size, hidden]`` hidden states.
        """
        kv_collection = self._cache(kv_collections, block_embeds)
        freqs_cis = self._freqs_cis(block_embeds)
        h = block_embeds
        for layer in self.layers:
            h = layer(h, kv_collection, freqs_cis, input_row_offsets)
        return self.norm(h)
