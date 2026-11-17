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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/dflash2_qwen3_5/dflash2_qwen3_5.py
# at main 4df8111afba.
"""The DFlash2 drafter of a Qwen3.5 target, on ModuleV3.

DFlash2 is DFlash's non-causal block drafter plus two mechanisms: a two-tap
grouped dynamic convolution around each sublayer, and a candidate-path
selector that keeps ``selector_top_k`` tokens at every block position and
traces one coherent path through them. The rest is the DFlash body: the
``fc``/``hidden_norm`` tap fusion, K/V written from the target's hidden
states, and a windowed non-causal block forward.

The drafter holds no embedding and no head: its proposer embeds the block
and scores the drafts through the target's. It runs on one device.
"""

from __future__ import annotations

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module, ModuleList
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import QKVLinear
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMesh

from .layers import DFlash2CandidateSelector, DFlash2GroupedConv, DFlash2RMSNorm
from .model_config import DFlash2Qwen3_5Config

__all__ = ["DFlash2Attention", "DFlash2DecoderLayer", "DFlash2Qwen3_5"]


class DFlash2Attention(Module[..., Tensor]):
    """A drafter layer's attention, with per-head ``q_norm`` and ``k_norm``.

    The K/V a block query attends to must be computed the same way whether
    it came from a target tap or from the block itself, so both paths go
    through :meth:`_rope_and_store`. Inside a block every query attends to
    every block key, bounded only by the layer's window.
    """

    def __init__(self, config: DFlash2Qwen3_5Config, layer_idx: int) -> None:
        self.kv_params = config.kv_params
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = config.attention_scale
        self.window = config.layer_windows[layer_idx]
        hidden = config.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = config.num_key_value_heads * self.head_dim
        self.qkv_proj = QKVLinear(hidden, q_dim, kv_dim)
        self.o_proj = Linear(q_dim, hidden, bias=False)
        self.q_norm = DFlash2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlash2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _layer_idx(self) -> Tensor:
        return F.constant(self.layer_idx, DType.uint32, device=CPU())

    def _rope_and_store(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        layer_idx: Tensor,
    ) -> Tensor:
        """Writes the roped K and the V of ``x``'s rows; returns roped Q.

        The rows land at the cache's lengths. Q is projected and normed
        either way, since the store reads the fused QKV rows.
        """
        head_dim = self.head_dim
        q_dim, kv_dim = self.qkv_proj.q_dim, self.qkv_proj.kv_dim
        # Not ``self.qkv_proj(x)``, which adds a zero bias.
        qkv = x @ self.qkv_proj.fused_weight.T
        x_q, x_k, x_v = F.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
        x_q = self.q_norm(x_q.reshape([-1, head_dim])).reshape(x_q.shape)
        x_k = self.k_norm(x_k.reshape([-1, head_dim])).reshape(x_k.shape)
        qkv = F.concat([x_q, x_k, x_v], axis=-1)
        return rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis.cast(qkv.dtype).to(qkv.device),
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.num_heads,
            interleaved=False,
        )

    def store_kv(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes the context K/V of ``x``'s rows, with no attention."""
        self._rope_and_store(
            x, kv_collection, freqs_cis, input_row_offsets, self._layer_idx()
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Attends the block's rows over the context and the whole block."""
        layer_idx = self._layer_idx()
        xq = self._rope_and_store(
            x, kv_collection, freqs_cis, input_row_offsets, layer_idx
        ).reshape([-1, self.num_heads, self.head_dim])
        attn = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=(
                MHAMaskVariant.NULL_MASK
                if self.window is None
                else MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
            ),
            scale=self.scale,
            local_window_size=-1 if self.window is None else self.window,
        )
        return self.o_proj(
            attn.reshape([x.shape[0], self.num_heads * self.head_dim])
        )


class DFlash2DecoderLayer(Module[..., Tensor]):
    """A pre-norm drafter layer with a convolution around each sublayer.

    ``r = x; h = conv_a.prepare(norm(r)); h = attn(h); r += conv_a.finish(h)``
    and the same again for the MLP.
    """

    def __init__(self, config: DFlash2Qwen3_5Config, layer_idx: int) -> None:
        dflash = config.dflash

        def conv() -> DFlash2GroupedConv:
            return DFlash2GroupedConv(
                config.hidden_size,
                taps=dflash.conv_kernel_size,
                group_size=dflash.conv_group_size,
                block_size=dflash.block_size,
            )

        self.input_layernorm = DFlash2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DFlash2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = DFlash2Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.attention_conv = conv()
        self.mlp_conv = conv()

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the layer over the block's rows."""
        h, coefficients = self.attention_conv.prepare(self.input_layernorm(x))
        h = self.self_attn(h, kv_collection, freqs_cis, input_row_offsets)
        residual = x + self.attention_conv.finish(h, coefficients)
        h, coefficients = self.mlp_conv.prepare(
            self.post_attention_layernorm(residual)
        )
        h = self.mlp(h)
        return residual + self.mlp_conv.finish(h, coefficients)


class DFlash2Qwen3_5(Module[[], None]):
    """The drafter: the tap projection, the layer stack and the selector.

    No ``forward``: the fused graph drives the drafter through its named
    entries, never ``__call__``.
    """

    def __init__(self, config: DFlash2Qwen3_5Config) -> None:
        self.config = config
        hidden = config.hidden_size
        dflash = config.dflash
        with default_dtype(config.dtype):
            self.rope = RotaryEmbedding(
                # RoPE spans heads x head_dim, which differs from the hidden
                # size when head_dim isn't hidden_size // heads.
                dim=config.head_dim * config.num_attention_heads,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                device=CPU(),
                head_dim=config.head_dim,
                interleaved=False,
            )
            self.layers = ModuleList(
                [
                    DFlash2DecoderLayer(config, i)
                    for i in range(config.num_hidden_layers)
                ]
            )
            self.norm = DFlash2RMSNorm(hidden, eps=config.rms_norm_eps)
            self.fc = Linear(
                len(config.target_layer_ids) * hidden, hidden, bias=False
            )
            self.hidden_norm = DFlash2RMSNorm(hidden, eps=config.rms_norm_eps)
            self.candidate_selector = DFlash2CandidateSelector(
                hidden,
                vocab_size=config.vocab_size,
                rank=dflash.selector_rank,
                top_k=dflash.selector_top_k,
            )

    @property
    def mesh(self) -> DeviceMesh:
        """The device the drafter runs on, as a mesh."""
        return self.norm.weight.mesh

    def materialize(
        self,
        target_hidden: Tensor,
        input_row_offsets: Tensor,
        kv_collection: PagedCacheValues,
    ) -> None:
        """Writes every layer's context K/V from the target's taps.

        No ``input_layernorm`` and no convolution on this path.

        Args:
            target_hidden: The target's taps, concatenated in
                ``target_layer_ids`` order, ``[rows, taps * hidden]``.
            input_row_offsets: Ragged offsets of the rows.
            kv_collection: The drafter cache, whose lengths place the rows.
        """
        ctx_hidden = self.hidden_norm(self.fc(target_hidden))
        freqs_cis = self.rope.freqs_cis
        for layer in self.layers:
            layer.self_attn.store_kv(
                ctx_hidden, kv_collection, freqs_cis, input_row_offsets
            )

    def forward_block(
        self,
        input_embeds: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs ``block_size`` query rows per sequence through the stack.

        ``input_embeds`` must be dense ``block_size``-row blocks, one per
        sequence: the convolution's block axis is a reshape of the row axis,
        not a function of ``input_row_offsets``.

        Returns:
            The final-normed hidden states.
        """
        h = input_embeds
        freqs_cis = self.rope.freqs_cis
        for layer in self.layers:
            h = layer(h, kv_collection, freqs_cis, input_row_offsets)
        return self.norm(h)

    def compute_candidates(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Returns each slot's top-k candidate ids and their logits.

        Args:
            logits: ``[batch, steps, vocab]`` float32 logits of the borrowed
                head at each mask slot.

        Returns:
            ``(candidate_ids, unary_logits)``, both
            ``[batch, steps, selector_top_k]``.
        """
        top_k = self.candidate_selector.top_k
        unary_logits, candidate_ids = F.top_k(
            logits.reshape([-1, logits.shape[-1]]), top_k
        )
        shape = [logits.shape[0], logits.shape[1], top_k]
        return candidate_ids.reshape(shape), unary_logits.reshape(shape)
