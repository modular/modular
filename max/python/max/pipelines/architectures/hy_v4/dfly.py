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
"""The Hy4 VL DFly2 drafter (ModuleV3).

One parallel pass over the block ``[bonus, mask x N]`` of each request, then
N serial hidden-correction and ``lm_head`` steps. The drafter has no KV cache
and no indexer of its own: layer ``i`` reads and writes the MLA cache of
target layer ``dfly_target_kv_layers[i]`` and attends to that target layer's
top-k, followed by the block.
"""

from __future__ import annotations

from dataclasses import replace

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.quant_ops import is_fp8_block_quantized
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.norm import RMSNorm
from max.nn.sequential import ModuleList
from max.tensor import Tensor, default_dtype
from max.tensor.compilation import as_subgraph

from .layers.decoder import HYV4DecoderLayer
from .layers.mlp import HYV4MLP
from .model_config import HYV4VLConfig
from .mrope import mla_rope_tables

__all__ = ["HYV4DFlyDrafter", "draft_config"]


def draft_config(config: HYV4VLConfig, num_draft_tokens: int) -> HYV4VLConfig:
    """The target config narrowed to the drafter's layers.

    The block is the bonus token and N masks, so ``dfly.block_size`` becomes
    ``N + 1``, the width the drafter runs at, which need not be the trained
    one. A query row holds the target's top-k and then the block, so the
    sparse row is ``round_up(target_index_topk + N + 1, 128)`` wide, as in
    vLLM.
    """
    n = config.dfly.num_hidden_layers
    width = config.dfly.target_index_topk + num_draft_tokens + 1
    quant = config.quant_config
    if quant is not None and is_fp8_block_quantized(quant):
        quant = replace(quant, attn_quantized_layers=set(range(n)))
    return replace(
        config,
        quant_config=quant,
        dfly=replace(config.dfly, block_size=num_draft_tokens + 1),
        num_hidden_layers=n,
        indexer_types=["shared"] * n,
        mlp_layer_types=["dense"] * n,
        index_topk=-(-width // 128) * 128,
        dfly_draft=True,
    )


class HYV4DFlyFusion(Module[[Tensor, Tensor], Tensor]):
    """``e + gate * eh_proj(cat(enorm(e), hnorm(h)))`` on the bonus embedding."""

    def __init__(self, config: HYV4VLConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.enorm = RMSNorm(
            hidden, config.rms_norm_eps, multiply_before_cast=True
        )
        self.hnorm = RMSNorm(
            hidden, config.rms_norm_eps, multiply_before_cast=True
        )
        self.eh_proj = Linear(2 * hidden, hidden, bias=False)
        self.gate = Tensor.zeros([hidden])

    def forward(self, embeds: Tensor, target_hidden: Tensor) -> Tensor:
        fused = self.eh_proj(
            F.concat([self.enorm(embeds), self.hnorm(target_hidden)], axis=-1)
        )
        return embeds + self.gate * fused


class HYV4DFlyHiddenCorrection(HYV4MLP):
    """``h + mlp(cat(hidden_norm(h), embed_norm(e)))``.

    The MLP maps the ``2 * hidden`` concat through ``hidden`` back to
    ``hidden`` (vLLM's ``hidden_correction_intermediate_size`` default).
    """

    def __init__(self, config: HYV4VLConfig) -> None:
        hidden = config.hidden_size
        quant = config.projection_quant_config
        super().__init__(2 * hidden, hidden, quant, out_dim=hidden)
        self.hidden_norm = RMSNorm(
            hidden, config.rms_norm_eps, multiply_before_cast=True
        )
        self.embed_norm = RMSNorm(
            hidden, config.rms_norm_eps, multiply_before_cast=True
        )

    def correct(self, hidden: Tensor, embeds: Tensor) -> Tensor:
        """Returns ``hidden`` corrected by the previous token's embedding."""
        return hidden + self(
            F.concat(
                [self.hidden_norm(hidden), self.embed_norm(embeds)], axis=-1
            )
        )


def _slot(index: int) -> Tensor:
    return F.constant(index, DType.uint32, device=CPU())


class HYV4DFlyDrafter(Module[..., Tensor]):
    """The DFly2 drafter: five dense Hy4 layers without iHC or indexer.

    ``embed_tokens`` and ``lm_head`` are the target's; :meth:`propose` takes
    them. The weights are ``model.dfly_layers.*`` of the target's checkpoint,
    as ``hy_v4_dfly_weight_rules`` names them. On several devices the layers
    split as the target's do.

    Args:
        config: The draft config, from :func:`draft_config`.
    """

    def __init__(self, config: HYV4VLConfig) -> None:
        super().__init__()
        assert config.dfly_draft
        assert isinstance(config.kv_params, MultiKVCacheParams)
        mla = config.kv_params.children["mla"]
        assert isinstance(mla, KVCacheParams)
        self.page_size = mla.page_size
        self.width = config.dfly.block_size
        self.mask_token_id = config.dfly.mask_token_id
        self.kv_layers = tuple(config.dfly.target_kv_layers)
        self.index_topk = config.index_topk
        self.mrope_section = tuple(config.mrope_section)
        # Unquantized parameters are BF16, as the target decoder's are.
        with default_dtype(DType.bfloat16):
            self.layers = ModuleList[HYV4DecoderLayer](
                HYV4DecoderLayer(config, i, ihc=False)
                for i in range(config.num_hidden_layers)
            )
            for layer in self.layers:
                # The checkpoint stores the drafter's sinks BF16, the
                # target's float32; the attention widens them.
                attn = layer.self_attn
                attn.learnable_sink_param = Tensor.zeros(
                    attn.learnable_sink_param.shape, dtype=DType.bfloat16
                )
            self.final_norm = RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                multiply_before_cast=True,
            )
            self.mtp_fusion = HYV4DFlyFusion(config)
            self.hidden_correction = HYV4DFlyHiddenCorrection(config)
        if len(config.devices) > 1:
            for layer in self.layers:
                layer.shard()

    def _sparse_rows(
        self, target_topk: Tensor, kv_collection: PagedCacheValues
    ) -> Tensor:
        """Returns each block row's ``[batch * W, index_topk]`` cache rows.

        Row ``j`` of a request is the target's valid top-k, then the block
        positions ``anchor .. anchor + j``, then ``-1``. The rows are packed
        because the sparse decode kernels end a row at its first ``-1``; vLLM
        keeps the top-k's ``-1`` padding before the block, with the same
        valid keys. vLLM drops top-k positions at or past the anchor, which a
        causal top-k at ``anchor - 1`` never holds.

        Args:
            target_topk: ``[batch, target_topk]`` int32 physical rows of the
                target layer's top-k at each request's last accepted token,
                valid rows first, as the indexer returns them.
            kv_collection: The target's MLA cache; ``cache_lengths`` is each
                request's anchor, and its pages hold ``anchor + W`` tokens.
        """
        batch, topk = target_topk.shape
        w, page = self.width, self.page_size
        device = target_topk.mesh
        anchor = F.rebind(
            kv_collection.cache_lengths.cast(DType.int32).reshape([-1, 1]),
            [batch, 1],
        )
        # The block's physical rows, [batch, W].
        positions = anchor + F.range(0, w, dtype=DType.int32, device=device)
        lookup = kv_collection.lookup_table
        pages = F.gather_nd(
            F.rebind(lookup, [batch, lookup.shape[1]]).cast(DType.int32),
            (positions // page).unsqueeze(-1),
            batch_dims=1,
        )
        block = pages * page + positions % page

        valid = F.sum((target_topk >= 0).cast(DType.int32), axis=-1)
        cols = F.range(0, self.index_topk, dtype=DType.int32, device=device)
        # [batch, W, index_topk]: the block offset of each column.
        k = F.broadcast_to(
            cols.reshape([1, 1, -1]) - valid.reshape([-1, 1, 1]),
            [batch, w, self.index_topk],
        )
        from_block = F.gather_nd(
            block, F.clamp(k, 0, w - 1).reshape([batch, -1, 1]), batch_dims=1
        ).reshape([batch, w, self.index_topk])
        history = F.concat(
            [
                target_topk,
                F.full(
                    [batch, self.index_topk - int(topk)],
                    -1,
                    dtype=DType.int32,
                    device=device,
                ),
            ],
            axis=1,
        ).unsqueeze(1)
        query = F.range(0, w, dtype=DType.int32, device=device).reshape(
            [1, -1, 1]
        )
        rows = F.where(k < 0, history, F.where(k <= query, from_block, -1))
        return rows.cast(DType.int32).reshape([-1, self.index_topk])

    def forward(
        self,
        embeds: Tensor,
        target_topk: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        token_positions: Tensor,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
    ) -> Tensor:
        """Returns the final-normalized ``[batch * W, hidden]`` block states.

        Args:
            embeds: ``[batch * W, hidden]`` block embeddings, slot 0 fused.
            target_topk: ``[layers, batch, target_topk]`` int32 physical
                rows, one top-k per drafter layer, as :meth:`_sparse_rows`
                takes them.
            kv_collection: The target's MLA cache.
            freqs_cis: The target's rope table.
            token_positions: ``[3, batch * W]`` M-RoPE positions.
            input_row_offsets: ``[batch + 1]`` row offsets, W apart.
            batch_context_length: The host prefill buffer length, as
                ``HYV4TextModel.streams`` takes it.
        """
        freqs_cis, token_rope = mla_rope_tables(
            freqs_cis, token_positions, self.mrope_section
        )
        # A subgraph signature takes named dims only, not batch * W.
        h = F.rebind(embeds, [token_positions.shape[1], embeds.shape[1]])
        for i, layer in enumerate(self.layers):
            # Every drafter layer has the same kind, so they share a
            # subgraph and take their cache slot as an input.
            h, _ = as_subgraph(layer, name="dfly_layer")(
                h,
                _slot(self.kv_layers[i]),
                _slot(0),
                kv_collection,
                # Unread: a layer without an indexer has no indexer cache.
                kv_collection,
                freqs_cis,
                input_row_offsets,
                F.rebind(
                    self._sparse_rows(target_topk[i], kv_collection),
                    [h.shape[0], self.index_topk],
                ),
                batch_context_length,
                token_rope,
            )
        return self.final_norm(h)
