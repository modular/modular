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
"""The Hy4 MTP draft layer (ModuleV3)."""

from __future__ import annotations

from dataclasses import replace

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, Linear
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.nn.kv_cache import MultiKVCacheParams

from ..hy_v4.layers.decoder import HYV4DecoderLayer
from ..hy_v4.model_config import HYV4Config

__all__ = ["HYV4MultiTokenPredictor", "draft_config"]


def draft_config(
    config: HYV4Config, kv_params: MultiKVCacheParams
) -> HYV4Config:
    """The target config narrowed to the one MTP layer and its own caches.

    The layer is a ``full`` indexer layer with a sparse MLP, at slot 0 of
    single-layer MLA and indexer caches.
    """
    return replace(
        config,
        kv_params=kv_params,
        num_hidden_layers=1,
        indexer_types=["full"],
        mlp_layer_types=["sparse"],
    )


def _zero_first_positions(
    embeds: Tensor, input_row_offsets: Tensor, cache_lengths: Tensor
) -> Tensor:
    """Zeroes the rows at sequence position 0: a request's first token on an
    empty cache."""
    starts = F.cast(input_row_offsets[:-1], DType.int64)
    keep = F.rebind(F.cast(cache_lengths != 0, embeds.dtype), starts.shape)
    keep = keep.unsqueeze(-1)
    first = F.gather(embeds, starts, axis=0) * keep
    return F.scatter_nd(embeds, first, starts.unsqueeze(-1))


class HYV4MultiTokenPredictor(HYV4DecoderLayer):
    """``enorm``/``hnorm``, ``eh_proj``, one Hy4 block without iHC, and
    ``final_layernorm``. ``embed_tokens`` and ``lm_head`` are the target's.

    The block's own sublayers sit at the top level, where the checkpoint's
    ``model.mtp_layers.0.*`` names put them; :meth:`predict` is the draft's
    entry point and the inherited ``forward`` is the bare block.

    Args:
        config: The draft config, from :func:`draft_config`.
    """

    def __init__(self, config: HYV4Config) -> None:
        super().__init__(config, 0, ihc=False)
        hidden = config.hidden_size
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.eh_proj = Linear(2 * hidden, hidden, bias=False)
        self.final_layernorm = RMSNorm(hidden, config.rms_norm_eps)

    def predict(
        self,
        embed_tokens: Embedding,
        tokens: Tensor,
        hidden: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        prev_topk_indices: Tensor | None,
        batch_context_length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns the normalized draft hidden state and the top-k it used.

        Args:
            embed_tokens: The target's embedding.
            tokens: ``[tokens]`` the token after each position.
            hidden: ``[tokens, hidden]`` the target's final hidden state, or
                the previous draft step's output.
            kv_collection: The draft's MLA cache.
            indexer_kv_collection: The draft's indexer K cache.
            freqs_cis: The target's rope table.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            prev_topk_indices: Step 0's top-k. Given, the indexer is skipped
                and the block attends to these positions (R3 ``set_skip_topk``).
            batch_context_length: The host prefill buffer length, as
                ``HYV4TextModel.streams`` takes it.
        """
        embeds = F.rebind(embed_tokens(tokens), hidden.shape)
        # R3 mtp.py:500: position 0 has no previous token to predict from.
        embeds = _zero_first_positions(
            embeds, input_row_offsets, kv_collection.cache_lengths
        )
        h = self.eh_proj(
            F.concat([self.enorm(embeds), self.hnorm(hidden)], axis=-1)
        )
        indexer = self.self_attn.indexer
        indexer.skip_topk = prev_topk_indices is not None
        try:
            # The draft caches hold its one layer at slot 0.
            slot = F.constant(0, DType.uint32, device=CPU())
            h, topk = self(
                h,
                slot,
                slot,
                kv_collection,
                indexer_kv_collection,
                freqs_cis,
                input_row_offsets,
                prev_topk_indices,
                batch_context_length,
            )
        finally:
            indexer.skip_topk = False
        return self.final_layernorm(h), topk
