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
"""The DeepSeek-V3.2 NextN (multi-token prediction) draft layer, on ModuleV3."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import mtp_eh_norm
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.nn.comm.ep import EPBatchManager, EPCommBuffers

from .layers.transformer_block import DeepseekV3_2TransformerBlock
from .model_config import DeepseekV3_2Config


class DeepseekV3_2NextN(Module[..., tuple[Tensor, Tensor]]):
    """One sparse decoder layer fed by an embedding and a target hidden state.

    The draft shares the target's embedding and ``lm_head``, so it holds
    neither: it takes the embedded tokens and returns the decoder output, and
    the caller projects that through :attr:`shared_head_norm`.
    """

    def __init__(
        self,
        config: DeepseekV3_2Config,
        attention_scale: float,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        if config.indexer_types:
            raise ValueError(
                "The NextN draft computes its own top-k at step 0, so its"
                " config must leave indexer_types empty."
            )
        self.config = config
        self.ep_batch_manager = ep_batch_manager
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        # At or past ``first_k_dense_replace``, so the layer is a MoE layer.
        self.decoder_layer = DeepseekV3_2TransformerBlock(
            config=config,
            layer_idx=max(
                config.num_hidden_layers, config.first_k_dense_replace
            ),
            attention_scale=attention_scale,
            ep_batch_manager=ep_batch_manager,
        )
        self.shared_head_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        embedded: Tensor,
        hidden: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
        comm_buffers: EPCommBuffers | None = None,
        prev_topk_indices: Tensor | None = None,
        reuse_prev_topk: bool = False,
        selection_lag: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Returns the decoder output and its top-k selection.

        Args:
            embedded: The tokens' embeddings, ``[seq_len, hidden_size]``.
            hidden: The target's hidden state at the same rows.
            kv_collection: The draft's single-layer MLA cache.
            indexer_kv_collection: The draft's single-layer indexer cache.
            input_row_offsets: Offsets of each request's rows.
            freqs_cis: The rotary embedding table.
            mla_prefill_metadata: The MLA prefill plan, or ``None`` in
                decode.
            comm_buffers: The expert-parallel buffers, or ``None``.
            prev_topk_indices: A top-k selection to reuse, or ``None``.
            reuse_prev_topk: Whether to reuse ``prev_topk_indices`` rather
                than run the indexer.
            selection_lag: The positions between the token a reused
                selection was computed for and this call's token.
        """
        dtype = embedded.dtype
        # The enorm and hnorm normalizations and their concatenation, fused.
        normed = mtp_eh_norm(
            embedded,
            F.rebind(hidden, embedded.shape),
            self.enorm.weight.cast(dtype),
            self.hnorm.weight.cast(dtype),
            self.enorm.eps,
        ).rebind_mapping(embedded.mapping)
        h, topk_indices = self.decoder_layer(
            # The draft's caches hold this one layer.
            F.constant(0, DType.uint32, device=CPU()),
            self.eh_proj(normed),
            kv_collection,
            indexer_kv_collection,
            input_row_offsets,
            freqs_cis,
            mla_prefill_metadata,
            comm_buffers,
            prev_topk_indices,
            reuse_prev_topk,
            selection_lag=selection_lag,
        )
        return h, topk_indices
