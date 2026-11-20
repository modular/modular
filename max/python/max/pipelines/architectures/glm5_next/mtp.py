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
"""GLM-5.3-Flash's MTP draft layer, on ModuleV3.

The checkpoint's layer ``num_hidden_layers`` is the target's own sparse-MLA
block and MoE with a plain residual in place of the hyper-connections, so it is
built from this package's sublayers. It owns the last slot of the target's
latent, indexer and tail ring caches, which the target's config declares when
speculating.
"""

from __future__ import annotations

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module, RMSNorm
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMapping

from .layers.decoder import feed_forward
from .layers.indexer import KPoolTail
from .layers.mlp import build_mlp
from .layers.sparse_mla import Glm5NextSparseMLA
from .model_config import Glm5NextConfig

__all__ = ["Glm5NextMTP"]


class Glm5NextMTPLayer(Module[..., tuple[Tensor, Tensor]]):
    """The draft's decoder layer: sparse MLA and MoE, pre-norm residual."""

    def __init__(
        self,
        config: Glm5NextConfig,
        ep_batch_manager: EPBatchManager | None,
    ) -> None:
        layer_idx = config.num_hidden_layers
        quant_scheme = config.quant_scheme
        if quant_scheme is None:
            raise ValueError(
                "GLM-5.3-Flash's MTP layer needs a resolved quant scheme: its"
                " precision map is read off the checkpoint's weight scales."
            )
        self.self_attn = Glm5NextSparseMLA(
            config, layer_idx, quant_scheme.attn_config(layer_idx)
        )
        self.mlp = build_mlp(config, layer_idx, ep_batch_manager)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        tail: KPoolTail,
        comm_buffers: EPCommBuffers | None,
        prev_selection: Tensor | None,
        selection_lag: int,
    ) -> tuple[Tensor, Tensor]:
        y, selection = self.self_attn.attend(
            self.input_layernorm(x),
            kv_collection,
            indexer_kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            tail,
            prev_selection=prev_selection,
            selection_lag=selection_lag,
        )
        # The attention output is a sum of per-device partials.
        h = x + y.to(DeviceMapping.replicated(x.mesh))
        y = feed_forward(
            self.mlp, self.post_attention_layernorm(h), comm_buffers
        )
        return h + y, selection


class Glm5NextMTP(Module[..., tuple[Tensor, Tensor]]):
    """The draft: ``eh_proj(enorm(embed(t)) ++ hnorm(h))``, one decoder layer.

    It embeds and projects through the target's embedding and head, so it
    holds neither: it takes the embedded tokens and returns the decoder
    output, which the caller projects through :attr:`shared_head_norm`. It
    dispatches its experts through an expert-parallel manager of its own,
    over the buffers its config's ``ep_config`` sizes.

    Raises:
        NotImplementedError: If the checkpoint does not share the draft's
            top-k selection across steps, the only mode implemented.
    """

    def __init__(self, config: Glm5NextConfig) -> None:
        if not config.index_share_for_mtp_iteration:
            raise NotImplementedError(
                "GLM-5.3-Flash MTP reuses its first step's top-k selection at"
                " every later step; index_share_for_mtp_iteration=False is not"
                " implemented"
            )
        self.config = config
        self.ep_batch_manager = (
            None
            if config.ep_config is None
            else EPBatchManager(config.ep_config)
        )
        self.cache_layer_idx = len(config.sparse_attention_layers)
        """The draft's slot in the latent, indexer and tail ring caches, one
        past the target's sparse layers."""
        # Quantized checkpoints keep norms, biases and embeddings in bf16.
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_dtype(dtype):
            self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.eh_proj = Linear(
                2 * config.hidden_size, config.hidden_size, bias=False
            )
            self.decoder_layer = Glm5NextMTPLayer(config, self.ep_batch_manager)
            self.shared_head_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    @property
    def indexer(self) -> Module[..., Tensor]:
        """The draft layer's indexer, which only its first step runs."""
        return self.decoder_layer.self_attn.indexer

    @property
    def selection_width(self) -> int:
        """Width of the token selection the draft's later steps reuse."""
        return self.decoder_layer.self_attn.indexer.selection_width

    def forward(
        self,
        embedded: Tensor,
        hidden: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        tail: KPoolTail,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        comm_buffers: EPCommBuffers | None = None,
        prev_selection: Tensor | None = None,
        selection_lag: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Returns the decoder output and its token selection.

        Args:
            embedded: The tokens' embeddings, ``[rows, hidden_size]``.
            hidden: The target's normalized hidden state at the same rows,
                or the previous step's after :attr:`shared_head_norm`.
            kv_collection: The target's latent cache, whose last layer is
                the draft's.
            indexer_kv_collection: The target's indexer cache, likewise.
            tail: The k-pool tail ring at the draft's layer.
            input_row_offsets: Offsets of each request's rows.
            freqs_cis: The target's padding rotary table.
            comm_buffers: The expert-parallel buffers, or ``None``.
            prev_selection: A selection to reuse rather than run the indexer.
            selection_lag: Positions between the token ``prev_selection`` was
                made for and this call's token.
        """
        # The embedding and the carried hidden state come off different row
        # dims that are equal at runtime.
        x = self.eh_proj(
            F.concat(
                [
                    self.enorm(embedded),
                    self.hnorm(F.rebind(hidden, embedded.shape)),
                ],
                axis=-1,
            )
        )
        return self.decoder_layer(
            x,
            kv_collection,
            indexer_kv_collection,
            F.constant(self.cache_layer_idx, DType.uint32, device=CPU()),
            input_row_offsets,
            freqs_cis,
            tail,
            comm_buffers,
            prev_selection,
            selection_lag,
        )
