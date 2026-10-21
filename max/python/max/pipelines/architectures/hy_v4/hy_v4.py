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
"""The Hy4 model (ModuleV3)."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.compilation import as_subgraph
from max.experimental.nn import Embedding, Linear, Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib import ModuleV3Outputs

from .layers.decoder import HYV4DecoderLayer
from .layers.hyper_connection import HYV4HyperHead
from .model_config import HYV4Config

__all__ = ["HYV4"]


def _slot(index: int) -> Tensor:
    return F.constant(index, DType.uint32, device=CPU())


class HYV4TextModel(Module[..., Tensor]):
    """Embedding, iHC decoder stack, iHC head, norm and fp32 ``lm_head``."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.rope = RotaryEmbedding(
            dim=config.qk_rope_head_dim,
            n_heads=1,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
        )
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList[HYV4DecoderLayer](
            HYV4DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.layer_kinds = tuple(
            zip(config.indexer_types, config.mlp_layer_types, strict=True)
        )
        # A shared layer has no indexer cache slot; it is passed 0 unread.
        self.indexer_slots = tuple(
            config.indexer_cache_layer(i) if kind == "full" else 0
            for i, kind in enumerate(config.indexer_types)
        )
        self.hc_head = HYV4HyperHead(config)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def shard(self) -> None:
        """Splits attention by heads and the routed experts by expert over the
        one-axis mesh the model was built on; everything else, the iHC
        weights, indexers, routers, norms, embedding and ``lm_head`` included,
        stays replicated."""
        for layer in self.layers:
            layer.shard()

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
    ) -> Tensor:
        streams = self.streams(
            tokens,
            kv_collection,
            indexer_kv_collection,
            input_row_offsets,
            batch_context_length,
        )
        last = F.gather(streams, input_row_offsets[1:] - 1, axis=0)
        return self.logits(self.head(last))

    def streams(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
    ) -> Tensor:
        """Returns the ``[tokens, hc_mult, hidden]`` iHC streams.

        ``batch_context_length`` is the host ``[1]`` int32 prefill buffer
        length every attention block hands the MLA kernel.
        """
        h = self.embed_tokens(tokens)
        streams = F.broadcast_to(
            h.unsqueeze(1), [h.shape[0], self.hc_mult, h.shape[1]]
        )
        freqs_cis = self.rope.freqs_cis.to(h.mesh)
        # Layers of one indexer and MLP kind share a subgraph, so their cache
        # slots are inputs rather than constants traced from the first one.
        # A shared layer reuses the top-k of the most recent full layer;
        # layer 0 is always full, so no shared layer reads None.
        topk: Tensor | None = None
        for idx, layer in enumerate(self.layers):
            indexer, mlp = self.layer_kinds[idx]
            streams, topk = as_subgraph(layer, name=f"{indexer}_{mlp}_layer")(
                streams,
                _slot(idx),
                _slot(self.indexer_slots[idx]),
                kv_collection,
                indexer_kv_collection,
                freqs_cis,
                input_row_offsets,
                None if indexer == "full" else topk,
                batch_context_length,
            )
        return streams

    def head(self, streams: Tensor) -> Tensor:
        """Collapses the streams to the final normalized hidden state."""
        return self.norm(self.hc_head(streams))

    def logits(self, h: Tensor) -> Tensor:
        return self.lm_head(h).cast(DType.float32)


class HYV4(Module[..., ModuleV3Outputs]):
    """Unflattens the MLA and indexer cache inputs around the text model."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        assert isinstance(config.kv_params, MultiKVCacheParams)
        self.cache_params: list[KVCacheParams] = []
        for params in config.kv_params.children.values():
            assert isinstance(params, KVCacheParams)
            self.cache_params.append(params)
        self.language_model = HYV4TextModel(config)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        del return_n_logits
        mesh = self.language_model.embed_tokens.weight.mesh
        if mesh.num_devices > 1:
            tokens = F.distributed_broadcast(tokens, mesh)
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)
        values = iter(t._graph_value for t in kv_inputs)
        mla, indexer = (
            PagedCacheValues.from_upstream(
                params.unflatten_kv_inputs(values), tokens.mapping
            )
            for params in self.cache_params
        )
        logits = self.language_model(
            tokens, mla, indexer, input_row_offsets, batch_context_length
        )
        return ModuleV3Outputs(next_token_logits=logits.to(mesh.devices[0]))
