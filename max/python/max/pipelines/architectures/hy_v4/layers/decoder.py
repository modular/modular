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
"""The Hy4 decoder layer (ModuleV3)."""

from __future__ import annotations

from collections.abc import Callable

from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor

from ..model_config import HYV4Config
from .attention import HYV4Attention
from .hyper_connection import HYV4HyperConnection
from .mlp import HYV4MLP
from .moe import HYV4MoE

__all__ = ["HYV4DecoderLayer"]


class HYV4DecoderLayer(Module[..., tuple[Tensor, Tensor]]):
    """Attention then MLP or MoE, each inside a residual site.

    With iHC a site is ``hc -> norm -> sublayer -> write_back`` over
    ``[tokens, hc_mult, hidden]`` streams; without it (the MTP block) the
    residual is a plain ``[tokens, hidden]`` sum.

    Args:
        config: The model config.
        layer_idx: The decoder layer, which picks the indexer and MLP types.
        ihc: Whether the layer carries iHC streams.
    """

    def __init__(
        self,
        config: HYV4Config,
        layer_idx: int,
        *,
        ihc: bool = True,
    ) -> None:
        super().__init__()
        self.ihc = ihc
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn = HYV4Attention(config, layer_idx)
        self.mlp: Module[[Tensor], Tensor] = (
            HYV4MoE(config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else HYV4MLP(
                config.hidden_size,
                config.intermediate_size,
                config.quant_config,
            )
        )
        if ihc:
            self.hc_attn = HYV4HyperConnection(config)
            self.hc_mlp = HYV4HyperConnection(config)

    def shard(self) -> None:
        """Splits attention by heads and the routed experts by expert over the
        mesh the layer was built on; the rest stays replicated."""
        self.self_attn.shard_heads()
        if isinstance(self.mlp, HYV4MoE):
            self.mlp.shard_experts()

    def forward(
        self,
        h: Tensor,
        layer_idx: Tensor,
        indexer_layer: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        prev_topk_indices: Tensor | None,
        buffer_length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns the updated residual and the attention's top-k positions.

        ``layer_idx`` and ``indexer_layer`` are the layer's host-side cache
        slots, as :meth:`HYV4Attention.forward` takes them.
        """
        topk: Tensor | None = None

        def attn(x: Tensor) -> Tensor:
            nonlocal topk
            y, topk = self.self_attn(
                x,
                layer_idx,
                indexer_layer,
                kv_collection,
                indexer_kv_collection,
                freqs_cis,
                input_row_offsets,
                prev_topk_indices,
                buffer_length,
            )
            return y

        h = self._site(
            h, self.input_layernorm, attn, self.hc_attn if self.ihc else None
        )
        h = self._site(
            h,
            self.post_attention_layernorm,
            self.mlp,
            self.hc_mlp if self.ihc else None,
        )
        assert topk is not None
        return h, topk

    @staticmethod
    def _site(
        h: Tensor,
        norm: RMSNorm,
        sublayer: Callable[[Tensor], Tensor],
        hc: HYV4HyperConnection | None,
    ) -> Tensor:
        if hc is None:
            return h + sublayer(norm(h))
        post, x = hc(h)
        return hc.write_back(h, sublayer(norm(x)), post)
