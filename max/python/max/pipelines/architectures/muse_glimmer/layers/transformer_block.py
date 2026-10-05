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

"""Muse Glimmer transformer block for the ModuleV3 API."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.norm import GemmaRMSNorm
from max.experimental.tensor import Tensor


class MuseGlimmerTransformerBlock(Module[..., Tensor]):
    """Muse Glimmer decoder layer: ``(1 + w)`` sandwich norms around attention and MLP."""

    def __init__(
        self,
        attention: Module[..., Tensor],
        mlp: Module[[Tensor], Tensor],
        hidden_size: int,
        rms_norm_eps: float,
        post_norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            hidden_size, eps=post_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            hidden_size, eps=post_norm_eps
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
        )
        h = x + self.post_attention_layernorm(attn_out)
        mlp_out = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(mlp_out)
