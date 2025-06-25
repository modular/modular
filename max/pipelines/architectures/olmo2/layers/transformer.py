# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from max.graph import TensorValue
from max.nn.attention.interfaces import AttentionImpl, AttentionImplQKV
from max.nn.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    PagedKVCacheCollection,
)
from max.nn.layer import Layer, Module


class Olmo2TransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers.

    Llama3 Transformer block: Norm -> Attention -> Norm -> MLP
    Olmo2 Transformer block: Attention -> Norm -> MLP -> Norm
    """

    def __init__(
        self,
        attention: AttentionImpl | AttentionImplQKV | Module,
        mlp: Layer,
        post_attention_layer_norm: Layer,
        post_feedforward_layer_norm: Layer,
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp

        self.post_attention_layernorm = post_attention_layer_norm
        self.post_feedforward_layernorm = post_feedforward_layer_norm
        self.residual_multiplier = residual_multiplier

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        h = self.self_attn(
            layer_idx,
            x,
            kv_collection,
            input_row_offsets,
        )
        h = self.post_attention_layernorm(h)

        h = x + h
        residual = h

        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)

        return h + residual
