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
"""EAGLE draft model extending Llama3 with hidden state fusion."""

from __future__ import annotations

from max.graph import TensorValue, TensorValueLike, ops
from max.nn.identity import Identity
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.transformer import TransformerBlock

from ..llama3.llama3 import Llama3
from ..llama3.model_config import Llama3Config


class EagleLlama3(Llama3):
    """EAGLE draft model that extends Llama3 with hidden state fusion."""

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config)

        self.fc = Linear(
            config.hidden_size * 2,
            config.hidden_size,
            config.dtype,
            config.devices[0],
            has_bias=True,
        )

        # Replace norms with Identity for EAGLE
        # The draft model receives already-normalized hidden states from target
        identity = Identity()

        # Replace first layer's input norm (receives target's normalized states)
        assert isinstance(self.layers[0], TransformerBlock)
        assert isinstance(self.layers[0].input_layernorm, Module)
        self.layers[0].input_layernorm = identity

        # Replace final norm before lm_head
        self.norm = identity

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        hidden_states: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        assert isinstance(hidden_states, TensorValue)
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )
        fused = ops.concat([h, hidden_states], axis=-1)
        h = self.fc(fused)

        return self._process_hidden_states(
            h, kv_collection, return_n_logits, input_row_offsets
        )
