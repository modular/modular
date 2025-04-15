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

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn import EmbeddingV2, Module, RMSNormV2


@dataclass
class Qwen2_5VLRotaryEmbedding(Module):
    dim: int
    n_heads: int
    theta: float

    _inv_freq: Optional[TensorValue] = None

    def __post_init__(self):
        super().__init__()

    def _compute_inv_freqs(self) -> TensorValue:
        n = self.dim // self.n_heads
        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        iota = ops.range(
            ops.constant(0, DType.float64),
            ops.constant(n - 1, DType.float64),
            ops.constant(2, DType.float64),
            out_dim=n // 2,
        )
        inv_freq = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
        return inv_freq

    def freqs_cis_base(
        self,
        pos_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Computes the frequency tensor for complex exponentials (cis)
        for a given pos_ids.

        Args:
            pos_ids: of shape [3, batch_size, seq_len]

        Returns:
            Tuple of 2 tensors representing positional embeddings.
            Each tensor is of shape [3, batch_size, seq_len, head_dim]
        """
        # expand inv_freqs from [head_dim // 2] to [2, batch_size, head_dim // 2, 1]
        inv_freq_expanded = ops.unsqueeze(
            ops.unsqueeze(ops.unsqueeze(self.inv_freqs, 0), 0), -1
        )
        inv_freq_expanded = ops.tile(
            inv_freq_expanded, [3, pos_ids.shape[1], 1, 1]
        )
        # expand pos_ids from [3, batch_size, seq_len] to [3, batch_size, 1, seq_len]
        position_ids_expanded = ops.unsqueeze(pos_ids, 2)

        # TODO: maybe cast to float32 before multiplication
        freqs = (
            ops.cast(inv_freq_expanded, DType.float32)
            @ ops.cast(position_ids_expanded, DType.float32)
        ).transpose(2, 3)

        emb = ops.concat((freqs, freqs), -1)
        cos = ops.cos(emb)
        sin = ops.sin(emb)
        return cos, sin

    @cached_property
    def inv_freqs(self) -> TensorValue:
        self._inv_freqs = self._compute_inv_freqs()
        return self._inv_freqs

    def __call__(
        self,
        x: TensorValue,
    ) -> TensorValue:
        raise NotImplementedError


class Transformer(Module):
    """Transformer model consisting for TransformerBlock layers.

    Compared to nn.Transformer:
    - It doesn't have an output linear layer `self.lm_head`
    - It takes `token_embeds` as input rather than tokens. Hence, it doesn't call `self.embed_tokens(tokens)` at the beginning.
    """

    embed_tokens: EmbeddingV2
    rotary_emb: Qwen2_5VLRotaryEmbedding
    norm: RMSNormV2

    def __post_init__(self):
        super().__init__()

    def __call__(
        self,
        inputs_embeds: TensorValue,
        position_ids: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        """Outputs raw hidden states of the transformer model on input `inputs_embeds`.

        Args:
            inputs_embeds: Tensor of text and vision token embeddings of shape (batch_size, seq_len, hidden_size)
                For Qwen2.5VL, shape = (seq_len, 2048)
            attention_mask: Tensor of shape (batch_size, seq_len)
            position_ids: Tensor of position ids for rotary embeddings. These ids are generated for the pre-fill
            phase using data_processing.get_rope_index() shape (3, batch_size, seq_len). For generation phase,
            position_ids shape = (3, batch_size, seq_len=1)

        Returns:
            TensorValue : output of vision transformer projected into the decoder's hidden_size.

        Shapes:
            Input:
                inputs_embeds => (batch_size, seq_len, hidden_size)
                attention_mask => (batch_size, seq_len)
                position_ids => (3, batch_size, seq_len) 3 is hard-coded to represent t, h, w dims of videos.
            Output:

        """
        # expand position_ids to (3, inputs_embeds.shape[0], -1)
        # update causal mask
        position_embeddings = self.rotary_emb.freqs_cis_base(position_ids)
        raise NotImplementedError
