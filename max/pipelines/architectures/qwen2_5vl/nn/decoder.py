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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import (
    Embedding,
    Module,
    RMSNorm,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
)
from max.nn.layer import LayerList


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
            ops.constant(0, DType.float64, device=DeviceRef.CPU()),
            ops.constant(n - 1, DType.float64, device=DeviceRef.CPU()),
            ops.constant(2, DType.float64, device=DeviceRef.CPU()),
            out_dim=n // 2,
            device=DeviceRef.CPU(),
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


class Qwen2_5VLDecoderTransformer(Module):
    """Transformer model consisting for TransformerBlock layers.

    Compared to nn.Transformer:
    - It doesn't have an output linear layer `self.lm_head`
    - It takes `token_embeds` as input rather than tokens. Hence, it doesn't call `self.embed_tokens(tokens)` at the beginning.
    - It has a 3D rotary embedding layer that is applied to all tokens (text and visual) in attention.

    For the TransformerBlock, differences compared to nn.TransformerBlock:
    1. Attention Layer
    2. This is a Callable layer. It is called with inputs: position_ids and position_embedding
        which are passed as input to Attention.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_emb: Qwen2_5VLRotaryEmbedding,
        layers: list[TransformerBlock],
        norm: RMSNorm,
        embedding: Embedding,
        kv_params: KVCacheParams,
        kv_collection_constructor: (
            FetchContinuousBatchingKVCacheCollection
            | FetchPagedKVCacheCollection
        ),
        embedding_multiplier: float = 1.0,
        logits_postprocessor: Callable[[TensorValue], TensorValue]
        | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.rotary_emb = rotary_emb
        self.layers = LayerList(layers)
        self.norm = norm
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.kv_collection_constructor = kv_collection_constructor
        self.embedding_multiplier = embedding_multiplier
        self.logits_postprocessor = logits_postprocessor

    # copied from nn.Transformer
    def _apply_logits_postprocessor(
        self, output: tuple[TensorValue, ...]
    ) -> tuple[TensorValue, ...]:
        if self.logits_postprocessor is None:
            return output
        return tuple(self.logits_postprocessor(elem) for elem in output)

    def __call__(
        self,
        inputs_embeds: TensorValue,
        position_ids: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
        **kwargs,
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
        position_embeddings = self.rotary_emb.freqs_cis_base(position_ids)
        h = inputs_embeds
        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=DeviceRef.CPU()
            )

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)
        cache_lengths = kv_cache_inputs[1]

        input_row_offsets = kwargs["input_row_offsets"]
        prompt_lengths = ops.rebind(
            input_row_offsets[1:] - input_row_offsets[:-1],
            cache_lengths.shape,
        )

        for _, layer in enumerate(self.layers):
            h = layer(
                h,
                position_ids,
                position_embeddings,
                kv_collection,
                **kwargs,
            )

        return self.norm(h)
