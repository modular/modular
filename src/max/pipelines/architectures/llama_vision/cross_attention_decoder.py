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

"""Llama 3.2 Transformer Vision Language Model cross attention decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import MLP, RMSNorm
from max.nn.kernels import (
    MHAMaskVariant,
    cross_attention_ragged,
    matmul_kv_cache_ragged,
    rms_norm_key_cache,
)
from max.nn.kv_cache import ContinuousBatchingKVCacheCollection, KVCacheParams
from max.nn.layer import Layer
from max.nn.linear import Linear


@dataclass
class CrossSdpaAttention(Layer):
    """Cross attention layer using SDPA (Scaled Dot Product Attention)."""

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    layer_idx: int
    """Index into the cross attention layers' KV cache."""

    q_proj: Linear
    """A linear layer for the query projection."""

    wk: Weight
    """The k weight vector. Combines with wv to form a Linear."""

    wv: Weight
    """The v weight vector. Combines with wk to form a Linear."""

    o_proj: Linear
    """A linear layer for the output projection."""

    q_norm: RMSNorm
    """Layer normalization."""

    k_norm: RMSNorm
    """Layer normalization."""

    def __call__(
        self,
        hidden_states: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_attention_states: TensorValue,
        cross_input_row_offsets: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection,
    ) -> TensorValue:
        """Computes attention on hidden (query) and cross (key and value).

        Returns:
            Attended hidden activation.
        """
        wkv = ops.concat((self.wk, self.wv), axis=0)

        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(
            [
                -1,
                self.n_heads,
                self.kv_params.head_dim,
            ]
        )
        query_states = self.q_norm(query_states)

        matmul_kv_cache_ragged(
            kv_params=self.kv_params,
            # Here, hidden_states correspond to cross_attention_states.
            hidden_states=cross_attention_states,
            layer_idx=self.layer_idx,
            input_row_offsets=cross_input_row_offsets,
            weight=wkv,
            kv_collection=kv_collection,
        )
        rms_norm_key_cache(
            self.kv_params,
            kv_collection,
            gamma=ops.cast(self.k_norm.weight, hidden_states.dtype),
            epsilon=self.k_norm.eps,
            layer_idx=self.layer_idx,
            # Use the total sequence length of the cross attention states.
            total_seq_len=cross_attention_states.shape[0],
            input_row_offsets=cross_input_row_offsets,
        )

        # Calculate Flash Attention.
        attn_out = cross_attention_ragged(
            self.kv_params,
            input=query_states,
            kv_collection=kv_collection,
            layer_idx=ops.constant(
                self.layer_idx, DType.uint32, device=DeviceRef.CPU()
            ),
            input_row_offsets=hidden_input_row_offsets,
            # Use the null mask to attend to all vision tokens.
            mask_variant=MHAMaskVariant.NULL_MASK,
            kv_input_row_offsets=cross_input_row_offsets,
            q_max_seq_len=hidden_max_seq_len,
            scale=math.sqrt(1.0 / self.kv_params.head_dim),
        )

        # Reshape back to (hidden total seq len, hidden size).
        attn_out = ops.reshape(attn_out, shape=[hidden_states.shape[0], -1])

        return self.o_proj(attn_out)


@dataclass
class CrossAttentionDecoderLayer(Layer):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    cross_attn: CrossSdpaAttention
    input_layernorm: RMSNorm
    cross_attn_attn_gate: Weight
    mlp: MLP
    post_attention_layernorm: RMSNorm
    cross_attn_mlp_gate: Weight

    def __call__(
        self,
        hidden_states: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_attention_states: TensorValue,
        cross_input_row_offsets: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection,
    ) -> TensorValue:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states,
            hidden_input_row_offsets,
            hidden_max_seq_len,
            cross_attention_states,
            cross_input_row_offsets,
            kv_collection,
        )
        hidden_states = (
            residual + ops.tanh(self.cross_attn_attn_gate) * hidden_states
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + ops.tanh(self.cross_attn_mlp_gate) * hidden_states
