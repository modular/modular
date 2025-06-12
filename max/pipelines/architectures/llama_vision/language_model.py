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

"""Llama 3.2 Transformer Vision Language Model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.graph.weights import Weights
from max.nn import (
    MLPV1,
    AttentionWithRopeQKV,
    EmbeddingV1,
    LinearV1,
    OptimizedRotaryEmbedding,
    RMSNormV1,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
)
from max.nn.layer import Layer

from .cross_attention_decoder import (
    CrossAttentionDecoderLayer,
    CrossSdpaAttention,
)


@dataclass
class TextModel(Layer):
    """
    The Llama text model which consists of transformer with self and cross attention layers.
    """

    dtype: DType
    kv_params: KVCacheParams
    vision_kv_params: KVCacheParams
    embed_tokens: EmbeddingV1
    layers: list[CrossAttentionDecoderLayer | SelfAttentionDecoderLayer]
    norm: RMSNormV1
    cross_attention_layers: list[int]

    def __call__(
        self,
        text_kv_cache_inputs: tuple[TensorValue, ...],
        vision_kv_cache_inputs: tuple[TensorValue, ...],
        input_ids: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_attention_states: TensorValue,
        cross_input_row_offsets: TensorValue,
    ) -> TensorValue:
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = ops.cast(inputs_embeds, self.dtype)

        before_attention_blocks_shape = hidden_states.shape

        # Assume that text and vision KV caches have the same KV params for now.
        # So they can share the KV collection constructor object.
        text_kv_collection_constructor = FetchPagedKVCacheCollection(
            self.kv_params
        )
        vision_kv_collection_constructor = FetchPagedKVCacheCollection(
            self.vision_kv_params
        )

        # Construct text and vision KV collections with their distinct inputs.
        text_kv_collection = text_kv_collection_constructor(
            *text_kv_cache_inputs
        )
        vision_kv_collection = vision_kv_collection_constructor(
            *vision_kv_cache_inputs
        )

        for decoder_layer in self.layers:
            # For text-only path we should skip cross attention layers.
            # We expect cross_attention_states to be zeroes if it's a text-only path.
            # The underlying implementation should be a no-op when a zeroed out cross
            # attention states is passed in.

            if isinstance(decoder_layer, CrossAttentionDecoderLayer):
                hidden_states = decoder_layer(
                    hidden_states,
                    hidden_input_row_offsets,
                    hidden_max_seq_len,
                    cross_attention_states,
                    cross_input_row_offsets,
                    vision_kv_collection,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states, text_kv_collection, hidden_input_row_offsets
                )

        assert hidden_states.shape == before_attention_blocks_shape

        return self.norm(hidden_states)


@dataclass
class CausalLanguageModel(Layer):
    """The Llama Vision Text Model with a language modeling head on top."""

    dtype: DType
    kv_params: KVCacheParams
    model: TextModel
    lm_head: LinearV1

    def __call__(
        self,
        text_kv_cache_inputs: tuple[TensorValue, ...],
        vision_kv_cache_inputs: tuple[TensorValue, ...],
        input_ids: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_attention_states: TensorValue,
        cross_input_row_offsets: TensorValue,
    ) -> TensorValue:
        last_hidden_state = self.model(
            text_kv_cache_inputs,
            vision_kv_cache_inputs,
            input_ids,
            hidden_input_row_offsets,
            hidden_max_seq_len,
            cross_attention_states,
            cross_input_row_offsets,
        )

        # For ragged tensors gather the last tokens from packed dim 0.
        last_token_indices = hidden_input_row_offsets[1:] - 1
        last_token_logits = ops.gather(
            last_hidden_state, last_token_indices, axis=0
        )
        return ops.cast(self.lm_head(last_token_logits), self.dtype)  # logits


def cross_attention_decoder_layer(
    dtype: DType,
    num_attention_heads: int,
    hidden_size: int,
    num_key_value_heads: int,
    rms_norm_eps: float,
    kv_params: KVCacheParams,
    vision_kv_params: KVCacheParams,
    intermediate_size: int,
    weights: Weights,
    layer_idx: int,
    device: DeviceRef,
) -> CrossAttentionDecoderLayer:
    head_dim = hidden_size // num_attention_heads
    sdpa_attn = CrossSdpaAttention(
        n_heads=num_attention_heads,
        vision_kv_params=vision_kv_params,
        layer_idx=layer_idx,
        q_proj=LinearV1(
            weights.cross_attn.q_proj.weight.allocate(
                dtype,
                [
                    num_attention_heads * head_dim,
                    hidden_size,
                ],
                device=device,
            ),
            bias=None,
        ),
        wk=weights.cross_attn.k_proj.weight.allocate(
            dtype,
            [
                num_key_value_heads * head_dim,
                hidden_size,
            ],
            device=device,
        ),
        wv=weights.cross_attn.v_proj.weight.allocate(
            dtype,
            [
                num_key_value_heads * head_dim,
                hidden_size,
            ],
            device=device,
        ),
        o_proj=LinearV1(
            weight=weights.cross_attn.o_proj.weight.allocate(
                dtype,
                [
                    hidden_size,
                    num_attention_heads * head_dim,
                ],
                device=device,
            ),
            bias=None,
        ),
        q_norm=RMSNormV1(
            weight=weights.cross_attn.q_norm.weight.allocate(
                dtype, [head_dim], device=device
            ),
            eps=rms_norm_eps,
        ),
        k_norm=RMSNormV1(
            weight=weights.cross_attn.k_norm.weight.allocate(
                dtype, [head_dim], device=device
            ),
            eps=rms_norm_eps,
        ),
    )
    return CrossAttentionDecoderLayer(
        cross_attn=sdpa_attn,
        input_layernorm=RMSNormV1(
            weight=weights.input_layernorm.weight.allocate(
                dtype, [hidden_size], device=device
            ),
            eps=rms_norm_eps,
        ),
        cross_attn_attn_gate=weights.cross_attn_attn_gate.allocate(
            dtype, [1], device=device
        ),
        mlp=MLPV1(
            gate_proj=LinearV1(
                weight=weights.mlp.gate_proj.weight.allocate(
                    dtype,
                    [
                        intermediate_size,
                        hidden_size,
                    ],
                    device=device,
                ),
                bias=None,
            ),
            down_proj=LinearV1(
                weight=weights.mlp.down_proj.weight.allocate(
                    dtype,
                    [
                        hidden_size,
                        intermediate_size,
                    ],
                    device=device,
                ),
                bias=None,
            ),
            up_proj=LinearV1(
                weight=weights.mlp.up_proj.weight.allocate(
                    dtype,
                    [
                        intermediate_size,
                        hidden_size,
                    ],
                    device=device,
                ),
                bias=None,
            ),
        ),
        post_attention_layernorm=RMSNormV1(
            weight=weights.post_attention_layernorm.weight.allocate(
                dtype, [hidden_size], device=device
            ),
            eps=rms_norm_eps,
        ),
        cross_attn_mlp_gate=weights.cross_attn_mlp_gate.allocate(
            dtype, [1], device=device
        ),
    )


class SelfAttentionDecoderLayer(Layer):
    def __init__(self, layer_idx: int, transformer_block: TransformerBlock):
        self.layer_idx = layer_idx
        self.transformer_block = transformer_block

    def __call__(self, *args, **kwargs):
        return self.transformer_block(
            ops.constant(self.layer_idx, DType.uint32, device=DeviceRef.CPU()),
            *args,
            **kwargs,
        )


def self_attention_decoder_layer(
    dtype: DType,
    num_attention_heads: int,
    hidden_size: int,
    num_key_value_heads: int,
    intermediate_size: int,
    rms_norm_eps: float,
    kv_params: KVCacheParams,
    weights: Weights,
    layer_idx: int,
    rotary_embedding: OptimizedRotaryEmbedding,
    device: DeviceRef,
) -> SelfAttentionDecoderLayer:
    head_dim = hidden_size // num_attention_heads

    wq = weights.self_attn.q_proj.weight.allocate(
        dtype,
        shape=[num_attention_heads * head_dim, hidden_size],
        device=device,
    )
    wk = weights.self_attn.k_proj.weight.allocate(
        dtype,
        shape=[num_key_value_heads * head_dim, hidden_size],
        device=device,
    )
    wv = weights.self_attn.v_proj.weight.allocate(
        dtype,
        shape=[num_key_value_heads * head_dim, hidden_size],
        device=device,
    )
    o_proj = LinearV1(
        weight=weights.self_attn.o_proj.weight.allocate(
            dtype,
            shape=[hidden_size, num_attention_heads * head_dim],
            device=device,
        )
    )

    attention = AttentionWithRopeQKV(
        n_heads=num_attention_heads,
        kv_params=kv_params,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=o_proj,
        rope=rotary_embedding,
        scale=math.sqrt(1.0 / head_dim),
    )

    return SelfAttentionDecoderLayer(
        layer_idx=layer_idx,
        transformer_block=TransformerBlock(
            attention=attention,
            mlp=MLPV1(
                gate_proj=LinearV1(
                    weight=weights.mlp.gate_proj.weight.allocate(
                        dtype, [intermediate_size, hidden_size], device=device
                    ),
                    bias=None,
                ),
                down_proj=LinearV1(
                    weight=weights.mlp.down_proj.weight.allocate(
                        dtype, [hidden_size, intermediate_size], device=device
                    ),
                    bias=None,
                ),
                up_proj=LinearV1(
                    weight=weights.mlp.up_proj.weight.allocate(
                        dtype, [intermediate_size, hidden_size], device=device
                    ),
                    bias=None,
                ),
            ),
            attention_norm=RMSNormV1(
                weight=weights.input_layernorm.weight.allocate(
                    dtype, [hidden_size], device=device
                ),
                eps=rms_norm_eps,
            ),
            mlp_norm=RMSNormV1(
                weight=weights.post_attention_layernorm.weight.allocate(
                    dtype, [hidden_size], device=device
                ),
                eps=rms_norm_eps,
            ),
        ),
    )


def instantiate_language_model(
    dtype: DType,
    hidden_size: int,
    n_heads: int,
    rope_theta: int,
    max_seq_len: int,
    num_hidden_layers: int,
    cross_attention_layers: list[int],
    vocab_size: int,
    rms_norm_eps: float,
    num_key_value_heads: int,
    intermediate_size: int,
    kv_params: KVCacheParams,
    vision_kv_params: KVCacheParams,
    weights: Weights,
    device: DeviceRef,
) -> CausalLanguageModel:
    layers: list[CrossAttentionDecoderLayer | SelfAttentionDecoderLayer] = []

    # We don't really have a rotary embedding layer within the graph as it's largely
    # folded into the custom kernel, but leaving this here for now.
    # TODO: this should be Llama3RotaryEmbedding with rope scaling params.
    rotary_embedding = OptimizedRotaryEmbedding(
        dim=hidden_size,
        n_heads=n_heads,
        theta=rope_theta,
        max_seq_len=max_seq_len,
        interleaved=False,
        device=device,
    )

    # Track the cross attention KV cache layer index to compute the self
    # attention KV layer index.
    cross_kv_layer_idx = -1
    for layer_idx in range(num_hidden_layers):
        curr_layer_weight = weights.language_model.model.layers[layer_idx]

        if layer_idx in cross_attention_layers:
            cross_kv_layer_idx = cross_attention_layers.index(layer_idx)
            layers.append(
                cross_attention_decoder_layer(
                    dtype=dtype,
                    num_attention_heads=n_heads,
                    hidden_size=hidden_size,
                    num_key_value_heads=num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                    kv_params=kv_params,
                    vision_kv_params=vision_kv_params,
                    intermediate_size=intermediate_size,
                    weights=curr_layer_weight,
                    layer_idx=cross_kv_layer_idx,
                    device=device,
                )
            )
        else:
            layers.append(
                self_attention_decoder_layer(
                    dtype=dtype,
                    num_attention_heads=n_heads,
                    hidden_size=hidden_size,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    kv_params=kv_params,
                    weights=curr_layer_weight,
                    layer_idx=layer_idx - cross_kv_layer_idx + 1,
                    rotary_embedding=rotary_embedding,
                    device=device,
                )
            )

    text_model = TextModel(
        dtype=dtype,
        kv_params=kv_params,
        vision_kv_params=vision_kv_params,
        embed_tokens=EmbeddingV1(
            weights.language_model.model.embed_tokens.weight.allocate(
                dtype,
                [
                    # Upstream in the Huggingface llama reference, 8 is added to the vocab size.
                    vocab_size + 8,
                    hidden_size,
                ],
                device=device,
            ),
            device=device,
        ),
        norm=RMSNormV1(
            weight=weights.language_model.model.norm.weight.allocate(
                dtype, [hidden_size], device=device
            ),
            eps=rms_norm_eps,
        ),
        layers=layers,
        cross_attention_layers=cross_attention_layers,
        # TODO: Verify if these values passed are even correct.
    )

    return CausalLanguageModel(
        dtype=dtype,
        kv_params=kv_params,
        model=text_model,
        lm_head=LinearV1(
            weights.language_model.lm_head.weight.allocate(
                dtype,
                [
                    vocab_size,
                    hidden_size,
                ],
                device=device,
            ),
            bias=None,
        ),
    )
