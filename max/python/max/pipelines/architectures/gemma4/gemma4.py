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

"""Implements the Gemma4 model."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, ShardingStrategy, TensorValue, ops
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.moe.moe import MoE
from max.nn.moe.moe_fp8 import MoEQuantized
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from .layers.attention import Gemma4Attention
from .layers.decoder_layer import Gemma4TextDecoderLayer
from .layers.moe import Gemma4MoEGate
from .layers.rms_norm import Gemma4RMSNorm
from .layers.rotary_embedding import ProportionalRotaryEmbedding
from .model_config import Gemma4ForConditionalGenerationConfig

# Map from layer type string to the index in MultiKVCacheParams.params.
_LAYER_TYPE_TO_KV_INDEX = {
    "sliding_attention": 0,
    "full_attention": 1,
}


class Gemma4TextModel(DistributedLogitsPostprocessMixin, Module):
    """The Gemma 4 language model."""

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        text_config = config.text_config
        self.devices = config.devices

        # Build per-layer-type rotary embeddings.
        # sliding_attention uses default rope, full_attention uses proportional.
        rope_sliding = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.sliding_window_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,
        )

        rope_global = ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.global_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=text_config.global_rope_scaling,
        )

        unquantized_dtype = config.unquantized_dtype

        embedding_output_dtype = config.dtype
        quant_config = text_config.quant_config
        if quant_config and quant_config.embedding_output_dtype:
            embedding_output_dtype = quant_config.embedding_output_dtype

        self.embed_tokens = ScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            embedding_output_dtype,
            config.devices,
            embed_scale=text_config.hidden_size**0.5,
        )

        self.norm = Gemma4RMSNorm(
            text_config.hidden_size,
            unquantized_dtype,
            text_config.rms_norm_eps,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = ColumnParallelLinear(
            text_config.hidden_size,
            text_config.vocab_size,
            dtype=unquantized_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        # Resolve per-layer KVCacheParams from MultiKVCacheParams.
        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_layer_type: dict[str, KVCacheParams] = {
            layer_type: config.kv_params.params[kv_idx]
            for layer_type, kv_idx in _LAYER_TYPE_TO_KV_INDEX.items()
        }

        layer_type_counts: dict[str, int] = {
            "sliding_attention": 0,
            "full_attention": 0,
        }
        layers = []
        for i in range(text_config.num_hidden_layers):
            layer_type = text_config.layer_types[i]
            kv_params = kv_params_by_layer_type[layer_type]

            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            is_sliding = layer_type == "sliding_attention"

            is_nvfp4 = quant_config is not None and quant_config.is_nvfp4
            moe_nvfp4 = is_nvfp4 and text_config.enable_moe_block

            moe_block: MoE | None = None
            if text_config.enable_moe_block:
                moe_gate_cls = functools.partial(
                    Gemma4MoEGate,
                    eps=text_config.rms_norm_eps,
                )
                moe_norm_cls = functools.partial(
                    Gemma4RMSNorm,
                    text_config.hidden_size,
                    unquantized_dtype,
                    eps=text_config.rms_norm_eps,
                )
                if is_nvfp4:
                    moe_block = MoEQuantized(
                        devices=config.devices,
                        hidden_dim=text_config.hidden_size,
                        num_experts=text_config.num_experts,
                        num_experts_per_token=text_config.top_k_experts,
                        moe_dim=text_config.moe_intermediate_size,
                        gate_cls=moe_gate_cls,
                        gate_activation="gelu_tanh",
                        pre_expert_norm_cls=moe_norm_cls,
                        dtype=config.dtype,
                        quant_config=quant_config,
                    )
                else:
                    moe_block = MoE(
                        devices=config.devices,
                        hidden_dim=text_config.hidden_size,
                        num_experts=text_config.num_experts,
                        num_experts_per_token=text_config.top_k_experts,
                        moe_dim=text_config.moe_intermediate_size,
                        gate_cls=moe_gate_cls,
                        gate_activation="gelu_tanh",
                        pre_expert_norm_cls=moe_norm_cls,
                        dtype=config.dtype,
                    )

            layers.append(
                Gemma4TextDecoderLayer(
                    attention=Gemma4Attention(
                        rope_global=rope_global,
                        rope_local=rope_sliding,
                        num_attention_heads=text_config.num_attention_heads,
                        num_key_value_heads=text_config.num_key_value_heads,
                        num_global_key_value_heads=text_config.num_global_key_value_heads,
                        attention_k_eq_v=text_config.attention_k_eq_v,
                        hidden_size=text_config.hidden_size,
                        kv_params=kv_params,
                        global_head_dim=text_config.global_head_dim,
                        layer_idx=i,
                        layer_idx_in_cache=layer_idx_in_cache,
                        is_sliding=is_sliding,
                        dtype=unquantized_dtype if is_nvfp4 else config.dtype,
                        devices=config.devices,
                        qk_norm_eps=text_config.rms_norm_eps,
                        local_window_size=text_config.sliding_window,
                        quant_config=None if is_nvfp4 else quant_config,
                    ),
                    mlp=MLP(
                        dtype=unquantized_dtype if moe_nvfp4 else config.dtype,
                        quantization_encoding=None,
                        hidden_dim=text_config.hidden_size,
                        feed_forward_length=text_config.intermediate_size,
                        devices=config.devices,
                        activation_function=text_config.hidden_activation,
                        quant_config=None if moe_nvfp4 else quant_config,
                    ),
                    hidden_size=text_config.hidden_size,
                    rms_norm_eps=text_config.rms_norm_eps,
                    devices=config.devices,
                    unquantized_dtype=unquantized_dtype,
                    enable_moe_block=text_config.enable_moe_block,
                    moe_block=moe_block,
                )
            )

        # Store per-layer mapping to kv collection index so __call__ can
        # route the correct cache to each layer.
        self._layer_kv_index = [
            _LAYER_TYPE_TO_KV_INDEX[text_config.layer_types[i]]
            for i in range(text_config.num_hidden_layers)
        ]

        self.dim = text_config.hidden_size
        self.n_heads = text_config.num_attention_heads
        self.layers = LayerList(layers)
        self.kv_params = config.kv_params
        self.return_logits = text_config.return_logits

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue],
        image_token_indices: Sequence[TensorValue],
        **kwargs: object,
    ) -> tuple[TensorValue, ...]:
        kv_collections_by_type = [
            sliding_kv_collections,
            global_kv_collections,
        ]

        h = self.embed_tokens(tokens, signal_buffers)

        h = [
            merge_multimodal_embeddings(
                inputs_embeds=h_device,
                multimodal_embeddings=img_embed,
                image_token_indices=img_tok_indices,
            )
            for h_device, img_embed, img_tok_indices in zip(
                h, image_embeddings, image_token_indices, strict=True
            )
        ]

        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            kv_collections = kv_collections_by_type[self._layer_kv_index[idx]]
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
                **kwargs,
            )

        return self._postprocess_logits(
            h, input_row_offsets, return_n_logits, signal_buffers
        )
