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
"""Build a Llama3 model that runs on multiple devices."""

from __future__ import annotations

import functools
import logging

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    ColumnParallelLinear,
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    DistributedTransformer,
    DistributedTransformerBlock,
    Linear,
    VocabParallelEmbedding,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)

logger = logging.getLogger("max.pipelines")
from .model_config import Llama3Config, create_rope_embedding


class DistributedLlama3(DistributedTransformer):
    def __init__(self, config: Llama3Config) -> None:
        assert len(config.devices) > 1

        if config.quantization_config:
            raise ValueError(
                "Model contains GPTQ weights. This is currently not supported with multiple GPUs."
            )

        if config.stacked_mlp:
            raise ValueError(
                "Model contains stacked MLP weights. This is currently not supported with multiple GPUs."
            )

        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError(
                "`norm_method` must be `RMSNorm` and `rms_norm_eps` cannot be "
                "None for model that uses `RMSNorm`."
            )

        rope = create_rope_embedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=config.interleaved_rope_weights,
            rope_scaling_params=config.rope_scaling_params,
            longrope_scaling_params=config.longrope_scaling_params,
            device=config.devices[0],
        )

        create_distributed_norm = functools.partial(
            DistributedRMSNorm,
            dim=config.hidden_size,
            dtype=config.norm_dtype or config.dtype,
            eps=config.rms_norm_eps,
            devices=config.devices,
        )

        fp8_cfg = config.float8_config
        linear_cls = functools.partial(Linear, float8_config=fp8_cfg)

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            # Deal with the float8 case where individual layers are ignored
            # specially: assume bfloat16 dtype for "ignored" layers in fp8
            # quantized models.
            attn_qkv_dtype = (
                DType.bfloat16
                if fp8_cfg and layer_idx not in fp8_cfg.attn_qkv_in_float8
                else config.dtype
            )
            mlp_dtype = (
                DType.bfloat16
                if fp8_cfg and layer_idx not in fp8_cfg.mlp_in_float8
                else config.dtype
            )
            layers.append(
                DistributedTransformerBlock(
                    attention=DistributedAttentionWithRope(
                        stacked_qkv=config.stacked_qkv,
                        scale=config.attention_multiplier,
                        clip_qkv=config.clip_qkv,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        dtype=attn_qkv_dtype,
                        rope=rope,
                        linear_cls=linear_cls,
                        devices=config.devices,
                        has_bias=config.attention_bias,
                        # Only pass the float8 config if this attention layer is quantized.
                        float8_config=(
                            fp8_cfg
                            if fp8_cfg
                            and (layer_idx in fp8_cfg.attn_qkv_in_float8)
                            else None
                        ),
                    ),
                    mlp=DistributedMLP(
                        mlp_dtype,
                        config.model_quantization_encoding,
                        config.hidden_size,
                        config.intermediate_size,
                        config.devices,
                        linear_cls,
                        # Only pass the float8 config if this MLP layer is quantized.
                        float8_config=(
                            fp8_cfg
                            if fp8_cfg and (layer_idx in fp8_cfg.mlp_in_float8)
                            else None
                        ),
                    ),
                    attention_norm=create_distributed_norm(),
                    mlp_norm=create_distributed_norm(),
                    devices=config.devices,
                    # TODO: Support residual_multiplier
                    # residual_multiplier=config.residual_multiplier,
                )
            )

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if fp8_cfg and fp8_cfg.embedding_output_dtype:
            embedding_output_dtype = fp8_cfg.embedding_output_dtype

        embedding_layer = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=embedding_output_quantization,
        )
        output = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            devices=config.devices,
            tied_weight=(
                embedding_layer.weight if config.tie_word_embeddings else None
            ),
            quantization_encoding=embedding_output_quantization,
        )

        kv_collection_cls: (
            type[FetchContinuousBatchingKVCacheCollection]
            | type[FetchPagedKVCacheCollection]
        )
        if config.kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy "
                + str(config.kv_params.cache_strategy)
            )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_distributed_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            devices=config.devices,
            return_logits=config.return_logits,
            use_subgraphs=config.use_subgraphs,
            # TODO: Support the following config options.
            # embedding_multiplier=config.embedding_multiplier,
            # logits_postprocessor=config.logits_postprocessor,
        )
