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
from typing import Callable

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    AttentionWithRope,
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    DistributedTransformer,
    DistributedTransformerBlock,
    GPTQAttentionWithRope,
    GPTQLinear,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    RMSNorm,
    VocabParallelEmbedding,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    FetchPagedKVCacheCollectionFA3Fallback,
    KVCacheStrategy,
)

from .naive_llama3 import StackedMLP

logger = logging.getLogger("max.pipelines")
from .model_config import Llama3Config


class DistributedLlama3(DistributedTransformer):
    def __init__(self, config: Llama3Config):
        assert len(config.devices) > 1

        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
            device=config.devices[0],
        )

        # Select norm layer class.
        assert config.norm_method == "rms_norm"
        if config.rms_norm_eps is None:
            raise ValueError(
                "rms_norm_eps cannot be None for model that uses RMSNorm."
            )
        create_distributed_norm = functools.partial(
            DistributedRMSNorm,
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            devices=config.devices,
        )
        create_norm = functools.partial(
            RMSNorm,
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]
        if config.quantization_config:
            logger.warning(
                "Model contains GPTQ weights. This is currently not supported with multiple GPUs, and will run on %s.",
                config.devices[0],
            )
            linear_cls = functools.partial(
                GPTQLinear, quantization_config=config.quantization_config
            )
        else:
            linear_cls = Linear

        # Select MLP class.
        mlp_cls: Callable[..., Module]
        if config.stacked_mlp:
            logger.warning(
                "Model contains stacked MLP weights. This is currently not supported with multiple GPUs, and will run on %s.",
                config.devices[0],
            )
            mlp_cls = StackedMLP
        else:
            mlp_cls = DistributedMLP

        # Select attention class.
        attention_cls: Callable[..., AttentionWithRope]
        if config.quantization_config:
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=config.quantization_config,
                scale=config.attention_multiplier,
            )
        else:
            attention_cls = functools.partial(
                DistributedAttentionWithRope,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
            )

        layers = [
            DistributedTransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_distributed_norm(),
                mlp_norm=create_distributed_norm(),
                devices=config.devices,
                # TODO: Support residual_multiplier
                # residual_multiplier=config.residual_multiplier,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None

        embedding_layer = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=embedding_output_quantization,
        )
        # TODO: Shard the output layer.
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        kv_collection_cls: (
            type[FetchContinuousBatchingKVCacheCollection]
            | type[FetchPagedKVCacheCollection]
            | type[FetchPagedKVCacheCollectionFA3Fallback]
        )
        if config.kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        elif (
            config.kv_params.cache_strategy
            == KVCacheStrategy.PAGED_FA3_FALLBACK
        ):
            kv_collection_cls = FetchPagedKVCacheCollectionFA3Fallback
        else:
            raise ValueError(
                "Unsupported caching strategy "
                + str(config.kv_params.cache_strategy)
            )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            devices=config.devices,
            return_logits=config.return_logits,
            # TODO: Support the following config options.
            # embedding_multiplier=config.embedding_multiplier,
            # logits_postprocessor=config.logits_postprocessor,
        )
