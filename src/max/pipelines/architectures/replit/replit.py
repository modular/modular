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
"""Implements the Replit model."""

from __future__ import annotations

from max.graph import ops
from max.nn import (
    Embedding,
    LayerNorm,
    Linear,
    MHAMaskVariant,
    Module,
    RaggedAttention,
    Sequential,
    Transformer,
    TransformerBlock,
)
from max.nn.kv_cache import FetchContinuousBatchingKVCacheCollection

from .model_config import ReplitConfig


class Replit(Transformer):
    """Defines the replit transformer model."""

    def __init__(self, config: ReplitConfig):
        assert len(config.devices) == 1

        layers = [
            TransformerBlock(
                attention=RaggedAttention(
                    mask_variant=MHAMaskVariant.CAUSAL_ALIBI_MASK,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    scale=config.attention_multiplier,
                    stacked_qkv=True,
                    has_bias=False,
                ),
                mlp=Sequential(
                    layers=[
                        Linear(
                            config.hidden_size,
                            12288,
                            config.dtype,
                            config.devices[0],
                        ),
                        Gelu(),
                        Linear(
                            12288,
                            config.hidden_size,
                            config.dtype,
                            config.devices[0],
                        ),
                    ]
                ),
                attention_norm=LayerNorm(
                    config.hidden_size,
                    config.devices[0],
                    1e-5,
                    use_bias=False,
                ),
                mlp_norm=LayerNorm(
                    config.hidden_size,
                    config.devices[0],
                    1e-5,
                    use_bias=False,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
        )
        output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=LayerNorm(
                config.hidden_size,
                config.devices[0],
                1e-5,
                use_bias=False,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=FetchContinuousBatchingKVCacheCollection(
                config.kv_params
            ),
            return_logits=config.return_logits,
        )


class Gelu(Module):
    """Basic layer that applies GELU (Gaussian Error Linear Units) on the input."""

    def __call__(self, x):
        return ops.gelu(x)
