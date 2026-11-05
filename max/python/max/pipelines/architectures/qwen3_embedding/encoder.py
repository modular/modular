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
"""The Qwen3 embedding encoder an architecture registers for ``ENCODER``."""

from __future__ import annotations

import functools
import math

from max.dtype import DType
from max.graph import TensorType, TensorValue
from max.nn.legacy.embedding import Embedding
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import MLP, Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import Llama3RotaryEmbedding
from max.nn.rope_scaling import Llama3RopeScalingParams
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.registration_encoder import EncoderInputs

from .layers import (
    Qwen3AttentionNoCache,
    Qwen3EmbeddingTransformer,
    Qwen3EmbeddingTransformerBlock,
)
from .model_config import Qwen3EmbeddingConfig


class _Qwen3EmbeddingTransformer(Qwen3EmbeddingTransformer):
    # The checkpoint names the transformer's weights from its root.
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class Qwen3EmbeddingEncoder(Module):
    """Qwen3 embedding, registered for the ``ENCODER`` role.

    It takes a ragged batch and returns last-token pooled, normalized
    embeddings, or the float32 hidden states when the pipeline doesn't pool.
    """

    graph_name = "qwen3_embedding"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    ragged_batch = True
    override_quantization_encoding = True
    weight_alignment = 1

    def __init__(self, config: Qwen3EmbeddingConfig) -> None:
        """Builds the transformer from the checkpoint's config."""
        super().__init__()
        hf = config.huggingface_config
        dtype = config.dtype
        device_refs = config.devices
        head_dim = hf.head_dim
        rope_scaling_params: Llama3RopeScalingParams | None = None
        rope_scaling = getattr(hf, "rope_scaling", None)
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type") or rope_scaling.get(
                "rope_type"
            )
            if rope_type == "llama3":
                rope_scaling_params = Llama3RopeScalingParams(
                    factor=rope_scaling["factor"],
                    low_freq_factor=rope_scaling["low_freq_factor"],
                    high_freq_factor=rope_scaling["high_freq_factor"],
                    orig_max_position=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
        rope = Llama3RotaryEmbedding(
            dim=hf.hidden_size,
            n_heads=hf.num_attention_heads,
            theta=get_rope_theta(hf),
            max_seq_len=config.max_seq_len,
            head_dim=head_dim,
            interleaved=False,
            scaling_params=rope_scaling_params,
        )
        attention_multiplier = getattr(
            hf, "attention_multiplier", 1.0 / math.sqrt(float(head_dim))
        )
        norm_eps = getattr(hf, "rms_norm_eps", 1e-6)
        create_norm = functools.partial(
            RMSNorm,
            hf.hidden_size,
            dtype=dtype,
            eps=norm_eps,
            multiply_before_cast=False,
        )
        layers = [
            Qwen3EmbeddingTransformerBlock(
                attention=Qwen3AttentionNoCache(
                    rope=rope,
                    num_attention_heads=hf.num_attention_heads,
                    num_key_value_heads=hf.num_key_value_heads,
                    hidden_size=hf.hidden_size,
                    head_dim=head_dim,
                    dtype=dtype,
                    devices=device_refs,
                    scale=attention_multiplier,
                    qk_norm_eps=norm_eps,
                ),
                mlp=MLP(
                    dtype=dtype,
                    quantization_encoding=None,
                    hidden_dim=hf.hidden_size,
                    feed_forward_length=hf.intermediate_size,
                    devices=device_refs,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=1.0,
            )
            for _ in range(hf.num_hidden_layers)
        ]
        embedding_quantization = config.embedding_quantization
        embedding_dtype = dtype if not embedding_quantization else DType.uint8
        embedding = Embedding(
            hf.vocab_size,
            hf.hidden_size,
            embedding_dtype,
            device_refs[0],
            quantization_encoding=embedding_quantization,
        )
        # Kept for weight sharing with the embedding; the encoder doesn't
        # project to logits.
        output = Linear(
            hf.hidden_size,
            hf.vocab_size,
            embedding_dtype,
            device_refs[0],
            quantization_encoding=embedding_quantization,
        )
        tie_word_embeddings = bool(getattr(hf, "tie_word_embeddings", False))
        if tie_word_embeddings:
            output.set_shared_weight("weight", embedding.weight)
        # Loads strictly only when tied, as it did before registration.
        self.strict_state_dict_loading = tie_word_embeddings
        self.model = _Qwen3EmbeddingTransformer(
            dim=hf.hidden_size,
            n_heads=hf.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding,
            rope=rope,
            embedding_multiplier=1.0,
            device=device_refs[0],
            pool_embeddings=config.pool_embeddings,
        )

    def legacy_input_types(self) -> tuple[TensorType, ...]:
        """Returns the input types the graph had before registration.

        Their dimension names differ from the ragged layout's, and the MEF
        key includes them.
        """
        return self.model.input_types()

    def __call__(self, inputs: EncoderInputs[TensorValue]) -> TensorValue:
        """Embeds the ragged batch."""
        assert inputs.input_row_offsets is not None
        assert inputs.return_n_logits is not None
        return self.model(
            inputs.tokens, inputs.input_row_offsets, inputs.return_n_logits
        )
