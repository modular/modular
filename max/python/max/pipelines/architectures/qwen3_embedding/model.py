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
"""Qwen3 Embedding pipeline model without KV caching."""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, ops
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.legacy.embedding import Embedding
from max.nn.legacy.kv_cache import KVCacheInputs
from max.nn.legacy.linear import MLP, Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import Llama3RotaryEmbedding
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers import AutoConfig

from .layers import (
    Qwen3AttentionNoCache,
    Qwen3EmbeddingTransformer,
    Qwen3EmbeddingTransformerBlock,
    last_token_pool,
    normalize_embeddings,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen3EmbeddingInputs(ModelInputs):
    """Input structure for Qwen3 embedding models."""

    tokens: Buffer
    """Input token IDs [total_seq_len]"""

    input_row_offsets: Buffer
    """Row offsets for ragged tensors [batch_size + 1]"""

    return_n_logits: Buffer
    """Number of logits to return (kept for interface compatibility)"""


class Qwen3EmbeddingModel(PipelineModel[TextContext]):
    """Qwen3 embedding pipeline model without KV caching.

    This model is optimized for embedding generation with:
    - No KV cache overhead
    - Single-pass forward computation
    - Flash attention without cache operations
    - Last token pooling with L2 normalization
    """

    model: Model
    """Compiled and initialized model."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization method."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    state_dict: dict[str, Any]
    """Model weights."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        """Initialize the Qwen3 embedding pipeline model.

        Args:
            pipeline_config: Pipeline configuration
            session: Inference session
            huggingface_config: HuggingFace model configuration
            encoding: Encoding configuration
            devices: List of devices
            weights: Model weights
            adapter: Optional weight adapter
        """
        self.pipeline_config = pipeline_config
        self.session = session
        self.huggingface_config = huggingface_config
        self.encoding = encoding
        self.devices = devices

        # Build and compile graph
        logger.info(f"Building {self.__class__.__name__} graph")
        graph = self._build_graph(weights, adapter, session)
        logger.info(f"Compiling {self.__class__.__name__} model")
        self.model = session.load(graph, weights_registry=self.state_dict)
        logger.info("Model loaded successfully")

    @property
    def dtype(self) -> DType:
        """Get the model's data type."""
        return self.encoding.dtype

    def _get_state_dict(
        self, weights: Weights, adapter: WeightsAdapter | None
    ) -> dict[str, WeightData]:
        """Get state dictionary from weights.

        Args:
            weights: Model weights
            adapter: Optional adapter

        Returns:
            State dictionary
        """
        if adapter:
            return adapter(
                dict(weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        return {key: value.data() for key, value in weights.items()}

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        """Build the embedding model graph.

        Args:
            weights: Model weights
            adapter: Optional weight adapter
            session: Optional inference session

        Returns:
            Compiled graph
        """
        # Load weights
        state_dict = self._get_state_dict(weights, adapter)

        # Get configuration
        dtype = self.encoding.dtype
        device_refs = [DeviceRef.from_device(d) for d in self.devices]

        # Create RoPE
        head_dim = self.huggingface_config.head_dim
        max_seq_len = self.pipeline_config.model.max_length or 32768
        rope = Llama3RotaryEmbedding(
            dim=self.huggingface_config.hidden_size,
            n_heads=self.huggingface_config.num_attention_heads,
            theta=self.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=False,  # Qwen3 uses non-interleaved RoPE
            scaling_params=getattr(
                self.huggingface_config, "rope_scaling", None
            ),
        )

        # Calculate Qwen3-specific attention multiplier
        attention_multiplier = getattr(
            self.huggingface_config,
            "attention_multiplier",
            1.0 / math.sqrt(float(head_dim)),
        )

        # Create normalization layer
        norm_eps = getattr(self.huggingface_config, "rms_norm_eps", 1e-6)
        create_norm = functools.partial(
            RMSNorm,
            self.huggingface_config.hidden_size,
            dtype=dtype,
            eps=norm_eps,
            multiply_before_cast=False,
        )

        # Create transformer layers
        layers = []
        for _layer_idx in range(self.huggingface_config.num_hidden_layers):
            # Create attention layer
            attention = Qwen3AttentionNoCache(
                rope=rope,
                num_attention_heads=self.huggingface_config.num_attention_heads,
                num_key_value_heads=self.huggingface_config.num_key_value_heads,
                hidden_size=self.huggingface_config.hidden_size,
                head_dim=head_dim,
                dtype=dtype,
                devices=device_refs,
                scale=attention_multiplier,
                qk_norm_eps=norm_eps,
            )

            # Create MLP
            mlp = MLP(
                dtype=dtype,
                quantization_encoding=None,
                hidden_dim=self.huggingface_config.hidden_size,
                feed_forward_length=self.huggingface_config.intermediate_size,
                devices=device_refs,
            )

            # Create transformer block
            block = Qwen3EmbeddingTransformerBlock(
                attention=attention,
                mlp=mlp,
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=1.0,
            )
            layers.append(block)

        # Create embedding layer
        embed_weight = state_dict.get("embed_tokens.weight")
        embedding_quantization = (
            embed_weight.quantization_encoding if embed_weight else None
        )
        embedding_dtype = dtype if not embedding_quantization else DType.uint8

        embedding_layer = Embedding(
            self.huggingface_config.vocab_size,
            self.huggingface_config.hidden_size,
            embedding_dtype,
            device_refs[0],
            quantization_encoding=embedding_quantization,
        )

        # Create output layer (for weight sharing with embedding)
        output = Linear(
            self.huggingface_config.hidden_size,
            self.huggingface_config.vocab_size,
            embedding_dtype,
            device_refs[0],
            quantization_encoding=embedding_quantization,
        )

        # Share weights if configured
        if getattr(self.huggingface_config, "tie_word_embeddings", False):
            output.set_shared_weight("weight", embedding_layer.weight)

        # Create transformer
        nn_model = Qwen3EmbeddingTransformer(
            dim=self.huggingface_config.hidden_size,
            n_heads=self.huggingface_config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            rope=rope,
            return_hidden_states=ReturnHiddenStates.ALL,  # Return un-normalized states, pooling+norm happens after
            embedding_multiplier=1.0,
            device=device_refs[0],
        )

        # Load weights into model
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=getattr(
                self.huggingface_config, "tie_word_embeddings", False
            ),
        )

        self.state_dict = nn_model.state_dict()

        # Build graph
        graph_inputs = nn_model.input_types()

        with Graph("qwen3_embedding", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits = graph.inputs

            # Forward pass - returns (hidden_states,)
            outputs = nn_model(
                tokens.tensor,
                input_row_offsets.tensor,
                return_n_logits.tensor,
            )

            # Extract hidden states
            hidden_states = outputs[0]

            if self.pipeline_config.pool_embeddings:
                # Apply last token pooling
                embeddings = last_token_pool(
                    hidden_states, input_row_offsets.tensor
                )

                # Apply L2 normalization
                embeddings_normalized = normalize_embeddings(embeddings)

                # Output pooled and normalized embeddings [batch_size, hidden_size]
                graph.output(embeddings_normalized)
            else:
                # Return raw hidden states [total_seq_len, hidden_size]
                hidden_states_f32 = ops.cast(hidden_states, DType.float32)
                graph.output(hidden_states_f32)

        return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model.

        Args:
            model_inputs: Model inputs

        Returns:
            Model outputs with embeddings in the logits field
        """
        assert isinstance(model_inputs, Qwen3EmbeddingInputs)

        # Execute model
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
        )

        # Return embeddings in logits field for pipeline compatibility
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3EmbeddingInputs:
        """Prepare initial inputs for embedding generation.

        Args:
            replica_batches: Batches of text contexts
            kv_cache_inputs: Ignored (no KV cache for embeddings)
            return_n_logits: Number of logits (ignored for embeddings)

        Returns:
            Prepared inputs
        """
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        device = self.devices[0]

        # Collect all tokens from the batch
        all_tokens: list[int] = []
        row_offsets = [0]

        for ctx in context_batch:
            tokens = ctx.tokens.active
            all_tokens.extend(tokens)
            row_offsets.append(len(all_tokens))

        # Convert to numpy arrays
        tokens_array = np.array(all_tokens, dtype=np.uint32)
        row_offsets_array = np.array(row_offsets, dtype=np.uint32)

        # Create buffers on CPU (inputs are expected on CPU)
        tokens_buffer = Buffer.from_numpy(tokens_array)
        row_offsets_buffer = Buffer.from_numpy(row_offsets_array)
        return_n_logits_buffer = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.uint32)
        )

        return Qwen3EmbeddingInputs(
            tokens=tokens_buffer.to(device),
            input_row_offsets=row_offsets_buffer,
            return_n_logits=return_n_logits_buffer,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Qwen3EmbeddingInputs:
        """Prepare next token inputs (not supported for embedding models).

        Args:
            next_tokens: Next tokens
            prev_model_inputs: Previous inputs

        Raises:
            NotImplementedError: Embedding models don't support autoregressive generation
        """
        raise NotImplementedError(
            "Qwen3 embedding model does not support autoregressive generation"
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length.

        Args:
            pipeline_config: Pipeline configuration
            huggingface_config: HuggingFace configuration

        Returns:
            Maximum sequence length
        """
        # Use configured max_length, bounded by model's max_position_embeddings
        model_max = getattr(
            huggingface_config, "max_position_embeddings", 32768
        )
        configured_max = pipeline_config.model.max_length or 8192

        if configured_max > model_max:
            raise ValueError(
                f"Configured max_length ({configured_max}) exceeds model's "
                f"max_position_embeddings ({model_max})"
            )

        return configured_max
