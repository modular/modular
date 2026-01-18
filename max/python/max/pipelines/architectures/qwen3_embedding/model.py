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
"""Defines the Qwen3 Embedding pipeline model.

Implementation reuses the Qwen3 transformer with last token pooling
for generating embeddings.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, ops
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnHiddenStates, ReturnLogits
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    SupportedEncoding,
)
from max.python.max.nn.kv_cache.input_types import KVCacheInputs
from transformers import AutoConfig

from ..llama3.model import Llama3Inputs
from ..qwen3.model import Qwen3Model
from ..qwen3.model_config import Qwen3Config
from ..qwen3.qwen3 import Qwen3
from .pooling import last_token_pool, normalize_embeddings

logger = logging.getLogger("max.pipelines")


class Qwen3EmbeddingPipelineModel(Qwen3Model):
    """Pipeline model for Qwen3 Embedding models.

    This model extends Qwen3Model and overrides the graph building
    to add last token pooling for generating embeddings. This allows
    maximum code reuse from the generative Qwen3 model, including
    all KV cache handling logic.
    """

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
        # Call parent constructor with return_hidden_states enabled for pooling
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states=ReturnHiddenStates.ALL,  # Enable hidden states for pooling
        )

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        """Build graph with pooling layer for embeddings.

        Overrides parent's _build_graph to add last token pooling
        and normalization for embedding generation.
        """
        # Get state dict using parent's helper method
        state_dict = self._get_state_dict(weights, adapter)

        # Create model config with hidden states enabled
        model_config = Qwen3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
            return_hidden_states=ReturnHiddenStates.ALL,  # Enable hidden states for pooling
        )

        # Build the Qwen3 model
        nn_model = Qwen3(model_config)

        # Load weights
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            # For embedding models, we don't need lm_head weights
            # The smaller variants have tied word embeddings but for the 8B variant
            # we have to set strict=False to avoid errors
            strict=getattr(
                self.huggingface_config, "tie_word_embeddings", False
            ),
        )

        self.state_dict = nn_model.state_dict()

        # Get graph input types
        graph_inputs = nn_model.input_types(self.kv_params)

        with Graph("qwen3_embedding", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )

            # Construct KV cache collection (same as parent)
            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )

            # Call the model - returns multiple outputs depending on return_logits and return_hidden_states
            # With return_logits=ALL and return_hidden_states=ALL:
            # outputs = (next_token_logits, logits, hidden_states)
            outputs = nn_model(
                tokens.tensor,
                kv_collection,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )

            # Extract hidden states from output tuple - it's the last element
            # Model returns: (next_token_logits, logits, hidden_states) or (logits, hidden_states)
            hidden_states = outputs[-1]  # Hidden states are always last

            if self.pipeline_config.pool_embeddings:
                # Apply last token pooling to get embeddings
                # The pooling function extracts the last token from each sequence
                # using input_row_offsets to identify sequence boundaries
                embeddings = last_token_pool(
                    hidden_states, input_row_offsets.tensor
                )

                # Apply L2 normalization
                embeddings_normalized = normalize_embeddings(embeddings)

                # Output the pooled and normalized embeddings [batch_size, hidden_size]
                graph.output(embeddings_normalized)
            else:
                # Return raw hidden states without pooling [total_seq_len, hidden_size]
                hidden_states_f32 = ops.cast(hidden_states, DType.float32)
                graph.output(hidden_states_f32)

        return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model and return embeddings.

        For embedding models, we return the pooled embeddings in the logits field
        for compatibility with the pipeline interface. Unlike the generative model,
        our graph outputs a single tensor (embeddings) rather than multiple outputs.
        """
        assert isinstance(model_inputs, Llama3Inputs)

        # Get KV cache inputs
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        # Execute the model - unlike parent, we don't pass through signal_buffers
        # because the Qwen3 embedding graph doesn't include them as inputs
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *curr_kv_cache_inputs,
        )

        # Our graph outputs a single tensor (embeddings)
        assert isinstance(model_outputs[0], Buffer)
        # Store embeddings in the logits field for pipeline compatibility
        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        """Prepare inputs for embedding generation.

        Embeddings are computed in a single forward pass, so we don't need
        persistent KV cache. However, the graph still requires KV cache tensors
        as inputs, so we allocate temporary KV cache entries for this batch.
        """
        # First, use parent to prepare basic inputs (tokens, row_offsets, etc.)
        # But we pass None for kv_cache_inputs initially
        inputs = super().prepare_initial_token_inputs(
            replica_batches, None, return_n_logits
        )

        # Now populate KV cache inputs from the manager
        # For embeddings, we need to temporarily allocate KV cache entries
        if kv_cache_inputs is None:
            # Flatten batch to get all contexts
            batch = [ctx for replica in replica_batches for ctx in replica]

            # Allocate KV cache for this batch
            # Each context needs space for its tokens
            # First claim, then alloc
            for ctx in batch:
                if not self.kv_manager.contains(ctx.request_id):
                    self.kv_manager.claim(ctx.request_id, replica_idx=0)
                    self.kv_manager.alloc(
                        ctx,
                        num_steps=1,  # Only need 1 step for embedding generation
                    )

            # Get the runtime inputs after allocation
            kv_cache_inputs_list = self.kv_manager.get_runtime_inputs(batch)
            if kv_cache_inputs_list:
                kv_cache_inputs = kv_cache_inputs_list[
                    0
                ]  # Get first device's KV inputs

        # Update the inputs with KV cache data
        inputs.kv_cache_inputs = kv_cache_inputs
        return inputs
