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

Implementation uses the Qwen3 transformer with last token pooling
for generating embeddings.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
from max._core.engine import Model
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.kv_cache import NullKVCacheManager
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.pipelines.core import TextContext
from max.pipelines.dataprocessing import collate_batch
from max.pipelines.lib import (
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from transformers import AutoConfig

from .graph import build_graph
from .model_config import Qwen3EmbeddingConfig

logger = logging.getLogger("max.pipelines")

PAD_VALUE = 1


class Qwen3EmbeddingInputs(ModelInputs):
    """A class representing inputs for the Qwen3 Embedding model.

    This class encapsulates the input tensors required for the Qwen3 Embedding
    model execution:
    - attention_mask: A tensor containing the attention mask
    - next_tokens_batch: A tensor containing the input token IDs (flattened/ragged)
    - row_offsets: Row offsets for ragged tensor format
    - return_n_logits: Number of logits to return
    - kv_cache_inputs: KV cache tensors (optional, for compatibility with generative model)
    """

    attention_mask: Tensor
    next_tokens_batch: Tensor
    row_offsets: Tensor
    return_n_logits: Tensor

    def __init__(
        self,
        attention_mask: Tensor,
        next_tokens_batch: Tensor,
        row_offsets: Tensor,
        return_n_logits: Tensor,
        kv_cache_inputs: tuple[Tensor, ...] | None = None,
    ) -> None:
        self.attention_mask = attention_mask
        self.next_tokens_batch = next_tokens_batch
        self.row_offsets = row_offsets
        self.return_n_logits = return_n_logits
        self.kv_cache_inputs = kv_cache_inputs


class Qwen3EmbeddingPipelineModel(PipelineModel[TextContext], KVCacheMixin):
    """Pipeline model for Qwen3 Embedding models.
    
    This model processes text inputs and generates embeddings using
    the Qwen3 architecture with last token pooling.
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
        )
        self.model = self.load_model(session)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Qwen3EmbeddingConfig.get_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Qwen3EmbeddingConfig.get_num_layers(huggingface_config)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Qwen3 Embedding, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Qwen3EmbeddingInputs)
        
        # Get KV cache inputs if available, otherwise use empty tuple
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        
        model_outputs = self.model.execute(
            model_inputs.attention_mask,
            model_inputs.next_tokens_batch,
            model_inputs.row_offsets,
            model_inputs.return_n_logits,
            *curr_kv_cache_inputs,
        )
        assert isinstance(model_outputs[0], Tensor)

        # For embedding models, the output is the embeddings, not logits
        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3EmbeddingInputs:
        if len(replica_batches) > 1:
            raise ValueError("Qwen3 Embedding does not support DP>1")

        context_batch = replica_batches[0]

        # Get tokens from contexts.
        tokens = [ctx.tokens.active for ctx in context_batch]

        # Pad tokens for the batch.
        # Use EOS token as pad token for Qwen3
        pad_value = getattr(
            self.huggingface_config,
            "pad_token_id",
            getattr(self.huggingface_config, "eos_token_id", 151643),
        )
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )

        # Compute attention mask.
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)

        # Flatten tokens to ragged format [total_seq_len] like generative model
        # This is what the Qwen3 model expects
        tokens_flat = np.concatenate(tokens)

        # Compute row offsets for ragged tensor format
        # These mark the start/end boundaries of each sequence in the flattened array
        row_offsets = np.zeros(len(tokens) + 1, dtype=np.uint32)
        np.cumsum(
            [0] + [len(t) for t in tokens],
            dtype=np.uint32,
            out=row_offsets,
        )

        # return_n_logits tensor - must be on CPU
        return_n_logits_tensor = Tensor.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )  # Defaults to CPU

        # Get KV cache inputs from manager
        # For embeddings, we need to provide KV cache tensors even though we don't use them
        # between requests. Create empty/zero KV cache inputs.
        if kv_cache_inputs is None:
            # Create empty KV cache inputs manually
            # Get the KV blocks tensor from the manager (first device)
            kv_blocks = self.kv_manager._replica_managers[0].device_tensors[0]
            
            batch_size = len(tokens)
            # Cache lengths: all zeros since we don't have cached tokens
            cache_lengths = Tensor.from_numpy(
                np.zeros(batch_size, dtype=np.uint32)
            ).to(self.devices[0])
            
            # Lookup table: maps sequences to blocks (just use sequential blocks)
            # Calculate max_pages based on max sequence length and page size
            page_size = self.kv_params.page_size
            max_seq_len = self.pipeline_config.max_length
            max_pages = (max_seq_len + page_size - 1) // page_size  # Ceiling division
            
            lookup_table = Tensor.from_numpy(
                np.arange(batch_size * max_pages, dtype=np.uint32).reshape(batch_size, max_pages)
            ).to(self.devices[0])
            
            # Max lengths: [max_seq_length, max_cache_length] per sequence
            # This must be on CPU according to KVCacheParams.get_symbolic_inputs
            max_lengths = Tensor.from_numpy(
                np.array([[max_seq_len, max_seq_len] for _ in range(batch_size)], dtype=np.uint32)
            )  # Defaults to CPU
            
            kv_inputs_tuple = (kv_blocks, cache_lengths, lookup_table, max_lengths)
        else:
            kv_inputs_tuple = (
                kv_cache_inputs.kv_blocks,
                kv_cache_inputs.cache_lengths,
                kv_cache_inputs.lookup_table,
                kv_cache_inputs.max_lengths,
            )

        return Qwen3EmbeddingInputs(
            attention_mask=Tensor.from_numpy(attention_mask).to(
                self.devices[0]
            ),
            next_tokens_batch=Tensor.from_numpy(tokens_flat).to(
                self.devices[0]
            ),
            row_offsets=Tensor.from_numpy(row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=return_n_logits_tensor,
            kv_cache_inputs=kv_inputs_tuple,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> Qwen3EmbeddingInputs:
        raise NotImplementedError(
            "Qwen3 Embedding does not support preparing next token inputs "
            "(embeddings are generated in a single forward pass)."
        )

    def load_model(self, session: InferenceSession) -> Model:
        logger.info("Building and compiling Qwen3 Embedding model...")
        before = time.perf_counter()
        
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        
        graph = build_graph(
            self.pipeline_config,
            state_dict,
            self.huggingface_config,
            self.dtype,
            DeviceRef.from_device(self.devices[0]),
            self.kv_params,
            self.kv_cache_config,
        )
        after_build = time.perf_counter()

        logger.info(f"Building graph took {after_build - before:.6f} seconds")

        before_compile = time.perf_counter()
        model = session.load(graph, weights_registry=state_dict)
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model
