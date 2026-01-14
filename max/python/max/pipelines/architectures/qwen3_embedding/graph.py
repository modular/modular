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
"""Graph builder for Qwen3 Embedding models with last token pooling."""

from __future__ import annotations

from collections.abc import Mapping

from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.nn import ReturnHiddenStates, ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.architectures.qwen3.qwen3 import Qwen3
import numpy as np

def last_token_pool(
    hidden_states: TensorValue, attention_mask: TensorValue, input_row_offsets: TensorValue
) -> TensorValue:
    """Apply last token pooling to extract embeddings.
    
    This function extracts the hidden state of the last non-padding token
    for each sequence in the batch, as used by Qwen3-Embedding models.
    
    Args:
        hidden_states: Output from the transformer model in ragged format [total_seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len] marking valid tokens (1) vs padding (0)
        input_row_offsets: Row offsets defining sequence boundaries in flattened format [batch_size + 1]
    
    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    # Compute sequence lengths by summing attention mask
    # attention_mask shape: [batch_size, seq_len]
    sequence_lengths = ops.sum(attention_mask, axis=1)  # [batch_size], float32
    
    # Convert to int64 for indexing
    sequence_lengths_int = ops.cast(sequence_lengths, DType.int64)
    
    # Get starting offsets for each sequence using Python slicing
    # For batch_size=1: input_row_offsets = [0, total_len], so start_offsets = [0]
    start_offsets = input_row_offsets[:-1]  # Remove last element
    start_offsets_int = ops.cast(start_offsets, DType.int64)
    
    # Compute the index of the last valid token for each sequence
    # last_index = start_offset + sequence_length - 1
    one = ops.constant(1, DType.int64, device=hidden_states.device)
    last_token_indices = start_offsets_int + sequence_lengths_int - one
    
    # Use gather to extract the embeddings - cast to int32 for gather
    last_token_indices_i32 = ops.cast(last_token_indices, DType.int32)
    
    # Gather the hidden states at the last valid token positions
    # For batch_size=1, this will give us [1, hidden_size]
    pooled = ops.gather(hidden_states, last_token_indices_i32, axis=0)
    
    return pooled


def build_graph(
    pipeline_config: PipelineConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
    huggingface_config: AutoConfig,
    dtype: DType,
    input_device: DeviceRef,
    kv_params: KVCacheParams,
    kv_cache_config: KVCacheConfig,
) -> Graph:
    """Build the computation graph for Qwen3 Embedding model.
    
    Args:
        pipeline_config: Pipeline configuration
        state_dict: Model weights
        huggingface_config: HuggingFace model configuration
        dtype: Data type for computation
        input_device: Device to place inputs on
        kv_params: KV cache parameters for proper graph input construction
        kv_cache_config: KV cache configuration
    
    Returns:
        Compiled graph for embedding generation
    """
    # Convert state_dict to the expected format  
    # Most entries are already WeightData, just ensure the type is correct
    state_dict_converted: dict[str, WeightData] = {}
    for k, v in state_dict.items():
        # Assume all values can be converted to WeightData or are already WeightData
        state_dict_converted[k] = v  # type: ignore

    # Create model config for Qwen3 - we'll use it to build the model
    # but configure it to return hidden states instead of logits
    model_config = Qwen3Config.generate(
        pipeline_config=pipeline_config,
        huggingface_config=huggingface_config,
        state_dict=state_dict_converted,
        dtype=dtype,
        n_devices=1,
        norm_method="rms_norm",
        attention_bias=getattr(huggingface_config, "attention_bias", False),
        cache_dtype=dtype,
        kv_cache_config=kv_cache_config,
        # For embedding generation, we need ALL hidden states to apply last token pooling
        # ReturnHiddenStates.ALL returns the hidden states from the last layer
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.ALL,  # Enable hidden states for embeddings
    )

    # Build the Qwen3 model - Qwen3-Embedding uses the full CausalLM model
    # with tied embeddings (lm_head shares weights with embed_tokens)
    qwen3_model = Qwen3(model_config)
    
    # Load all weights - the model has lm_head with tied weights
    qwen3_model.load_state_dict(
        state_dict_converted,
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=False,  # Allow missing weights if any
    )

    # Get proper graph input types from the model, including KV cache inputs
    graph_inputs = qwen3_model.input_types(kv_params)
    
    # Add attention_mask as an additional input for pooling
    # attention_mask shape: [batch_size, seq_len]
    attention_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len"], device=input_device
    )
    # Prepend attention mask to the graph inputs
    graph_inputs_with_mask = (attention_mask_type,) + graph_inputs

    with Graph(
        "qwen3_embedding", input_types=graph_inputs_with_mask
    ) as graph:
        # Extract inputs: attention_mask, then the standard model inputs
        attention_mask = graph.inputs[0].tensor
        tokens = graph.inputs[1].tensor
        input_row_offsets = graph.inputs[2].tensor
        return_n_logits = graph.inputs[3].tensor
        kv_cache_inputs = graph.inputs[4:]
        
        # Construct KV cache collection from graph inputs (like generative model)
        from max.nn.kv_cache import PagedCacheValues
        kv_collection = PagedCacheValues(
            kv_blocks=kv_cache_inputs[0].buffer,
            cache_lengths=kv_cache_inputs[1].tensor,
            lookup_table=kv_cache_inputs[2].tensor,
            max_lengths=kv_cache_inputs[3].tensor,
        )
        
        # Call the model - returns tuple (logits, hidden_states)
        outputs = qwen3_model(
            tokens,
            kv_collection,
            return_n_logits,
            input_row_offsets,
        )
        
        # Extract hidden states from output tuple
        # outputs is (logits, hidden_states) where hidden_states is [total_seq_len, hidden_size] in ragged format
        logits, hidden_states = outputs
        
        # Apply last token pooling to get embeddings
        # Use attention_mask to find the last valid (non-padding) token
        embeddings = last_token_pool(hidden_states, attention_mask, input_row_offsets)
        
        # Cast to float32 for compatibility with numpy/dlpack
        # bfloat16 is not supported by dlpack's to_numpy conversion
        embeddings_f32 = ops.cast(embeddings, DType.float32)
        
        # Output the embeddings [batch_size, hidden_size]
        graph.output(embeddings_f32)

    return graph
