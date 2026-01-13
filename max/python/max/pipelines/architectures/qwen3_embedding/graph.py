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
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.architectures.qwen3.qwen3 import Qwen3


def last_token_pool(
    hidden_states: TensorValue, attention_mask: TensorValue
) -> TensorValue:
    """Apply last token pooling to extract embeddings.
    
    This function extracts the hidden state of the last non-padding token
    for each sequence in the batch, as used by Qwen3-Embedding models.
    
    Args:
        hidden_states: Output from the transformer model [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]
    
    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    # Calculate sequence lengths by summing attention mask along seq dimension
    # Subtract 1 to get the index of the last token (0-indexed)
    sequence_lengths = ops.sum(attention_mask, axis=1).cast(DType.int64) - 1
    
    # Clamp to ensure we're within bounds (at least 0)
    sequence_lengths = ops.max(
        sequence_lengths,
        ops.constant(0, DType.int64, device=sequence_lengths.device),
    )
    
    # Gather the last token's hidden state for each sequence in the batch
    # We create indices for gathering
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    
    # Reshape to [batch_size * seq_len, hidden_size]
    hidden_flat = ops.reshape(hidden_states, (-1, hidden_size))
    
    # Calculate flat indices: batch_idx * seq_len + sequence_length
    seq_len = hidden_states.shape[1]
    
    # Create batch offsets using range
    batch_offsets = ops.range(
        0,
        batch_size,
        1,
        out_dim=batch_size,
        dtype=DType.int64,
        device=DeviceRef.CPU(),
    )
    # Transfer to the correct device
    batch_offsets = ops.transfer_to(batch_offsets, device=hidden_states.device)
    # Get seq_len as a TensorValue from the shape
    seq_len_tensor = TensorValue(hidden_states.shape[1])
    batch_offsets = batch_offsets * seq_len_tensor
    flat_indices = batch_offsets + sequence_lengths
    
    # Gather the last tokens
    last_token_embeddings = ops.gather(hidden_flat, flat_indices, axis=0)
    
    return last_token_embeddings


def build_graph(
    pipeline_config: PipelineConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
    huggingface_config: AutoConfig,
    dtype: DType,
    input_device: DeviceRef,
) -> Graph:
    """Build the computation graph for Qwen3 Embedding model.
    
    Args:
        pipeline_config: Pipeline configuration
        state_dict: Model weights
        huggingface_config: HuggingFace model configuration
        dtype: Data type for computation
        input_device: Device to place inputs on
    
    Returns:
        Compiled graph for embedding generation
    """
    # Graph input types
    input_ids_type = TensorType(
        DType.int64, shape=["batch_size", "seq_len"], device=input_device
    )
    attention_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len"], device=input_device
    )

    # Convert state_dict to the expected format  
    # Most entries are already WeightData, just ensure the type is correct
    state_dict_converted: dict[str, WeightData] = {}
    for k, v in state_dict.items():
        # Assume all values can be converted to WeightData or are already WeightData
        state_dict_converted[k] = v  # type: ignore

    # Create a minimal KV cache config for single forward pass
    # We don't actually cache anything for embeddings, but Qwen3 model needs it
    kv_cache_config = KVCacheConfig(
        enable_prefix_caching=False,
        cache_strategy="paged",
    )

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
        return_hidden_states=ReturnHiddenStates.ALL,
    )

    with Graph(
        "qwen3_embedding", input_types=[input_ids_type, attention_mask_type]
    ) as graph:
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
        
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor
        
        # Flatten input_ids to ragged format [total_tokens]
        input_ids_flat = ops.reshape(input_ids, (-1,))
        
        # Get dimensions
        batch_size_dim = input_ids_type.shape[0]
        seq_len_dim = input_ids_type.shape[1]
        
        # Use max_seq_len from model config (known at graph build time)
        max_seq_len = model_config.max_seq_len
        
        # Create row offsets: [0, max_seq_len, 2*max_seq_len, ...]
        input_row_offsets = ops.range(
            0,
            batch_size_dim + 1,
            1,
            out_dim=batch_size_dim + 1,
            dtype=DType.uint32,
            device=input_device,
        ) * max_seq_len
        
        # For LAST_TOKEN logits mode, return_n_logits should be 1 (just the last token)
        return_n_logits = 1
        
        # Create dummy KV cache for single-pass embedding generation
        from max.nn.kv_cache import PagedCacheValues
        
        # For embedding generation, we use dummy KV cache since it's a single forward pass
        # The model still needs these parameters but won't use them effectively
        num_layers = model_config.num_hidden_layers
        num_kv_heads = model_config.num_key_value_heads
        head_dim = model_config.kv_params.head_dim
        page_size = kv_cache_config.kv_cache_page_size
        
        # Create a minimal buffer (won't be used)
        # Shape must be: (num_blocks, 2, num_layers, page_size, num_kv_heads, head_dim)
        # where 2 is for key and value
        from max.graph import BufferType
        
        kv_block_buffer = ops.buffer_create(
            BufferType(
                dtype,
                shape=(1, 2, num_layers, page_size, num_kv_heads, head_dim),
                device=input_device,
            )
        )
        
        # Create cache metadata (all should be uint32 for paged cache)
        # cache_lengths: [batch_size] - number of cached tokens per sequence
        cache_lengths = ops.range(
            0, 0, 1, out_dim=batch_size_dim, dtype=DType.uint32, device=DeviceRef.CPU()
        )
        cache_lengths = ops.transfer_to(cache_lengths, device=input_device)
        
        # lookup_table: [batch_size, max_blocks] - maps sequence indices to block IDs
        # For embedding generation we just need 1 block per batch
        import numpy as np
        max_blocks = 1  # Minimal for single forward pass
        lookup_table_np = np.zeros((1, max_blocks), dtype=np.uint32)
        lookup_table = ops.constant(lookup_table_np, device=DeviceRef.CPU())
        lookup_table = ops.transfer_to(lookup_table, device=input_device)
        
        # max_lengths: [1, 2] containing [max_seq_length, max_cache_length]
        # For embeddings, both are the same (max_seq_len)
        max_lengths_np = np.array([[max_seq_len, max_seq_len]], dtype=np.uint32)
        max_lengths = ops.constant(max_lengths_np, device=DeviceRef.CPU())
        max_lengths = ops.transfer_to(max_lengths, device=input_device)
        
        kv_collection = PagedCacheValues(
            kv_blocks=kv_block_buffer,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_lengths=max_lengths,
        )
        
        # Call the model - with return_hidden_states=ALL, it returns (logits, all_hidden_states)
        # The all_hidden_states is the last layer's hidden states for ALL tokens
        outputs = qwen3_model(
            input_ids_flat,
            kv_collection,
            return_n_logits,
            input_row_offsets,
        )
        
        # Extract hidden states from the tuple (logits, hidden_states)
        # hidden_states shape: [batch_size * seq_len, hidden_size]
        _, hidden_states_flat = outputs
        
        # Reshape back to [batch_size, seq_len, hidden_size] for last token pooling
        hidden_size = model_config.hidden_size
        hidden_states = ops.reshape(
            hidden_states_flat, (batch_size_dim, seq_len_dim, hidden_size)
        )
        
        # Apply last token pooling
        embeddings = last_token_pool(hidden_states, attention_mask)
        
        graph.output(embeddings)

    return graph
