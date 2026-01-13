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
from max.pipelines.lib import PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig

from ..qwen3.model_config import Qwen3Config
from ..qwen3.qwen3 import Qwen3


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
    batch_offsets = batch_offsets * ops.constant(
        seq_len, DType.int64, device=hidden_states.device
    )
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
    state_dict_converted: dict[str, WeightData] = {}
    for k, v in state_dict.items():
        if isinstance(v, WeightData):
            state_dict_converted[k] = v
        elif hasattr(v, "__dlpack__"):
            from max.graph.weights import WeightData as WD
            # Create WeightData from DLPackArray
            state_dict_converted[k] = WD(v)
        else:
            state_dict_converted[k] = v

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
        kv_cache_config=pipeline_config.kv_cache,
        # Configure to return LAST_TOKEN logits but also LAST_NORMALIZED hidden states
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.LAST_NORMALIZED,
    )

    with Graph(
        "qwen3_embedding", input_types=[input_ids_type, attention_mask_type]
    ) as graph:
        # Build the Qwen3 model
        qwen3_model = Qwen3(model_config)
        
        # Load weights (excluding LM head which we don't need for embeddings)
        filtered_state_dict = {
            k: v
            for k, v in state_dict_converted.items()
            if not k.startswith("lm_head") and not k.startswith("output")
        }
        qwen3_model.load_state_dict(
            filtered_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,  # Allow missing lm_head weights
        )
        
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor
        
        # For embedding generation, we need to:
        # 1. Flatten input_ids to [total_seq_len]
        # 2. Create input_row_offsets for ragged tensors
        # 3. Call the model to get hidden states
        # 4. Reshape and pool
        
        # Flatten input_ids to [total_seq_len]
        input_ids_flat = ops.reshape(input_ids, (-1,))
        
        # Create input_row_offsets for ragged tensors
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # input_row_offsets indicates where each sequence starts
        # [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
        total_tokens = batch_size * seq_len
        input_row_offsets = ops.range(
            0,
            batch_size + 1,
            1,
            out_dim=batch_size + 1,
            dtype=DType.uint32,
            device=DeviceRef.CPU(),
        )
        # Transfer to device and multiply by seq_len
        input_row_offsets = ops.transfer_to(input_row_offsets, device=input_device)
        input_row_offsets = input_row_offsets * ops.constant(
            seq_len, DType.uint32, device=input_device
        )
        
        # Return all tokens' hidden states
        return_n_logits = ops.constant(
            total_tokens, DType.int64, device=DeviceRef.CPU()
        )
        
        # Create dummy KV cache for single-pass embedding generation
        from max.nn.kv_cache import PagedCacheValues
        
        # For embedding generation, we use dummy KV cache since it's a single forward pass
        # The model still needs these parameters but won't use them effectively
        num_layers = model_config.num_hidden_layers
        num_kv_heads = model_config.num_key_value_heads
        head_dim = model_config.kv_params.head_dim
        
        # Create a minimal buffer (won't be used)
        kv_block_buffer = ops.buffer_create(
            (1, num_layers, num_kv_heads, 1, head_dim),
            dtype=dtype,
            device=input_device,
        )
        
        # Create cache metadata
        cache_lengths = ops.range(
            0, 0, 1, out_dim=batch_size, dtype=DType.int64, device=DeviceRef.CPU()
        )
        cache_lengths = ops.transfer_to(cache_lengths, device=input_device)
        
        lookup_table = ops.range(
            0, 0, 1, out_dim=batch_size, dtype=DType.int64, device=DeviceRef.CPU()
        )
        lookup_table = ops.transfer_to(lookup_table, device=input_device)
        
        max_lengths_val = ops.constant(seq_len, DType.int64, device=DeviceRef.CPU())
        max_lengths = ops.broadcast_to(
            ops.unsqueeze(max_lengths_val, 0), (batch_size,)
        )
        max_lengths = ops.transfer_to(max_lengths, device=input_device)
        
        kv_collection = PagedCacheValues(
            kv_blocks=kv_block_buffer,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_lengths=max_lengths,
        )
        
        # Call the model - it will return (logits, hidden_states) tuple
        # since we configured return_hidden_states=LAST_NORMALIZED
        outputs = qwen3_model(
            input_ids_flat,
            kv_collection,
            return_n_logits,
            input_row_offsets,
        )
        
        # Extract hidden states (second element of tuple)
        # Shape: [batch_size * seq_len, hidden_size]
        hidden_states_flat = outputs[1]
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        hidden_size = model_config.hidden_size
        hidden_states = ops.reshape(
            hidden_states_flat, (batch_size, seq_len, hidden_size)
        )
        
        # Apply last token pooling
        embeddings = last_token_pool(hidden_states, attention_mask)
        
        graph.output(embeddings)

    return graph
