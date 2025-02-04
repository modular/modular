# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines cross attention layer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from llama_vision.cross_attention_decoder import CrossSdpaAttention
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, TensorValue, Weight
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    load_kv_manager,
)
from max.pipelines.nn import Linear, RMSNorm
from test_common.distance_metrics import is_euclidean_distance_close
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.models.mllama.modeling_mllama import (
    MllamaTextCrossSdpaAttention,
)

FAKE_TOKEN = 999


class CrossAttentionModel:
    """Model containing fetch and cross attention layers."""

    fetch: FetchContinuousBatchingKVCacheCollection
    """Layer for fetching a kv cache collection."""

    cross_attention: CrossSdpaAttention
    """Layer for computing multimodal cross attention."""

    dtype: DType
    """DType of the model weights."""

    def __init__(
        self,
        config: MllamaTextConfig,
        kv_params: KVCacheParams,
        torch_cross_attn: MllamaTextCrossSdpaAttention,
        dtype: DType,
    ) -> None:
        """Inits fetch and cross attention layers using the torch model."""
        self.dtype = dtype

        self.fetch = FetchContinuousBatchingKVCacheCollection(kv_params)

        # Use torch model weights to initialize MAX graph cross attention
        # shapes.
        self.cross_attention = CrossSdpaAttention(
            config.num_attention_heads,
            kv_params,
            layer_idx=0,
            q_proj=Linear(
                Weight(
                    name="wq",
                    dtype=self.dtype,
                    shape=torch_cross_attn.q_proj.weight.shape,
                )
            ),
            wk=Weight(
                name="wk",
                dtype=self.dtype,
                shape=torch_cross_attn.k_proj.weight.shape,
            ),
            wv=Weight(
                name="wv",
                dtype=self.dtype,
                shape=torch_cross_attn.v_proj.weight.shape,
            ),
            o_proj=Linear(
                Weight(
                    name="wo",
                    dtype=self.dtype,
                    shape=torch_cross_attn.o_proj.weight.shape,
                )
            ),
            q_norm=RMSNorm(
                Weight(
                    name="q_norm",
                    dtype=self.dtype,
                    shape=torch_cross_attn.q_norm.weight.shape,
                )
            ),
            k_norm=RMSNorm(
                Weight(
                    name="k_norm",
                    dtype=self.dtype,
                    shape=torch_cross_attn.k_norm.weight.shape,
                )
            ),
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_attention_states: TensorValue,
        cross_input_row_offsets: TensorValue,
        *fetch_args: TensorValue,
    ) -> TensorValue:
        """Builds the cross attention model graph."""
        kv_collection = self.fetch(*fetch_args)
        return self.cross_attention(
            hidden_states,
            hidden_input_row_offsets,
            hidden_max_seq_len,
            cross_attention_states,
            cross_input_row_offsets,
            kv_collection,
        )


@pytest.mark.parametrize(
    "hidden_seq_lens",
    [
        [10, 4],
        [1, 2],
    ],
)
def test_cross_attention(
    session: InferenceSession, hidden_seq_lens: list[int]
) -> None:
    # Globally disable saving activations for backprop.
    torch.set_grad_enabled(False)

    num_tiles = 4
    # image_dim**2 // patch_dim**2 + 1 (cls token)
    num_vision_tokens = 1025
    cross_seq_len = num_tiles * num_vision_tokens
    config = MllamaTextConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        rope_theta=10000.0,
        max_position_embeddings=8192,
    )
    # Set up PyTorch attention layer.
    torch_dtype = torch.float32
    torch_cross_attn = MllamaTextCrossSdpaAttention(config, layer_idx=0)
    torch_cross_attn.to(torch_dtype)

    # Set up MAX graph attention layer.
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    batch_size = len(hidden_seq_lens)

    dtype = DType.float32
    hidden_states_type = TensorType(
        dtype, ["total_seq_len", config.hidden_size]
    )
    cross_attention_states_type = TensorType(
        dtype, shape=[batch_size * cross_seq_len, config.hidden_size]
    )

    input_row_offsets_type = TensorType(DType.uint32, [batch_size + 1])
    hidden_max_seq_len_type = TensorType(DType.uint32, [1])

    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
    )
    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=batch_size,
        max_seq_len=config.max_position_embeddings,
        num_layers=config.num_hidden_layers,
        session=session,
        devices=[CPU()],
    )

    # Phase 1: op staging.

    # Construct and compile the MAX graph cross attention.
    graph = Graph(
        "test_cross_attn",
        forward=CrossAttentionModel(config, kv_params, torch_cross_attn, dtype),
        input_types=[
            # NOTE: 2 input row offsets: for hidden and cross attention states.
            hidden_states_type,
            input_row_offsets_type,
            hidden_max_seq_len_type,
            cross_attention_states_type,
            input_row_offsets_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Phase 2: model compilation and weight initialization.

    # Map torch weight values to their MAX graph counterparts.
    weights_registry = {
        "wq": torch_cross_attn.q_proj.weight.detach(),
        "wk": torch_cross_attn.k_proj.weight.detach(),
        "wv": torch_cross_attn.v_proj.weight.detach(),
        "wo": torch_cross_attn.o_proj.weight.detach(),
        "q_norm": torch_cross_attn.q_norm.weight.detach(),
        "k_norm": torch_cross_attn.k_norm.weight.detach(),
    }
    cross_attn_model = session.load(graph, weights_registry=weights_registry)

    # Phase 3: execution.

    seq_ids = kv_manager.claim(n=batch_size)
    # Use cross states sequence length when fetching from the KV manager since
    # KV are cross states.
    seq_ids_and_prompts = {
        s: np.array([FAKE_TOKEN] * cross_seq_len) for i, s in enumerate(seq_ids)
    }
    kv_cache_inputs = kv_manager.fetch(seq_ids_and_prompts)[0]

    # Initialize model inputs.
    total_seq_len = sum(hidden_seq_lens)
    hidden_states = torch.randn(
        [total_seq_len, config.hidden_size], dtype=torch_dtype
    )
    cross_attention_states = torch.randn(
        cross_attention_states_type.shape.static_dims, dtype=torch_dtype
    )
    hidden_input_row_offsets = torch.tensor(
        [0, *np.cumsum(hidden_seq_lens)], dtype=torch.uint32
    )
    cross_input_row_offsets = torch.tensor(
        [i * num_tiles * num_vision_tokens for i in range(batch_size + 1)],
        dtype=torch.uint32,
    )
    hidden_max_seq_len = np.array([max(hidden_seq_lens)], dtype=np.uint32)

    predicted = cross_attn_model(
        hidden_states,
        hidden_input_row_offsets,
        hidden_max_seq_len,
        cross_attention_states,
        cross_input_row_offsets,
        *kv_cache_inputs,
    )[0]
    assert isinstance(predicted, Tensor)

    # Marshal extra inputs for torch.
    # Create padded inputs since the torch model doesn't support ragged
    # tensors.
    hidden_states_padded = torch.zeros(
        size=[batch_size, max(hidden_seq_lens), config.hidden_size],
        dtype=torch_dtype,
    )
    # Convert to int since torch can't subtract uint32.
    hidden_input_row_offsets = hidden_input_row_offsets.to(dtype=torch.int32)
    for batch_idx, (start, stop) in enumerate(
        zip(hidden_input_row_offsets[:-1], hidden_input_row_offsets[1:])
    ):
        hidden_states_padded[batch_idx, : stop - start] = hidden_states[
            start:stop
        ]

    attention_mask = torch.ones(
        [1, 1, max(hidden_seq_lens), num_tiles * num_vision_tokens],
        dtype=torch.bool,
    )
    expected = (
        torch_cross_attn(
            hidden_states=hidden_states_padded,
            cross_attention_states=cross_attention_states.reshape(
                [batch_size, num_tiles * num_vision_tokens, config.hidden_size]
            ),
            attention_mask=attention_mask,
        )[0]
        .detach()
        .numpy()
    )
    expected_ragged = np.empty(
        shape=[total_seq_len, config.hidden_size], dtype=dtype.to_numpy()
    )
    for batch_idx, (start, stop) in enumerate(
        zip(hidden_input_row_offsets[:-1], hidden_input_row_offsets[1:])
    ):
        expected_ragged[start:stop] = expected[batch_idx, : stop - start]

    # Compare the outputs.
    assert is_euclidean_distance_close(
        predicted.to_numpy(), expected_ragged, rtol=1e-4
    )
