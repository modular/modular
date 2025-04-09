# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test Llama vision language model self attention layer."""

import math
from typing import Any

import numpy as np
import pytest
import torch
from context_utils import create_text_context
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.nn import AttentionWithRopeQKV, Linear, OptimizedRotaryEmbedding
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    load_kv_manager,
)
from modular_graph_test import are_all_tensor_values, modular_graph_test
from torch import nn
from transformers import DynamicCache
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.models.mllama.modeling_mllama import (
    MllamaTextSelfSdpaAttention,
)

BATCH_SIZE = 1
ACCURACY_RTOL = 1e-10
ACCURACY_ATOL = 1e-10


class TorchAttention(nn.Module):
    def __init__(
        self,
        config: MllamaTextConfig,
        start_pos: int,
        seq_len: int,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.attention = MllamaTextSelfSdpaAttention(
            self.config, layer_idx=layer_idx
        )
        self.start_pos = start_pos
        self.seq_len = seq_len

    def forward(self, x, wq, wk, wv, wo, position_embeddings):
        self.attention.load_state_dict(
            {
                "q_proj.weight": wq,
                "k_proj.weight": wk,
                "v_proj.weight": wv,
                "o_proj.weight": wo,
            }
        )
        return self.attention(
            hidden_states=x,
            attention_mask=None,
            position_embeddings=position_embeddings,
            past_key_values=DynamicCache(),
        )[0]


def _attention_layer(
    config: MllamaTextConfig, seq_len: int, layer_idx: int
) -> tuple[Graph, Any]:
    dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = dim // n_heads

    dtype = DType.float32
    input_type = TensorType(dtype, [BATCH_SIZE * seq_len, dim])
    wq_type = TensorType(dtype, [n_heads * head_dim, config.hidden_size])
    wk_type = TensorType(dtype, [n_kv_heads * head_dim, config.hidden_size])
    wv_type = TensorType(dtype, [n_kv_heads * head_dim, config.hidden_size])
    wo_type = TensorType(dtype, [config.hidden_size, n_heads * head_dim])
    weight_types = [wq_type, wk_type, wv_type, wo_type]

    input_row_offsets_type = TensorType(
        DType.uint32,
        [BATCH_SIZE + 1],
    )
    session = InferenceSession()

    # define kv_params
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=config.hidden_size // n_kv_heads,
    )
    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=BATCH_SIZE,  # verify this.
        max_seq_len=config.max_position_embeddings,
        num_layers=config.num_hidden_layers,
        session=session,
        devices=[CPU()],
    )

    seq_ids = kv_manager.claim(n=BATCH_SIZE)
    batch = [create_text_context(s, np.empty(seq_len)) for s in seq_ids]
    kv_cache_inputs = kv_manager.fetch(batch)[0]

    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    kv_cache_types = [
        element for tup in kv_manager.input_symbols() for element in tup
    ]

    graph = Graph(
        "self_attn",
        input_types=[input_type]
        + weight_types
        + [input_row_offsets_type]
        + kv_cache_types,
    )

    with graph:
        assert are_all_tensor_values(graph.inputs)
        x, wq, wk, wv, wo, input_row_offsets, *graph_kv_cache_inputs = (
            graph.inputs
        )

        # Get KV Collection
        kv_collection = fetch_op(*graph_kv_cache_inputs)
        # TODO: this should be Llama3RotaryEmbedding with rope scaling params.
        rotary_embedding = OptimizedRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            # TODO: Check if this param value used is correct for "max_seq_len".
            max_seq_len=config.max_position_embeddings,
        )
        attention = AttentionWithRopeQKV(
            n_heads=config.num_attention_heads,
            kv_params=KVCacheParams(
                dtype=dtype,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
            ),
            layer_idx=layer_idx,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=Linear(wo),
            rope=rotary_embedding,
            scale=math.sqrt(1.0 / head_dim),
        )

        graph.output(
            attention(
                x=x,
                kv_collection=kv_collection,
                input_row_offsets=input_row_offsets,
            )
        )
    return graph, kv_cache_inputs


@pytest.mark.parametrize(("start_pos", "seq_len"), [(0, 128)])
def test_self_attention(session, start_pos, seq_len):
    layer_idx = 0

    test_config = MllamaTextConfig(
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=14336,
        max_position_embeddings=512,  # original implementation is 131072.
        num_attention_heads=32,
        num_hidden_layers=5,  # original implementation is 40.
        num_key_value_heads=8,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 256,  # original implementation is 8192.
            "rope_type": "llama3",
        },
    )

    # Set up MAX Graph attention layer.
    layer_graph, kv_cache_inputs = _attention_layer(
        config=test_config, seq_len=seq_len, layer_idx=layer_idx
    )
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_cache_inputs
    )

    prompt_lens = np.array([30])
    assert len(prompt_lens) == BATCH_SIZE
    input_row_offsets = np.array(
        [0, *prompt_lens.cumsum()],
        dtype=cache_lengths.dtype.to_numpy(),
    )

    # Set up PyTorch attention layer.
    torch_attention = TorchAttention(
        config=test_config,
        start_pos=start_pos,
        seq_len=seq_len,
        layer_idx=layer_idx,
    )

    @modular_graph_test(
        session,
        layer_graph,
        static_dims={
            "input_row_offsets": 1,
        },
        provided_inputs={
            5: input_row_offsets,
            6: blocks,
            7: cache_lengths,
            8: lookup_table_tensor,
            9: is_cache_empty_buf,
        },
    )
    def test_correctness(execute, inputs, torch_inputs):
        inputs = list(inputs)
        result = execute(inputs).to_numpy()

        x, wq, wk, wv, wo, *_ = torch_inputs
        # position_embeddings
        partial_tensor = torch.randn(
            BATCH_SIZE, 1, seq_len, dtype=torch.bfloat16
        )
        positional_embeddings = [partial_tensor, partial_tensor]

        # Reshape to match reference implementation with input rank of 3.
        x = x.reshape(BATCH_SIZE, seq_len, test_config.hidden_size)
        expected = torch_attention(x, wq, wk, wv, wo, positional_embeddings)
        expected = (
            expected.reshape(BATCH_SIZE * seq_len, test_config.hidden_size)
            .detach()
            .numpy()
        )

        # TODO(MSDK-1071): Consolidate and figure out how to call
        # assert_allclose(result, expected) to fire again on mismatched
        # tensor values.
        np.testing.assert_allclose(
            result,
            expected,
            atol=ACCURACY_ATOL,
            rtol=ACCURACY_RTOL,
            equal_nan=True,
        )
