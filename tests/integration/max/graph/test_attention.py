# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines attention layer."""

import pytest
import torch
from conftest import assert_allclose, modular_graph_test
from max.dtype import DType
from max.graph import Graph, TensorType
from torch import nn
from transformers import StaticCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaSdpaAttention

from nn import Attention, Linear, RotaryEmbedding


class TorchAttention(nn.Module):
    def __init__(self, config, start_pos, seq_len):
        super().__init__()
        self.config = config
        self.attention = LlamaSdpaAttention(self.config, layer_idx=0)
        self.start_pos = start_pos
        self.seq_len = seq_len

    def forward(self, x, attention_mask, k_cache, v_cache, wq, wk, wv, wo):
        # Unsqueeze from (batch, seq_len, post_seq_len) to
        # (batch, nheads, seq_len, post_seq_len).
        attention_mask = attention_mask.unsqueeze(1).tile(
            (
                1,
                self.config.num_attention_heads,
                1,
                1,
            )
        )

        self.attention.load_state_dict(
            {
                "q_proj.weight": wq,
                "k_proj.weight": wk,
                "v_proj.weight": wv,
                "o_proj.weight": wo,
            }
        )
        cache = StaticCache(
            self.config,
            max_batch_size=k_cache.shape[2],
            max_cache_len=k_cache.shape[0],
            device=torch.get_default_device(),
        )

        # MAX KV cache has shape:
        #   [start_pos, n_layers, batch_size, n_kv_heads, head_dim]
        # Torch cache stores it as:
        #   [max_batch_size, n_kv_heads, max_cache_len, head_dim] per layer
        k_cache = k_cache[:, 0, ...].movedim(0, 2)
        v_cache = v_cache[:, 0, ...].movedim(0, 2)

        cache.update(
            k_cache,
            v_cache,
            0,
            cache_kwargs={"cache_position": None},
        )
        positional_ids = torch.arange(
            self.start_pos,
            self.start_pos + self.seq_len,
            device=x.device,
        )
        positional_ids = positional_ids.unsqueeze(0)
        position_embeddings = self.attention.rotary_emb(x, positional_ids)

        return self.attention(
            x,
            attention_mask,
            past_key_values=cache,
            position_embeddings=position_embeddings,
        )[0]


def _attention_layer(config: LlamaConfig):
    dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = dim // n_heads
    theta = config.rope_theta
    max_seq_len = config.max_position_embeddings

    input_dtype = DType.float32
    input_type = TensorType(input_dtype, ["batch_size", "seq_len", dim])
    attn_mask_type = TensorType(
        DType.float32, ["batch_size", "seq_len", "post_seq_len"]
    )
    cache_type = TensorType(
        DType.float32,
        shape=[
            "start_pos",
            1,
            "batch_size",
            n_kv_heads,
            head_dim,
        ],
    )
    attn_input_types = [input_type, attn_mask_type, cache_type, cache_type]

    wq_type = TensorType(input_dtype, [n_heads * head_dim, dim])
    wk_type = TensorType(input_dtype, [n_kv_heads * head_dim, dim])
    wv_type = TensorType(input_dtype, [n_kv_heads * head_dim, dim])
    wo_type = TensorType(input_dtype, [dim, n_heads * head_dim])
    weight_types = [wq_type, wk_type, wv_type, wo_type]

    graph = Graph(
        "attn",
        input_types=attn_input_types + weight_types,
    )

    with graph:
        x, attn_mask, k_cache, v_cache, wq, wk, wv, wo = graph.inputs
        graph.output(
            Attention(
                n_heads,
                n_kv_heads,
                head_dim,
                dim,
                Linear(wq),
                Linear(wk),
                Linear(wv),
                Linear(wo),
                rope=RotaryEmbedding(
                    dim=dim,
                    n_heads=n_heads,
                    theta=theta,
                    max_seq_len=max_seq_len,
                    rope_scaling=None,
                ),
            )(x, attn_mask, k_cache, v_cache)[0]
        )
    return graph


@pytest.mark.parametrize(
    "start_pos,seq_len",
    [
        (0, 10),
        (9, 1),
    ],
)
def test_attention(session, start_pos, seq_len):
    config = LlamaConfig(
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_theta=10000.0,
    )
    # Set up pytorch attention layer.
    torch_attention = TorchAttention(config, start_pos, seq_len)

    # Set up max graph attention layer.
    layer_graph = _attention_layer(config)

    @modular_graph_test(
        session,
        layer_graph,
        static_dims={
            "start_pos": start_pos,
            "seq_len": seq_len,
            "post_seq_len": start_pos + seq_len,
        },
    )
    def test_correctness(execute, inputs, torch_inputs):
        inputs = list(inputs)
        result = execute(inputs)
        expected = torch_attention(*torch_inputs).detach().numpy()
        assert_allclose(result, expected)
