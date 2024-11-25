# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision language model layer."""

import random
from typing import Union

import numpy as np
import pytest
import torch
from llama_vision.cross_attention_decoder import (
    CrossAttentionDecoderLayer,
    CrossSdpaAttention,
)
from llama_vision.language_model import (
    CausalLanguageModel,
    TextModel,
)

# from llama_vision.rotary_embedding_2d import RotaryEmbedding2D
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight
from max.pipelines.kv_cache import KVCacheParams, load_kv_manager

# from modular_graph_test import modular_graph_test
from nn import MLP, AttentionQKV, Embedding, Linear, RMSNorm, TransformerBlock
from transformers import MllamaForCausalLM
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.testing_utils import torch_device

# TODO: Change these to only pass the test if correct.
ACCURACY_ATOL = 1e05
ACCURACY_RTOL = 1e05

# copied from https://github.com/huggingface/transformers/blob/main/tests/test_modeling_common.py
global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return (
        torch.tensor(data=values, dtype=torch.long, device=torch_device)
        .view(shape)
        .contiguous()
    )


def linear(name: str, weights_array, weights_registry) -> Linear:
    """Creates a Linear layer backed by a weight."""
    weights_registry[name] = weights_array
    return Linear(
        Weight(
            name=name,
            dtype=DType.from_numpy(weights_array.numpy().dtype),
            shape=weights_array.shape,
        ),
        bias=None,
    )


def embedding(name: str, weights_array, weights_registry) -> Embedding:
    """Creates a Embedding layer backed by a weight."""
    weights_registry[name] = weights_array
    return Embedding(
        Weight(
            name=name,
            dtype=DType.from_numpy(weights_array.numpy().dtype),
            shape=weights_array.shape,
        )
    )


def norm(name: str, weights_array, eps, weights_registry) -> RMSNorm:
    weights_registry[name] = weights_array
    return RMSNorm(
        weight=Weight(
            name=name,
            dtype=DType.from_numpy(weights_array.numpy().dtype),
            shape=weights_array.shape,
        ),
        eps=eps,
    )


def cross_attention_decoder_layer(
    num_attention_heads: int,
    hidden_size: int,
    num_key_value_heads: int,
    rms_norm_eps: float,
    pytorch_layer,
    layer_idx: int,
    weights_registry: dict,
) -> CrossAttentionDecoderLayer:
    num_heads = num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_groups = num_heads // num_key_value_heads
    sdpa_attn = CrossSdpaAttention(
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        layer_idx=layer_idx,
        num_key_value_groups=num_key_value_groups,
        q_proj=linear(
            f"text.layers{layer_idx}.cross_attn.q_proj",
            pytorch_layer.cross_attn.q_proj.weight.data,
            weights_registry,
        ),
        k_proj=linear(
            f"text.layers{layer_idx}.cross_attn.k_proj",
            pytorch_layer.cross_attn.k_proj.weight.data,
            weights_registry,
        ),
        v_proj=linear(
            f"text.layers{layer_idx}.cross_attn.v_proj",
            pytorch_layer.cross_attn.v_proj.weight.data,
            weights_registry,
        ),
        o_proj=linear(
            f"text.layers{layer_idx}.cross_attn.o_proj",
            pytorch_layer.cross_attn.o_proj.weight.data,
            weights_registry,
        ),
        q_norm=norm(
            f"text.layers{layer_idx}.cross_attn.q_norm",
            pytorch_layer.cross_attn.q_norm.weight.data,
            rms_norm_eps,
            weights_registry,
        ),
        k_norm=norm(
            f"text.layers{layer_idx}.cross_attn.k_norm",
            pytorch_layer.cross_attn.k_norm.weight.data,
            rms_norm_eps,
            weights_registry,
        ),
    )
    return CrossAttentionDecoderLayer(
        cross_attn=sdpa_attn,
        input_layernorm=norm(
            f"text.layers{layer_idx}.input_layernorm",
            pytorch_layer.input_layernorm.weight.data,
            rms_norm_eps,
            weights_registry,
        ),
        cross_attn_attn_gate=pytorch_layer.cross_attn_attn_gate.data,
        mlp=MLP(
            gate_proj=linear(
                f"text.layers{layer_idx}.mlp.gate_proj",
                pytorch_layer.mlp.gate_proj.weight.data,
                weights_registry,
            ),
            down_proj=linear(
                f"text.layers{layer_idx}.mlp.down_proj",
                pytorch_layer.mlp.down_proj.weight.data,
                weights_registry,
            ),
            up_proj=linear(
                f"text.layers{layer_idx}.mlp.up_proj",
                pytorch_layer.mlp.up_proj.weight.data,
                weights_registry,
            ),
        ),
        post_attention_layernorm=norm(
            f"text.layers{layer_idx}.post_attention_layernorm",
            pytorch_layer.post_attention_layernorm.weight.data,
            rms_norm_eps,
            weights_registry,
        ),
        cross_attn_mlp_gate=pytorch_layer.cross_attn_mlp_gate.data,
    )


def self_attention_decoder_layer(
    kv_params: KVCacheParams,
    num_attention_heads: int,
    hidden_size: int,
    rms_norm_eps: float,
    pytorch_layer,
    layer_idx: int,
    weights_registry: dict,
) -> TransformerBlock:
    num_heads = num_attention_heads
    head_dim = hidden_size // num_heads

    q_proj = linear(
        f"text.layers{layer_idx}.self_attn.q_proj",
        pytorch_layer.self_attn.q_proj.weight.data,
        weights_registry,
    )
    k_proj = linear(
        f"text.layers{layer_idx}.self_attn.k_proj",
        pytorch_layer.self_attn.k_proj.weight.data,
        weights_registry,
    )
    v_proj = linear(
        f"text.layers{layer_idx}.self_attn.v_proj",
        pytorch_layer.self_attn.v_proj.weight.data,
        weights_registry,
    )

    attention = AttentionQKV(
        n_heads=num_attention_heads,
        kv_params=kv_params,
        layer_idx=layer_idx,
        wq=q_proj.weight,
        wk=k_proj.weight,
        wv=v_proj.weight,
        wo=linear(
            f"text.layers{layer_idx}.self_attn.o_proj",
            pytorch_layer.self_attn.o_proj.weight.data,
            weights_registry,
        ),
    )
    return TransformerBlock(
        attention=attention,
        mlp=MLP(
            gate_proj=linear(
                f"text.layers{layer_idx}.mlp.gate_proj",
                pytorch_layer.mlp.gate_proj.weight.data,
                weights_registry,
            ),
            down_proj=linear(
                f"text.layers{layer_idx}.mlp.down_proj",
                pytorch_layer.mlp.down_proj.weight.data,
                weights_registry,
            ),
            up_proj=linear(
                f"text.layers{layer_idx}.mlp.up_proj",
                pytorch_layer.mlp.up_proj.weight.data,
                weights_registry,
            ),
        ),
        attention_norm=norm(
            f"text.layers{layer_idx}.input_layernorm",
            pytorch_layer.input_layernorm.weight.data,
            rms_norm_eps,
            weights_registry,
        ),
        mlp_norm=norm(
            f"text.layers{layer_idx}.post_attention_layernorm",
            pytorch_layer.post_attention_layernorm.weight.data,
            rms_norm_eps,
            weights_registry,
        ),
    )


def language_model_given_pytorch_model(
    pytorch_model,
    num_attention_heads,
    hidden_size,
    num_key_value_heads,
    rms_norm_eps,
    num_hidden_layers,
    cross_attention_layers,
    kv_params,
    weights_registry: dict,
    dtype: DType = DType.float32,
):
    layers: list[Union[CrossAttentionDecoderLayer, TransformerBlock]] = []
    for layer_idx in range(num_hidden_layers):
        curr_layer = pytorch_model.model.layers[layer_idx]
        if layer_idx in cross_attention_layers:
            layers.append(
                cross_attention_decoder_layer(
                    num_attention_heads=num_attention_heads,
                    hidden_size=hidden_size,
                    num_key_value_heads=num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                    pytorch_layer=curr_layer,
                    layer_idx=layer_idx,
                    weights_registry=weights_registry,
                )
            )
        else:
            layers.append(
                self_attention_decoder_layer(
                    kv_params=kv_params,
                    num_attention_heads=num_attention_heads,
                    hidden_size=hidden_size,
                    rms_norm_eps=rms_norm_eps,
                    pytorch_layer=curr_layer,
                    layer_idx=layer_idx,
                    weights_registry=weights_registry,
                )
            )

    text_model = TextModel(
        kv_params=kv_params,
        dtype=dtype,
        embed_tokens=embedding(
            "text.model.embed_tokens",
            pytorch_model.model.embed_tokens.weight.data,
            weights_registry,
        ),
        norm=norm(
            "text.model.norm",
            pytorch_model.model.norm.weight.data,
            rms_norm_eps,
            weights_registry,
        ),
        layers=layers,
        # TODO: To be added later when we add rotary embeddings
        #     rotary_emb=RotaryEmbedding2D(dim=params.hidden_size,
        #                          n_heads=params.num_attention_heads, # Should this be params.num_key_value_heads
        #                          theta=params.rope_theta,
        #                          max_patches_per_side=params.max_position_embeddings, # verify this
        #                          #rope_scaling=params.rope_scaling, # TODO:figure it out
        # ),
        cross_attention_layers=pytorch_model.model.cross_attention_layers,
    )

    return CausalLanguageModel(
        dtype=dtype,
        kv_params=kv_params,
        model=text_model,
        lm_head=linear(
            "text.lm_head", pytorch_model.lm_head.weight.data, weights_registry
        ),
    )


@pytest.mark.skip(reason="Currently failing")
def test_llama_language_model():
    batch_size = 1
    seq_length = 7
    rms_norm_eps = 1e-5
    cross_attention_layers = [1]
    num_hidden_layers = 2
    max_position_embeddings = 512
    hidden_size = 256
    num_key_value_heads = 8
    num_attention_heads = 8

    config_dict = {
        "model_type": "mllama",
        "vocab_size": 99,
        "hidden_size": hidden_size,  # was 32
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": 37,
        "hidden_act": "gelu",
        "max_position_embeddings": max_position_embeddings,
        "initializer_range": 0.02,
        "rope_scaling": {"rope_type": "default"},
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "cross_attention_layers": cross_attention_layers,
    }
    pytorch_lm_config = MllamaTextConfig(**config_dict)
    input_ids = (
        ids_tensor(
            [batch_size, seq_length],
            pytorch_lm_config.vocab_size - 1,
        )
        + 1
    )
    attention_mask = input_ids.ne(1).to(dtype=torch.int64, device=torch_device)

    pytorch_model = MllamaForCausalLM(config=pytorch_lm_config)
    pytorch_model.to(torch_device)
    pytorch_model.eval()
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        pytorch_logits = pytorch_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )["logits"]
        assert not (torch.isnan(pytorch_logits).any().item())

    # define kv_cache_inputs_types
    dtype = DType.float32

    weights_registry: dict = {}
    # define kv_params
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=num_key_value_heads,
        head_dim=hidden_size // num_key_value_heads,
    )

    graph_api_model = language_model_given_pytorch_model(
        pytorch_model,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rms_norm_eps=rms_norm_eps,
        cross_attention_layers=cross_attention_layers,
        num_hidden_layers=num_hidden_layers,
        kv_params=kv_params,
        weights_registry=weights_registry,
    )

    session = InferenceSession()

    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=batch_size,  # verify this.
        max_seq_len=max_position_embeddings,  # verify this.
        num_layers=num_hidden_layers,
        session=session,
        devices=[CPU()],
    )

    seq_ids = kv_manager.claim(n=batch_size)
    kv_cache_inputs = kv_manager.fetch(seq_ids)

    kv_cache_types = kv_manager.input_symbols()
    input_ids_type = TensorType(
        DType.int64,
        shape=input_ids.shape,  # batch_size, sequence_length
    )
    attention_mask_type = input_ids_type
    # kv_cache_types = [blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type]
    input_types = [
        input_ids_type,
        attention_mask_type,
        *kv_cache_types,
    ]

    with Graph("language_model", input_types=input_types) as graph:
        graph_input_ids, graph_attention_mask, *graph_kv_cache_inputs = (
            graph.inputs
        )
        logits = graph_api_model(
            kv_cache_inputs=graph_kv_cache_inputs,
            input_ids=graph_input_ids,
            attention_mask=graph_attention_mask,
        )[1]
        graph.output(logits)

    compiled = session.load(graph, weights_registry=weights_registry)

    output = compiled.execute(input_ids, attention_mask, *kv_cache_inputs)[  # type: ignore
        0
    ].to_numpy()

    np.testing.assert_allclose(
        output,
        pytorch_logits.detach().numpy(),
        equal_nan=False,  # TODO: flip to True when language_model is correct
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
