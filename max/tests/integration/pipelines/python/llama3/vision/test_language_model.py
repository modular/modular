# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs tests for the Llama3.2 vision language model layer."""

from __future__ import annotations

from typing import Union

import numpy as np
import pytest
import torch
from llama_vision.cross_attention_decoder import (
    CrossAttentionDecoderLayer,
    CrossSdpaAttention,
)
from llama_vision.language_model import CausalLanguageModel, TextModel
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight
from max.pipelines.kv_cache import KVCacheParams, load_kv_manager
from nn import (
    MLP,
    AttentionWithRopeQKV,
    Embedding,
    Linear,
    OptimizedRotaryEmbedding,
    RMSNorm,
    TransformerBlock,
)
from transformers import MllamaForCausalLM
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.testing_utils import torch_device

FAKE_TOKEN = 999


def weight(
    name: str, weights_array: torch.Tensor, weights_registry: dict
) -> Weight:
    """Creates a Linear layer backed by a weight."""
    weights_registry[name] = weights_array
    return Weight(
        name=name,
        dtype=DType.from_numpy(weights_array.numpy().dtype),
        shape=weights_array.shape,
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
    kv_params: KVCacheParams,
    num_attention_heads: int,
    rms_norm_eps: float,
    pytorch_layer: torch.nn.Module,
    layer_idx: int,
    weights_registry: dict,
) -> CrossAttentionDecoderLayer:
    num_heads = num_attention_heads
    sdpa_attn = CrossSdpaAttention(
        n_heads=num_heads,
        kv_params=kv_params,
        layer_idx=layer_idx,
        q_proj=linear(
            f"text.layers{layer_idx}.cross_attn.q_proj",
            pytorch_layer.cross_attn.q_proj.weight.data,
            weights_registry,
        ),
        wk=weight(
            f"text.layers{layer_idx}.cross_attn.k_proj",
            pytorch_layer.cross_attn.k_proj.weight.data,
            weights_registry,
        ),
        wv=weight(
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
            name=f"text.layers{layer_idx}.input_layernorm",
            weights_array=pytorch_layer.input_layernorm.weight.data,
            eps=rms_norm_eps,
            weights_registry=weights_registry,
        ),
        cross_attn_attn_gate=weight(
            name=f"text.layers{layer_idx}.cross_attn_attn_gate",
            weights_array=pytorch_layer.cross_attn_attn_gate.data,
            weights_registry=weights_registry,
        ),
        mlp=MLP(
            gate_proj=linear(
                name=f"text.layers{layer_idx}.mlp.gate_proj",
                weights_array=pytorch_layer.mlp.gate_proj.weight.data,
                weights_registry=weights_registry,
            ),
            down_proj=linear(
                name=f"text.layers{layer_idx}.mlp.down_proj",
                weights_array=pytorch_layer.mlp.down_proj.weight.data,
                weights_registry=weights_registry,
            ),
            up_proj=linear(
                name=f"text.layers{layer_idx}.mlp.up_proj",
                weights_array=pytorch_layer.mlp.up_proj.weight.data,
                weights_registry=weights_registry,
            ),
        ),
        post_attention_layernorm=norm(
            name=f"text.layers{layer_idx}.post_attention_layernorm",
            weights_array=pytorch_layer.post_attention_layernorm.weight.data,
            eps=rms_norm_eps,
            weights_registry=weights_registry,
        ),
        cross_attn_mlp_gate=weight(
            name=f"text.layers{layer_idx}.cross_attn_mlp_gate",
            weights_array=pytorch_layer.cross_attn_mlp_gate.data,
            weights_registry=weights_registry,
        ),
    )


def self_attention_decoder_layer(
    kv_params: KVCacheParams,
    num_attention_heads: int,
    hidden_size: int,
    rms_norm_eps: float,
    pytorch_layer,
    layer_idx: int,
    weights_registry: dict,
    rotary_embedding: OptimizedRotaryEmbedding,
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

    attention = AttentionWithRopeQKV(
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
        rope=rotary_embedding,
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
    num_attention_heads: int,
    hidden_size: int,
    rms_norm_eps: float,
    num_hidden_layers: int,
    cross_attention_layers: list[int],
    max_seq_len: int,
    rope_theta: float,
    kv_params: KVCacheParams,
    weights_registry: dict,
    dtype: DType,
):
    rotary_embedding = OptimizedRotaryEmbedding(
        dim=hidden_size,
        n_heads=num_attention_heads,
        theta=rope_theta,
        # TODO: Check if this param value used is correct for "max_seq_len".
        max_seq_len=max_seq_len,
        # TODO: Figure out how we want to pass this
        # rope_scaling=params.rope_scaling,
    )

    layers: list[Union[CrossAttentionDecoderLayer, TransformerBlock]] = []
    for layer_idx in range(num_hidden_layers):
        curr_layer = pytorch_model.model.layers[layer_idx]
        if layer_idx in cross_attention_layers:
            layers.append(
                cross_attention_decoder_layer(
                    kv_params=kv_params,
                    num_attention_heads=num_attention_heads,
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
                    rotary_embedding=rotary_embedding,
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
        cross_attention_layers=pytorch_model.model.cross_attention_layers,
        rotary_emb=rotary_embedding,
    )

    return CausalLanguageModel(
        dtype=dtype,
        kv_params=kv_params,
        model=text_model,
        lm_head=linear(
            "text.lm_head", pytorch_model.lm_head.weight.data, weights_registry
        ),
    )


@pytest.mark.parametrize(
    # num_vision_tokens = image_dim**2 // patch_dim**2 + 1 (cls token)
    "seq_lens,num_tiles,num_vision_tokens",
    [
        ([7, 512], 4, 1025),
    ],
)
def test_llama_language_model(
    session: InferenceSession,
    seq_lens: list[int],
    num_tiles: int,
    num_vision_tokens: int,
):
    config_dict = {
        "bos_token_id": 128000,
        "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38],
        "dropout": 0,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 1024,  # original: 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,  # original: 131072
        "model_type": "mllama_text_model",
        "num_attention_heads": 8,  # original: 32
        "num_hidden_layers": 4,  # original: 40
        "num_key_value_heads": 8,
        "pad_token_id": 128004,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 511,  # original: 8192
            "rope_type": "llama3",
        },
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",  # original: "bfloat16"
        "use_cache": True,
        "vocab_size": 128256,
    }

    config = MllamaTextConfig(**config_dict)

    pytorch_model = MllamaForCausalLM(config=config)

    dtype = DType.float32

    # define kv_params
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_key_value_heads,
    )

    weights_registry: dict = {}
    graph_api_model = language_model_given_pytorch_model(
        pytorch_model,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        rms_norm_eps=config.rms_norm_eps,
        num_hidden_layers=config.num_hidden_layers,
        cross_attention_layers=config.cross_attention_layers,
        max_seq_len=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        kv_params=kv_params,
        weights_registry=weights_registry,
        dtype=dtype,
    )

    session = InferenceSession(devices=[CPU()])

    batch_size = len(seq_lens)
    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=batch_size,  # verify this.
        max_seq_len=config.max_position_embeddings,
        num_layers=config.num_hidden_layers,
        session=session,
        devices=[CPU()],
    )

    pytorch_input_ids = torch.randint(
        0,
        config.vocab_size - 1,
        [batch_size, max(seq_lens)],
        dtype=torch.long,
    )

    pytorch_attention_mask = pytorch_input_ids.ne(1).to(
        dtype=torch.int64, device=torch_device
    )

    # Define graph API input types
    input_ids_type = TensorType(
        DType.int64,
        shape=[batch_size * max(seq_lens)],
    )
    cross_attention_states_type = TensorType(
        dtype,
        shape=[batch_size * num_tiles * num_vision_tokens, config.hidden_size],
    )
    hidden_input_row_offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
    )
    cross_input_row_offsets_type = hidden_input_row_offsets_type

    input_types = [
        element for tup in kv_manager.input_symbols() for element in tup
    ] + [
        input_ids_type,
        hidden_input_row_offsets_type,
        cross_attention_states_type,
        cross_input_row_offsets_type,
    ]
    with Graph("test_language_model", input_types=input_types) as graph:
        (
            *graph_kv_cache_inputs,
            graph_input_ids,
            graph_hidden_input_row_offsets,
            graph_cross_attention_states,
            graph_cross_input_row_offsets,
        ) = graph.inputs

        logits = graph_api_model(
            kv_cache_inputs=tuple(graph_kv_cache_inputs),
            input_ids=graph_input_ids,
            hidden_input_row_offsets=graph_hidden_input_row_offsets,
            cross_attention_states=graph_cross_attention_states,
            cross_input_row_offsets=graph_cross_input_row_offsets,
        )
        graph.output(logits)

    compiled = session.load(graph, weights_registry=weights_registry)

    seq_ids = kv_manager.claim(n=batch_size)
    seq_ids_and_prompts = {
        s: np.array([FAKE_TOKEN] * seq_lens[i]) for i, s in enumerate(seq_ids)
    }
    kv_cache_inputs = kv_manager.fetch(seq_ids_and_prompts)[0]

    torch_dtype = torch.float32
    # Initialize model inputs.
    cross_attention_states = torch.randn(
        cross_attention_states_type.shape.static_dims, dtype=torch_dtype
    )
    hidden_input_row_offsets = torch.tensor(
        [0, *np.cumsum(seq_lens)], dtype=torch.uint32
    )
    cross_input_row_offsets = torch.tensor(
        [i for i in range(batch_size + 1)],
        dtype=torch.uint32,
    )

    # predicted = compiled.execute(
    #     *kv_cache_inputs,
    #     pytorch_input_ids.to(dtype=torch.int64),
    #     hidden_input_row_offsets,
    #     cross_attention_states,
    #     cross_input_row_offsets,
    # )[0]
    # assert isinstance(predicted, Tensor)

    expected = pytorch_model(
        input_ids=pytorch_input_ids,
        attention_mask=pytorch_attention_mask,
        return_dict=True,
    )["logits"]
    assert not (torch.isnan(expected).any().item())
    expected = expected.detach().numpy()

    # Compare the outputs.
    # np.testing.assert_array_equal(predicted.to_numpy().shape, expected.shape)

    # assert is_euclidean_distance_close(
    #     result=predicted.to_numpy(), expected=expected, rtol=1e-4
    # )
