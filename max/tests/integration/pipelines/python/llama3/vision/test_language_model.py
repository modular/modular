# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs tests for the Llama3.2 vision language model layer."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import pytest
import torch
from llama_vision.cross_attention_decoder import (
    CrossAttentionDecoderLayer,
    CrossSdpaAttention,
)
from llama_vision.language_model import (
    CausalLanguageModel,
    TextModel,
    instantiate_language_model,
)
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight, ops
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig, SupportedEncoding
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


def weight(
    name: str, weights_array: torch.tensor, weights_registry: dict
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
    num_attention_heads,
    hidden_size,
    num_key_value_heads,
    rms_norm_eps,
    num_hidden_layers,
    cross_attention_layers,
    max_position_embeddings: int,
    rope_theta: float,
    kv_params,
    weights_registry: dict,
    dtype: DType = DType.float32,
):
    rotary_embedding = OptimizedRotaryEmbedding(
        dim=hidden_size,
        n_heads=num_attention_heads,
        theta=rope_theta,
        # TODO: Check if this param value used is correct for "max_seq_len".
        max_seq_len=max_position_embeddings,
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


def generate_test_language_model() -> Graph:
    """
    This helper function generates a test vision model instance for testing purposes.
    """

    pipeline_config = PipelineConfig(
        architecture="MllamaForConditionalGeneration",
        huggingface_repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization_encoding=SupportedEncoding.bfloat16,
        weight_path=[
            Path("model-00001-of-00005.safetensors"),
            Path("model-00002-of-00005.safetensors"),
            Path("model-00003-of-00005.safetensors"),
            Path("model-00004-of-00005.safetensors"),
            Path("model-00005-of-00005.safetensors"),
        ],
    )
    text_config = pipeline_config.huggingface_config.text_config

    weights = pipeline_config.load_weights()
    assert isinstance(
        weights, SafetensorWeights
    ), "only safetensor weights supported currently"

    kv_params = KVCacheParams(
        dtype=pipeline_config.dtype,
        n_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.hidden_size // text_config.num_key_value_heads,
    )
    with Graph("test_llama_vision") as graph:
        print("building vision model...")
        language_model = instantiate_language_model(
            dtype=pipeline_config.dtype,
            hidden_size=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            rope_theta=text_config.rope_theta,
            max_seq_len=512,
            num_hidden_layers=text_config.num_hidden_layers,
            cross_attention_layers=text_config.cross_attention_layers,
            vocab_size=text_config.vocab_size,
            rms_norm_eps=text_config.rms_norm_eps,
            num_key_value_heads=text_config.num_key_value_heads,
            intermediate_size=text_config.intermediate_size,
            kv_params=kv_params,
            weights=weights,
        )

        graph.output(
            ops.constant(
                language_model.kv_params.n_kv_heads, dtype=DType.int32
            ),
        )

        return graph


@pytest.mark.skip("requires internet and is very large")
def test_build_language_model():
    """
    This test is not meant to be run in CI.
    It will require the internet and download over 20gb of weights.
    It is primarily meant to be run as a double check function that the vision model continues to build.
    """

    vision_model = generate_test_language_model()
    assert isinstance(vision_model, Graph)


@pytest.mark.skip("doesnt work")
def test_llama_language_model():
    batch_size = 1
    seq_length = 7
    rms_norm_eps = 1e-5
    cross_attention_layers = [1]
    num_hidden_layers = 2
    max_position_embeddings = 512
    hidden_size = 4096
    num_key_value_heads = 8
    num_attention_heads = 8
    rope_theta = 500000.0

    config_dict = {
        "model_type": "mllama",
        "vocab_size": 99,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": 37,
        "hidden_act": "silu",
        "max_position_embeddings": max_position_embeddings,
        "initializer_range": 0.02,
        "rope_scaling": {"rope_type": "default"},
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "cross_attention_layers": cross_attention_layers,
        "rope_theta": rope_theta,
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
        num_hidden_layers=num_hidden_layers,
        cross_attention_layers=cross_attention_layers,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
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
    input_ids_type = TensorType(
        DType.int64,
        shape=input_ids.shape,  # batch_size, sequence_length
    )
    attention_mask_type = input_ids_type
    input_row_offset_type = TensorType(
        DType.uint32,
        [batch_size + 1],
    )
    input_types = [
        input_ids_type,
        attention_mask_type,
        input_row_offset_type,
    ] + [element for tup in kv_manager.input_symbols() for element in tup]
    with Graph("test_language_model", input_types=input_types) as graph:
        (
            graph_input_ids,
            graph_attention_mask,
            graph_input_row_offset,
            *graph_kv_cache_inputs,
        ) = graph.inputs

        logits = graph_api_model(
            kv_cache_inputs=graph_kv_cache_inputs,
            input_ids=graph_input_ids,
            attention_mask=graph_attention_mask,
            input_row_offset=graph_input_row_offset,
        )
        graph.output(logits)

    compiled = session.load(graph, weights_registry=weights_registry)

    prompt_lens = [30]
    assert len(prompt_lens) == batch_size
    input_row_offset = Tensor(
        [batch_size + 1],
        DType.uint32,
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offset[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offset[batch_size] = running_sum

    # output = compiled.execute(
    #     input_ids, attention_mask, input_row_offset, *kv_cache_inputs
    # )[  # type: ignore
    #     0
    # ].to_numpy()
    #
    # np.testing.assert_allclose(
    #     output,
    #     pytorch_logits.detach().numpy(),
    #     equal_nan=False,  # TODO: flip to True when language_model is correct
    #     rtol=ACCURACY_RTOL,
    #     atol=ACCURACY_ATOL,
    # )
