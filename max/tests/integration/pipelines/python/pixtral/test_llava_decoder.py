# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import math
import random

import hf_repo_lock
import numpy as np
import pytest
import torch
from context_utils import create_text_context
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops
from max.nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    OptimizedRotaryEmbedding,
    RMSNorm,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    load_kv_manager,
)
from max.pipelines.architectures.pixtral.llava.llava_decoder import (
    Transformer as LLavaTransformer,
)
from max.pipelines.lib import generate_local_model_path
from transformers import (
    LlavaForConditionalGeneration,
    MistralConfig,
    MistralForCausalLM,
)
from transformers.testing_utils import torch_device

# TODO: Change these to only pass the test if correct.
ACCURACY_ATOL = 1e-05
ACCURACY_RTOL = 1e-05

REPO_ID = "mistral-community/pixtral-12b"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)
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


@pytest.fixture
def pytorch_mistral_and_config() -> tuple[MistralForCausalLM, MistralConfig]:
    """Loads the language model in PyTorch pixtral model. Using the community
    pixtral version. It also returns pixtral_model.config.text_config
    """
    local_model_path = generate_local_model_path(REPO_ID, REVISION)
    model = LlavaForConditionalGeneration.from_pretrained(local_model_path)
    return model.language_model, model.config.text_config


def mistral_given_pytorch_mistral(
    pytorch_model: MistralForCausalLM,
    config: MistralConfig,
    kv_params: KVCacheParams,
    kv_collection_constructor: FetchContinuousBatchingKVCacheCollection,
    weights_registry: dict,
) -> LLavaTransformer:
    # refer to:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L682
    # kv_params is passed to AttentionWithRope
    # kv_collection_constructor is passed to LLavaTransformer

    #######################Init weights with pytorch weights ###################
    mlp_gate_weights = [
        pytorch_model.model.layers[i].mlp.gate_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    mlp_up_weights = [
        pytorch_model.model.layers[i].mlp.up_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    mlp_down_weights = [
        pytorch_model.model.layers[i].mlp.down_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]

    ############ Define Graph-API layers with weights with pytorch #############
    def linear(name: str, array) -> Linear:
        """Creates a Linear layer backed by a weight."""
        weights_registry[name] = array
        return Linear(
            Weight(
                name=name,
                dtype=DType.from_numpy(array.numpy().dtype),
                shape=array.shape,
                device=DeviceRef.CPU(),
            )
        )

    def _weight(name: str, array) -> Weight:
        weights_registry[name] = array
        return Weight(
            name=name,
            dtype=DType.from_numpy(array.numpy().dtype),
            shape=array.shape,
            device=DeviceRef.CPU(),
        )

    def attention(kv_params, rope: OptimizedRotaryEmbedding, layer_idx: int):
        wq = ops.transpose(
            _weight(
                f"text.wq_weights_{layer_idx}",
                pytorch_model.model.layers[
                    layer_idx
                ].self_attn.q_proj.weight.data,
            ),
            0,
            1,
        )
        wk = ops.transpose(
            _weight(
                f"text.wk_weights_{layer_idx}",
                pytorch_model.model.layers[
                    layer_idx
                ].self_attn.k_proj.weight.data,
            ),
            0,
            1,
        )
        wv = ops.transpose(
            _weight(
                f"text.wv_weights_{layer_idx}",
                pytorch_model.model.layers[
                    layer_idx
                ].self_attn.v_proj.weight.data,
            ),
            0,
            1,
        )
        wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

        return AttentionWithRope(
            n_heads=config.num_attention_heads,
            kv_params=kv_params,
            wqkv=wqkv,
            wo=linear(
                name=f"text.wo_weights_{i}",
                array=pytorch_model.model.layers[
                    layer_idx
                ].self_attn.o_proj.weight.data,
            ),
            rope=rope,
            layer_idx=ops.constant(layer_idx, DType.uint32),
            scale=math.sqrt(1.0 / kv_params.head_dim),
        )

    rope = OptimizedRotaryEmbedding(
        dim=config.num_attention_heads * config.head_dim,
        n_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        theta=config.rope_theta,
        max_seq_len=config.max_length,
        device=DeviceRef.CPU(),
    )

    transformer_layers = []
    for i in range(config.num_hidden_layers):
        gate_proj = linear(
            name=f"text.mlp_gate_weights_{i}", array=mlp_gate_weights[i]
        )
        down_proj = linear(
            name=f"text.mlp_down_weights_{i}", array=mlp_down_weights[i]
        )
        up_proj = linear(
            name=f"text.mlp_up_weights_{i}", array=mlp_up_weights[i]
        )

        layer = TransformerBlock(
            attention=attention(kv_params, rope, i),
            mlp=MLP(gate_proj, down_proj, up_proj),
            attention_norm=RMSNorm(
                pytorch_model.model.layers[
                    i
                ].post_attention_layernorm.weight.data,
                config.rms_norm_eps,
            ),
            mlp_norm=RMSNorm(
                pytorch_model.model.layers[i].input_layernorm.weight.data,
                config.rms_norm_eps,
            ),
        )
        transformer_layers.append(layer)

    norm_layer = RMSNorm(
        pytorch_model.model.norm.weight.data, config.rms_norm_eps
    )
    embedding_layer = Embedding(
        _weight(
            "text.embed_tokens", pytorch_model.model.embed_tokens.weight.data
        ),
        device=DeviceRef.CPU(),
    )
    output_linear = linear(
        "text.output_linear", pytorch_model.lm_head.weight.data
    )

    model = LLavaTransformer(
        config.hidden_size,
        config.num_attention_heads,
        transformer_layers,
        norm_layer,
        output_linear,
        embedding_layer,
        kv_params,
        kv_collection_constructor,
    )
    return model


@pytest.mark.skip("doesnt work and needs refactoring similar to llama vision")
def test_llava_mistral_decoder(pytorch_mistral_and_config):
    batch_size = 1
    seq_length = 7

    pytorch_model, pytorch_config = pytorch_mistral_and_config
    pytorch_model.to(torch_device)
    pytorch_model.eval()

    # Prep torch inputs
    input_ids = (
        ids_tensor([batch_size, seq_length], pytorch_config.vocab_size - 1) + 1
    )

    attention_mask = input_ids.ne(1).to(dtype=torch.int64, device=torch_device)

    with torch.autocast(device_type="cuda", dtype=torch.float32):
        pytorch_logits = pytorch_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )["logits"]
        assert not (torch.isnan(pytorch_logits).any().item())

    # prep graph API inputs
    # embeddings of text and image tokens [batch_size, n_tokens_and_patches, hidden_dim] = [1, 7, 5120]
    embeds = pytorch_model.model.embed_tokens(input_ids)
    print("embeddings generated by pytorch model", embeds.shape)

    # define kv_cache_inputs_types
    dtype = DType.float32

    # define kv_params
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=pytorch_config.num_key_value_heads,
        head_dim=pytorch_config.head_dim,
    )

    kv_collection_constructor = FetchContinuousBatchingKVCacheCollection(
        kv_params
    )

    weights_registry: dict = {}

    session = InferenceSession()

    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=batch_size,  # verify this.
        max_seq_len=pytorch_config.max_position_embeddings,  # verify this.
        num_layers=pytorch_config.num_hidden_layers,
        session=session,
        devices=[Accelerator()],
    )

    seq_ids = kv_manager.claim(n=batch_size)
    batch = [create_text_context(s, np.empty(seq_length)) for s in seq_ids]
    _ = kv_manager.fetch(batch)[0]

    embeds_type = TensorType(
        DType.float32,
        shape=embeds.shape,  # [batch_size, n_tokens_and_patches, hidden_dim]
        device=DeviceRef.CPU(),
    )

    input_row_offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=DeviceRef.CPU(),
    )

    input_types = [
        embeds_type,
        input_row_offsets_type,
    ] + [element for tup in kv_manager.input_symbols() for element in tup]

    with Graph("test_llava_decoder_llm", input_types=input_types) as graph:
        graph_embeds, graph_input_row_offsets, *graph_kv_cache_inputs = (
            graph.inputs
        )

        graph_api_model = mistral_given_pytorch_mistral(
            pytorch_model=pytorch_model,
            config=pytorch_config,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_constructor,
            weights_registry=weights_registry,
        )
    """

        logits = graph_api_model(
            kv_cache_inputs=graph_kv_cache_inputs,
            embeds=graph_embeds,
            input_row_offsets=graph_input_row_offsets,
        )
        graph.output(logits)

    compiled = session.load(graph, weights_registry=weights_registry)

    prompt_lens = [30]
    assert len(prompt_lens) == batch_size
    input_row_offsets = Tensor(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    # graph_api_logits = compiled.execute(embeds, input_row_offsets, *kv_cache_inputs)[
    #     0
    # ].to_numpy()

    # np.testing.assert_allclose(
    #     graph_api_logits,
    #     pytorch_logits.detach().numpy(),
    #     equal_nan=False,  # TODO: flip to True when language_model is correct
    #     rtol=ACCURACY_RTOL,
    #     atol=ACCURACY_ATOL,
    # )
    """
