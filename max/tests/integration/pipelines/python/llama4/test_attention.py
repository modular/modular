# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn import OptimizedRotaryEmbedding
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from max.pipelines import KVCacheConfig
from max.pipelines.architectures.llama4.layers.attention import (
    _Llama4TextAttention,
)
from max.pipelines.architectures.llama4.model_config import (
    Llama4Config as MaxLlama4Config,
)
from test_common.context_utils import create_text_context
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextModel,
)

# TODO(KERN-1729): this is load bearing and MAX_SEQ_LEN = 100 fails on AMD GPU.
MAX_SEQ_LEN = 64


@torch.no_grad()
def generate_torch_outputs(
    text_config: Llama4TextConfig,
    input_tensor: torch.Tensor,
    weights: dict[str, torch.Tensor],
    dtype: DType,
    device: torch.device,
) -> list[torch.Tensor]:
    """Runs the Transformers Llama4 attention layer.

    Returns the outputs:
    1) Layer with rope
    2) Attention without rope but with attention tuning
    """
    # Construct full Llama4 text model to help with setting up inputs
    # to the attention layers.
    model = Llama4TextModel(text_config)
    assert model.layers[0].self_attn.use_rope
    # Hack - Transformers `LLama4TextAttention` doesn't actually use the config
    # to configure whether the layer uses rope. Set this manually.
    model.layers[1].self_attn.use_rope = False

    # Create layer inputs.
    input_tensor = input_tensor.to(device)
    attention_mask = torch.ones(
        [1, input_tensor.shape[1]], dtype=torch.int64
    ).to(device)
    cache_position = torch.arange(0, input_tensor.shape[1], device=device)

    rotary_emb = model.rotary_emb
    position_ids = cache_position.unsqueeze(0)
    freq_cis = rotary_emb(input_tensor, position_ids)

    causal_mask, chunk_causal_mask = model._update_causal_mask(
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values=None,
        output_attentions=False,
    )

    outputs = []
    for layer_idx in (0, 1):
        if layer_idx == 0:
            continue
        layer = model.layers[layer_idx].self_attn.to(dtype).to(device)
        layer.training = False

        # Update attention weights
        for name, param in layer.named_parameters():
            param.data = weights[name].to(device)

        outputs.append(
            layer(
                input_tensor,
                attention_mask=causal_mask,
                chunk_causal_mask=chunk_causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=freq_cis,
            )[0]
        )
    return outputs


def unflatten_kv_inputs(
    kv_manager: KVCacheManager,
    kv_params: KVCacheParams,
    kv_inputs_flat: Sequence[TensorValue],
) -> list[tuple[TensorValue, ...]]:
    n_devices = kv_params.n_devices
    fetch_types = kv_manager.input_symbols()[0]
    len_of_kv_tuple_per_dev = len(list(fetch_types))
    kv_caches_per_dev = [
        tuple(
            kv_inputs_flat[
                i * len_of_kv_tuple_per_dev : (i + 1) * len_of_kv_tuple_per_dev
            ]
        )
        for i in range(n_devices)
    ]
    return kv_caches_per_dev


def generate_max_outputs(
    config: Llama4Config,
    input_tensor: torch.Tensor,
    weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> list[torch.Tensor]:
    """Runs the MAX Llama4 attention layer.

    Returns the outputs:
    1) Layer with rope
    2) Attention without rope but with attention tuning
    """
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    input_seq_len = input_tensor.shape[1]

    state_dict = {}
    for weight_name, value in weights.items():
        if weight_name.startswith("self_attn."):
            weight_name = weight_name[len("self_attn.") :]
        state_dict[weight_name] = value.cpu()

    text_config = config.text_config
    kv_cache_config = KVCacheConfig(cache_strategy=KVCacheStrategy.PAGED)
    kv_params = MaxLlama4Config.get_kv_params(
        config,
        1,
        kv_cache_config,
        dtype,
    )
    kv_collection_constructor = FetchPagedKVCacheCollection(
        kv_params, num_layers=text_config.num_hidden_layers
    )

    session = InferenceSession(devices=[Accelerator(0)])
    outputs = []
    for layer_idx, use_rope in enumerate([True, False]):
        if layer_idx == 0:
            continue
        attention = _Llama4TextAttention(
            rope=OptimizedRotaryEmbedding(
                text_config.hidden_size,
                text_config.num_attention_heads,
                text_config.rope_theta,
                MAX_SEQ_LEN,
                interleaved=True,
            ),
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            hidden_size=text_config.hidden_size,
            kv_params=kv_params,
            dtype=dtype,
            attn_temperature_tuning=text_config.attn_temperature_tuning,
            floor_scale=text_config.floor_scale,
            attn_scale=text_config.attn_scale,
            devices=[device_ref],
            use_qk_norm=True,
            layer_idx=layer_idx,
            use_rope=use_rope,
        )
        attention.load_state_dict(state_dict)

        # Set up blank KV cache.
        kv_manager = load_kv_manager(
            params=kv_params,
            max_batch_size=1,
            max_seq_len=MAX_SEQ_LEN,
            num_layers=MaxLlama4Config.get_num_layers(
                huggingface_config=config
            ),
            devices=[device],
            available_cache_memory=10 * 1024 * 1024 * 1024,
            page_size=kv_cache_config.kv_cache_page_size,
            session=session,
        )

        # Construct input types.
        input_type = TensorType(
            dtype,
            ["total_seq_len", text_config.hidden_size],
            device=device_ref,
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        cache_positions_type = TensorType(
            DType.uint32,
            ["total_seq_len"],
            device=device_ref,
        )
        kv_cache_args = kv_manager.input_symbols()
        flattened_kv_types = [
            kv_type for sublist in kv_cache_args for kv_type in sublist
        ]

        # Build graph.
        with Graph(
            "Llama4Attention",
            input_types=(
                input_type,
                input_row_offsets_type,
                cache_positions_type,
                *flattened_kv_types,
            ),
        ) as graph:
            inputs, input_row_offsets, cache_positions, *kv_cache = graph.inputs
            kv_cache_inputs_per_dev = unflatten_kv_inputs(
                kv_manager, kv_params, [k.tensor for k in kv_cache]
            )
            kv_collections = [
                kv_collection_constructor(*kv_cache_inputs)
                for kv_cache_inputs in kv_cache_inputs_per_dev
            ]
            graph.output(
                attention(
                    [inputs.tensor],
                    [cache_positions.tensor],
                    kv_collections,
                    input_row_offsets=input_row_offsets.tensor,
                )[0]
            )

        compiled = session.load(graph, weights_registry=attention.state_dict())

        # Set up cache inputs and call the compiled model.
        seq_id = 0
        kv_manager.external_claim(seq_ids=[seq_id])
        batch = [create_text_context(seq_id, np.empty(input_seq_len))]
        blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
            kv_manager.fetch(batch)[0]
        )
        cache_positions_input = np.arange(input_seq_len, dtype=np.uint32)
        outputs.append(
            compiled.execute(
                Tensor.from_dlpack(input_tensor[0]).to(device),
                Tensor.from_numpy(
                    np.array([0, input_seq_len], dtype=np.uint32)
                ).to(device),
                Tensor.from_numpy(cache_positions_input).to(device),
                blocks.to(device),
                cache_lengths.to(device),
                lookup_table_tensor.to(device),
                is_cache_empty_buf,
            )[0]
        )
    return outputs


def attention_weights(
    config: Llama4Config, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    text_config = config.text_config
    hidden_size = text_config.hidden_size
    num_attention_heads = text_config.num_attention_heads
    num_key_value_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    std = 0.001

    def random_weight(*size):
        return torch.normal(0, std, size, dtype=dtype)

    return {
        "q_proj.weight": random_weight(
            num_attention_heads * head_dim, hidden_size
        ),
        "k_proj.weight": random_weight(
            num_key_value_heads * head_dim, hidden_size
        ),
        "v_proj.weight": random_weight(
            num_key_value_heads * head_dim, hidden_size
        ),
        "o_proj.weight": random_weight(
            hidden_size, num_attention_heads * head_dim
        ),
    }


def test_attention(
    text_config: Llama4TextConfig,
    input_tensor: torch.Tensor,
) -> None:
    # Update TextConfig to contain two decoder layers, one that uses attention
    # with rope, and one that uses attention without rope with attention tuning.
    text_config = Llama4TextConfig(**text_config.to_dict())
    text_config.attn_temperature_tuning = True
    text_config.num_hidden_layers = 2
    text_config.no_rope_layers = [1]

    config = Llama4Config()
    config.text_config = text_config

    torch_dtype = torch.bfloat16
    max_dtype = DType.bfloat16
    weights = attention_weights(config, torch_dtype)

    torch_outputs = generate_torch_outputs(
        text_config, input_tensor, weights, torch_dtype, "cuda"
    )

    max_outputs = generate_max_outputs(
        config, input_tensor, weights, max_dtype, Accelerator()
    )
    max_output_pt = [torch.from_dlpack(x).to(torch_dtype) for x in max_outputs]

    for torch_out, max_out in zip(torch_outputs, max_output_pt):
        torch.testing.assert_close(
            torch_out.squeeze(),
            max_out,
            rtol=1e-3,
            atol=2 * torch.finfo(torch.bfloat16).eps,
        )
