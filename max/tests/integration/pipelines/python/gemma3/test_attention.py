# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import copy
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
    load_kv_manager,
)
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines import KVCacheConfig
from max.pipelines.architectures.gemma3.layers.attention import (
    _Gemma3Attention as MaxGemma3Attention,
)
from max.pipelines.architectures.gemma3.model_config import (
    Gemma3Config as MaxGemma3Config,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3RotaryEmbedding,
)

MAX_SEQ_LEN = 1152


@pytest.fixture
def input_tensor(text_config: Gemma3TextConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return (
        torch.randn(1, 11, text_config.hidden_size)
        .to(torch.bfloat16)
        .to("cuda")
    )


def _get_position_embeddings(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    use_global_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""
    seq_len = input_tensor.shape[1]
    if use_global_rope:
        rotary_emb = Gemma3RotaryEmbedding(config=text_config, device="cuda")
    else:
        config = copy.deepcopy(text_config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        rotary_emb = Gemma3RotaryEmbedding(config=config, device="cuda")
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    cos, sin = rotary_emb(input_tensor, position_ids)
    return cos.to(torch.bfloat16).to("cuda"), sin.to(torch.bfloat16).to("cuda")


def _causal_attention_mask(seq_len: int) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        1, 1, seq_len, seq_len, dtype=torch.bfloat16, device="cuda"
    )
    attention_mask = attention_mask.masked_fill(
        causal_mask[None, None, :, :], torch.finfo(torch.bfloat16).min
    )
    return attention_mask


@torch.no_grad()
def generate_torch_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    """Generates the outputs of the MAX and PyTorch attention layers.

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local `RoPE`
    is used.
    """
    layer = (
        Gemma3Attention(
            text_config,
            layer_idx=layer_idx,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(torch.bfloat16).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    use_global_rope = layer_idx % 6 == 5
    position_embeddings = _get_position_embeddings(
        text_config, input_tensor, use_global_rope
    )

    return layer(input_tensor, position_embeddings, attention_mask)[0]


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
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
    layer_idx: int,
) -> torch.Tensor:
    """Runs the MAX Llama4 attention layer.

    Returns the outputs:
    1) Layer with rope
    2) Attention without rope but with attention tuning

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local `RoPE`
    is used.
    """
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    input_seq_len = input_tensor.shape[1]

    state_dict = {}
    for weight_name, value in attention_weights.items():
        state_dict[weight_name] = value.cpu()

    kv_cache_config = KVCacheConfig(
        cache_strategy=KVCacheStrategy.PAGED, kv_cache_page_size=256
    )
    kv_params = MaxGemma3Config.get_kv_params(
        text_config,
        1,
        kv_cache_config,
        dtype,
    )
    kv_collection_constructor = FetchPagedKVCacheCollection(
        kv_params, num_layers=text_config.num_hidden_layers
    )

    session = InferenceSession(devices=[Accelerator(0)])

    attention = MaxGemma3Attention(
        rope_global=Llama3RotaryEmbedding(
            text_config.hidden_size,
            text_config.num_attention_heads,
            text_config.rope_theta,
            MAX_SEQ_LEN,
            interleaved=False,
            head_dim=text_config.head_dim,
            device=device_ref,
        ),
        rope_local=Llama3RotaryEmbedding(
            text_config.hidden_size,
            text_config.num_attention_heads,
            text_config.rope_local_base_freq,
            MAX_SEQ_LEN,
            interleaved=False,
            head_dim=text_config.head_dim,
            device=device_ref,
        ),
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        hidden_size=text_config.hidden_size,
        kv_params=kv_params,
        dtype=dtype,
        devices=[device_ref],
        layer_idx=layer_idx,
        sliding_window_pattern=text_config.sliding_window_pattern,
    )
    attention.load_state_dict(state_dict)

    # Set up blank KV cache.
    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=MaxGemma3Config.get_num_layers(
            huggingface_config=text_config
        ),
        devices=[device],
        available_cache_memory=30 * 1024 * 1024,
        page_size=kv_cache_config.kv_cache_page_size,
        session=session,
    )
    assert isinstance(kv_manager, PagedKVCacheManager)

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
        "Gemma3Attention",
        input_types=(
            input_type,
            input_row_offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        inputs, input_row_offsets, *kv_cache = graph.inputs
        kv_inputs_flat = [k.tensor for k in kv_cache]
        kv_collection = kv_collection_constructor(*kv_inputs_flat)

        graph.output(
            attention(
                inputs.tensor,
                kv_collection,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    compiled = session.load(graph, weights_registry=attention.state_dict())

    # Set up cache inputs and call the compiled model.
    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.external_claim(batch[0].request_id)
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(batch)[0]
    )

    output = compiled.execute(
        Tensor.from_dlpack(input_tensor[0]).to(device),
        Tensor.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
            device
        ),
        blocks.to(device),
        cache_lengths.to(device),
        lookup_table_tensor.to(device),
        is_cache_empty_buf,
    )[0]

    return output


def test_attention_local_rope(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights, layer_idx=0
    )

    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        attention_weights,
        DType.bfloat16,
        Accelerator(),
        layer_idx=0,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        from_dlpack(max_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def test_attention_global_rope(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        attention_weights,
        layer_idx=5,
    )

    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        attention_weights,
        DType.bfloat16,
        Accelerator(),
        layer_idx=5,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        from_dlpack(max_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
