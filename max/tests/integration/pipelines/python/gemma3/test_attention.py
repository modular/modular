# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections.abc import Sequence

import numpy as np
import pytest
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
)

MAX_SEQ_LEN = 1152


@pytest.fixture
def position_embeddings(
    text_config: Gemma3TextConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 11, 256).to(torch.bfloat16).to("cuda") * 0.7,
        torch.randn(1, 11, 256).to(torch.bfloat16).to("cuda") * 0.5,
    )


@pytest.fixture
def input_tensor(text_config: Gemma3TextConfig) -> torch.Tensor:
    return torch.randn(1, 11, 1152).to(torch.bfloat16).to("cuda")


def generate_torch_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    layer = (
        Gemma3Attention(
            text_config,
            layer_idx=0,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(torch.bfloat16).to("cuda")

    attention_mask = torch.zeros(1, 1, 1, 60).to(torch.bfloat16).to("cuda")

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
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
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
    for weight_name, value in attention_weights.items():
        if "norm" in weight_name:
            state_dict[weight_name] = value.to(torch.float32).cpu()
        else:
            state_dict[weight_name] = value.cpu()

    kv_cache_config = KVCacheConfig(cache_strategy=KVCacheStrategy.PAGED)
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
        rope=OptimizedRotaryEmbedding(
            text_config.hidden_size,
            text_config.num_attention_heads,
            text_config.rope_theta,
            MAX_SEQ_LEN,
            interleaved=True,
            head_dim=text_config.head_dim,
            device=device_ref,
        ),
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        hidden_size=text_config.hidden_size,
        kv_params=kv_params,
        dtype=dtype,
        devices=[device_ref],
        use_qk_norm=True,
        layer_idx=0,
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
        available_cache_memory=10 * 1024 * 1024,
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
        "Gemma3Attention",
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
    output = compiled.execute(
        Tensor.from_dlpack(input_tensor[0]).to(device),
        Tensor.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
            device
        ),
        Tensor.from_numpy(cache_positions_input).to(device),
        blocks.to(device),
        cache_lengths.to(device),
        lookup_table_tensor.to(device),
        is_cache_empty_buf,
    )[0]
    return output


@pytest.mark.skip(reason="Accuracy debugging in progress.")
def test_attention(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights, position_embeddings
    )

    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        attention_weights,
        position_embeddings,
        DType.bfloat16,
        Accelerator(),
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        from_dlpack(max_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
