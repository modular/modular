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
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
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
from max.pipelines.architectures.olmo2.layers.attention import (
    Olmo2Attention as MaxOlmo2Attention,
)
from max.pipelines.architectures.olmo2.model_config import (
    Olmo2Config as MaxOlmo2Config,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding
from transformers.models.olmo2.modular_olmo2 import Olmo2Attention, Olmo2Config

# Max position embeddings for OLMo2-7B
# Based on OLMo2 configuration
MAX_SEQ_LEN = 4096


@pytest.fixture
def input_tensor(text_config: Olmo2Config) -> torch.Tensor:
    torch.manual_seed(39)
    # Use the hidden size from the config
    return torch.randn(1, 11, 4096).to(torch.float32).to("cuda")


def _get_position_embeddings(
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""

    seq_len = input_tensor.shape[1]
    rotary_emb = Olmo2RotaryEmbedding(config=text_config, device="cuda")
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    cos, sin = rotary_emb(input_tensor, position_ids)
    return cos.to(torch.float32).to("cuda"), sin.to(torch.float32).to("cuda")


def _causal_attention_mask(seq_len: int) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        1, 1, seq_len, seq_len, dtype=torch.float32, device="cuda"
    )
    attention_mask = attention_mask.masked_fill(
        causal_mask[None, None, :, :], torch.finfo(torch.float32).min
    )
    return attention_mask


@torch.no_grad()
def generate_torch_outputs(
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = (
        Olmo2Attention(
            text_config,
            layer_idx=0,
        )
        .to(torch.float32)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(torch.float32).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    position_embeddings = _get_position_embeddings(text_config, input_tensor)

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
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Runs the MAX Olmo2 attention layer.
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
        state_dict[weight_name] = value.to(dtype.to_torch()).cpu()

    kv_cache_config = KVCacheConfig(cache_strategy=KVCacheStrategy.PAGED)
    kv_params = MaxOlmo2Config.get_kv_params(
        text_config,
        n_devices=1,
        kv_cache_config=kv_cache_config,
        cache_dtype=dtype,
    )
    kv_collection_constructor = FetchPagedKVCacheCollection(
        kv_params, num_layers=text_config.num_hidden_layers
    )

    session = InferenceSession(devices=[Accelerator(0)])

    assert text_config.head_dim == 128
    print("inja inja text_config.head_dim")
    print(text_config.head_dim)

    attention = MaxOlmo2Attention(
        rope=Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.rope_theta,
            max_seq_len=MAX_SEQ_LEN,
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
        layer_idx=0,
    )
    attention.load_state_dict(state_dict)

    # Set up blank KV cache.
    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=MaxOlmo2Config.get_num_layers(
            huggingface_config=text_config
        ),
        devices=[device],
        available_cache_memory=40 * 4096 * 4096,
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
        "Olmo2Attention",
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
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                inputs.tensor,
                kv_collection,
                freqs_cis=attention.rope.freqs_cis,
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


def test_attention(
    text_config: Olmo2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights
    )

    max_output = generate_max_outputs(
        text_config=text_config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        dtype=DType.float32,
        device=Accelerator(),
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.float32),
        from_dlpack(max_output).to(torch.float32),
        rtol=0.01,
        atol=0.01,
    )
