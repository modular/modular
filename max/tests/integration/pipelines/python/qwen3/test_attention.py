# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections.abc import Sequence

import max.driver as md
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
from max.pipelines.architectures.qwen3.layers.attention import (
    Qwen3Attention as MaxQwen3Attention,
)
from max.pipelines.architectures.qwen3.model_config import (
    Qwen3Config as MaxQwen3Config,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
)

MAX_SEQ_LEN = 1024


@pytest.fixture
def input_tensor(text_config: Qwen3Config) -> torch.Tensor:
    torch.manual_seed(42)
    # https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
    # 2048 per Qwen3-1.7B Hidden Size in config.json (5120 if you want to test the 32B attention)
    return torch.randn(1, 11, 2048).to(torch.bfloat16).to("cuda")


def _get_position_embeddings(
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""
    seq_len = input_tensor.shape[1]
    rotary_emb = Qwen3RotaryEmbedding(config=text_config, device="cuda")
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
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = (
        Qwen3Attention(
            text_config,
            layer_idx=0,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(torch.bfloat16).to("cuda")

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
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Runs the MAX Qwen3 attention layer.

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
        state_dict[weight_name] = value.cpu()

    kv_cache_config = KVCacheConfig(cache_strategy=KVCacheStrategy.PAGED)
    kv_params = MaxQwen3Config.get_kv_params(
        text_config,
        1,
        kv_cache_config,
        dtype,
    )
    kv_collection_constructor = FetchPagedKVCacheCollection(
        kv_params, num_layers=text_config.num_hidden_layers
    )

    session = InferenceSession(devices=[Accelerator(0)])

    rope = Llama3RotaryEmbedding(
        text_config.hidden_size,
        text_config.num_attention_heads,
        text_config.rope_theta,
        MAX_SEQ_LEN,
        interleaved=False,
        head_dim=text_config.head_dim,
        device=device_ref,
    )
    attention = MaxQwen3Attention(
        rope=rope,
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
        num_layers=MaxQwen3Config.get_num_layers(
            huggingface_config=text_config
        ),
        devices=[device],
        available_cache_memory=30
        * 1024
        * 1024,  # Use 32 instead of 30 for 32B model attention test
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
        "Qwen3Attention",
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
                input_row_offsets.tensor,
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
    text_config: Qwen3Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    # TODO: Remove this once we figure out the attention error on AMD GPUs.
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")

    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights
    )

    max_output = generate_max_outputs(
        text_config=text_config,
        input_tensor=input_tensor,
        attention_weights=attention_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        from_dlpack(max_output).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
