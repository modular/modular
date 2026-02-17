# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import functools
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.interfaces import TextGenerationContext
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy import AttentionWithRope, Linear, RotaryEmbedding
from max.nn.legacy.float8_config import (
    Float8Config,
)
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.pipelines.architectures.llama3_legacy.model_config import (
    create_rope_embedding,
)
from max.pipelines.lib.float8 import parse_float8_config
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_h100_h200
from torch.utils.dlpack import from_dlpack
from transformers.models.llama.configuration_llama import LlamaConfig

RTOL = 0.006
ATOL = 0.006


@pytest.fixture
def config() -> LlamaConfig:
    config = LlamaConfig()
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config_nvfp4.json"
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


@pytest.fixture
def input_tensor(
    config: LlamaConfig,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def attention_mask(
    seq_len: int = 7,
    batch_size: int = 1,
) -> torch.Tensor:
    # Create causal mask where future tokens can't attend to past tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    causal_mask = torch.zeros(
        1, batch_size, seq_len, seq_len, dtype=torch.bfloat16
    )
    causal_mask.masked_fill_(mask, float("-inf")).to(torch.bfloat16)
    return causal_mask


@pytest.fixture
def attention_weights_and_scales(
    config: LlamaConfig,
    block_size: int = 16,
) -> dict[str, torch.Tensor]:
    hidden_size = config.hidden_size
    state_dict = {}

    for key in "koqv":
        outdim = config.num_attention_heads * config.head_dim
        if key in "kv":
            outdim = config.head_dim * config.num_key_value_heads
            # state_dict.update({f"{key}_proj.{key}_scale": torch.randn(1, dtype=torch.bfloat16).abs()})

        state_dict.update(
            {
                f"{key}_proj.input_scale": torch.tensor(
                    [0.0015], dtype=torch.float32
                ),
                f"{key}_proj.weight": torch.randint(
                    255, (outdim, hidden_size // 2), dtype=torch.uint8
                ),
                f"{key}_proj.weight_scale": torch.randn(
                    outdim, hidden_size // block_size
                )
                .abs()
                .to(torch.float8_e4m3fn),
                f"{key}_proj.weight_scale_2": torch.tensor(
                    [0.0002], dtype=torch.float32
                ),
            }
        )

    return state_dict


def get_state_dict(
    attention_weights_and_scales: dict[str, torch.Tensor],
) -> dict[str, WeightData]:
    # Convert torch tensors to WeightData objects
    normalized_state_dict: dict[str, WeightData] = {}
    for name, tensor in attention_weights_and_scales.items():
        if tensor.dtype == torch.float8_e4m3fn:
            normalized_state_dict[name] = WeightData(
                Buffer.from_dlpack(tensor.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),  # avoids BufferError: float8 types are not supported by dlpack, when loading state_dict
                name,
                DType.float8_e4m3fn,
                Shape(tensor.shape),
            )
        else:
            normalized_state_dict[name] = WeightData(
                data=tensor,
                name=name,
                dtype=DType.from_torch(tensor.dtype),
                shape=Shape(tensor.shape),
            )
    return normalized_state_dict


def model(
    config: LlamaConfig,
    float8_config: Float8Config,
    kv_params: KVCacheParams,
    state_dict: dict[str, WeightData],
    dtype: DType,
) -> tuple[AttentionWithRope, RotaryEmbedding]:
    rope = create_rope_embedding(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        rope_theta=config.rope_theta,
        max_seq_len=7,  # config.max_seq_len,
        interleaved_rope_weights=False,  # config.interleaved_rope_weights,
        rope_scaling_params=None,  # config.rope_scaling_params, TODO
        longrope_scaling_params=None,  # config.longrope_scaling_params, TODO
        device=DeviceRef.GPU(),  # config.devices[0],
    )
    attention_multiplier = rope.compute_scale()

    linear_cls = functools.partial(Linear, float8_config=float8_config)

    layer = AttentionWithRope(
        stacked_qkv="layers.0.self_attn.qkv_proj.weight" in state_dict,
        scale=attention_multiplier,
        clip_qkv=getattr(config, "clip_qkv", None),
        has_bias=False,
        float8_config=float8_config,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_size=config.hidden_size,
        kv_params=kv_params,
        dtype=dtype,
        rope=rope,
        linear_cls=linear_cls,
        devices=[DeviceRef.GPU()],  # config.devices,
    )
    # raise Exception(layer.q_proj.weight)
    layer.load_state_dict(state_dict)

    return layer, rope


def generate_max_outputs_fp4(
    state_dict: dict[str, WeightData],
    config: LlamaConfig,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    device = Accelerator()
    # FP4 weights are stored as uint8; pass that so the layer expects uint8.
    weight_dtype = DType.uint8
    cache_dtype = DType.bfloat16

    # Parse float8 config for fp4
    float8_config = parse_float8_config(
        config,
        state_dict,
        weight_dtype,  # uint8 for fp4-e2m1fnX2
    )

    if float8_config is None:
        raise ValueError("Failed to parse float8 config for FP4")

    kv_params = KVCacheParams(
        dtype=cache_dtype,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.GPU()],
    )

    layer, rope = model(
        config, float8_config, kv_params, state_dict, weight_dtype
    )

    device_ref = DeviceRef.GPU()
    input_seq_len = input_tensor.shape[1]

    # Construct input types.
    input_type = TensorType(
        dtype=DType.bfloat16,
        shape=["total_seq_len", config.hidden_size],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    kv_cache_args = kv_params.get_symbolic_inputs()
    flattened_kv_types = [
        kv_type for sublist in kv_cache_args for kv_type in sublist
    ]

    session = InferenceSession(devices=[Accelerator()])

    # Set up blank KV cache.
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    # Build graph with context manager
    with Graph(
        "AttentionWithRope",
        input_types=(
            input_type,
            input_row_offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        inputs, input_row_offsets, *kv_cache = graph.inputs
        # Unflatten KV cache inputs
        kv_collection = PagedCacheValues(
            kv_blocks=kv_cache[0].buffer,
            cache_lengths=kv_cache[1].tensor,
            lookup_table=kv_cache[2].tensor,
            max_lengths=kv_cache[3].tensor,
        )

        # Create layer_idx constant
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

        # Call attention layer
        graph.output(
            layer(
                layer_idx,
                inputs.tensor,
                kv_collection,
                freqs_cis=rope.freqs_cis,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    compiled = session.load(graph, weights_registry=layer.state_dict())

    # Set up KV cache for execution (single replica: replica_idx=0)
    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0].request_id, replica_idx=0)
    kv_manager.alloc(batch[0], replica_idx=0)
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.get_runtime_inputs(
            cast(list[list[TextGenerationContext]], [batch])
        )[0]
    )

    # Prepare inputs - flatten batch and sequence dimensions
    input_tensor_flat = input_tensor[0].reshape(-1, config.hidden_size)
    input_row_offsets_input = np.array([0, input_seq_len], dtype=np.uint32)

    out = compiled.execute(
        Buffer.from_dlpack(input_tensor_flat).to(device),
        Buffer.from_numpy(input_row_offsets_input).to(device),
        blocks.to(device),
        cache_lengths.to(device),
        lookup_table_tensor.to(device),
        is_cache_empty_buf,
    )[0]
    return from_dlpack(out).to(torch.bfloat16)


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="FP4 kernel only supports Nvidia GPUs"
)
@pytest.mark.skipif(
    is_h100_h200(),
    reason="FP4 kernel requires SM100 (B200), not supported on H100/H200 (SM90)",
)
def test_llama_attention_fp4(
    config: LlamaConfig,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights_and_scales: dict[str, torch.Tensor],
) -> None:
    state_dict = get_state_dict(attention_weights_and_scales)

    max_output = generate_max_outputs_fp4(state_dict, config, input_tensor)

    # Check that outputs are not all zeros or NaN
    assert not torch.all(max_output == 0.0), (
        "MAX output should not be all zeros"
    )
    assert not torch.any(torch.isnan(max_output)), (
        "MAX output should not contain NaN"
    )
    assert not torch.any(torch.isinf(max_output)), (
        "MAX output should not contain Inf"
    )
