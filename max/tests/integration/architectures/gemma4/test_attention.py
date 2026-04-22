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


import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    Gemma4RotaryEmbedding,
    Gemma4TextAttention,
)
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.kv_cache import PagedKVCacheManager
from max.nn.kernels import KVCacheParams
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma4.layers.attention import (
    Gemma4Attention as MaxGemma4Attention,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
    ProportionalScalingParams,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

MAX_SEQ_LEN = 1152

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


def _generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


@pytest.fixture
def text_config() -> Gemma3TextConfig:
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    # Use "text_config" for the multimodal variants
    if "text_config" in data:
        return Gemma3TextConfig(
            **data["text_config"], attn_implementation="eager"
        )
    else:
        return Gemma3TextConfig(**data, attn_implementation="eager")


@pytest.fixture
def input_tensor(text_config: Gemma3TextConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return _generate_tensor((1, 11, text_config.hidden_size)).to("cuda")


@pytest.fixture
def attention_weights_local(
    text_config: Gemma3TextConfig,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    # calculated from google/gemma-3-1b-it checkpoint
    O_PROJ_STD = 0.0237
    K_PROJ_STD = 0.0309
    Q_PROJ_STD = 0.0284
    V_PROJ_STD = 0.0309
    K_NORM_STD = 0.793
    Q_NORM_STD = 0.68

    q_dim = text_config.head_dim * text_config.num_attention_heads
    kv_dim = text_config.head_dim * text_config.num_key_value_heads
    hidden_size = text_config.hidden_size

    return {
        "k_norm.weight": _generate_tensor((text_config.head_dim,)) * K_NORM_STD,
        "k_proj.weight": _generate_tensor((kv_dim, hidden_size)) * K_PROJ_STD,
        "o_proj.weight": _generate_tensor((hidden_size, q_dim)) * O_PROJ_STD,
        "q_norm.weight": _generate_tensor((text_config.head_dim,)) * Q_NORM_STD,
        "q_proj.weight": _generate_tensor((q_dim, hidden_size)) * Q_PROJ_STD,
        "v_proj.weight": _generate_tensor((kv_dim, hidden_size)) * V_PROJ_STD,
    }


@pytest.fixture
def attention_weights_global(
    text_config: Gemma3TextConfig,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    # calculated from google/gemma-3-1b-it checkpoint
    O_PROJ_STD = 0.0237
    K_PROJ_STD = 0.0309
    Q_PROJ_STD = 0.0284
    K_NORM_STD = 0.793
    Q_NORM_STD = 0.68

    q_dim = text_config.global_head_dim * text_config.num_attention_heads
    kv_dim = (
        text_config.global_head_dim * text_config.num_global_key_value_heads
    )
    hidden_size = text_config.hidden_size

    return {
        "k_norm.weight": _generate_tensor((text_config.global_head_dim,))
        * K_NORM_STD,
        "k_proj.weight": _generate_tensor((kv_dim, hidden_size)) * K_PROJ_STD,
        "o_proj.weight": _generate_tensor((hidden_size, q_dim)) * O_PROJ_STD,
        "q_norm.weight": _generate_tensor((text_config.global_head_dim,))
        * Q_NORM_STD,
        "q_proj.weight": _generate_tensor((q_dim, hidden_size)) * Q_PROJ_STD,
    }


def _get_position_embeddings(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    use_global_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""
    seq_len = input_tensor.shape[1]
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)

    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "sliding_attention" in rope_params:
        # v5: single embedding handles both layer types natively
        rotary_emb = Gemma4RotaryEmbedding(config=text_config, device="cuda")
        layer_type = (
            "full_attention" if use_global_rope else "sliding_attention"
        )
        cos, sin = rotary_emb(input_tensor, position_ids, layer_type=layer_type)
    else:
        # v4: need separate embedding with hacked config for local rope
        if use_global_rope:
            rotary_emb = Gemma4RotaryEmbedding(
                config=text_config, device="cuda"
            )
        else:
            config = copy.deepcopy(text_config)
            config.rope_theta = config.rope_local_base_freq
            config.rope_scaling = {"rope_type": "default"}
            rotary_emb = Gemma4RotaryEmbedding(config=config, device="cuda")
        cos, sin = rotary_emb(input_tensor, position_ids)

    return cos.to(TORCH_DTYPE).to("cuda"), sin.to(TORCH_DTYPE).to("cuda")


def _causal_attention_mask(seq_len: int) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        1, 1, seq_len, seq_len, dtype=TORCH_DTYPE, device="cuda"
    )
    attention_mask = attention_mask.masked_fill(
        causal_mask[None, None, :, :], torch.finfo(TORCH_DTYPE).min
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
        Gemma4TextAttention(
            text_config,
            layer_idx=layer_idx,
        )
        .to(TORCH_DTYPE)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(TORCH_DTYPE).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    use_global_rope = layer_idx % 6 == 5
    position_embeddings = _get_position_embeddings(
        text_config, input_tensor, use_global_rope
    )

    return layer(input_tensor, position_embeddings, attention_mask)[0]


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

    # No remapping required for either sliding/local (QKV) or
    # global/full (QK) layer types.
    state_dict = {
        weight_name: value.cpu()
        for weight_name, value in attention_weights.items()
    }

    kv_params_local = KVCacheParams(
        dtype=dtype,
        devices=[device_ref],
        n_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        num_layers=len(
            [lt for lt in text_config.layer_types if lt == "sliding_attention"]
        ),
        page_size=256,
    )

    kv_params_global = KVCacheParams(
        dtype=dtype,
        devices=[device_ref],
        n_kv_heads=text_config.num_global_key_value_heads,
        head_dim=text_config.global_head_dim,
        num_layers=len(
            [lt for lt in text_config.layer_types if lt == "full_attention"]
        ),
        page_size=256,
    )

    kv_params = (
        kv_params_local
        if text_config.layer_types[layer_idx] == "sliding_attention"
        else kv_params_global
    )

    session = InferenceSession(devices=[Accelerator(0)])

    attention = MaxGemma4Attention(
        rope_global=ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=1000000.0,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=ProportionalScalingParams(0.25),
        ),
        rope_local=Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=10000.0,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,
        ),
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        num_global_key_value_heads=text_config.num_global_key_value_heads,
        attention_k_eq_v=text_config.attention_k_eq_v,
        hidden_size=text_config.hidden_size,
        kv_params=kv_params,
        global_head_dim=text_config.global_head_dim,
        layer_idx=layer_idx,
        layer_idx_in_cache=0,
        is_sliding=text_config.layer_types[layer_idx] == "sliding_attention",
        dtype=dtype,
        devices=[device_ref],
        qk_norm_eps=text_config.rms_norm_eps,
        local_window_size=text_config.sliding_window,
    )
    attention.load_state_dict(state_dict)

    # Set up blank KV cache.
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
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
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

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
        kv_collection = (
            kv_params.get_symbolic_inputs().unflatten(iter(kv_cache)).inputs[0]
        )

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
    kv_manager.claim(batch[0].request_id, replica_idx=0)
    kv_manager.alloc(batch[0], replica_idx=0, num_steps=1)
    kv_runtime_inputs = kv_manager.runtime_inputs([batch]).inputs[0]
    assert kv_runtime_inputs.attention_dispatch_metadata is not None

    output = compiled.execute(
        Buffer.from_dlpack(input_tensor[0]).to(device),
        Buffer.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
            device
        ),
        kv_runtime_inputs.kv_blocks.to(device),
        kv_runtime_inputs.cache_lengths.to(device),
        kv_runtime_inputs.lookup_table.to(device),
        kv_runtime_inputs.max_lengths,
        kv_runtime_inputs.attention_dispatch_metadata,
    )[0]

    return output


def test_attention_local(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_local: dict[str, torch.Tensor],
) -> None:
    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        attention_weights_local,
        MAX_DTYPE,
        Accelerator(),
        layer_idx=0,
    )

    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights_local, layer_idx=0
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )


def test_attention_global(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_global: dict[str, torch.Tensor],
) -> None:
    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        attention_weights_global,
        MAX_DTYPE,
        Accelerator(),
        layer_idx=5,
    )
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        attention_weights_global,
        layer_idx=5,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )
