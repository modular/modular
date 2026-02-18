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
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Value
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy import RotaryEmbedding
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.pipelines.architectures.llama4.layers.attention import (
    _Llama4TextAttention,
)
from test_common.context_utils import create_text_context
from transformers.masking_utils import (
    create_causal_mask,
    create_chunked_causal_mask,
)
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import Llama4TextModel

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
    model.to(device)
    assert model.layers[0].self_attn.use_rope
    # Hack - Transformers `LLama4TextAttention` doesn't actually use the config
    # to configure whether the layer uses rope. Set this manually.
    model.layers[1].self_attn.use_rope = False
    del model.layers[1].self_attn.qk_norm  # Also unset qk_norm.

    # Create layer inputs.
    input_tensor = input_tensor.to(device)
    attention_mask = torch.ones(
        [1, input_tensor.shape[1]], dtype=torch.int64
    ).to(device)
    cache_position = torch.arange(0, input_tensor.shape[1], device=device)

    rotary_emb = model.rotary_emb
    position_ids = cache_position.unsqueeze(0)
    freqs_cis = rotary_emb(input_tensor, position_ids)

    mask_kwargs = {
        "config": text_config,
        "input_embeds": input_tensor,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask = create_causal_mask(**mask_kwargs)
    chunk_causal_mask = create_chunked_causal_mask(**mask_kwargs)

    outputs = []
    for layer_idx in (0, 1):
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
                position_embeddings=freqs_cis,
            )[0]
        )
    return outputs


def unflatten_kv_inputs(
    kv_params: KVCacheParams,
    kv_inputs_flat: Sequence[Value[Any]],
) -> list[PagedCacheValues]:
    n_devices = kv_params.n_devices
    kv_caches_per_dev: list[PagedCacheValues] = []
    for i in range(n_devices):
        start_idx = i * n_devices
        kv_caches_per_dev.append(
            PagedCacheValues(
                kv_blocks=kv_inputs_flat[start_idx].buffer,
                cache_lengths=kv_inputs_flat[start_idx + 1].tensor,
                lookup_table=kv_inputs_flat[start_idx + 2].tensor,
                max_lengths=kv_inputs_flat[start_idx + 3].tensor,
            )
        )
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
        weight_name = weight_name.removeprefix("self_attn.")
        state_dict[weight_name] = value.cpu()

    text_config = config.text_config
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        num_layers=text_config.num_hidden_layers,
        devices=[DeviceRef.from_device(device)],
    )

    session = InferenceSession(devices=[Accelerator(0)])
    outputs = []
    for layer_idx, use_rope in enumerate([True, False]):
        attention = _Llama4TextAttention(
            rope=RotaryEmbedding(
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
        cache_positions_type = TensorType(
            DType.uint32,
            ["total_seq_len"],
            device=device_ref,
        )
        kv_cache_args = kv_params.get_symbolic_inputs()
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
            kv_collections = unflatten_kv_inputs(kv_params, kv_cache)
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
        batch = [create_text_context(np.empty(input_seq_len))]
        kv_manager.claim(batch[0].request_id, replica_idx=0)
        kv_manager.alloc(batch[0], replica_idx=0, num_steps=1)
        blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
            kv_manager.get_runtime_inputs([batch])[0]
        )
        cache_positions_input = np.arange(input_seq_len, dtype=np.uint32)
        outputs.append(
            compiled.execute(
                Buffer.from_dlpack(input_tensor[0]).to(device),
                Buffer.from_numpy(
                    np.array([0, input_seq_len], dtype=np.uint32)
                ).to(device),
                Buffer.from_numpy(cache_positions_input).to(device),
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

    std = 0.008

    def random_weight(*size):  # noqa: ANN202
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
) -> None:
    # Update TextConfig to contain two decoder layers, one that uses attention
    # with rope, and one that uses attention without rope with attention tuning.
    text_config = Llama4TextConfig(**text_config.to_dict())
    text_config.attn_temperature_tuning = True
    text_config.num_hidden_layers = 2
    text_config.no_rope_layer_interval = 2

    # Decrease actual head sizes to avoid OOMing
    text_config.hidden_size = 512
    text_config.num_attention_heads = 4
    text_config.num_key_value_heads = 4
    text_config.head_dim = (
        text_config.hidden_size // text_config.num_attention_heads
    )

    config = Llama4Config()
    config.text_config = text_config

    input_tensor = torch.randn(
        1,  # batch size
        6,  # seq len
        text_config.hidden_size,  # hidden size
        dtype=torch.bfloat16,
    )

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

    for torch_out, max_out in zip(torch_outputs, max_output_pt, strict=True):
        torch.testing.assert_close(
            torch_out.squeeze(),
            max_out,
            rtol=1e-3,
            atol=2 * torch.finfo(torch.bfloat16).eps,
        )
