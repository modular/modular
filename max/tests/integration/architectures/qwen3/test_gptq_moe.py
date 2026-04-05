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

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightData
from max.kv_cache import PagedKVCacheManager
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheParams, unflatten_ragged_attention_inputs
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.architectures.qwen3.qwen3 import Qwen3
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_nvidia_gpu
from torch.utils.dlpack import from_dlpack

DESC_ACT = False
GROUP_SIZE = 128
HIDDEN_SIZE = 2048
HEAD_DIM = 128
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 4
Q_PROJ_DIM = HEAD_DIM * NUM_ATTENTION_HEADS
KV_PROJ_DIM = HEAD_DIM * NUM_KEY_VALUE_HEADS
MOE_INTERMEDIATE_SIZE = 768


def _weight_data_from_numpy(name: str, value: np.ndarray) -> WeightData:
    return WeightData(
        Buffer.from_numpy(value),
        name,
        DType.from_numpy(value.dtype),
        Shape(value.shape),
    )


def _gptq_scales_shape(in_dim: int, out_dim: int) -> tuple[int, int]:
    return (in_dim // GROUP_SIZE, out_dim)


def _make_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        num_hidden_layers=1,
        rope_theta=1000000.0,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=6144,
        interleaved_rope_weights=False,
        vocab_size=256,
        dtype=DType.bfloat16,
        model_quantization_encoding=QuantizationEncoding.GPTQ,
        quantization_config=QuantizationConfig(
            quant_method="gptq",
            bits=4,
            group_size=GROUP_SIZE,
            desc_act=DESC_ACT,
            sym=True,
        ),
        kv_params=_kv_params(),
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        rms_norm_eps=1e-6,
        clip_qkv=None,
        norm_method="rms_norm",
        norm_dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        num_experts=8,
        num_experts_per_tok=8,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        mlp_only_layers=[],
        norm_topk_prob=False,
        decoder_sparse_step=1,
        use_subgraphs=True,
    )


def _kv_params() -> KVCacheParams:
    return KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.GPU()],
        data_parallel_degree=1,
    )


def _make_state_dict() -> dict[str, object]:
    rng = np.random.default_rng(7)
    vocab_size = 256

    state_dict: dict[str, object] = {
        "embed_tokens.weight": torch.randn(
            vocab_size, HIDDEN_SIZE, dtype=torch.bfloat16
        )
        * 0.02,
        "lm_head.weight": torch.randn(
            vocab_size, HIDDEN_SIZE, dtype=torch.bfloat16
        )
        * 0.02,
        "norm.weight": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16),
        "layers.0.input_layernorm.weight": torch.ones(
            HIDDEN_SIZE, dtype=torch.bfloat16
        ),
        "layers.0.post_attention_layernorm.weight": torch.ones(
            HIDDEN_SIZE, dtype=torch.bfloat16
        ),
        "layers.0.self_attn.q_norm.weight": torch.ones(
            HEAD_DIM, dtype=torch.bfloat16
        ),
        "layers.0.self_attn.k_norm.weight": torch.ones(
            HEAD_DIM, dtype=torch.bfloat16
        ),
        "layers.0.mlp.gate.gate_score.weight": torch.randn(
            8, HIDDEN_SIZE, dtype=torch.bfloat16
        )
        * 0.02,
    }

    for proj_name, out_dim in (
        ("q_proj", Q_PROJ_DIM),
        ("k_proj", KV_PROJ_DIM),
        ("v_proj", KV_PROJ_DIM),
        ("o_proj", HIDDEN_SIZE),
    ):
        prefix = f"layers.0.self_attn.{proj_name}"
        state_dict[f"{prefix}.qweight"] = _weight_data_from_numpy(
            f"{prefix}.qweight",
            rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=(HIDDEN_SIZE // 8, out_dim),
                dtype=np.uint32,
            ),
        )
        state_dict[f"{prefix}.scales"] = _weight_data_from_numpy(
            f"{prefix}.scales",
            rng.standard_normal(
                size=_gptq_scales_shape(HIDDEN_SIZE, out_dim)
            ).astype(np.float16),
        )
        if DESC_ACT:
            state_dict[f"{prefix}.perm_idx"] = np.arange(
                HIDDEN_SIZE, dtype=np.int32
            )[::-1].copy()

    for expert_idx in range(8):
        for proj_name, in_dim, out_dim in (
            ("gate_proj", HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
            ("up_proj", HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
            ("down_proj", MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ):
            prefix = f"layers.0.mlp.experts.{expert_idx}.{proj_name}"
            state_dict[f"{prefix}.qweight"] = _weight_data_from_numpy(
                f"{prefix}.qweight",
                rng.integers(
                    0,
                    np.iinfo(np.uint32).max,
                    size=(in_dim // 8, out_dim),
                    dtype=np.uint32,
                ),
            )
            state_dict[f"{prefix}.scales"] = _weight_data_from_numpy(
                f"{prefix}.scales",
                rng.standard_normal(
                    size=_gptq_scales_shape(in_dim, out_dim)
                ).astype(np.float16),
            )
            if DESC_ACT:
                state_dict[f"{prefix}.perm_idx"] = np.arange(
                    in_dim, dtype=np.int32
                )[::-1].copy()

    return state_dict


@pytest.mark.skipif(
    not is_nvidia_gpu(), reason="Qwen3 GPTQ MoE requires NVIDIA GPU"
)
def test_qwen3_gptq_moe_smoke() -> None:
    config = _make_config()
    model = Qwen3(config)
    model.load_state_dict(
        cast(Mapping[str, DLPackArray | WeightData], _make_state_dict()),
        strict=True,
    )
    first_layer = cast(Any, model.layers[0])
    assert hasattr(first_layer.self_attn.q_proj, "weight")
    assert hasattr(first_layer.mlp.experts[0].gate_proj, "packed_weight_tensor")
    assert not hasattr(first_layer.mlp.experts[0].gate_proj, "weight")
    session = InferenceSession(devices=[Accelerator()])
    with Graph(
        "qwen3_gptq_moe_smoke", input_types=model.input_types(config.kv_params)
    ) as graph:
        graph_tokens, input_row_offsets, return_n_logits, *variadic_args = (
            graph.inputs
        )
        signal_buffers = [variadic_args[0].buffer]
        kv_collection = unflatten_ragged_attention_inputs(
            variadic_args[1:], n_devices=1
        )[0]
        graph.output(
            *model(
                graph_tokens.tensor,
                [kv_collection],
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
            )
        )

    compiled = session.load(graph, weights_registry=model.state_dict())

    kv_manager = PagedKVCacheManager(
        params=config.kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=8,
    )
    token_ids = np.array([1, 2, 3], dtype=np.int64)
    batch = [create_text_context(token_ids)]
    kv_manager.claim(batch[0].request_id, replica_idx=0)
    kv_manager.alloc(batch[0], replica_idx=0, num_steps=1)
    kv_runtime_inputs = kv_manager.runtime_inputs([batch]).inputs[0]

    result = compiled.execute(
        Buffer.from_numpy(token_ids).to(Accelerator()),
        Buffer.from_numpy(np.array([0, len(token_ids)], dtype=np.uint32)).to(
            Accelerator()
        ),
        Buffer.from_numpy(np.array([1], dtype=np.int64)),
        *Signals(devices=[DeviceRef.GPU()]).buffers(),
        kv_runtime_inputs.blocks.to(Accelerator()),
        kv_runtime_inputs.cache_lengths.to(Accelerator()),
        kv_runtime_inputs.lookup_table.to(Accelerator()),
        kv_runtime_inputs.max_lengths,
        cast(Any, kv_runtime_inputs.attention_dispatch_metadata),
    )[0]

    output = from_dlpack(result)
    assert output.shape[0] == 1
    assert torch.all(torch.isfinite(output))
