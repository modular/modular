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

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightData
from max.graph.weights.weights import Weights
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.qwen3 import qwen3_arch, qwen3_moe_arch
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.architectures.qwen3.qwen3 import Qwen3
from max.pipelines.architectures.qwen3.weight_adapters import (
    convert_qwen3_moe_state_dict,
)


class _FakeWeights:
    def __init__(self, array: np.ndarray, name: str):
        self._weight_data = WeightData.from_numpy(array, name)

    def data(self) -> WeightData:
        return self._weight_data


def _make_pipeline_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(_quant=True, quantization_encoding="gptq")
    )


def _kv_params(devices: list[DeviceRef]) -> KVCacheParams:
    return KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=128,
        num_layers=1,
        devices=devices,
        data_parallel_degree=1,
    )


def _make_qwen3_gptq_config(devices: list[DeviceRef]) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
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
            group_size=128,
            desc_act=False,
            sym=True,
        ),
        kv_params=_kv_params(devices),
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        rms_norm_eps=1e-6,
        clip_qkv=None,
        norm_method="rms_norm",
        norm_dtype=DType.bfloat16,
        devices=devices,
        num_experts=8,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        mlp_only_layers=[],
        norm_topk_prob=False,
        decoder_sparse_step=1,
        use_subgraphs=True,
    )


def test_qwen3_architectures_support_gptq() -> None:
    assert "gptq" in qwen3_arch.supported_encodings
    assert "gptq" in qwen3_moe_arch.supported_encodings


def test_convert_qwen3_moe_gptq_state_dict_preserves_needed_perm_idx() -> None:
    state_dict = {
        "model.layers.0.self_attn.q_proj.g_idx": _FakeWeights(
            np.array([3, 0, 2, 1], dtype=np.int32),
            "model.layers.0.self_attn.q_proj.g_idx",
        ),
        "model.layers.0.self_attn.k_proj.g_idx": _FakeWeights(
            np.array([2, 1, 0, 3], dtype=np.int32),
            "model.layers.0.self_attn.k_proj.g_idx",
        ),
        "model.layers.0.self_attn.v_proj.g_idx": _FakeWeights(
            np.array([1, 0, 3, 2], dtype=np.int32),
            "model.layers.0.self_attn.v_proj.g_idx",
        ),
        "model.layers.0.mlp.experts.0.gate_proj.g_idx": _FakeWeights(
            np.array([1, 0, 3, 2], dtype=np.int32),
            "model.layers.0.mlp.experts.0.gate_proj.g_idx",
        ),
        "model.layers.0.mlp.experts.0.gate_proj.qzeros": _FakeWeights(
            np.zeros((1, 1), dtype=np.uint32),
            "model.layers.0.mlp.experts.0.gate_proj.qzeros",
        ),
        "model.layers.0.mlp.gate.weight": _FakeWeights(
            np.ones((2, 4), dtype=np.float16),
            "model.layers.0.mlp.gate.weight",
        ),
        "model.layers.0.mlp.experts.0.gate_proj.scales": _FakeWeights(
            np.ones((1, 4), dtype=np.float16),
            "model.layers.0.mlp.experts.0.gate_proj.scales",
        ),
    }
    huggingface_config = SimpleNamespace(quantization_config={"desc_act": True})
    pipeline_config = _make_pipeline_config()

    new_state_dict = convert_qwen3_moe_state_dict(
        cast(dict[str, Weights], state_dict),
        huggingface_config,
        cast(Any, pipeline_config),
    )

    assert "layers.0.self_attn.q_proj.perm_idx" in new_state_dict
    assert "layers.0.mlp.experts.0.gate_proj.perm_idx" in new_state_dict
    assert "layers.0.self_attn.k_proj.perm_idx" not in new_state_dict
    assert "layers.0.self_attn.v_proj.perm_idx" not in new_state_dict
    assert "layers.0.mlp.experts.0.gate_proj.qzeros" not in new_state_dict
    assert "layers.0.mlp.gate.gate_score.weight" in new_state_dict

    q_perm = np.from_dlpack(
        cast(Any, new_state_dict["layers.0.self_attn.q_proj.perm_idx"])
    )
    expert_perm = np.from_dlpack(
        cast(
            Any,
            new_state_dict["layers.0.mlp.experts.0.gate_proj.perm_idx"],
        )
    )
    assert np.array_equal(q_perm, np.array([1, 3, 2, 0], dtype=np.int32))
    assert np.array_equal(expert_perm, np.array([1, 0, 3, 2], dtype=np.int32))

    assert (
        new_state_dict["layers.0.mlp.gate.gate_score.weight"].dtype
        == DType.bfloat16
    )
    assert (
        new_state_dict["layers.0.mlp.experts.0.gate_proj.scales"].dtype
        == DType.float16
    )


def test_convert_qwen3_moe_gptq_state_dict_drops_perm_idx_without_desc_act() -> None:
    state_dict = {
        "model.layers.0.self_attn.q_proj.g_idx": _FakeWeights(
            np.array([3, 0, 2, 1], dtype=np.int32),
            "model.layers.0.self_attn.q_proj.g_idx",
        ),
        "model.layers.0.mlp.experts.0.gate_proj.g_idx": _FakeWeights(
            np.array([1, 0, 3, 2], dtype=np.int32),
            "model.layers.0.mlp.experts.0.gate_proj.g_idx",
        ),
        "model.layers.0.mlp.experts.0.gate_proj.qzeros": _FakeWeights(
            np.zeros((1, 1), dtype=np.uint32),
            "model.layers.0.mlp.experts.0.gate_proj.qzeros",
        ),
        "model.layers.0.mlp.gate.weight": _FakeWeights(
            np.ones((2, 4), dtype=np.float16),
            "model.layers.0.mlp.gate.weight",
        ),
    }
    huggingface_config = SimpleNamespace(
        quantization_config={"desc_act": False}
    )

    new_state_dict = convert_qwen3_moe_state_dict(
        cast(dict[str, Weights], state_dict),
        huggingface_config,
        cast(Any, _make_pipeline_config()),
    )

    assert "layers.0.self_attn.q_proj.perm_idx" not in new_state_dict
    assert "layers.0.mlp.experts.0.gate_proj.perm_idx" not in new_state_dict
    assert "layers.0.mlp.experts.0.gate_proj.qzeros" not in new_state_dict
    assert "layers.0.mlp.gate.gate_score.weight" in new_state_dict
    assert (
        new_state_dict["layers.0.mlp.gate.gate_score.weight"].dtype
        == DType.bfloat16
    )


def test_qwen3_gptq_requires_single_device() -> None:
    with pytest.raises(ValueError) as exc_info:
        Qwen3(
            _make_qwen3_gptq_config(
                [DeviceRef.CPU(), DeviceRef.CPU()]
            )
        )

    assert "single-device execution only" in str(exc_info.value)
