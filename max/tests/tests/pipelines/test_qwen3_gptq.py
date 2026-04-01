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
from max.dtype import DType
from max.graph.weights import WeightData
from max.graph.weights.weights import Weights
from max.pipelines.architectures.qwen3 import qwen3_arch, qwen3_moe_arch
from max.pipelines.architectures.qwen3.weight_adapters import (
    convert_qwen3_moe_state_dict,
)


class _FakeWeights:
    def __init__(self, array: np.ndarray, name: str):
        self._weight_data = WeightData.from_numpy(array, name)

    def data(self) -> WeightData:
        return self._weight_data


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
    pipeline_config = SimpleNamespace(
        model=SimpleNamespace(_quant=True, quantization_encoding="gptq")
    )

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
