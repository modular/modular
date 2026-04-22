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

from dataclasses import dataclass
from unittest.mock import NonCallableMock

from max.graph.quantization import QuantizationEncoding
from max.pipelines.architectures.llama3.weight_adapters import (
    convert_gguf_state_dict,
)


@dataclass
class _ModelConfigStub:
    graph_quantization_encoding: QuantizationEncoding | None


@dataclass
class _PipelineConfigStub:
    model: _ModelConfigStub


def _weight_mock() -> NonCallableMock:
    weight = NonCallableMock()
    weight.data.return_value = "mock_weight_data"
    return weight


def test_convert_gguf_state_dict_non_quantized_uses_stacked_linear_keys() -> (
    None
):
    state_dict = {
        "blk.0.attn_q.weight": _weight_mock(),
        "blk.0.attn_k.weight": _weight_mock(),
        "blk.0.attn_v.weight": _weight_mock(),
        "rope_freqs.weight": _weight_mock(),
    }
    pipeline_config = _PipelineConfigStub(
        model=_ModelConfigStub(graph_quantization_encoding=None)
    )

    converted = convert_gguf_state_dict(
        state_dict,  # type: ignore[arg-type]
        pipeline_config=pipeline_config,  # type: ignore[arg-type]
    )

    assert "layers.0.self_attn.q_proj.weight" in converted
    assert "layers.0.self_attn.k_proj.weight" in converted
    assert "layers.0.self_attn.v_proj.weight" in converted
    assert "rope_freqs.weight" not in converted


def test_convert_gguf_state_dict_quantized_keeps_legacy_qkv_keys() -> None:
    state_dict = {
        "blk.0.attn_q.weight": _weight_mock(),
        "blk.0.attn_k.weight": _weight_mock(),
        "blk.0.attn_v.weight": _weight_mock(),
    }
    pipeline_config = _PipelineConfigStub(
        model=_ModelConfigStub(
            graph_quantization_encoding=QuantizationEncoding.Q4_K
        )
    )

    converted = convert_gguf_state_dict(
        state_dict,  # type: ignore[arg-type]
        pipeline_config=pipeline_config,  # type: ignore[arg-type]
    )

    assert "layers.0.self_attn.q_proj.weight" in converted
    assert "layers.0.self_attn.k_proj.weight" in converted
    assert "layers.0.self_attn.v_proj.weight" in converted
