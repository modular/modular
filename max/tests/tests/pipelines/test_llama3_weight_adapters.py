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
from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData
from max.pipelines.architectures.llama3.weight_adapters import (
    convert_gguf_state_dict,
    convert_safetensor_state_dict,
)


@dataclass
class _ModelConfigStub:
    graph_quantization_encoding: QuantizationEncoding | None = None
    _quant: bool = False
    quantization_encoding: str | None = None
    _applied_dtype_cast_from: str | None = None
    _applied_dtype_cast_to: str | None = None


@dataclass
class _PipelineConfigStub:
    model: _ModelConfigStub


def _weight_mock() -> NonCallableMock:
    weight = NonCallableMock()
    weight.data.return_value = "mock_weight_data"
    return weight


def _weight_data_mock(data: WeightData) -> NonCallableMock:
    weight = NonCallableMock()
    weight.data.return_value = data
    return weight


def _compressed_tensors_nvfp4_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "weights": {
                        "dynamic": False,
                        "group_size": 16,
                        "num_bits": 4,
                        "strategy": "tensor_group",
                        "type": "float",
                    },
                    "input_activations": {
                        "dynamic": "local",
                        "group_size": 16,
                        "num_bits": 4,
                        "strategy": "tensor_group",
                        "type": "float",
                    },
                }
            },
        }
    )


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


def test_convert_safetensor_state_dict_normalizes_compressed_tensors_nvfp4() -> (
    None
):
    prefix = "model.layers.0.mlp.down_proj"
    packed = WeightData.from_numpy(
        np.array([[0x21, 0x43]], dtype=np.uint8), "packed"
    )
    group_scale = WeightData(
        data=np.array([[0x38, 0x40]], dtype=np.uint8),
        name="group_scale",
        dtype=DType.float8_e4m3fn,
        shape=packed.shape,
    )
    weight_global_scale = WeightData.from_numpy(
        np.array([5408.0], dtype=np.float32), "weight_global_scale"
    )
    input_global_scale = WeightData.from_numpy(
        np.array([300.0], dtype=np.float32), "input_global_scale"
    )
    lm_head = WeightData(
        data=np.zeros((1, 2), dtype=np.uint16),
        name="lm_head.weight",
        dtype=DType.bfloat16,
        shape=packed.shape,
    )
    state_dict = {
        f"{prefix}.weight_packed": _weight_data_mock(packed),
        f"{prefix}.weight_scale": _weight_data_mock(group_scale),
        f"{prefix}.weight_global_scale": _weight_data_mock(weight_global_scale),
        f"{prefix}.input_global_scale": _weight_data_mock(input_global_scale),
        "lm_head.weight": _weight_data_mock(lm_head),
    }

    converted = convert_safetensor_state_dict(
        state_dict,  # type: ignore[arg-type]
        _compressed_tensors_nvfp4_hf_config(),
        _PipelineConfigStub(model=_ModelConfigStub()),  # type: ignore[arg-type]
    )

    canonical = "layers.0.mlp.down_proj"
    assert converted[f"{canonical}.weight"] is packed
    assert converted[f"{canonical}.weight_scale"] is group_scale
    assert converted["lm_head.weight"] is lm_head
    assert converted["lm_head.weight"].dtype == DType.bfloat16
    weight_scale_2 = converted[f"{canonical}.weight_scale_2"]
    input_scale = converted[f"{canonical}.input_scale"]
    assert tuple(weight_scale_2.shape) == (1,)
    assert tuple(input_scale.shape) == (1,)
    np.testing.assert_allclose(
        np.from_dlpack(weight_scale_2), 1.0 / 5408.0, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.from_dlpack(input_scale), 1.0 / 300.0, rtol=1e-6
    )


def test_convert_safetensor_state_dict_reinterprets_raw_nvfp4_scale() -> None:
    raw_scale = WeightData.from_numpy(
        np.array([[0x38, 0x40]], dtype=np.uint8), "raw_scale"
    )
    converted = convert_safetensor_state_dict(
        {
            "model.layers.0.mlp.down_proj.weight_scale": _weight_data_mock(
                raw_scale
            )
        },
        _compressed_tensors_nvfp4_hf_config(),
        _PipelineConfigStub(model=_ModelConfigStub()),  # type: ignore[arg-type]
    )

    scale = converted["layers.0.mlp.down_proj.weight_scale"]
    assert scale.dtype == DType.float8_e4m3fn
    assert scale.data is raw_scale.data


def test_convert_safetensor_state_dict_does_not_reinvert_canonical_scale() -> (
    None
):
    canonical_scale = WeightData.from_numpy(
        np.array([1.0 / 5408.0], dtype=np.float32), "canonical_scale"
    )
    converted = convert_safetensor_state_dict(
        {
            "model.layers.0.mlp.down_proj.weight_scale_2": _weight_data_mock(
                canonical_scale
            )
        },
        _compressed_tensors_nvfp4_hf_config(),
        _PipelineConfigStub(model=_ModelConfigStub()),  # type: ignore[arg-type]
    )

    assert converted["layers.0.mlp.down_proj.weight_scale_2"] is canonical_scale
