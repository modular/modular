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
"""Tests the routed-expert format a checkpoint's quantization config selects.

DeepSeek's own release stores the trunk's routed experts in MXFP4, the
format of the DSpark stages' experts; NVIDIA's ModelOpt release stores them
in NVFP4. The format decides the quant-config validation, the weight
adapter's stacking, the MoE module and the expert-parallel dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.pipelines.architectures.deepseekV41 import weight_adapters
from max.pipelines.architectures.deepseekV41.layers.moe import (
    DeepseekV41MoE,
    DeepseekV41Mxfp4RoutedExperts,
    DeepseekV41NvFp4RoutedExperts,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
    RoutedExpertFormat,
)
from max.pipelines.architectures.deepseekV41.weight_adapters import (
    DeferredExpertStack,
    convert_weight_data,
    interleave_nvfp4_scales,
    resolve_quantization_info,
    with_dtype,
)
from transformers.configuration_utils import PretrainedConfig

# DeepSeek's own quantization config: MXFP4 experts, MXFP8 linears.
MXFP4_QUANTIZATION_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}
# The ModelOpt fields of NVIDIA's NVFP4 release.
NVFP4_QUANTIZATION_CONFIG = {
    **MXFP4_QUANTIZATION_CONFIG,
    "group_size": 16,
    "ignore": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
    "kv_cache_quant_algo": None,
    "moe_quant_algo": "NVFP4",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
        "layers.0.ffn.experts": {"quant_algo": "NVFP4", "group_size": 16}
    },
}

HIDDEN = 128
INTER = 128
EXPERTS = 8
PREFIX = "layers.0.ffn.experts"


@pytest.mark.parametrize(
    ("quantization_config", "expected"),
    [
        (MXFP4_QUANTIZATION_CONFIG, RoutedExpertFormat.MXFP4),
        (NVFP4_QUANTIZATION_CONFIG, RoutedExpertFormat.NVFP4),
        (
            {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            RoutedExpertFormat.NVFP4,
        ),
        (None, RoutedExpertFormat.NVFP4),
    ],
)
def test_format_follows_the_quantization_config(
    quantization_config: Mapping[str, Any] | None,
    expected: RoutedExpertFormat,
) -> None:
    """ModelOpt configs are NVFP4; DeepSeek's ``expert_dtype: fp4`` without
    a ModelOpt map is MXFP4."""
    got = RoutedExpertFormat.from_quantization_config(quantization_config)
    assert got is expected


def test_unknown_expert_quantization_is_refused() -> None:
    with pytest.raises(ValueError, match="expert_dtype"):
        RoutedExpertFormat.from_quantization_config(
            {"quant_method": "fp8", "expert_dtype": "fp8"}
        )


def test_mxfp4_activations_dispatch_as_mxfp8() -> None:
    fmt = RoutedExpertFormat.MXFP4
    assert fmt.dispatch_dtype == DType.float8_e4m3fn
    assert fmt.activation_quant_config.is_mxfp8
    assert RoutedExpertFormat.NVFP4.activation_quant_config.is_nvfp4


def _hf_config(quantization_config: Mapping[str, Any]) -> PretrainedConfig:
    return PretrainedConfig(
        quantization_config=dict(quantization_config),
        text_config=PretrainedConfig(
            num_hidden_layers=1,
            n_routed_experts=EXPERTS,
            dspark_n_routed_experts=EXPERTS,
        ),
    )


def test_native_quantization_config_is_accepted() -> None:
    """DeepSeek's config has no ModelOpt ``quantized_layers`` map."""
    info = resolve_quantization_info(
        _hf_config(MXFP4_QUANTIZATION_CONFIG), {}, num_hidden_layers=1
    )
    assert info is not None
    assert info.routed_expert_format is RoutedExpertFormat.MXFP4
    assert info.nvfp4_expert_layers == frozenset()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight_block_size", [128, 128]),
        ("scale_fmt", "fp32"),
        ("quant_method", "mxfp8"),
    ],
)
def test_native_quantization_config_checks_its_fields(
    field: str, value: object
) -> None:
    config = {**MXFP4_QUANTIZATION_CONFIG, field: value}
    with pytest.raises(ValueError, match=field):
        resolve_quantization_info(_hf_config(config), {}, num_hidden_layers=1)


def _bytes(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.integers(0, 256, shape, dtype=np.uint8)


def _mxfp4_experts(
    prefix: str, n_experts: int, rng: np.random.Generator
) -> dict[str, WeightData]:
    """Per-expert MXFP4 tensors as stored: int8 E2M1 pairs, E8M0 per 32."""
    sd: dict[str, WeightData] = {}
    for e in range(n_experts):
        for proj, n, k in (
            ("w1", INTER, HIDDEN),
            ("w3", INTER, HIDDEN),
            ("w2", HIDDEN, INTER),
        ):
            base = f"{prefix}.{e}.{proj}"
            sd[f"{base}.weight"] = with_dtype(
                _bytes(rng, n, k // 2).view(np.int8),
                DType.int8,
                f"{base}.weight",
            )
            sd[f"{base}.scale"] = with_dtype(
                _bytes(rng, n, k // 32),
                DType.float8_e8m0fnu,
                f"{base}.scale",
            )
    return sd


def _raw(sd: Mapping[str, WeightData], name: str) -> np.ndarray:
    return weight_adapters.as_uint8(sd[name].data)


def test_trunk_mxfp4_experts_stack_by_expert() -> None:
    """``gate_up_proj`` stacks ``[w1; w3]`` per expert in id order, with
    host-interleaved scales; ``down_proj`` stacks ``w2``."""
    checkpoint = _mxfp4_experts(PREFIX, EXPERTS, np.random.default_rng(0))
    out = convert_weight_data(
        checkpoint,
        num_routed_experts=EXPERTS,
        routed_expert_format=RoutedExpertFormat.MXFP4,
    )
    assert set(out) == {
        f"{PREFIX}.{proj}.{kind}"
        for proj in ("gate_up_proj", "down_proj")
        for kind in ("weight", "weight_scale")
    }
    expected = {
        "gate_up_proj": (DType.uint8, (EXPERTS, 2 * INTER, HIDDEN // 2)),
        "down_proj": (DType.uint8, (EXPERTS, HIDDEN, INTER // 2)),
    }
    for proj, (dtype, shape) in expected.items():
        weight = out[f"{PREFIX}.{proj}.weight"]
        assert weight.dtype == dtype
        assert tuple(int(d) for d in weight.shape) == shape
        scale = out[f"{PREFIX}.{proj}.weight_scale"]
        assert scale.dtype == DType.float8_e8m0fnu
        assert tuple(int(d) for d in scale.shape) == (
            *shape[:2],
            shape[2] // 16,
        )

    gate_up = _raw(out, f"{PREFIX}.gate_up_proj.weight")
    gate_up_scale = _raw(out, f"{PREFIX}.gate_up_proj.weight_scale")
    down = _raw(out, f"{PREFIX}.down_proj.weight")
    down_scale = _raw(out, f"{PREFIX}.down_proj.weight_scale")
    for e in range(EXPERTS):
        x = f"{PREFIX}.{e}"
        np.testing.assert_array_equal(
            gate_up[e],
            np.concatenate(
                [
                    _raw(checkpoint, f"{x}.w1.weight"),
                    _raw(checkpoint, f"{x}.w3.weight"),
                ]
            ),
        )
        np.testing.assert_array_equal(
            gate_up_scale[e],
            interleave_nvfp4_scales(
                np.concatenate(
                    [
                        _raw(checkpoint, f"{x}.w1.scale"),
                        _raw(checkpoint, f"{x}.w3.scale"),
                    ]
                )
            ),
        )
        np.testing.assert_array_equal(
            down[e], _raw(checkpoint, f"{x}.w2.weight")
        )
        np.testing.assert_array_equal(
            down_scale[e],
            interleave_nvfp4_scales(_raw(checkpoint, f"{x}.w2.scale")),
        )


def test_dspark_experts_keep_row_major_scales() -> None:
    """DSpark's stages interleave their scales in the graph."""
    prefix = "mtp.0.ffn.experts"
    checkpoint = _mxfp4_experts(prefix, EXPERTS, np.random.default_rng(1))
    out = convert_weight_data(
        checkpoint,
        num_dspark_experts=EXPERTS,
        routed_expert_format=RoutedExpertFormat.NVFP4,
    )
    np.testing.assert_array_equal(
        _raw(out, f"{prefix}.down_proj.weight_scale")[3],
        _raw(checkpoint, f"{prefix}.3.w2.scale"),
    )


def test_deferred_trunk_mxfp4_experts_materialize_the_same_bytes() -> None:
    checkpoint = _mxfp4_experts(PREFIX, EXPERTS, np.random.default_rng(2))
    eager = convert_weight_data(
        checkpoint,
        num_routed_experts=EXPERTS,
        routed_expert_format=RoutedExpertFormat.MXFP4,
    )
    deferred = convert_weight_data(
        checkpoint,
        num_routed_experts=EXPERTS,
        routed_expert_format=RoutedExpertFormat.MXFP4,
        defer_routed_experts=True,
    )
    assert set(deferred) == set(eager)
    for name, value in deferred.items():
        assert isinstance(value, DeferredExpertStack), name
        assert value.dtype == eager[name].dtype
        assert value.shape == eager[name].shape
        np.testing.assert_array_equal(value.materialize(), _raw(eager, name))
        np.testing.assert_array_equal(
            value.materialize(2, 5), _raw(eager, name)[2:5]
        )


def test_a_384_expert_trunk_layer_flushes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trunk stacks at its own expert count, not DSpark's 128."""
    stacked: list[tuple[str, int]] = []
    stack_experts = weight_adapters.stack_experts

    def spy(
        experts: Mapping[int, Mapping[str, np.ndarray]],
        prefix: str,
        **kwargs: bool,
    ) -> dict[str, WeightData]:
        stacked.append((prefix, len(experts)))
        return stack_experts(experts, prefix, **kwargs)

    monkeypatch.setattr(weight_adapters, "stack_experts", spy)
    rng = np.random.default_rng(3)
    checkpoint = _mxfp4_experts(PREFIX, 384, rng) | _mxfp4_experts(
        "mtp.0.ffn.experts", 128, rng
    )
    out = convert_weight_data(
        checkpoint,
        num_routed_experts=384,
        num_dspark_experts=128,
        routed_expert_format=RoutedExpertFormat.MXFP4,
        defer_routed_experts=True,
    )
    assert sorted(stacked) == [
        ("layers.0.ffn.experts", 384),
        ("mtp.0.ffn.experts", 128),
    ]
    assert int(out[f"{PREFIX}.gate_up_proj.weight"].shape[0]) == 384


def test_packed_weights_of_the_other_format_are_refused() -> None:
    checkpoint = _mxfp4_experts(PREFIX, EXPERTS, np.random.default_rng(4))
    with pytest.raises(ValueError, match="nvfp4"):
        convert_weight_data(
            checkpoint,
            num_routed_experts=EXPERTS,
            routed_expert_format=RoutedExpertFormat.NVFP4,
        )


def _config(expert_format: RoutedExpertFormat) -> DeepseekV41Config:
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=MagicMock(num_draft_tokens=0),
        devices=[DeviceRef.CPU()],
        max_seq_len=64,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        n_routed_experts=EXPERTS,
        moe_intermediate_size=INTER,
        num_experts_per_tok=2,
        routed_expert_format=expert_format,
    )


def test_trunk_moe_builds_the_format_experts() -> None:
    nvfp4 = DeepseekV41MoE(
        _config(RoutedExpertFormat.NVFP4), 0, DeviceRef.CPU()
    )
    assert isinstance(nvfp4.experts, DeepseekV41NvFp4RoutedExperts)
    mxfp4 = DeepseekV41MoE(
        _config(RoutedExpertFormat.MXFP4), 0, DeviceRef.CPU()
    )
    assert isinstance(mxfp4.experts, DeepseekV41Mxfp4RoutedExperts)


def test_converted_mxfp4_experts_strict_load_into_the_trunk_experts() -> None:
    """Names, dtypes and shapes match the trunk's MXFP4 experts exactly."""
    moe = DeepseekV41MoE(_config(RoutedExpertFormat.MXFP4), 0, DeviceRef.CPU())
    out = convert_weight_data(
        _mxfp4_experts(PREFIX, EXPERTS, np.random.default_rng(5)),
        num_routed_experts=EXPERTS,
        routed_expert_format=RoutedExpertFormat.MXFP4,
    )
    part = {name.removeprefix(f"{PREFIX}."): data for name, data in out.items()}
    moe.experts.load_state_dict(part, weight_alignment=1, strict=True)
    declared = moe.experts.raw_state_dict()
    assert set(declared) == set(part)
    for name, weight in declared.items():
        assert part[name].dtype == weight.dtype, name
        assert tuple(int(d) for d in part[name].shape) == tuple(
            int(d) for d in weight.shape
        ), name
