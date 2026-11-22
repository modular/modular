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
"""GLM-5.3-Flash's precision maps, read off the checkpoints' scales.

The stub holds the checkpoint's scale and expert names and dtypes as the
weight rules leave them (``model.language_model.`` becomes
``language_model.``), with two routed experts per layer instead of 288. Its
``ignore`` globs list the KDA layers under ``self_attn``, which a reader keyed
on ``linear_attn`` takes for quantized, and its MTP layer's experts carry no
scale. The FP8 checkpoint's stub checks that its map is unchanged, the MTP
layer included.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from max.dtype import DType
from max.nn.quant_config import QuantFormat
from max.pipelines.architectures.glm5_next.checkpoint import (
    checkpoint_quant_scheme,
)
from max.pipelines.architectures.glm5_next.quantization import (
    Glm5NextQuantScheme,
)

NUM_LAYERS = 45
MTP = NUM_LAYERS
SPARSE = [i for i in range(NUM_LAYERS) if i % 4 == 3]
KDA = [i for i in range(NUM_LAYERS) if i % 4 != 3]
NVFP4_QUANT_CONFIG = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    # Two of the checkpoint's 134 globs: a KDA layer's attention and an MTP
    # layer, which the map must not read.
    "ignore": [
        "model.language_model.layers.0.self_attn*",
        "model.language_model.layers.45*",
    ],
}

_NVFP4 = {
    "weight": DType.uint8,
    "weight_scale": DType.float8_e4m3fn,
    "weight_scale_2": DType.float32,
    "input_scale": DType.float32,
}
_KDA = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_b_proj")
_MLA = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
_INDEXER = ("wq_b", "wk", "weights_proj")


def _stub(*, with_mtp: bool) -> dict[str, SimpleNamespace]:
    names: dict[str, DType] = {
        "language_model.embed_tokens.weight": DType.bfloat16,
        "language_model.lm_head.weight": DType.bfloat16,
    }
    proj = ("gate_proj", "up_proj", "down_proj")
    for layer in range(NUM_LAYERS + with_mtp):
        prefix = f"language_model.layers.{layer}"
        attn = _MLA if layer in SPARSE or layer == MTP else _KDA
        for p in attn:
            names[f"{prefix}.self_attn.{p}.weight"] = DType.bfloat16
        if layer in SPARSE or layer == MTP:
            for p in _INDEXER:
                names[f"{prefix}.self_attn.indexer.{p}.weight"] = DType.bfloat16
        if layer < 3:
            for p in proj:
                for suffix, dtype in _NVFP4.items():
                    names[f"{prefix}.mlp.{p}.{suffix}"] = dtype
            continue
        names[f"{prefix}.mlp.gate.weight"] = DType.bfloat16
        for p in proj:
            names[f"{prefix}.mlp.shared_experts.{p}.weight"] = DType.bfloat16
            for expert in range(2):
                name = f"{prefix}.mlp.experts.{expert}.{p}"
                if layer == MTP:
                    names[f"{name}.weight"] = DType.bfloat16
                    continue
                for suffix, dtype in _NVFP4.items():
                    names[f"{name}.{suffix}"] = dtype
    return {name: SimpleNamespace(dtype=dtype) for name, dtype in names.items()}


def _fp8_stub() -> dict[str, SimpleNamespace]:
    """The FP8 checkpoint: every MLP, and the sparse and MTP layers' four
    attention projections, carry a block scale; ``kv_b_proj`` and the
    indexer do not."""
    names: dict[str, DType] = {}
    proj = ("gate_proj", "up_proj", "down_proj")
    for layer in range(NUM_LAYERS + 1):
        prefix = f"language_model.layers.{layer}"
        mlps = (
            [f"mlp.{p}" for p in proj]
            if layer < 3
            else [
                f"mlp.{block}.{p}"
                for block in ("experts.0", "experts.1", "shared_experts")
                for p in proj
            ]
        )
        sparse = layer in SPARSE or layer == MTP
        scaled = [
            *mlps,
            *(f"self_attn.{p}" for p in _MLA if sparse and p != "kv_b_proj"),
        ]
        for module in scaled:
            names[f"{prefix}.{module}.weight"] = DType.float8_e4m3fn
            names[f"{prefix}.{module}.weight_scale_inv"] = DType.float32
        if sparse:
            names[f"{prefix}.self_attn.kv_b_proj.weight"] = DType.bfloat16
            for p in _INDEXER:
                names[f"{prefix}.self_attn.indexer.{p}.weight"] = DType.bfloat16
    return {name: SimpleNamespace(dtype=dtype) for name, dtype in names.items()}


def _scheme(state_dict: dict[str, SimpleNamespace]) -> Glm5NextQuantScheme:
    scheme = checkpoint_quant_scheme(
        NVFP4_QUANT_CONFIG, DType.bfloat16, state_dict
    )
    assert scheme is not None
    return scheme


def test_the_mlps_are_nvfp4_and_everything_else_bf16() -> None:
    scheme = _scheme(_stub(with_mtp=True))
    config = scheme.config
    assert config.is_nvfp4
    assert scheme.mlp_layers == frozenset(range(NUM_LAYERS))
    assert scheme.attn_layers == frozenset()
    assert config.mlp_quantized_layers == set(range(NUM_LAYERS))
    assert config.attn_quantized_layers == set()
    assert all(scheme.attn_config(i) is None for i in [*SPARSE, *KDA, MTP])
    assert not config.shared_experts_use_quant(DType.uint8)
    assert config.shared_experts_dtype(DType.uint8) == DType.bfloat16
    assert config.embedding_output_dtype == DType.bfloat16
    assert scheme.compute_dtype == DType.bfloat16


def test_the_mtp_layer_builds_bf16_experts_and_dispatches_bf16() -> None:
    draft = _scheme(_stub(with_mtp=True))
    assert draft.mlp_config(MTP) is None
    assert draft.moe_layers == frozenset(range(3, NUM_LAYERS + 1))
    assert not draft.quantizes_every_moe
    # The target's rules drop the MTP layer, so its exchange stays NVFP4.
    target = _scheme(_stub(with_mtp=False))
    assert target.mlp_config(3) is target.config
    assert target.quantizes_every_moe


@pytest.mark.parametrize(
    "scaled",
    [
        "language_model.layers.3.self_attn.q_b_proj",
        "language_model.layers.3.self_attn.indexer.wq_b",
        "language_model.layers.0.self_attn.q_proj",
        "language_model.layers.3.mlp.shared_experts.gate_proj",
    ],
)
def test_a_scale_on_an_excluded_projection_raises(scaled: str) -> None:
    state_dict = _stub(with_mtp=True)
    state_dict[f"{scaled}.weight_scale_2"] = SimpleNamespace(
        dtype=DType.float32
    )
    with pytest.raises(ValueError, match="stores it in bfloat16"):
        _scheme(state_dict)


def test_another_modelopt_algorithm_raises() -> None:
    with pytest.raises(ValueError, match="quant_algo 'NVFP4'"):
        checkpoint_quant_scheme(
            {**NVFP4_QUANT_CONFIG, "quant_algo": "FP8"},
            DType.bfloat16,
            _stub(with_mtp=False),
        )


def test_the_fp8_map_quantizes_the_mtp_layer_and_dispatches_fp8() -> None:
    scheme = checkpoint_quant_scheme(
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
        DType.bfloat16,
        _fp8_stub(),
    )
    assert scheme is not None
    assert scheme.config.format == QuantFormat.BLOCKSCALED_FP8
    assert scheme.mlp_layers == frozenset(range(NUM_LAYERS + 1))
    assert scheme.attn_layers == frozenset({*SPARSE, MTP})
    assert scheme.mlp_config(MTP) is scheme.config
    assert scheme.quantizes_every_moe
