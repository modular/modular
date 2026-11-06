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
"""Checkpoint names map onto the ModuleV3 GLM-5.3-Flash module tree.

The checkpoint is mixed precision layer by layer: an FP8 projection keeps its
checkpoint names, and the config lists the projections left unquantized. Every
KDA layer has a BF16 ``q_proj`` that the DeepSeek naming would mistake for an
MLA projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from unittest.mock import NonCallableMock

import numpy as np
import numpy.typing as npt
import pytest
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.glm5_next_modulev3.checkpoint import (
    _stored_hc_fn_dtype,
    check_raw_projections,
)
from max.pipelines.architectures.glm5_next_modulev3.weight_adapters import (
    glm5_next_weight_rules,
)
from max.pipelines.lib import WeightTransform
from transformers import PretrainedConfig

_LM = "model.language_model."


def _weight(
    name: str, shape: tuple[int, ...], dtype: npt.DTypeLike = np.float32
) -> Weights:
    weight = NonCallableMock()
    weight.data.return_value = WeightData.from_numpy(
        np.zeros(shape, dtype=dtype), name
    )
    return weight


def _convert(
    names: Mapping[str, tuple[int, ...]],
    dtype: npt.DTypeLike = np.float32,
    not_converted: Sequence[str] = ("model.layers.3.self_attn.kv_b_proj",),
) -> dict[str, WeightData]:
    state = {name: _weight(name, shape, dtype) for name, shape in names.items()}
    config = PretrainedConfig(
        text_config=PretrainedConfig(num_hidden_layers=4),
        quantization_config={
            "quant_method": "fp8",
            "modules_to_not_convert": list(not_converted),
        },
    )
    return WeightTransform(glm5_next_weight_rules(config))(state)


def test_mixed_precision_names() -> None:
    out = _convert(
        {
            f"{_LM}layers.0.self_attn.q_proj.weight": (8, 4),
            f"{_LM}layers.0.self_attn.q_conv1d.weight": (8, 1, 4),
            f"{_LM}layers.0.mlp.gate_proj.weight": (8, 4),
            f"{_LM}layers.0.mlp.gate_proj.weight_scale_inv": (1, 1),
            f"{_LM}layers.3.self_attn.q_a_proj.weight": (8, 4),
            f"{_LM}layers.3.self_attn.q_a_proj.weight_scale_inv": (1, 1),
            f"{_LM}layers.3.self_attn.kv_b_proj.weight": (8, 4),
            f"{_LM}layers.3.self_attn.o_proj.weight": (4, 8),
            f"{_LM}layers.3.self_attn.o_proj.weight_scale_inv": (1, 1),
            f"{_LM}layers.3.self_attn.kv_a_layernorm.weight": (4,),
            f"{_LM}layers.3.self_attn.indexer.index_kpool_compress_gate": (
                8,
                4,
            ),
            f"{_LM}layers.3.mlp.gate.weight": (16, 4),
            "lm_head.weight": (16, 4),
        }
    )
    lm = "language_model."
    assert set(out) == {
        # A BF16 KDA projection is a plain linear, whatever its name.
        f"{lm}layers.0.self_attn.q_proj.weight",
        f"{lm}layers.0.self_attn.q_conv1d",
        f"{lm}layers.0.mlp.gate_proj.weight",
        f"{lm}layers.0.mlp.gate_proj.weight_scale_inv",
        # A BF16 raw MLA projection drops `.weight`; an FP8 one keeps its
        # checkpoint names, as a linear does.
        f"{lm}layers.3.self_attn.q_a_proj.weight",
        f"{lm}layers.3.self_attn.q_a_proj.weight_scale_inv",
        f"{lm}layers.3.self_attn.kv_b_proj",
        f"{lm}layers.3.self_attn.o_proj.weight",
        f"{lm}layers.3.self_attn.o_proj.weight_scale_inv",
        f"{lm}layers.3.self_attn.kv_a_proj_layernorm",
        f"{lm}layers.3.self_attn.indexer.index_kpool_compress_gate.weight",
        f"{lm}layers.3.mlp.gate.gate_score.weight",
        f"{lm}lm_head.weight",
    }
    # The depthwise conv keeps the checkpoint's shape.
    assert list(out[f"{lm}layers.0.self_attn.q_conv1d"].shape) == [8, 1, 4]


def test_raw_projections_must_match_their_scales() -> None:
    """A bare MLA projection is one without a scale, whichever way the
    checkpoint's ``modules_to_not_convert`` disagrees."""
    names = {
        f"{_LM}layers.3.self_attn.kv_b_proj.weight": (8, 4),
        f"{_LM}layers.3.self_attn.q_a_proj.weight": (8, 4),
        f"{_LM}layers.3.self_attn.q_a_proj.weight_scale_inv": (1, 1),
    }
    check_raw_projections(_convert(names))
    with pytest.raises(ValueError, match="omits"):
        check_raw_projections(_convert(names, not_converted=()))
    with pytest.raises(ValueError, match="lists"):
        check_raw_projections(
            _convert(
                names,
                not_converted=(
                    "model.layers.3.self_attn.kv_b_proj",
                    "model.layers.3.self_attn.q_a_proj",
                ),
            )
        )


def test_mhc_sites_keep_the_checkpoint_dtype() -> None:
    """The modules hold every tensor in the dtype the checkpoint stores."""
    out = _convert(
        {
            f"{_LM}layers.0.hc_attn_fn": (24, 16),
            f"{_LM}layers.0.hc_ffn_base": (24,),
            f"{_LM}layers.0.self_attn.o_norm.weight": (2,),
        },
        np.float16,
    )
    lm = "language_model."
    assert set(out) == {
        f"{lm}layers.0.hc_attn.hc_fn",
        f"{lm}layers.0.hc_ffn.hc_base",
        f"{lm}layers.0.self_attn.o_norm.weight",
    }
    assert all(w.dtype == DType.float16 for w in out.values())


def test_stored_hc_fn_dtype() -> None:
    """One checkpoint dtype is shared by every site; a mix is rejected."""

    def hc_fn(dtype: DType) -> WeightData:
        return WeightData.from_numpy(np.zeros((2, 2), np.float32), "w").astype(
            dtype
        )

    assert _stored_hc_fn_dtype({}) == DType.float32
    assert (
        _stored_hc_fn_dtype(
            {
                "layers.0.hc_attn.hc_fn": hc_fn(DType.bfloat16),
                "layers.0.hc_ffn.hc_fn": hc_fn(DType.bfloat16),
            }
        )
        == DType.bfloat16
    )
    with pytest.raises(ValueError, match="mixes"):
        _stored_hc_fn_dtype(
            {
                "layers.0.hc_attn.hc_fn": hc_fn(DType.bfloat16),
                "layers.1.hc_attn.hc_fn": hc_fn(DType.float32),
            }
        )


def test_drops_mtp_layer_and_vision_tower() -> None:
    out = _convert(
        {
            f"{_LM}layers.4.input_layernorm.weight": (4,),
            "model.visual.patch_embed.proj.weight": (4, 4),
            f"{_LM}norm.weight": (4,),
        }
    )
    assert set(out) == {"language_model.norm.weight"}
