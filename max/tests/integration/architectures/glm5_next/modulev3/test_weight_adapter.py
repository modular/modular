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

The checkpoint is mixed precision layer by layer, so an FP8 weight has to be
recognized by its scale, not by its projection's name: every KDA layer has a
BF16 ``q_proj`` that the DeepSeek adapter would mistake for an MLA projection.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.glm5_next_modulev3.weight_adapters import (
    convert_glm5_next_state_dict,
)

_LM = "model.language_model."


class _Weight:
    def __init__(self, data: WeightData) -> None:
        self._data = data

    def data(self) -> WeightData:
        return self._data


def _weight(
    name: str, shape: tuple[int, ...], dtype: npt.DTypeLike = np.float32
) -> _Weight:
    return _Weight(WeightData.from_numpy(np.zeros(shape, dtype=dtype), name))


def _convert(names: dict[str, tuple[int, ...]]) -> dict[str, WeightData]:
    state = {name: _weight(name, shape) for name, shape in names.items()}
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=4))
    return convert_glm5_next_state_dict(state, config)  # type: ignore[arg-type]


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
    # The depthwise conv loses its singleton group axis.
    assert list(out[f"{lm}layers.0.self_attn.q_conv1d"].shape) == [8, 4]


def test_mhc_sites_and_float32_parameters() -> None:
    out = _convert(
        {
            f"{_LM}layers.0.hc_attn_fn": (24, 16),
            f"{_LM}layers.0.hc_ffn_base": (24,),
            f"{_LM}layers.0.self_attn.A_log": (2,),
        }
    )
    lm = "language_model."
    assert set(out) == {
        f"{lm}layers.0.hc_attn.hc_fn",
        f"{lm}layers.0.hc_ffn.hc_base",
        f"{lm}layers.0.self_attn.A_log",
    }
    assert all(w.dtype == DType.float32 for w in out.values())


def test_drops_mtp_layer_and_vision_tower() -> None:
    out = _convert(
        {
            f"{_LM}layers.4.input_layernorm.weight": (4,),
            "model.visual.patch_embed.proj.weight": (4, 4),
            f"{_LM}norm.weight": (4,),
        }
    )
    assert set(out) == {"language_model.norm.weight"}
