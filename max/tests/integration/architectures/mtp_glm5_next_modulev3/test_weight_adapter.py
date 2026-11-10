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
"""Where the MTP layer's weights go on ModuleV3: dropped by the target's
rules, named for the draft's module by its own."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from unittest.mock import NonCallableMock

import numpy as np
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.glm5_next_modulev3.weight_adapters import (
    glm5_next_mtp_weight_rules,
    glm5_next_weight_rules,
    glm5_next_with_mtp_weight_rules,
)
from max.pipelines.lib import WeightRule, WeightTransform
from transformers import PretrainedConfig

NUM_HIDDEN_LAYERS = 4

_MTP_KEYS = (
    "model.language_model.layers.4.enorm.weight",
    "model.language_model.layers.4.hnorm.weight",
    "model.language_model.layers.4.eh_proj.weight",
    "model.language_model.layers.4.shared_head.norm.weight",
    "model.language_model.layers.4.input_layernorm.weight",
    "model.language_model.layers.4.self_attn.q_a_proj.weight",
    "model.language_model.layers.4.self_attn.q_a_proj.weight_scale_inv",
    "model.language_model.layers.4.mlp.gate.weight",
)
_TARGET_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "model.language_model.layers.3.self_attn.q_a_proj.weight",
    "model.language_model.layers.3.self_attn.q_a_proj.weight_scale_inv",
    "lm_head.weight",
)


def _weight(name: str) -> Weights:
    weight = NonCallableMock()
    weight.data.return_value = WeightData.from_numpy(
        np.zeros((2, 2), dtype=np.float32), name
    )
    return weight


def _checkpoint() -> dict[str, Weights]:
    return {name: _weight(name) for name in (*_MTP_KEYS, *_TARGET_KEYS)}


def _hf_config() -> PretrainedConfig:
    return PretrainedConfig(
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        quantization_config={"quant_method": "fp8"},
    )


def _apply(
    rules: Callable[[PretrainedConfig], Sequence[WeightRule]],
) -> dict[str, WeightData]:
    return WeightTransform(rules(_hf_config()))(_checkpoint())


def test_the_target_rules_drop_the_mtp_layer() -> None:
    out = _apply(glm5_next_weight_rules)
    assert not [n for n in out if n.startswith("language_model.layers.4.")]
    assert "language_model.layers.3.self_attn.q_a_proj.weight" in out


def test_the_mtp_layer_keeps_modulev3_names_for_the_precision_map() -> None:
    """Checks the layer keeps its FP8 scale's checkpoint name, as every
    ModuleV3 FP8 weight does, and the target's weights are unchanged."""
    out = _apply(glm5_next_with_mtp_weight_rules)
    mtp = {n for n in out if n.startswith("language_model.layers.4.")}
    assert mtp == {
        "language_model.layers.4.enorm.weight",
        "language_model.layers.4.hnorm.weight",
        "language_model.layers.4.eh_proj.weight",
        "language_model.layers.4.shared_head.norm.weight",
        "language_model.layers.4.input_layernorm.weight",
        "language_model.layers.4.self_attn.q_a_proj.weight",
        "language_model.layers.4.self_attn.q_a_proj.weight_scale_inv",
        "language_model.layers.4.mlp.gate.gate_score.weight",
    }
    assert set(out) - mtp == set(_apply(glm5_next_weight_rules))


def test_the_draft_rules_name_only_the_mtp_layer() -> None:
    """Checks the draft's rules name the layer as its module does and leave
    the target's weights to the target."""
    assert set(_apply(glm5_next_mtp_weight_rules)) == {
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "shared_head_norm.weight",
        "decoder_layer.input_layernorm.weight",
        "decoder_layer.self_attn.q_a_proj.weight",
        "decoder_layer.self_attn.q_a_proj.weight_scale_inv",
        "decoder_layer.mlp.gate.gate_score.weight",
    }
