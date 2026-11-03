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
adapter, kept for the draft's."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.glm5_next_modulev3.weight_adapters import (
    convert_glm5_next_state_dict,
)
from max.pipelines.architectures.mtp_glm5_next_modulev3.weight_adapters import (
    convert_glm5_next_with_mtp_state_dict,
)

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


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(num_hidden_layers=NUM_HIDDEN_LAYERS)


def test_the_target_adapter_drops_the_mtp_layer() -> None:
    out = convert_glm5_next_state_dict(_checkpoint(), _hf_config())
    assert not [n for n in out if n.startswith("language_model.layers.4.")]
    assert "language_model.layers.3.self_attn.q_a_proj.weight" in out


def test_the_draft_adapter_keeps_the_mtp_layer_under_modulev3_names() -> None:
    """Checks the layer keeps its FP8 scale's checkpoint name, as every
    ModuleV3 FP8 weight does, and the target's weights are unchanged."""
    out = convert_glm5_next_with_mtp_state_dict(_checkpoint(), _hf_config())
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
    base = convert_glm5_next_state_dict(_checkpoint(), _hf_config())
    assert set(out) - mtp == set(base)
