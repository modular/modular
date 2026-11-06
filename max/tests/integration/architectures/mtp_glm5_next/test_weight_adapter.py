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
"""Where the MTP layer's weights go: dropped by the base, kept for the draft."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.glm5_next.weight_adapters import (
    glm5_next_mtp_weight_rules,
    glm5_next_weight_rules,
    glm5_next_with_mtp_weight_rules,
)
from max.pipelines.lib import WeightTransform

NUM_HIDDEN_LAYERS = 4

_MTP_KEYS = (
    "model.layers.4.enorm.weight",
    "model.layers.4.hnorm.weight",
    "model.layers.4.eh_proj.weight",
    "model.layers.4.shared_head.norm.weight",
    "model.layers.4.input_layernorm.weight",
    "model.layers.4.self_attn.q_a_proj.weight",
    "model.layers.4.self_attn.q_a_proj.weight_scale_inv",
    "model.layers.4.mlp.experts.0.gate_proj.weight",
)
_TARGET_KEYS = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.3.self_attn.q_a_proj.weight",
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


def _base(checkpoint: dict[str, Weights]) -> dict[str, WeightData]:
    return WeightTransform(glm5_next_weight_rules(_hf_config()))(checkpoint)


def test_the_base_rules_drop_the_mtp_layer() -> None:
    """Checks the base graph's strict load sees no MTP layer."""
    out = _base(_checkpoint())
    assert not [name for name in out if name.startswith("layers.4.")]
    assert "layers.3.self_attn.q_a_proj.weight" in out


def test_the_mtp_rules_keep_the_mtp_layer() -> None:
    """Checks the draft's weights survive under the checkpoint's layer index,
    renamed like every other layer's."""
    out = WeightTransform(glm5_next_with_mtp_weight_rules(_hf_config()))(
        _checkpoint()
    )
    mtp = {name for name in out if name.startswith("layers.4.")}
    assert mtp == {
        "layers.4.enorm.weight",
        "layers.4.hnorm.weight",
        "layers.4.eh_proj.weight",
        "layers.4.shared_head.norm.weight",
        "layers.4.input_layernorm.weight",
        "layers.4.self_attn.q_a_proj.weight",
        "layers.4.self_attn.q_a_proj.weight_scale",
        "layers.4.mlp.experts.0.gate_proj.weight",
    }
    assert set(out) - mtp == set(_base(_checkpoint()))


def test_the_draft_rules_name_only_the_mtp_layer() -> None:
    """Checks the MTP layer's own modules sit beside its decoder layer, which
    is renamed like the target's, and the target's weights are left out."""
    out = WeightTransform(glm5_next_mtp_weight_rules(_hf_config()))(
        _checkpoint()
    )
    assert set(out) == {
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "shared_head_norm.weight",
        "decoder_layer.input_layernorm.weight",
        "decoder_layer.self_attn.q_a_proj.weight",
        "decoder_layer.self_attn.q_a_proj.weight_scale",
        "decoder_layer.mlp.experts.0.gate_proj.weight",
    }
