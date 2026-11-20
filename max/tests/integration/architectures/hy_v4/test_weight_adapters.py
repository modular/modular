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
"""Checks the Hy4 weight rules against the published checkpoint's names.

One name per pattern from ``tencent/Hy4-preview-FP8``'s index. The rules
must keep each tensor as stored under the name its module declares,
relabel the U8 E8M0 scales, and give the MTP layer to the draft alone.
"""

from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.hy_v4.weight_adapters import (
    hy_v4_mtp_weight_rules,
    hy_v4_weight_rules,
)
from max.pipelines.lib.weight_transform import WeightTransform

_L = "model.layers.3."
_T = "language_model.layers.3."
_MTP = "model.mtp_layers.0."

# Checkpoint name -> the decoder's name.
_TARGET = {
    "model.embed_tokens.weight": "language_model.embed_tokens.weight",
    "model.norm.weight": "language_model.norm.weight",
    "lm_head.weight": "language_model.lm_head.weight",
    "model.hc_head.hc_head_fn": "language_model.hc_head.hc_fn",
    _L + "hc_attn_layer.hc_pre.hc_scale": _T + "hc_attn.scale",
    _L + "hc_mlp_layer.hc_pre.hc_base": _T + "hc_mlp.base",
    _L + "self_attn.q_a_proj.weight": _T + "self_attn.q_a_proj.weight",
    _L + "self_attn.kv_a_proj_with_mqa.weight_scale": (
        _T + "self_attn.kv_a_proj_with_mqa.weight_scale"
    ),
    _L + "self_attn.kv_a_layernorm.weight": _T
    + "self_attn.kv_a_proj_layernorm",
    _L + "self_attn.kv_b_proj.weight": _T + "self_attn.kv_b_proj.weight",
    _L + "self_attn.learnable_sink_param": _T
    + "self_attn.learnable_sink_param",
    _L + "self_attn.indexer.wk.weight": _T + "self_attn.indexer.wk.weight",
    _L + "self_attn.indexer.weights_proj.weight": (
        _T + "self_attn.indexer.weights_proj.weight"
    ),
    _L + "mlp.gate.weight": _T + "mlp.gate.gate_score.weight",
    _L + "mlp.experts.gate_up_proj": _T + "mlp.experts.gate_up_proj.weight",
    _L + "mlp.experts.down_proj_scale": _T
    + "mlp.experts.down_proj.weight_scale",
    _L + "mlp.shared_experts.up_proj.weight": (
        _T + "mlp.shared_experts.up_proj.weight"
    ),
    "model.layers.0.mlp.gate_proj.weight_scale": (
        "language_model.layers.0.mlp.gate_proj.weight_scale"
    ),
}

# Checkpoint name -> the draft's name.
_DRAFT = {
    _MTP + "enorm.weight": "enorm.weight",
    _MTP + "eh_proj.weight": "eh_proj.weight",
    _MTP
    + "self_attn.kv_b_proj.weight_scale": "self_attn.kv_b_proj.weight_scale",
    _MTP + "self_attn.kv_a_layernorm.weight": "self_attn.kv_a_proj_layernorm",
    _MTP
    + "mlp.gate.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    _MTP
    + "mlp.experts.gate_up_proj_scale": "mlp.experts.gate_up_proj.weight_scale",
}


def _weights(names: list[str]) -> dict[str, NonCallableMock]:
    out = {}
    for name in names:
        scale = name.endswith("_scale")
        bits = np.zeros((2, 4), dtype=np.uint8 if scale else np.uint16)
        dtype = DType.uint8 if scale else DType.bfloat16
        buffer = Buffer.from_numpy(bits)
        if not scale:
            buffer = buffer.view(dtype)
        weight = NonCallableMock()
        weight.data.return_value = WeightData(
            buffer, name, dtype, Shape(bits.shape)
        )
        out[name] = weight
    return out


@pytest.mark.parametrize(
    "rules, kept, dropped",
    [
        (hy_v4_weight_rules, _TARGET, _DRAFT),
        (hy_v4_mtp_weight_rules, _DRAFT, _TARGET),
    ],
)
def test_each_rule_set_keeps_its_graph_tensors_and_drops_the_rest(
    rules: object, kept: dict[str, str], dropped: dict[str, str]
) -> None:
    transform = WeightTransform(rules(NonCallableMock()))  # type: ignore[operator]
    adapted = transform(_weights([*kept, *dropped]))
    assert sorted(adapted) == sorted(kept.values())
    for source, name in kept.items():
        expected = (
            DType.float8_e8m0fnu
            if source.endswith("_scale")
            else DType.bfloat16
        )
        assert adapted[name].dtype == expected, source
