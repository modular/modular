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
"""The vision converter keeps the calibration scalars.

A checkpoint whose vision config sets ``use_clipped_linears`` ships four
per-module bounds (``input_max``, ``input_min``, ``output_max``, ``output_min``)
beside every vision projection, hanging off a wrapper module whose real weight
is a child named ``linear``. They are model weights, so the converter maps them
onto the vision tower's own names like any other tensor.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures.gemma4.weight_adapters import (
    convert_safetensor_vision_state_dict,
)


def _convert(keys: list[str]) -> set[str]:
    placeholder = Buffer.from_numpy(np.zeros(1, dtype=np.float32))
    weights = SafetensorWeights(
        [],
        tensors=set(keys),
        tensors_to_file_idx={},
        _st_weight_map=dict.fromkeys(keys, placeholder),
    )
    return set(convert_safetensor_vision_state_dict(dict(weights.items())))


def test_vision_calibration_scalars_are_kept() -> None:
    converted = _convert(
        [
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.output_max",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.output_min",
            "model.embed_vision.weight",
        ]
    )
    assert converted == {
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.q_proj.input_max",
        "encoder.layers.0.self_attn.q_proj.input_min",
        "encoder.layers.0.self_attn.q_proj.output_max",
        "encoder.layers.0.self_attn.q_proj.output_min",
        "embed_vision.weight",
    }


def test_language_tensors_stay_out_of_the_vision_converter() -> None:
    """The vision converter only claims the vision tower's tensors."""
    converted = _convert(
        [
            "model.vision_tower.encoder.layers.0.mlp.down_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
        ]
    )
    assert converted == {"encoder.layers.0.mlp.down_proj.weight"}
