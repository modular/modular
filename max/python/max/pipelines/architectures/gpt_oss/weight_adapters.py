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

import dataclasses

import numpy as np
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights

GPT_OSS_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
    # MoE weight mappings
    ".mlp.router": ".mlp.gate.gate_score",
}


def _convert_mxfp4_weight(
    name: str, data: WeightData
) -> tuple[str, WeightData]:
    """Converts an MXFP4 weight, handling both block and scale tensors.

    Block weights (``*_blocks``) are flattened from 4D to 3D and the suffix
    is stripped.  Scale weights (``*_scales``) are renamed to ``*_scale``
    and reinterpreted as float8_e8m0fnu (safetensors stores them as uint8).
    Non-MXFP4 weights are returned unchanged.
    """
    if name.endswith("_blocks"):
        arr = np.from_dlpack(data)  # type: ignore[arg-type]
        if arr.ndim != 4:
            raise ValueError(
                f"Expected 4D MXFP4 block tensor, got {arr.ndim}D"
                f" for '{data.name}'"
            )
        arr = arr.reshape(*arr.shape[:2], -1)
        data = dataclasses.replace(data, data=arr, shape=Shape(arr.shape))
        return name.removesuffix("_blocks"), data

    if name.endswith("_scales"):
        data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)
        return name.removesuffix("_scales") + "_scale", data

    return name, data


# Native OpenAI-format keys to skip.  The HF-compatible shards
# (model-0000X-of-*.safetensors) provide the same weights under
# standard ``model.layers.*`` naming.
_OPENAI_NATIVE_KEYS = frozenset(
    {"embedding.weight", "unembedding.weight", "norm.scale"}
)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Args:
        state_dict: Dictionary of weight tensors

    Returns:
        Dictionary of converted weight data
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if (
            weight_name.startswith("block.")
            or weight_name in _OPENAI_NATIVE_KEYS
        ):
            continue

        data = value.data()
        max_name, data = _convert_mxfp4_weight(weight_name, data)

        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        if max_name != data.name:
            data = dataclasses.replace(data, name=max_name)
        new_state_dict[max_name] = data

    return new_state_dict
