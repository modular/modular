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

from max.graph.weights import WeightData


def _replace_prefix(key: str, old: str, new: str) -> str:
    if key.startswith(old):
        return new + key[len(old) :]
    return key


def convert_z_image_transformer_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    converted: dict[str, WeightData] = {}

    dropped_prefixes = (
        "x_pad_token",
        "cap_pad_token",
        "siglip_",
    )

    for original_key, value in state_dict.items():
        key = original_key

        if key.startswith(dropped_prefixes):
            continue

        key = _replace_prefix(key, "all_x_embedder.2-1.", "x_embedder.")
        key = _replace_prefix(key, "all_final_layer.2-1.", "final_layer.")
        key = _replace_prefix(key, "t_embedder.mlp.0.", "t_embedder.linear_1.")
        key = _replace_prefix(key, "t_embedder.mlp.2.", "t_embedder.linear_2.")
        key = _replace_prefix(key, "cap_embedder.0.", "cap_norm.")
        key = _replace_prefix(key, "cap_embedder.1.", "cap_proj.")
        key = key.replace("adaLN_modulation.0.", "adaLN_modulation.")
        key = _replace_prefix(
            key,
            "final_layer.adaLN_modulation.1.",
            "final_layer.adaLN_modulation.",
        )

        converted[key] = value

    required_prefixes = (
        "x_embedder.",
        "t_embedder.",
        "cap_norm.",
        "cap_proj.",
        "noise_refiner.0.",
        "context_refiner.0.",
        "layers.0.",
        "final_layer.",
    )
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in converted):
            raise ValueError(
                f"Missing required z-image transformer weights with prefix '{prefix}'"
            )

    allowed_prefixes = (
        "x_embedder.",
        "noise_refiner.",
        "context_refiner.",
        "t_embedder.",
        "cap_norm.",
        "cap_proj.",
        "layers.",
        "final_layer.",
    )
    unexpected_keys = [
        k
        for k in converted
        if not any(k.startswith(p) for p in allowed_prefixes)
    ]
    if unexpected_keys:
        sample = ", ".join(unexpected_keys[:8])
        raise ValueError(
            f"Unexpected z-image transformer keys in phase-1 adapter: {sample}"
        )

    return converted
