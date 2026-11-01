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
"""Maps Qwen3 embedding checkpoint names onto the encoder's tree."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.llama3_modulev3 import weight_adapters


def _for_encoder(state_dict: Mapping[str, WeightData]) -> dict[str, WeightData]:
    # Qwen3-Embedding checkpoints name their weights without the `model.`
    # prefix the Llama 3 adapter maps to `language_model.`, so add it where
    # it is missing. The encoder projects no logits, so `lm_head` is dropped.
    return {
        name
        if name.startswith("language_model.")
        else f"language_model.{name}": value
        for name, value in state_dict.items()
        if "lm_head" not in name
    }


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs: Any
) -> dict[str, WeightData]:
    """Maps the checkpoint's safetensors names onto the encoder."""
    return _for_encoder(
        weight_adapters.convert_safetensor_state_dict(state_dict, **kwargs)
    )


def convert_gguf_state_dict(
    state_dict: dict[str, Weights], **kwargs: Any
) -> dict[str, WeightData]:
    """Maps the checkpoint's GGUF names onto the encoder."""
    return _for_encoder(
        weight_adapters.convert_gguf_state_dict(state_dict, **kwargs)
    )
