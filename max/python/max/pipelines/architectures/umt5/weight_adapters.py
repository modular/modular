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
"""Weight adapters for UMT5 models."""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData


def _clone_weight(weight: WeightData, new_name: str) -> WeightData:
    return WeightData(
        data=weight.data,
        name=new_name,
        dtype=weight.dtype,
        shape=weight.shape,
        quantization_encoding=weight.quantization_encoding,
    )


def convert_safetensor_state_dict(
    state_dict: Mapping[str, WeightData],
) -> dict[str, WeightData]:
    """Ensure shared UMT5 embeddings are available under MAX expected names.

    Hugging Face UMT5 ties `encoder.embed_tokens.weight` to `shared.weight`.
    Some checkpoints include only one of these keys, so this adapter duplicates
    the available embedding to whichever name is missing.
    """
    new_state_dict = dict(state_dict)
    shared_weight = new_state_dict.get("shared.weight")
    encoder_weight = new_state_dict.get("encoder.embed_tokens.weight")

    if shared_weight is None and encoder_weight is None:
        raise ValueError(
            "Missing UMT5 embedding weights. Expected one of "
            "`shared.weight` or `encoder.embed_tokens.weight` to be present."
        )

    if shared_weight is None and encoder_weight is not None:
        new_state_dict["shared.weight"] = _clone_weight(
            encoder_weight, "shared.weight"
        )

    if encoder_weight is None and shared_weight is not None:
        new_state_dict["encoder.embed_tokens.weight"] = _clone_weight(
            shared_weight, "encoder.embed_tokens.weight"
        )

    return new_state_dict
