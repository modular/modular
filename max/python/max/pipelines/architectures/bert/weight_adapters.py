# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Weight adapters for Bert models.

This module converts HuggingFace BERT weight names to MAX weight names.

HuggingFace Bert naming convention:
- embeddings.word_embeddings.weight
- embeddings.LayerNorm.weight
- encoder.layer.0.attention.self.query.weight
- pooler.dense.weight (not used for embeddings)

MAX naming convention:
- embeddings.word_embeddings.weight
- embeddings.layer_norm.weight
- encoder.layer.0.attention.self.query.weight
"""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights

BERT_SAFETENSOR_MAP: dict[str, str | None] = {
    ".LayerNorm.": ".layer_norm.",
    ".position_ids": None,
    "pooler.": None,
}


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    value: Weights | None

    for weight_name, value in state_dict.items():
        max_name = weight_name

        for before, after in BERT_SAFETENSOR_MAP.items():
            if after is None:
                if before in max_name:
                    value = None
                    break
            else:
                max_name = max_name.replace(before, after)

        if value is not None:
            new_state_dict[max_name] = value.data()

    return new_state_dict
