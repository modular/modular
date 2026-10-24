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

from max.graph.weights import WeightData, Weights


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs: object
) -> dict[str, WeightData]:
    """Names the assistant checkpoint's weights as its module does.

    Its own embedding, which only its tied head reads, is
    ``draft_embed_tokens``: ``embed_tokens`` is the target's.
    """
    converted: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        name = name.removeprefix("model.")
        if name.startswith("embed_tokens."):
            name = f"draft_{name}"
        converted[name] = value.data()
    return converted
