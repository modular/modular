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
"""Weight adapters for Gemma4 ModuleV3: reuse the graph arch's converter."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.gemma4.weight_adapters import (
    convert_safetensor_language_state_dict,
)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Converts safetensor weights and re-prefixes them under `language_model.`.

    The graph arch's converter filters checkpoint keys down to
    `model.language_model.*` / `language_model.*` and strips those prefixes
    (the graph arch's module tree consumes bare, unprefixed parameter paths).
    Our ModuleV3 module tree nests everything under `language_model`, so
    re-add the prefix here to match the module's parameter paths.
    """
    language = convert_safetensor_language_state_dict(state_dict)
    return {f"language_model.{name}": data for name, data in language.items()}
