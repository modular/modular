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
"""Maps Beam checkpoint names onto the ModuleV3 module's parameters."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Strips ``model.`` and moves the router weight under ``gate_score``.

    Args:
        state_dict: The checkpoint tensors, by HuggingFace name.
        **unused_kwargs: Accepted so every adapter shares one signature.

    Returns:
        The weights under their ModuleV3 attribute paths.
    """
    return {
        name.removeprefix("model.").replace(
            ".mlp.gate.weight", ".mlp.gate.gate_score.weight"
        ): value.data()
        for name, value in state_dict.items()
    }
