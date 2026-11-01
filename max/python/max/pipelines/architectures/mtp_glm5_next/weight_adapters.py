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
"""Weight adapter for the MTP layer a GLM-5.3-Flash checkpoint ships."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

from ..glm5_next.weight_adapters import convert_glm5_next_state_dict

__all__ = ["convert_glm5_next_with_mtp_state_dict"]


def convert_glm5_next_with_mtp_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Converts the checkpoint, keeping the MTP layer for the draft.

    The MTP layer stays under ``layers.{num_hidden_layers}.``, the name the
    base adapter gives it. The draft's builder splits it off by that prefix,
    after the precision map has read its weight scales.
    """
    return convert_glm5_next_state_dict(
        state_dict, huggingface_config, keep_mtp_layer=True
    )
