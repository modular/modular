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

from ..glm5_next_modulev3.weight_adapters import convert_glm5_next_state_dict

__all__ = ["convert_glm5_next_with_mtp_state_dict"]


def convert_glm5_next_with_mtp_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Converts the checkpoint to the ModuleV3 names, keeping the MTP layer.

    The MTP layer stays under ``language_model.layers.{num_hidden_layers}.``.
    The draft's builder reads the precision map off the whole checkpoint, its
    layer included, and splits the layer off by that prefix.
    """
    return convert_glm5_next_state_dict(
        state_dict, huggingface_config, keep_mtp_layer=True
    )
