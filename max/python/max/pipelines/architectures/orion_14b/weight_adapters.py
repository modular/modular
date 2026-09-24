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

"""Weight adapters for Orion checkpoints.

The llama3 donor adapter drops every key ending in ``.bias`` when the Hugging
Face config carries a ``quantization_config``. Orion stores 81 learned
LayerNorm biases under that suffix, so the mapping is reimplemented here rather
than delegated.
"""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.llama3.weight_adapters import (
    convert_gguf_state_dict,
)
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_dtype_cast
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig

from .model_config import OrionConfig

# Maps from safetensor to MAX weight names.
ORION_SAFETENSOR_MAPPING = {
    "model.": "",
}

__all__ = ["convert_gguf_state_dict", "convert_safetensor_state_dict"]


def _convert_safetensor_with_model_config(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    model_config: MAXModelConfig,
) -> dict[str, WeightData]:
    del huggingface_config

    new_state_dict: dict[str, WeightData] = {}
    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in ORION_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    cast_from, cast_to = _select_dtype_cast(
        model_config, OrionConfig.DEFAULT_ENCODING
    )
    if cast_from is not None:
        assert cast_to is not None
        for key, weight_data in new_state_dict.items():
            if weight_data.dtype == supported_encoding_dtype(cast_from):
                new_state_dict[key] = weight_data.astype(
                    supported_encoding_dtype(cast_to)
                )

    return new_state_dict


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    return _convert_safetensor_with_model_config(
        state_dict, huggingface_config, pipeline_config.model
    )
