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
"""Weight adapters for the DFlash draft checkpoint."""

from __future__ import annotations

from typing import Any

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig

from ..llama3.weight_adapters import _convert_safetensor_with_model_config


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: Any,
    pipeline_config: PipelineConfig,
    **unused_kwargs: Any,
) -> dict[str, WeightData]:
    """Converts the draft checkpoint in the draft's own encoding."""
    assert pipeline_config.draft_model is not None
    return _convert_safetensor_with_model_config(
        state_dict, huggingface_config, pipeline_config.draft_model
    )
