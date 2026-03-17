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
from max.pipelines.lib import PipelineConfig
from transformers import LlamaConfig

from ..llama3.weight_adapters import (
    _convert_safetensor_with_model_config,
)
from ..llama3.weight_adapters import (
    convert_gguf_state_dict as llama_convert_gguf_state_dict,
)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: LlamaConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict for EAGLE draft models.

    Uses the shared Llama conversion logic with draft_model.
    """
    assert pipeline_config.draft_model is not None
    return _convert_safetensor_with_model_config(
        state_dict, huggingface_config, pipeline_config.draft_model
    )


def convert_gguf_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    return llama_convert_gguf_state_dict(state_dict, **unused_kwargs)


def convert_unified_safetensor_state_dict(
    target_state_dict: dict[str, WeightData],
    draft_state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Convert and merge target + draft weights for the unified model.

    Prefixes target weights with ``target.*`` and draft weights with
    ``draft.*``.  Skips ``embed_tokens`` and ``lm_head`` from the draft
    since those are shared via module aliasing.
    """
    unified: dict[str, WeightData] = {}

    for name, value in target_state_dict.items():
        unified[f"target.{name}"] = value

    shared_prefixes = ("embed_tokens.", "lm_head.")
    for name, value in draft_state_dict.items():
        if name.startswith(shared_prefixes):
            continue
        unified[f"draft.{name}"] = value

    return unified
