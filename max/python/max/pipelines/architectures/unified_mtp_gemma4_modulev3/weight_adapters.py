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

"""Weight adapters for the ModuleV3 unified Gemma4 MTP model."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from ..gemma4.weight_adapters import (
    convert_safetensor_language_state_dict,
)

__all__ = [
    "convert_draft_state_dict",
    "convert_safetensor_state_dict",
]


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Registry adapter: converts target language weights only.

    Identical to the graph-API MTP arch's adapter: the draft checkpoint is
    loaded separately in ``model.py`` from the draft model's own files.
    """
    return convert_safetensor_language_state_dict(state_dict)


def convert_draft_state_dict(
    draft_state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Renames the assistant checkpoint's embedding to the draft module path.

    Input is the output of the graph-API model's ``_convert_draft_weights``
    (run by the inherited ``_load_state_dict``), which already stripped the
    HF ``model.`` prefix and materialized ``WeightData`` — but, by design,
    leaves ``embed_tokens.*`` untouched. This function applies that one
    rename, so each transform in the two-step pipeline is applied exactly
    once. It is idempotent (``model.``-stripping no-ops and
    ``draft_embed_tokens.`` no longer starts with ``embed_tokens.``), so it
    is also correct on raw HF keys.

    The draft module holds the assistant's own 1024-dim embedding as
    ``draft_embed_tokens`` (it also serves as its tied lm_head). The
    target's ``embed_tokens.weight`` is shared separately by module
    aliasing in ``model._build_module``.
    """
    converted: dict[str, WeightData] = {}
    for name, data in draft_state_dict.items():
        # Idempotent no-op after _convert_draft_weights' strip; kept so
        # this function is correct on raw HF keys too.
        name = name.removeprefix("model.")
        # Rename embed_tokens -> draft_embed_tokens.
        if name.startswith("embed_tokens."):
            name = "draft_" + name
        converted[name] = data
    return converted
