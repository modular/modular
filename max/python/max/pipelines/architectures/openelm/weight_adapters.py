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
"""Weight adapters for OpenELM safetensors checkpoints."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# OpenELM's checkpoint names match max.nn's module attribute names almost
# exactly, with two exceptions:
# - the attention output projection: the checkpoint calls it "out_proj"
#   (matching Apple's reference implementation) while AttentionWithRope
#   names its attribute "o_proj".
# - the final norm: the checkpoint stores it under the "transformer."
#   prefix, but LogitsPostprocessMixin requires it as a top-level "norm"
#   attribute on OpenELMLanguageModel (see model.py for why it can't also
#   live nested under "transformer").
OPENELM_SAFETENSOR_MAPPING = {
    "attn.out_proj.": "attn.o_proj.",
    "transformer.norm.": "norm.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Adapter for OpenELM safetensors checkpoints.

    Nearly a pass-through: OpenELM's checkpoint key names already match
    the names the MAX graph expects, aside from the one rename in
    OPENELM_SAFETENSOR_MAPPING.
    """
    del huggingface_config, pipeline_config, unused_kwargs
    new_state_dict: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        max_name = name
        for before, after in OPENELM_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    return new_state_dict


def list_weight_names(model_dir: str) -> list[str]:
    """Returns all weight key names found in a safetensors model directory.

    Handles both single-file and sharded layouts. Useful for debugging key
    mismatches between the safetensors file and the graph's weight
    references.
    """
    from pathlib import Path

    from safetensors import safe_open

    model_path = Path(model_dir)
    names = []

    safetensor_files = list(model_path.glob("model.safetensors")) + list(
        model_path.glob("model-*.safetensors")
    )

    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt") as f:
            names.extend(f.keys())

    return sorted(names)
