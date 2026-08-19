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
"""Weight adapter for Eagle3 draft checkpoints (e.g. ``nvidia/Kimi-K2.5-Thinking-Eagle3``)."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import PretrainedConfig

# Eagle3 checkpoint key prefix -> Eagle3DeepseekV3 module path.
_EAGLE3_KEY_MAP: dict[str, str] = {
    "layers.0.hidden_norm.": "hidden_norm.",
    "layers.0.input_layernorm.": "decoder_layer.input_layernorm.",
    "layers.0.self_attn.": "decoder_layer.self_attn.",
    "layers.0.post_attention_layernorm.": "decoder_layer.post_attention_layernorm.",
    "layers.0.mlp.": "decoder_layer.mlp.",
}


def convert_eagle3_draft_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> dict[str, WeightData]:
    """Convert an Eagle3 draft checkpoint to Eagle3DeepseekV3 module keys.

    The Eagle3 checkpoint (``nvidia/Kimi-K2.5-Thinking-Eagle3``) has keys:
    ``fc.*``, ``layers.0.*``, ``norm.*``, ``lm_head.*``.

    All weights are loaded; ``norm`` and ``lm_head`` are kept independent
    from the target model.  Only ``embed_tokens`` is shared from the target.

    Args:
        state_dict: Raw Eagle3 checkpoint.

    Returns:
        State dict with keys matching ``Eagle3DeepseekV3`` module hierarchy.
    """
    result: dict[str, WeightData] = {}

    for name, weight in state_dict.items():
        # Apply key mapping
        mapped = False
        for before, after in _EAGLE3_KEY_MAP.items():
            if name.startswith(before):
                new_name = after + name[len(before) :]
                result[new_name] = weight.data()
                mapped = True
                break

        if not mapped:
            # fc.*, norm.*, lm_head.* pass through directly
            result[name] = weight.data()

    return result
