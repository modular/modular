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

"""Maps Kimi K2.5 checkpoints onto the ModuleV3 decoder and vision encoder.

:func:`kimik2_5_weight_rules` names a checkpoint's language model under
``language_model.`` and its vision tower and projector under
``vision_encoder.``, the prefixes the config's ``weight_prefixes`` take each
module's weights from.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers.configuration_utils import PretrainedConfig

_VISION_RENAMES: tuple[WeightRule, ...] = (
    WeightRenaming(r"^vision_tower\.", "vision_encoder."),
    # Kimi K2.5 numbers the projector's linears as a Sequential's;
    # Kimi-VL names them.
    WeightRenaming(
        r"^(?:mm|multi_modal)_projector\.", "vision_encoder.patch_merger."
    ),
    WeightRenaming(
        r"^(vision_encoder\.patch_merger\.)(?:proj\.0|linear_1)\.",
        r"\1linear1.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.patch_merger\.)(?:proj\.2|linear_2)\.",
        r"\1linear2.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.)final_layernorm\.", r"\1norm."
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.blocks\.\d+\.)(wqkv|wo)\.",
        r"\1attn.\2.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.blocks\.\d+\.mlp\.)fc0\.",
        r"\1up_proj.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.blocks\.\d+\.mlp\.)fc1\.",
        r"\1down_proj.",
    ),
)


def kimik2_5_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a Kimi K2.5 checkpoint onto its modules.

    The decoder holds each MLA projection as a bare tensor. The checkpoint's
    NextN layer, past the decoder's last one, is left out.

    Args:
        huggingface_config: The checkpoint's config.
    """
    text_config = huggingface_config.get_text_config()
    return (
        Ignore(
            rf"^language_model\.model\.layers\.{text_config.num_hidden_layers}\.",
            reason="the NextN layer, which only multi-token prediction loads",
        ),
        Ignore(
            r"\.rotary_emb\.inv_freq$",
            reason="rotary frequencies, which the model computes",
        ),
        Ignore(
            r"\.(k|v)_scale$",
            reason="modelopt FP8 KV-cache scales, which MAX does not read",
        ),
        WeightRenaming(r"^language_model\.model\.", "language_model."),
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
        WeightRenaming(
            r"\.self_attn\.kv_a_layernorm\.weight$",
            ".self_attn.kv_a_proj_layernorm",
        ),
        WeightRenaming(
            r"(\.self_attn\.(?:q_a_proj|q_b_proj|q_proj|kv_a_proj_with_mqa"
            r"|kv_b_proj))\.weight$",
            r"\1",
        ),
        *_VISION_RENAMES,
    )
