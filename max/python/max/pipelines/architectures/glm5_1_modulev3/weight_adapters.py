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
"""Maps GLM-5 checkpoints onto the ModuleV3 modules."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig


def _layer_rules(
    huggingface_config: PretrainedConfig,
) -> tuple[WeightRule, ...]:
    """Returns the renames of a decoder layer's weights.

    The attention holds an MLA projection as a bare tensor, unless the
    checkpoint quantizes it as an FP8 one does: a quantized projection keeps
    its checkpoint names.
    """
    rules: tuple[WeightRule, ...] = (
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
        WeightRenaming(
            r"\.self_attn\.kv_a_layernorm\.weight$",
            ".self_attn.kv_a_proj_layernorm",
        ),
    )
    quantization = getattr(huggingface_config, "quantization_config", None)
    if quantization and quantization.get("quant_method") == "fp8":
        return rules
    return (
        *rules,
        WeightRenaming(
            r"(\.self_attn\.(?:q_a_proj|q_b_proj|q_proj|kv_a_proj_with_mqa"
            r"|kv_b_proj))\.weight$",
            r"\1",
        ),
    )


def glm5_1_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a GLM-5 checkpoint onto the decoder.

    The checkpoint's NextN layer, past the decoder's last one, is left out:
    the multi-token prediction head loads it.

    Args:
        huggingface_config: The checkpoint's config.
    """
    return (
        Ignore(
            rf"^model\.layers\.{huggingface_config.num_hidden_layers}\.",
            reason="the NextN layer, which the multi-token prediction head"
            " loads",
        ),
        *_layer_rules(huggingface_config),
        WeightRenaming(r"^model\.", "language_model."),
        WeightRenaming(r"^lm_head\.", "language_model.lm_head."),
    )


def glm5_1_mtp_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that name the NextN layer as the head's module does.

    The layer ships in the target's checkpoint, past the target's last layer.
    It holds no copy of the embedding and head: the head reads the target's.

    Args:
        huggingface_config: The checkpoint's config.
    """
    nextn = rf"^model\.layers\.{huggingface_config.num_hidden_layers}\."
    return (
        WeightRenaming(nextn + r"shared_head\.norm\.", "shared_head_norm."),
        WeightRenaming(nextn + r"(enorm|hnorm|eh_proj)\.", r"\1."),
        WeightRenaming(nextn, "decoder_layer."),
        Ignore(
            r"^(?:model|lm_head)\.",
            reason="the target's weights, which the target's rules name",
        ),
        *_layer_rules(huggingface_config),
    )
