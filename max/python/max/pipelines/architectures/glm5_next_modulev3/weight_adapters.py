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
"""Maps a GLM-5.3-Flash checkpoint onto the ModuleV3 module tree.

The modules hold every tensor as the checkpoint stores it, in its dtype and
shape, so the rules only rename and drop.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig

__all__ = [
    "glm5_next_mtp_weight_rules",
    "glm5_next_weight_rules",
    "glm5_next_with_mtp_weight_rules",
]

_MLA_RAW = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj")
"""MLA projections the module holds as raw tensors."""

_UNQUANTIZED_RAW = re.compile(
    rf"layers\.(\d+)\.self_attn\.({'|'.join(_MLA_RAW)})$"
)

_PREFIX_RULES: tuple[WeightRule, ...] = (
    WeightRenaming(r"^model\.language_model\.", "language_model."),
    WeightRenaming(r"^model\.", "language_model."),
    WeightRenaming(r"^lm_head\.", "language_model.lm_head."),
)


def _num_hidden_layers(huggingface_config: PretrainedConfig) -> int:
    text_config = getattr(huggingface_config, "text_config", huggingface_config)
    return int(text_config.num_hidden_layers)


def _raw_projection_rules(
    huggingface_config: PretrainedConfig,
) -> tuple[WeightRule, ...]:
    """Returns the renames of the MLA projections held as bare tensors.

    A quantized projection keeps its checkpoint names beside its scales. An
    FP8 checkpoint is mixed precision layer by layer and lists the
    projections it leaves unquantized; any other checkpoint quantizes none.
    """
    quantization = getattr(huggingface_config, "quantization_config", None)
    if not quantization or quantization.get("quant_method") != "fp8":
        return (
            WeightRenaming(
                rf"(\.self_attn\.(?:{'|'.join(_MLA_RAW)}))\.weight$", r"\1"
            ),
        )
    layers: dict[str, set[str]] = {}
    for module in quantization.get("modules_to_not_convert") or ():
        if (match := _UNQUANTIZED_RAW.search(module)) is not None:
            layers.setdefault(match.group(2), set()).add(match.group(1))
    return tuple(
        WeightRenaming(
            rf"(\.layers\.(?:{'|'.join(sorted(ids, key=int))})"
            rf"\.self_attn\.{projection})\.weight$",
            r"\1",
        )
        for projection, ids in sorted(layers.items())
    )


def _layer_rules(
    huggingface_config: PretrainedConfig,
) -> tuple[WeightRule, ...]:
    """Returns the renames of a decoder layer's weights."""
    return (
        # The mHC sites are flat tensors in the checkpoint, modules here.
        WeightRenaming(r"\.(hc_attn|hc_ffn)_(fn|base|scale)$", r".\1.hc_\2"),
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
        WeightRenaming(
            r"\.self_attn\.kv_a_layernorm\.weight$",
            ".self_attn.kv_a_proj_layernorm",
        ),
        # The k-pool gate is a bare `[128, 4096]` tensor applied as a linear.
        WeightRenaming(
            r"\.indexer\.index_kpool_compress_gate$",
            ".indexer.index_kpool_compress_gate.weight",
        ),
        # KDA's depthwise convs are held as raw tensors.
        WeightRenaming(
            r"\.self_attn\.([qkv])_conv1d\.weight$", r".self_attn.\1_conv1d"
        ),
        *_raw_projection_rules(huggingface_config),
    )


def glm5_next_with_mtp_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the decoder's rules, keeping the checkpoint's MTP layer.

    The layer stays under ``language_model.layers.{num_hidden_layers}.``, so
    the draft's builder reads the precision map off the whole checkpoint.

    Args:
        huggingface_config: The checkpoint's config.
    """
    return (
        Ignore(
            r"^model\.visual\.",
            reason="the vision tower, which the text-only graph does not build",
        ),
        *_layer_rules(huggingface_config),
        *_PREFIX_RULES,
    )


def glm5_next_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a GLM-5.3-Flash checkpoint onto the decoder.

    The checkpoint's MTP layer, past the decoder's last one, is left out:
    the draft loads it.

    Args:
        huggingface_config: The checkpoint's config.
    """
    return (
        Ignore(
            rf"^model\.(?:language_model\.)?layers\."
            rf"{_num_hidden_layers(huggingface_config)}\.",
            reason="the MTP layer, which the draft loads",
        ),
        *glm5_next_with_mtp_weight_rules(huggingface_config),
    )


def glm5_next_mtp_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that name the MTP layer as the draft's module does.

    The layer ships in the target's checkpoint, past the target's last
    layer. Its own embedding and head are left to the target's.

    Args:
        huggingface_config: The checkpoint's config.
    """
    layer = (
        rf"^model\.(?:language_model\.)?layers\."
        rf"{_num_hidden_layers(huggingface_config)}\."
    )
    return (
        # Before the layer moves, while the unquantized projections still
        # carry its index.
        *_layer_rules(huggingface_config),
        WeightRenaming(layer + r"shared_head\.norm\.", "shared_head_norm."),
        WeightRenaming(layer + r"(enorm|hnorm|eh_proj)\.", r"\1."),
        WeightRenaming(layer, "decoder_layer."),
        Ignore(
            r"^(?:model|lm_head)\.",
            reason="the target's weights, which the target's rules name",
        ),
    )
