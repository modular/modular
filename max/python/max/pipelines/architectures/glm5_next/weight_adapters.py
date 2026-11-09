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
"""Maps a GLM-5.3-Flash checkpoint onto the graph-API modules.

The modules hold every tensor as the checkpoint stores it, in its dtype and
shape, so the rules only rename and drop. Beyond prefix stripping, the FP8
scales take the ``weight_scale`` name MAX's blockscaled-FP8 path reads, and
:mod:`.quantization` derives the whole precision map from which tensors have
one.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig

__all__ = [
    "VISION_CHECKPOINT_PREFIX",
    "VISION_WEIGHT_PREFIX",
    "glm5_next_decoder_weight_rules",
    "glm5_next_mtp_weight_rules",
    "glm5_next_vision_weight_rules",
    "glm5_next_weight_rules",
    "glm5_next_with_mtp_weight_rules",
]

# Checkpoint prefix for the vision tower, and the MAX module it loads into.
VISION_CHECKPOINT_PREFIX = "model.visual."
VISION_WEIGHT_PREFIX = "vision_encoder."

_VISION_RULES: tuple[WeightRule, ...] = (
    WeightRenaming(r"^model\.visual\.", VISION_WEIGHT_PREFIX),
    # The tower's stacked QKV loads into a StackedLinear.
    WeightRenaming(r"^(vision_encoder\..*)\.attn\.qkv\.", r"\1.attn.qkv_proj."),
)

_LAYER_RULES: tuple[WeightRule, ...] = (
    # The checkpoint stores the pool-compression gate as a bare `[128, 4096]`
    # tensor, which the reference applies as `F.linear(hidden_states, gate)`.
    # MAX declares it as a bias-less `Linear`, so it needs the `.weight` leaf.
    # Its sibling `index_kpool_compress_ape` is genuinely a bare parameter --
    # it is *added* to the gate logits, not multiplied -- and keeps its name.
    WeightRenaming(
        r"\.indexer\.index_kpool_compress_gate$",
        ".indexer.index_kpool_compress_gate.weight",
    ),
    # The MoE router lives behind `gate_score` in MAX's MoE gate. Anchored on
    # `mlp.gate.weight` so it cannot touch a dense layer's `gate_proj` or the
    # router's `e_score_correction_bias`.
    WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
    WeightRenaming(r"\.weight_scale_inv$", ".weight_scale"),
)

_LANGUAGE_RULES: tuple[WeightRule, ...] = (
    # The multimodal prefix before the plain one, or `model.` would strip
    # only half of it.
    WeightRenaming(r"^model\.language_model\.", ""),
    WeightRenaming(r"^model\.", ""),
    *_LAYER_RULES,
)


def _mtp_layer(huggingface_config: PretrainedConfig) -> str:
    text_config = getattr(huggingface_config, "text_config", huggingface_config)
    return (
        rf"^model\.(?:language_model\.)?layers\."
        rf"{text_config.num_hidden_layers}\."
    )


def glm5_next_with_mtp_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the decoder's and the vision tower's rules, keeping the MTP layer.

    The layer stays under ``layers.{num_hidden_layers}.``, where the draft
    reads it. The tower's weights go under :data:`VISION_WEIGHT_PREFIX`.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (*_VISION_RULES, *_LANGUAGE_RULES)


def glm5_next_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a GLM-5.3-Flash checkpoint onto the model.

    The checkpoint's MTP layer, past the decoder's last one, is left out:
    only the speculative architecture builds it.

    Args:
        huggingface_config: The checkpoint's config.
    """
    return (
        Ignore(
            _mtp_layer(huggingface_config),
            reason="the MTP layer, which only the draft loads",
        ),
        *glm5_next_with_mtp_weight_rules(huggingface_config),
    )


def glm5_next_decoder_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that name the decoder graph's weights.

    Args:
        huggingface_config: The checkpoint's config.
    """
    return (
        Ignore(
            r"^model\.visual\.",
            reason="the vision tower, which its own graph loads",
        ),
        *glm5_next_weight_rules(huggingface_config),
    )


def glm5_next_vision_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that name the vision graph's weights.

    The graph holds the tower under :data:`VISION_WEIGHT_PREFIX`.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        *_VISION_RULES,
        Ignore(
            r"^(?:model|lm_head)\.",
            reason="the language model, which the decoder graph loads",
        ),
    )


def glm5_next_mtp_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that name the MTP layer as the draft's module does.

    The layer ships in the target's checkpoint, past the target's last
    layer. Its own modules sit beside its decoder layer, which is renamed
    like any other layer, and its embedding and head are the target's.

    Args:
        huggingface_config: The checkpoint's config.
    """
    layer = _mtp_layer(huggingface_config)
    return (
        WeightRenaming(layer + r"shared_head\.norm\.", "shared_head_norm."),
        WeightRenaming(layer + r"(enorm|hnorm|eh_proj)\.", r"\1."),
        WeightRenaming(layer, "decoder_layer."),
        Ignore(
            r"^(?:model|lm_head)\.",
            reason="the target's weights, which the target's rules name",
        ),
        *_LAYER_RULES,
    )
