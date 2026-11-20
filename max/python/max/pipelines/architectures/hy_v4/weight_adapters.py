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
"""Maps the modelopt MXFP8 Hy4 checkpoints onto the ModuleV3 modules.

The modules declare every tensor as stored; the graph stacks the fused
projections and dequantizes what runs BF16. The rules only rename, relabel
the U8 E8M0 scales as ``float8_e8m0fnu`` and drop the MTP layer.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.pipelines.lib import Bitcast, Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig

_E8M0_SCALES = Bitcast(r"_scale$", src=DType.uint8, dst=DType.float8_e8m0fnu)

# Inside a decoder block, the main layers' and the MTP layer's alike.
_BLOCK_RULES: tuple[WeightRule, ...] = (
    WeightRenaming(
        r"\.hc_(attn|mlp)_layer\.hc_pre\.hc_(fn|base|scale)$", r".hc_\1.\2"
    ),
    WeightRenaming(
        r"\.self_attn\.kv_a_layernorm\.weight$",
        ".self_attn.kv_a_proj_layernorm",
    ),
    WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
    WeightRenaming(
        r"\.mlp\.experts\.(gate_up_proj|down_proj)$", r".mlp.experts.\1.weight"
    ),
    WeightRenaming(
        r"\.mlp\.experts\.(gate_up_proj|down_proj)_scale$",
        r".mlp.experts.\1.weight_scale",
    ),
)


def hy_v4_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a Hy4 checkpoint onto the decoder.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        Ignore(
            r"^model\.mtp_layers\.",
            reason="the MTP layer, which the multi-token prediction draft loads",
        ),
        _E8M0_SCALES,
        *_BLOCK_RULES,
        WeightRenaming(
            r"^model\.hc_head\.hc_head_", "language_model.hc_head.hc_"
        ),
        WeightRenaming(r"^model\.", "language_model."),
        WeightRenaming(r"^lm_head\.", "language_model.lm_head."),
    )


def hy_v4_mtp_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that name the MTP layer as the draft's module does.

    The layer ships in the target's checkpoint; the draft reads the target's
    embedding and head.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        Ignore(
            r"^(?:model\.(?:layers|embed_tokens|norm|hc_head)|lm_head)\.",
            reason="the target's weights, which the target's rules name",
        ),
        _E8M0_SCALES,
        WeightRenaming(r"^model\.mtp_layers\.0", ""),
        *_BLOCK_RULES,
        WeightRenaming(r"^\.", ""),
    )
