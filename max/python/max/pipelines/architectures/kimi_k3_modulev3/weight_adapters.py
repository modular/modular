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
"""Maps a Kimi K3 checkpoint onto the text tower's module tree.

:class:`~.model.KimiK3Model` is the registry's top-level module, so its
weights lose the checkpoint's ``language_model.`` prefixes. The modules
declare every tensor as the checkpoint stores it, and the graph fuses and
reshapes them, so the rules only rename, relabel and drop.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.pipelines.lib import Bitcast, Ignore, WeightRenaming, WeightRule
from transformers.configuration_utils import PretrainedConfig

__all__ = ["kimi_k3_weight_rules"]


def kimi_k3_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a Kimi K3 checkpoint onto the text tower.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        Ignore(
            r"^(?:vision_tower|mm_projector)\.",
            reason="the vision tower, which the text tower doesn't serve",
        ),
        WeightRenaming(r"^language_model\.lm_head\.weight$", "lm_head_weight"),
        WeightRenaming(
            r"^language_model\.model\.norm\.weight$", "final_norm_weight"
        ),
        WeightRenaming(
            r"^language_model\.model\.(output_attn_res_(?:proj|norm))\.weight$",
            r"\1_weight",
        ),
        WeightRenaming(r"^language_model\.model\.", ""),
        WeightRenaming(
            r"^(layers\.\d+\.(?:input_layernorm|post_attention_layernorm"
            r"|self_attention_res_proj|self_attention_res_norm|mlp_res_proj"
            r"|mlp_res_norm))\.weight$",
            r"\1_weight",
        ),
        WeightRenaming(r"\.self_attn\.", ".attention."),
        WeightRenaming(r"\.(?:block_sparse_moe|mlp)\.", ".ffn."),
        WeightRenaming(r"\.ffn\.gate\.weight$", ".ffn.gate.gate_score.weight"),
        WeightRenaming(r"(_conv1d)\.weight$", r"\1"),
        WeightRenaming(r"\.A_log$", ".a_log"),
        # MXFP4 experts store their packed bytes as `weight_packed` and their
        # E8M0 scales as `uint8`, which safetensors has no name for.
        WeightRenaming(r"\.weight_packed$", ".weight"),
        Bitcast(
            r"\.experts\.\d+\.w[123]\.weight_scale$",
            src=DType.uint8,
            dst=DType.float8_e8m0fnu,
        ),
    )
