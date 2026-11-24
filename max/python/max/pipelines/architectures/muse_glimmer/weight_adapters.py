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
"""Maps Muse Glimmer checkpoints onto the ModuleV3 modules."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import WeightRenaming, WeightRule
from transformers import PretrainedConfig


def muse_glimmer_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a Muse Glimmer checkpoint onto its modules.

    The text model's weights go under ``decoder.`` and the vision tower's
    under ``vision.``, the prefixes each registered module takes its weights
    from.

    Args:
        huggingface_config: The checkpoint's config.
    """
    del huggingface_config
    return (
        WeightRenaming(r"^model\.language_model\.", "decoder.language_model."),
        WeightRenaming(r"^lm_head\.", "decoder.language_model.lm_head."),
        # A bare tensor in the module, an nn.Embedding in the checkpoint.
        WeightRenaming(
            r"^model\.vision_tower\.patch_embedder\.position_embedding_table"
            r"\.weight$",
            "vision.vision_tower.patch_embedder.position_embedding_table",
        ),
        WeightRenaming(
            r"^model\.(vision_tower|vision_adapter|vision_projection)\.",
            r"vision.\1.",
        ),
    )
