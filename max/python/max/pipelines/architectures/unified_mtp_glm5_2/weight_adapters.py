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
"""Maps a GLM-5 checkpoint onto the unified MTP module."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import WeightRenaming, WeightRule
from transformers import PretrainedConfig

from ..glm5_1.weight_adapters import GLM5_1_RENAMES


def unified_mtp_glm5_2_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that name a checkpoint for the unified MTP module.

    The target's weights go under ``target.`` and the NextN layer's under
    ``draft.``. The checkpoint holds no copy of the embedding and head under
    the NextN layer: the draft shares the target's.

    Args:
        huggingface_config: The checkpoint's config.
    """
    nextn = rf"^layers\.{huggingface_config.num_hidden_layers}\."
    return (
        *GLM5_1_RENAMES,
        WeightRenaming(
            nextn + r"shared_head\.norm\.", "draft.shared_head_norm."
        ),
        WeightRenaming(nextn + r"(enorm|hnorm|eh_proj)\.", r"draft.\1."),
        WeightRenaming(nextn, "draft.decoder_layer."),
        WeightRenaming.prefix("target."),
        WeightRenaming(r"^target\.draft\.", "draft."),
    )
