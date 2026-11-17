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
"""Rules that map the MiMo-V2 DFlash drafter's checkpoint onto its module.

``dflash/dflash_draft_model.safetensors`` names every tensor as the drafter
module does, so no rule renames, relabels or drops one. The trained mask
embedding in ``mask_embedding.pt`` is no weight: the module bakes it into
the graph as a constant.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib.weight_transform import WeightRule
from transformers import PretrainedConfig


def dflash_mimo_v2_weight_rules(
    huggingface_config: PretrainedConfig | None = None,
) -> Sequence[WeightRule]:
    """Returns the rules that map the drafter's checkpoint onto its module.

    Args:
        huggingface_config: The drafter's config, which no rule reads.
    """
    del huggingface_config
    return ()
