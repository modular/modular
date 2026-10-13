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
"""Memory planning for MiMo-V2.6-Flash with its DFlash drafter."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from ..dflash_mimo_v2.weight_adapters import DRAFT_WEIGHTS_FILE
from ..mimo_v2.memory_planner import MiMoV2MemoryPlanner
from .model_config import DFLASH_DIR, repo_file


class MiMoV2DFlashMemoryPlanner(MiMoV2MemoryPlanner):
    """Counts the drafter's weights beside the target's.

    The fused speculative graph loads the drafter from the draft model. The
    base graph that writes the drafter's context reads only its context
    projections, from the drafter the target checkpoint ships in
    ``dflash/``; counting the whole file overstates that by the drafter's
    other weights.
    """

    @override
    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Returns the target's adapted weights plus the drafter's file."""
        target = super().estimate_weights_size(pipeline_config)
        if pipeline_config.draft_model is not None:
            return target + pipeline_config.draft_model.weights_size()
        drafter = repo_file(
            pipeline_config.model.huggingface_weight_repo,
            f"{DFLASH_DIR}/{DRAFT_WEIGHTS_FILE}",
        )
        return target + drafter.stat().st_size
