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
"""Memory planning for the fused MiMo-V2.6-Flash + DFlash spec graph.

The base planner's reservation, with the expert-parallel buffers sized for
the verify rows the fused graph carries past the scheduler's active-token
budget.
"""

from __future__ import annotations

from max.pipelines.lib.config import PipelineConfig
from typing_extensions import override

from ..mimo_v2_modulev3.memory_planner import MiMoV2ModuleV3MemoryPlanner
from .model_config import verify_widened_row_budget

__all__ = ["UnifiedDflashMiMoV2ModuleV3MemoryPlanner"]


class UnifiedDflashMiMoV2ModuleV3MemoryPlanner(MiMoV2ModuleV3MemoryPlanner):
    """The base planner, counting a verify step's merged draft rows.

    The fused graph's dispatch buffers hold a full decode step's verify
    rows -- every request's drafts beside its active token -- so their
    reservation follows the same widened budget the model sizes the
    buffers with (:func:`~.model_config.verify_widened_row_budget`).
    """

    @override
    def ep_dispatch_row_budget(self, pipeline_config: PipelineConfig) -> int:
        return verify_widened_row_budget(pipeline_config)
