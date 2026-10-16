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
"""Memory planning for Nemotron-H with its DSpark draft."""

from __future__ import annotations

from typing import Any

from max.pipelines.lib.config import PipelineConfig
from typing_extensions import override

from ..nemotron_h_modulev3.memory_planner import NemotronHMemoryPlanner
from .model_config import UnifiedDSparkNemotronHConfig


class UnifiedDSparkNemotronHMemoryPlanner(NemotronHMemoryPlanner):
    """The Nemotron-H planner, driven by the fused config's target half.

    The draft's KV leaf is priced through the fused config's KV tree. What
    differs is the weights: the draft loads from a second checkpoint that
    ``pipeline_config.model`` does not describe.
    """

    def __init__(self, config: Any) -> None:
        assert isinstance(config, UnifiedDSparkNemotronHConfig)
        super().__init__(config.target)
        self._fused = config

    @override
    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Counts the draft's checkpoint alongside the target's.

        The checkpoint's frozen embedding is counted though it is not loaded,
        and the weights every device holds whole, the context projection and
        the Markov head, once more per further device.
        """
        total = super().estimate_weights_size(pipeline_config)
        draft_model = pipeline_config.draft_model
        if draft_model is not None:
            total += draft_model.weights_size()
            draft = self._fused.draft
            devices = len(self._fused.devices)
            replicated = (
                # fc, NVFP4 or BF16: at most two bytes a weight.
                2
                * draft.num_context_features
                * draft.hidden_size
                * draft.hidden_size
                # markov_w1 in BF16 and markov_w2 at most as much.
                + 4 * draft.vocab_size * draft.markov_rank
            )
            total += replicated * (devices - 1)
        return total
