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
"""Memory planning for the fused Qwen3.5 MTP graph."""

from __future__ import annotations

from max.pipelines.lib.config import PipelineConfig
from typing_extensions import override

from ..qwen3_5.memory_planner import Qwen3_5MemoryPlanner
from ..qwen3_5.model_config import Qwen3_5Config
from .unified_mtp_qwen3_5 import ring_len_for_config


class UnifiedMTPQwen3_5MemoryPlanner(Qwen3_5MemoryPlanner):
    """Plans memory for the fused Qwen3.5 MTP graph.

    When it infers a batch size, it counts the verify ring's scratch leaf,
    which each request holds beside its state set. The cache budgets the
    state and ring leaves it holds itself.
    """

    @override
    def spec_state_bytes(self, pipeline_config: PipelineConfig) -> int:
        assert isinstance(self._config, Qwen3_5Config)
        return self._config._per_request_ring_bytes(
            ring_len_for_config(pipeline_config.speculative)
        )
