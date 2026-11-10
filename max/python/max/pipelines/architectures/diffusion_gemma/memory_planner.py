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
"""Memory planner for the DiffusionGemma architecture."""

from __future__ import annotations

from max.pipelines.architectures.gemma4.memory_planner import (
    Gemma4MemoryPlanner,
)


class DiffusionGemmaMemoryPlanner(Gemma4MemoryPlanner):
    """The Gemma4 planner plus the second copy of the shared text weights.

    The encoder and denoiser graphs each hold a copy of the text weights.
    """

    def estimate_weights_size(self, pipeline_config: object) -> int:
        """Returns the checkpoint's weights plus the second text-graph copy."""
        return 2 * super().estimate_weights_size(pipeline_config)
