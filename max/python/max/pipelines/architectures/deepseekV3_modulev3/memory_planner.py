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

"""Memory planner for the ModuleV3 DeepseekV3 family."""

from __future__ import annotations

from max.pipelines.lib.config import PipelineConfig
from transformers import AutoConfig

from ..deepseekV3.memory_planner import DeepseekV3MemoryPlanner


class DeepseekV3ModuleV3MemoryPlanner(DeepseekV3MemoryPlanner):
    """Plans the graph-API terms for a ModuleV3 graph that never fuses the send.

    The ModuleV3 expert-parallel MoE always materializes the FFN output before
    the combine, so reserving for it is required whatever
    ``ep_fuse_ffn_combine_send`` resolves to.
    """

    def _ep_fuse_ffn_combine_send(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> bool:
        return False
