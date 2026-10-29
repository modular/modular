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
"""Memory planner for GLM-5.3-Flash on ModuleV3.

Adds the costs the graph-API planner adds -- the KDA state pools, which cap
concurrency below roughly 12K context, the 4x mHC residual and, on gfx950, the
``kda_chunk`` prefill workspaces -- to the
ModuleV3 DeepSeek planner, which reserves the EP buffers the way the ModuleV3
MoE uses them.
"""

from __future__ import annotations

from max.driver import Device
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3_modulev3.memory_planner import DeepseekV3ModuleV3MemoryPlanner
from ..glm5_next.memory_planner import amd_kda_chunk_workspace_bytes
from .model_config import Glm5NextConfig

__all__ = ["Glm5NextMemoryPlanner"]


class Glm5NextMemoryPlanner(DeepseekV3ModuleV3MemoryPlanner):
    """Adds the KDA state pools, the mHC residual and the chunk workspaces."""

    _always_signal_buffers = True

    _inferred_max_batch_size: int | None = None
    """The batch size :meth:`infer_max_batch_size` chose, which planning does
    not write back to ``runtime.max_batch_size`` before it asks
    :meth:`estimate_activation_memory`."""

    @override
    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        inferred = config.infer_optimal_batch_size(
            devices,
            weights_size=weights_size,
            device_memory_utilization=(
                pipeline_config.model.kv_cache.device_memory_utilization
            ),
        )
        self._inferred_max_batch_size = inferred
        return inferred

    @override
    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        # The state pools are cache leaves and already paid for; the residual
        # streams are a real activation, replicated over the whole batch, and
        # so are the chunk workspaces gfx950 prefill allocates per call.
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        residual_bytes = (
            pipeline_config.runtime.max_batch_input_tokens
            * config.activation_bytes_per_token()
        )
        chunk_workspace_bytes = amd_kda_chunk_workspace_bytes(
            pipeline_config.runtime,
            num_heads=config.linear_num_heads,
            head_dim=config.linear_head_dim,
            lower_bound=config.linear_lower_bound,
            state_dtype=config.state_pool_dtype,
            max_batch_size=(
                pipeline_config.runtime.max_batch_size
                or self._inferred_max_batch_size
            ),
            max_length=pipeline_config.model.max_length,
        )
        return (
            super().estimate_activation_memory(
                pipeline_config, huggingface_config
            )
            + residual_bytes
            + chunk_workspace_bytes
        )
