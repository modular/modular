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

Adds the two costs the graph-API planner adds -- the KDA state pools, which cap
concurrency below roughly 12K context, and the 4x mHC residual -- to the
ModuleV3 DeepSeek planner, which reserves the EP buffers the way the ModuleV3
MoE uses them.
"""

from __future__ import annotations

from max.driver import Device
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3_modulev3.memory_planner import DeepseekV3ModuleV3MemoryPlanner
from .model_config import Glm5NextConfig

__all__ = ["Glm5NextMemoryPlanner"]


class Glm5NextMemoryPlanner(DeepseekV3ModuleV3MemoryPlanner):
    """Adds the KDA state pools and the widened mHC residual."""

    _always_signal_buffers = True

    @override
    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        return config.infer_optimal_batch_size(
            devices,
            weights_size=weights_size,
            device_memory_utilization=(
                pipeline_config.model.kv_cache.device_memory_utilization
            ),
        )

    @override
    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        # The state pools are cache leaves and already paid for; the residual
        # streams are a real activation, replicated over the whole batch.
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        residual_bytes = (
            pipeline_config.runtime.max_batch_input_tokens
            * config.activation_bytes_per_token()
        )
        return (
            super().estimate_activation_memory(
                pipeline_config, huggingface_config
            )
            + residual_bytes
        )
