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

from typing import Any

from max.driver import Device
from max.pipelines.lib.config import PipelineConfig
from typing_extensions import override

from ..qwen3_5.memory_planner import Qwen3_5MemoryPlanner
from ..qwen3_5.model_config import Qwen3_5Config
from .unified_mtp_qwen3_5 import ring_len_for_config


class UnifiedMTPQwen3_5MemoryPlanner(Qwen3_5MemoryPlanner):
    """Plans memory for the fused Qwen3.5 MTP graph.

    Reserves the verify's shadow pools as activation memory at the resolved
    batch size, since they are allocated outside the state cache after it is
    sized. When it infers a batch size, it also counts the shadows and, on
    the ring rollback, the ring's scratch leaf, which each request holds
    beside its state set. The cache budgets the state and ring leaves it
    holds itself.
    """

    _inferred_max_batch_size: int | None = None
    """Set by :meth:`infer_max_batch_size` for :meth:`estimate_activation_memory`.

    The memory estimator calls both on one planner instance.
    """

    def _config_typed(self) -> Qwen3_5Config:
        assert isinstance(self._config, Qwen3_5Config)
        return self._config

    def shadow_bytes_per_request(self, pipeline_config: PipelineConfig) -> int:
        """Returns the bytes one request occupies in the verify's shadows."""
        return self._config_typed()._per_request_shadow_bytes(
            ring_len_for_config(pipeline_config.speculative)
        )

    @override
    def shadow_state_bytes(self, pipeline_config: PipelineConfig) -> int:
        config = self._config_typed()
        return self.shadow_bytes_per_request(
            pipeline_config
        ) + config._per_request_ring_bytes(
            ring_len_for_config(pipeline_config.speculative)
        )

    @override
    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        self._inferred_max_batch_size = super().infer_max_batch_size(
            pipeline_config, devices, weights_size
        )
        return self._inferred_max_batch_size

    @override
    def estimate_activation_memory(
        self,
        pipeline_config: Any,
        huggingface_config: Any,
    ) -> int:
        """Reserves the verify's shadow pools.

        The ring is drawn from the cache, so it is not reserved here.
        """
        del huggingface_config
        per_request = self.shadow_bytes_per_request(pipeline_config)
        if per_request == 0:
            return super().estimate_activation_memory(pipeline_config, None)
        max_batch = pipeline_config.runtime.max_batch_size
        if max_batch is None:
            max_batch = self._inferred_max_batch_size
        assert max_batch is not None, (
            "infer_max_batch_size must run before estimate_activation_memory"
            " when max_batch_size is unset"
        )
        return max_batch * per_request + super().estimate_activation_memory(
            pipeline_config, None
        )
