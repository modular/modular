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
"""Memory planner for GLM-5.3-Flash.

Two costs sit outside the framework's default KV accounting, and either can be
the binding constraint depending on context length:

* **The KDA state pools.** 146 MiB per sequence at float32 -- 136 MiB of
  recurrent state plus 9.6 MiB of conv state across 34 layers -- and
  *independent of context length*. Below roughly 12K context this dominates a
  sequence's cost and caps concurrency before the KV pool fills. SGLang says
  the same thing operationally: "the KDA state pool can limit concurrency
  before the KV pool is full."
* **The 4x mHC residual widening.** A prefill chunk of ``T`` tokens holds
  ``T * hc_mult * hidden_size`` elements per live residual tensor, so the chunk
  size that fits a given activation budget drops by roughly ``hc_mult``
  relative to a single-stream model of the same width -- 268 MB at ``T`` =
  8192. Folded in here rather than discovered as an out-of-memory error at the
  first long prefill.

Everything the V3 family already accounts for -- the MLA up-projection, the
expert-parallel routing buffers and the persistent EP SHMEM buffers -- comes
from :class:`~..deepseekV3.memory_planner.DeepseekV3MemoryPlanner`, which this
extends rather than replaces. Deriving from :class:`PagedMemoryPlanner`
instead reserved nothing for any of them, because that base returns ``0``.

Above roughly 12K context the KV cache dominates instead: 11 KiB per token
across the 11 sparse-MLA layers, 11.8 GB for a 1M-token sequence.
"""

from __future__ import annotations

from typing import Any

from max.driver import Device
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3.memory_planner import DeepseekV3MemoryPlanner
from .model_config import Glm5NextConfig

__all__ = ["Glm5NextMemoryPlanner"]


class Glm5NextMemoryPlanner(DeepseekV3MemoryPlanner):
    """Adds the KDA state pools and the widened mHC residual to V3's estimate."""

    _always_signal_buffers = True

    @override
    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        """Infers a memory-safe default ``max_batch_size``.

        The framework's default inference assumes per-request GPU cost is the
        KV cache alone, which understates GLM-5.3-Flash by 146 MiB per
        sequence and OOMs on short-context traffic.
        """
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        inferred = config.infer_optimal_batch_size(
            devices,
            weights_size=weights_size,
            device_memory_utilization=(
                pipeline_config.model.kv_cache.device_memory_utilization
            ),
        )
        return inferred

    @override
    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Adds the widened residual to V3's estimate, and nothing else.

        The conv and recurrent pools are leaves of the multi-cache, so the
        cache allocation already pays for their pages --
        :meth:`~.model_config.Glm5NextConfig.per_request_state_bytes` reads
        the same leaves the pool is built from. Reserving them here as well
        would subtract one request's state twice and shrink the cache by that
        much, which is why the sibling Qwen3.5 planner reserves nothing for
        it either.

        What is left is the mHC residual, which is genuinely an activation:
        ``hc_mult`` copies of the hidden state, live for the whole stack and
        owned by no cache. It is replicated, so it is sized from the whole
        batch. The EP dispatch buffers are not: the feed-forward sublayer
        splits the token axis, which is what makes the inherited per-rank
        estimate right without an override here.
        """
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

    def describe(self) -> dict[str, Any]:
        """Returns the per-sequence costs, for logging a capacity decision."""
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        return {
            "kda_layers": len(config.kda_layers),
            "sparse_attention_layers": len(config.sparse_attention_layers),
            "state_bytes_per_request": config.per_request_state_bytes(),
            "state_dtype": str(config.state_dtype),
            "residual_bytes_per_token": config.activation_bytes_per_token(),
            "mla_latent_width": config.mla_head_dim,
        }
