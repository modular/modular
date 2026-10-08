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
"""Memory planner for GLM-5.x without the MTP draft."""

from __future__ import annotations

from typing import Any

from max.pipelines.architectures.deepseekV3.memory_planner import (
    _ep_max_rank_send_tokens_for_pipeline,
    fused_ep_moe_memory,
    fused_moe_counter_reserve_memory,
)
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype

from .model_config import Glm5_1Config


class Glm5_1MemoryPlanner(PagedMemoryPlanner):
    """The paged planner, plus what EP init allocates for the fused MoE: the
    counter reserve, which this model's EP config always gets, and the
    one-launch fused MoE workspace when ``MODULAR_EP_FUSED_MOE=1`` asks for
    it. The EP communication buffers and the rest of the counter buffers stay
    unplanned here, as on main."""

    def estimate_activation_memory(
        self,
        pipeline_config: Any,
        huggingface_config: Any,
    ) -> int:
        """Estimates activation memory beyond model weights.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated activation memory in bytes.
        """
        memory = super().estimate_activation_memory(
            pipeline_config, huggingface_config
        )
        if pipeline_config.runtime.ep_size <= 1:
            return memory
        encoding = _select_quantization_encoding(
            pipeline_config.model, Glm5_1Config.DEFAULT_ENCODING
        )
        return (
            memory
            + fused_moe_counter_reserve_memory(
                pipeline_config, huggingface_config
            )
            + fused_ep_moe_memory(
                pipeline_config,
                huggingface_config,
                supported_encoding_dtype(encoding),
                _ep_max_rank_send_tokens_for_pipeline(pipeline_config),
            )
        )
