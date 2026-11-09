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
"""Memory planner for the Beam ModuleV3 architecture.

Beam serves more than one device only with its routed experts expert
parallel over all of them, so every multi-device run reserves the
double-buffered dispatch/combine buffers and the MoE's received-row
activations, as MiMo-V2's planner does for the same kernels.
"""

from __future__ import annotations

import logging

from max.dtype import DType
from max.nn.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

from .model_config import BeamConfig

logger = logging.getLogger(__name__)


class BeamMemoryPlanner(PagedMemoryPlanner):
    """Reserves the EP communication buffers and their peak activations."""

    def estimate_activation_memory(
        self, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Estimates the activation memory one step needs."""
        del huggingface_config
        ep_size = pipeline_config.runtime.ep_size
        if ep_size <= 1:
            return 0
        n_devices = len(pipeline_config.model.device_specs)
        config = self._config
        assert isinstance(config, BeamConfig)
        # BeamConfig.expert_parallel_config dispatches in the weight dtype:
        # packed NVFP4 on a quantized checkpoint, bfloat16 otherwise.
        dispatch_dtype = config.dtype
        top_k = config.num_experts_per_tok
        max_tokens_per_rank = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=1,
        )
        # Worst-case tokens received per rank during the all-to-all routing.
        max_recv_tokens_per_rank = max_tokens_per_rank * min(
            config.num_experts, ep_size * top_k
        )

        # Peak MoE activation: the second grouped matmul's input
        # [max_recv, moe_dim] plus its [max_recv, hidden] bf16 output.
        moe_activation_memory = (
            max_recv_tokens_per_rank
            * config.moe_intermediate_size
            * dispatch_dtype.size_in_bytes
            + max_recv_tokens_per_rank
            * config.hidden_size
            * DType.bfloat16.size_in_bytes
        )
        moe_activation_memory *= n_devices

        per_device_ep_memory = estimate_ep_memory_usage(
            hidden_size=config.hidden_size,
            dispatch_dtype=dispatch_dtype,
            combine_dtype=DType.bfloat16,
            max_tokens_per_rank=max_tokens_per_rank,
            n_experts=config.num_experts,
            n_nodes=1,
            n_gpus_per_node=n_devices,
            top_k=top_k,
        )
        # EPCommInitializer double-buffers (NUM_GROUPS=2) the dispatch and
        # combine buffers, one pair per device.
        ep_buffer_memory = per_device_ep_memory * n_devices * 2

        activation_memory = moe_activation_memory + ep_buffer_memory
        logger.info(
            "Estimated activation memory: %s (ep_buffers=%s, "
            "moe_activation=%s)",
            to_human_readable_bytes(activation_memory),
            to_human_readable_bytes(ep_buffer_memory),
            to_human_readable_bytes(moe_activation_memory),
        )
        return activation_memory
