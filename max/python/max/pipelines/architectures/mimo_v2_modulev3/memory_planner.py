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

"""Memory planner for the ModuleV3 MiMo-V2 architecture.

The graph-API architecture's planner reserves nothing: its MoE is
tensor-parallel, and the residual's own activations fit the planner's
fallback. This package's expert-parallel build allocates double-buffered
dispatch/combine buffers per device beside the MoE's received-row
activations, so the EP shapes reserve them, as MiniMax-M2's planner does
for the same kernels.
"""

from __future__ import annotations

import logging

from max.dtype import DType
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class MiMoV2ModuleV3MemoryPlanner(PagedMemoryPlanner):
    """Reserves the EP communication buffers and their peak activations."""

    _always_signal_buffers = True

    def ep_dispatch_row_budget(self, pipeline_config: PipelineConfig) -> int:
        """Returns the rows one forward step carries; sizes the EP buffers.

        The scheduler's ``max_batch_input_tokens`` bounds a step's active
        rows -- chunked prefill makes it hard -- so the base build sizes
        its dispatch buffers to it. A build whose steps carry rows the
        budget never charges overrides this with its own bound.

        Args:
            pipeline_config: The pipeline whose scheduler budget bounds
                the step's rows.
        """
        return pipeline_config.runtime.max_batch_input_tokens

    def estimate_activation_memory(
        self, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        ep_size = pipeline_config.runtime.ep_size
        if ep_size <= 1:
            # The tensor-parallel build owns no EP buffers; the inherited
            # planner's fallback covers its activations.
            return 0

        n_gpus_per_node = len(pipeline_config.model.device_specs)
        num_experts = huggingface_config.n_routed_experts
        top_k = huggingface_config.num_experts_per_tok
        hidden_size = huggingface_config.hidden_size
        moe_dim = huggingface_config.moe_intermediate_size
        # The dispatch quantizes tokens to the experts' MXFP8 activation
        # format in transit (MXFP8_DISPATCH_QUANT_CONFIG): E4M3 data.
        ep_dispatch_dtype = DType.float8_e4m3fn

        ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.ep_dispatch_row_budget(pipeline_config),
            ep_size=ep_size,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            use_allreduce=pipeline_config.runtime.ep_use_allreduce,
        )
        # Worst-case tokens received per rank during the all-to-all routing.
        max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
            num_experts,
            ep_size * top_k,
        )

        # Peak MoE activation: the second grouped matmul's input
        # [max_recv, moe_dim] plus its [max_recv, hidden] bf16 output.
        moe_activation_memory = (
            max_recv_tokens_per_rank * moe_dim * ep_dispatch_dtype.size_in_bytes
            + max_recv_tokens_per_rank
            * hidden_size
            * DType.bfloat16.size_in_bytes
        )
        moe_activation_memory *= n_gpus_per_node

        n_nodes = max(ep_size // n_gpus_per_node, 1)
        per_device_ep_memory = estimate_ep_memory_usage(
            hidden_size=hidden_size,
            dispatch_dtype=ep_dispatch_dtype,
            combine_dtype=DType.bfloat16,
            max_tokens_per_rank=ep_max_rank_send_tokens,
            n_experts=num_experts,
            n_nodes=n_nodes,
            n_gpus_per_node=n_gpus_per_node,
            top_k=top_k,
            use_allreduce=pipeline_config.runtime.ep_use_allreduce,
        )
        # EPCommInitializer double-buffers (NUM_GROUPS=2) the dispatch and
        # combine buffers, one pair per device.
        ep_buffer_memory = per_device_ep_memory * n_gpus_per_node * 2

        activation_memory = moe_activation_memory + ep_buffer_memory
        logger.info(
            "Estimated activation memory: %s (ep_buffers=%s, "
            "moe_activation=%s)",
            to_human_readable_bytes(activation_memory),
            to_human_readable_bytes(ep_buffer_memory),
            to_human_readable_bytes(moe_activation_memory),
        )
        return activation_memory
