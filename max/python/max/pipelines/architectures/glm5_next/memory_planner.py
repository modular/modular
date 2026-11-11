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

Plans the MLA up-projection buffers and the expert-parallel buffers the way the
ModuleV3 MoE uses them, plus GLM-5.3-Flash's own costs: the KDA state pools,
which cap concurrency below roughly 12K context, the 4x mHC residual, the
vision tower and, on gfx950, the ``kda_chunk`` prefill workspaces.

The vision tower runs on the first device, which also holds its share of the
evenly split KV cache, so that device bounds the cache. Each device is
charged a tower and its activations, as if each held one, so the first
device's fit.
"""

from __future__ import annotations

import logging
import math

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.nn._graph_ops import nvfp4_mla_prefill_convert_bytes
from max.nn._graph_ops.state_space.kimi_delta import (
    amd_kda_chunk_workspace_bytes,
)
from max.nn.ep_config import (
    EPConfig,
    _ep_fused_moe_requested,
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.nn.legacy.comm.ep.ep_manager import (
    _bound_fused_moe_workspace,
    _ep_sync_counter_bytes,
)
from max.pipelines.kv_cache import (
    cache_dtype_for_encoding,
    is_fp4_kv_cache_format,
)
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.support.human_readable_formatter import to_human_readable_bytes
from max.tensor import default_dtype
from transformers import AutoConfig
from typing_extensions import override

from .model_config import Glm5NextConfig
from .vision.image_processor import Glm5NextImageProcessor
from .vision.tower import Glm5NextVisionTransformer

__all__ = ["Glm5NextMemoryPlanner"]

logger = logging.getLogger("max.pipelines")

# Peak live activation per patch inside the vision tower, in units of the
# tower's hidden width in bytes. A block holds the bfloat16 gate and up
# projections at 4x the width, the float32 q and k the rotary embedding
# upcasts, and the residual, about 40 widths in all; rounded up.
_VISION_PEAK_HIDDEN_WIDTHS_PER_PATCH = 48


def _vision_tower_bytes(config: Glm5NextConfig) -> int:
    """Bytes of one tower, read off the module the model builds."""
    if config.vision_config is None:
        return 0
    with F.lazy(), default_dtype(config.vision_config.dtype):
        tower = Glm5NextVisionTransformer(config.vision_config)
    return sum(
        math.prod(int(dim) for dim in weight.shape) * weight.dtype.size_in_bytes
        for _, weight in tower.parameters
    )


def _ep_max_rank_send_tokens(
    pipeline_config: PipelineConfig,
) -> int:
    """Upper bound on EP dispatch tokens held on one rank for this pipeline."""
    return calculate_ep_max_tokens_per_rank(
        max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
        ep_size=pipeline_config.runtime.ep_size,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        use_allreduce=pipeline_config.runtime.ep_use_allreduce,
    )


def fused_moe_counter_reserve_memory(
    pipeline_config: PipelineConfig, huggingface_config: AutoConfig
) -> int:
    """Bytes of the fused MoE counters' reserve on all of this node's devices.

    An EP config that gives the experts' FFN width, as the DeepSeek-V3.2 model
    code does, ends each of a device's EP sync-counter buffers in a fixed
    reserve for the fused MoE's counters, whether or not the fused MoE runs.
    The rest of those buffers is main's and stays unplanned. This planner also
    serves models that give no width and so allocate no reserve; it cannot
    tell them apart, so it plans the reserve for them too.

    Args:
        pipeline_config: Pipeline configuration.
        huggingface_config: HuggingFace model configuration.

    Returns:
        The reserve bytes over all devices of the node.
    """
    n_gpus_per_node = len(pipeline_config.model.device_specs)
    n_experts = huggingface_config.n_routed_experts
    use_allreduce = pipeline_config.runtime.ep_use_allreduce
    with_reserve = _ep_sync_counter_bytes(
        n_experts,
        n_gpus_per_node,
        use_allreduce,
        huggingface_config.moe_intermediate_size,
    )
    without_reserve = _ep_sync_counter_bytes(
        n_experts, n_gpus_per_node, use_allreduce, 0
    )
    return n_gpus_per_node * (with_reserve - without_reserve)


def fused_ep_moe_memory(
    pipeline_config: PipelineConfig,
    huggingface_config: AutoConfig,
    dispatch_dtype: DType,
    max_tokens_per_rank: int,
) -> int:
    """Bytes the one-launch fused EP MoE can allocate on this node's devices.

    Nothing is allocated unless ``MODULAR_EP_FUSED_MOE=1`` asks EP init for
    it, then the backend's bound for the EP geometry and dispatch element
    type.

    The plan is made before the checkpoint is parsed, so the dispatch format,
    its global scales and the shared-expert rows are not known yet. The
    backend answers the most EP init can allocate for any of them, and a
    model it refuses gets nothing, so the plan can be high but never short.

    Args:
        pipeline_config: Pipeline configuration.
        huggingface_config: HuggingFace model configuration.
        dispatch_dtype: The target's EP dispatch element type. EP init sets the
            fused MoE up only on the target's EP config: when an MTP draft
            makes it allocate the shared buffers from a bfloat16 copy, the
            backend refuses that copy, so a bound for the target covers both.
        max_tokens_per_rank: The EP per-rank token capacity.

    Returns:
        The workspace bytes over all devices of the node.
    """
    if not _ep_fused_moe_requested():
        return 0
    n_gpus_per_node = len(pipeline_config.model.device_specs)
    arena, refusal = _bound_fused_moe_workspace(
        EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=huggingface_config.hidden_size,
            top_k=huggingface_config.num_experts_per_tok,
            n_experts=huggingface_config.n_routed_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            n_gpus_per_node=n_gpus_per_node,
            n_nodes=pipeline_config.runtime.ep_size // n_gpus_per_node,
            moe_dim=huggingface_config.moe_intermediate_size,
            use_allreduce=pipeline_config.runtime.ep_use_allreduce,
        ),
        dispatch_dtype,
    )
    if refusal:
        logger.info(
            "Not planning a fused EP MoE workspace: the backend does not"
            " serve this configuration (%s)",
            refusal,
        )
        return 0
    logger.info(
        "Planning the fused EP MoE workspace: "
        f"{to_human_readable_bytes(arena)} per device"
    )
    return n_gpus_per_node * arena


def nvfp4_mla_prefill_convert_memory(
    pipeline_config: PipelineConfig, huggingface_config: AutoConfig
) -> int:
    """Bytes the NVFP4 MLA prefill's FP8 conversion can allocate on this node.

    Over an NVFP4 cache, a prefill step may convert its context to FP8 once
    and allocate the copy inside the attention op, outside the graph's
    planned buffers. Every device converts its own whole latent cache, under
    TP as under DP, so each reserves the bound.

    Args:
        pipeline_config: Pipeline configuration.
        huggingface_config: HuggingFace model configuration.

    Returns:
        The bytes over all devices of the node, or 0 for another cache
        format, a decode-only worker, or the conversion turned off.
    """
    if pipeline_config.runtime.pipeline_role == "decode_only" or (
        not is_fp4_kv_cache_format(
            pipeline_config.model.kv_cache.kv_cache_format
        )
    ):
        return 0
    runtime = pipeline_config.runtime
    max_query_tokens = (
        runtime.max_batch_input_tokens
        if runtime.enable_chunked_prefill
        else pipeline_config.model.max_length or 0
    )
    per_device = nvfp4_mla_prefill_convert_bytes(
        max_query_tokens,
        top_k=huggingface_config.index_topk,
        row_dim=huggingface_config.kv_lora_rank
        + huggingface_config.qk_rope_head_dim,
        page_size=pipeline_config.model.kv_cache.kv_cache_page_size,
    )
    memory = len(pipeline_config.model.device_specs) * per_device
    logger.info(
        "Estimated NVFP4 MLA prefill conversion scratch: "
        f"{to_human_readable_bytes(memory)}"
    )
    return memory


class Glm5NextMemoryPlanner(PagedMemoryPlanner):
    """Memory planner for GLM-5.3-Flash on ModuleV3.

    Accounts for MLA up-projection buffers, expert-parallel routing memory,
    EP SHMEM communication buffers, the mHC residual, the vision tower and
    the ``kda_chunk`` workspaces. The KDA state pools are cache leaves,
    planned with the cache.
    """

    _always_signal_buffers = True

    _inferred_max_batch_size: int | None = None
    """The batch size :meth:`infer_max_batch_size` chose, which planning does
    not write back to ``runtime.max_batch_size`` before it asks
    :meth:`estimate_activation_memory`."""

    @override
    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Estimates weight memory.

        Adjusts the raw weight file size to account for expert-parallel
        sharding, shared expert replication, and DP attention duplication.
        The checkpoint carries the one vision tower the first device holds;
        every other device adds one.

        Args:
            pipeline_config: Pipeline configuration.

        Returns:
            Estimated weight memory in bytes.
        """
        model_config = pipeline_config.model
        weights_size = model_config.weights_size()
        n_gpus_per_node = len(model_config.device_specs)

        encoding = _select_quantization_encoding(
            pipeline_config.model, Glm5NextConfig.DEFAULT_ENCODING
        )

        config = model_config.huggingface_config
        assert config is not None
        n_sparse_layers = (
            config.num_hidden_layers - config.first_k_dense_replace
        )
        n_mtp_layers = config.num_nextn_predict_layers

        # Note: All the following calculations are not exact, but they are
        # better than directly using the raw weights size.

        # First, Calculate the lm_head/embed_tokens size.
        # These are always in BF16.
        lm_head_size = (
            config.vocab_size
            * config.hidden_size
            * DType.bfloat16.size_in_bytes
        )
        embed_tokens_size = lm_head_size

        # Subtract the lm_head/embed_tokens size from the weights size
        weights_size -= lm_head_size + embed_tokens_size
        weights_size -= (lm_head_size + embed_tokens_size) * n_mtp_layers

        # We don't use the MTP module for now, so subtract the MTP attn/moe size.
        # Estimate the MTP module size by assuming the MTP layer is of the same
        # size as a sparse model layer.
        weights_size = int(
            weights_size * n_sparse_layers / (n_sparse_layers + n_mtp_layers)
        )

        # Calculate the routing experts and the shared experts size.
        expert_elems = (
            config.moe_intermediate_size * config.hidden_size * 3
        )  # A factor of 3 accounts for the gate/up/down proj weights.
        expert_size = (
            expert_elems * supported_encoding_dtype(encoding).size_in_bytes
        )
        routing_experts_size = (
            n_sparse_layers * config.n_routed_experts * expert_size
        )
        shared_experts_size = (
            n_sparse_layers * config.n_shared_experts * expert_size
        )

        # Estimate the size of the attention weights.
        attn_weights_size = (
            weights_size - routing_experts_size - shared_experts_size
        )

        # If we use DP attention, attention weights are duplicated on each DP rank.
        total_size = attn_weights_size * model_config.data_parallel_degree

        # The shared experts are duplicated on each device.
        total_size += shared_experts_size * n_gpus_per_node

        ep_size = max(pipeline_config.runtime.ep_size, 1)
        if ep_size == 1:
            total_size += routing_experts_size
        else:
            # we don't support mixing EP and TP strategies yet.
            # ep_size must be equal to n_gpus_per_node * n_nodes
            assert ep_size % n_gpus_per_node == 0
            n_nodes = ep_size // n_gpus_per_node
            total_size += routing_experts_size // n_nodes

        # Add back the lm_head/embed_tokens size, they will never be duplicated.
        total_size += lm_head_size + embed_tokens_size

        config = self._config
        assert isinstance(config, Glm5NextConfig)
        replicas = len(config.devices) - 1
        return total_size + replicas * _vision_tower_bytes(config)

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
        """Estimates activation memory.

        Accounts for MLA up-projection buffers during prefill, expert-parallel
        routing buffers, EP SHMEM communication buffers, the FP8 conversion
        scratch of sparse prefill over an NVFP4 cache, the mHC residual and
        the ``kda_chunk`` workspaces.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated activation memory in bytes.
        """
        encoding = _select_quantization_encoding(
            pipeline_config.model, Glm5NextConfig.DEFAULT_ENCODING
        )
        mla_activation_memory: int = 0
        moe_activation_memory: int = 0
        ep_buffer_memory = 0

        # During the prefill, we need to up-project all the KV cache for
        # current requests. The total context length of requests in a batch
        # should be limited by max_batch_total_tokens.
        if pipeline_config.runtime.pipeline_role != "decode_only":
            max_kv_length: int = 0

            if pipeline_config.runtime.max_batch_total_tokens is None:
                # If max_batch_total_tokens is not set, we use max_length.
                max_kv_length = pipeline_config.model.max_length or 0
            else:
                max_kv_length = pipeline_config.runtime.max_batch_total_tokens

            cache_dtype = cache_dtype_for_encoding(
                encoding,
                pipeline_config.model.kv_cache.kv_cache_format,
            )
            mla_activation_memory += (
                pipeline_config.model.data_parallel_degree
                * 2  # 2 for K and V
                * max_kv_length
                * huggingface_config.num_attention_heads
                * huggingface_config.qk_nope_head_dim
                * cache_dtype.size_in_bytes
            )

        # Estimate buffer and activation memory during Expert Parallel MoE.
        if pipeline_config.runtime.ep_size > 1:
            n_gpus_per_node = len(pipeline_config.model.device_specs)

            ep_max_rank_send_tokens = _ep_max_rank_send_tokens(pipeline_config)

            # Calculate the maximum number of tokens a rank may receive during
            # all-to-all routing. Each token selects top_k experts, and in the
            # worst case all selections land on one rank.
            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                huggingface_config.n_routed_experts,
                pipeline_config.runtime.ep_size
                * huggingface_config.num_experts_per_tok,
            )

            if pipeline_config.runtime.ep_use_allreduce:
                max_recv_tokens_per_rank = (
                    pipeline_config.runtime.max_batch_input_tokens
                    * min(
                        huggingface_config.n_routed_experts // n_gpus_per_node,
                        huggingface_config.num_experts_per_tok,
                    )
                )

            # The maximal activation memory usage happens at the second
            # grouped_matmul in the MoE layer. The input for that matmul would
            # of shape [max_recv_tokens_per_rank, moe_intermediate_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.moe_intermediate_size
                * supported_encoding_dtype(encoding).size_in_bytes
            )

            # The output is of shape [max_recv_tokens_per_rank, hidden_size]:
            # the ModuleV3 expert-parallel MoE always materializes it before
            # the combine.
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.hidden_size
                * DType.bfloat16.size_in_bytes  # output is always bfloat16.
            )

            # Adding 256MB per GPU to account for misc items (e.g. FP8 scalars).
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            # EP SHMEM communication buffers are persistent (allocated once at
            # model init, not freed between layers).
            n_nodes = pipeline_config.runtime.ep_size // n_gpus_per_node

            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=huggingface_config.hidden_size,
                dispatch_dtype=supported_encoding_dtype(encoding),
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=huggingface_config.n_routed_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=huggingface_config.num_experts_per_tok,
                use_allreduce=pipeline_config.runtime.ep_use_allreduce,
            )
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node

            reserve_memory = fused_moe_counter_reserve_memory(
                pipeline_config, huggingface_config
            )
            logger.info(
                "Estimated fused MoE counter reserve: "
                f"{to_human_readable_bytes(reserve_memory)}"
            )
            ep_buffer_memory += reserve_memory
            ep_buffer_memory += fused_ep_moe_memory(
                pipeline_config,
                huggingface_config,
                supported_encoding_dtype(encoding),
                ep_max_rank_send_tokens,
            )

            logger.info(
                "Estimated EP SHMEM buffer memory: "
                f"{to_human_readable_bytes(ep_buffer_memory)}"
            )

        # We only need to consider the maximum of the MLA and MoE activation
        # memories, because the MLA and MoE layers are executed sequentially.
        activation_memory = max(mla_activation_memory, moe_activation_memory)
        activation_memory += ep_buffer_memory
        activation_memory += nvfp4_mla_prefill_convert_memory(
            pipeline_config, huggingface_config
        )

        # The state pools are cache leaves and already paid for; the residual
        # streams are a real activation, replicated over the whole batch, and
        # so are the chunk workspaces gfx950 prefill allocates per call. The
        # vision tower's activations overlap the decoder's under the overlap
        # scheduler, so they are added rather than shared.
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        runtime = pipeline_config.runtime
        activation_memory += (
            runtime.max_batch_input_tokens * config.activation_bytes_per_token()
        )
        activation_memory += amd_kda_chunk_workspace_bytes(
            num_heads=config.linear_num_heads,
            head_dim=config.linear_head_dim,
            lower_bound=config.linear_lower_bound,
            state_dtype=config.state_pool_dtype,
            decode_only=runtime.pipeline_role == "decode_only",
            max_batch_input_tokens=runtime.max_batch_input_tokens,
            enable_chunked_prefill=runtime.enable_chunked_prefill,
            max_batch_size=(
                runtime.max_batch_size or self._inferred_max_batch_size
            ),
            max_length=pipeline_config.model.max_length,
        )
        activation_memory += self._vision_activation_bytes(pipeline_config)

        if activation_memory != 0:
            logger.info(
                f"Estimated activation memory: {to_human_readable_bytes(activation_memory)}"
            )

        return activation_memory

    def _vision_activation_bytes(self, pipeline_config: PipelineConfig) -> int:
        """The tower's peak activations, charged to every device.

        The model chunks an encode at ``max_batch_input_tokens`` output rows
        but never splits an image, so one encode is at most the larger of
        that and the largest image.
        """
        config = self._config
        assert isinstance(config, Glm5NextConfig)
        vision_config = config.vision_config
        if vision_config is None:
            return 0
        rows = max(
            pipeline_config.runtime.max_batch_input_tokens,
            Glm5NextImageProcessor().max_image_tokens,
        )
        patches = rows * vision_config.spatial_merge_size**2
        return (
            len(config.devices)
            * patches
            * vision_config.hidden_size
            * _VISION_PEAK_HIDDEN_WIDTHS_PER_PATCH
        )
