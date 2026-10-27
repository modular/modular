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
"""Memory planner for DeepSeek-V4.1-Flash."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from max.graph import DeviceRef
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.support.human_readable_formatter import to_human_readable_bytes

from .inputs import v41_config
from .interface import V41Config, extra_inputs
from .layers.index_ops import INDEXER_SCORES_BUDGET_BYTES
from .model_config import DeepseekV41Config

logger = logging.getLogger("max.pipelines")

# Transient activation bytes one prefill token costs on each device, outside
# the indexer scores. Seeded from DeepSeek-V4's fit (deepseekV4/
# memory_planner.py), whose trunk has the same hidden size and four residual
# streams. The seed only plans memory, so it holds until a full-checkpoint
# run can be observed: refit from full-model allocation logs before
# trusting it to size a deployment.
_ARENA_BYTES_PER_TOKEN_BASE = 1_090_000
_ARENA_BYTES_PER_TOKEN_PER_LAYER = 450

# Live float32 [rows, num_candidates] tensors at the peak of one indexer layer:
# the kernel's scores, and the all-reduced or masked copy the top-k ranks.
# The fused SM100 path replaced them with the scorer's budget-capped
# transient score window, the candidate ops' window scratch, the kept block
# ids, and the shared RoPE table.
# The candidate source's window scratch holds one float32 block maximum per
# block of a window row, then one int32 live block count per row; the
# consumer's one flag byte per block is smaller, and the two never overlap.
_WINDOW_BYTES_PER_BLOCK = 4
_WINDOW_BYTES_PER_ROW = 4
_KEPT_ID_BYTES = 4
_ROPE_TABLE_BYTES_PER_POSITION = 256


def engram_table_bytes(config: V41Config) -> int:
    """Bytes of the Engram tables, which live in pinned host memory.

    Read off the table graph inputs, so it follows the ABI.
    """
    return sum(
        math.prod(t.shape.static_dims) * t.dtype.size_in_bytes
        for name, t in extra_inputs(config, [DeviceRef.GPU(0)])
        if name.startswith("engram_table_")
    )


def indexer_score_bytes(
    config: V41Config,
    rows: int,
    max_length: int,
    *,
    index_source_layer_ids: Sequence[int],
    candidate_topk_blocks: int,
    candidate_block_size: int,
) -> int:
    """Bytes of the indexer's transient scratch for ``rows`` query rows.

    The fused path's score matrix is budget-capped (512 MiB, the SM100
    scorer's row window, matching vLLM's ``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB``
    default); beside it the candidate ops score ``ceil(cap /
    candidate_block_size)`` blocks one window of rows at a time (the scratch
    fits a quarter of the score budget), the candidate source hands its
    consumers ``[rows, candidate_topk_blocks]`` int32 kept block ids, and
    every indexer layer builds one ``[max_length, 64]`` float32 RoPE table.
    """
    ratios = [config.ratio(l) for l in index_source_layer_ids]
    if not ratios or rows <= 0:
        return 0
    cap = math.ceil(max_length / min(ratios))
    nblocks = math.ceil(cap / candidate_block_size)
    window_rows = max(
        1,
        min(
            rows,
            INDEXER_SCORES_BUDGET_BYTES // (nblocks * candidate_block_size * 2),
        ),
    )
    window = window_rows * (
        nblocks * _WINDOW_BYTES_PER_BLOCK + _WINDOW_BYTES_PER_ROW
    )
    kept = rows * candidate_topk_blocks * _KEPT_ID_BYTES
    return (
        INDEXER_SCORES_BUDGET_BYTES
        + window
        + kept
        + max_length * _ROPE_TABLE_BYTES_PER_POSITION
    )


def prefill_arena_bytes(config: V41Config, rows: int) -> int:
    """Per-device transient activation bytes of one forward of ``rows`` tokens."""
    per_token = (
        _ARENA_BYTES_PER_TOKEN_BASE
        + _ARENA_BYTES_PER_TOKEN_PER_LAYER * config.num_hidden_layers
    )
    return rows * per_token


def forward_rows(runtime: Any, max_length: int) -> int:
    """Query rows of the widest forward ``runtime`` can schedule.

    Without chunked prefill the batch token budget is a soft limit: the
    scheduler admits a whole request that overshoots what is left of it
    (BUDGET_REACHED in token_budget.py), so a batch can reach the budget
    less one token plus one prompt of max_length.
    """
    return runtime.max_num_input_tokens or (
        runtime.max_batch_input_tokens - 1 + max_length
    )


class DeepseekV41MemoryPlanner(PagedMemoryPlanner):
    """Reserves the forward's transient scratch, and counts only HBM weights."""

    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Weights on device: the checkpoint minus the host-resident Engram tables."""
        config = v41_config(pipeline_config.model.huggingface_config)
        size = super().estimate_weights_size(pipeline_config)
        text = DeepseekV41Config.get_text_config(
            pipeline_config.model.huggingface_config
        )
        woa_elements = (
            int(text.o_groups)
            * int(text.o_lora_rank)
            * (int(text.num_attention_heads) // int(text.o_groups))
            * int(text.head_dim)
        )
        expansion = int(text.num_hidden_layers) * (
            woa_elements - woa_elements // (32 * 32)
        )
        return max(size - engram_table_bytes(config) + expansion, 0)

    def estimate_activation_memory(
        self, pipeline_config: Any, huggingface_config: Any
    ) -> int:
        """Estimates the arena and indexer scores of the largest forward.

        Only transient scratch. The KV leaves are sized afterwards out of what
        this leaves free (``memory_estimation.plan_from_sizes``), so counting
        them here would subtract them twice.

        Returns:
            Bytes summed over the devices, which each run their own forward.
        """
        runtime = pipeline_config.runtime
        max_length = pipeline_config.model.max_length
        config = v41_config(huggingface_config)
        rows = forward_rows(runtime, max_length)
        n_devices = len(pipeline_config.model.device_specs)
        logger.warning(
            "DeepSeek-V4.1-Flash activation estimates use unmeasured V4 "
            "coefficients (%d B/token + %d B/token/layer arena, %s score "
            "window); they are refit when the real forward lands.",
            _ARENA_BYTES_PER_TOKEN_BASE,
            _ARENA_BYTES_PER_TOKEN_PER_LAYER,
            to_human_readable_bytes(INDEXER_SCORES_BUDGET_BYTES),
        )
        arena = prefill_arena_bytes(config, rows)
        text = DeepseekV41Config.get_text_config(huggingface_config)
        scores = indexer_score_bytes(
            config,
            rows,
            max_length,
            index_source_layer_ids=text.index_source_layer_ids,
            candidate_topk_blocks=text.candidate_topk_blocks,
            candidate_block_size=text.candidate_block_size,
        )
        logger.info(
            "Estimated DeepSeek-V4.1-Flash scratch: %s arena + %s indexer "
            "scores per device x %d devices (%d rows, max_length %d)",
            to_human_readable_bytes(arena),
            to_human_readable_bytes(scores),
            n_devices,
            rows,
            max_length,
        )
        return (arena + scores) * n_devices

    def forward_scratch_bytes_per_device(self, pipeline_config: Any) -> int:
        """Per-device transient scratch of the widest forward this config runs.

        The same arena and indexer-score estimate
        :meth:`estimate_activation_memory` sums over every device, for one
        device: the exporter records this figure in the input-spec sidecar
        (``forward_scratch_bytes_per_device``) and the engine's KV sizing
        holds it back per device, so the first forward still finds its own
        room. Mach sizes the KV pool before any forward runs, so without it
        the pool spends the scratch's room and the forward OOMs.
        """
        runtime = pipeline_config.runtime
        max_length = pipeline_config.model.max_length
        config = v41_config(pipeline_config.model.huggingface_config)
        rows = forward_rows(runtime, max_length)
        return prefill_arena_bytes(config, rows) + indexer_score_bytes(
            config,
            rows,
            max_length,
            index_source_layer_ids=DeepseekV41Config.get_text_config(
                pipeline_config.model.huggingface_config
            ).index_source_layer_ids,
            candidate_topk_blocks=DeepseekV41Config.get_text_config(
                pipeline_config.model.huggingface_config
            ).candidate_topk_blocks,
            candidate_block_size=DeepseekV41Config.get_text_config(
                pipeline_config.model.huggingface_config
            ).candidate_block_size,
        )
