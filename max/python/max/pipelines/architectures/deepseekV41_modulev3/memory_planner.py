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
from typing import Any, NamedTuple

from max.graph import DeviceRef
from max.nn.legacy.comm.ep import EPConfig
from max.nn.legacy.quant_config import fp4_packed_k
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.support.human_readable_formatter import to_human_readable_bytes

from ..deepseekV41.memory_planner import (
    _SCALE_TILE_BYTES,
    _SCALE_TILE_ROWS,
    _SCALE_TILE_SCALES,
    _mm_pages,
)
from .inputs import v41_config
from .interface import V41Config, extra_inputs
from .layers.index_ops import INDEXER_SCORES_BUDGET_BYTES
from .model import build_ep_config
from .model_config import (
    DeepseekV41Config,
    dspark_expert_counts,
    num_dspark_stages,
)

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


def _dp_ep_config(pipeline_config: Any, text: Any) -> EPConfig | None:
    """The EP config whose buffers a DP run allocates, or ``None``.

    Only the multi-GPU shape the trunk wires (``dp == ep_size == the
    device count``) carries EP buffers; every other shape the model
    refuses before load, so the planner leaves them alone rather than
    fronting its own error.
    """
    ep_size = pipeline_config.runtime.ep_size
    n_devices = len(pipeline_config.model.device_specs)
    if (
        ep_size <= 1
        or ep_size != n_devices
        or pipeline_config.model.data_parallel_degree != n_devices
    ):
        return None
    return build_ep_config(pipeline_config, text, n_devices)


def _ep_reserve_bytes(pipeline_config: Any, text: Any) -> tuple[int, int]:
    """(comm buffers, receive pool) per device under expert parallelism.

    The comm buffers are what EP init allocates beside the KV pool. The
    receive pool is the receive-side rows the grouped EP chain holds
    live: the dispatched tokens with their per-block scales and source
    map, the packed down input the fused gate/up+SwiGLU kernel writes
    (its epilogue runs the activation and the NVFP4 re-quant, so no bf16
    intermediate exists), and the down projection's bf16 output -- all
    sized for the most rows a device can receive.

    Returns:
        ``(0, 0)`` when the run carries no EP buffers.
    """
    config = _dp_ep_config(pipeline_config, text)
    if config is None:
        return 0, 0
    rows = int(config.get_max_recv_tokens())
    hidden = int(config.hidden_size)
    n_ranks = int(config.n_gpus_per_node) * int(config.n_nodes)
    n_local_experts = int(config.n_experts) // n_ranks
    inter = int(text.moe_intermediate_size)
    quant_config = config.dispatch_quant_config
    assert quant_config is not None, "the trunk's dispatch is NVFP4-quantized"
    assert quant_config.input_scale.block_size is not None
    block = quant_config.input_scale.block_size[1]
    tile_rows = -(
        -(rows + n_local_experts * _SCALE_TILE_ROWS) // _SCALE_TILE_ROWS
    )

    def scale_bytes(width: int) -> int:
        tile_cols = -(-width // (block * _SCALE_TILE_SCALES))
        return tile_rows * tile_cols * _SCALE_TILE_BYTES

    pool = (
        # The dispatched rows, NVFP4-packed with their block scales.
        rows * fp4_packed_k(hidden, quant_config)
        + scale_bytes(hidden)
        # The source map, one [rows, 2] int32 entry per row.
        + rows * 2 * 4
        # The fused gate/up+SwiGLU kernel's output: the down projection's
        # NVFP4-packed input rows with their block scales.
        + rows * fp4_packed_k(inter, quant_config)
        + scale_bytes(inter)
        # The down projection's bf16 output.
        + rows * hidden * 2
    )
    return config.estimate_memory_usage(), _mm_pages(pool)


class DeepseekV41MemoryPlanner(PagedMemoryPlanner):
    """Reserves the forward's transient scratch, and counts only HBM weights.

    The DP and EP corrections split by consumer. The export-time plan
    (``memory_estimation.plan_from_sizes``) charges these estimates
    against ``device_memory_utilization`` of free memory read before
    anything loads, and the runtime owns the replication and the EP
    dispatch's allocations: the weights count once (the EP shards and
    the vocab-parallel embedding and head spread the rest over the
    ranks -- the engine's own KV sizing reads device free with the
    weights already resident, so the plan never holds the replicas
    back), and the EP comm buffers and receive pool stay out of that
    plan because the engine books them itself, through the sidecar
    figure below. The sidecar is the exception: the engine sizes and
    allocates the unified KV arena before EP init allocates the
    dispatch heaps, so the EP reserve travels there (see
    :meth:`forward_scratch_bytes_per_device`). The frozen DP4/EP4
    export plan is pinned in test_memory_planner.py.
    """

    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Weights on device: the checkpoint minus the host-resident Engram tables.

        A DSpark pairing loads the drafter beside the trunk, so it adds what
        :func:`dspark_extra_weight_bytes` says the checkpoint size leaves
        out; without it the plan sizes the KV pool into memory the
        drafter's replicas already hold.
        """
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
        size = max(size - engram_table_bytes(config) + expansion, 0)
        speculative = pipeline_config.speculative
        if (
            speculative is not None
            and speculative.speculative_method == "dspark"
        ):
            size += dspark_extra_weight_bytes(
                text,
                len(pipeline_config.model.device_specs),
                pipeline_config.model.data_parallel_degree > 1,
            )
        return size

    def estimate_activation_memory(
        self, pipeline_config: Any, huggingface_config: Any
    ) -> int:
        """Estimates the arena and indexer scores of the largest forward.

        Only transient scratch. The KV leaves are sized afterwards out of what
        this leaves free (``memory_estimation.plan_from_sizes``), so counting
        them here would subtract them twice. The EP dispatch's comm buffers
        and receive pool are not forward scratch either: they allocate after
        the plan has priced the KV budget, and the engine books them through
        :meth:`forward_scratch_bytes_per_device` instead.

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
        """Per-device bytes the engine's KV sizing must hold back.

        The forward's arena and indexer-score estimate
        :meth:`estimate_activation_memory` sums over every device, for one
        device -- plus the EP dispatch's comm buffers and receive pool. The
        exporter records this figure in the input-spec sidecar and the
        engine's KV sizing holds it back per device, and the boot order is
        why the EP terms ride here and nowhere else: the engine sizes and
        allocates the unified KV arena before EP init runs its handshake
        and allocates the dispatch heaps (the C4b boot's serve log shows
        ``unified KV arena allocated`` at 10:21:57.545Z preceding ``running
        EP init handshake`` at 10:21:57.699Z), so at KV-sizing time those
        buffers are not yet resident -- without this reserve an unpinned
        boot sizes the pool into their room, exactly as it would into the
        forward scratch's, which travels here for the same reason.
        """
        runtime = pipeline_config.runtime
        max_length = pipeline_config.model.max_length
        config = v41_config(pipeline_config.model.huggingface_config)
        text = DeepseekV41Config.get_text_config(
            pipeline_config.model.huggingface_config
        )
        rows = forward_rows(runtime, max_length)
        ep_buffers, ep_pool = _ep_reserve_bytes(pipeline_config, text)
        return (
            prefill_arena_bytes(config, rows)
            + indexer_score_bytes(
                config,
                rows,
                max_length,
                index_source_layer_ids=text.index_source_layer_ids,
                candidate_topk_blocks=text.candidate_topk_blocks,
                candidate_block_size=text.candidate_block_size,
            )
            + ep_buffers
            + ep_pool
        )


class DSparkParameter(NamedTuple):
    """One drafter parameter's bytes, as stored and as resident."""

    name: str
    stored: int
    """Bytes in the checkpoint."""
    resident: int
    """Bytes of one whole copy on device."""
    split: bool
    """Whether tensor parallelism splits it over the mesh rather than
    replicating it."""


def _mxfp8_bytes(n: int, k: int) -> int:
    """An MXFP8 ``[n, k]`` projection: e4m3 codes, one E8M0 per 32x32 block."""
    return n * k + -(-n // 32) * (k // 32)


def dspark_parameters(
    text: Any,
) -> tuple[list[DSparkParameter], list[DSparkParameter]]:
    """The drafter's parameters: ``(per stage, once)``.

    Layouts follow the checkpoint and ``weight_adapters``: dense MXFP8
    projections keep compact 32x32-block scales, the four
    ``_DSPARK_HOST_DEQUANT`` projections load as bf16, and the routed
    experts are MXFP4 with an E8M0 per 32 elements. Under tensor
    parallelism ``layers/dspark.py`` splits ``attn_sink``, ``wq_b``,
    ``wo_a`` and ``wo_b`` and ``layers/moe.py`` the shared expert; the
    rest replicates, the routed experts included
    (``layers/dspark_moe.py``). The tensors the loader drops
    (``gate.bias_vl``, the confidence head by default) are left out.
    """
    hidden = int(text.hidden_size)
    inter = int(text.moe_intermediate_size)
    heads = int(text.num_attention_heads)
    head_dim = int(text.head_dim)
    q_lora = int(text.q_lora_rank)
    o_groups = int(text.o_groups)
    o_rows = o_groups * int(text.o_lora_rank)
    hc = int(text.hc_mult)
    mix = (2 + hc) * hc
    vocab = int(text.vocab_size)
    markov = int(getattr(text, "dspark_markov_rank", 256))
    n_taps = len(text.dspark_target_layer_ids)
    n_experts, _ = dspark_expert_counts(text)

    def plain(name: str, nbytes: int, split: bool = False) -> DSparkParameter:
        return DSparkParameter(name, nbytes, nbytes, split)

    def mxfp8(name: str, n: int, k: int, split: bool) -> DSparkParameter:
        return plain(name, _mxfp8_bytes(n, k), split)

    def dequantized(name: str, n: int, k: int, split: bool) -> DSparkParameter:
        return DSparkParameter(name, _mxfp8_bytes(n, k), 2 * n * k, split)

    stage = [
        plain("attn.attn_sink", heads * 4, split=True),
        plain("attn.q_norm", q_lora * 2),
        plain("attn.kv_norm", head_dim * 2),
        dequantized("attn.wq_a", q_lora, hidden, split=False),
        dequantized("attn.wq_b", heads * head_dim, q_lora, split=True),
        dequantized("attn.wkv", head_dim, hidden, split=False),
        dequantized(
            "attn.wo_a", o_rows, heads * head_dim // o_groups, split=True
        ),
        mxfp8("attn.wo_b", hidden, o_rows, split=True),
        plain("attn_norm", hidden * 2),
        plain("ffn_norm", hidden * 2),
        plain("hc_mixes", 2 * (mix * hc * hidden + mix + 3) * 4),
        plain("ffn.gate", n_experts * (hidden * 2 + 4)),
        # gate_up [E, 2I, H/2] and down [E, H, I/2] uint8 with an E8M0 per
        # 32 elements: 3*I*H elements at 17/32 bytes apiece.
        plain("ffn.experts", n_experts * 3 * inter * hidden * 17 // 32),
        mxfp8("ffn.shared_experts.w1", inter, hidden, split=True),
        mxfp8("ffn.shared_experts.w2", hidden, inter, split=True),
        mxfp8("ffn.shared_experts.w3", inter, hidden, split=True),
    ]
    once = [
        mxfp8("main_proj", hidden, hidden * n_taps, split=False),
        plain("main_norm", hidden * 2),
        plain("norm", hidden * 2),
        plain("markov_head.embed", vocab * markov * 2),
        plain("markov_head.head", vocab * markov * 2),
    ]
    return stage, once


def dspark_extra_weight_bytes(
    text: Any, n_devices: int, data_parallel: bool
) -> int:
    """Device bytes the DSpark drafter holds beyond its checkpoint bytes.

    The checkpoint size counts each drafter tensor once, as stored. On
    device a split tensor holds one whole copy over the mesh and a
    replicated one ``n_devices``; under data parallelism every tensor
    replicates (TP1 per rank). Each parameter of :func:`dspark_parameters`
    adds its resident copies less its stored bytes.
    """

    def extra(parameters: list[DSparkParameter]) -> int:
        return sum(
            (1 if p.split and not data_parallel else n_devices) * p.resident
            - p.stored
            for p in parameters
        )

    stage, once = dspark_parameters(text)
    return num_dspark_stages(text) * extra(stage) + extra(once)
