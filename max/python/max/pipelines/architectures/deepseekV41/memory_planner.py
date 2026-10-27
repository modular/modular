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
"""Memory planner for DeepSeek-V4.1-Flash.

Estimates device weight bytes from checkpoint headers and one forward step's
transient memory, so the device memory that remains can go to the paged KV
cache. KV storage is sized afterward from what remains, so the activation
estimate excludes it to avoid counting it twice.
"""

from __future__ import annotations

import functools
import json
import logging
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from max.dtype import DType
from max.nn.kv_cache import spec_decode_cache_slack
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.support.human_readable_formatter import to_human_readable_bytes

from . import kv_leaves
from .layers.csa2 import (
    DECODE_SPLIT_ROWS,
    HEAD_DIM,
    PREFILL_ROWS,
    decode_workspace_floats,
)
from .layers.indexer import (
    INDEXER_SCORES_BUDGET_BYTES,
    candidate_keep_blocks,
    candidate_window_rows,
)
from .layers.moe import router_reads_bf16_gate
from .model_config import (
    DeepseekV41Config,
    compressor_bf16_enabled,
    data_parallel_ep_config,
    forward_row_bound,
    head_bf16_enabled,
    max_images_per_request,
    nvfp4_sf_preinterleave_enabled,
    q_mxfp8_enabled,
    woa_mxfp8_enabled,
)
from .weight_adapters import (
    COMPRESSOR_WEIGHT_SUFFIXES,
    DSPARK_CHECKPOINT_PREFIX,
    FLOAT32_WEIGHT_SUFFIXES,
    FUSED_QUERY_PROJECTIONS,
    HEAD_WEIGHT,
    MXFP8_BLOCK_SIZE,
    VISION_NORM_WEIGHT,
    is_host_dequantized,
    is_skipped_checkpoint_tensor,
)

logger = logging.getLogger("max.pipelines")

# The forward graph's planned arena: ~2 MiB of activations per query row and
# a row-independent part. It barely depends on the tensor-parallel degree
# (MoE intermediates are ``T * top_k`` rows on every rank).
_ARENA_ROW_BYTES = 2 << 20
# Under expert-parallel dispatch the routed experts' intermediates live in the
# statically sized receive tensors (:func:`ep_receive_pool_bytes`), not per
# query row, so the per-row part is smaller: a DP4 8192-row prefill chunk
# peaks at ~10.4 GiB of transients outside those tensors.
_EP_ARENA_ROW_BYTES = 5 << 18  # 1.25 MiB
# Upper bound on the MoE's transient bytes per row (gathered input, routed
# top-k intermediates, float32 partial sum), for the other replicas' rows a
# data-parallel device also runs the MoE on.
_MOE_ROW_BYTES = 256 << 10

# Device 0 runs the LM head and sampling for every data-parallel replica, so
# its captured decode graphs hold extra bytes per ``[rows, vocab]`` entry: the
# head's bfloat16 logits and their float32 cast (6 B), plus sampling and
# draft verification working memory, measured at 4.33 B per entry and growing
# exactly with the rows but not broken down further (rounded up to 4.5 B).
_HEAD_LOGITS_BYTES = 6
_HEAD_SAMPLING_BYTES = 4.5
# NVFP4 block scales: one byte per 16 values, in 128-row x 4-column tiles.
_NVFP4_SCALE_TILE_ROWS = 128
_NVFP4_SCALE_TILE_BYTES = 512
_NVFP4_SCALE_TILE_COLS = 64
_ARENA_FIXED_BYTES = 5 << 28  # 1.25 GiB
# The candidate ops' window scratch holds one float32 block maximum (source)
# or one flag byte (consumer) per block of a window row; they never overlap.
_WINDOW_BYTES_PER_BLOCK = 4
# The source's window scratch also holds one int32 live block count per row.
_WINDOW_BYTES_PER_ROW = 4
# Kept block ids pass from the candidate source to its consumers as int32.
_KEPT_ID_BYTES = 4
# The device MemoryManager allocates whole 256 MiB pages.
_MM_PAGE_BYTES = 256 << 20


def _mm_pages(nbytes: int) -> int:
    """``nbytes`` rounded up to whole MemoryManager pages."""
    return ceil(nbytes / _MM_PAGE_BYTES) * _MM_PAGE_BYTES if nbytes > 0 else 0


def indexer_activation_bytes(
    *, live_rows: int, candidate_blocks: int, kept_blocks: int, block_size: int
) -> int:
    """Returns per-device bytes of the indexer's transient memory.

    The kernel's score window, the candidate ops' window scratch (at most a
    quarter of the score window) and the ``[T, kept_blocks]`` kept block ids;
    the scratch and score window each take whole MemoryManager pages.

    Args:
        live_rows: Query rows ``T`` in one forward.
        candidate_blocks: Candidate blocks per row, or 0 if no layer keeps
            candidate blocks.
        kept_blocks: Blocks each row keeps, or 0 if no layer keeps candidate
            blocks.
        block_size: Keys per candidate block.
    """
    if live_rows <= 0:
        return 0
    blocks = max(0, candidate_blocks)
    window = (
        candidate_window_rows(live_rows, blocks, block_size)
        * (blocks * _WINDOW_BYTES_PER_BLOCK + _WINDOW_BYTES_PER_ROW)
        if blocks
        else 0
    )
    kept = live_rows * max(0, kept_blocks) * _KEPT_ID_BYTES
    return _mm_pages(INDEXER_SCORES_BUDGET_BYTES) + _mm_pages(window) + kept


def csa2_activation_bytes(*, live_rows: int, num_heads: int) -> int:
    """Returns per-device bytes of the CSA2 attention's workspaces.

    The decode's split partials plus the prefill's ``PREFILL_ROWS``
    dequantized rows, in whole MemoryManager pages. The prefill's per-slot
    map (four bytes per slot of the caches it reads) grows with the KV pool
    and is not counted.

    Args:
        live_rows: Query rows ``T`` in one forward.
        num_heads: Query heads a device attends.
    """
    if live_rows <= 0:
        return 0
    decode = decode_workspace_floats(num_heads, DECODE_SPLIT_ROWS) * 4
    return _mm_pages(decode + PREFILL_ROWS * (HEAD_DIM + 2) * 2)


def graph_arena_bytes(
    *, live_rows: int, row_bytes: int = _ARENA_ROW_BYTES
) -> int:
    """Returns per-device bytes of the forward graph's planned arena.

    A fixed part plus a per-row part, rounded up to whole pages.

    Args:
        live_rows: Query rows in one forward.
        row_bytes: Transient bytes per query row.
    """
    if live_rows <= 0:
        return 0
    return _mm_pages(_ARENA_FIXED_BYTES + live_rows * row_bytes)


def ep_receive_pool_bytes(ep_config: Any) -> int:
    """Returns the routed experts' statically sized receive-side bytes.

    The dispatched NVFP4 rows, their block scales and source info stay live
    while the fused expert kernel writes the bf16 down projection, all sized
    for the most rows a device can receive, whatever the batch. This is the
    largest single transient of a forward, and every forward and captured
    graph holds one.

    Args:
        ep_config: The expert-parallel :class:`EPConfig`.
    """
    rows = int(ep_config.get_max_recv_tokens())
    hidden = int(ep_config.hidden_size)
    n_ranks = int(ep_config.n_gpus_per_node) * int(ep_config.n_nodes)
    n_local_experts = int(ep_config.n_experts) // n_ranks
    scale_tiles = -(
        -(rows + n_local_experts * _NVFP4_SCALE_TILE_ROWS)
        // _NVFP4_SCALE_TILE_ROWS
    ) * -(-hidden // _NVFP4_SCALE_TILE_COLS)
    tokens = rows * hidden // 2
    src_info = rows * 2 * 4
    down = rows * hidden * 2
    return _mm_pages(
        tokens + scale_tiles * _NVFP4_SCALE_TILE_BYTES + src_info + down
    )


def graph_capture_pool_bytes(
    *,
    capture_rows: int,
    head_rows: int,
    vocab_size: int,
    row_bytes: int,
    indexer_bytes: int,
    attention_bytes: int,
    ep_pool_bytes: int,
) -> int:
    """Returns per-device bytes of the captured device graphs' memory pool.

    Captured graphs run one at a time and each capture reuses the previous
    graph's working memory, so they share one pool sized by the largest
    graph: the batch-independent receive tensors, indexer scratch and CSA2
    workspaces plus the largest decode batch's per-row transients. Device 0
    also holds the LM head's logits and the sampling working memory over
    ``[rows, vocab]`` for every replica's rows. KV is the same size on every
    device, so every device reserves device 0's pool.

    Args:
        capture_rows: Query rows per replica of the largest captured graph.
        head_rows: Rows the LM head scores in that graph, over all replicas.
        vocab_size: Vocabulary size.
        row_bytes: Transient bytes per query row.
        indexer_bytes: The indexer's transient bytes at ``capture_rows``.
        attention_bytes: :func:`csa2_activation_bytes` at ``capture_rows``.
        ep_pool_bytes: :func:`ep_receive_pool_bytes`, or 0 without
            expert-parallel dispatch.
    """
    if capture_rows <= 0:
        return 0
    return (
        graph_arena_bytes(live_rows=capture_rows, row_bytes=row_bytes)
        + indexer_bytes
        + attention_bytes
        + ep_pool_bytes
        + _mm_pages(
            ceil(
                head_rows
                * vocab_size
                * (_HEAD_LOGITS_BYTES + _HEAD_SAMPLING_BYTES)
            )
        )
    )


def forward_activation_bytes(
    *, live_rows: int, candidate_blocks: int, kept_blocks: int, block_size: int
) -> int:
    """Returns one forward's per-device transient peak bytes.

    The graph arena plus the indexer's transient memory.
    """
    return graph_arena_bytes(live_rows=live_rows) + indexer_activation_bytes(
        live_rows=live_rows,
        candidate_blocks=candidate_blocks,
        kept_blocks=kept_blocks,
        block_size=block_size,
    )


def kv_leaf_bytes_per_token(
    text_config: Any,
    *,
    page_size: int,
    cache_dtype: DType,
    csa_records: bool = kv_leaves.CSA_KV_RECORDS,
) -> dict[str, float]:
    """Returns each KV leaf's device bytes per token a page covers.

    A page's bytes over its ``page_size`` tokens, summed over the leaf's
    owning layers, for one data-parallel replica. Sliding-window leaves
    hold only a window's pages per request, so their bytes do not grow
    with context length.

    Args:
        text_config: The model's ``text_config``.
        page_size: Tokens a page covers.
        cache_dtype: Element type of the leaves that store element rows.
        csa_records: Passed to :func:`kv_leaves.kv_leaf_specs`.
    """
    return {
        spec.key: spec.page_bytes(page_size, cache_dtype) / page_size
        for spec in kv_leaves.kv_leaf_specs(
            text_config, csa_records=csa_records
        )
    }


def candidate_blocks_per_row(
    text_config: Any, max_length: int, position_capacity: int
) -> int:
    """Returns the candidate blocks a row scores, or 0 if none is dropped.

    Sized for the smallest compress ratio, which gives the most blocks. As
    in :func:`candidate_keep_blocks`, the gate reads ``max_length`` and the
    blocks cover ``position_capacity``.
    """
    ratios = [r for r in text_config.compress_ratios if r > 0]
    if not ratios:
        return 0
    return candidate_keep_blocks(
        max_length,
        position_capacity,
        min(ratios),
        block_size=getattr(text_config, "candidate_block_size", 0) or 0,
        topk_blocks=getattr(text_config, "candidate_topk_blocks", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Weight sizing from safetensors headers. Checkpoint file sizes are wrong for
# this model: Engram's hash tables live in pinned host memory, DSpark and
# vision tensors are not loaded by the trunk (the vision model holds them on
# device 0), and replicated tensors are held once per device.
# ---------------------------------------------------------------------------

# Engram's hash tables (weight and e8m0 scale) live in pinned host memory.
_HOST_RESIDENT_MARKER = ".engram.embed."

# Split across the devices (each holds ``1 / num_devices``): the routed
# experts (expert parallel), the head-sharded attention projections, and
# Engram's column-parallel ``wkv`` (weight and scale). Everything else is
# replicated, including ``wo_b`` and the indexer (held whole on each device).
# Under data parallelism only the routed experts are split.
_EXPERT_MARKER = ".ffn.experts."
_SHARDED_MARKERS = (
    ".ffn.experts.",
    ".attn.wq_b.",
    ".attn.wo_a.",
    ".attn.attn_sink",
    ".engram.wkv.",
)
# The query projections. Unless they run as MXFP8, their fp8 weights are
# dequantized (2x) and their e8m0 scales folded in.
_QUERY_PROJECTIONS = tuple(
    f".attn.{projection}"
    for projection in (*FUSED_QUERY_PROJECTIONS, "wq_b", "indexer.wq_b")
)
# The router gate's bf16 weight, held in float32 unless the fused route
# reads it as is.
_ROUTER_GATE_SUFFIX = ".ffn.gate.weight"
# Routed-expert NVFP4 block scales. Unless pre-interleaved on the host, the
# compiler hoists the in-graph interleave into a second resident copy (2x).
_ROUTED_EXPERT_SCALE_MARKER = ".ffn.experts."
_ROUTED_EXPERT_SCALE_SUFFIX = ".weight_scale"


@dataclass(frozen=True)
class CheckpointTensor:
    """One safetensors header entry, reduced to what weight sizing needs."""

    name: str
    dtype: str
    nbytes: int


def read_safetensors_headers(paths: Iterable[Path]) -> list[CheckpointTensor]:
    """Reads every tensor's name, dtype and byte size from safetensors headers.

    Reads only each file's length prefix and JSON header.
    """
    tensors: list[CheckpointTensor] = []
    for path in paths:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            tensors.append(CheckpointTensor(name, entry["dtype"], end - begin))
    return tensors


@dataclass(frozen=True)
class DeviceWeightLayout:
    """The load-time switches that change how many bytes a tensor takes."""

    head_bf16: bool
    host_dequantized: Callable[[str], bool]
    """Whether a projection base (no ``.weight``) is dequantized to bf16."""
    expert_scales_preinterleaved: bool
    router_gate_float32: bool = True
    """Whether the router holds a float32 copy of its bf16 gate weight."""
    q_mxfp8: bool = True
    """Whether the query projections keep their fp8 weights."""
    compressor_bf16: bool = True
    """Whether the compressors' ``wkv``/``wgate`` stay bf16."""

    @classmethod
    def current(cls, *, router_gate_float32: bool = True) -> DeviceWeightLayout:
        """The layout the weight adapter uses under the current env switches."""
        return cls(
            head_bf16=head_bf16_enabled(),
            host_dequantized=functools.partial(
                is_host_dequantized, woa_mxfp8=woa_mxfp8_enabled()
            ),
            # Assumes every expert scale shape tiles the interleave tile
            # exactly, so the env switch alone decides.
            expert_scales_preinterleaved=nvfp4_sf_preinterleave_enabled(),
            router_gate_float32=router_gate_float32,
            q_mxfp8=q_mxfp8_enabled(),
            compressor_bf16=compressor_bf16_enabled(),
        )


def _device_bytes_factor(
    tensor: CheckpointTensor, layout: DeviceWeightLayout
) -> float:
    """How many bytes one checkpoint byte becomes once loaded on a device.

    bf16 weights widened to float32 double. fp8 weights dequantized to bf16
    on the host double, and their e8m0 scales never reach a device. Every
    other MXFP8 weight's compact ``[N / block, K / block]`` e8m0 scale is
    row-expanded to ``[N, K / block]``.
    """
    name, dtype = tensor.name, tensor.dtype
    if name == HEAD_WEIGHT:
        return 1.0 if layout.head_bf16 else 2.0
    if dtype == "BF16" and name.endswith(FLOAT32_WEIGHT_SUFFIXES):
        return 2.0
    if dtype == "BF16" and name.endswith(COMPRESSOR_WEIGHT_SUFFIXES):
        return 1.0 if layout.compressor_bf16 else 2.0
    if dtype == "BF16" and name.endswith(_ROUTER_GATE_SUFFIX):
        return 2.0 if layout.router_gate_float32 else 1.0
    if dtype == "F8_E4M3" and name.endswith(".weight"):
        base = name[: -len(".weight")]
        if _ROUTED_EXPERT_SCALE_MARKER not in name and (
            layout.host_dequantized(base)
            or (not layout.q_mxfp8 and base.endswith(_QUERY_PROJECTIONS))
        ):
            return 2.0
        return 1.0
    if dtype == "F8_E8M0" and name.endswith(".scale"):
        base = name[: -len(".scale")]
        if layout.host_dequantized(base) or (
            not layout.q_mxfp8 and base.endswith(_QUERY_PROJECTIONS)
        ):
            return 0.0
        return float(MXFP8_BLOCK_SIZE)
    if (
        dtype == "F8_E4M3"
        and _ROUTED_EXPERT_SCALE_MARKER in name
        and name.endswith(_ROUTED_EXPERT_SCALE_SUFFIX)
        and not layout.expert_scales_preinterleaved
    ):
        return 2.0
    return 1.0


def device_weight_bytes(
    tensors: Iterable[CheckpointTensor],
    num_devices: int,
    layout: DeviceWeightLayout | None = None,
    *,
    data_parallel: bool = False,
) -> tuple[int, int]:
    """Returns per-device weight bytes, plus the bytes kept in host memory.

    Args:
        tensors: The checkpoint's tensors (:func:`read_safetensors_headers`).
        num_devices: Devices the trunk is split across (the EP degree).
        layout: The load-time switches; defaults to
            :meth:`DeviceWeightLayout.current`.
        data_parallel: Whether attention runs data-parallel, so only the
            routed experts split across devices.

    Returns:
        ``(per_device_bytes, host_bytes)``: what each device holds after
        load, and the pinned host-resident Engram table bytes.
    """
    num_devices = max(1, num_devices)
    if layout is None:
        layout = DeviceWeightLayout.current()
    per_device = 0.0
    host = 0
    for tensor in tensors:
        name = tensor.name
        # DSpark stages and vision tensors are not loaded with the trunk.
        if name.startswith(
            DSPARK_CHECKPOINT_PREFIX
        ) or is_skipped_checkpoint_tensor(name):
            continue
        if _HOST_RESIDENT_MARKER in name:
            host += tensor.nbytes
            continue
        nbytes = tensor.nbytes * _device_bytes_factor(tensor, layout)
        markers = (_EXPERT_MARKER,) if data_parallel else _SHARDED_MARKERS
        if any(marker in name for marker in markers):
            nbytes /= num_devices
        per_device += nbytes
    return ceil(per_device), host


def vision_weight_bytes(tensors: Iterable[CheckpointTensor]) -> int:
    """Returns the vision model's device bytes, held on device 0 only.

    Its bf16 RMSNorm weights are widened to float32.
    """
    return sum(
        tensor.nbytes
        * (
            2
            if tensor.dtype == "BF16"
            and VISION_NORM_WEIGHT.fullmatch(tensor.name)
            else 1
        )
        for tensor in tensors
        if is_skipped_checkpoint_tensor(tensor.name)
    )


# Measured on a B200 with the memory manager off: one encode of 9189 patches
# (the most one image's span allows) peaks 1032 MiB above the loaded model,
# 117764 bytes per patch. Reserved with a 25% margin.
_VISION_BYTES_PER_PATCH = 117_764
_VISION_PEAK_MARGIN = 1.25


def vision_activation_bytes(vision_config: Any, image_tokens: int) -> int:
    """Returns the vision model's peak for images spanning ``image_tokens``.

    An aligner row takes ``downsample_ratio**2`` patches and fills one span
    row, so the images have at most ``downsample_ratio**2 * image_tokens``
    patches. In whole MemoryManager pages.
    """
    patches = vision_config.downsample_ratio**2 * max(0, image_tokens)
    return _mm_pages(
        ceil(patches * _VISION_BYTES_PER_PATCH * _VISION_PEAK_MARGIN)
    )


def check_planned_max_length(
    *, planned: int | None, requested: int, forward_rows: int
) -> None:
    """Refuses a memory plan whose KV pool cannot hold one prefill chunk.

    Without this check the plan would succeed with ``max_length`` clamped to
    a few tokens. The error is worded like other out-of-memory planning
    errors so callers that retry with smaller configs recognize it.

    Args:
        planned: The plan's ``planned_max_length``.
        requested: The ``max_length`` the config asked for.
        forward_rows: Query rows one forward carries (the prefill chunk).

    Raises:
        RuntimeError: If ``planned < min(requested, forward_rows)``.
    """
    if not all(isinstance(v, int) for v in (planned, requested, forward_rows)):
        return  # No plan (virtual devices) or a partial test double.
    assert planned is not None
    if planned >= min(requested, forward_rows):
        return
    raise RuntimeError(
        f"DeepSeek-V4.1-Flash: the KV pool left after the weights and the "
        f"forward's {forward_rows}-row activation reserve holds only "
        f"{planned} tokens (max_length {requested} requested), so the "
        "model, activations, and signal buffers don't leave room for KV "
        "cache. Lower max_length or the prefill chunk "
        "(max_batch_input_tokens), or raise device_memory_utilization."
    )


class DeepseekV41MemoryPlanner(PagedMemoryPlanner):
    """Reserves the forward's transient memory out of the KV budget.

    The graph arena and indexer scratch (:func:`forward_activation_bytes`),
    the captured device graphs' pool (:func:`graph_capture_pool_bytes`) and
    fragmentation headroom can reach tens of GiB per device, so they must be
    subtracted before the KV leaves are sized.
    """

    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Estimates device weight bytes, summed over devices.

        Free memory is compared summed over devices, so this returns the
        per-device size times the device count, plus the vision model's
        weights on device 0. Falls back to the checkpoint's file sizes when
        the headers cannot be read (e.g. a remote repo).
        """
        model_config = pipeline_config.model
        num_devices = max(1, len(model_config.device_specs))
        try:
            paths = model_config.resolved_weight_paths()
            tensors = read_safetensors_headers(
                p for p in paths if str(p).endswith(".safetensors")
            )
        except (OSError, ValueError) as e:
            logger.warning(
                "DeepSeek-V4.1-Flash weight sizing fell back to file sizes "
                "(could not read safetensors headers: %s)",
                e,
            )
            return model_config.weights_size()
        if not tensors:
            return model_config.weights_size()
        text_config = DeepseekV41Config.get_text_config(
            model_config.huggingface_config
        )
        layout = DeviceWeightLayout.current(
            router_gate_float32=not router_reads_bf16_gate(
                model_config.device_specs[0].device_type == "gpu",
                int(text_config.n_routed_experts),
            )
        )
        per_device, host = device_weight_bytes(
            tensors,
            num_devices,
            layout,
            data_parallel=int(model_config.data_parallel_degree) > 1,
        )
        vision = (
            vision_weight_bytes(tensors)
            if max_images_per_request(pipeline_config) != 0
            else 0
        )
        logger.info(
            "Estimated DeepSeek-V4.1-Flash weights: %s per device x %d "
            "devices, plus %s (%d bytes) of vision weights on device 0 (%s "
            "of Engram tables kept in host memory)",
            to_human_readable_bytes(per_device),
            num_devices,
            to_human_readable_bytes(vision),
            vision,
            to_human_readable_bytes(host),
        )
        return per_device * num_devices + vision

    def estimate_activation_memory(
        self, pipeline_config: Any, huggingface_config: Any
    ) -> int:
        """Estimates one forward's per-device transient peak bytes.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration (the
                top-level config, with text fields under ``text_config``).

        Returns:
            :func:`graph_arena_bytes`, :func:`indexer_activation_bytes`,
            the expert-parallel buffers, :func:`graph_capture_pool_bytes`
            with device graph capture, :func:`csa2_activation_bytes`, and
            fragmentation headroom per device, summed over devices (free
            memory is summed the same way), plus
            :func:`vision_activation_bytes` of one largest image on device 0.
            Excludes KV leaf storage (see module docstring).
        """
        text_config = DeepseekV41Config.get_text_config(huggingface_config)
        specs = kv_leaves.kv_leaf_specs(text_config)
        vision = (
            self._vision_activation_bytes(huggingface_config)
            if max_images_per_request(pipeline_config) != 0
            else 0
        )

        if not any(spec.kind == "comp" for spec in specs):
            # No compressed layer, so no indexer.
            return vision

        runtime = pipeline_config.runtime
        max_length = pipeline_config.model.max_length
        # Decode also keeps candidate blocks, so this is not zeroed for
        # decode-only roles.
        live_rows = forward_row_bound(
            runtime.max_num_input_tokens,
            runtime.max_batch_input_tokens,
            max_length,
        )
        num_devices = max(1, len(pipeline_config.model.device_specs))
        dp = int(pipeline_config.model.data_parallel_degree)
        blocks = candidate_blocks_per_row(
            text_config,
            max_length,
            max_length + spec_decode_cache_slack(self._config.get_kv_params()),
        )
        kept_blocks = (
            getattr(text_config, "candidate_topk_blocks", 0) or 0
            if blocks
            else 0
        )
        block_size = getattr(text_config, "candidate_block_size", 0) or 0
        indexer = indexer_activation_bytes(
            live_rows=live_rows,
            candidate_blocks=blocks,
            kept_blocks=kept_blocks,
            block_size=block_size,
        )
        attention = csa2_activation_bytes(
            live_rows=live_rows, num_heads=text_config.num_attention_heads
        )
        ep_config = data_parallel_ep_config(pipeline_config, text_config)
        row_bytes = (
            _ARENA_ROW_BYTES if ep_config is None else _EP_ARENA_ROW_BYTES
        )
        arena = graph_arena_bytes(live_rows=live_rows, row_bytes=row_bytes)
        ep_pool = 0
        if ep_config is not None:
            # Expert-parallel dispatch: the persistent communication buffers,
            # and the experts' intermediates over the statically sized
            # receive rows.
            ep_buffers = ep_config.estimate_memory_usage()
            ep_pool = ep_receive_pool_bytes(ep_config)
            moe_rows = ep_buffers + ep_pool
            speculative = pipeline_config.speculative
            if speculative is not None and speculative.is_dspark():
                # The drafter's experts still run every replica's draft
                # blocks on each device.
                assert runtime.max_batch_size is not None
                draft_rows = runtime.max_batch_size * (
                    (speculative.num_speculative_tokens or 0) + 1
                )
                moe_rows += _mm_pages((dp - 1) * draft_rows * _MOE_ROW_BYTES)
            logger.info(
                "Estimated DeepSeek-V4.1-Flash expert-parallel memory: %s "
                "per device (communication buffers %s, %d receive rows)",
                to_human_readable_bytes(moe_rows),
                to_human_readable_bytes(ep_buffers),
                ep_config.get_max_recv_tokens(),
            )
        else:
            # Each replica gets its own row budget; under data parallelism
            # without expert dispatch, the MoE runs on every replica's rows.
            moe_rows = _mm_pages((dp - 1) * live_rows * _MOE_ROW_BYTES)
        graph_pool = 0
        if runtime.device_graph_capture is True:
            # Captured graphs keep their own pool, which eager forwards never
            # reuse. Each decode request carries its draft rows.
            assert runtime.max_batch_size is not None
            speculative = pipeline_config.speculative
            draft_tokens = (
                speculative.num_speculative_tokens or 0
                if speculative is not None and speculative.is_dspark()
                else 0
            )
            capture_rows = runtime.max_batch_size * (draft_tokens + 1)
            graph_pool = graph_capture_pool_bytes(
                capture_rows=capture_rows,
                head_rows=max(1, dp) * capture_rows,
                vocab_size=int(text_config.vocab_size),
                row_bytes=row_bytes,
                indexer_bytes=indexer_activation_bytes(
                    live_rows=capture_rows,
                    candidate_blocks=blocks,
                    kept_blocks=kept_blocks,
                    block_size=block_size,
                ),
                attention_bytes=csa2_activation_bytes(
                    live_rows=capture_rows,
                    num_heads=text_config.num_attention_heads,
                ),
                ep_pool_bytes=ep_pool,
            )
        # Freed transients are reused only in place, so a fragmented free
        # list must still fit a fresh copy of the largest one.
        headroom = max(ep_pool, _MM_PAGE_BYTES)
        per_device = (
            arena + indexer + attention + moe_rows + graph_pool + headroom
        )

        logger.info(
            "Estimated DeepSeek-V4.1-Flash forward activation memory: %s per "
            "device x %d devices (graph arena %s + indexer %s + attention %s "
            "+ device graph pool %s + fragmentation headroom %s; %d live "
            "rows, %d candidate blocks)",
            to_human_readable_bytes(per_device),
            num_devices,
            to_human_readable_bytes(arena),
            to_human_readable_bytes(indexer),
            to_human_readable_bytes(attention),
            to_human_readable_bytes(graph_pool),
            to_human_readable_bytes(headroom),
            live_rows,
            blocks,
        )
        return per_device * num_devices + vision

    @staticmethod
    def _vision_activation_bytes(huggingface_config: Any) -> int:
        """The vision model's peak for one largest image."""
        vision_config = getattr(huggingface_config, "vision_config", None)
        if vision_config is None:
            return 0
        # TODO: a forward whose rows overlap several unencoded images
        # encodes them in one call, whose peak grows with their patches.
        peak = vision_activation_bytes(
            vision_config, vision_config.max_image_tokens
        )
        logger.info(
            "Estimated DeepSeek-V4.1-Flash vision activation memory on "
            "device 0: %s for one %d-token image",
            to_human_readable_bytes(peak),
            vision_config.max_image_tokens,
        )
        return peak
