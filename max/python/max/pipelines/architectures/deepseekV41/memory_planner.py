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

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.support.human_readable_formatter import to_human_readable_bytes

from . import kv_leaves
from .layers.indexer import candidate_blocks_active, candidate_num_blocks
from .model_config import (
    DeepseekV41Config,
    data_parallel_ep_config,
    forward_row_bound,
    head_bf16_enabled,
    max_images_per_request,
    nvfp4_sf_preinterleave_enabled,
    woa_mxfp8_enabled,
)
from .weight_adapters import (
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
# Upper bound on the MoE's transient bytes per row (gathered input, routed
# top-k intermediates, float32 partial sum), for the other replicas' rows a
# data-parallel device also runs the MoE on.
_MOE_ROW_BYTES = 256 << 10

# Upper bound on the routed experts' transient bytes per row of an
# expert-parallel receive buffer: the rows arrive quantized, so this is the
# bf16 down projection output plus slack.
_EP_RECV_ROW_BYTES = 12 << 10
_ARENA_FIXED_BYTES = 5 << 28  # 1.25 GiB
# The indexer kernel scores query rows in windows that fit this transient
# budget, whatever the batch or context length.
_INDEXER_SCORES_BUDGET_BYTES = 512 << 20
# The candidate keep step's ``[T, blocks]`` tensors: two float32 (the block
# maxima and their forced copy) and up to four one-byte masks; plus its
# top-k's float32 values, int64 ids and int32 tie ids, at most one per block.
_KEEP_BYTES_PER_ENTRY = 2 * 4 + 4 + 4 + 8 + 4
# The device MemoryManager allocates whole 256 MiB pages.
_MM_PAGE_BYTES = 256 << 20


def _mm_pages(nbytes: int) -> int:
    """``nbytes`` rounded up to whole MemoryManager pages."""
    return ceil(nbytes / _MM_PAGE_BYTES) * _MM_PAGE_BYTES if nbytes > 0 else 0


def indexer_activation_bytes(*, live_rows: int, candidate_blocks: int) -> int:
    """Returns per-device bytes of the indexer's transient memory.

    The kernel's score window plus the candidate keep step's ``[T, blocks]``
    tensors, each in whole MemoryManager pages.

    Args:
        live_rows: Query rows ``T`` in one forward.
        candidate_blocks: Candidate blocks per row, or 0 if no layer keeps
            candidate blocks.
    """
    if live_rows <= 0:
        return 0
    return _mm_pages(_INDEXER_SCORES_BUDGET_BYTES) + _mm_pages(
        live_rows * max(0, candidate_blocks) * _KEEP_BYTES_PER_ENTRY
    )


def graph_arena_bytes(*, live_rows: int) -> int:
    """Returns per-device bytes of the forward graph's planned arena.

    A fixed part plus a per-row part, rounded up to whole pages.
    """
    if live_rows <= 0:
        return 0
    return _mm_pages(_ARENA_FIXED_BYTES + live_rows * _ARENA_ROW_BYTES)


def forward_activation_bytes(*, live_rows: int, candidate_blocks: int) -> int:
    """Returns one forward's per-device transient peak bytes.

    The graph arena plus the indexer's transient memory.
    """
    return graph_arena_bytes(live_rows=live_rows) + indexer_activation_bytes(
        live_rows=live_rows, candidate_blocks=candidate_blocks
    )


def candidate_blocks_per_row(text_config: Any, max_length: int) -> int:
    """Returns the candidate blocks per row the keep step builds, or 0.

    Sized for the smallest compress ratio, which gives the most blocks.
    """
    block_size = getattr(text_config, "candidate_block_size", 0) or 0
    topk_blocks = getattr(text_config, "candidate_topk_blocks", 0) or 0
    ratios = [r for r in text_config.compress_ratios if r > 0]
    if not ratios:
        return 0
    ratio = min(ratios)
    if not candidate_blocks_active(
        max_length, ratio, block_size=block_size, topk_blocks=topk_blocks
    ):
        return 0
    return candidate_num_blocks(max_length, ratio, block_size)


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
# The fused query projection's fp8 weights are dequantized (2x) and their
# e8m0 scales folded in.
_FUSED_BF16_PROJECTIONS = tuple(
    f".attn.{projection}" for projection in FUSED_QUERY_PROJECTIONS
)
# The router gate's bf16 weight is held in float32.
_FLOAT32_GATE_SUFFIX = ".ffn.gate.weight"
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

    @classmethod
    def current(cls) -> DeviceWeightLayout:
        """The layout the weight adapter uses under the current env switches."""
        return cls(
            head_bf16=head_bf16_enabled(),
            host_dequantized=functools.partial(
                is_host_dequantized, woa_mxfp8=woa_mxfp8_enabled()
            ),
            # Assumes every expert scale shape tiles the interleave tile
            # exactly, so the env switch alone decides.
            expert_scales_preinterleaved=nvfp4_sf_preinterleave_enabled(),
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
    if dtype == "BF16" and name.endswith(
        (*FLOAT32_WEIGHT_SUFFIXES, _FLOAT32_GATE_SUFFIX)
    ):
        return 2.0
    if dtype == "F8_E4M3" and name.endswith(".weight"):
        base = name[: -len(".weight")]
        if _ROUTED_EXPERT_SCALE_MARKER not in name and (
            layout.host_dequantized(base)
            or base.endswith(_FUSED_BF16_PROJECTIONS)
        ):
            return 2.0
        return 1.0
    if dtype == "F8_E8M0" and name.endswith(".scale"):
        base = name[: -len(".scale")]
        if layout.host_dequantized(base) or base.endswith(
            _FUSED_BF16_PROJECTIONS
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

    The graph arena and indexer scratch (:func:`forward_activation_bytes`)
    can reach tens of GiB per device, so they must be subtracted before the
    KV leaves are sized.
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
        per_device, host = device_weight_bytes(
            tensors,
            num_devices,
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
            :func:`graph_arena_bytes` plus :func:`indexer_activation_bytes`
            per device, summed over devices (free memory is summed the same
            way), plus :func:`vision_activation_bytes` of one largest image
            on device 0. Excludes KV leaf storage (see module docstring).
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
        # Decode also runs the keep step over every block, so this is not
        # zeroed for decode-only roles.
        live_rows = forward_row_bound(
            runtime.max_num_input_tokens,
            runtime.max_batch_input_tokens,
            max_length,
        )
        num_devices = max(1, len(pipeline_config.model.device_specs))
        dp = int(pipeline_config.model.data_parallel_degree)
        blocks = candidate_blocks_per_row(text_config, max_length)
        arena = graph_arena_bytes(live_rows=live_rows)
        indexer = indexer_activation_bytes(
            live_rows=live_rows, candidate_blocks=blocks
        )
        ep_config = data_parallel_ep_config(pipeline_config, text_config)
        if ep_config is not None:
            # Expert-parallel dispatch: the persistent communication buffers,
            # and the experts' intermediates over the statically sized
            # receive rows.
            ep_buffers = ep_config.estimate_memory_usage()
            moe_rows = ep_buffers + _mm_pages(
                ep_config.get_max_recv_tokens() * _EP_RECV_ROW_BYTES
            )
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
        per_device = arena + indexer + moe_rows

        logger.info(
            "Estimated DeepSeek-V4.1-Flash forward activation memory: %s per "
            "device x %d devices (graph arena %s + indexer %s; %d live rows, "
            "%d candidate blocks)",
            to_human_readable_bytes(per_device),
            num_devices,
            to_human_readable_bytes(arena),
            to_human_readable_bytes(indexer),
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
