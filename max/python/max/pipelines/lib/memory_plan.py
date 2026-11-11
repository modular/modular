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

"""The memory plan a pipeline is built with.

The registry resolves the batch and length bounds before any device memory
is touched; the model worker measures what the pipeline allocates and fills
in the KV-cache budget (see :mod:`max.pipelines.lib.memory_ledger`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from max.driver import Device, DeviceSpec, is_virtual_device_mode, load_devices
from max.dtype import DType
from max.support.human_readable_formatter import to_human_readable_bytes

from .vision_encoder_cache import (
    DEFAULT_VISION_CACHE_BLOCK_TOKENS,
    VisionCachePlan,
)

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .registry import SupportedArchitecture

__all__ = [
    "FreeMemory",
    "MemoryPlan",
    "plan_memory",
    "vision_cache_plan",
]

logger = logging.getLogger("max.pipelines")

_DEFAULT_BATCH_SIZE = 512


@dataclass(frozen=True)
class FreeMemory:
    """Bounds on free device memory, in bytes.

    ``lo`` is the memory known to be free and ``hi`` the most that could be.
    They are equal when the driver measures free memory; when it can't,
    ``lo`` is 0 and ``hi`` is the device's capacity.
    """

    lo: int
    hi: int

    def __add__(self, other: FreeMemory) -> FreeMemory:
        return FreeMemory(self.lo + other.lo, self.hi + other.hi)

    @classmethod
    def from_stats(cls, device: Device) -> FreeMemory:
        """Reads the free-memory range from ``device.stats``."""
        stats = device.stats
        free = stats["free_memory"]
        if free is None:
            return cls(0, int(stats["total_memory"]))
        return cls(int(free), int(free))

    @classmethod
    def summed(cls, devices: Iterable[Device]) -> FreeMemory:
        """Returns the free memory summed across ``devices``."""
        return sum((cls.from_stats(d) for d in devices), cls(0, 0))

    @classmethod
    def smallest(cls, devices: Iterable[Device]) -> FreeMemory:
        """Returns the least free memory on any one of ``devices``."""
        ranges = [cls.from_stats(d) for d in devices]
        return cls(
            min((r.lo for r in ranges), default=0),
            min((r.hi for r in ranges), default=0),
        )


@dataclass(frozen=True)
class MemoryPlan:
    """The memory plan computed when a pipeline is loaded.

    Carries the effective batch size, sequence-length bound, and memory
    budgets that the pipeline and its schedulers consume. Produced by
    :func:`plan_memory`.
    """

    planned_max_batch_size: int
    """The maximum number of requests scheduled together in one batch: the
    user's ``runtime.max_batch_size``, or the value planning inferred when
    the user left it unset."""

    footprint: int
    """Unused; kept for callers that still construct plans with it."""

    planned_max_length: int | None
    """The resolved maximum sequence length after memory planning lowered
    it to fit device memory: the construction-resolved
    ``config.model.max_length``, clamped to what the KV cache and any
    draft model can hold. ``None`` for pipelines with no main language
    model, such as diffusion pipelines."""

    available_cache_memory: int | None = None
    """The device memory committed to the KV cache, in bytes. ``None`` when
    the plan reserves no KV-cache budget, such as for models without a KV
    cache."""

    device_specs: tuple[DeviceSpec, ...] | None = None
    """The device specs the plan was computed for, kept as specs rather
    than ``Device`` objects so the plan can cross process boundaries.
    ``None`` for plans that never load devices, such as diffusion
    pipelines."""

    planned_max_batch_total_tokens: int | None = None
    """Cap on the total context tokens resident across a batch: the user's
    ``runtime.max_batch_total_tokens``, or ``planned_max_length`` for
    architectures that require a cap. ``None`` means no cap is configured."""

    vision_cache_plan: VisionCachePlan | None = None
    """Block-mode vision cache reservation; ``None`` means entry-count mode."""

    def require_device_specs(self) -> tuple[DeviceSpec, ...]:
        """Returns the device specs, which must be set on this plan.

        Raises:
            AssertionError: If the plan carries no device specs.
        """
        assert self.device_specs is not None, (
            "memory plan lacks device specs; pipelines require a "
            "plan built by the registry's memory-planning step"
        )
        return self.device_specs


def _bounded_by_draft(
    max_length: int | None, draft_max_seq_len: int | None
) -> int | None:
    """Lowers ``max_length`` to the draft model's own sequence limit."""
    if (
        draft_max_seq_len is None
        or max_length is None
        or max_length <= draft_max_seq_len
    ):
        return max_length
    logger.info(
        "Clamping max_length from %d to %d (draft model max sequence length)",
        max_length,
        draft_max_seq_len,
    )
    return draft_max_seq_len


def _max_batch_total_tokens(
    pipeline_config: PipelineConfig,
    arch: SupportedArchitecture | None,
    max_length: int | None,
) -> int | None:
    """The cap on context tokens resident across a batch.

    Architectures that require chunked prefill need a cap; it defaults to the
    bounded ``max_length`` so the scheduler never admits more resident
    context than the KV cache holds.
    """
    configured = pipeline_config.runtime.max_batch_total_tokens
    if (
        configured is not None
        or arch is None
        or not arch.requires_max_batch_context_length
    ):
        return configured
    logger.warning(
        "Architecture '%s' requires max-batch-total-tokens to be specified "
        "but found None. Defaulting to the max sequence length of the model: %s",
        arch.name,
        max_length,
    )
    return max_length


def _vision_row(
    pipeline_config: PipelineConfig, arch: SupportedArchitecture
) -> VisionCachePlan | None:
    """The vision encoder cache's row layout, with no bytes reserved yet.

    ``None`` when the model has no vision tower, the cache is turned off
    (``vision_cache_utilization`` 0), or the architecture names no row.
    """
    from .interfaces import ArchConfigWithVisionCache, arch_has_vision_tower

    hf_config = pipeline_config.model.huggingface_config
    if pipeline_config.runtime.vision_cache_utilization == 0 or not (
        arch_has_vision_tower(arch.config, hf_config)
    ):
        return None
    assert issubclass(arch.config, ArchConfigWithVisionCache)
    row_spec = arch.config.get_vision_cache_row_spec(hf_config)
    if row_spec is None:
        logger.warning(
            "%s's arch config reports no vision cache row spec "
            "(get_vision_cache_row_spec); no vision encoder cache will be "
            "reserved.",
            arch.name,
        )
        return None
    hidden_size, dtype = row_spec
    assert isinstance(dtype, DType)
    return VisionCachePlan(
        bytes_per_device=0, hidden_size=hidden_size, dtype=dtype
    )


def vision_cache_plan(
    pipeline_config: PipelineConfig,
    row: VisionCachePlan | None,
    cache_bytes: int,
    n_devices: int,
) -> tuple[int, VisionCachePlan | None]:
    """Carves the vision encoder cache out of the cache budget.

    ``vision_cache_utilization`` is the share of the budget it gets, rounded
    down to whole blocks sharded across the devices. Returns the bytes taken
    across all devices and the block plan (``None`` when there is none).

    Args:
        pipeline_config: The pipeline's config.
        row: The row layout from planning, or ``None`` for no cache.
        cache_bytes: The cache budget across all devices.
        n_devices: The devices the budget spans.
    """
    if row is None:
        return 0, None
    utilization = pipeline_config.runtime.vision_cache_utilization
    requested = int(cache_bytes * utilization) // n_devices
    block_bytes = (
        DEFAULT_VISION_CACHE_BLOCK_TOKENS
        * row.hidden_size
        * row.dtype.size_in_bytes
    )
    num_blocks = requested // block_bytes // n_devices * n_devices
    if num_blocks == 0:
        raise ValueError(
            f"vision_cache_utilization={utilization} reserves "
            f"{to_human_readable_bytes(requested)} of the KV cache pool, too "
            f"small to fit one {DEFAULT_VISION_CACHE_BLOCK_TOKENS}-token block "
            f"({to_human_readable_bytes(block_bytes)}) per device. Increase "
            "the fraction or set 0 to disable the vision encoder cache."
        )
    total = num_blocks * block_bytes
    logger.info(
        "Vision encoder cache: %d blocks x %d tokens sharded across %d "
        "device(s), %s reserved (%s per device).",
        num_blocks,
        DEFAULT_VISION_CACHE_BLOCK_TOKENS,
        n_devices,
        to_human_readable_bytes(total),
        to_human_readable_bytes(total // n_devices),
    )
    return total, VisionCachePlan(
        bytes_per_device=total // n_devices,
        hidden_size=row.hidden_size,
        dtype=row.dtype,
    )


def plan_memory(
    pipeline_config: PipelineConfig,
    arch: SupportedArchitecture,
    draft_arch: SupportedArchitecture | None = None,
) -> MemoryPlan:
    """Plans a pipeline's batch size and sequence length before it loads.

    On accelerators the KV-cache budget is left unset: the pipeline measures
    its warmups once its weights are loaded and fills the budget in, clamping
    ``max_length`` and halving an unset ``max_batch_size`` as it goes (see
    :mod:`max.pipelines.lib.memory_ledger`). Host-only pipelines, virtual
    devices and ``MAX_MEMORY_MEASURE_DISABLE=1`` get a budget here from free
    memory and a fixed reserve.
    """
    from . import memory_ledger
    from .interfaces import ArchConfigWithKVCache

    runtime = pipeline_config.runtime
    if "main" not in pipeline_config.models:
        # Multi-component pipelines (diffusion) have no KV cache.
        return MemoryPlan(
            planned_max_batch_size=runtime.max_batch_size or 1,
            footprint=0,
            planned_max_length=None,
            planned_max_batch_total_tokens=runtime.max_batch_total_tokens,
        )

    model_config = pipeline_config.model
    device_specs = tuple(model_config.device_specs)
    logger.info(
        "devices: %s",
        ", ".join(f"{d.device_type}[{d.id}]" for d in device_specs),
    )
    if model_config.max_length is None:
        raise ValueError(
            "max_length is unresolved. Construct the config through "
            "PipelineConfig.from_args, which runs the architecture's "
            "sequence-length policy, or set max_length explicitly."
        )
    draft_max_seq_len = None
    if draft_arch is not None and pipeline_config.draft_model is not None:
        draft_max_seq_len = pipeline_config.draft_model.max_length
        if draft_max_seq_len is None:
            raise ValueError(
                "The draft model's max_length is unresolved. Construct the "
                "config through PipelineConfig.from_args, which runs the "
                "draft architecture's sequence-length policy, or set "
                "max_length on the draft model config explicitly."
            )
    max_length = _bounded_by_draft(model_config.max_length, draft_max_seq_len)
    assert max_length is not None

    devices = load_devices(device_specs)
    arch_config = arch.config.initialize(
        pipeline_config, model_config=model_config, max_seq_len=max_length
    )
    has_kv = isinstance(arch_config, ArchConfigWithKVCache)

    user_batch = runtime.max_batch_size
    if user_batch is not None:
        batch = user_batch
    elif not has_kv or (len(devices) == 1 and devices[0].is_host):
        batch = 1
    else:
        batch = _DEFAULT_BATCH_SIZE
    if batch > runtime.max_batch_input_tokens:
        logger.info(
            "max_batch_size of %d cannot be larger than max_batch_input_tokens "
            "of %d; using %d",
            batch,
            runtime.max_batch_input_tokens,
            runtime.max_batch_input_tokens,
        )
        batch = runtime.max_batch_input_tokens

    def finish(
        batch_size: int,
        length: int | None,
        cache_memory: int | None,
        vision: VisionCachePlan | None = None,
    ) -> MemoryPlan:
        if cache_memory is not None:
            logger.info(
                "cache_memory: %s", to_human_readable_bytes(cache_memory)
            )
        return MemoryPlan(
            planned_max_batch_size=batch_size,
            footprint=0,
            planned_max_length=length,
            available_cache_memory=cache_memory,
            device_specs=device_specs,
            planned_max_batch_total_tokens=_max_batch_total_tokens(
                pipeline_config, arch, length
            ),
            vision_cache_plan=vision,
        )

    if is_virtual_device_mode():
        # Cross-compilation allocates nothing; any budget works.
        logger.info("Skipping memory planning in virtual device mode")
        return finish(user_batch or 1, max_length, 1 << 40)
    if not has_kv:
        return finish(batch, max_length, None)
    vision_row = _vision_row(pipeline_config, arch)
    if memory_ledger.measures(devices):
        # The pipeline measures its warmups and fills the budget in.
        return finish(batch, max_length, None, vision_row)

    from max.pipelines.kv_cache import (
        kv_cache_memory_size,
        max_seq_len_fitting_in_cache,
    )

    assert isinstance(arch_config, ArchConfigWithKVCache)
    params = arch_config.get_kv_params()
    utilization = model_config.kv_cache.device_memory_utilization
    cache_bytes = memory_ledger.unmeasured_budget(
        devices, utilization, model_config.weights_size()
    )
    if cache_bytes <= 0:
        if memory_ledger.measuring_disabled():
            # An exporter (gen_mef) plans shapes without allocating a cache.
            logger.warning(
                "No memory is left for a KV cache at "
                "device_memory_utilization=%s; planning without one because "
                "measuring is disabled.",
                utilization,
            )
            return finish(batch, max_length, None)
        raise RuntimeError(
            "The model leaves no memory for a KV cache at "
            f"device_memory_utilization={utilization}. Try a smaller model, a "
            "smaller precision, or more devices."
        )
    # One buffer per device; a device caps its largest allocation.
    cache_bytes = min(
        cache_bytes, sum(d.max_single_alloc_size for d in devices)
    )
    vision_bytes, vision_plan = vision_cache_plan(
        pipeline_config, vision_row, cache_bytes, len(devices)
    )
    kv_bytes = cache_bytes - vision_bytes
    capacity = max_seq_len_fitting_in_cache(
        params=params,
        available_cache_memory=kv_bytes,
        is_di_enabled=runtime.is_disaggregated,
        model_name=model_config.model_name,
    )
    if capacity is not None and capacity < max_length:
        logger.warning(
            "Clamping max_length from %d to %d: the KV cache (%s) holds no "
            "longer request",
            max_length,
            capacity,
            to_human_readable_bytes(kv_bytes),
        )
        max_length = capacity
    if all(d.is_host for d in devices):
        # A host-only run allocates what the planned batch at max_length
        # needs rather than the whole budget, which would claim gigabytes of
        # RAM for a cache it never fills. Accelerators keep the budget.
        kv_bytes = kv_cache_memory_size(
            params=params,
            available_cache_memory=kv_bytes,
            max_batch_size=batch,
            max_seq_len=max_length,
            is_di_enabled=runtime.is_disaggregated,
            model_name=model_config.model_name,
        )
    return finish(batch, max_length, kv_bytes, vision_plan)
