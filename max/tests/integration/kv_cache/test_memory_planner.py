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

"""Unit tests for MemoryPlanner and PagedMemoryPlanner."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
)
from max.pipelines.kv_cache import (
    ModelConfig,
    ModelConfigWithKVCache,
    PagedMemoryPlanner,
)
from max.pipelines.lib.memory_estimation import MemoryEstimator
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.vision_encoder_cache import VisionCachePlan

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.lib.config.model_config import MAXModelConfig
    from max.pipelines.lib.interfaces import ArchConfig
    from max.pipelines.lib.registry import SupportedArchitecture

# ---------------------------------------------------------------------------
# Minimal protocol conformers
# ---------------------------------------------------------------------------


class _MinimalConfig:
    """Satisfies ModelConfig (has ``devices``)."""

    @property
    def devices(self) -> list[Device]:
        return []


class _KVConfig(_MinimalConfig):
    """Satisfies ModelConfigWithKVCache (adds ``get_kv_params``)."""

    def get_kv_params(self) -> KVCacheParams:
        return MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=8,
            head_dim=128,
            num_layers=1,
            page_size=128,
            data_parallel_degree=1,
            devices=[DeviceRef.CPU()],
            kvcache_quant_config=KVCacheQuantizationConfig(),
        )


class _BadConfig:
    """Does NOT satisfy ModelConfigWithKVCache (no ``get_kv_params``)."""

    @property
    def devices(self) -> list[Device]:
        return []


# ---------------------------------------------------------------------------
# Protocol isinstance checks
# ---------------------------------------------------------------------------


def test_minimal_config_satisfies_model_config() -> None:
    assert isinstance(_MinimalConfig(), ModelConfig)


def test_kv_config_satisfies_model_config_with_kv_cache() -> None:
    assert isinstance(_KVConfig(), ModelConfigWithKVCache)


def test_bad_config_does_not_satisfy_model_config_with_kv_cache() -> None:
    assert not isinstance(_BadConfig(), ModelConfigWithKVCache)


# ---------------------------------------------------------------------------
# PagedMemoryPlanner
# ---------------------------------------------------------------------------


def test_paged_planner_rejects_non_kv_config() -> None:
    with pytest.raises(TypeError, match="ModelConfigWithKVCache"):
        PagedMemoryPlanner(_BadConfig())


def test_paged_planner_accepts_kv_config() -> None:
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner is not None


def test_paged_planner_estimate_vision_cache_entry_bytes_zero() -> None:
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner.estimate_vision_cache_entry_bytes(None) == 0


def test_paged_planner_estimate_activation_memory_zero_by_default() -> None:
    """Default estimate_activation_memory should return 0."""
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner.estimate_activation_memory(MagicMock(), MagicMock()) == 0


def test_paged_planner_infer_max_batch_size_none_by_default() -> None:
    """Default infer_max_batch_size defers to the framework inference."""
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner.infer_max_batch_size(MagicMock(), [], 0) is None


def test_with_activation_reservation_returns_correct_bytes() -> None:
    """with_activation_reservation should return the configured value."""
    reservation = 15 * 1024**3
    planner_cls = PagedMemoryPlanner.with_activation_reservation(reservation)
    planner = planner_cls(_KVConfig())
    assert (
        planner.estimate_activation_memory(MagicMock(), MagicMock())
        == reservation
    )


def test_paged_planner_vision_cache_row_spec_none() -> None:
    """Default hook means no vision cache (text-only architectures)."""
    planner = PagedMemoryPlanner(_KVConfig())
    assert planner.get_vision_cache_row_spec(None) is None


# ---------------------------------------------------------------------------
# Block-mode vision cache reservation
# ---------------------------------------------------------------------------


def _block_reserve(
    utilization: float,
    row_bytes: int,
    available_memory: int,
    n_devices: int = 1,
) -> tuple[int, VisionCachePlan, "PipelineRuntimeConfig"]:
    """Run _reserve_vision_cache_blocks against a real runtime config."""
    runtime = PipelineRuntimeConfig(
        experimental_vision_cache_utilization=utilization
    )
    pipeline_config = SimpleNamespace(runtime=runtime)
    total, plan = MemoryEstimator._reserve_vision_cache_blocks(
        cast("PipelineConfig", pipeline_config),
        (row_bytes, DType.uint8),
        available_memory,
        n_devices,
    )
    return total, plan, runtime


def test_reserve_vision_cache_blocks_rounds_to_whole_blocks() -> None:
    available = 1024**3
    total, plan, runtime = _block_reserve(
        utilization=0.001,
        row_bytes=10,
        available_memory=available,
    )
    block_bytes = 128 * 10
    assert total == (int(available * 0.001) // block_bytes) * block_bytes
    assert 0 < total <= available * 0.001
    assert plan.bytes_per_device == total
    assert (plan.hidden_size, plan.dtype) == (10, DType.uint8)
    # Block mode leaves the legacy entry knob untouched.
    assert runtime.max_vision_cache_entries == 256


def test_reserve_vision_cache_blocks_scales_with_pool() -> None:
    small, _, runtime_small = _block_reserve(
        utilization=0.5,
        row_bytes=8,
        available_memory=10 * 1024**2,
    )
    large, _, runtime_large = _block_reserve(
        utilization=0.5,
        row_bytes=8,
        available_memory=20 * 1024**2,
    )
    assert 0 < small <= 5 * 1024**2
    assert small < large <= 10 * 1024**2
    assert runtime_small.max_vision_cache_entries == 256
    assert runtime_large.max_vision_cache_entries == 256


def test_reserve_vision_cache_blocks_raises_when_no_block_fits() -> None:
    with pytest.raises(ValueError, match="too small to fit one"):
        _block_reserve(
            utilization=0.001,
            row_bytes=1024**2,
            available_memory=10 * 1024**2,
        )


def test_reserve_vision_cache_blocks_splits_budget_across_devices() -> None:
    available = 1024**3
    total, plan, _ = _block_reserve(
        utilization=0.001,
        row_bytes=10,
        available_memory=available,
        n_devices=2,
    )
    block_bytes = 128 * 10 * 2
    assert total == (int(available * 0.001) // block_bytes) * block_bytes
    assert plan.bytes_per_device == total // 2


# ---------------------------------------------------------------------------
# Entry-count vision cache reservation
# ---------------------------------------------------------------------------


def _entry_reserve(
    max_entries: int,
    per_entry_bytes: int,
    available_memory: int,
    n_devices: int = 1,
    row_spec: tuple[int, DType] | None = None,
    utilization: float = 0.0,
) -> tuple[int, VisionCachePlan | None, "PipelineRuntimeConfig"]:
    """Run _reserve_vision_cache_memory against a real runtime config."""
    runtime = PipelineRuntimeConfig(
        max_vision_cache_entries=max_entries,
        experimental_vision_cache_utilization=utilization,
    )
    pipeline_config = SimpleNamespace(runtime=runtime)
    planner = MagicMock()
    planner.estimate_vision_cache_entry_bytes.return_value = per_entry_bytes
    planner.get_vision_cache_row_spec.return_value = row_spec
    arch = SimpleNamespace(memory_planner=MagicMock(return_value=planner))
    total, plan = MemoryEstimator._reserve_vision_cache_memory(
        cast("PipelineConfig", pipeline_config),
        cast("MAXModelConfig", SimpleNamespace(huggingface_config=None)),
        available_memory,
        [cast(Device, MagicMock())] * n_devices,
        cast("ArchConfig", MagicMock()),
        arch=cast("SupportedArchitecture", arch),
    )
    return total, plan, runtime


def test_reserve_vision_cache_entries_full_request_fits() -> None:
    total, plan, runtime = _entry_reserve(
        max_entries=4,
        per_entry_bytes=1024,
        available_memory=1024**3,
    )
    assert total == 4 * 1024
    assert plan is None
    assert runtime.max_vision_cache_entries == 4


def test_reserve_vision_cache_entries_reduced_to_budget() -> None:
    # 20% of a 100 MiB pool fits only 20 of the 256 requested 1 MiB entries.
    per_entry = 1024**2
    total, plan, runtime = _entry_reserve(
        max_entries=256,
        per_entry_bytes=per_entry,
        available_memory=100 * 1024**2,
    )
    assert total == 20 * per_entry
    assert plan is None
    assert runtime.max_vision_cache_entries == 20


def test_reserve_vision_cache_entries_disabled_when_pool_too_small() -> None:
    per_entry = 1024**2
    total, plan, runtime = _entry_reserve(
        max_entries=8,
        per_entry_bytes=per_entry,
        available_memory=per_entry,
    )
    assert total == 0
    assert plan is None
    assert runtime.max_vision_cache_entries == 0


def test_reserve_vision_cache_memory_block_mode_returns_plan() -> None:
    total, plan, runtime = _entry_reserve(
        max_entries=256,
        per_entry_bytes=1024,
        available_memory=1024**3,
        row_spec=(10, DType.uint8),
        utilization=0.001,
    )
    assert plan is not None
    assert plan.bytes_per_device == total
    assert (plan.hidden_size, plan.dtype) == (10, DType.uint8)
    assert runtime.max_vision_cache_entries == 256
