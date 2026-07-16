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


def test_with_activation_reservation_returns_correct_bytes() -> None:
    """with_activation_reservation should return the configured value."""
    reservation = 15 * 1024**3
    planner_cls = PagedMemoryPlanner.with_activation_reservation(reservation)
    planner = planner_cls(_KVConfig())
    assert (
        planner.estimate_activation_memory(MagicMock(), MagicMock())
        == reservation
    )
