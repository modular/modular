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

"""Regression tests for per-batch KV connector metric reset (MXSERV-203)."""

from __future__ import annotations

from collections.abc import Sequence

from max.nn.kv_cache import KVHashAlgo
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.connectors.local_connector import LocalConnector
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.connectors.tiered_connector import TieredConnector
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager


class _CountingConnector:
    """Minimal connector stub with mutable transfer counters."""

    def __init__(self) -> None:
        self._h2d_blocks_copied = 0
        self._d2h_blocks_copied = 0
        self._disk_blocks_written = 0
        self._disk_blocks_read = 0

    @property
    def name(self) -> str:
        return "counting"

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        return frozenset({"ahash64", "sha256", "sha256_64"})

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        return 0

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> None:
        pass

    def wait_for_loads(self) -> None: ...
    def wait_for_offloads(self) -> None: ...
    def shutdown(self) -> None: ...
    def reset_prefix_cache(self) -> None: ...

    @property
    def num_host_blocks(self) -> int:
        return 0

    @property
    def num_used_host_blocks(self) -> int:
        return 0

    @property
    def num_disk_blocks(self) -> int:
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        return 0

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
            disk_blocks_written=self._disk_blocks_written,
            disk_blocks_read=self._disk_blocks_read,
        )

    def reset_metrics(self) -> None:
        self._h2d_blocks_copied = 0
        self._d2h_blocks_copied = 0
        self._disk_blocks_written = 0
        self._disk_blocks_read = 0


def test_block_manager_reset_metrics_clears_connector_counters() -> None:
    connector = _CountingConnector()
    connector._d2h_blocks_copied = 5
    connector._h2d_blocks_copied = 2
    bm = BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=64,
        block_size=16,
        enable_prefix_caching=True,
        connector=connector,
    )

    assert bm.metrics.d2h_blocks_copied == 5
    assert bm.metrics.h2d_blocks_copied == 2

    bm.reset_metrics()

    assert bm.metrics.d2h_blocks_copied == 0
    assert bm.metrics.h2d_blocks_copied == 0

    connector._d2h_blocks_copied = 3
    assert bm.metrics.d2h_blocks_copied == 3


def test_local_connector_reset_metrics() -> None:
    connector = LocalConnector.__new__(LocalConnector)
    connector._h2d_blocks_copied = 4
    connector._d2h_blocks_copied = 7

    connector.reset_metrics()

    assert connector.metrics.h2d_blocks_copied == 0
    assert connector.metrics.d2h_blocks_copied == 0


def test_tiered_connector_reset_metrics() -> None:
    connector = TieredConnector.__new__(TieredConnector)
    connector._h2d_blocks_copied = 1
    connector._d2h_blocks_copied = 2
    connector._disk_blocks_written = 3
    connector._disk_blocks_read = 4

    connector.reset_metrics()

    assert connector._h2d_blocks_copied == 0
    assert connector._d2h_blocks_copied == 0
    assert connector._disk_blocks_written == 0
    assert connector._disk_blocks_read == 0


def test_scheduler_sampling_cycle_reports_per_batch_deltas() -> None:
    """Models BatchMetrics.create(): sample aggregated metrics, then reset.

    Before MXSERV-203, batch 2 would report 8 (5+3 cumulative) because
    connector counters were never reset. Telemetry must emit 5 then 3.
    """
    connector = _CountingConnector()
    bm = BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=64,
        block_size=16,
        enable_prefix_caching=True,
        connector=connector,
    )

    def sample_and_reset(d2h_delta: int, h2d_delta: int) -> KVCacheMetrics:
        connector._d2h_blocks_copied += d2h_delta
        connector._h2d_blocks_copied += h2d_delta
        sampled = bm.metrics
        bm.reset_metrics()
        return sampled

    batch_one = sample_and_reset(d2h_delta=5, h2d_delta=2)
    batch_two = sample_and_reset(d2h_delta=3, h2d_delta=0)

    assert batch_one.d2h_blocks_copied == 5
    assert batch_one.h2d_blocks_copied == 2
    assert batch_two.d2h_blocks_copied == 3
    assert batch_two.h2d_blocks_copied == 0

    # OTEL counter.add() with these per-batch samples totals 8, not 23.
    otel_counter_total = (
        batch_one.d2h_blocks_copied + batch_two.d2h_blocks_copied
    )
    assert otel_counter_total == 8
    assert otel_counter_total != 5 + 8  # pre-fix cumulative double-count


def test_null_connector_reset_metrics_is_noop() -> None:
    NullConnector().reset_metrics()
