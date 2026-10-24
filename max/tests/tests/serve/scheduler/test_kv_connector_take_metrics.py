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

"""Tests for KV connector metric accumulation (MXSERV-203)."""

from __future__ import annotations

from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.connectors.null_connector import NullConnector


def test_null_connector_take_metrics_is_noop() -> None:
    assert NullConnector().take_metrics() == KVCacheMetrics()


def test_kv_cache_metrics_add_sums_dkv_health_fields() -> None:
    """The dKV health fields sum across per-replica connector clients."""
    connected = KVCacheMetrics(
        dkv_connected_clients=1,
        dkv_total_clients=1,
        dkv_reconnect_attempts=2,
    )
    down = KVCacheMetrics(
        dkv_connected_clients=0,
        dkv_total_clients=1,
        dkv_reconnect_attempts=5,
    )

    total = connected + down

    assert total.dkv_connected_clients == 1
    assert total.dkv_total_clients == 2
    assert total.dkv_reconnect_attempts == 7
    # one of two clients down, so the aggregate is degraded
    assert total.dkv_degraded


def test_kv_cache_metrics_add_folds_the_dkv_settle_fields() -> None:
    """The settle depth and finished count sum, the age and the halt take the max.

    Two settle workers' oldest jobs are still one job's wait each, so adding
    them would report a wait neither saw, and two halts are still one flag.
    """
    halted = KVCacheMetrics(
        dkv_settle_queue_depth=3,
        dkv_settle_queue_oldest_age_ms=250.0,
        dkv_settle_jobs_finished=10,
        dkv_owner_releases_halted=1,
    )
    draining = KVCacheMetrics(
        dkv_settle_queue_depth=4,
        dkv_settle_queue_oldest_age_ms=40.0,
        dkv_settle_jobs_finished=7,
        dkv_owner_releases_halted=0,
    )

    for total in (halted + draining, draining + halted):
        assert total.dkv_settle_queue_depth == 7
        assert total.dkv_settle_queue_oldest_age_ms == 250.0
        assert total.dkv_settle_jobs_finished == 17
        assert total.dkv_owner_releases_halted == 1

    # two halts up are still one flag, not a count of them
    assert (halted + halted).dkv_owner_releases_halted == 1
    assert KVCacheMetrics().dkv_settle_queue_oldest_age_ms == 0.0


def test_kv_cache_metrics_add_sums_tier_attribution_fields() -> None:
    """device_blocks_served and cross_replica_{blocks,bytes}_copied sum
    across replicas (CENG-845 G0/G0-DP tier-attribution counters)."""
    replica_0 = KVCacheMetrics(
        device_blocks_served=5,
        cross_replica_blocks_copied=2,
        cross_replica_bytes_copied=1024,
        h2d_bytes_copied=10,
        disk_bytes_read=3,
    )
    replica_1 = KVCacheMetrics(
        device_blocks_served=7,
        cross_replica_blocks_copied=1,
        cross_replica_bytes_copied=512,
        h2d_bytes_copied=4,
        disk_bytes_read=1,
    )

    total = replica_0 + replica_1

    assert total.device_blocks_served == 12
    assert total.cross_replica_blocks_copied == 3
    assert total.cross_replica_bytes_copied == 1536
    assert total.h2d_bytes_copied == 14
    assert total.disk_bytes_read == 4
    # defaults stay zero
    assert KVCacheMetrics().device_blocks_served == 0
    assert KVCacheMetrics().cross_replica_bytes_copied == 0


def test_kv_cache_metrics_add_maxes_the_read_latency_peak() -> None:
    """The READ latency peak takes the max across replicas, never the sum.

    The slowest read two replicas saw is still one read. Adding their peaks
    would report a latency neither replica measured, and the pooled figure
    would grow with the replica count rather than with the tail.
    """
    replica_0 = KVCacheMetrics(
        nixl_read_latency_total_ms=2.0,
        nixl_read_latency_count=10,
        nixl_read_latency_max_ms=16.4,
    )
    replica_1 = KVCacheMetrics(
        nixl_read_latency_total_ms=1.0,
        nixl_read_latency_count=5,
        nixl_read_latency_max_ms=0.3,
    )

    total = replica_0 + replica_1

    # The pair beside it still sums, so the average is over both replicas.
    assert total.nixl_read_latency_total_ms == 3.0
    assert total.nixl_read_latency_count == 15
    assert total.nixl_read_latency_max_ms == 16.4
    # Order must not matter, which a sum would also satisfy but an assign
    # would not.
    assert (replica_1 + replica_0).nixl_read_latency_max_ms == 16.4
    assert KVCacheMetrics().nixl_read_latency_max_ms == 0.0


def test_kv_cache_metrics_dkv_degraded_predicate() -> None:
    """dkv_degraded is true only with a dKV tier and a client not connected."""
    # no dKV tier attached
    assert not KVCacheMetrics().dkv_degraded
    # every client connected
    assert not KVCacheMetrics(
        dkv_connected_clients=2, dkv_total_clients=2
    ).dkv_degraded
    # a client down
    assert KVCacheMetrics(
        dkv_connected_clients=1, dkv_total_clients=2
    ).dkv_degraded
