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
"""Regression test for MLA graph-capture cache-length bucketing coverage.

Graph capture probes a set of cache lengths and captures one graph per distinct
resolved ``num_partitions``. At replay a runtime cache length is bucketed *up*
to the nearest probed length and the graph captured at that length is replayed.
This relies on ``num_partitions`` being monotonic non-decreasing in cache
length, so the bucketed (longer) probe always yields ``num_partitions >=`` the
runtime value -- a captured graph with enough partitions. These tests verify
that invariant against the real ``mo.mla.compute_dispatch_args.scalar`` op.
"""

from collections.abc import Sequence

import pytest
from max.graph import DeviceRef
from max.nn.kv_cache.utils import AttentionDispatchResolver

NUM_HEADS = 128
BATCH_SIZES = [1, 2, 4, 8, 16, 31, 32, 33, 63, 64, 65, 96, 128]
MAX_CACHE_LENGTH = 16384


def _resolve_np(
    resolver: AttentionDispatchResolver,
    batch_size: int,
    cache_length: int,
) -> int:
    """Returns num_partitions from the real Mojo dispatch kernel."""
    return resolver.resolve_attn_key(batch_size, 1, cache_length).num_partitions


def _bucket_cache_length(
    cache_length: int, probe_lengths_sorted: Sequence[int]
) -> int:
    """Rounds a cache length up to the smallest probed length >= it."""
    candidates = [p for p in probe_lengths_sorted if p >= cache_length]
    return min(candidates) if candidates else max(probe_lengths_sorted)


@pytest.fixture(scope="module")
def mla_resolver() -> AttentionDispatchResolver:
    """Builds an MLA dispatch resolver backed by the real custom op."""
    device = DeviceRef.GPU()
    return AttentionDispatchResolver(
        devices=[device],
        is_mla=True,
        n_kv_heads_per_device=1,
        num_q_heads_per_device=NUM_HEADS // 1,
    )


@pytest.fixture(scope="module")
def mla_resolver_fp8() -> AttentionDispatchResolver:
    """Builds an MLA dispatch resolver with ``is_fp8_kv=True``."""
    device = DeviceRef.GPU()
    return AttentionDispatchResolver(
        devices=[device],
        is_mla=True,
        n_kv_heads_per_device=1,
        num_q_heads_per_device=NUM_HEADS // 1,
        is_fp8_kv=True,
    )


def _assert_bucketing_covers(resolver: AttentionDispatchResolver) -> None:
    """For every cache length, the bucketed probe covers its num_partitions."""
    probe_lengths = sorted(set(resolver.probe_lengths(MAX_CACHE_LENGTH)))
    for batch_size in BATCH_SIZES:
        probe_np = {
            length: _resolve_np(resolver, batch_size, length)
            for length in probe_lengths
        }
        for cache_length in range(1, MAX_CACHE_LENGTH + 1):
            runtime_np = _resolve_np(resolver, batch_size, cache_length)
            bucketed = _bucket_cache_length(cache_length, probe_lengths)
            assert probe_np[bucketed] >= runtime_np, (
                f"batch_size={batch_size}, cache_length={cache_length}: "
                f"bucketed probe {bucketed} has num_partitions "
                f"{probe_np[bucketed]} < runtime num_partitions {runtime_np}."
            )


def test_mla_bucketing_coverage(
    mla_resolver: AttentionDispatchResolver,
) -> None:
    """Bucketing a cache length up always reaches a graph with enough np."""
    _assert_bucketing_covers(mla_resolver)


def test_mla_bucketing_coverage_fp8(
    mla_resolver_fp8: AttentionDispatchResolver,
) -> None:
    """Coverage holds when ``is_fp8_kv=True`` (different np thresholds)."""
    _assert_bucketing_covers(mla_resolver_fp8)


def test_fp8_produces_fewer_partitions(
    mla_resolver: AttentionDispatchResolver,
    mla_resolver_fp8: AttentionDispatchResolver,
) -> None:
    """FP8 never produces more partitions than BF16 (``np_fp8 <= np_bf16``).

    FP8 uses higher ``min_pages_per_split`` thresholds and a doubled
    ``_np1_cache_threshold``, both of which can only reduce (never increase)
    the partition count relative to BF16.  This test sweeps a broad range of
    ``(batch_size, cache_length)`` combinations to verify the invariant.
    """
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 96, 128]:
        for cl in [
            1,
            64,
            128,
            256,
            300,
            384,
            450,
            511,
            512,
            1024,
            2048,
            2176,
            2177,
            3000,
            4096,
            8192,
            16384,
        ]:
            np_bf16 = _resolve_np(mla_resolver, batch_size, cl)
            np_fp8 = _resolve_np(mla_resolver_fp8, batch_size, cl)
            assert np_fp8 <= np_bf16, (
                f"FP8 produced more partitions than BF16: "
                f"bs={batch_size}, cl={cl}, np_fp8={np_fp8}, "
                f"np_bf16={np_bf16}"
            )
