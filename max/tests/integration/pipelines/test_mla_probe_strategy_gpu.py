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
"""Regression test for MLAProbeStrategy coverage.

For representative (batch_size, cache_length) combinations, verifies that the
probing strategy discovers every reachable num_partitions bucket using the real
``mo.mla.compute_dispatch_args.scalar`` custom op.
"""

import pytest
from max.graph import DeviceRef
from max.nn.kv_cache.utils import AttentionDispatchResolver
from max.pipelines.lib.graph_capture import MLAProbeStrategy

NUM_HEADS = 128
BATCH_SIZES = [1, 2, 4, 8, 16, 31, 32, 33, 63, 64, 65, 96, 128]


def _resolve_np(
    resolver: AttentionDispatchResolver,
    batch_size: int,
    cache_length: int,
) -> int:
    """Returns num_partitions from the real Mojo dispatch kernel."""
    metadata = resolver(batch_size, 1, cache_length).to_numpy()
    return int(metadata[2])


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


def test_mla_probe_coverage(
    mla_resolver: AttentionDispatchResolver,
) -> None:
    """Every reachable num_partitions must be covered by the probe strategy.

    For each representative batch size, we:
    1. Sweep all cache lengths [1..16384] to find reachable num_partitions.
    2. Run the MLAProbeStrategy to find probed (captured) num_partitions.
    3. Verify that every reachable np can be bucketed to a captured np.
    """
    strategy = MLAProbeStrategy()
    max_cache_length = 16384

    for batch_size in BATCH_SIZES:
        # 1. Find all reachable np values by sweeping cache lengths.
        reachable_nps: set[int] = set()
        for cl in range(1, max_cache_length + 1):
            reachable_nps.add(_resolve_np(mla_resolver, batch_size, cl))

        # 2. Simulate what the probe strategy would capture.
        probe_lengths = strategy.probe_lengths(max_cache_length)
        captured_nps: set[int] = set()
        for cl in probe_lengths:
            captured_nps.add(_resolve_np(mla_resolver, batch_size, cl))
        captured_nps_sorted = sorted(captured_nps)

        # 3. Every reachable np must be bucketable to some captured np.
        for runtime_np in sorted(reachable_nps):
            bucketed = strategy.bucket_num_partitions(
                runtime_np, captured_nps_sorted
            )
            assert bucketed is not None, (
                f"batch_size={batch_size}: runtime num_partitions={runtime_np} "
                f"cannot be bucketed. captured={captured_nps_sorted}, "
                f"reachable={sorted(reachable_nps)}"
            )


def test_mla_probe_coverage_negative(
    mla_resolver: AttentionDispatchResolver,
) -> None:
    """Without bucket_num_partitions, granularity=256 misses np values.

    At granularity=256, probing misses np=2 for many batch sizes.
    """
    max_cache_length = 16384

    class CoarseMLAProbeStrategy(MLAProbeStrategy):
        granularity = 256

    strategy = CoarseMLAProbeStrategy()

    for batch_size in BATCH_SIZES:
        reachable_nps: set[int] = set()
        for cl in range(1, max_cache_length + 1):
            reachable_nps.add(_resolve_np(mla_resolver, batch_size, cl))

        probe_lengths = strategy.probe_lengths(max_cache_length)
        captured_nps: set[int] = set()
        for cl in probe_lengths:
            captured_nps.add(_resolve_np(mla_resolver, batch_size, cl))

        # Exact match (no bucketing) — should find missing np values.
        missing = reachable_nps - captured_nps
        if missing:
            return

    pytest.fail(
        "Expected granularity=256 without bucketing to miss some "
        "num_partitions, but all were covered — the test harness may "
        "be broken."
    )


def test_mla_probe_coverage_fp8(
    mla_resolver_fp8: AttentionDispatchResolver,
) -> None:
    """Probe coverage holds when ``is_fp8_kv=True``.

    FP8 doubles the ``min_pages_per_split`` thresholds, which changes
    reachable num_partitions values.
    """
    strategy = MLAProbeStrategy()
    max_cache_length = 16384

    for batch_size in BATCH_SIZES:
        reachable_nps: set[int] = set()
        for cl in range(1, max_cache_length + 1):
            reachable_nps.add(_resolve_np(mla_resolver_fp8, batch_size, cl))

        probe_lengths = strategy.probe_lengths(max_cache_length)
        captured_nps: set[int] = set()
        for cl in probe_lengths:
            captured_nps.add(_resolve_np(mla_resolver_fp8, batch_size, cl))
        captured_nps_sorted = sorted(captured_nps)

        for runtime_np in sorted(reachable_nps):
            bucketed = strategy.bucket_num_partitions(
                runtime_np, captured_nps_sorted
            )
            assert bucketed is not None, (
                f"batch_size={batch_size}: runtime "
                f"num_partitions={runtime_np} cannot be bucketed. "
                f"captured={captured_nps_sorted}, "
                f"reachable={sorted(reachable_nps)}"
            )


def test_fp8_produces_fewer_partitions(
    mla_resolver: AttentionDispatchResolver,
    mla_resolver_fp8: AttentionDispatchResolver,
) -> None:
    """FP8 never produces more partitions than BF16 (``np_fp8 <= np_bf16``).

    FP8 uses higher ``min_pages_per_split`` thresholds and a doubled
    ``_np1_cache_threshold``, both of which can only reduce (never increase)
    the partition count relative to BF16.  This test sweeps a broad range of
    ``(batch_size, cache_length)`` combinations to verify the invariant.

    The dispatch optimizations (``batch_size >= 64 and
    effective_max_cache_len <= 2176 -> min_partitions=1``) may cause FP8 and
    BF16 to agree on most short-cache configs; the invariant ``<=`` still
    holds.
    """
    # Broad sweep: FP8 should never produce *more* partitions than BF16.
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
