# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Benchmarks for DiskTier write and read throughput at various block sizes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from max.kv_cache.connectors.disk_tier import DiskTier
from pytest_benchmark.fixture import BenchmarkFixture

ITERATIONS = 5
WARMUP_ROUNDS = 1
MIB = 1024 * 1024
BYTES = 4 * MIB  # Kimi 128 token page size is ~4MiB


@pytest.mark.parametrize("use_direct_io", [False, True])
@pytest.mark.parametrize("num_workers", [4, 8])
@pytest.mark.parametrize("batch_size", [512, 2048])  # 75k tokens is ~585 pages
def test_benchmark_batch_write(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    batch_size: int,
    num_workers: int,
    use_direct_io: bool,
) -> None:
    """Measure throughput of submitting a batch of writes concurrently."""
    total_batches = ITERATIONS + WARMUP_ROUNDS + 1
    total_blocks = total_batches * batch_size

    # Reuse a small pool of blocks to avoid allocating total_blocks x 4 MiB
    # of memory.  The write path doesn't depend on block contents.
    pool_size = min(batch_size, 64)
    block_pool = [np.zeros(BYTES, dtype=np.uint8) for _ in range(pool_size)]
    hash_counter = iter(range(total_blocks))

    tier = DiskTier(
        cache_dir=str(tmp_path / "bench_batch_write"),
        block_nbytes=BYTES,
        max_disk_size_bytes=total_blocks * BYTES,
        num_workers=num_workers,
        use_direct_io=use_direct_io,
    )

    def write_batch() -> None:
        for i in range(batch_size):
            h = next(hash_counter)
            tier.write_block_async(block_hash=h, src=block_pool[i % pool_size])
        tier.wait_for_writes()

    benchmark.pedantic(
        write_batch,
        rounds=ITERATIONS,
        iterations=1,
        warmup_rounds=WARMUP_ROUNDS,
    )

    tier.shutdown()
