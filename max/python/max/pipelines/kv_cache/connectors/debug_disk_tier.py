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

"""Disk-backed block storage for KV cache tiered offloading."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_pool import BlockPool

logger = logging.getLogger("max.pipelines")


class DebugDiskTier:
    def __init__(
        self,
        cache_dir: str,
        block_nbytes: int,
        max_disk_size_bytes: int,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._num_blocks = max_disk_size_bytes // block_nbytes

        self._block_pool = BlockPool(
            MemoryTier.MEMORY_TIER_DISK,
            self._num_blocks,
            enable_prefix_caching=True,
            enable_runtime_checks=False,
        )

    def contains(self, block_hash: int) -> bool:
        return block_hash in self._block_pool.prefix_cache

    def read_block(
        self,
        block_hash: int,
        dest: npt.NDArray[np.uint8],
    ) -> None:
        if not self.contains(block_hash):
            raise ValueError("Block not on disk")

        block = self._block_pool.prefix_cache[block_hash]
        self._block_pool.touch(block)

        path = self._hash_to_path(block_hash)
        with open(path, "rb") as f:
            assert dest.data.contiguous
            n = f.readinto(dest.data)
            if n != dest.nbytes:
                raise OSError(f"Short read: got {n}, expected {dest.nbytes}")

        self._block_pool.free_block(block)

    def write_block(
        self,
        block_hash: int,
        src: npt.NDArray[np.uint8],
    ) -> None:
        if block_hash in self._block_pool.prefix_cache:
            raise ValueError("Block already on disk")

        # Check if fits
        if self._block_pool.num_free_blocks == 0:
            raise ValueError("Too many blocks")

        block, evicted_hash = self._block_pool.alloc_block()

        # delete the evicted block from the disk
        if evicted_hash is not None:
            path = self._hash_to_path(evicted_hash)
            os.remove(path)

        path = self._hash_to_path(block_hash)

        with open(path, "wb") as f:
            assert src.data.contiguous
            f.write(src.data)

        self._block_pool.commit_into_prefix_cache(block_hash, block)
        self._block_pool.free_block(block)

    def _hash_to_path(self, block_hash: int) -> Path:
        return self._cache_dir / f"{block_hash:016x}.bin"

    # on exit, delete all files in the cache directory
    def __del__(self) -> None:
        for path in self._cache_dir.glob("*.bin"):
            path.unlink(missing_ok=True)

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_used_blocks(self) -> int:
        return len(self._block_pool.prefix_cache)
