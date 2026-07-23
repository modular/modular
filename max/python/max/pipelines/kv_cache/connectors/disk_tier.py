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

"""Disk-backed block storage for KV cache tiered offloading.

Provides a disk cache with async I/O and LRU eviction. Each block hash maps to
a single binary file containing all TP shards concatenated, stored under a
hex-named subdirectory keyed by the first byte of the hash so no single
directory grows unboundedly. Reads are prioritized over writes via a
priority-based thread pool.

Credits to LMCache for inspiring this design.
https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/local_disk_backend.py
"""

from __future__ import annotations

import itertools
import logging
import queue
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil
from max.profiler import Tracer

logger = logging.getLogger("max.pipelines")

_SENTINEL = None

# Block files are sharded across this many hex-named subdirectories (by the
# first byte of the block hash) so no single directory holds more than
# ~total/256 entries. A flat directory of ~1-2M files makes per-file
# open/create/unlink metadata operations slow on ext4/xfs; bucketing keeps each
# directory small. 256 == one byte, matching the two-hex bucket name.
_NUM_SHARD_BUCKETS = 256

# Internal type for items in the priority queue: (priority, count, fn, args, kwargs, future)
_WorkItem = tuple[
    int,
    int,
    Callable[..., Any] | None,
    tuple[Any, ...],
    dict[str, Any],
    Future[None] | None,
]


class PriorityExecutor:
    """Thread pool with priority-based job scheduling.

    Lower priority number = higher urgency. Reads (0) preempt deletes (1)
    preempt writes (2).
    Uses stdlib ``queue.PriorityQueue`` for ordering without asyncio overhead.
    Returns ``concurrent.futures.Future`` for compatibility with ``wait()``.
    """

    READ_PRIORITY = 0
    DELETE_PRIORITY = 1
    WRITE_PRIORITY = 2

    def __init__(self, num_workers: int = 4) -> None:
        self._queue: queue.PriorityQueue[_WorkItem] = queue.PriorityQueue()
        self._counter = itertools.count()
        self._workers: list[threading.Thread] = []

        # In-flight job count, so callers can drain the pool without retaining
        # completed Futures (which would pile up and slow cyclic GC).
        self._inflight = 0
        self._idle = threading.Condition()

        for i in range(num_workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)

    def submit(
        self, priority: int, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[None]:
        """Submit a callable with the given priority.

        Args:
            priority: Lower number = higher urgency.
            fn: Callable to execute on a worker thread.
            *args: Positional arguments for *fn*.
            **kwargs: Keyword arguments for *fn*.

        Returns:
            A Future that resolves when *fn* completes.
        """
        future: Future[None] = Future()
        with self._idle:
            self._inflight += 1
        self._queue.put(
            (priority, next(self._counter), fn, args, kwargs, future)
        )
        return future

    def _worker(self, worker_id: int) -> None:
        while True:
            item = self._queue.get()
            _, _, fn, args, kwargs, future = item
            if fn is _SENTINEL:
                break
            try:
                with Tracer(f"DiskWorker-{worker_id} running {fn.__name__}"):
                    result = fn(*args, **kwargs)
                future.set_result(result)  # type: ignore[union-attr]
            except Exception as e:
                future.set_exception(e)  # type: ignore[union-attr]
            finally:
                with self._idle:
                    self._inflight -= 1
                    if self._inflight == 0:
                        self._idle.notify_all()

    def wait_until_idle(self) -> None:
        """Block until every submitted job has finished (success or failure)."""
        with self._idle:
            while self._inflight > 0:
                self._idle.wait()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down all worker threads.

        Args:
            wait: If True, block until all workers have exited.
        """
        for _ in self._workers:
            self._queue.put((999, 0, _SENTINEL, (), {}, None))
        if wait:
            for t in self._workers:
                t.join()

    @property
    def inflight_disk_ops(self) -> int:
        """Number of submitted jobs not yet finished (queued or running)."""
        return self._inflight


class DiskTier:
    """Sharded disk cache for KV blocks.

    One file per block hash, bucketed into 256 hex-named subdirectories by the
    first byte of the hash. All TP shards are concatenated into a single file.
    Writes are async, reads return a Future.
    LRU eviction keeps disk usage within a configurable budget.

    The cache is ephemeral scratch: the index starts empty and the whole
    ``cache_dir`` tree is removed on :meth:`shutdown`. Blocks are never reloaded
    from a directory left behind by a previous process.
    """

    def __init__(
        self,
        cache_dir: str,
        block_nbytes: int,
        max_disk_size_bytes: int,
        num_workers: int = 16,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Pre-create the shard buckets so the write path can open files without
        # a per-write mkdir.
        for bucket in range(_NUM_SHARD_BUCKETS):
            (self._cache_dir / f"{bucket:02x}").mkdir(exist_ok=True)

        self._block_nbytes = block_nbytes
        self._max_disk_size_bytes = max_disk_size_bytes

        # LRU tracking: hashes that have been saved to disk
        # The value for the dict is ignored.
        self._saved_hashes: OrderedDict[bytes, None] = OrderedDict()

        # Thread safety for _saved_hashes, _pending_hashes
        self._lock = threading.Lock()

        # Hashes with in-flight writes (not yet on disk but "claimed")
        self._pending_hashes: set[bytes] = set()

        # Hashes evicted from the live index whose files have an in-flight
        # async unlink. A hash here blocks re-writes until the delete completes,
        # which avoids a write/delete race over the same content-addressed file.
        self._pending_deletes: set[bytes] = set()

        # Priority executor: reads preempt deletes preempt writes.
        self._executor = PriorityExecutor(num_workers=num_workers)

        available_bytes = psutil.disk_usage(str(self._cache_dir)).free
        if self._max_disk_size_bytes > available_bytes:
            raise RuntimeError(
                "disk_offload_max_gb requests "
                f"{self._max_disk_size_bytes / (1024**3):.1f} GiB at "
                f"{self._cache_dir} but only "
                f"{available_bytes / (1024**3):.1f} GiB is available. Reduce "
                "disk_offload_max_gb or free space on the target filesystem."
            )

    @property
    def num_blocks(self) -> int:
        """Total disk block capacity (max_disk_size_bytes / block_nbytes)."""
        if self._block_nbytes == 0:
            return 0
        return self._max_disk_size_bytes // self._block_nbytes

    @property
    def num_used_blocks(self) -> int:
        """Number of blocks currently saved on disk."""
        with self._lock:
            return len(self._saved_hashes)

    def contains(self, block_hash: bytes) -> bool:
        """Check if a block hash is saved on disk and eligible for cache hit.

        Note that block hashes that have active in-flight writes are not eligible
        for cache hit from the disk tier. Instead, the caller should serve the cache
        hit from the cpu tier instead.
        """
        with self._lock:
            return block_hash in self._saved_hashes

    def read_block_async(
        self,
        block_hash: bytes,
        dest: npt.NDArray[np.uint8],
    ) -> Future[None]:
        """Submit an async read from disk into *dest* numpy view.

        Args:
            block_hash: Hash of the block to read.
            dest: Numpy view into host tensor at the target bid.

        Returns:
            A Future that completes when dest is populated.
        """
        with self._lock:
            self._saved_hashes.move_to_end(block_hash)  # LRU touch

        return self._executor.submit(
            PriorityExecutor.READ_PRIORITY,
            self._read_block_sync,
            block_hash,
            dest,
        )

    def write_block_async(
        self,
        block_hash: bytes,
        src: npt.NDArray[np.uint8],
    ) -> Future[None] | None:
        """Submit an async write to disk.

        Returns the Future if a write was submitted, or None if the block
        is already on disk (or has an in-flight write).  The caller can
        use the Future to track when it is safe to release the source
        memory.

        Args:
            block_hash: Hash of the block to write.
            src: Numpy array of block data.

        Returns:
            A Future that completes when the write is done, or None.
        """
        with self._lock:
            if (
                block_hash in self._saved_hashes
                or block_hash in self._pending_hashes
                or block_hash in self._pending_deletes
            ):
                # Already on disk, mid-write, or mid-delete. Skipping a write
                # while a delete is in flight avoids racing the unlink against a
                # fresh create of the same file.
                return None

            # Reserve space by selecting LRU victims. Their files are unlinked
            # asynchronously (see below) so the calling thread never blocks on
            # filesystem metadata operations.
            evictions = self._select_evictions(self._block_nbytes)
            self._pending_hashes.add(block_hash)

        # Submit the unlinks off the lock and off the caller's thread. Deletes
        # preempt writes (DELETE_PRIORITY < WRITE_PRIORITY) so freed space is
        # reclaimed promptly.
        for evicted_hash, path in evictions:
            self._executor.submit(
                PriorityExecutor.DELETE_PRIORITY,
                self._delete_block_sync,
                evicted_hash,
                path,
            )

        future = self._executor.submit(
            PriorityExecutor.WRITE_PRIORITY,
            self._write_block_sync,
            block_hash,
            src,
        )
        return future

    def remove(self, block_hash: bytes) -> None:
        """Remove a block from disk."""
        with self._lock:
            if block_hash not in self._saved_hashes:
                return
            self._saved_hashes.pop(block_hash)

        self._hash_to_path(block_hash).unlink(missing_ok=True)

    def wait_for_writes(self) -> None:
        """Block until all in-flight disk I/O (writes, evictions, reads) completes."""
        self._executor.wait_until_idle()

    def shutdown(self) -> None:
        """Wait for pending writes, stop the executor, and delete the cache dir.

        The disk cache is ephemeral scratch, so the whole ``cache_dir`` tree is
        removed here. ``ignore_errors`` keeps a filesystem hiccup from turning a
        best-effort cleanup into a shutdown failure.
        """
        self.wait_for_writes()
        self._executor.shutdown(wait=True)
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def reset(self) -> None:
        """Clear all blocks from disk."""
        with self._lock:
            self._saved_hashes.clear()
            self._pending_hashes.clear()
            self._pending_deletes.clear()
        for path in self._cache_dir.glob("*/*.bin"):
            path.unlink(missing_ok=True)

    # -- sync I/O (runs on worker threads) --

    def _read_block_sync(
        self,
        block_hash: bytes,
        dest: npt.NDArray[np.uint8],
    ) -> None:
        """Reads a block from disk.

        This method is called on a worker thread and must not contain code that
        hogs the gil for any significant amount of time.
        """
        path = self._hash_to_path(block_hash)
        with open(path, "rb") as f:
            assert dest.data.contiguous
            n = f.readinto(dest.data)
            if n != dest.nbytes:
                raise OSError(f"Short read: got {n}, expected {dest.nbytes}")

    def _write_block_sync(
        self,
        block_hash: bytes,
        src: npt.NDArray[np.uint8],
    ) -> None:
        """Writes a block out to disk.

        This method is called on a worker thread and must not contain code that
        hogs the gil for any significant amount of time.
        """
        path = self._hash_to_path(block_hash)
        with open(path, "wb") as f:
            assert src.data.contiguous
            f.write(src.data)

        with self._lock:
            self._pending_hashes.discard(block_hash)
            self._saved_hashes[block_hash] = None

    # -- file paths --

    def _hash_to_path(self, block_hash: bytes) -> Path:
        bucket = f"{block_hash[0]:02x}"
        return self._cache_dir / bucket / f"{block_hash.hex()}.bin"

    # -- eviction --

    def _select_evictions(self, needed_bytes: int) -> list[tuple[bytes, Path]]:
        """Select LRU blocks to evict so *needed_bytes* fits.

        Removes the victims from the live index and marks them in
        ``_pending_deletes`` so their hashes can't be re-written until the
        unlink completes. Returns ``(block_hash, path)`` pairs whose files the
        caller must unlink off the lock and off its own thread (see
        ``_delete_block_sync``).

        Caller must hold ``self._lock``.
        """
        evictions: list[tuple[bytes, Path]] = []
        while (
            len(self._saved_hashes) * self._block_nbytes + needed_bytes
            > self._max_disk_size_bytes
        ):
            if not self._saved_hashes:
                logger.warning("Disk cache full, no blocks to evict")
                break
            evicted_hash = self._saved_hashes.popitem(last=False)[0]
            self._pending_deletes.add(evicted_hash)
            evictions.append((evicted_hash, self._hash_to_path(evicted_hash)))
        return evictions

    def _delete_block_sync(self, block_hash: bytes, path: Path) -> None:
        """Unlink an evicted block file on a worker thread.

        This method is called on a worker thread and must not contain code that
        hogs the gil for any significant amount of time.
        """
        try:
            path.unlink(missing_ok=True)
        finally:
            with self._lock:
                self._pending_deletes.discard(block_hash)

    @property
    def inflight_disk_ops(self) -> int:
        """Number of in-flight disk operations."""
        return self._executor.inflight_disk_ops
