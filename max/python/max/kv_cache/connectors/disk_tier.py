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

Provides a flat-file disk cache with async I/O and LRU eviction. Each block
hash maps to a single binary file containing all TP shards concatenated.
Reads are prioritized over writes via a priority-based thread pool.

Credits to LMCache for inspiring this design.
https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/local_disk_backend.py
"""

from __future__ import annotations

import itertools
import json
import logging
import mmap
import os
import queue
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("max.pipelines")

_SENTINEL = None  # poison pill for worker shutdown

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

    Lower priority number = higher urgency. Reads (0) preempt writes (2).
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
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
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
        self._queue.put(
            (priority, next(self._counter), fn, args, kwargs, future)
        )
        return future

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            _, _, fn, args, kwargs, future = item
            if fn is _SENTINEL:
                break
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)  # type: ignore[union-attr]
            except Exception as e:
                future.set_exception(e)  # type: ignore[union-attr]

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


class DiskTier:
    """Flat-file disk cache for KV blocks.

    One file per block hash. All TP shards are concatenated into a single
    file. Writes are async, reads return a Future.
    LRU eviction keeps disk usage within a configurable budget.
    Metadata is persisted for warm restarts across process lifetimes.
    """

    def __init__(
        self,
        cache_dir: str,
        block_nbytes: int,
        num_devices: int,
        max_disk_size_bytes: int,
        num_workers: int = 4,
        has_scales: bool = False,
        scale_block_nbytes: int = 0,
        use_direct_io: bool = False,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._block_nbytes = block_nbytes
        self._num_devices = num_devices
        self._max_disk_size_bytes = max_disk_size_bytes
        self._has_scales = has_scales
        self._scale_block_nbytes = scale_block_nbytes

        # Total bytes per block on disk (all TP shards concatenated)
        self._total_block_nbytes = block_nbytes * num_devices
        self._total_scale_nbytes = (
            scale_block_nbytes * num_devices if has_scales else 0
        )

        # Optional O_DIRECT for bypassing OS page cache.
        self._use_direct_io = use_direct_io
        self._fs_block_size = 4096
        if self._use_direct_io:
            if not hasattr(os, "O_DIRECT"):
                logger.warning(
                    "O_DIRECT not available on this platform, "
                    "falling back to buffered I/O"
                )
                self._use_direct_io = False
            else:
                stat = os.statvfs(str(self._cache_dir))
                self._fs_block_size = stat.f_bsize
                total = self._total_block_nbytes + self._total_scale_nbytes
                if total % self._fs_block_size != 0:
                    logger.warning(
                        "Block size (%d) not aligned to FS block size "
                        "(%d). Disabling O_DIRECT.",
                        total,
                        self._fs_block_size,
                    )
                    self._use_direct_io = False
                else:
                    logger.info(
                        "O_DIRECT enabled for disk cache at %s "
                        "(FS block size=%d)",
                        self._cache_dir,
                        self._fs_block_size,
                    )

        # Pre-allocated aligned buffer for O_DIRECT (one per worker thread).
        # mmap anonymous regions are always page-aligned, satisfying
        # O_DIRECT's buffer alignment requirement.  Allocated lazily in
        # _get_aligned_buf() on first use per thread.
        self._aligned_buf_size = (
            max(self._total_block_nbytes, self._total_scale_nbytes)
            if self._use_direct_io
            else 0
        )
        self._tls = threading.local()

        # LRU tracking: hash -> file size on disk
        self._hash_to_size: OrderedDict[int, int] = OrderedDict()
        self._total_bytes_used: int = 0

        # Thread safety for _hash_to_size, _total_bytes_used, _pending_hashes
        self._lock = threading.Lock()

        # Hashes with in-flight writes (not yet on disk but "claimed")
        self._pending_hashes: set[int] = set()

        # Priority executor: reads preempt writes
        self._executor = PriorityExecutor(num_workers=num_workers)
        self._write_futures: list[Future[None]] = []

        # Save metadata every N completed writes
        self._writes_since_metadata_save: int = 0
        self._metadata_save_interval: int = 32

        # Rebuild from persisted metadata if available
        self._load_existing()

    def contains(self, block_hash: int) -> bool:
        """Check if a block hash exists on disk or has an in-flight write."""
        with self._lock:
            return (
                block_hash in self._hash_to_size
                or block_hash in self._pending_hashes
            )

    def read_block_async(
        self,
        block_hash: int,
        dest: list[npt.NDArray[np.uint8]],
        scale_dest: list[npt.NDArray[np.uint8]] | None = None,
    ) -> Future[None]:
        """Submit an async read from disk into *dest* numpy views.

        Args:
            block_hash: Hash of the block to read.
            dest: Per-device numpy views into host tensors at the target bid.
            scale_dest: Per-device scale numpy views, or None.

        Returns:
            A Future that completes when dest (and scale_dest) are populated.
        """
        with self._lock:
            self._hash_to_size.move_to_end(block_hash)  # LRU touch

        return self._executor.submit(
            PriorityExecutor.READ_PRIORITY,
            self._read_block_sync,
            block_hash,
            dest,
            scale_dest,
        )

    def write_block_async(
        self,
        block_hash: int,
        src: list[npt.NDArray[np.uint8]],
        scale_src: list[npt.NDArray[np.uint8]] | None = None,
    ) -> Future[None] | None:
        """Submit an async write to disk.

        Returns the Future if a write was submitted, or None if the block
        is already on disk (or has an in-flight write).  The caller can
        use the Future to track when it is safe to release the source
        memory.

        Args:
            block_hash: Hash of the block to write.
            src: Per-device numpy arrays (views or copies) of block data.
            scale_src: Per-device scale numpy arrays, or None.

        Returns:
            A Future that completes when the write is done, or None.
        """
        with self._lock:
            if (
                block_hash in self._hash_to_size
                or block_hash in self._pending_hashes
            ):
                return None  # already on disk or write in progress

            # Reserve space, evicting LRU blocks if needed
            needed = self._total_block_nbytes + self._total_scale_nbytes
            self._evict_until_fits(needed)

            self._pending_hashes.add(block_hash)

        future = self._executor.submit(
            PriorityExecutor.WRITE_PRIORITY,
            self._write_block_sync,
            block_hash,
            src,
            scale_src,
        )
        self._write_futures.append(future)
        return future

    def remove(self, block_hash: int) -> None:
        """Remove a block from disk."""
        with self._lock:
            if block_hash not in self._hash_to_size:
                return
            size = self._hash_to_size.pop(block_hash)
            self._total_bytes_used -= size

        self._hash_to_path(block_hash).unlink(missing_ok=True)
        if self._has_scales:
            self._scale_path(block_hash).unlink(missing_ok=True)

    def wait_for_writes(self) -> None:
        """Block until all pending async writes complete."""
        for f in self._write_futures:
            f.result()
        self._write_futures.clear()

    def shutdown(self) -> None:
        """Wait for pending writes, save metadata, and shut down executor."""
        self.wait_for_writes()
        self._save_metadata()
        self._executor.shutdown(wait=True)

    def reset(self) -> None:
        """Clear all blocks from disk."""
        with self._lock:
            self._hash_to_size.clear()
            self._total_bytes_used = 0
            self._pending_hashes.clear()
        for path in self._cache_dir.glob("*.bin"):
            path.unlink(missing_ok=True)
        meta_path = self._cache_dir / "_metadata.json"
        meta_path.unlink(missing_ok=True)

    # -- sync I/O (runs on worker threads) --

    def _read_block_sync(
        self,
        block_hash: int,
        dest: list[npt.NDArray[np.uint8]],
        scale_dest: list[npt.NDArray[np.uint8]] | None,
    ) -> None:
        path = self._hash_to_path(block_hash)
        if self._use_direct_io:
            self._read_file_direct(path, dest)
        else:
            with open(path, "rb") as f:
                for arr in dest:
                    f.readinto(arr)  # type: ignore[arg-type]

        if self._has_scales and scale_dest is not None:
            scale_path = self._scale_path(block_hash)
            if self._use_direct_io:
                self._read_file_direct(scale_path, scale_dest)
            else:
                with open(scale_path, "rb") as f:
                    for arr in scale_dest:
                        f.readinto(arr)  # type: ignore[arg-type]

    def _write_block_sync(
        self,
        block_hash: int,
        src: list[npt.NDArray[np.uint8]],
        scale_src: list[npt.NDArray[np.uint8]] | None,
    ) -> None:
        path = self._hash_to_path(block_hash)
        if self._use_direct_io:
            self._write_file_direct(path, src)
        else:
            with open(path, "wb") as f:
                for arr in src:
                    f.write(arr.tobytes())

        total_size = self._total_block_nbytes

        if self._has_scales and scale_src is not None:
            scale_path = self._scale_path(block_hash)
            if self._use_direct_io:
                self._write_file_direct(scale_path, scale_src)
            else:
                with open(scale_path, "wb") as f:
                    for arr in scale_src:
                        f.write(arr.tobytes())
            total_size += self._total_scale_nbytes

        with self._lock:
            self._pending_hashes.discard(block_hash)
            self._hash_to_size[block_hash] = total_size
            self._total_bytes_used += total_size
            self._writes_since_metadata_save += 1
            should_save = (
                self._writes_since_metadata_save >= self._metadata_save_interval
            )
            if should_save:
                self._writes_since_metadata_save = 0

        # Periodically persist metadata so a crash loses at most
        # ~64 blocks instead of everything (no clean shutdown required).
        if should_save:
            self._save_metadata()

    # -- O_DIRECT helpers --

    def _get_aligned_buf(self) -> mmap.mmap:
        """Return a thread-local page-aligned staging buffer for O_DIRECT.

        Allocated once per worker thread via ``threading.local()`` and
        reused for every subsequent I/O operation — no per-call allocation.
        """
        buf = getattr(self._tls, "aligned_buf", None)
        if buf is None or buf.closed:
            buf = mmap.mmap(-1, self._aligned_buf_size)
            self._tls.aligned_buf = buf
        return buf

    def _write_file_direct(
        self, path: Path, arrays: list[npt.NDArray[np.uint8]]
    ) -> None:
        """Write *arrays* to *path* using O_DIRECT (bypass OS page cache).

        Concatenates all arrays into the pre-allocated page-aligned staging
        buffer, then issues a single ``os.write()`` via a ``memoryview``
        slice — one syscall, aligned pointer, no per-call allocation.
        """
        buf = self._get_aligned_buf()
        buf.seek(0)
        for arr in arrays:
            buf.write(arr.tobytes())
        total = buf.tell()

        fd = os.open(
            str(path),
            os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_DIRECT,
            0o644,
        )
        try:
            # memoryview is zero-copy — preserves the mmap's page-aligned
            # address so os.write sees an aligned buffer.
            os.write(fd, memoryview(buf)[:total])
        finally:
            os.close(fd)

    def _read_file_direct(
        self, path: Path, arrays: list[npt.NDArray[np.uint8]]
    ) -> None:
        """Read into *arrays* from *path* using O_DIRECT.

        Single ``os.read()`` into the pre-allocated page-aligned buffer,
        then scatter into the target numpy arrays — one syscall, no
        per-call allocation.
        """
        total = sum(a.nbytes for a in arrays)
        buf = self._get_aligned_buf()

        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECT)
        try:
            # Read the whole file in one aligned read.
            n = os.readv(fd, [memoryview(buf)[:total]])
            if n != total:
                raise OSError(f"Short O_DIRECT read: got {n}, expected {total}")
        finally:
            os.close(fd)

        # Scatter the staging buffer into destination arrays.
        offset = 0
        for arr in arrays:
            nbytes = arr.nbytes
            arr.flat[:] = np.frombuffer(
                buf[offset : offset + nbytes], dtype=arr.dtype
            )
            offset += nbytes

    # -- file paths --

    def _hash_to_path(self, block_hash: int) -> Path:
        return self._cache_dir / f"{block_hash:016x}.bin"

    def _scale_path(self, block_hash: int) -> Path:
        return self._cache_dir / f"{block_hash:016x}.scale.bin"

    # -- eviction --

    def _evict_until_fits(self, needed_bytes: int) -> None:
        """Evict LRU blocks until *needed_bytes* can be accommodated.

        Caller must hold ``self._lock``.
        """
        while self._total_bytes_used + needed_bytes > self._max_disk_size_bytes:
            if not self._hash_to_size:
                logger.warning("Disk cache full, no blocks to evict")
                return
            evicted_hash, evicted_size = self._hash_to_size.popitem(last=False)
            self._total_bytes_used -= evicted_size
            self._hash_to_path(evicted_hash).unlink(missing_ok=True)
            if self._has_scales:
                self._scale_path(evicted_hash).unlink(missing_ok=True)

    # -- persistence --

    def _save_metadata(self) -> None:
        """Persist hash index to disk for warm restarts.

        Uses write-to-temp + atomic rename to avoid corrupt JSON if
        the process crashes mid-write.
        """
        with self._lock:
            meta = {
                "block_nbytes": self._block_nbytes,
                "num_devices": self._num_devices,
                "hashes": list(self._hash_to_size.keys()),
            }
        tmp_path = self._cache_dir / "_metadata.json.tmp"
        final_path = self._cache_dir / "_metadata.json"
        tmp_path.write_text(json.dumps(meta))
        os.replace(tmp_path, final_path)

    def _load_existing(self) -> None:
        """Rebuild index from persisted metadata if compatible."""
        meta_path = self._cache_dir / "_metadata.json"
        if not meta_path.exists():
            logger.info("Disk cache cold start: no metadata at %s", meta_path)
            return
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt disk cache metadata, starting fresh")
            self.reset()
            return

        if (
            meta.get("block_nbytes") != self._block_nbytes
            or meta.get("num_devices") != self._num_devices
        ):
            logger.info("Disk cache config mismatch, clearing stale cache")
            self.reset()
            return

        for h in meta.get("hashes", []):
            path = self._hash_to_path(h)
            if path.exists():
                file_size = path.stat().st_size
                self._hash_to_size[h] = file_size
                self._total_bytes_used += file_size

        if self._hash_to_size:
            logger.info(
                "Disk cache warm start: loaded %d blocks (%.1f GB) from %s",
                len(self._hash_to_size),
                self._total_bytes_used / (1024**3),
                self._cache_dir,
            )
        else:
            logger.info(
                "Disk cache cold start: metadata found but no valid "
                "block files in %s",
                self._cache_dir,
            )
