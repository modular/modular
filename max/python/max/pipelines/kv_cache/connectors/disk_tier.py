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
import json
import logging
import mmap
import os
import queue
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max.nn.kv_cache.cache_params import KVHashAlgo
from max.profiler import Tracer

logger = logging.getLogger("max.pipelines")

_META_FILE = "kv-disk-cache.meta.json"

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
    The cached set is the ``*.bin`` files themselves: a warm start rebuilds the
    in-memory index by scanning the shard subdirectories, so no metadata is
    persisted and the write path keeps no index in sync. A cache directory
    written by an older flat-layout build is treated as a cold start.

    ``block_nbytes`` is assumed constant for a given ``cache_dir``. Reusing a
    directory across a block-size change (page size, model, dtype, or TP degree)
    is unsupported: the stale, wrong-sized blocks are not detected and may be
    read as valid data. Point a changed configuration at a fresh ``cache_dir``.

    The `use_direct_io` flag controls whether the OS page cache is bypassed.
    This should be turned on for better performance if most local CPU memory is
    consumed (ie: for CPU KVCache). In other cases, leaving this off will yield
    better performance as reads and writes will be buffered by the OS page cache.
    """

    def __init__(
        self,
        cache_dir: str,
        block_nbytes: int,
        max_disk_size_bytes: int,
        kv_hash_algo: KVHashAlgo = "ahash64",
        num_workers: int = 16,
        use_direct_io: bool = False,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Pre-create the shard buckets so the write path can open files without
        # a per-write mkdir. Created before _load_existing scans them.
        for bucket in range(_NUM_SHARD_BUCKETS):
            (self._cache_dir / f"{bucket:02x}").mkdir(exist_ok=True)

        self._block_nbytes = block_nbytes
        self._max_disk_size_bytes = max_disk_size_bytes

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
                total = self._block_nbytes
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
                        "O_DIRECT enabled for disk cache at %s (FS block size=%d)",
                        self._cache_dir,
                        self._fs_block_size,
                    )

        # Pre-allocated aligned buffer for O_DIRECT (one per worker thread).
        # mmap anonymous regions are always page-aligned, satisfying
        # O_DIRECT's buffer alignment requirement.  Allocated lazily in
        # _get_aligned_buf() on first use per thread.
        self._aligned_buf_size = (
            self._block_nbytes if self._use_direct_io else 0
        )
        self._tls = threading.local()

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

        self._hash_algo: KVHashAlgo = kv_hash_algo
        self._verify_or_record_algo()

        self._load_existing()

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
        """Wait for pending writes and shut down the executor."""
        self.wait_for_writes()
        self._executor.shutdown(wait=True)

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
        if self._use_direct_io:
            self._read_file_direct(path, dest)
        else:
            with open(path, "rb") as f:
                assert dest.data.contiguous
                n = f.readinto(dest.data)
                if n != dest.nbytes:
                    raise OSError(
                        f"Short read: got {n}, expected {dest.nbytes}"
                    )

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
        if self._use_direct_io:
            self._write_file_direct(path, src)
        else:
            with open(path, "wb") as f:
                assert src.data.contiguous
                f.write(src.data)

        with self._lock:
            self._pending_hashes.discard(block_hash)
            self._saved_hashes[block_hash] = None

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
        self, path: Path, src: npt.NDArray[np.uint8]
    ) -> None:
        """Write src to path using O_DIRECT (bypass OS page cache).

        Issues a single ``os.write()`` via a ``memoryview``slice.
        """
        buf = self._get_aligned_buf()
        buf.seek(0)
        assert src.data.contiguous
        buf.write(src.data)
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
        self, path: Path, array: npt.NDArray[np.uint8]
    ) -> None:
        """Read into *array* from *path* using O_DIRECT.

        Single ``os.read()`` into the pre-allocated page-aligned buffer.
        """
        nbytes = array.nbytes
        buf = self._get_aligned_buf()
        assert len(buf) == nbytes
        assert array.data.contiguous

        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECT)
        try:
            # Read the whole file in one aligned read.
            n = os.readv(fd, [memoryview(buf)])
            if n != nbytes:
                raise OSError(
                    f"Short O_DIRECT read: got {n}, expected {nbytes}"
                )
        finally:
            os.close(fd)

        # np.copyto releases the GIL during the memcpy.
        # np.frombuffer on the mmap directly is zero-copy (buffer protocol);
        np.copyto(
            array.reshape(-1),
            np.frombuffer(buf, dtype=array.dtype),
        )

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

    # -- persistence --

    def _load_existing(self) -> None:
        """Rebuild the in-memory index by scanning the shard subdirectories.

        Blocks live at ``<xx>/<hex>.bin`` where ``<hex>`` is either a 16-char
        u64 (ahash64) or a 64-char SHA-256 digest. Other lengths are skipped
        with a warning so a mis-placed sidecar file does not abort warm start.
        Only the ``<xx>/`` bucket subdirectories are scanned, so a cache
        directory written by an older flat-layout build is treated as a cold
        start (its root-level files are not indexed); point a changed
        configuration at a fresh ``cache_dir``.
        """
        scanned = 0
        for bucket in os.scandir(self._cache_dir):
            if not bucket.is_dir():
                continue
            for entry in os.scandir(bucket.path):
                scanned += 1
                name = entry.name
                if not name.endswith(".bin"):
                    continue
                stem = name[:-4]
                if len(stem) not in (16, 64):
                    logger.warning(
                        "Skipping disk cache file with unexpected stem "
                        "length: %s",
                        name,
                    )
                    continue
                try:
                    block_hash = bytes.fromhex(stem)
                except ValueError:
                    continue
                self._saved_hashes[block_hash] = None

        if self._saved_hashes:
            logger.info(
                "Disk cache warm start: indexed %d blocks (%.1f GB) from "
                "%d files in %s",
                len(self._saved_hashes),
                len(self._saved_hashes) * self._block_nbytes / (1024**3),
                scanned,
                self._cache_dir,
            )
        else:
            logger.info(
                "Disk cache cold start at %s (scanned %d files)",
                self._cache_dir,
                scanned,
            )

    def _verify_or_record_algo(self) -> None:
        """Verify the on-disk cache hash algo matches ``self._hash_algo``.
        Maintains a ``kv-disk-cache.meta.json`` sidecar that pins the
        algorithm used for filenames in the cache directory.
        On startup:
            - If the meta file exists, compare its ``hash_algo`` to
              ``self._hash_algo`` and raise ``RuntimeError`` on mismatch
              with a clear remediation message.
            - If the meta file is missing but ``.bin`` files exist, infer
              the algo from the first filename's stem length: 64 hex chars
              must be ``sha256`` (32-byte digests). 16 hex chars are
              ambiguous between ``ahash64`` and ``sha256_64`` (both store
              64-bit ints) so we trust the configured algo. Refuse startup
              on a clear mismatch (e.g. 64-char stems with configured
              ``ahash64``).
            - If no meta and no ``.bin`` files exist, write a fresh meta
              file with the configured algo.
        """
        meta_path = self._cache_dir / _META_FILE
        if meta_path.exists():
            try:
                recorded = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Failed to read disk cache meta at {meta_path}: "
                    f"{exc}. Delete {self._cache_dir} and restart to "
                    "start fresh."
                ) from exc
            recorded_algo = recorded.get("hash_algo")
            if recorded_algo != self._hash_algo:
                raise RuntimeError(
                    f"Disk cache at {self._cache_dir} was created with "
                    f"hash_algo={recorded_algo!r}; current configuration "
                    f"requires {self._hash_algo!r}. Delete "
                    f"{self._cache_dir} and restart to start fresh."
                )
            return
        existing_stem_len: int | None = None
        for entry in os.scandir(self._cache_dir):
            if entry.name.endswith(".bin"):
                existing_stem_len = len(entry.name) - 4
                break
        if existing_stem_len == 64 and self._hash_algo != "sha256":
            raise RuntimeError(
                f"Disk cache at {self._cache_dir} contains SHA-256 files "
                f"(64-char stems) but current configuration requires "
                f"hash_algo={self._hash_algo!r}. Delete "
                f"{self._cache_dir} and restart to start fresh."
            )
        if existing_stem_len == 16 and self._hash_algo == "sha256":
            raise RuntimeError(
                f"Disk cache at {self._cache_dir} contains int-hash files "
                "(16-char stems) but current configuration requires "
                "hash_algo='sha256' (64-char stems). Delete "
                f"{self._cache_dir} and restart to start fresh."
            )
        payload = json.dumps(
            {"hash_algo": self._hash_algo}, separators=(",", ":")
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=meta_path.parent,
            prefix=".kv-disk-cache.meta.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        os.replace(tmp_path, meta_path)

    @property
    def inflight_disk_ops(self) -> int:
        """Number of in-flight disk operations."""
        return self._executor.inflight_disk_ops
