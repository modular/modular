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

"""Byte-budgeted cache for preprocessed vision inputs."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["VisionPreprocessCache"]

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    """One cached payload plus the host bytes it retains."""

    value: T
    nbytes: int


class VisionPreprocessCache(Generic[T]):
    """LRU cache of preprocessed image tensors, bounded by total bytes.

    Keyed on the same raw-encoded-bytes digest that
    :class:`~max.pipelines.lib.vision_encoder_cache.VisionEncoderCache` uses,
    so both caches hit and miss together for a given image.

    This sits *upstream* of the vision encoder cache: it is consulted in the
    tokenizer, before preprocessing, whereas the encoder cache is consulted in
    the model worker after preprocessing has already run. A hit therefore skips
    the resize, rescale and patchify -- work the encoder cache cannot avoid no
    matter how often it hits.

    The decode itself is not saved on the serving path, because the API server
    already decodes every image once at admission and hands the tokenizer the
    decoded image. Offline callers, which pass raw bytes through to the
    tokenizer, save the decode too.

    Bounded by bytes rather than by entry count (unlike
    :class:`~max.pipelines.lib.utils.BoundedCache`) because a preprocessed
    entry's size tracks the resized image area: a thumbnail and a full-budget
    image differ by more than an order of magnitude, so an entry count bounds
    host memory far too loosely to be a safe default.

    Args:
        max_bytes: Host-memory budget for cached payloads. ``0`` disables the
            cache, in which case :meth:`put` is a no-op and :meth:`get` always
            misses.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(0, max_bytes)
        self._cache: OrderedDict[int, _Entry[T]] = OrderedDict()
        self._total_bytes = 0
        # Preprocessing may be dispatched to worker threads, so guard the
        # ordered dict rather than relying on the caller running single
        # threaded under the event loop.
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled (``max_bytes > 0``)."""
        return self._max_bytes > 0

    @property
    def total_bytes(self) -> int:
        """Host bytes currently retained by cached payloads."""
        return self._total_bytes

    @property
    def hits(self) -> int:
        """Lookups served from the cache."""
        return self._hits

    @property
    def misses(self) -> int:
        """Lookups that had to preprocess."""
        return self._misses

    def __len__(self) -> int:
        return len(self._cache)

    def __getstate__(self) -> dict[str, int]:
        """Pickles as an empty cache, carrying only the budget.

        The tokenizer that owns this cache is pickled into the spawned model
        worker, because the pipeline factory captures it (see
        ``PIPELINE_REGISTRY.retrieve_factory``). A :class:`threading.Lock`
        cannot be pickled, so without this the whole server fails to start.

        Dropping the entries is not merely a workaround, it is the correct
        semantics: this cache is process-local. The worker preprocesses
        nothing -- it is handed already-preprocessed tensors -- so a copied
        entry would be dead weight there, and each process must own its own
        lock regardless.
        """
        return {"max_bytes": self._max_bytes}

    def __setstate__(self, state: dict[str, int]) -> None:
        """Restores an empty cache with a fresh lock in the new process."""
        self._max_bytes = state["max_bytes"]
        self._cache = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: int) -> T | None:
        """Look up a payload by image hash, refreshing LRU order."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: int, value: T, nbytes: int) -> None:
        """Insert a payload, evicting least-recently-used entries to fit.

        A payload larger than the whole budget is dropped rather than cached,
        so one oversized image cannot flush every useful entry.

        Args:
            key: The image hash to key on.
            value: The preprocessed payload to retain.
            nbytes: Host bytes ``value`` retains, used against the budget.
        """
        if not self.enabled or nbytes > self._max_bytes:
            return
        with self._lock:
            existing = self._cache.pop(key, None)
            if existing is not None:
                self._total_bytes -= existing.nbytes
            while self._cache and self._total_bytes + nbytes > self._max_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._total_bytes -= evicted.nbytes
            self._cache[key] = _Entry(value=value, nbytes=nbytes)
            self._total_bytes += nbytes
