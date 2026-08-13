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

"""Tests for the byte-budgeted preprocessed-vision cache."""

from __future__ import annotations

import pickle
import threading

from max.pipelines.lib.vision_preprocess_cache import VisionPreprocessCache


class TestVisionPreprocessCache:
    def test_hit_returns_the_stored_payload(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1024)
        cache.put(7, "seven", 10)

        assert cache.get(7) == "seven"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_miss_returns_none_and_counts(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1024)

        assert cache.get(7) is None
        assert cache.misses == 1

    def test_disabled_cache_never_retains(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(0)
        cache.put(7, "seven", 10)

        assert not cache.enabled
        assert cache.get(7) is None
        assert len(cache) == 0
        assert cache.total_bytes == 0

    def test_evicts_least_recently_used_to_fit_budget(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        cache.put(1, "a", 40)
        cache.put(2, "b", 40)

        # Touch 1 so 2 becomes the least recently used.
        assert cache.get(1) == "a"

        cache.put(3, "c", 40)

        assert cache.get(2) is None
        assert cache.get(1) == "a"
        assert cache.get(3) == "c"
        assert cache.total_bytes == 80

    def test_budget_is_tracked_in_bytes_not_entries(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        for key in range(10):
            cache.put(key, "small", 10)
        assert len(cache) == 10
        assert cache.total_bytes == 100

        # One large payload displaces as many small ones as it needs.
        cache.put(99, "large", 50)
        assert cache.total_bytes <= 100
        assert cache.get(99) == "large"

    def test_payload_larger_than_budget_is_dropped_not_cached(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        cache.put(1, "a", 60)
        cache.put(2, "oversized", 101)

        # The oversized payload must not flush the entries that do fit.
        assert cache.get(2) is None
        assert cache.get(1) == "a"
        assert cache.total_bytes == 60

    def test_reinsert_does_not_double_count_bytes(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(100)
        cache.put(1, "a", 40)
        cache.put(1, "a2", 30)

        assert cache.get(1) == "a2"
        assert len(cache) == 1
        assert cache.total_bytes == 30

    def test_concurrent_writers_keep_the_budget_consistent(self) -> None:
        # Preprocessing may run in worker threads, so the accounting has to
        # hold up without the caller serializing access.
        cache: VisionPreprocessCache[int] = VisionPreprocessCache(1000)
        barrier = threading.Barrier(8)

        def hammer(worker: int) -> None:
            barrier.wait()
            for i in range(100):
                key = worker * 100 + i
                cache.put(key, key, 10)
                cache.get(key)

        threads = [threading.Thread(target=hammer, args=(w,)) for w in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert cache.total_bytes <= 1000
        assert cache.total_bytes == 10 * len(cache)

    def test_survives_pickling_as_an_empty_cache(self) -> None:
        """The owning tokenizer is pickled into the spawned model worker.

        A ``threading.Lock`` cannot be pickled, so before this the whole
        server failed to start for every VLM. The cache is process-local:
        it must come back empty, usable, and with the budget intact.
        """
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(1024)
        cache.put(7, "seven", 10)
        assert len(cache) == 1

        revived: VisionPreprocessCache[str] = pickle.loads(pickle.dumps(cache))

        assert revived.enabled
        assert len(revived) == 0
        assert revived.total_bytes == 0
        assert revived.get(7) is None

        # Usable in the new process: a fresh lock was installed, not shared.
        revived.put(7, "seven", 10)
        assert revived.get(7) == "seven"
        assert revived.total_bytes == 10

    def test_disabled_cache_survives_pickling_disabled(self) -> None:
        cache: VisionPreprocessCache[str] = VisionPreprocessCache(0)

        revived: VisionPreprocessCache[str] = pickle.loads(pickle.dumps(cache))

        assert not revived.enabled
        revived.put(1, "one", 1)
        assert revived.get(1) is None
