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

from max.pipelines.lib.utils import BoundedCache


class TestBoundedCache:
    def test_basic_get_set(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(4)
        cache["a"] = 1
        cache["b"] = 2
        assert cache["a"] == 1
        assert cache["b"] == 2
        assert len(cache) == 2

    def test_evicts_lru_on_overflow(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Cache is full; inserting "d" should evict "a" (oldest).
        cache["d"] = 4
        assert "a" not in cache
        assert len(cache) == 3
        assert list(cache.keys()) == ["b", "c", "d"]

    def test_access_promotes_entry(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Access "a" so it becomes most-recently-used.
        _ = cache["a"]
        # Now "b" is the LRU entry and should be evicted.
        cache["d"] = 4
        assert "b" not in cache
        assert "a" in cache
        assert list(cache.keys()) == ["c", "a", "d"]

    def test_overwrite_promotes_entry(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Overwrite "a" — it should move to the end (most recent).
        cache["a"] = 10
        cache["d"] = 4
        assert "b" not in cache
        # Use dict access to avoid __getitem__ promoting "a".
        assert dict.__getitem__(cache, "a") == 10
        assert list(cache.keys()) == ["c", "a", "d"]

    def test_contains_does_not_promote(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # `in` check should NOT promote "a".
        assert "a" in cache
        cache["d"] = 4
        assert "a" not in cache

    def test_maxsize_one(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(1)
        cache["a"] = 1
        assert cache["a"] == 1
        cache["b"] = 2
        assert "a" not in cache
        assert cache["b"] == 2
        assert len(cache) == 1

    def test_dict_compatible_iteration(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(4)
        cache["x"] = 10
        cache["y"] = 20
        assert dict(cache) == {"x": 10, "y": 20}
        assert list(cache.items()) == [("x", 10), ("y", 20)]
