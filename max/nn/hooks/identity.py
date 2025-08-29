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
"""Utility classes for using objects as keys in data structures."""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSet
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


# From https://stackoverflow.com/questions/16994307/identityset-in-python
class IdentitySet(Generic[_T], MutableSet[_T]):
    """Set that uses object `id` as keys to support unhashable types."""

    def __init__(self, iterable=()) -> None:  # noqa: ANN001
        self.map: dict[Any, Any] = {}  # id -> object
        self |= iterable  # add elements from iterable to the set (union)

    def __len__(self) -> int:
        return len(self.map)

    def __iter__(self):
        return iter(self.map.values())

    def __contains__(self, x) -> bool:  # noqa: ANN001
        return id(x) in self.map

    def add(self, value) -> None:  # noqa: ANN001
        """Add an element."""
        self.map[id(value)] = value

    def discard(self, value) -> None:  # noqa: ANN001
        """Remove an element.  Do not raise an exception if absent."""
        self.map.pop(id(value), None)

    def __repr__(self) -> str:
        if not self:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({list(self)!r})"


class IdentityMap(Generic[_K, _V], MutableMapping[_K, _V]):
    """Map that uses object `id` as keys to support unhashable types."""

    def __init__(self) -> None:
        self.key_map: dict[Any, Any] = {}  # id -> object
        self.value_map: dict[Any, Any] = {}  # id -> Value

    def __getitem__(self, key):  # noqa: ANN001
        return self.value_map[id(key)]

    def __setitem__(self, key, value) -> None:  # noqa: ANN001
        self.key_map[id(key)] = key
        self.value_map[id(key)] = value

    def __delitem__(self, key) -> None:  # noqa: ANN001
        del self.key_map[id(key)]
        del self.value_map[id(key)]

    def __iter__(self):
        return iter(self.key_map.values())

    def __len__(self) -> int:
        return len(self.key_map)
