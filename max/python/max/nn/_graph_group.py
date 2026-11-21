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
"""The graph group base class, and the groups that ``max.nn`` builds.

:mod:`max.nn.legacy.graph_groups` re-exports these classes. A kind names
exactly one class, so ``max.nn`` and ``max.nn.legacy`` share each class
rather than define their own.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import ClassVar

from max import tree
from max.graph import BufferType, TensorType
from typing_extensions import Self, TypeVar

__all__ = ["EPPayload", "GraphGroup"]

_K = TypeVar("_K", bound=str)
_V = TypeVar("_V")


class GraphGroup(OrderedDict[_K, _V]):
    """The named graph inputs or outputs of one feature, in graph order.

    An ``OrderedDict`` keeps insertion order when flattened, where a plain
    ``dict`` would sort its keys. Each subclass fixes the names and values it
    holds.
    """

    kind: ClassVar[str | None] = None
    """Identifies the feature wherever the group sits in a tree."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # A kind names one class, so a subclass that inherits its base's kind
        # leaves the registration to the base.
        if "kind" in cls.__dict__ and cls.kind is not None:
            tree.kind(cls.kind)(cls)

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        """Creates the group, rejecting a subclass with no ``kind``."""
        if cls.kind is None:
            raise TypeError(
                f"{cls.__qualname__} has no kind, so an exported graph could"
                " not name it: subclass GraphGroup with a kind"
            )
        return super().__new__(cls)

    def __tree_flatten__(self) -> tuple[Mapping[_K, _V], None]:
        return self, None

    @classmethod
    def __tree_unflatten__(cls, meta: None, children: Mapping[_K, _V]) -> Self:
        # Not through ``__init__``, which a subclass may make keyword-only.
        group = cls.__new__(cls)
        OrderedDict.update(group, children)
        return group


class EPPayload(GraphGroup[str, tuple[TensorType | BufferType, ...]]):
    """Expert-parallel communication state, built once at startup."""

    kind = "ep_payload"
