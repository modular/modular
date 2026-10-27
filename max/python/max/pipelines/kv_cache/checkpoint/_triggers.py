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
"""When each heuristic's callbacks run."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ._context import Context
from ._types import Checkpoint, EventKind

Callback = Callable[[Context], Iterable[Checkpoint]]


class Triggers:
    """When each subscribed callback runs."""

    def __init__(self) -> None:
        self._by_kind: dict[EventKind, set[Callback]] = {}

    def on_event(self, callback: Callback, *kinds: EventKind) -> None:
        """Runs ``callback`` at the next step after any event of ``kinds``."""
        for kind in kinds:
            self._by_kind.setdefault(kind, set()).add(callback)

    def handles(self, kind: EventKind) -> bool:
        """Returns whether any callback runs on ``kind``."""
        return kind in self._by_kind

    def triggered(self, kinds: Iterable[EventKind]) -> set[Callback]:
        """Returns the callbacks any of ``kinds`` triggers."""
        return {cb for kind in kinds for cb in self._by_kind.get(kind, ())}
