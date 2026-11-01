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
"""Context needed by checkpoint heuristics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from max.pipelines.modeling.types import RequestID

from ._types import Event, EventKind


class Context:
    """Context needed by checkpoint heuristics."""

    def __init__(
        self, candidate_at_or_before: Callable[[RequestID, int], int | None]
    ) -> None:
        """Initializes the context.

        Args:
            candidate_at_or_before: Returns the last token position at or
                before the given one that the request can be checkpointed at
                now, or ``None`` if there is none.
        """
        self._candidate_at_or_before = candidate_at_or_before
        self._events: dict[EventKind, list[Event]] = {}
        self._candidates: dict[tuple[RequestID, int], int | None] = {}

    def events(self, kind: EventKind) -> Sequence[Event]:
        """Returns the events of ``kind`` reported since the last step."""
        return self._events.get(kind, [])

    def candidate_at_or_before(
        self, request_id: RequestID, position: int
    ) -> int | None:
        """Returns the last checkpointable position at or before ``position``."""
        key = (request_id, position)
        if key not in self._candidates:
            self._candidates[key] = self._candidate_at_or_before(
                request_id, position
            )
        return self._candidates[key]

    def record(self, event: Event) -> None:
        """Records an event."""
        self._events.setdefault(event.kind, []).append(event)

    def reset(self) -> None:
        """Forgets the step's events and query results."""
        self._events.clear()
        self._candidates.clear()
