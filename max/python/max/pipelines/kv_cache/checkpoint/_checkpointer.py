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
"""Collects lifecycle events and runs heuristics."""

from __future__ import annotations

from collections.abc import Callable

from max.pipelines.modeling.types import RequestID
from typing_extensions import Self

from ._context import Context
from ._heuristic import Heuristic
from ._triggers import Triggers
from ._types import Checkpoint, Event, EventKind


class Checkpointer:
    """Collects lifecycle events and runs heuristics."""

    @staticmethod
    def factory() -> CheckpointerFactory:
        """Returns a factory to configure and build a checkpointer."""
        return CheckpointerFactory()

    def __init__(self, ctx: Context) -> None:
        """Initializes the checkpointer.

        Args:
            ctx: The context the heuristics read.
        """
        self._ctx = ctx
        self._triggers = Triggers()
        self._pending: set[EventKind] = set()

    def register(self, *heuristics: Heuristic) -> Self:
        """Subscribes each of ``heuristics`` and returns this checkpointer."""
        for heuristic in heuristics:
            heuristic.subscribe(self._triggers)
        return self

    def record(self, event: Event) -> None:
        """Records an event for the next checkpoint step.

        Call it before the request's cache can change, for example before
        releasing it. Events no heuristic handles are dropped.
        """
        if not self._triggers.handles(event.kind):
            return
        self._ctx.record(event)
        self._pending.add(event.kind)

    def checkpoint(self) -> set[Checkpoint]:
        """Runs each callback the step's events triggered, once.

        Returns:
            The distinct checkpoints chosen.
        """
        checkpoints: set[Checkpoint] = set()
        for callback in self._triggers.triggered(self._pending):
            checkpoints.update(callback(self._ctx))
        self._pending.clear()
        self._ctx.reset()
        return checkpoints


class CheckpointerFactory:
    """Builds a checkpointer once every query its context needs is provided."""

    def __init__(self) -> None:
        self._heuristics: list[Heuristic] = []
        self._candidate_at_or_before: (
            Callable[[RequestID, int], int | None] | None
        ) = None
        self._disabled = False

    def with_heuristics(self, *heuristics: Heuristic) -> Self:
        """Adds heuristics for the checkpointer to run."""
        self._heuristics.extend(heuristics)
        return self

    def with_candidate_at_or_before(
        self, query: Callable[[RequestID, int], int | None]
    ) -> Self:
        """Sets the query that answers the context's candidates."""
        self._candidate_at_or_before = query
        return self

    def disabled(self) -> Self:
        """Makes the checkpointer drop every event."""
        self._disabled = True
        return self

    def build(self) -> Checkpointer:
        """Returns the checkpointer.

        Raises:
            ValueError: If the checkpointer is enabled and lacks a candidate query.
        """
        if self._disabled:
            return Checkpointer(Context(lambda request_id, position: None))
        if self._candidate_at_or_before is None:
            raise ValueError(
                "A checkpointer needs a candidate_at_or_before query"
            )
        return Checkpointer(Context(self._candidate_at_or_before)).register(
            *self._heuristics
        )
