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
"""Checkpoints a request where an event happened to it."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .._context import Context
from .._triggers import Triggers
from .._types import Checkpoint, Event, EventKind


class AtEventHeuristic:
    """Checkpoints requests at the events of the kinds it is given.

    Each event's request is checkpointed at the last candidate at or before
    each position ``targets`` returns for the event, by default just the
    event's own position.
    """

    def __init__(
        self,
        *kinds: EventKind,
        targets: Callable[[Event], Sequence[int]] = lambda event: (
            event.position,
        ),
    ) -> None:
        self._kinds = kinds
        self._targets = targets

    def subscribe(self, triggers: Triggers) -> None:
        """Runs on this heuristic's event kinds."""
        triggers.on_event(self.checkpoint, *self._kinds)

    def checkpoint(self, ctx: Context) -> list[Checkpoint]:
        """Returns a checkpoint per target that has a candidate at or before it."""
        checkpoints = []
        for kind in self._kinds:
            for event in ctx.events(kind):
                for target in self._targets(event):
                    position = ctx.candidate_at_or_before(
                        event.request_id, target
                    )
                    if position is not None:
                        checkpoints.append(
                            Checkpoint(event.request_id, position)
                        )
        return checkpoints


def end_of_prompt() -> AtEventHeuristic:
    """Checkpoints where a prompt ends and where a replay of it resumes.

    A replay of the same prompt recomputes its last token to sample from, so
    it resumes one token short of the prompt's end.
    """
    return AtEventHeuristic(
        EventKind.END_OF_PROMPT,
        targets=lambda event: (event.position, event.position - 1),
    )


def release() -> AtEventHeuristic:
    """Checkpoints a request where it releases its cache."""
    return AtEventHeuristic(EventKind.RELEASE)
