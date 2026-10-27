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

from .._context import Context
from .._triggers import Triggers
from .._types import Checkpoint, EventKind


class AtEventHeuristic:
    """Checkpoints requests at the events of the kinds it is given.

    Each event's request is checkpointed at the last candidate at or before
    the event's position.
    """

    def __init__(self, *kinds: EventKind) -> None:
        self._kinds = kinds

    def subscribe(self, triggers: Triggers) -> None:
        """Runs on this heuristic's event kinds."""
        triggers.on_event(self.checkpoint, *self._kinds)

    def checkpoint(self, ctx: Context) -> list[Checkpoint]:
        """Returns a checkpoint per event that has a candidate at or before it."""
        checkpoints = []
        for kind in self._kinds:
            for event in ctx.events(kind):
                position = ctx.candidate_at_or_before(
                    event.request_id, event.position
                )
                if position is not None:
                    checkpoints.append(Checkpoint(event.request_id, position))
        return checkpoints


def end_of_prompt() -> AtEventHeuristic:
    """Checkpoints a request where its prompt ends."""
    return AtEventHeuristic(EventKind.END_OF_PROMPT)


def release() -> AtEventHeuristic:
    """Checkpoints a request where it releases its cache."""
    return AtEventHeuristic(EventKind.RELEASE)
