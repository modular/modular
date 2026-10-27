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
"""The checkpointer, its context, and the at-event heuristics."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from max.pipelines.kv_cache.checkpoint import (
    Checkpoint,
    CheckpointContext,
    Checkpointer,
    CheckpointEvent,
    CheckpointEventKind,
    checkpoint_at_end_of_prompt,
    checkpoint_at_release,
)
from max.pipelines.kv_cache.checkpoint._triggers import Triggers
from max.pipelines.modeling.types import RequestID


class _Candidates:
    """Answers from fixed candidate positions per request, counting queries."""

    def __init__(self, positions: dict[RequestID, Sequence[int]]) -> None:
        self.positions = positions
        self.queries: list[tuple[RequestID, int]] = []

    def __call__(self, request_id: RequestID, position: int) -> int | None:
        self.queries.append((request_id, position))
        before = [
            p for p in self.positions.get(request_id, ()) if p <= position
        ]
        return before[-1] if before else None


class _CountingHeuristic:
    """Runs on the given kinds and records each run."""

    def __init__(self, *kinds: CheckpointEventKind) -> None:
        self.kinds = kinds
        self.runs = 0

    def subscribe(self, triggers: Triggers) -> None:
        triggers.on_event(self.checkpoint, *self.kinds)

    def checkpoint(self, ctx: CheckpointContext) -> list[Checkpoint]:
        self.runs += 1
        return []


def test_an_event_checkpoints_at_the_last_candidate_at_or_before_it() -> None:
    request = RequestID()
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(checkpoint_at_end_of_prompt())
        .with_candidate_at_or_before(_Candidates({request: [4, 8, 12]}))
        .build()
    )

    checkpointer.record(
        CheckpointEvent(request, CheckpointEventKind.END_OF_PROMPT, 10)
    )

    assert checkpointer.checkpoint() == {Checkpoint(request, 8)}


def test_an_event_with_no_candidate_before_it_checkpoints_nothing() -> None:
    request = RequestID()
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(checkpoint_at_end_of_prompt())
        .with_candidate_at_or_before(_Candidates({request: [8]}))
        .build()
    )

    checkpointer.record(
        CheckpointEvent(request, CheckpointEventKind.END_OF_PROMPT, 5)
    )

    assert checkpointer.checkpoint() == set()


def test_events_no_heuristic_handles_are_dropped() -> None:
    request = RequestID()
    candidates = _Candidates({request: [4]})
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(checkpoint_at_end_of_prompt())
        .with_candidate_at_or_before(candidates)
        .build()
    )

    checkpointer.record(
        CheckpointEvent(request, CheckpointEventKind.RELEASE, 4)
    )

    assert checkpointer.checkpoint() == set()
    assert candidates.queries == []


def test_a_heuristic_runs_once_per_step_however_many_events_trigger_it() -> (
    None
):
    heuristic = _CountingHeuristic(
        CheckpointEventKind.END_OF_PROMPT, CheckpointEventKind.RELEASE
    )
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(heuristic)
        .with_candidate_at_or_before(_Candidates({}))
        .build()
    )

    for kind in (
        CheckpointEventKind.END_OF_PROMPT,
        CheckpointEventKind.RELEASE,
    ):
        checkpointer.record(CheckpointEvent(RequestID(), kind, 1))
    checkpointer.checkpoint()

    assert heuristic.runs == 1


def test_a_step_with_no_events_runs_no_heuristic() -> None:
    heuristic = _CountingHeuristic(CheckpointEventKind.RELEASE)
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(heuristic)
        .with_candidate_at_or_before(_Candidates({}))
        .build()
    )

    checkpointer.checkpoint()

    assert heuristic.runs == 0


def test_a_step_forgets_its_events() -> None:
    request = RequestID()
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(checkpoint_at_release())
        .with_candidate_at_or_before(_Candidates({request: [4]}))
        .build()
    )

    checkpointer.record(
        CheckpointEvent(request, CheckpointEventKind.RELEASE, 4)
    )
    assert checkpointer.checkpoint() == {Checkpoint(request, 4)}

    assert checkpointer.checkpoint() == set()


def test_heuristics_choosing_the_same_checkpoint_yield_it_once() -> None:
    request = RequestID()
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(checkpoint_at_end_of_prompt(), checkpoint_at_release())
        .with_candidate_at_or_before(_Candidates({request: [4]}))
        .build()
    )

    checkpointer.record(
        CheckpointEvent(request, CheckpointEventKind.END_OF_PROMPT, 5)
    )
    checkpointer.record(
        CheckpointEvent(request, CheckpointEventKind.RELEASE, 6)
    )

    assert checkpointer.checkpoint() == {Checkpoint(request, 4)}


def test_the_context_asks_the_cache_once_per_query_per_step() -> None:
    request = RequestID()
    candidates = _Candidates({request: [4]})
    ctx = CheckpointContext(candidates)

    assert ctx.candidate_at_or_before(request, 6) == 4
    assert ctx.candidate_at_or_before(request, 6) == 4
    assert candidates.queries == [(request, 6)]

    ctx.reset()
    ctx.candidate_at_or_before(request, 6)
    assert candidates.queries == [(request, 6), (request, 6)]


def test_an_enabled_manager_needs_a_candidate_query() -> None:
    with pytest.raises(ValueError, match="candidate_at_or_before"):
        Checkpointer.factory().with_heuristics(checkpoint_at_release()).build()


def test_a_disabled_manager_drops_every_event() -> None:
    """A cache that cannot persist disables the checkpointer it is offered.

    Its heuristics never run, so the events nothing would clear are dropped.
    """
    request = RequestID()
    checkpointer = (
        Checkpointer.factory()
        .with_heuristics(checkpoint_at_end_of_prompt(), checkpoint_at_release())
        .with_candidate_at_or_before(_Candidates({request: [4]}))
        .disabled()
        .build()
    )

    checkpointer.record(
        CheckpointEvent(request, CheckpointEventKind.RELEASE, 4)
    )

    assert checkpointer.checkpoint() == set()
