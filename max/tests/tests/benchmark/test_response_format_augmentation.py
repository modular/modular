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

"""Tests for dataset-agnostic structured-output mixing."""

from __future__ import annotations

import dataclasses
import logging

import pytest
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeChatSamples,
)
from max.benchmark.benchmark_shared.datasets.response_format_augmentation import (
    augment_samples_with_response_format,
    move_first_turn_response_format,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    ChatSession,
    RequestSamples,
    ResponseFormat,
    SessionMessage,
    TurnSelector,
)
from tests.benchmark.benchmark_fakes import make_sampled_request, make_session

_RF: ResponseFormat = {"type": "json_object"}


def _constrained_turn_indices(session: ChatSession) -> list[int]:
    user_turns = [m for m in session.messages if m.source == "user"]
    return [
        i for i, m in enumerate(user_turns) if m.response_format is not None
    ]


def test_zero_fraction_is_noop() -> None:
    samples = RequestSamples(
        requests=[make_sampled_request() for _ in range(5)]
    )
    augment_samples_with_response_format(
        samples, response_format=_RF, fraction=0.0
    )
    assert all(r.response_format is None for r in samples.requests)
    assert all(r.ignore_eos for r in samples.requests)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_rejects_out_of_range_fraction(fraction: float) -> None:
    samples = RequestSamples(requests=[make_sampled_request()])
    with pytest.raises(ValueError, match="must be in"):
        augment_samples_with_response_format(
            samples, response_format=_RF, fraction=fraction
        )


def test_full_fraction_constrains_every_request_and_clears_ignore_eos() -> None:
    samples = RequestSamples(
        requests=[make_sampled_request() for _ in range(5)]
    )
    augment_samples_with_response_format(
        samples, response_format=_RF, fraction=1.0
    )
    assert all(r.response_format == _RF for r in samples.requests)
    # A constrained response ends at its schema, so the drawn length caps it.
    assert not any(r.ignore_eos for r in samples.requests)


def test_request_partial_fraction_converges() -> None:
    n = 2000
    samples = RequestSamples(
        requests=[make_sampled_request() for _ in range(n)]
    )
    augment_samples_with_response_format(
        samples, response_format=_RF, fraction=0.3
    )
    constrained = sum(1 for r in samples.requests if r.response_format)
    # Loose bound: a Bernoulli(0.3) draw over 2000 trials essentially never
    # lands outside +/- 0.1 of the target fraction.
    assert 0.2 * n < constrained < 0.4 * n


def test_chat_fraction_lands_on_turns_not_sessions() -> None:
    """A partial fraction lands on turns, leaving sessions partly constrained.

    Per-session selection would leave every session all-on or all-off.
    """
    turns_per_session = 4
    sessions = [make_session(i, turns_per_session) for i in range(500)]
    augment_samples_with_response_format(
        ChatSamples(chat_sessions=sessions),
        response_format=_RF,
        fraction=0.25,
        turn="every",
    )
    total = 500 * turns_per_session
    constrained = sum(len(_constrained_turn_indices(s)) for s in sessions)
    assert 0.15 * total < constrained < 0.35 * total
    # Per-session selection would leave whole sessions all-on or all-off.
    partially = sum(
        1
        for s in sessions
        if 0 < len(_constrained_turn_indices(s)) < turns_per_session
    )
    assert partially > 0


@pytest.mark.parametrize(
    ("turn", "expected"),
    [("first", [0]), ("last", [2]), ("every", [0, 1, 2])],
)
def test_turn_selector_restricts_eligible_turns(
    turn: TurnSelector, expected: list[int]
) -> None:
    session = make_session(0, num_user_turns=3)
    augment_samples_with_response_format(
        ChatSamples(chat_sessions=[session]),
        response_format=_RF,
        fraction=1.0,
        turn=turn,
    )
    assert _constrained_turn_indices(session) == expected


def test_assistant_messages_are_never_constrained() -> None:
    session = make_session(0, num_user_turns=3)
    augment_samples_with_response_format(
        ChatSamples(chat_sessions=[session]),
        response_format=_RF,
        fraction=1.0,
        turn="every",
    )
    assert all(
        m.response_format is None
        for m in session.messages
        if m.source != "user"
    )


def test_chat_judge_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its driver builds its own requests and never reads the per-turn field,
    so constraining here would be reported but never sent."""
    session = make_session(0, num_user_turns=2)
    samples = ChatJudgeChatSamples(chat_sessions=[session])
    with caplog.at_level(logging.WARNING):
        augment_samples_with_response_format(
            samples, response_format=_RF, fraction=1.0
        )
    assert "chat-judge" in caplog.text
    assert all(m.response_format is None for m in session.messages)


def _first_turn_constrained(prefix_turns: int) -> ChatSession:
    session = make_session(0, num_user_turns=4)
    augment_samples_with_response_format(
        ChatSamples(chat_sessions=[session]),
        response_format=_RF,
        fraction=1.0,
        turn="first",
    )
    return dataclasses.replace(session, prefix_turns=prefix_turns)


def test_first_turn_constraint_moves_to_the_first_sent_turn() -> None:
    """A warmed session replays its opening turns locally and never sends their
    response_format, so the constraint goes on turn `prefix_turns`."""
    warmed = _first_turn_constrained(prefix_turns=2)
    (moved,) = move_first_turn_response_format([warmed])
    assert _constrained_turn_indices(moved) == [2]
    assert moved.messages[4].response_format == _RF
    assert moved.prefix_turns == 2


def test_first_turn_move_leaves_the_sampled_session_unchanged() -> None:
    """Each run picks start turns again from the same sampled sessions."""
    warmed = _first_turn_constrained(prefix_turns=3)
    move_first_turn_response_format([warmed])
    assert _constrained_turn_indices(warmed) == [0]


def test_first_turn_move_skips_cold_and_unconstrained_sessions() -> None:
    cold = _first_turn_constrained(prefix_turns=0)
    unconstrained = dataclasses.replace(
        make_session(1, num_user_turns=4), prefix_turns=2
    )
    moved = move_first_turn_response_format([cold, unconstrained])
    assert moved[0] is cold
    assert moved[1] is unconstrained


def test_seed_makes_selection_reproducible() -> None:
    """A private RNG stream, so a rerun repeats and the global stream that
    later augmentations draw from is left where it was."""

    def constrained_with(seed: int) -> list[int]:
        samples = RequestSamples(
            requests=[make_sampled_request() for _ in range(200)]
        )
        augment_samples_with_response_format(
            samples, response_format=_RF, fraction=0.5, seed=seed
        )
        return [
            i
            for i, r in enumerate(samples.requests)
            if r.response_format is not None
        ]

    assert constrained_with(7) == constrained_with(7)
    assert constrained_with(7) != constrained_with(8)


def test_session_with_no_user_turns_is_skipped() -> None:
    session = ChatSession(
        id=0,
        messages=[SessionMessage(source="assistant", content="", num_tokens=5)],
    )
    augment_samples_with_response_format(
        ChatSamples(chat_sessions=[session]),
        response_format=_RF,
        fraction=1.0,
    )
    assert all(m.response_format is None for m in session.messages)
