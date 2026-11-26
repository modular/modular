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

"""Dataset-agnostic structured-output mixing for benchmark workloads.

Mixes a `response_format` into a fraction of already-sampled requests/turns, so
a workload can carry the share of constrained-decoding traffic production
actually sees rather than all of it or none of it.

The fraction is per request and per user turn, not per session as
`image_augmentation` selects, so it lands on the share of *requests* that set
`response_format` -- `maxserve.structured_output.requests` over the
chat-completion `maxserve.request_count`.
"""

from __future__ import annotations

import dataclasses
import logging
import random
from collections.abc import Sequence

from typing_extensions import assert_never

from .chat_judge import ChatJudgeChatSamples
from .types import (
    ChatSamples,
    ChatSession,
    RequestSamples,
    ResponseFormat,
    Samples,
    SessionMessage,
    TurnSelector,
)

logger = logging.getLogger(__name__)


def augment_samples_with_response_format(
    samples: Samples,
    *,
    response_format: ResponseFormat,
    fraction: float,
    turn: TurnSelector = "every",
    seed: int | None = None,
) -> None:
    """Constrain a fraction of already-sampled requests/turns to a schema.

    Dataset-agnostic: operates on the `Samples` any dataset produces, after
    sampling, so datasets never need their own structured-output logic.

    A constrained request also has `ignore_eos` cleared. A schema-shaped
    response ends where its schema is satisfied, so the drawn output length
    caps it rather than pinning it; generating past that point makes the
    server drop enforcement for the rest of the request, which would report
    constrained-decoding numbers for output that was not constrained.

    Args:
        samples: Already-sampled requests or chat sessions, mutated in place.
        response_format: The OpenAI `response_format` to apply.
        fraction: Fraction (0.0-1.0) of requests (single-turn) or of eligible
            user turns (multi-turn) to constrain. Drawn per request/turn, so
            it lands directly on the share of requests that set
            `response_format`.
        turn: Which user turns of a multi-turn session are eligible: "every"
            (any turn), "first", or "last". Ignored for single-turn requests.
        seed: Seed for this function's own RNG. Drawing from a private stream
            rather than the global one keeps the selection reproducible without
            displacing the draws any later augmentation makes.

    Raises:
        ValueError: If `fraction` is outside [0, 1].
        TypeError: If `samples` is neither `RequestSamples` nor `ChatSamples`.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(
            f"response_format_fraction must be in [0, 1], got {fraction}"
        )
    if fraction == 0:
        logger.info(
            "Structured-output mixing: fraction is 0, leaving the workload"
            " unconstrained"
        )
        return

    rng = random.Random(seed)

    if isinstance(samples, RequestSamples):
        constrained = 0
        for request in samples.requests:
            if rng.random() >= fraction:
                continue
            request.response_format = response_format
            request.ignore_eos = False
            constrained += 1
        logger.info(
            "Structured-output mixing: constrained %d/%d requests",
            constrained,
            len(samples.requests),
        )
    elif isinstance(samples, ChatJudgeChatSamples):
        # chat_judge_session_driver builds its own requests and never reads the
        # per-turn field, so constraining here would be reported but never sent.
        logger.warning(
            "Structured-output mixing: skipping chat-judge workload; its"
            " driver does not send response_format."
        )
    elif isinstance(samples, ChatSamples):
        constrained = 0
        eligible = 0
        for session in samples.chat_sessions:
            for message in _eligible_turns(session.messages, turn):
                eligible += 1
                if rng.random() >= fraction:
                    continue
                message.response_format = response_format
                constrained += 1
        logger.info(
            "Structured-output mixing: constrained %d/%d eligible turns"
            " (turn=%s) across %d chat sessions",
            constrained,
            eligible,
            turn,
            len(samples.chat_sessions),
        )
    else:
        raise TypeError(f"Unsupported samples type: {type(samples)}")


def _eligible_turns(
    messages: Sequence[SessionMessage], turn: TurnSelector
) -> list[SessionMessage]:
    """The user turns `turn` makes eligible, in session order.

    Indexes the whole session, prefix turns included. `prefix_turns` is still 0
    here -- `pick_warmup_population` assigns it after sampling -- so "first"
    names the opening turn, and `move_first_turn_response_format` later moves
    the mark to the first turn a warmed session sends.
    """
    user_turns = [m for m in messages if m.source == "user"]
    if not user_turns:
        return []
    if turn == "every":
        return user_turns
    if turn == "first":
        return [user_turns[0]]
    if turn == "last":
        return [user_turns[-1]]
    assert_never(turn)


def move_first_turn_response_format(
    sessions: Sequence[ChatSession],
) -> list[ChatSession]:
    """Moves each warmed session's first-turn constraint to its first sent turn.

    `chat_session_driver` replays a session's first `prefix_turns` turns
    locally and never sends their `response_format`. With
    `--response-format-turn first`, the constraint must go on user turn
    `prefix_turns`, the first turn the driver sends. Run this after
    `pick_warmup_population` assigns `prefix_turns`.

    The sessions share their messages with the sampled workload, which later
    runs pick start turns from again, so this copies the messages it changes.

    Args:
        sessions: Sessions whose `prefix_turns` are assigned, each below the
            session's user-turn count. Only the opening user turn of each may
            carry a `response_format`.

    Returns:
        The sessions, with each warmed session's constraint moved.
    """
    moved: list[ChatSession] = []
    for session in sessions:
        user_indices = [
            i for i, m in enumerate(session.messages) if m.source == "user"
        ]
        if session.prefix_turns == 0 or not user_indices:
            moved.append(session)
            continue
        first = session.messages[user_indices[0]]
        if first.response_format is None:
            moved.append(session)
            continue
        target = user_indices[session.prefix_turns]
        messages = list(session.messages)
        messages[user_indices[0]] = dataclasses.replace(
            first, response_format=None
        )
        messages[target] = dataclasses.replace(
            messages[target], response_format=first.response_format
        )
        moved.append(dataclasses.replace(session, messages=messages))
    return moved
