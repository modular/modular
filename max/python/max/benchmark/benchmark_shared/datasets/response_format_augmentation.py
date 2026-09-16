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
`response_format` -- the quantity `maxserve.response_format.requests` reports
via its `kind` tag.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence

from typing_extensions import assert_never

from .chat_judge import ChatJudgeChatSamples
from .types import (
    ChatSamples,
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
        if turn == "first":
            # A session warmed to steady state replays its opening turns
            # locally, so a constraint on the first one is never sent
            # (CENG-1086).
            logger.warning(
                "Structured-output mixing: --response-format-turn first loses"
                " the constraint on any session that starts mid-conversation"
                " (--warmup-to-steady-state, on by default); use 'last' or"
                " 'every' to constrain a turn the driver is guaranteed to send."
            )
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
    here -- `pick_warmup_population` assigns it after sampling -- so "first" can
    name a turn that `chat_session_driver` later replays locally and never
    sends. See CENG-1086; `augment_samples_with_response_format` warns when the
    combination is reachable.
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
