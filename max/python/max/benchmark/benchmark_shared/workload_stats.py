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

"""Offline statistics of the requests a workload would send.

Computed from sampled requests alone, without a server, so a workload can be
compared with production before any GPU time is spent on it. Each figure is
defined the way MAX Serve's own metrics define it, because production is
measured from those:

- Input length is the whole prompt of a request, history included, which is
  what ``maxserve.input_tokens_per_request`` records.
- A request is a first turn when its conversation has no assistant message
  yet, matching that metric's ``turn`` label.
- The tool share counts requests that send a ``tools`` field, like the
  ``maxserve.tool_call.requests`` counter.

Multi-turn requests follow the session driver: warm-up prefix turns are built
locally and never sent, and a session ends at the first turn whose context
would exceed the tokenizer's context limit.

The stats describe the sampled workload, not any single run of it:
``--warmup-to-steady-state`` later drops unpicked candidate sessions and
gives the picked ones warm-up prefix turns, so a run with it on sends fewer
requests than these stats report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeChatSamples,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    ImageContentBlock,
    RequestSamples,
    SampledRequest,
    Samples,
)
from pydantic import BaseModel
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

WORKLOAD_STATS_SCHEMA_VERSION = 1

# The percentiles kimberlite records for production, so the two line up.
STATS_PERCENTILES: tuple[int, ...] = (50, 75, 90, 95, 99)

# Long-context thresholds. A five-point percentile summary says nothing about
# the tail past p99, and production's tail is where prefill cost and KV
# pressure concentrate. 192k is the Kadabra workloads' model-max-length.
INPUT_LEN_THRESHOLDS: tuple[int, ...] = (128_000, 192_000, 256_000)


@dataclass(frozen=True)
class WorkloadRequest:
    """One request a workload would send, reduced to what the stats need."""

    input_len: int
    output_len: int | None
    first_turn: bool
    num_tools: int
    constrained: bool
    carries_image: bool


class Distribution(BaseModel):
    """Summary of one per-request quantity.

    Deliberately not a
    :class:`~max.benchmark.benchmark_shared.percentile_metrics.PercentileMetrics`,
    which summarizes a measured run: its fixed p50/p90/p95/p99 fields and
    their flattened ``median_*`` columns are a BigQuery contract, and its
    confidence interval and positive-mean check judge a measurement's
    reliability. This summarizes requests built offline and is compared
    against production, so it carries kimberlite's percentiles (p75
    included), the count and bounds, and the long-context tail past p99,
    and it round-trips through the versioned ``--workload-stats-json``
    schema.
    """

    count: int
    mean: float
    min: float
    max: float
    percentiles: dict[int, float]
    # Share of requests strictly above each threshold; set for input lengths.
    fraction_above: dict[int, float] | None = None

    @classmethod
    def of(
        cls,
        values: Sequence[int | float],
        thresholds: Sequence[int] = (),
    ) -> Distribution | None:
        """Summarizes ``values``, or returns None when there are none."""
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        points = np.percentile(arr, STATS_PERCENTILES)
        return cls(
            count=len(arr),
            mean=float(arr.mean()),
            min=float(arr.min()),
            max=float(arr.max()),
            percentiles={
                p: float(v)
                for p, v in zip(STATS_PERCENTILES, points, strict=True)
            },
            fraction_above=(
                {t: float((arr > t).mean()) for t in thresholds}
                if thresholds
                else None
            ),
        )


class WorkloadStats(BaseModel):
    """What a workload would send, in the terms production is measured in."""

    schema_version: int = WORKLOAD_STATS_SCHEMA_VERSION
    requests: int
    # The limit multi-turn sessions are cut at; None when unbounded.
    context_limit: int | None
    # Turns never sent because their session hit ``context_limit`` first.
    turns_dropped_at_context_limit: int
    input_len: Distribution | None
    input_len_first_turn: Distribution | None
    input_len_later_turn: Distribution | None
    output_len: Distribution | None
    tool_request_fraction: float | None
    # Over requests that send tools, as production's tools-per-request
    # histogram only records those.
    tools_per_request: Distribution | None
    image_request_fraction: float | None
    constrained_request_fraction: float | None

    @property
    def dropped_turn_fraction(self) -> float | None:
        """Share of the workload's turns lost to the context limit."""
        total = self.requests + self.turns_dropped_at_context_limit
        return self.turns_dropped_at_context_limit / total if total else None


def _prompt_has_assistant(prompt: str | list[ChatMessage]) -> bool:
    return isinstance(prompt, list) and any(
        m.role == "assistant" for m in prompt
    )


def _prompt_has_image(prompt: str | list[ChatMessage]) -> bool:
    return isinstance(prompt, list) and any(
        isinstance(block, ImageContentBlock)
        for m in prompt
        if isinstance(m.content, list)
        for block in m.content
    )


def _single_turn_request(
    request: SampledRequest, run_prefix_len: int
) -> WorkloadRequest:
    return WorkloadRequest(
        input_len=request.prompt_len + run_prefix_len,
        output_len=request.output_len,
        first_turn=not _prompt_has_assistant(request.prompt_formatted),
        num_tools=len(request.tools or []),
        constrained=request.response_format is not None,
        carries_image=bool(request.encoded_images)
        or _prompt_has_image(request.prompt_formatted),
    )


def _chat_judge_requests(session: ChatSession) -> list[WorkloadRequest]:
    # Each turn is sent as [system?, user] with no history, tools, images or
    # response format.
    messages = list(session.messages)
    system_tokens = 0
    if messages and messages[0].source == "system":
        system_tokens = messages[0].num_tokens
        messages = messages[1:]
    return [
        WorkloadRequest(
            input_len=system_tokens + m.num_tokens,
            output_len=None,
            first_turn=True,
            num_tools=0,
            constrained=False,
            carries_image=False,
        )
        for m in messages
    ]


def _chat_session_requests(
    session: ChatSession,
    context_limit: int | None,
    run_prefix_len: int,
) -> tuple[list[WorkloadRequest], int]:
    """Walks a session the way ``chat_session_driver`` sends it.

    Returns the requests sent and the number of measured turns dropped at the
    context limit.
    """
    messages = session.messages
    prefix_end = session.prefix_turns * 2
    requests: list[WorkloadRequest] = []
    chat_len = 0
    # Earlier turns' images are resent with the history.
    image_seen = False
    idx = 0
    while idx + 1 < len(messages):
        user = messages[idx]
        chat_len += user.num_tokens
        if idx == 0 and prefix_end == 0:
            # The driver prepends the --force-unique-runs prefix to a cold
            # session's first measured turn only; a warmed session's
            # measured turns start past idx 0.
            chat_len += run_prefix_len
        image_seen = image_seen or bool(user.images)
        output_len = messages[idx + 1].num_tokens
        if context_limit is not None and chat_len + output_len > context_limit:
            first_unsent = max(idx, prefix_end)
            dropped = len(range(first_unsent, len(messages) - 1, 2))
            return requests, dropped
        if idx >= prefix_end:
            requests.append(
                WorkloadRequest(
                    input_len=chat_len,
                    output_len=output_len,
                    first_turn=idx == 0,
                    num_tools=len(user.tools or []),
                    constrained=user.response_format is not None,
                    carries_image=image_seen,
                )
            )
        # The run adds the server's completion length; offline, the scripted
        # one is the best estimate.
        chat_len += output_len
        idx += 2
    return requests, 0


def workload_requests(
    samples: Samples,
    context_limit: int | None = None,
    run_prefix_len: int = 0,
) -> tuple[list[WorkloadRequest], int]:
    """Lists the requests ``samples`` would send.

    Args:
        samples: The sampled workload.
        context_limit: The tokenizer's ``model_max_length``, which ends a
            multi-turn session. None leaves sessions uncut.
        run_prefix_len: Tokens ``--force-unique-runs`` prepends to every
            single-turn request and to a cold session's first turn. Zero
            when the flag is off; pixel-generation tasks, which do not
            track prompt length, must pass zero.

    Returns:
        The requests, and how many turns the context limit dropped.
    """
    if isinstance(samples, RequestSamples):
        return (
            [_single_turn_request(r, run_prefix_len) for r in samples.requests],
            0,
        )

    assert isinstance(samples, ChatSamples)
    requests: list[WorkloadRequest] = []
    dropped = 0
    for session in samples.chat_sessions:
        if isinstance(samples, ChatJudgeChatSamples):
            requests += _chat_judge_requests(session)
            continue
        session_requests, session_dropped = _chat_session_requests(
            session, context_limit, run_prefix_len
        )
        requests += session_requests
        dropped += session_dropped
    return requests, dropped


def _fraction(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def compute_workload_stats(
    samples: Samples,
    context_limit: int | None = None,
    run_prefix_len: int = 0,
) -> WorkloadStats:
    """Computes the stats of the requests ``samples`` would send.

    Args:
        samples: The sampled workload.
        context_limit: The tokenizer's ``model_max_length``. Values at or
            above transformers' unbounded sentinel are treated as no limit.
        run_prefix_len: Tokens ``--force-unique-runs`` prepends to every
            single-turn request and to a cold session's first turn. See
            :func:`workload_requests`.

    Returns:
        The workload's statistics.
    """
    if context_limit is not None and context_limit >= VERY_LARGE_INTEGER:
        context_limit = None
    requests, dropped = workload_requests(
        samples, context_limit, run_prefix_len
    )
    return WorkloadStats(
        requests=len(requests),
        context_limit=context_limit,
        turns_dropped_at_context_limit=dropped,
        input_len=Distribution.of(
            [r.input_len for r in requests], INPUT_LEN_THRESHOLDS
        ),
        input_len_first_turn=Distribution.of(
            [r.input_len for r in requests if r.first_turn],
            INPUT_LEN_THRESHOLDS,
        ),
        input_len_later_turn=Distribution.of(
            [r.input_len for r in requests if not r.first_turn],
            INPUT_LEN_THRESHOLDS,
        ),
        output_len=Distribution.of(
            [r.output_len for r in requests if r.output_len is not None]
        ),
        tool_request_fraction=_fraction([r.num_tools > 0 for r in requests]),
        tools_per_request=Distribution.of(
            [r.num_tools for r in requests if r.num_tools > 0]
        ),
        image_request_fraction=_fraction([r.carries_image for r in requests]),
        constrained_request_fraction=_fraction(
            [r.constrained for r in requests]
        ),
    )


def write_workload_stats(stats: WorkloadStats, path: str | Path) -> None:
    """Writes ``stats`` as JSON to ``path``."""
    Path(path).write_text(stats.model_dump_json(indent=2) + "\n")
