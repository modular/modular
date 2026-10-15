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

"""Tests for benchmark_shared.workload_stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeChatSamples,
)
from max.benchmark.benchmark_shared.datasets.image_augmentation import (
    generate_random_image,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    OpenAIImage,
    RequestSamples,
    SampledRequest,
    SessionMessage,
    TextContentBlock,
    encode_image,
)
from max.benchmark.benchmark_shared.serving_result_output import (
    print_workload_stats,
)
from max.benchmark.benchmark_shared.workload_stats import (
    WORKLOAD_STATS_SCHEMA_VERSION,
    Distribution,
    WorkloadStats,
    compute_workload_stats,
    workload_requests,
    write_workload_stats,
)
from openai.types.chat.completion_create_params import ResponseFormat
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": f"tool_{i}"}} for i in range(3)
]
_JSON_OBJECT: ResponseFormat = {"type": "json_object"}


def _image() -> OpenAIImage:
    return encode_image(generate_random_image(16, 16))


def _request(
    prompt_len: int = 10,
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: ResponseFormat | None = None,
    encoded_images: list[OpenAIImage] | None = None,
    prompt: str | list[ChatMessage] = "hello",
) -> SampledRequest:
    return SampledRequest(
        prompt_formatted=prompt,
        prompt_len=prompt_len,
        output_len=5,
        encoded_images=encoded_images or [],
        ignore_eos=False,
        response_format=response_format,
        tools=tools,
    )


def _user(
    num_tokens: int,
    *,
    tools: list[dict[str, Any]] | None = None,
    images: list[OpenAIImage] | None = None,
) -> SessionMessage:
    return SessionMessage(
        source="user",
        content="u",
        num_tokens=num_tokens,
        tools=tools,
        images=images or [],
    )


def _assistant(num_tokens: int) -> SessionMessage:
    return SessionMessage(
        source="assistant", content="a", num_tokens=num_tokens
    )


def _session(
    *turns: tuple[SessionMessage, int], prefix_turns: int = 0
) -> ChatSession:
    messages: list[SessionMessage] = []
    for user, output_len in turns:
        messages += [user, _assistant(output_len)]
    return ChatSession(id=0, messages=messages, prefix_turns=prefix_turns)


def test_single_turn_mix_counts_each_request_once() -> None:
    samples = RequestSamples(
        requests=[
            _request(tools=_TOOLS),
            _request(tools=_TOOLS[:1], response_format=_JSON_OBJECT),
            _request(encoded_images=[_image()]),
            _request(),
        ]
    )
    stats = compute_workload_stats(samples)
    assert stats.requests == 4
    assert stats.tool_request_fraction == 0.5
    assert stats.constrained_request_fraction == 0.25
    assert stats.image_request_fraction == 0.25
    assert stats.tools_per_request is not None
    assert stats.tools_per_request.count == 2
    assert stats.tools_per_request.mean == 2.0


def test_empty_tools_list_is_not_a_tool_request() -> None:
    """The runner only sends ``tools`` when the list is non-empty."""
    stats = compute_workload_stats(
        RequestSamples(requests=[_request(tools=[])])
    )
    assert stats.tool_request_fraction == 0.0
    assert stats.tools_per_request is None


def test_single_turn_prompt_with_history_is_a_later_turn() -> None:
    history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="a"),
        ChatMessage(role="user", content=[TextContentBlock(text="q2")]),
    ]
    requests, _ = workload_requests(
        RequestSamples(requests=[_request(prompt=history), _request()])
    )
    assert [r.first_turn for r in requests] == [False, True]


def test_chat_input_is_the_whole_context_at_each_turn() -> None:
    samples = ChatSamples(
        chat_sessions=[_session((_user(100), 10), (_user(20), 10))]
    )
    requests, dropped = workload_requests(samples)
    assert [r.input_len for r in requests] == [100, 130]
    assert [r.first_turn for r in requests] == [True, False]
    assert [r.output_len for r in requests] == [10, 10]
    assert dropped == 0


def test_chat_prefix_turns_are_not_sent_but_count_toward_context() -> None:
    samples = ChatSamples(
        chat_sessions=[
            _session(
                (_user(100), 10),
                (_user(20), 10),
                (_user(30), 10),
                prefix_turns=2,
            )
        ]
    )
    requests, _ = workload_requests(samples)
    assert len(requests) == 1
    assert requests[0].input_len == 100 + 10 + 20 + 10 + 30
    assert not requests[0].first_turn


def test_chat_images_are_resent_with_later_turns() -> None:
    samples = ChatSamples(
        chat_sessions=[
            _session((_user(5), 1), (_user(5, images=[_image()]), 1)),
            _session((_user(5), 1), (_user(5), 1)),
        ]
    )
    stats = compute_workload_stats(samples)
    assert stats.image_request_fraction == 0.25

    resent = ChatSamples(
        chat_sessions=[
            _session((_user(5, images=[_image()]), 1), (_user(5), 1))
        ]
    )
    assert compute_workload_stats(resent).image_request_fraction == 1.0


def test_chat_tools_are_per_turn() -> None:
    samples = ChatSamples(
        chat_sessions=[_session((_user(5, tools=_TOOLS), 1), (_user(5), 1))]
    )
    stats = compute_workload_stats(samples)
    assert stats.tool_request_fraction == 0.5


def test_context_limit_ends_the_session_like_the_driver() -> None:
    samples = ChatSamples(
        chat_sessions=[
            _session((_user(100), 10), (_user(100), 10), (_user(100), 10))
        ]
    )
    stats = compute_workload_stats(samples, context_limit=250)
    # Turn 2 needs 100 + 10 + 100 + 10 = 220; turn 3 needs 330 > 250.
    assert stats.requests == 2
    assert stats.turns_dropped_at_context_limit == 1
    assert stats.dropped_turn_fraction == pytest.approx(1 / 3)
    assert stats.context_limit == 250


def test_prefix_over_the_limit_drops_every_measured_turn() -> None:
    samples = ChatSamples(
        chat_sessions=[
            _session(
                (_user(300), 10),
                (_user(5), 1),
                (_user(5), 1),
                prefix_turns=1,
            )
        ]
    )
    stats = compute_workload_stats(samples, context_limit=250)
    assert stats.requests == 0
    assert stats.turns_dropped_at_context_limit == 2
    assert stats.input_len is None


def test_run_prefix_adds_to_each_single_turn_request() -> None:
    samples = RequestSamples(
        requests=[_request(prompt_len=10), _request(prompt_len=20)]
    )
    requests, _ = workload_requests(samples, run_prefix_len=4)
    assert [r.input_len for r in requests] == [14, 24]


def test_run_prefix_adds_to_a_cold_sessions_first_turn_only() -> None:
    samples = ChatSamples(
        chat_sessions=[_session((_user(100), 10), (_user(20), 10))]
    )
    requests, _ = workload_requests(samples, run_prefix_len=4)
    # Only the first turn grows; later turns carry it inside chat_len.
    assert [r.input_len for r in requests] == [104, 134]


def test_run_prefix_counts_toward_the_first_turns_limit() -> None:
    samples = ChatSamples(
        chat_sessions=[_session((_user(100), 10), (_user(20), 10))]
    )
    # Turn 1 needs 100 + 4 (prefix) + 10 = 114 > 113, so the session ends
    # before it; without the prefix it would fit.
    stats = compute_workload_stats(samples, context_limit=113, run_prefix_len=4)
    assert stats.requests == 0
    assert stats.turns_dropped_at_context_limit == 2


def test_run_prefix_is_not_added_to_a_warmed_session() -> None:
    """The driver only prefixes a session whose measured turns start at 0."""
    samples = ChatSamples(
        chat_sessions=[
            _session(
                (_user(100), 10),
                (_user(20), 10),
                (_user(30), 10),
                prefix_turns=2,
            )
        ]
    )
    requests, _ = workload_requests(samples, run_prefix_len=4)
    assert requests[0].input_len == 100 + 10 + 20 + 10 + 30


def test_run_prefix_is_not_added_to_chat_judge_turns() -> None:
    session = ChatSession(
        id=0,
        messages=[
            SessionMessage(source="system", content="s", num_tokens=50),
            _user(10),
        ],
    )
    requests, _ = workload_requests(
        ChatJudgeChatSamples(chat_sessions=[session]), run_prefix_len=4
    )
    assert [r.input_len for r in requests] == [60]


def test_unbounded_tokenizer_limit_is_no_limit() -> None:
    samples = ChatSamples(chat_sessions=[_session((_user(10), 1))])
    stats = compute_workload_stats(samples, context_limit=VERY_LARGE_INTEGER)
    assert stats.context_limit is None
    assert stats.requests == 1


def test_chat_judge_sends_system_plus_user_without_history() -> None:
    session = ChatSession(
        id=0,
        messages=[
            SessionMessage(source="system", content="s", num_tokens=50),
            _user(10),
            _user(20),
        ],
    )
    requests, _ = workload_requests(
        ChatJudgeChatSamples(chat_sessions=[session])
    )
    assert [r.input_len for r in requests] == [60, 70]
    assert all(r.first_turn for r in requests)


def test_input_distribution_reports_long_context_shares() -> None:
    lengths = [100_000] * 6 + [150_000] * 2 + [200_000] + [300_000]
    stats = compute_workload_stats(
        RequestSamples(requests=[_request(n) for n in lengths])
    )
    assert stats.input_len is not None
    assert stats.input_len.fraction_above == {
        128_000: pytest.approx(0.4),
        192_000: pytest.approx(0.2),
        256_000: pytest.approx(0.1),
    }
    assert stats.input_len.max == 300_000
    assert stats.input_len.percentiles[50] == 100_000
    assert stats.output_len is not None
    assert stats.output_len.fraction_above is None


def test_distribution_of_nothing_is_none() -> None:
    assert Distribution.of([]) is None


def test_written_json_round_trips(tmp_path: Path) -> None:
    samples = ChatSamples(
        chat_sessions=[_session((_user(100, tools=_TOOLS), 10), (_user(20), 5))]
    )
    stats = compute_workload_stats(samples, context_limit=1000)
    path = tmp_path / "stats.json"
    write_workload_stats(stats, path)

    blob = json.loads(path.read_text())
    assert blob["schema_version"] == WORKLOAD_STATS_SCHEMA_VERSION
    assert blob["input_len"]["percentiles"]["50"] == 115.0
    assert blob["input_len"]["fraction_above"]["192000"] == 0.0
    assert WorkloadStats.model_validate(blob) == stats


def test_print_workload_stats_prints_requests_sent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = ChatSamples(
        chat_sessions=[
            _session((_user(100, tools=_TOOLS), 10), (_user(100), 10))
        ]
    )
    print_workload_stats(
        samples, compute_workload_stats(samples, context_limit=150)
    )
    out = capsys.readouterr().out
    assert "Requests Sent" in out
    assert "Input length (first turn)" in out
    assert "Turns dropped at the limit:" in out
    assert "Tool-offering requests:" in out
    assert "100.00%" in out


def test_print_workload_stats_sections_share_columns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Input length lines up across sections; raw values add the spread."""
    samples = ChatSamples(
        chat_sessions=[_session((_user(100), 10), (_user(100), 10))]
    )
    print_workload_stats(samples, compute_workload_stats(samples))
    lines = capsys.readouterr().out.splitlines()
    headers = {
        line.strip(): lines[i + 1].split()
        for i, line in enumerate(lines)
        if line.startswith("  Input length")
    }
    assert headers["Input length (per turn):"] == (
        "count min mean std p5 p25 p50 p75 p90 p95 p99 max".split()
    )
    assert headers["Input length:"] == (
        "count min mean p50 p75 p90 p95 p99 max".split()
    )


def test_print_workload_stats_without_stats_omits_requests_sent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_workload_stats(RequestSamples(requests=[_request()]))
    assert "Requests Sent" not in capsys.readouterr().out
