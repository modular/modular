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

"""Tests for dataset-agnostic tool-definition mixing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from max.benchmark.benchmark_serving import parse_tools
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeChatSamples,
)
from max.benchmark.benchmark_shared.datasets.tool_augmentation import (
    augment_samples_with_tools,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    RequestSamples,
    SampledRequest,
    SessionMessage,
    TextContentBlock,
)

_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "read_file"}},
    {"type": "function", "function": {"name": "run_shell"}},
]
# Words the fake template spends on each tool, so the block for _TOOLS costs 8.
_WORDS_PER_TOOL = 4


def _word_tokenizer(*, template_renders_tools: bool = True) -> MagicMock:
    """A tokenizer whose tokens are whitespace-separated words.

    Its chat template renders each tool as ``_WORDS_PER_TOOL`` words, so the
    carve-out the augmentation computes is known exactly.
    """
    vocab: dict[str, int] = {}
    words: list[str] = []

    def encode(text: str, add_special_tokens: bool = True) -> list[int]:
        ids = []
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(words)
                words.append(word)
            ids.append(vocab[word])
        return ids

    def decode(ids: list[int]) -> str:
        return " ".join(words[i] for i in ids)

    def apply_chat_template(
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        rendered = ["<sys>"]
        if tools and template_renders_tools:
            rendered += ["tool"] * (_WORDS_PER_TOOL * len(tools))
        rendered += [m["content"] for m in messages]
        return " ".join(rendered)

    tokenizer = MagicMock()
    tokenizer.encode.side_effect = encode
    tokenizer.decode.side_effect = decode
    tokenizer.apply_chat_template.side_effect = apply_chat_template
    return tokenizer


def _first_turn_text(num_words: int) -> str:
    return " ".join(f"w{i}" for i in range(num_words))


def _make_session(session_id: int, num_user_turns: int = 3) -> ChatSession:
    messages: list[SessionMessage] = []
    for i in range(num_user_turns):
        messages.append(
            SessionMessage(
                source="user",
                content=_first_turn_text(20) if i == 0 else f"turn {i}",
                num_tokens=20 if i == 0 else 2,
            )
        )
        messages.append(
            SessionMessage(source="assistant", content="", num_tokens=5)
        )
    return ChatSession(id=session_id, messages=messages)


def _make_request(prompt: str = "hi") -> SampledRequest:
    return SampledRequest(
        prompt_formatted=prompt,
        prompt_len=len(prompt.split()),
        output_len=64,
        encoded_images=[],
        ignore_eos=True,
    )


def _user_turns(session: ChatSession) -> list[SessionMessage]:
    return [m for m in session.messages if m.source == "user"]


def test_zero_fraction_is_noop() -> None:
    sessions = [_make_session(i) for i in range(5)]
    augment_samples_with_tools(
        ChatSamples(chat_sessions=sessions),
        tools=_TOOLS,
        fraction=0.0,
        tokenizer=_word_tokenizer(),
    )
    assert all(m.tools is None for s in sessions for m in s.messages)
    assert _user_turns(sessions[0])[0].content == _first_turn_text(20)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_rejects_out_of_range_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="must be in"):
        augment_samples_with_tools(
            RequestSamples(requests=[_make_request()]),
            tools=_TOOLS,
            fraction=fraction,
            tokenizer=None,
        )


def test_rejects_empty_tools() -> None:
    with pytest.raises(ValueError, match="at least one"):
        augment_samples_with_tools(
            RequestSamples(requests=[_make_request()]),
            tools=[],
            fraction=1.0,
            tokenizer=None,
        )


def test_selection_is_per_session_and_covers_every_turn() -> None:
    """A selected session carries tools on every user turn, so no session
    toggles them mid-conversation and misses the prefix cache for it."""
    sessions = [_make_session(i, num_user_turns=4) for i in range(1000)]
    augment_samples_with_tools(
        ChatSamples(chat_sessions=sessions),
        tools=_TOOLS,
        fraction=0.68,
        tokenizer=None,
    )
    selected = 0
    for session in sessions:
        offered = [m.tools is not None for m in _user_turns(session)]
        assert all(offered) or not any(offered)
        selected += all(offered)
        assert all(
            m.tools is None for m in session.messages if m.source != "user"
        )
    # Loose bound: a Bernoulli(0.68) draw over 1000 sessions essentially never
    # lands outside +/- 0.08 of the target.
    assert 600 < selected < 760


def test_block_is_carved_out_of_the_first_turn() -> None:
    """The rendered block replaces prompt tokens rather than adding to them:
    the first turn loses exactly the block's tokens off its tail, and its
    drawn ``num_tokens`` still describes the whole prompt."""
    session = _make_session(0)
    augment_samples_with_tools(
        ChatSamples(chat_sessions=[session]),
        tools=_TOOLS,
        fraction=1.0,
        tokenizer=_word_tokenizer(),
    )
    first, *rest = _user_turns(session)
    block = _WORDS_PER_TOOL * len(_TOOLS)
    # Trimmed from the tail, so the shared system-prompt head survives.
    assert first.content == _first_turn_text(20 - block)
    assert first.num_tokens == 20
    assert [m.content for m in rest] == ["turn 1", "turn 2"]


def test_block_is_sized_from_json_when_the_template_ignores_tools() -> None:
    tokenizer = _word_tokenizer(template_renders_tools=False)
    session = _make_session(0)
    augment_samples_with_tools(
        ChatSamples(chat_sessions=[session]),
        tools=_TOOLS,
        fraction=1.0,
        tokenizer=tokenizer,
    )
    json_tokens = len(json.dumps(_TOOLS).split())
    assert _user_turns(session)[0].content == _first_turn_text(20 - json_tokens)


def test_without_a_tokenizer_tools_attach_untrimmed() -> None:
    session = _make_session(0)
    augment_samples_with_tools(
        ChatSamples(chat_sessions=[session]),
        tools=_TOOLS,
        fraction=1.0,
        tokenizer=None,
    )
    first = _user_turns(session)[0]
    assert first.tools == _TOOLS
    assert first.content == _first_turn_text(20)


def test_prompt_shorter_than_the_block_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = _make_request("too short")
    with caplog.at_level(logging.WARNING):
        augment_samples_with_tools(
            RequestSamples(requests=[request]),
            tools=_TOOLS,
            fraction=1.0,
            tokenizer=_word_tokenizer(),
        )
    assert "shorter than" in caplog.text
    assert request.tools == _TOOLS


def test_single_turn_requests_are_selected_and_trimmed() -> None:
    requests = [_make_request(_first_turn_text(20)) for _ in range(3)]
    augment_samples_with_tools(
        RequestSamples(requests=requests),
        tools=_TOOLS,
        fraction=1.0,
        tokenizer=_word_tokenizer(),
    )
    block = _WORDS_PER_TOOL * len(_TOOLS)
    for request in requests:
        assert request.tools == _TOOLS
        assert request.prompt_formatted == _first_turn_text(20 - block)
        # Tools do not change how the drawn output length is honored.
        assert request.ignore_eos


def test_dataset_tools_are_kept_and_left_unselected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    own_tools: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": "dataset_tool"}}
    ]
    request = _make_request(_first_turn_text(20))
    request.tools = own_tools
    with caplog.at_level(logging.WARNING):
        augment_samples_with_tools(
            RequestSamples(requests=[request]),
            tools=_TOOLS,
            fraction=1.0,
            tokenizer=_word_tokenizer(),
        )
    assert request.tools == own_tools
    assert request.prompt_formatted == _first_turn_text(20)
    assert "already carry their dataset's tools" in caplog.text


def test_chat_message_prompts_are_sent_untrimmed_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    messages = [ChatMessage(role="user", content=[TextContentBlock(text="hi")])]
    request = SampledRequest(
        prompt_formatted=messages,
        prompt_len=1,
        output_len=64,
        encoded_images=[],
        ignore_eos=True,
    )
    with caplog.at_level(logging.WARNING):
        augment_samples_with_tools(
            RequestSamples(requests=[request]),
            tools=_TOOLS,
            fraction=1.0,
            tokenizer=_word_tokenizer(),
        )
    assert request.tools == _TOOLS
    assert request.prompt_formatted == messages
    assert "not trimmed" in caplog.text


def test_chat_judge_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its driver builds its own requests and never reads the per-turn field,
    so attaching here would be reported but never sent."""
    session = _make_session(0)
    with caplog.at_level(logging.WARNING):
        augment_samples_with_tools(
            ChatJudgeChatSamples(chat_sessions=[session]),
            tools=_TOOLS,
            fraction=1.0,
            tokenizer=None,
        )
    assert "chat-judge" in caplog.text
    assert all(m.tools is None for m in session.messages)


def test_seed_makes_selection_reproducible() -> None:
    def selected_with(seed: int) -> list[int | None]:
        sessions = [_make_session(i) for i in range(200)]
        augment_samples_with_tools(
            ChatSamples(chat_sessions=sessions),
            tools=_TOOLS,
            fraction=0.5,
            tokenizer=None,
            seed=seed,
        )
        return [s.id for s in sessions if _user_turns(s)[0].tools is not None]

    assert selected_with(7) == selected_with(7)
    assert selected_with(7) != selected_with(8)


def test_parse_tools_reads_inline_json_and_files(tmp_path: Path) -> None:
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps(_TOOLS))
    assert parse_tools(json.dumps(_TOOLS)) == _TOOLS
    assert parse_tools(f"@{tools_file}") == _TOOLS


@pytest.mark.parametrize(
    ("arg", "match"),
    [
        ("[]", "empty"),
        ("{not json", "Invalid tools"),
        ('[{"type": "function"}]', "Invalid tools"),
        ("@/nonexistent/tools.json", "not found"),
    ],
)
def test_parse_tools_rejects_malformed_input(arg: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_tools(arg)
