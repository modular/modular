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

from unittest.mock import Mock

import numpy as np
import pytest
from max.interfaces.reasoning import ReasoningSpan
from max.pipelines.lib.reasoning import (
    KimiK2_5ReasoningParser,
    MiniMaxM2ReasoningParser,
    create,
)


def test_reasoning_span_extract_content_removes_delimited_span() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 4), reasoning=(1, 3))
    result = span.extract_content([10, 20, 30, 40, 50])
    assert result == [50]


def test_reasoning_span_extract_reasoning_excludes_delimiters() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 4), reasoning=(1, 3))
    result = span.extract_reasoning([10, 20, 30, 40, 50])
    assert result == [20, 30]


def test_reasoning_span_extract_content_returns_empty_when_empty() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 5), reasoning=(1, 4))
    result = span.extract_content([10, 20, 30, 40, 50])
    assert result == []


def test_reasoning_span_extract_reasoning_returns_empty_when_empty() -> None:
    """Zero-width reasoning within delimiters returns empty list."""
    span = ReasoningSpan(reasoning_with_delimiters=(0, 2), reasoning=(1, 1))
    result = span.extract_reasoning([10, 20, 30])
    assert result == []


def test_reasoning_span_zero_width() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 0), reasoning=(0, 0))
    content = span.extract_content([10, 20])
    reasoning = span.extract_reasoning([10, 20])
    assert content == [10, 20]
    assert reasoning == []


def test_reasoning_span_content_before_and_after() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(1, 4), reasoning=(2, 3))
    tokens = [10, 20, 30, 40, 50]
    assert span.extract_content(tokens) == [10, 50]
    assert span.extract_reasoning(tokens) == [30]


def test_reasoning_span_invalid_ranges_raise() -> None:
    with pytest.raises(AssertionError):
        # reasoning range extends beyond delimited range
        ReasoningSpan(reasoning_with_delimiters=(2, 4), reasoning=(1, 3))


def _mock_tokenizer(token_map: dict[str, int | None]) -> Mock:
    """Create a mock tokenizer whose encode() returns single-element arrays."""

    async def mock_encode(
        token: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        token_id = token_map.get(token)
        if token_id is None:
            # Simulate unrecognized token: encode produces multiple IDs
            return np.array([0, 0])
        return np.array([token_id])

    mock = Mock()
    mock.encode = mock_encode
    return mock


@pytest.mark.asyncio
async def test_register_and_create() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<|tool_calls_section_begin|>": None,
        }
    )
    parser = await create("kimik2_5", mock)
    assert isinstance(parser, KimiK2_5ReasoningParser)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_section_start_token_id is None


@pytest.mark.asyncio
async def test_create_unknown_parser_raises() -> None:
    mock = Mock()
    with pytest.raises(ValueError, match="Unknown reasoning parser"):
        await create("nonexistent", mock)


THINK_START_TOKEN_ID = 1
THINK_END_TOKEN_ID = 2
TOOL_SECTION_START_TOKEN_ID = 3


@pytest.mark.parametrize(
    "tokens,expected_reasoning,expected_content,expected_is_still_reasoning",
    [
        pytest.param(
            [THINK_START_TOKEN_ID, 10, 20, THINK_END_TOKEN_ID, 30],
            [10, 20],
            [30],
            False,
            id="complete_think_block",
        ),
        pytest.param(
            [THINK_START_TOKEN_ID, 10, 20],
            [10, 20],
            [],
            True,
            id="think_start_token_id_only",
        ),
        pytest.param(
            [10, 20, 30], [10, 20, 30], [], True, id="no_think_tokens"
        ),
        pytest.param(
            [10, THINK_END_TOKEN_ID, 30],
            [10],
            [30],
            False,
            id="think_end_token_id_only",
        ),
        pytest.param(
            [10, TOOL_SECTION_START_TOKEN_ID, 30],
            [10],
            [30],
            False,
            id="tool_start_token_id_ends",
        ),
        pytest.param([], [], [], True, id="empty_chunk"),
        pytest.param(
            [
                THINK_START_TOKEN_ID,
                10,
                THINK_START_TOKEN_ID,
                20,
                THINK_END_TOKEN_ID,
            ],
            [10, THINK_START_TOKEN_ID, 20],
            [],
            False,
            id="multiple_think_start_token_ids",
        ),
        pytest.param(
            [10, THINK_END_TOKEN_ID, THINK_START_TOKEN_ID, 20],
            [10],
            [THINK_START_TOKEN_ID, 20],
            False,
            id="think_end_token_id_before_think_start_token_id",
        ),
        pytest.param(
            [10, THINK_END_TOKEN_ID, 30, TOOL_SECTION_START_TOKEN_ID, 40],
            [10],
            [30, TOOL_SECTION_START_TOKEN_ID, 40],
            False,
            id="think_end_token_id_wins_over_tool_start_token_id",
        ),
    ],
)
def test_stream_parsing(
    tokens: list[int],
    expected_reasoning: list[int] | None,
    expected_content: list[int] | None,
    expected_is_still_reasoning: bool,
) -> None:
    parser = KimiK2_5ReasoningParser(
        THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, TOOL_SECTION_START_TOKEN_ID
    )
    span, is_still_reasoning = parser.stream(tokens)
    assert span.extract_reasoning(tokens) == expected_reasoning
    assert span.extract_content(tokens) == expected_content
    assert is_still_reasoning is expected_is_still_reasoning


def test_stream_no_tool_start_token_id_support() -> None:
    parser = KimiK2_5ReasoningParser(
        THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, None
    )
    tokens = [10, TOOL_SECTION_START_TOKEN_ID, 30]
    span, is_still_reasoning = parser.stream(tokens)
    assert span.extract_reasoning(tokens) == [
        10,
        TOOL_SECTION_START_TOKEN_ID,
        30,
    ]
    assert span.extract_content(tokens) == []
    assert is_still_reasoning


@pytest.mark.asyncio
async def test_from_tokenizer_missing_tokens_raises() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": None,
            "</think>": None,
            "<|tool_calls_section_begin|>": None,
        }
    )
    with pytest.raises(
        ValueError,
        match="could not locate think start/end tokens in the tokenizer",
    ):
        await KimiK2_5ReasoningParser.from_tokenizer(mock)


@pytest.mark.asyncio
async def test_from_tokenizer_with_tool_start_token_id() -> None:
    """from_tokenizer correctly picks up tool_section_start_token_id when present."""
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<|tool_calls_section_begin|>": 300,
        }
    )
    parser = await KimiK2_5ReasoningParser.from_tokenizer(mock)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_section_start_token_id == 300


@pytest.mark.asyncio
async def test_minimax_m2_register_and_create() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<minimax:tool_call>": 300,
        }
    )
    parser = await create("minimax_m2", mock)
    assert isinstance(parser, MiniMaxM2ReasoningParser)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_call_start_token_id == 300


@pytest.mark.asyncio
async def test_minimax_m2_missing_think_tokens_raises() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": None,
            "</think>": 200,
            "<minimax:tool_call>": 300,
        }
    )
    with pytest.raises(ValueError, match="MiniMaxM2ReasoningParser"):
        await create("minimax_m2", mock)


@pytest.mark.asyncio
async def test_minimax_m2_optional_tool_token() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<minimax:tool_call>": None,
        }
    )
    parser = await create("minimax_m2", mock)
    assert isinstance(parser, MiniMaxM2ReasoningParser)
    assert parser.tool_call_start_token_id is None


def _make_minimax_parser(
    tool_call_start_token_id: int | None = 300,
) -> MiniMaxM2ReasoningParser:
    return MiniMaxM2ReasoningParser(
        think_start_token_id=100,
        think_end_token_id=200,
        tool_call_start_token_id=tool_call_start_token_id,
    )


def test_minimax_m2_stream_finds_think_boundaries() -> None:
    parser = _make_minimax_parser()
    # Tokens: [prefix, <think>, r1, r2, </think>, suffix]
    tokens = [10, 100, 11, 12, 200, 13]
    span, is_still_reasoning = parser.stream(tokens)
    assert is_still_reasoning is False
    assert span.extract_reasoning(tokens) == [11, 12]
    assert span.extract_content(tokens) == [10, 13]


def test_minimax_m2_stream_implicit_start() -> None:
    parser = _make_minimax_parser()
    # Chat template appends <think>\n at assistant turn, so the model's
    # first tokens are already inside a reasoning section.
    # Tokens: [r1, r2, </think>, answer]
    tokens = [11, 12, 200, 42]
    span, is_still_reasoning = parser.stream(tokens)
    assert is_still_reasoning is False
    assert span.extract_reasoning(tokens) == [11, 12]
    assert span.extract_content(tokens) == [42]


def test_minimax_m2_stream_tool_call_ends_reasoning() -> None:
    parser = _make_minimax_parser()
    # Model jumps straight to a tool call without </think>.
    # Tokens: [<think>, r1, <minimax:tool_call>, tc1, tc2]
    tokens = [100, 11, 300, 77, 78]
    span, is_still_reasoning = parser.stream(tokens)
    assert is_still_reasoning is False
    # Reasoning excludes both the <think> start and <minimax:tool_call>.
    assert span.extract_reasoning(tokens) == [11]
    # <minimax:tool_call> is NOT consumed — stays in content region.
    assert span.extract_content(tokens) == [300, 77, 78]


def test_minimax_m2_stream_no_end_still_reasoning() -> None:
    parser = _make_minimax_parser()
    # Mid-chunk during reasoning; no end marker yet.
    tokens = [11, 12, 13]
    span, is_still_reasoning = parser.stream(tokens)
    assert is_still_reasoning is True
    # Entire chunk is reasoning; nothing extracted as content.
    assert span.extract_reasoning(tokens) == [11, 12, 13]
    assert span.extract_content(tokens) == []
