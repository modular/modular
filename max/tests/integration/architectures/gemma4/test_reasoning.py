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
from max.pipelines.architectures.gemma4.reasoning import (
    Gemma4ReasoningParser,
)


def _mock_tokenizer(token_map: dict[str, int | None]) -> Mock:
    """Create a mock tokenizer whose encode() returns single-element arrays."""

    async def mock_encode(
        token: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        token_id = token_map.get(token)
        if token_id is None:
            return np.array([0, 0])
        return np.array([token_id])

    mock = Mock()
    mock.encode = mock_encode
    return mock


CHANNEL_START_TOKEN_ID = 1
CHANNEL_END_TOKEN_ID = 2
TOOL_CALL_START_TOKEN_ID = 3
THINK_TOKEN_ID = 4


def _make_parser(
    tool_call_start_token_id: int | None = TOOL_CALL_START_TOKEN_ID,
    think_token_id: int | None = THINK_TOKEN_ID,
) -> Gemma4ReasoningParser:
    return Gemma4ReasoningParser(
        channel_start_token_id=CHANNEL_START_TOKEN_ID,
        channel_end_token_id=CHANNEL_END_TOKEN_ID,
        tool_call_start_token_id=tool_call_start_token_id,
        think_token_id=think_token_id,
    )


@pytest.mark.parametrize(
    "tokens,expected_reasoning,expected_content,expected_is_still_reasoning",
    [
        pytest.param(
            [CHANNEL_START_TOKEN_ID, 10, 20, CHANNEL_END_TOKEN_ID, 30],
            [10, 20],
            [30],
            False,
            id="complete_channel_block",
        ),
        pytest.param(
            [CHANNEL_START_TOKEN_ID, 10, 20],
            [10, 20],
            [],
            True,
            id="channel_start_only",
        ),
        pytest.param(
            [10, 20, 30],
            [10, 20, 30],
            [],
            True,
            id="no_channel_tokens",
        ),
        pytest.param(
            [10, CHANNEL_END_TOKEN_ID, 30],
            [10],
            [30],
            False,
            id="channel_end_only",
        ),
        pytest.param(
            [],
            [],
            [],
            True,
            id="empty_chunk",
        ),
        pytest.param(
            [
                CHANNEL_START_TOKEN_ID,
                10,
                CHANNEL_START_TOKEN_ID,
                20,
                CHANNEL_END_TOKEN_ID,
            ],
            [10, CHANNEL_START_TOKEN_ID, 20],
            [],
            False,
            id="multiple_channel_starts",
        ),
        pytest.param(
            [10, CHANNEL_END_TOKEN_ID, CHANNEL_START_TOKEN_ID, 20],
            [10],
            [CHANNEL_START_TOKEN_ID, 20],
            False,
            id="channel_end_before_channel_start",
        ),
    ],
)
def test_stream_parsing(
    tokens: list[int],
    expected_reasoning: list[int],
    expected_content: list[int],
    expected_is_still_reasoning: bool,
) -> None:
    parser = _make_parser()
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == expected_reasoning
    assert delta.span.extract_content(tokens) == expected_content
    assert delta.is_still_reasoning is expected_is_still_reasoning


def test_stream_tool_call_ends_reasoning() -> None:
    parser = _make_parser()
    tokens = [CHANNEL_START_TOKEN_ID, 11, TOOL_CALL_START_TOKEN_ID, 77, 78]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [11]
    # <|tool_call> is NOT consumed — stays in content region.
    assert delta.span.extract_content(tokens) == [
        TOOL_CALL_START_TOKEN_ID,
        77,
        78,
    ]


def test_stream_channel_end_wins_over_tool_call() -> None:
    parser = _make_parser()
    tokens = [
        10,
        CHANNEL_END_TOKEN_ID,
        30,
        TOOL_CALL_START_TOKEN_ID,
        40,
    ]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [10]
    assert delta.span.extract_content(tokens) == [
        30,
        TOOL_CALL_START_TOKEN_ID,
        40,
    ]


def test_stream_no_tool_call_support() -> None:
    parser = _make_parser(tool_call_start_token_id=None)
    tokens = [10, TOOL_CALL_START_TOKEN_ID, 30]
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == [
        10,
        TOOL_CALL_START_TOKEN_ID,
        30,
    ]
    assert delta.span.extract_content(tokens) == []
    assert delta.is_still_reasoning


def test_is_prompt_in_reasoning_think_token_enables() -> None:
    parser = _make_parser()
    prompt = [THINK_TOKEN_ID, 10, 20]
    assert parser.is_prompt_in_reasoning(prompt) is True


def test_is_prompt_in_reasoning_no_think_token_disables() -> None:
    parser = _make_parser()
    prompt = [10, 20, 30]
    assert parser.is_prompt_in_reasoning(prompt) is False


def test_is_prompt_in_reasoning_empty_prompt_defaults_false() -> None:
    parser = _make_parser()
    assert parser.is_prompt_in_reasoning([]) is False


def test_is_prompt_in_reasoning_multi_turn_with_think() -> None:
    parser = _make_parser()
    # <|think|> in system, prior turn has <channel|>, new turn starts
    prompt = [
        THINK_TOKEN_ID,
        CHANNEL_START_TOKEN_ID,
        10,
        CHANNEL_END_TOKEN_ID,
        20,
    ]
    assert parser.is_prompt_in_reasoning(prompt) is True


def test_is_prompt_in_reasoning_no_think_token_id_configured() -> None:
    parser = _make_parser(think_token_id=None)
    prompt = [10, 20, 30]
    assert parser.is_prompt_in_reasoning(prompt) is False


def test_format_reasoning_full_prefix_in_one_chunk() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10, 20])
    assert delta.reasoning_text_formatter is not None
    result = delta.reasoning_text_formatter("thought\nactual reasoning")
    assert result == "actual reasoning"


def test_format_reasoning_split_across_chunks() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    result1 = delta.reasoning_text_formatter("thou")
    assert result1 is None

    delta2 = parser.stream([20, 30])
    assert delta2.reasoning_text_formatter is not None
    result2 = delta2.reasoning_text_formatter("ght\nactual")
    assert result2 == "actual"


def test_format_reasoning_exact_prefix() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    result = delta.reasoning_text_formatter("thought\n")
    assert result is None


def test_format_reasoning_mismatch() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    result = delta.reasoning_text_formatter("unexpected")
    assert result == "unexpected"


def test_reset_clears_prefix_state() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    delta.reasoning_text_formatter("thou")

    parser.reset()

    delta2 = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta2.reasoning_text_formatter is not None
    result = delta2.reasoning_text_formatter("thought\nactual")
    assert result == "actual"


@pytest.mark.asyncio
async def test_from_tokenizer_missing_tokens_raises() -> None:
    mock = _mock_tokenizer(
        {
            "<|channel>": None,
            "<channel|>": None,
            "<|tool_call>": None,
            "<|think|>": None,
        }
    )
    with pytest.raises(
        ValueError,
        match="could not locate channel start/end tokens in the tokenizer",
    ):
        await Gemma4ReasoningParser.from_tokenizer(mock)


@pytest.mark.asyncio
async def test_from_tokenizer_optional_tokens() -> None:
    mock = _mock_tokenizer(
        {
            "<|channel>": 100,
            "<channel|>": 200,
            "<|tool_call>": None,
            "<|think|>": None,
        }
    )
    parser = await Gemma4ReasoningParser.from_tokenizer(mock)
    assert parser.channel_start_token_id == 100
    assert parser.channel_end_token_id == 200
    assert parser.tool_call_start_token_id is None
    assert parser.think_token_id is None


@pytest.mark.asyncio
async def test_from_tokenizer_with_all_tokens() -> None:
    mock = _mock_tokenizer(
        {
            "<|channel>": 100,
            "<channel|>": 200,
            "<|tool_call>": 300,
            "<|think|>": 400,
        }
    )
    parser = await Gemma4ReasoningParser.from_tokenizer(mock)
    assert parser.channel_start_token_id == 100
    assert parser.channel_end_token_id == 200
    assert parser.tool_call_start_token_id == 300
    assert parser.think_token_id == 400
