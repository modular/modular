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
from max.pipelines.architectures.kimik2_5.reasoning import (
    KimiK2_5ReasoningParser,
)


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
            [10, TOOL_SECTION_START_TOKEN_ID, 30],
            [],
            True,
            id="tool_start_token_id_does_not_end_reasoning",
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
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == expected_reasoning
    assert delta.span.extract_content(tokens) == expected_content
    assert delta.is_still_reasoning is expected_is_still_reasoning


def test_stream_no_tool_start_token_id_support() -> None:
    parser = KimiK2_5ReasoningParser(
        THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, None
    )
    tokens = [10, TOOL_SECTION_START_TOKEN_ID, 30]
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == [
        10,
        TOOL_SECTION_START_TOKEN_ID,
        30,
    ]
    assert delta.span.extract_content(tokens) == []
    assert delta.is_still_reasoning


def test_is_prompt_in_reasoning_tool_section_token_disables_reasoning() -> None:
    parser = KimiK2_5ReasoningParser(
        THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, TOOL_SECTION_START_TOKEN_ID
    )
    # Right-to-left scan: <|tool_calls_section_begin|> is treated as an
    # end-of-reasoning delimiter, so reasoning is disabled.
    prompt = [10, 20, TOOL_SECTION_START_TOKEN_ID, 30]
    assert parser.is_prompt_in_reasoning(prompt) is False


def test_is_prompt_in_reasoning_think_end_disables_reasoning() -> None:
    parser = KimiK2_5ReasoningParser(
        THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, TOOL_SECTION_START_TOKEN_ID
    )
    prompt = [10, THINK_START_TOKEN_ID, 20, THINK_END_TOKEN_ID, 30]
    assert parser.is_prompt_in_reasoning(prompt) is False


def test_is_prompt_in_reasoning_empty_prompt_stays_active() -> None:
    parser = KimiK2_5ReasoningParser(
        THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, TOOL_SECTION_START_TOKEN_ID
    )
    assert parser.is_prompt_in_reasoning([]) is True


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
