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
from max.pipelines.architectures.minimax_m2.reasoning import (
    MiniMaxM2ReasoningParser,
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


def _make_parser(
    tool_call_start_token_id: int | None = 300,
) -> MiniMaxM2ReasoningParser:
    return MiniMaxM2ReasoningParser(
        think_start_token_id=100,
        think_end_token_id=200,
        tool_call_start_token_id=tool_call_start_token_id,
    )


def test_stream_finds_think_boundaries() -> None:
    parser = _make_parser()
    # Tokens: [prefix, <think>, r1, r2, </think>, suffix]
    tokens = [10, 100, 11, 12, 200, 13]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [11, 12]
    assert delta.span.extract_content(tokens) == [10, 13]


def test_stream_implicit_start() -> None:
    parser = _make_parser()
    # Chat template appends <think>\n at assistant turn, so the model's
    # first tokens are already inside a reasoning section.
    # Tokens: [r1, r2, </think>, answer]
    tokens = [11, 12, 200, 42]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [11, 12]
    assert delta.span.extract_content(tokens) == [42]


def test_stream_tool_call_ends_reasoning() -> None:
    parser = _make_parser()
    # Model jumps straight to a tool call without </think>.
    # Tokens: [<think>, r1, <minimax:tool_call>, tc1, tc2]
    tokens = [100, 11, 300, 77, 78]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is False
    # Reasoning excludes both the <think> start and <minimax:tool_call>.
    assert delta.span.extract_reasoning(tokens) == [11]
    # <minimax:tool_call> is NOT consumed — stays in content region.
    assert delta.span.extract_content(tokens) == [300, 77, 78]


def test_stream_no_end_still_reasoning() -> None:
    parser = _make_parser()
    # Mid-chunk during reasoning; no end marker yet.
    tokens = [11, 12, 13]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is True
    # Entire chunk is reasoning; nothing extracted as content.
    assert delta.span.extract_reasoning(tokens) == [11, 12, 13]
    assert delta.span.extract_content(tokens) == []


def test_is_prompt_in_reasoning_tool_call_token_does_not_disable_reasoning() -> (
    None
):
    parser = _make_parser()
    # The MiniMax chat template embeds <minimax:tool_call> in the prompt when
    # tools are provided. This must NOT disable reasoning for generation.
    prompt = [10, 20, 300, 30, 40]  # 300 = tool_call_start_token_id
    assert parser.is_prompt_in_reasoning(prompt) is True


def test_is_prompt_in_reasoning_think_end_disables_reasoning() -> None:
    parser = _make_parser()
    # If the prompt already contains </think>, reasoning is disabled.
    prompt = [10, 100, 11, 200, 20]  # 200 = think_end_token_id
    assert parser.is_prompt_in_reasoning(prompt) is False


def test_is_prompt_in_reasoning_empty_prompt_stays_active() -> None:
    parser = _make_parser()
    assert parser.is_prompt_in_reasoning([]) is True


@pytest.mark.asyncio
async def test_from_tokenizer_missing_tokens_raises() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": None,
            "</think>": 200,
            "<minimax:tool_call>": 300,
        }
    )
    with pytest.raises(ValueError, match="MiniMaxM2ReasoningParser"):
        await MiniMaxM2ReasoningParser.from_tokenizer(mock)


@pytest.mark.asyncio
async def test_from_tokenizer_optional_tool_token() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<minimax:tool_call>": None,
        }
    )
    parser = await MiniMaxM2ReasoningParser.from_tokenizer(mock)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_call_start_token_id is None


@pytest.mark.asyncio
async def test_from_tokenizer_with_tool_token() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<minimax:tool_call>": 300,
        }
    )
    parser = await MiniMaxM2ReasoningParser.from_tokenizer(mock)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_call_start_token_id == 300
