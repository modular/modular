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
from max.pipelines.architectures.kimik2_5.reasoning import (
    KimiK2_5ReasoningParser,
)
from max.pipelines.architectures.minimax_m2.reasoning import (
    MiniMaxM2ReasoningParser,
)
from max.pipelines.lib.reasoning import create


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
