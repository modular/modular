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

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.architectures.hy_v4.reasoning import HYV4ReasoningParser

# Token ids from the published Hy4 checkpoint's tokenizer.
THINK_START = 120029
THINK_END = 120030
TOOL_CALL_START = 120033
TOOL_CALL_END = 120034
TOOL_SECTION_START = 120035
TOOL_SECTION_END = 120036

# A forced tool call written inside still-open reasoning: the grammar arms at
# the section start, so ``</think:opensource>`` never follows.
CALL = [TOOL_SECTION_START, TOOL_CALL_START, 7, TOOL_CALL_END, TOOL_SECTION_END]


def _parser() -> HYV4ReasoningParser:
    return HYV4ReasoningParser(THINK_START, THINK_END, TOOL_SECTION_START)


def _mock_tokenizer(token_map: dict[str, int]) -> Mock:
    async def encode(token: str, add_special_tokens: bool = False) -> Any:
        return np.array([token_map[token]] if token in token_map else [0, 0])

    mock = Mock()
    mock.encode = encode
    return mock


@pytest.mark.parametrize(
    "tokens,reasoning,content",
    [
        pytest.param([10, 20, *CALL], [10, 20], CALL, id="tool_start_ends"),
        pytest.param(
            [THINK_START, 10, *CALL], [10], CALL, id="tool_start_after_open"
        ),
        pytest.param(
            [10, THINK_END, *CALL], [10], CALL, id="think_end_consumed"
        ),
    ],
)
def test_stream_ends_reasoning(
    tokens: list[int], reasoning: list[int], content: list[int]
) -> None:
    delta = _parser().stream(tokens, is_currently_reasoning=True)
    assert delta.span.extract_reasoning(tokens) == reasoning
    assert delta.span.extract_content(tokens) == content
    assert delta.is_still_reasoning is False


def test_stream_tool_start_does_not_reopen_reasoning() -> None:
    tokens = [10, *CALL]
    delta = _parser().stream(tokens, is_currently_reasoning=False)
    assert delta.span.extract_reasoning(tokens) == []
    assert delta.span.extract_content(tokens) == tokens
    assert delta.is_still_reasoning is False


@pytest.mark.asyncio
async def test_from_tokenizer_resolves_tool_section_start() -> None:
    parser = await HYV4ReasoningParser.from_tokenizer(
        _mock_tokenizer(
            {
                "<think:opensource>": THINK_START,
                "</think:opensource>": THINK_END,
                "<tool_calls:opensource>": TOOL_SECTION_START,
            }
        )
    )
    assert parser.tool_section_start_token_id == TOOL_SECTION_START


@pytest.mark.asyncio
async def test_from_tokenizer_without_tool_section_ends_at_think_only() -> None:
    parser = await HYV4ReasoningParser.from_tokenizer(
        _mock_tokenizer(
            {
                "<think:opensource>": THINK_START,
                "</think:opensource>": THINK_END,
            }
        )
    )
    assert parser.tool_section_start_token_id is None
    tokens = [10, *CALL]
    delta = parser.stream(tokens, is_currently_reasoning=True)
    assert delta.span.extract_reasoning(tokens) == tokens
    assert delta.is_still_reasoning is True
