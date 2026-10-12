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
# ruff: noqa: RUF003

from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV4.reasoning import (
    DeepseekV4ReasoningParser,
)

# ``<think>`` / ``</think>`` ids in the DeepSeek-V4-Flash-0731 vocab.
_START = 128821
_END = 128822


def _make_parser() -> DeepseekV4ReasoningParser:
    return DeepseekV4ReasoningParser(
        think_start_token_id=_START, think_end_token_id=_END
    )


def test_stream_implicit_start() -> None:
    # Thinking mode prefills ``<think>``, so output opens inside reasoning.
    tokens = [11, 12, _END, 42, 43]
    delta = _make_parser().stream(tokens)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [11, 12]
    assert delta.span.extract_content(tokens) == [42, 43]


def test_stream_explicit_boundaries() -> None:
    tokens = [10, _START, 11, 12, _END, 13]
    delta = _make_parser().stream(tokens, is_currently_reasoning=False)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [11, 12]
    assert delta.span.extract_content(tokens) == [10, 13]


def test_stream_reasoning_spans_chunks() -> None:
    parser = _make_parser()
    first = [11, 12]
    delta = parser.stream(first)
    assert delta.is_still_reasoning is True
    assert delta.span.extract_reasoning(first) == [11, 12]

    second = [13, _END, 42]
    delta = parser.stream(second, is_currently_reasoning=True)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(second) == [13]
    assert delta.span.extract_content(second) == [42]


def test_stream_chat_mode_is_all_content() -> None:
    tokens = [42, 43, 44]
    delta = _make_parser().stream(tokens, is_currently_reasoning=False)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == []
    assert delta.span.extract_content(tokens) == tokens


def test_stream_stray_end_outside_reasoning_is_content() -> None:
    tokens = [42, _END, 43]
    delta = _make_parser().stream(tokens, is_currently_reasoning=False)
    assert delta.span.extract_content(tokens) == tokens


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        # Thinking mode: generation prompt ends ``<｜Assistant｜><think>``.
        ([1, 2, _START], True),
        # Chat mode: generation prompt ends ``<｜Assistant｜></think>``.
        ([1, 2, _END], False),
        # Tool instructions and kept earlier turns carry both delimiters;
        # the last one decides.
        ([_START, 5, _END, 6, _START, 7, _END, 8, _START], True),
        ([_START, 5, _END, 6, _START, 7, _END], False),
        ([1, 2, 3], False),
    ],
)
def test_will_reason_after_prompt(prompt: list[int], expected: bool) -> None:
    assert _make_parser().will_reason_after_prompt(prompt) is expected


@pytest.mark.asyncio
async def test_from_tokenizer() -> None:
    ids = {"<think>": _START, "</think>": _END}

    async def encode(
        token: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        return np.array([ids[token]])

    tokenizer = Mock()
    tokenizer.encode = encode
    parser = await DeepseekV4ReasoningParser.from_tokenizer(tokenizer)
    assert parser.think_start_token_id == _START
    assert parser.think_end_token_id == _END
    assert (
        await DeepseekV4ReasoningParser.reasoning_end_token_id(tokenizer)
        == _END
    )
