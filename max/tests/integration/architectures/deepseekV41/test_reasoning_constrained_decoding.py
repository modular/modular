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
"""Tests JSON-schema constrained decoding with the V4.1 reasoning parser.

Schema enforcement is suspended while the model reasons inside
``<think>``...``</think>`` and applies to the final answer after it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from max._core import xgrammar as xgr
from max.pipelines.architectures.deepseekV4.reasoning import (
    DeepseekV4ReasoningParser,
)
from max.pipelines.context import (
    GrammarEnforcementState,
    StructuredOutputRegionDelimiters,
)
from max.pipelines.lib.registry import _ThinkingRegionNewContext

THINK_START = DeepseekV4ReasoningParser.REASONING_START
THINK_END = DeepseekV4ReasoningParser.REASONING_END
_MARKERS = [THINK_START, THINK_END]

_SCHEMA = {
    "type": "object",
    "properties": {
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["unit"],
    "additionalProperties": False,
}


def _vocab() -> tuple[list[bytes], dict[str, int]]:
    """256 single-byte tokens, one per reasoning marker, and a stop token."""
    toks: list[bytes] = [bytes([i]) for i in range(256)]
    marker_ids: dict[str, int] = {}
    for marker in _MARKERS:
        marker_ids[marker] = len(toks)
        toks.append(marker.encode("utf-8"))
    marker_ids["<STOP>"] = len(toks)
    toks.append(b"<STOP>")
    return toks, marker_ids


def _compiled_schema_matcher() -> tuple[Any, dict[str, int]]:
    vocab, marker_ids = _vocab()
    info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[marker_ids["<STOP>"]],
    )
    compiled = xgr.GrammarCompiler(info).compile_json_schema(
        json.dumps(_SCHEMA), separators=(",", ":")
    )
    return xgr.GrammarMatcher(compiled), marker_ids


def _bytes_ids(s: str) -> list[int]:
    return list(s.encode("utf-8"))


def _mock_tokenizer(token_map: dict[str, int]) -> Any:
    """Returns a tokenizer stub whose async ``encode`` maps tokens to ids."""

    async def mock_encode(
        token: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        return np.array([token_map[token]])

    mock = Mock()
    mock.encode = mock_encode
    return mock


class _FakeContext:
    """Context double backed by a real ``GrammarEnforcementState``."""

    def __init__(
        self, prompt_token_ids: list[int], json_schema: str | None
    ) -> None:
        self.tokens = SimpleNamespace(prompt=prompt_token_ids)
        self.grammar = None
        self.json_schema = json_schema
        self.grammar_state = GrammarEnforcementState(
            has_json_schema=json_schema is not None,
            grammar_enforced=json_schema is not None,
        )

    def set_thinking_region(
        self, start_token_ids: list[int] | None, end_token_ids: list[int] | None
    ) -> None:
        # A thinking region is tracked only when its end marker is known.
        if end_token_ids is not None:
            self.grammar_state.thinking_region_delimiters = (
                StructuredOutputRegionDelimiters(
                    start_token_ids=start_token_ids,
                    end_token_ids=end_token_ids,
                )
            )


@pytest.mark.asyncio
async def test_v41_thinking_prompt_arms_suspended_json_schema_region() -> None:
    """A prompt ending in a prefilled ``<think>`` suspends enforcement."""
    _, marker_ids = _compiled_schema_matcher()
    tokenizer = _mock_tokenizer(
        {THINK_START: marker_ids[THINK_START], THINK_END: marker_ids[THINK_END]}
    )

    async def original_new_context(request: _FakeContext) -> _FakeContext:
        return request

    wrapper = _ThinkingRegionNewContext(
        tokenizer, original_new_context, "deepseekv4"
    )

    thinking_mode_prompt = [1, 2, marker_ids[THINK_START]]
    context = await wrapper(
        _FakeContext(thinking_mode_prompt, json.dumps(_SCHEMA))
    )

    assert context.grammar_state._in_thinking_region is True
    assert context.grammar_state.grammar_enforced is False
    assert context.grammar_state.thinking_region_delimiters == (
        StructuredOutputRegionDelimiters(
            start_token_ids=None, end_token_ids=[marker_ids[THINK_END]]
        )
    )


@pytest.mark.asyncio
async def test_v41_chat_mode_prompt_does_not_suspend_enforcement() -> None:
    """A prompt already past ``</think>`` keeps enforcement armed."""
    _, marker_ids = _compiled_schema_matcher()
    tokenizer = _mock_tokenizer(
        {THINK_START: marker_ids[THINK_START], THINK_END: marker_ids[THINK_END]}
    )

    async def original_new_context(request: _FakeContext) -> _FakeContext:
        return request

    wrapper = _ThinkingRegionNewContext(
        tokenizer, original_new_context, "deepseekv4"
    )

    chat_mode_prompt = [1, marker_ids[THINK_START], 5, marker_ids[THINK_END]]
    context = await wrapper(_FakeContext(chat_mode_prompt, json.dumps(_SCHEMA)))

    assert context.grammar_state.thinking_region_delimiters is None
    assert context.grammar_state._in_thinking_region is False
    assert context.grammar_state.grammar_enforced is True


def _armed_state(end_token_id: int) -> GrammarEnforcementState:
    """State as armed for a thinking-mode JSON-schema request."""
    state = GrammarEnforcementState(has_json_schema=True, grammar_enforced=True)
    state.thinking_region_delimiters = StructuredOutputRegionDelimiters(
        start_token_ids=None, end_token_ids=[end_token_id]
    )
    state._in_thinking_region = True
    state.grammar_enforced = False
    return state


def test_reasoning_tokens_excluded_then_final_json_enforced_against_schema() -> (
    None
):
    """Reasoning bypasses the matcher; the final answer is enforced."""
    matcher, marker_ids = _compiled_schema_matcher()
    state = _armed_state(marker_ids[THINK_END])

    # Schema-breaking text that the matcher would reject.
    reasoning_text = "not json at all {[}]"
    for token in _bytes_ids(reasoning_text):
        assert state.update_enforcement_state(token) is False

    # Control: the same text is rejected by a matcher fed directly.
    control_matcher, _ = _compiled_schema_matcher()
    garbage_rejected = False
    for token in _bytes_ids(reasoning_text):
        if not control_matcher.accept_token(token):
            garbage_rejected = True
            break
    assert garbage_rejected

    # The closing delimiter itself is not grammar content.
    assert state.update_enforcement_state(marker_ids[THINK_END]) is False
    assert state._in_thinking_region is False
    assert state.grammar_enforced is True

    final_answer = '{"unit":"celsius"}'
    for token in _bytes_ids(final_answer):
        assert state.update_enforcement_state(token) is True
        assert matcher.accept_token(token)


def test_schema_violating_final_answer_is_rejected_once_enforced() -> None:
    """Once enforcement re-arms, the matcher rejects an out-of-enum value."""
    matcher, marker_ids = _compiled_schema_matcher()
    state = _armed_state(marker_ids[THINK_END])

    state.update_enforcement_state(marker_ids[THINK_END])
    assert state.grammar_enforced is True

    bad_answer = '{"unit":"kelvin"'  # "kelvin" is not in the enum
    rejected = False
    for token in _bytes_ids(bad_answer):
        assert state.update_enforcement_state(token) is True
        if not matcher.accept_token(token):
            rejected = True
            break
    assert rejected
