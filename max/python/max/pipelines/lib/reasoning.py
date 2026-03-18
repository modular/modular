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

"""Reasoning parsers for identifying reasoning spans in model output."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from max.interfaces import PipelineTokenizer, ReasoningParser, ReasoningSpan

_REASONING_PARSERS: dict[str, type[ReasoningParser]] = {}

_GPT_OSS_INLINE_FINAL_PATTERN = re.compile(
    r"^\s*analysis(?P<reasoning>.*?)(?:assistant\s*final|assistantfinal)(?P<content>.*)$",
    re.DOTALL,
)
_GPT_OSS_REASONING_SUFFIX_FINAL_PATTERN = re.compile(
    r"^(?P<reasoning>.*?)(?:assistant\s*final|assistantfinal)(?P<content>.*)$",
    re.DOTALL,
)


def postprocess_decoded_reasoning(
    parser_name: str | None, decoded_reasoning: str | None
) -> str | None:
    """Clean up decoded reasoning text for parser-specific channel artifacts."""
    if decoded_reasoning is None:
        return None
    if parser_name == "gpt_oss_harmony" and decoded_reasoning.startswith(
        "analysis"
    ):
        return decoded_reasoning.removeprefix("analysis").lstrip()
    return decoded_reasoning


def postprocess_decoded_content_and_reasoning(
    parser_name: str | None,
    decoded_content: str | None,
    decoded_reasoning: str | None,
) -> tuple[str | None, str | None]:
    """Apply parser-specific cleanup to decoded content and reasoning text."""
    decoded_reasoning = postprocess_decoded_reasoning(
        parser_name, decoded_reasoning
    )
    if parser_name != "gpt_oss_harmony":
        return decoded_content, decoded_reasoning

    if decoded_reasoning:
        match = _GPT_OSS_REASONING_SUFFIX_FINAL_PATTERN.match(
            decoded_reasoning
        )
        if match is not None:
            reasoning_text = match.group("reasoning").strip() or None
            content_text = match.group("content").strip() or None
            if decoded_content:
                merged_content = (
                    f"{content_text}{decoded_content}"
                    if content_text is not None
                    else decoded_content
                )
            else:
                merged_content = content_text
            return merged_content, reasoning_text

    if decoded_content is None or decoded_reasoning is not None:
        return decoded_content, decoded_reasoning

    match = _GPT_OSS_INLINE_FINAL_PATTERN.match(decoded_content)
    if match is None:
        return decoded_content, decoded_reasoning

    reasoning_text = match.group("reasoning").strip() or None
    content_text = match.group("content").strip() or None
    return content_text, reasoning_text


def register(
    name: str,
) -> Callable[[type[ReasoningParser]], type[ReasoningParser]]:
    """Class decorator that registers a ReasoningParser under the given name."""

    def decorator(cls: type[ReasoningParser]) -> type[ReasoningParser]:
        _REASONING_PARSERS[name] = cls
        return cls

    return decorator


async def create(
    name: str,
    tokenizer: PipelineTokenizer[Any, Any, Any],
) -> ReasoningParser:
    """Look up a registered parser by name and construct it from a tokenizer."""
    cls = _REASONING_PARSERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown reasoning parser: {name!r}. "
            f"Available: {sorted(_REASONING_PARSERS)}"
        )
    return await cls.from_tokenizer(tokenizer)


async def _convert_token_to_id(
    tokenizer: PipelineTokenizer[Any, Any, Any],
    token: str,
) -> int | None:
    """Convert a token string to its token ID, or None if not a single token."""
    # Workaround: PipelineTokenizer does not expose convert_tokens_to_ids(),
    # so we encode the string and verify it maps to exactly one token ID.
    encoded = await tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1:
        return None
    return int(encoded[0])


@register("kimik2_5")
class KimiK2_5ReasoningParser(ReasoningParser):
    """Kimi K2.5 reasoning parser for <think>...</think> sections.

    Reasoning may end implicitly when a tool call section begins
    (<|tool_calls_section_begin|>).

    Reasoning may begin implicitly, without an explicit <think> token.

    Reasoning can be disabled through the chat template by including a </think>
    token in the prompt.
    """

    def __init__(
        self,
        think_start_token_id: int,
        think_end_token_id: int,
        tool_section_start_token_id: int | None = None,
    ) -> None:
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id
        self.tool_section_start_token_id = tool_section_start_token_id

    def stream(
        self,
        delta_token_ids: Sequence[int],
    ) -> tuple[ReasoningSpan, bool]:
        """Identify a reasoning span within a streaming delta chunk."""
        end_token_ids = (
            (self.think_end_token_id, self.tool_section_start_token_id)
            if self.tool_section_start_token_id is not None
            else (self.think_end_token_id,)
        )

        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.think_start_token_id
            ):
                # Take the earliest start token
                start_token_idx = i
            elif token_id in end_token_ids:
                # Take the earliest end token
                end_token_idx = i
                break

        if start_token_idx is None:
            start_reasoning = 0
            start_reasoning_with_delimiters = 0
        else:
            start_reasoning = start_token_idx + 1
            start_reasoning_with_delimiters = start_token_idx

        if end_token_idx is None:
            end_reasoning = len(delta_token_ids)
            end_reasoning_with_delimiters = len(delta_token_ids)
        else:
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1

        span = ReasoningSpan(
            reasoning_with_delimiters=(
                start_reasoning_with_delimiters,
                end_reasoning_with_delimiters,
            ),
            reasoning=(start_reasoning, end_reasoning),
        )
        is_still_reasoning = end_token_idx is None
        return span, is_still_reasoning

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> KimiK2_5ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        think_start_id = await _convert_token_to_id(tokenizer, "<think>")
        think_end_id = await _convert_token_to_id(tokenizer, "</think>")

        if think_start_id is None or think_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate think start/end tokens in the tokenizer"
            )

        tool_section_start_id = await _convert_token_to_id(
            tokenizer, "<|tool_calls_section_begin|>"
        )

        return cls(
            think_start_token_id=think_start_id,
            think_end_token_id=think_end_id,
            tool_section_start_token_id=tool_section_start_id,
        )


@register("gpt_oss_harmony")
class GptOssHarmonyReasoningParser(ReasoningParser):
    """GPT-OSS Harmony parser for ``analysis`` and ``final`` assistant channels.

    GPT-OSS emits assistant responses in Harmony format, typically beginning
    with:

    ``<|start|>assistant<|channel|>analysis<|message|>``

    and later switching to the visible response with:

    ``<|start|>assistant<|channel|>final<|message|>``

    The server should expose the ``analysis`` channel as reasoning and the
    ``final`` channel as the response content.
    """

    def __init__(
        self,
        start_token_id: int,
        channel_token_id: int,
        message_token_id: int,
        end_token_id: int,
        assistant_token_id: int,
        analysis_token_id: int,
        final_token_id: int,
    ) -> None:
        self._analysis_header = [
            start_token_id,
            assistant_token_id,
            channel_token_id,
            analysis_token_id,
            message_token_id,
        ]
        self._analysis_token_id = analysis_token_id
        self._final_header = [
            start_token_id,
            assistant_token_id,
            channel_token_id,
            final_token_id,
            message_token_id,
        ]
        self._final_header_with_end = [end_token_id, *self._final_header]
        self._analysis_header_consumed = False
        self._is_still_reasoning = True
        self._emitted_reasoning = False

    @staticmethod
    def _find_subsequence(
        tokens: Sequence[int], needle: Sequence[int], start: int = 0
    ) -> int | None:
        if len(needle) == 0:
            return start
        max_start = len(tokens) - len(needle)
        for idx in range(start, max_start + 1):
            if list(tokens[idx : idx + len(needle)]) == list(needle):
                return idx
        return None

    def _is_analysis_header_prefix(self, tokens: Sequence[int]) -> bool:
        return len(tokens) <= len(self._analysis_header) and list(tokens) == (
            self._analysis_header[: len(tokens)]
        )

    def stream(
        self,
        delta_token_ids: Sequence[int],
    ) -> tuple[ReasoningSpan, bool]:
        if not self._is_still_reasoning:
            span = ReasoningSpan(
                reasoning_with_delimiters=(0, 0), reasoning=(0, 0)
            )
            return span, False

        tokens = list(delta_token_ids)
        if len(tokens) == 0:
            span = ReasoningSpan(
                reasoning_with_delimiters=(0, 0), reasoning=(0, 0)
            )
            return span, True

        reasoning_start = 0
        header_prefix_only = False
        if not self._analysis_header_consumed:
            if tokens[: len(self._analysis_header)] == self._analysis_header:
                reasoning_start = len(self._analysis_header)
            elif tokens[0] == self._analysis_token_id:
                # Some decode paths suppress Harmony control tokens but still
                # surface the channel name token. Do not leak bare "analysis"
                # into the reasoning text.
                reasoning_start = 1
            elif self._is_analysis_header_prefix(tokens):
                header_prefix_only = True
            self._analysis_header_consumed = True
        elif not self._emitted_reasoning and tokens[0] == self._analysis_token_id:
            # Header/control tokens may be split across stream chunks. If we
            # have not emitted any reasoning payload yet, still strip a leading
            # bare "analysis" channel token.
            reasoning_start = 1

        final_idx = self._find_subsequence(
            tokens, self._final_header_with_end, start=reasoning_start
        )
        final_header_len = len(self._final_header_with_end)
        if final_idx is None:
            final_idx = self._find_subsequence(
                tokens, self._final_header, start=reasoning_start
            )
            final_header_len = len(self._final_header)

        if final_idx is None:
            span = ReasoningSpan(
                reasoning_with_delimiters=(0, len(tokens)),
                reasoning=(reasoning_start, len(tokens)),
            )
            emitted_reasoning = reasoning_start < len(tokens) and not header_prefix_only
            self._emitted_reasoning = self._emitted_reasoning or emitted_reasoning
            return span, True

        self._is_still_reasoning = False
        span = ReasoningSpan(
            reasoning_with_delimiters=(0, final_idx + final_header_len),
            reasoning=(reasoning_start, final_idx),
        )
        self._emitted_reasoning = self._emitted_reasoning or (
            reasoning_start < final_idx
        )
        return span, False

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> GptOssHarmonyReasoningParser:
        token_names = {
            "<|start|>": "start",
            "<|channel|>": "channel",
            "<|message|>": "message",
            "<|end|>": "end",
            "assistant": "assistant",
            "analysis": "analysis",
            "final": "final",
        }

        ids: dict[str, int] = {}
        for token, name in token_names.items():
            token_id = await _convert_token_to_id(tokenizer, token)
            if token_id is None:
                raise ValueError(
                    f"{cls.__name__} could not locate required Harmony token {token!r}"
                )
            ids[name] = token_id

        return cls(
            start_token_id=ids["start"],
            channel_token_id=ids["channel"],
            message_token_id=ids["message"],
            end_token_id=ids["end"],
            assistant_token_id=ids["assistant"],
            analysis_token_id=ids["analysis"],
            final_token_id=ids["final"],
        )
