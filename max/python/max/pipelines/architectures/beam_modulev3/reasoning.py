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

"""Beam reasoning parser for Harmony channels."""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from max.pipelines.lib.reasoning import Route, RoutedSpan, register
from max.pipelines.lib.tokenizer import convert_token_to_id
from max.pipelines.modeling.types import (
    ParsedReasoningDelta,
    PipelineTokenizer,
    ReasoningParser,
    ReasoningSpan,
)

from . import harmony

_MARKERS = (
    harmony.START,
    harmony.END,
    harmony.MESSAGE,
    harmony.CHANNEL,
    harmony.CONSTRAIN,
    harmony.RETURN,
    harmony.CALL,
)


class _State(enum.Enum):
    # Between messages: header text, which the response template drops.
    HEADER = enum.auto()
    REASONING = enum.auto()
    ANSWER = enum.auto()
    TOOL_CALL = enum.auto()


@register("beam")
class BeamReasoningParser(ReasoningParser):
    """Beam reasoning parser for Harmony channels.

    Matches vLLM's ``--reasoning-parser beam``, which reads the checkpoint's
    ``response_template``: an ``analysis`` message body is reasoning, a
    ``final`` message body is the answer, and the text between messages
    (``<|start|>assistant``, channel headers, anything the model writes
    outside a message) is dropped. A tool-call message, header and closing
    marker included, goes to content whole, so the tool parser sees the
    recipient and where each call ends.

    Markers end a message only as the template spells them, so a reasoning
    ``<|end|>`` or ``<|start|>`` that does not grow into
    ``<|end|><|start|>assistant`` stays reasoning text; until it resolves, it
    is held back, across deltas if needed. Header tokens are held the same
    way until their ``<|message|>`` shows which message they open. A
    reasoning parser cannot release held tokens once the stream ends, so the
    end-of-sequence tokens settle them, as end of output settles vLLM's.
    """

    # The text form, for consumers that split reasoning after detokenizing.
    # stream() is stricter: it ends reasoning on <|end|> only when
    # <|start|>assistant follows.
    REASONING_START: ClassVar[str] = harmony.REASONING_OPEN
    REASONING_END: ClassVar[str] = harmony.END

    def __init__(
        self,
        marker_ids: Sequence[int],
        decode: Callable[[Sequence[int]], str],
    ) -> None:
        if len(marker_ids) != len(_MARKERS):
            raise ValueError(f"expected one token id for each of {_MARKERS}")
        self._text_of = dict(zip(marker_ids, _MARKERS, strict=True))
        (
            self._start,
            self._end,
            self._message,
            _,
            self._constrain,
            self._return,
            self._call,
        ) = marker_ids
        self._decode = decode
        self.reset()

    def reset(self) -> None:
        """Clears the parser back to its initial state."""
        self._state: _State | None = None
        self._prompt_state: _State | None = None
        self._header = ""
        # A tool call's text since a possible close began.
        self._closing = ""
        # The texts of the tokens held at the end of the last delta, and
        # their items by type.
        self._held_texts: list[str] = []
        self._carried: dict[type, list[Any]] = {}

    def _text(self, token_id: int) -> str:
        text = self._text_of.get(token_id)
        return text if text is not None else self._decode([token_id])

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Routes a delta's tokens to reasoning, content or nothing.

        ``is_currently_reasoning`` seeds only the first delta: ``True`` starts
        where the prompt left the turn (the model writes its own channel
        header), ``False`` (a grammar constrains from the first token) starts
        in the answer. After that the channel headers decide.
        """
        if self._state is None:
            state = self._prompt_state or _State.HEADER
            if not is_currently_reasoning and state in (
                _State.HEADER,
                _State.REASONING,
            ):
                state = _State.ANSWER
            self._state = state
        if not delta_token_ids:
            # Keeps the carried items for the next delta.
            return ParsedReasoningDelta(
                span=ReasoningSpan((0, 0), (0, 0)),
                is_still_reasoning=self._is_reasoning(),
            )
        routes = [Route.HOLD] * len(self._held_texts)
        held_at = list(range(len(self._held_texts)))
        for token_id in delta_token_ids:
            routes.append(Route.DROP)
            self._step(token_id, len(routes) - 1, routes, held_at)
        carried, self._carried = self._carried, {}
        return ParsedReasoningDelta(
            span=RoutedSpan(routes, carried, self._carried),
            is_still_reasoning=self._is_reasoning(),
        )

    def _is_reasoning(self) -> bool:
        return self._state in (_State.HEADER, _State.REASONING)

    def _hold(
        self, at: int, text: str, routes: list[Route], held_at: list[int]
    ) -> None:
        routes[at] = Route.HOLD
        held_at.append(at)
        self._held_texts.append(text)

    def _resolve(
        self,
        count: int,
        route: Route,
        routes: list[Route],
        held_at: list[int],
    ) -> None:
        for at in held_at[:count]:
            routes[at] = route
        del held_at[:count], self._held_texts[:count]

    def _drop_all_but_tail(
        self, length: int, routes: list[Route], held_at: list[int]
    ) -> None:
        """Drops the held tokens before those spelling the last ``length`` chars."""
        count, covered = 0, 0
        while count < len(self._held_texts) and covered < length:
            count += 1
            covered += len(self._held_texts[-count])
        self._resolve(len(held_at) - count, Route.DROP, routes, held_at)

    def _close_message(self, rest: str = "") -> None:
        self._state = _State.HEADER
        self._header = rest
        self._closing = ""

    def _step(
        self,
        token_id: int,
        at: int,
        routes: list[Route],
        held_at: list[int],
    ) -> None:
        state = self._state
        if state is _State.HEADER:
            self._step_header(token_id, at, routes, held_at)
        elif state is _State.REASONING:
            self._step_reasoning(token_id, at, routes, held_at)
        elif state is _State.ANSWER:
            if token_id in (self._return, self._end, self._call):
                self._close_message()
            else:
                routes[at] = Route.CONTENT
        else:
            self._step_tool_call(token_id, at, routes)

    def _step_header(
        self,
        token_id: int,
        at: int,
        routes: list[Route],
        held_at: list[int],
    ) -> None:
        if token_id in (self._return, self._call):
            self._resolve(len(held_at), Route.DROP, routes, held_at)
            self._header = ""
            return
        text = self._text(token_id)
        self._hold(at, text, routes, held_at)
        self._header += text
        header = self._header
        if token_id == self._message:
            tool_header = harmony.TOOL_HEADER.search(header)
            if header.endswith(harmony.REASONING_OPEN):
                self._state = _State.REASONING
            elif header.endswith(harmony.ANSWER_OPEN):
                self._state = _State.ANSWER
            elif tool_header is not None and tool_header.end() == len(header):
                self._state = _State.TOOL_CALL
                self._closing = ""
                self._drop_all_but_tail(
                    len(tool_header.group()), routes, held_at
                )
                self._resolve(len(held_at), Route.CONTENT, routes, held_at)
            # A header can only grow past this <|message|> into the next one.
            self._resolve(len(held_at), Route.DROP, routes, held_at)
            self._header = ""
            return
        # Only a tool header's tokens are routed, so the others drop as soon
        # as no tool header can contain them.
        start = harmony.tool_header_start(header)
        keep = len(header) - start if start is not None else 0
        self._drop_all_but_tail(keep, routes, held_at)
        # The other open markers are at most this long, so older text is not
        # needed.
        self._header = header[-max(keep, len(harmony.REASONING_OPEN)) :]

    def _step_reasoning(
        self,
        token_id: int,
        at: int,
        routes: list[Route],
        held_at: list[int],
    ) -> None:
        if token_id in (self._return, self._call):
            # The output ends here, so vLLM's parser would complete a close
            # that ends in the message start and keep any other held text.
            held = "".join(self._held_texts)
            ends = held.startswith((harmony.END, harmony.CONSTRAIN))
            route = Route.DROP if ends else Route.REASONING
            self._resolve(len(held_at), route, routes, held_at)
            return
        if not held_at and token_id not in (
            self._end,
            self._constrain,
            self._start,
        ):
            routes[at] = Route.REASONING
            return
        self._hold(at, self._text(token_id), routes, held_at)
        while held_at:
            held = "".join(self._held_texts)
            close = next(
                (c for c in harmony.REASONING_CLOSES if held.startswith(c)),
                None,
            )
            if close is not None:
                self._resolve(len(held_at), Route.DROP, routes, held_at)
                self._close_message(held[len(close) :])
                return
            if any(c.startswith(held) for c in harmony.REASONING_CLOSES):
                return
            self._resolve(1, Route.REASONING, routes, held_at)

    def _step_tool_call(
        self, token_id: int, at: int, routes: list[Route]
    ) -> None:
        # The whole message goes to content for the tool parser, which reads
        # the same closes; tracking them only finds where the message ends.
        if token_id == self._return:
            return
        routes[at] = Route.CONTENT
        if token_id == self._call:
            self._close_message()
            return
        if not self._closing and token_id != self._end:
            return
        closing = self._closing + self._text(token_id)
        while closing:
            if closing.startswith(harmony.TOOL_CLOSE):
                self._close_message(closing[len(harmony.TOOL_CLOSE) :])
                return
            if harmony.TOOL_CLOSE.startswith(closing):
                break
            end = closing.find(harmony.END, 1)
            closing = closing[end:] if end > 0 else ""
        self._closing = closing

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Returns whether the prompt leaves the turn before or in reasoning.

        Reads the prompt from its last ``<|start|>assistant`` on, as vLLM's
        parser reads its prefix. The chat template's generation prompt ends
        there, leaving the model to write its channel header. A prompt with
        no assistant turn, such as a raw ``/v1/completions`` prompt, does not
        reason: its whole output is the answer.
        """
        self._prompt_state = None
        ids = list(prompt_token_ids)
        for i in range(len(ids) - 2, -1, -1):
            if ids[i] == self._start and self._text(ids[i + 1]) == harmony.ROLE:
                scratch = BeamReasoningParser(list(self._text_of), self._decode)
                scratch.stream(ids[i + 2 :])
                self._prompt_state = scratch._state
                break
        return self._prompt_state in (_State.HEADER, _State.REASONING)

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> BeamReasoningParser:
        """Constructs a reasoning parser from a tokenizer."""
        ids = [await convert_token_to_id(tokenizer, m) for m in _MARKERS]
        missing = [m for m, i in zip(_MARKERS, ids, strict=True) if i is None]
        if missing:
            raise ValueError(
                f"{cls.__name__} could not locate {missing} in the tokenizer"
            )
        # stream() is sync and PipelineTokenizer.decode is async, so header
        # tokens are decoded with the HF tokenizer the pipeline tokenizer wraps.
        delegate = getattr(tokenizer, "delegate", None)
        if delegate is None:
            raise ValueError(
                f"{cls.__name__} needs a tokenizer with an HF `delegate`"
            )
        return cls([i for i in ids if i is not None], delegate.decode)

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns ``None``: no single token ends Beam's reasoning.

        ``<|end|>`` ends the ``analysis`` message, but the model then writes a
        ``final`` header that a grammar released at ``<|end|>`` would reject,
        so a constrained request starts in the answer instead.
        """
        return None
