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

"""Muse Glimmer reasoning parser for ``to=self`` channels."""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypeVar

from max.pipelines.lib.reasoning import register
from max.pipelines.lib.tokenizer import convert_token_to_id
from max.pipelines.modeling.types import (
    ParsedReasoningDelta,
    PipelineTokenizer,
    ReasoningParser,
    ReasoningSpan,
)

_T = TypeVar("_T")

_HEADER_RECIPIENT = re.compile(r"to=([^\s<]+)")
_BARE_HEADER = re.compile(r"\s*to=([^\s<]+)")
# A body tail that may still grow into a bare ``to=NAME`` header.
_HEADER_PREFIX = re.compile(r"\s*(?:t|to|to=[^\s<]*)?")


class _Channel(enum.Enum):
    CLOSED = enum.auto()
    HEADER = enum.auto()
    REASONING = enum.auto()
    CONTENT = enum.auto()


class _Route(enum.Enum):
    REASONING = enum.auto()
    CONTENT = enum.auto()
    DROP = enum.auto()
    HOLD = enum.auto()


class _ChannelSpan(ReasoningSpan):
    """Routes each item of the carried-over items plus the delta.

    Items held back by the previous delta are prepended per element type,
    since llm.py extracts token ids and log-probabilities from separate
    sequences of the same length.
    """

    def __init__(
        self,
        routes: list[_Route],
        carried: dict[type, list[Any]],
        held: dict[type, list[Any]],
    ) -> None:
        super().__init__((0, 0), (0, 0))
        self._routes = routes
        self._carried = carried
        self._held = held

    def _select(self, seq: Sequence[_T], route: _Route) -> list[_T]:
        if not seq:
            return []
        key = type(seq[0])
        routed = list(
            zip(
                self._carried.get(key, []) + list(seq),
                self._routes,
                strict=True,
            )
        )
        self._held[key] = [x for x, r in routed if r is _Route.HOLD]
        return [x for x, r in routed if r is route]

    def extract_content(self, seq: Sequence[_T]) -> list[_T]:
        return self._select(seq, _Route.CONTENT)

    def extract_reasoning(self, seq: Sequence[_T]) -> list[_T]:
        return self._select(seq, _Route.REASONING)


@register("muse_glimmer")
class MuseGlimmerReasoningParser(ReasoningParser):
    """Muse Glimmer reasoning parser for channelled assistant output.

    Every assistant message is ``<|start|>assistant to=RECIPIENT<|message|>
    BODY`` closed by ``<|eom|>`` or ``<|eot|>``. ``to=self`` bodies are
    reasoning; ``to=user`` or no recipient is the answer; any other recipient
    is a tool channel whose body goes to content for the tool parser. Headers
    and framing tokens are dropped.

    The model sometimes leaves reasoning without ``<|eom|>`` and writes a bare
    `` to=NAME<|message|>`` header inside the body, so body tokens that may
    still become such a header are held back, across deltas if needed, until
    they are resolved.
    """

    REASONING_START: ClassVar[str] = "to=self<|message|>"
    REASONING_END: ClassVar[str] = "<|eom|>"

    def __init__(
        self,
        start_token_id: int,
        message_token_id: int,
        eom_token_id: int,
        eot_token_id: int,
        assistant_token_id: int,
        decode: Callable[[Sequence[int]], str],
    ) -> None:
        self.start_token_id = start_token_id
        self.message_token_id = message_token_id
        self.eom_token_id = eom_token_id
        self.eot_token_id = eot_token_id
        self.assistant_token_id = assistant_token_id
        self._boundary_ids = {start_token_id, eom_token_id, eot_token_id}
        self._decode = decode
        self.reset()

    def reset(self) -> None:
        self._channel: _Channel | None = None
        self._header: list[int] = []
        # Ids held at the end of the last delta, and their items by type.
        self._held: list[int] = []
        self._carried: dict[type, list[Any]] = {}

    def _open(self, recipient: str | None) -> None:
        self._channel = (
            _Channel.REASONING if recipient == "self" else _Channel.CONTENT
        )

    def _body_route(self) -> _Route:
        return (
            _Route.REASONING
            if self._channel is _Channel.REASONING
            else _Route.CONTENT
        )

    def _is_reasoning(self) -> bool:
        return self._channel in (
            _Channel.CLOSED,
            _Channel.HEADER,
            _Channel.REASONING,
        )

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Routes a delta's tokens to reasoning, content or nothing.

        ``is_currently_reasoning`` seeds only the first delta: ``True`` means
        the prompt ended inside an assistant header, ``False`` (no reasoning
        expected, or a grammar constrains from the first token) starts in the
        answer. After that the channel headers decide.
        """
        if self._channel is None:
            self._channel = (
                _Channel.HEADER if is_currently_reasoning else _Channel.CONTENT
            )
        if not delta_token_ids:
            # Keeps the carried items for the next delta.
            return ParsedReasoningDelta(
                span=ReasoningSpan((0, 0), (0, 0)),
                is_still_reasoning=self._is_reasoning(),
            )
        held_ids = self._held
        held_at = list(range(len(held_ids)))
        routes = [_Route.HOLD] * len(held_ids)

        def resolve_held(count: int, route: _Route) -> None:
            for i in held_at[:count]:
                routes[i] = route
            del held_at[:count], held_ids[:count]

        for token_id in delta_token_ids:
            routes.append(_Route.DROP)
            if (
                self._channel is _Channel.CLOSED
                and token_id not in self._boundary_ids
            ):
                # A grammar enforced from <|eom|> makes the model answer
                # without a <|start|> header.
                self._channel = _Channel.CONTENT
            if token_id in self._boundary_ids:
                resolve_held(len(held_at), self._body_route())
                self._channel = (
                    _Channel.HEADER
                    if token_id == self.start_token_id
                    else _Channel.CLOSED
                )
                self._header = []
            elif token_id == self.message_token_id:
                if self._channel is _Channel.HEADER:
                    match = _HEADER_RECIPIENT.search(self._decode(self._header))
                    self._open(match.group(1) if match else None)
                elif match := _BARE_HEADER.fullmatch(self._decode(held_ids)):
                    resolve_held(len(held_at), _Route.DROP)
                    self._open(match.group(1))
                else:
                    resolve_held(len(held_at), self._body_route())
            elif self._channel is _Channel.HEADER:
                self._header.append(token_id)
            else:
                routes[-1] = _Route.HOLD
                held_at.append(len(routes) - 1)
                held_ids.append(token_id)
                while held_ids and not _HEADER_PREFIX.fullmatch(
                    self._decode(held_ids)
                ):
                    resolve_held(1, self._body_route())

        carried, self._carried = self._carried, {}
        return ParsedReasoningDelta(
            span=_ChannelSpan(routes, carried, self._carried),
            is_still_reasoning=self._is_reasoning(),
        )

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Returns whether the prompt ends with ``<|start|>assistant``.

        The chat template's generation prompt stops there, leaving the model
        to write the recipient; ``to=self`` then opens reasoning.
        """
        return list(prompt_token_ids[-2:]) == [
            self.start_token_id,
            self.assistant_token_id,
        ]

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> MuseGlimmerReasoningParser:
        """Constructs a reasoning parser from a tokenizer."""
        names = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "assistant")
        ids = [await convert_token_to_id(tokenizer, name) for name in names]
        missing = [n for n, i in zip(names, ids, strict=True) if i is None]
        if missing:
            raise ValueError(
                f"{cls.__name__} could not locate {missing} in the tokenizer"
            )
        # stream() is sync and PipelineTokenizer.decode is async, so headers
        # are decoded with the HF tokenizer the pipeline tokenizer wraps.
        delegate = getattr(tokenizer, "delegate", None)
        if delegate is None:
            raise ValueError(
                f"{cls.__name__} needs a tokenizer with an HF `delegate`"
            )
        start, message, eom, eot, assistant = (i for i in ids if i is not None)
        return cls(start, message, eom, eot, assistant, delegate.decode)

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the ``<|eom|>`` token id, which closes a ``to=self`` body."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
