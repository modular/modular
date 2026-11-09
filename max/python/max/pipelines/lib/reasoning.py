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

import enum
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from max.pipelines.modeling.types import (
    PipelineTokenizer,
    ReasoningParser,
    ReasoningSpan,
)

__all__ = ["get_parser_cls"]

_T = TypeVar("_T")


class Route(enum.Enum):
    """Where a token-routing reasoning parser sends one token.

    ``HOLD`` keeps a token back until a later delta shows where it goes.
    """

    REASONING = enum.auto()
    CONTENT = enum.auto()
    DROP = enum.auto()
    HOLD = enum.auto()


class RoutedSpan(ReasoningSpan):
    """A reasoning span that routes each token on its own.

    Covers the items the previous delta held back followed by this delta's,
    one :class:`Route` each. Held items are prepended per element type, since
    llm.py extracts token ids and log-probabilities from separate sequences of
    the same length, and the ones this delta holds are stored in ``held``.

    Args:
        routes: One route per carried and delta item, in order.
        carried: The items the previous delta held, by element type.
        held: Where this span stores the items it holds, by element type.
    """

    def __init__(
        self,
        routes: list[Route],
        carried: dict[type, list[Any]],
        held: dict[type, list[Any]],
    ) -> None:
        super().__init__((0, 0), (0, 0))
        self._routes = routes
        self._carried = carried
        self._held = held

    def _select(self, seq: Sequence[_T], route: Route) -> list[_T]:
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
        self._held[key] = [x for x, r in routed if r is Route.HOLD]
        return [x for x, r in routed if r is route]

    def extract_content(self, seq: Sequence[_T]) -> list[_T]:
        """Returns the items routed to content."""
        return self._select(seq, Route.CONTENT)

    def extract_reasoning(self, seq: Sequence[_T]) -> list[_T]:
        """Returns the items routed to reasoning."""
        return self._select(seq, Route.REASONING)


_REASONING_PARSERS: dict[str, type[ReasoningParser]] = {}


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


def get_parser_cls(name: str | None) -> type[ReasoningParser] | None:
    """Look up a registered reasoning parser *class* by name.

    Returns ``None`` if no parser is registered under that name, or if
    ``name`` is ``None``. Useful for callers that need to invoke
    class-level hooks (such as :meth:`ReasoningParser.reasoning_end_token_id`)
    without instantiating the parser.
    """
    if name is None:
        return None
    return _REASONING_PARSERS.get(name)
