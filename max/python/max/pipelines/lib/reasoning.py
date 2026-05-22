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

from collections.abc import Callable
from typing import Any

from max.pipelines.modeling.types import PipelineTokenizer, ReasoningParser

__all__ = ["get_parser_cls"]

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
    class-level hooks (e.g., :meth:`ReasoningParser.reasoning_end_token_id`)
    without instantiating the parser.
    """
    if name is None:
        return None
    return _REASONING_PARSERS.get(name)
