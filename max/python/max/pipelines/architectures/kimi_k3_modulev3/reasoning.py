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

"""Kimi K3 reasoning parser for its ``think`` frame."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from max.pipelines.lib.reasoning import Route, RoutedSpan, register
from max.pipelines.lib.tokenizer import convert_token_to_id
from max.pipelines.modeling.types import (
    ParsedReasoningDelta,
    PipelineTokenizer,
    ReasoningParser,
)


@register("kimi_k3")
class KimiK3ReasoningParser(ReasoningParser):
    """Kimi K3 reasoning parser for its ``think`` frame.

    K3 frames each segment as ``<|open|>header<|sep|>...<|close|>name<|sep|>``,
    where the header is ordinary text that starts with the frame name, and
    ends the turn with ``<|end_of_msg|>``. The tokenizer does not mark these delimiters as
    special, so the parser drops every frame token from both reasoning and
    content.
    """

    REASONING_START: ClassVar[str] = "<|open|>think<|sep|>"
    REASONING_END: ClassVar[str] = "<|close|>think<|sep|>"

    def __init__(
        self,
        open_id: int,
        close_id: int,
        sep_id: int,
        end_of_msg_id: int,
        think_id: int,
    ) -> None:
        self.open_id = open_id
        self.close_id = close_id
        self.sep_id = sep_id
        self.think_id = think_id
        self._delimiters = frozenset((open_id, close_id, sep_id, end_of_msg_id))
        # The open or close tag of the header being read, up to its ``<|sep|>``.
        self._header_tag: int | None = None
        self._header_len = 0

    def reset(self) -> None:
        """Clears the frame-header state carried across chunks."""
        self._header_tag = None
        self._header_len = 0

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Routes each token of a streaming chunk to reasoning, content or drop."""
        reasoning = is_currently_reasoning
        routes: list[Route] = []
        for token_id in delta_token_ids:
            if token_id in (self.open_id, self.close_id):
                self._header_tag = token_id
                self._header_len = 0
                routes.append(Route.DROP)
            elif self._header_tag is not None:
                if token_id == self.sep_id:
                    self._header_tag = None
                elif self._header_len == 0 and token_id == self.think_id:
                    reasoning = self._header_tag == self.open_id
                self._header_len += 1
                routes.append(Route.DROP)
            elif token_id in self._delimiters:
                routes.append(Route.DROP)
            else:
                routes.append(Route.REASONING if reasoning else Route.CONTENT)

        # The parser holds no tokens back, so nothing is carried between
        # chunks.
        return ParsedReasoningDelta(
            span=RoutedSpan(routes, carried={}, held={}),
            is_still_reasoning=reasoning,
        )

    def will_reason_after_prompt(self, prompt_token_ids: Sequence[int]) -> bool:
        """Returns whether the prompt's last opened or closed frame is an open ``think``."""
        for i in range(len(prompt_token_ids) - 2, -1, -1):
            if prompt_token_ids[i] in (self.open_id, self.close_id):
                return (
                    prompt_token_ids[i] == self.open_id
                    and prompt_token_ids[i + 1] == self.think_id
                )
        return False

    @classmethod
    async def from_tokenizer(
        cls, tokenizer: PipelineTokenizer[Any, Any, Any]
    ) -> KimiK3ReasoningParser:
        """Constructs a reasoning parser from a tokenizer."""
        names = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>", "think")
        ids = [await convert_token_to_id(tokenizer, name) for name in names]
        if any(token_id is None for token_id in ids):
            raise ValueError(
                f"{cls.__name__} could not locate {names} as single tokens"
            )
        open_id, close_id, sep_id, end_of_msg_id, think_id = ids
        assert open_id is not None
        assert close_id is not None
        assert sep_id is not None
        assert end_of_msg_id is not None
        assert think_id is not None
        return cls(open_id, close_id, sep_id, end_of_msg_id, think_id)

    @classmethod
    async def reasoning_end_token_id(
        cls, tokenizer: PipelineTokenizer[Any, Any, Any]
    ) -> int | None:
        """Returns ``None``: the reasoning end is a multi-token frame."""
        return None
