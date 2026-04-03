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

"""Generic types for tool call parsing.

These types provide a server-agnostic representation of parsed tool calls
that can be translated to specific API schemas (e.g., OpenAI) by the
serving layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ParsedToolCall:
    """A parsed tool/function call extracted from model output.

    Attributes:
        id: Unique identifier for this tool call.
        name: The name of the function to call.
        arguments: The function arguments as a JSON string.
    """

    id: str
    name: str
    arguments: str


@dataclass
class ParsedToolCallDelta:
    """Incremental tool call data for streaming responses.

    Used during streaming to send partial tool call information
    as it becomes available.

    Attributes:
        index: The index of this tool call in the list of tool calls.
        id: The tool call ID (typically sent with the first chunk).
        name: The function name (typically sent with the first chunk).
        arguments: Partial arguments string (streamed incrementally).
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class ParsedToolResponse:
    """Result of parsing a complete model response for tool calls.

    Attributes:
        content: Text content from the response (before/after tool calls).
        tool_calls: List of parsed tool calls extracted from the response.
    """

    content: str | None = None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


class ToolParser(Protocol):
    """Protocol for parsing tool calls from model responses.

    Implementations parse model-specific tool calling formats into generic
    tool call structures. Supports both complete (non-streaming) and
    incremental (streaming) parsing modes.

    Different model architectures use different tool calling formats:
    - Llama models use JSON-based tool calls
    - Kimi K2.5 uses structural tags like <|tool_call_begin|>

    The serving layer is responsible for translating these generic types
    to API-specific schemas (e.g., OpenAI).
    """

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls.

        Args:
            response: The full model response text.

        Returns:
            A ParsedToolResponse containing any text content and parsed
            tool calls.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        ...

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses an incremental token delta for streaming tool calls.

        Accumulates tokens internally and returns tool call deltas when
        complete or partial tool calls can be extracted.

        Args:
            delta: The incremental token(s) to process.

        Returns:
            A list of tool call deltas if any can be extracted,
            or None if more tokens are needed.
        """
        ...

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        ...
