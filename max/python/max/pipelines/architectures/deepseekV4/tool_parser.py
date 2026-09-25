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
# ruff: noqa: RUF002

"""DeepSeek-V4 tool-call parser for the checkpoint's DSML format.

DeepSeek-V4 writes tool calls after its content as::

    \\n\\n<｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
    <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

Parsing is delegated to the checkpoint's reference decoder
(``encoding_dsv4.py``) so the accepted grammar is exactly the official one.
"""

from __future__ import annotations

from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    generate_call_id,
    register,
)
from max.pipelines.modeling.types import ParsedToolCall, ParsedToolResponse

from .encoding_dsv4 import (
    dsml_token,
    eos_token,
    parse_message_from_completion_text,
    parse_tool_calls,
    tool_calls_block_name,
)

_INVOKE_BEGIN = f"<{dsml_token}invoke"
_INVOKE_END = f"</{dsml_token}invoke>"
_PARAMETER_END = f"</{dsml_token}parameter>\n"
_HEADER_END = '">\n'
# Stands in for the real header when only the arguments are decoded.
_PLACEHOLDER_HEADER = f' name="_"{_HEADER_END}'


@register("deepseekv4")
class DeepseekV4ToolParser(StructuralTagToolParser):
    """Parses DeepSeek-V4 ``<｜DSML｜tool_calls>`` blocks into tool calls.

    The router hands this parser content only: the ``deepseekv4`` reasoning
    parser has already moved everything up to ``</think>`` into reasoning.
    A DSML block inside the reasoning is therefore never a tool call. The
    reference decoder rejects such output outright; here it stays verbatim in
    the reasoning.

    The section marker includes the ``\\n\\n`` the reference decoder requires
    before ``<｜DSML｜tool_calls>``, so the content ahead of a call streams
    without it, as it parses. Arguments stream one complete parameter at a
    time; ``string="true|false"`` carries each value's type, so no schema is
    needed.
    """

    SECTION_BEGIN = f"\n\n<{dsml_token}{tool_calls_block_name}>"
    SECTION_END = f"</{dsml_token}{tool_calls_block_name}>"
    CALL_BEGIN = _INVOKE_BEGIN
    CALL_END = _INVOKE_END

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response with the reference decoder.

        Output without any ``｜DSML｜`` token is returned as content, so plain
        replies never meet the decoder's special-token checks. Otherwise the
        reference decoder decides, in chat mode because reasoning was split
        off upstream; the response has its EOS stripped by detokenization, so
        it is restored first.

        Args:
            response: The content the model generated after its reasoning.

        Returns:
            The content before the tool-call block (``None`` when empty) and
            the parsed tool calls.

        Raises:
            ValueError: If the DSML is malformed or holds no tool call.
        """
        if dsml_token not in response:
            return ParsedToolResponse(content=response, tool_calls=[])

        text = (
            response if response.endswith(eos_token) else response + eos_token
        )
        try:
            message = parse_message_from_completion_text(
                text, thinking_mode="chat"
            )
        except AssertionError as e:
            # The reference decoder reports envelope errors with ``assert``.
            raise ValueError(f"Malformed DeepSeek-V4 tool calls: {e}") from e

        tool_calls = [
            ParsedToolCall(
                id=generate_call_id(),
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            )
            for call in message["tool_calls"]
        ]
        if not tool_calls:
            # Raising lets the router drop the empty block from the content.
            raise ValueError("DeepSeek-V4 tool-call block holds no tool call")
        return ParsedToolResponse(
            content=message["content"] or None, tool_calls=tool_calls
        )

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        """Decodes the invokes between the section markers.

        Raises:
            ValueError: If an invoke is malformed.
        """
        _, _, calls = parse_tool_calls(0, f">{tool_section}{self.SECTION_END}")
        return [
            ParsedToolCall(
                id=generate_call_id(),
                name=call["name"],
                arguments=call["arguments"],
            )
            for call in calls
        ]

    def _decode_invoke(self, header: str, parameters: str) -> ParsedToolCall:
        """Decodes one invoke from its header and complete parameter lines."""
        (call,) = self._parse_complete_section(
            f"\n{_INVOKE_BEGIN}{header}{parameters}{_INVOKE_END}\n"
        )
        return call

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits `` name="..."`` plus ``">\\n`` from the parameter lines."""
        end = body.find(_HEADER_END)
        if end == -1:
            return None, None
        end += len(_HEADER_END)
        return body[:end], body[end:]

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        try:
            call = self._decode_invoke(header, "")
        except ValueError:
            return None, None
        return call.id, call.name

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Decodes the complete parameters seen so far into JSON.

        While the invoke is open the closing brace is withheld, so the
        streamed pieces concatenate to the complete parse's arguments. A
        malformed invoke stops streaming its arguments; the complete parse
        reports it.
        """
        if not is_complete:
            end = args_text.rfind(_PARAMETER_END)
            if end == -1:
                return ""
            args_text = args_text[: end + len(_PARAMETER_END)]
        try:
            arguments = self._decode_invoke(
                _PLACEHOLDER_HEADER, args_text
            ).arguments
        except ValueError:
            return ""
        return arguments if is_complete else arguments[:-1]
