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

"""Tool call parser for MiniMax M2 models.

MiniMax M2 uses an XML-style format for tool calls:

.. code-block:: xml

    <minimax:tool_call>
    <invoke name="function_name">
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
    </invoke>
    </minimax:tool_call>

Reference: vllm/tool_parsers/minimax_m2_tool_parser.py
"""

from __future__ import annotations

import json
import re
from typing import Any

from llguidance import LLMatcher
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    escape_for_lark_string,
    generate_call_id,
    names_from_tools,
    register,
)
from max.pipelines.modeling.types import ParsedToolCall

# Regex patterns for complete parsing.
_INVOKE_PATTERN = re.compile(r"<invoke name=([^>]+)>(.*?)</invoke>", re.DOTALL)
_PARAMETER_PATTERN = re.compile(
    r"<parameter name=([^>]+)>(.*?)</parameter>", re.DOTALL
)


def _extract_name(name_str: str) -> str:
    """Extracts a name from a possibly-quoted string."""
    name_str = name_str.strip()
    if (name_str.startswith('"') and name_str.endswith('"')) or (
        name_str.startswith("'") and name_str.endswith("'")
    ):
        return name_str[1:-1]
    return name_str


def _convert_value(value_str: str) -> Any:
    """Converts a parameter value string to a Python object."""
    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        return value_str


def _parse_parameters(body: str) -> dict[str, Any]:
    """Parses all ``<parameter>`` blocks from an invoke body into a dict."""
    params: dict[str, Any] = {}
    for match in _PARAMETER_PATTERN.finditer(body):
        name = _extract_name(match.group(1))
        value_str = match.group(2).strip()
        params[name] = _convert_value(value_str)
    return params


@register("minimax_m2")
class MinimaxM2ToolParser(StructuralTagToolParser):
    """Parses MiniMax M2-style tool calls from model responses.

    MiniMax M2 wraps tool calls in ``<minimax:tool_call>`` and uses
    ``<invoke name=...>`` / ``</invoke>`` for individual calls. The base
    class drives buffer accumulation and section/call iteration; we
    customize the body split (``"name">`` is the header, the rest is
    parameter XML) and provide a structured-to-JSON conversion so the
    base's argument diffing can produce monotonically-growing JSON.
    """

    SECTION_BEGIN = "<minimax:tool_call>"
    SECTION_END = "</minimax:tool_call>"
    CALL_BEGIN = "<invoke name="
    CALL_END = "</invoke>"

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for invoke_match in _INVOKE_PATTERN.finditer(tool_section):
            name_attr = invoke_match.group(1)
            invoke_body = invoke_match.group(2)

            func_name = _extract_name(name_attr)
            if not func_name:
                continue
            params = _parse_parameters(invoke_body)

            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=func_name,
                    arguments=json.dumps(params),
                )
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits ``"name">parameters...`` into (name_attr, params_body)."""
        gt_pos = body.find(">")
        if gt_pos == -1:
            return None, None
        return body[:gt_pos].strip(), body[gt_pos + 1 :]

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        """Strips quotes from the ``name=...`` attribute value."""
        if not header:
            return None, None
        name = _extract_name(header)
        if not name:
            return None, None
        return generate_call_id(), name

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Builds a growing JSON string from complete ``<parameter>`` blocks.

        Returns JSON *without* the closing brace while the invoke is
        still streaming, so that successive argument diffs concatenate
        into valid JSON when the closing brace finally lands.
        """
        params = _parse_parameters(args_text)

        if not params:
            return "{}" if is_complete else ""

        parts = [
            f"{json.dumps(name)}: {json.dumps(value)}"
            for name, value in params.items()
        ]
        inner = ", ".join(parts)
        return "{" + inner + ("}" if is_complete else "")

    # ----- Constrained decoding grammar (MiniMax-specific) ------------------

    @staticmethod
    def _build_invoke_name_terminal(tool_names: list[str] | None) -> str:
        """Builds the Lark INVOKE_NAME terminal declaration.

        When ``tool_names`` is provided, returns an alternation of Lark
        string literals restricting the model to those exact names.
        Otherwise returns a permissive regex terminal.

        Args:
            tool_names: Optional allow-list of function names. ``None``
                accepts any double-quote-free name.
        """
        if tool_names is not None:
            quoted = [f'"{escape_for_lark_string(n)}"' for n in tool_names]
            return "INVOKE_NAME: " + " | ".join(quoted)
        return r'INVOKE_NAME: /[^"]+/'

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generates a Lark grammar for MiniMax M2 constrained decoding.

        Uses Lark rules for the envelope structure and regex terminals only
        at leaves. Special tokens ``<minimax:tool_call>`` and
        ``</minimax:tool_call>`` (IDs 200052/200053) are referenced as
        single-token symbols to avoid llguidance ParserTooComplex on
        multi-byte literal matches.

        Parameter values are captured via a ``param_value`` rule with
        ``suffix="</parameter>"`` so that the ``</parameter>`` boundary is
        handled atomically at the grammar level.  This allows values to
        contain literal ``<`` characters (HTML, JSON, comparison operators,
        etc.) without restriction.

        No trailing free-form text is allowed after the closing
        ``</minimax:tool_call>`` tag: under constrained decoding the model
        is expected to emit EOS immediately after the envelope.

        When ``response_format_schema`` is provided, adds an alternative
        branch that accepts JSON content matching the schema.

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a JSON response matching the schema.
            tools: Optional list of OpenAI-style tool dicts. ``None`` accepts
                any tool name.
            **kwargs: Ignored; accepts ``tokenizer`` and other future kwargs.

        Returns:
            A grammar string compatible with ``LLMatcher``.
        """
        tool_names = names_from_tools(tools)
        invoke_name_terminal = MinimaxM2ToolParser._build_invoke_name_terminal(
            tool_names
        )

        # Shared rules and terminals — same for both branches.
        rules = [
            "invokes: invoke_block+ NL",
            'invoke_block: NL "<invoke name=\\"" INVOKE_NAME "\\">" param_line* NL "</invoke>"',
            'param_line: NL "<parameter name=\\"" PARAM_NAME "\\">" param_value',
            'param_value[suffix="</parameter>"]: /[\\s\\S]*/',
        ]
        terminals = [
            invoke_name_terminal,
            r'PARAM_NAME: /[^"]+/',
            r"NL: /\n/",
        ]

        if response_format_schema is None:
            start_rule = (
                "start: <minimax:tool_call> invokes </minimax:tool_call>"
            )
            extra_rules: list[str] = []
        else:
            schema_str = json.dumps(response_format_schema)
            start_rule = "start: tool_calls_branch | json_response"
            extra_rules = [
                (
                    "tool_calls_branch: <minimax:tool_call> invokes "
                    "</minimax:tool_call>"
                ),
                f"json_response: %json {schema_str}",
            ]

        lark = (
            "\n".join(
                [
                    "%llguidance {}",
                    start_rule,
                    *rules,
                    *extra_rules,
                    *terminals,
                ]
            )
            + "\n"
        )
        return LLMatcher.grammar_from_lark(lark)
