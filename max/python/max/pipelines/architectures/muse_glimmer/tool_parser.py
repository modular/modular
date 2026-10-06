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

"""Tool call parser for Muse Glimmer models.

Muse Glimmer's chat template writes each tool call as an ATEM block:

.. code-block:: xml

    <atem:function_calls>
    <atem:invoke name="function_name">
    <atem:parameter name="param1">value1</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

Strings and scalars are written as is, lists and objects as JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    build_xgrammar_tool_grammar,
)
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    generate_call_id,
    register,
)
from max.pipelines.modeling.types import ParsedToolCall

_PARAMETER_PATTERN = re.compile(
    r'<atem:parameter name="(?P<key>[^"]+)"[^>]*>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)


def _convert_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_parameters(body: str) -> dict[str, Any]:
    # The template does not pad values and says string spaces are not
    # stripped, so the raw value is kept byte for byte.
    return {
        m["key"]: _convert_value(m["value"])
        for m in _PARAMETER_PATTERN.finditer(body)
    }


def _extract_name(header: str) -> str:
    return header.strip().strip('"')


@register("muse_glimmer")
class MuseGlimmerToolParser(StructuralTagToolParser):
    """Parses Muse Glimmer ``<atem:function_calls>`` tool calls."""

    SECTION_BEGIN = "<atem:function_calls>"
    SECTION_END = "</atem:function_calls>"
    CALL_BEGIN = "<atem:invoke name="
    CALL_END = "</atem:invoke>"

    _INVOKE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        re.escape(CALL_BEGIN)
        + r'"(?P<name>[^"]*)">(?P<body>.*?)'
        + re.escape(CALL_END),
        re.DOTALL,
    )

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        return [
            ParsedToolCall(
                id=generate_call_id(),
                name=m["name"],
                arguments=json.dumps(_parse_parameters(m["body"])),
            )
            for m in self._INVOKE_PATTERN.finditer(tool_section)
            if m["name"]
        ]

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits ``"name">parameters...`` into (name_attr, params_body)."""
        header, sep, args = body.partition(">")
        if not sep:
            return None, None
        return header, args

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        name = _extract_name(header)
        if not name:
            return None, None
        return generate_call_id(), name

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Builds a growing JSON object from the complete parameters so far.

        The closing brace is withheld until the invoke ends so successive
        diffs concatenate into valid JSON.
        """
        params = _parse_parameters(args_text)
        if not params:
            return "{}" if is_complete else ""
        inner = json.dumps(params)[:-1]
        return inner + ("}" if is_complete else "")

    XGRAMMAR_FORMAT: ClassVar[str] = "muse_glimmer"

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generates a constrained-decoding grammar for Muse Glimmer tool calls.

        Returns a serialized xgrammar ``StructuralTag`` that frames the
        ``<atem:function_calls>`` / ``<atem:invoke name="...">`` envelope.
        When ``response_format_schema`` is provided, the grammar also accepts
        a schema-conforming JSON response as an alternative to a tool call.

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a JSON response matching the schema.
            tools: OpenAI-style tool dicts.
            backend: Structured-output backend; must be ``"xgrammar"``.
            tool_choice: ``"auto"``, ``"required"``, or a named choice.
            **kwargs: Ignored; accepts ``tokenizer`` and other future kwargs.

        Returns:
            The StructuralTag serialized as a JSON string.

        Raises:
            InputError: If ``backend`` is not ``"xgrammar"``.
        """
        if backend != "xgrammar":
            raise InputError(
                "Muse Glimmer constrained tool calling requires the xgrammar "
                "backend; run with --structured-output-backend=xgrammar."
            )
        return build_xgrammar_tool_grammar(
            MuseGlimmerToolParser.XGRAMMAR_FORMAT,
            tools or [],
            tool_choice if tool_choice is not None else "auto",
            response_format_schema=response_format_schema,
        )
