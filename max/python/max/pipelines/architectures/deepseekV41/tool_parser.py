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

"""DeepSeek-V4.1 tool-call parser for the checkpoint's DSML format.

V4.1 keeps V4's DSML layout but renames the tags, each now with a leading
space::

    \\n\\n<｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="location" string="true">Beijing</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

so the V4 parser is reused with V4.1's markers and reference decoder
(``encoding_dsv41.py``).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar

from max.pipelines.architectures.deepseekV4.tool_parser import (
    DeepseekV4ToolParser,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    build_xgrammar_tool_grammar,
)
from max.pipelines.lib.tool_parsing import register

from . import encoding_dsv41
from .encoding_dsv41 import (
    dsml_token,
    tool_call_tag_name,
    tool_calls_block_name,
    tool_parameter_tag_name,
)


@register("deepseekv41")
class DeepseekV41ToolParser(DeepseekV4ToolParser):
    """Parses DeepSeek-V4.1 ``<｜DSML｜ calls>`` blocks into tool calls."""

    SECTION_BEGIN = f"<{dsml_token}{tool_calls_block_name}>\n"
    SECTION_END = f"</{dsml_token}{tool_calls_block_name}>"
    CALL_BEGIN = f"<{dsml_token}{tool_call_tag_name}"
    CALL_END = f"</{dsml_token}{tool_call_tag_name}>"
    PARAMETER_END: ClassVar[str] = f"</{dsml_token}{tool_parameter_tag_name}>\n"
    ENCODING: ClassVar[ModuleType] = encoding_dsv41
    # The reader takes nothing after the calls block.
    ENFORCE_TOOL_REGION_TO_EOS = True

    XGRAMMAR_FORMAT: ClassVar[str] = "deepseek_v41"

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        reject_unsupported: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generates a tool-call constrained-decoding grammar for DeepSeek-V4.1.

        Returns a serialized xgrammar ``StructuralTag`` that frames the
        ``<｜DSML｜ calls>`` / ``<｜DSML｜ invoke name="...">`` envelope and
        constrains each call's arguments to that tool's JSON schema, emitted
        as ``<｜DSML｜ parameter name=... string="true|false">`` values. When
        ``response_format_schema`` is provided, the grammar also accepts a
        schema-conforming JSON response as an alternative to a tool call.

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a JSON response matching the schema.
            tools: OpenAI-style tool dicts.
            backend: Structured-output backend; must be ``"xgrammar"``.
            tool_choice: ``"auto"``, ``"required"``, or a named choice.
            reject_unsupported: Whether to raise an error during grammar
                compilation when a tool's schema cannot be enforced. Otherwise,
                the schema is enforced best-effort.
            **kwargs: Ignored; accepts ``tokenizer`` and other future kwargs.

        Returns:
            The StructuralTag serialized as a JSON string.
        """
        if backend != "xgrammar":
            raise InputError(
                "DeepSeek-V4.1 constrained tool calling requires the xgrammar "
                "backend; run with --structured-output-backend=xgrammar."
            )
        normalized_choice = tool_choice if tool_choice is not None else "auto"
        return build_xgrammar_tool_grammar(
            DeepseekV41ToolParser.XGRAMMAR_FORMAT,
            tools or [],
            normalized_choice,
            response_format_schema=response_format_schema,
            reject_unsupported=reject_unsupported,
        )
