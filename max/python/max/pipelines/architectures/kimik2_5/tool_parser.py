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

"""Tool call parser for Kimi K2.5 models.

Kimi K2.5 uses a structural tag format for tool calls:

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>
    {"key": "value"}
    <|tool_call_end|>
    <|tool_calls_section_end|>

Reference: https://vllm.ai/blog/Kimi-K2-Accuracy
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from llguidance import LLMatcher
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    build_xgrammar_tool_grammar,
)
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    escape_for_lark_string,
    get_token_id,
    names_from_tools,
    register,
    resolve_lark_token_reference,
)
from max.pipelines.modeling.types import ParsedToolCall, PipelineTokenizer

# Structural tags used by Kimi K2.5
TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

# Reasoning and turn-terminator tokens. Kimi K2.5 interleaves
# ``<think>...</think>`` reasoning blocks with tool-call sections and ends
# the assistant turn with ``<|im_end|>``. These are referenced in the
# constrained-decoding grammar so the model may interleave reasoning
# between sections and stop early (see ``generate_tool_call_grammar``).
THINK_START = "<think>"
THINK_END = "</think>"
IM_END = "<|im_end|>"

# Bounds on the constrained-decoding grammar quantifiers. Without these,
# a model can spin emitting digits in the call index or an unbounded
# number of back-to-back calls/sections, holding a GPU slot until
# ``max_tokens``. The argument body is intentionally unbounded — tool
# arguments can be arbitrarily large (e.g. code blobs, embedded documents,
# search-result payloads being re-emitted) and a fixed cap would silently
# drop them. The ``max_tokens`` ceiling is the only meaningful upper bound
# there.
_MAX_TOOL_CALL_INDEX_DIGITS = 8  # up to 99_999_999 tool calls per turn
_MAX_TOOL_CALLS_PER_SECTION = 64
# Kimi interleaves multiple tool-call sections with reasoning in a single
# turn ("interleaved thinking"). The grammar admits up to this many
# sections; the model stops earlier by emitting ``<|im_end|>`` (allowed at
# every accepting state). A bounded cap keeps a stuck model from holding a
# slot forever (``max_tokens`` is the primary ceiling; this is a secondary
# backstop). Set with headroom for long interleaved turns so a legitimate
# extra section never trips the matcher-desync this grammar fixes — the
# bound is a counter in the compiled grammar, so raising it has no
# compile/per-token cost.
_MAX_TOOL_CALL_SECTIONS = 8

# Regex for one ``<|tool_call_begin|>...<|tool_call_end|>`` body. The
# function id and arguments are captured; the call markers are anchored.
_TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_BEGIN)}"
    rf"(?P<function_id>[^\n<]+)"
    rf"{re.escape(TOOL_CALL_ARGUMENT_BEGIN)}"
    rf"(?P<arguments>.*?)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)


def _parse_function_id(function_id: str) -> tuple[str, str]:
    """Parses a Kimi function ID into ``(name, call_id)``.

    Kimi function IDs have the format ``functions.{name}:{idx}``. Some
    IDs may lack the ``functions.`` prefix (for example, ``search:2``)
    or the index suffix. The call id always begins with ``call_`` and
    includes the index when one is present, matching the OpenAI-style
    tool id with a stable suffix per call.
    """
    function_id = function_id.strip()

    # Standard form: functions.{name}:{idx}
    if "." in function_id:
        try:
            _, rest = function_id.split(".", 1)
            if ":" in rest:
                name, _ = rest.rsplit(":", 1)
            else:
                name = rest
            short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
            return name, f"{name}:{short_uuid}"
        except (ValueError, IndexError):
            pass

    # Fallback for non-prefixed ids like "search:2"
    if ":" in function_id:
        name, _ = function_id.rsplit(":", 1)
        short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
        return name, f"{name}:{short_uuid}"

    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
    return function_id, f"{function_id}:{short_uuid}"


@register("kimik2_5")
class KimiToolParser(StructuralTagToolParser):
    """Parses Kimi K2.5-style tool calls from model responses.

    Kimi K2.5 wraps tool calls in section/call markers and embeds the
    function name as a compound ``functions.{name}:{idx}`` identifier
    before a dedicated argument-begin marker. Arguments are raw JSON,
    which the base class can diff directly.
    """

    SECTION_BEGIN = TOOL_CALLS_SECTION_BEGIN
    SECTION_END = TOOL_CALLS_SECTION_END
    CALL_BEGIN = TOOL_CALL_BEGIN
    CALL_END = TOOL_CALL_END

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for match in _TOOL_CALL_PATTERN.finditer(tool_section):
            function_id = match.group("function_id")
            arguments_str = match.group("arguments").strip()

            name, call_id = _parse_function_id(function_id)
            if not name:
                continue

            try:
                args_obj = json.loads(arguments_str)
                arguments_json = json.dumps(args_obj)
            except json.JSONDecodeError:
                # Pass through to surface upstream rather than dropping.
                arguments_json = arguments_str

            tool_calls.append(
                ParsedToolCall(id=call_id, name=name, arguments=arguments_json)
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits ``functions.foo:0<|tool_call_argument_begin|>{...}``."""
        arg_pos = body.find(TOOL_CALL_ARGUMENT_BEGIN)
        if arg_pos == -1:
            return None, None
        header = body[:arg_pos].strip()
        args = body[arg_pos + len(TOOL_CALL_ARGUMENT_BEGIN) :]
        return header, args

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        """Parses Kimi's ``functions.{name}:{idx}`` header.

        Delegates to :func:`_parse_function_id`, which handles all known
        Kimi header formats and always returns a valid (name, id) pair
        for non-empty input. Returns ``(None, None)`` only when the
        header is empty.
        """
        if not header:
            return None, None
        tool_name, tool_id = _parse_function_id(header)
        return tool_id, tool_name

    # ----- Constrained decoding grammar (Kimi-specific) -----------------

    @staticmethod
    def _build_envelope(
        tool_names: list[str] | None,
        refs: dict[str, str],
    ) -> tuple[str, list[str]]:
        """Builds the tool-call envelope rule and shared grammar lines.

        Returns the ``start``-body fragment (the repeated section/think
        sequence with an optional trailing ``<|im_end|>``) and the list of
        shared rule/terminal lines it references. Both the no-schema and
        json_schema branches reuse the same envelope.

        ``refs`` maps each marker to its single-token Lark reference
        (``<[id]>``). ``THINK_START``/``THINK_END`` and ``IM_END`` are
        optional: when absent the grammar simply omits interleaved
        reasoning / early termination rather than failing.
        """
        # ``functions.NAME:INDEX`` header. ``NAME`` is an alternation of the
        # offered tool names (or a length-capped fallback identifier).
        if tool_names is not None:
            name_terminal = "NAME: " + " | ".join(
                f'"{escape_for_lark_string(n)}"' for n in tool_names
            )
        else:
            name_terminal = r"NAME: /[a-zA-Z0-9_-]{1,128}/"

        # ``think?`` is only available when both delimiters resolve.
        has_think = "THINK_START" in refs and "THINK_END" in refs
        think_opt = "think? " if has_think else ""
        # ``<|im_end|>`` lets the model stop before the section cap; it is
        # an EOS-class token (handled by ``eos_tracker``) and is allowed at
        # every accepting state via this optional trailing reference.
        im_end_opt = f" {refs['IM_END']}?" if "IM_END" in refs else ""

        # 1..N sections, each optionally preceded by a reasoning block.
        envelope = (
            f"{think_opt}section "
            f"({think_opt}section){{0,{_MAX_TOOL_CALL_SECTIONS - 1}}}"
            f"{im_end_opt}"
        )

        rules = [
            (
                f"section: {refs['SECTION_BEGIN']} tool_call "
                f"(tool_call){{0,{_MAX_TOOL_CALLS_PER_SECTION - 1}}} "
                f"{refs['SECTION_END']}"
            ),
            (
                f'tool_call: {refs["CALL_BEGIN"]} "functions." NAME ":" '
                f"INDEX {refs['ARG_BEGIN']} ARGS {refs['CALL_END']}"
            ),
            name_terminal,
            rf"INDEX: /[0-9]{{1,{_MAX_TOOL_CALL_INDEX_DIGITS}}}/",
            # The argument body and reasoning body are byte-level ``/.*/``
            # terminals; each terminates naturally at its atomic closing
            # special token (``<|tool_call_end|>`` / ``</think>``), so they
            # accept ``<`` and other markup freely. Real argument validation
            # happens at parse time — the grammar only frames structure.
            r"ARGS: /[\s\S]*/",
        ]
        if has_think:
            rules.append(
                f"think: {refs['THINK_START']} THINK_BODY {refs['THINK_END']}"
            )
            rules.append(r"THINK_BODY: /[\s\S]*/")

        return envelope, rules

    @staticmethod
    def _resolve_token_refs(
        tokenizer: PipelineTokenizer[Any, Any, Any] | None,
    ) -> dict[str, str]:
        """Resolves Kimi structural tokens to single-token Lark references.

        Returns a ``name -> "<[id]>"`` map. The five tool-call markers are
        required (a missing one raises). ``<think>``/``</think>``/
        ``<|im_end|>`` are optional and simply absent from the map when the
        tokenizer does not define them.
        """
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required to generate the Kimi tool-call grammar"
            )

        required = {
            "SECTION_BEGIN": TOOL_CALLS_SECTION_BEGIN,
            "SECTION_END": TOOL_CALLS_SECTION_END,
            "CALL_BEGIN": TOOL_CALL_BEGIN,
            "CALL_END": TOOL_CALL_END,
            "ARG_BEGIN": TOOL_CALL_ARGUMENT_BEGIN,
        }
        optional = {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
            "IM_END": IM_END,
        }

        refs: dict[str, str] = {}
        for name, token in required.items():
            tid = get_token_id(tokenizer, token)
            if tid is None:
                raise ValueError(
                    f"tokenizer does not define required Kimi tool-call "
                    f"token {token!r}; cannot build constrained grammar"
                )
            refs[name] = resolve_lark_token_reference(tid)
        for name, token in optional.items():
            tid = get_token_id(tokenizer, token)
            if tid is not None:
                refs[name] = resolve_lark_token_reference(tid)
        return refs

    XGRAMMAR_FORMAT = "kimi"

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: PipelineTokenizer[Any, Any, Any] | None = None,
        backend: str = "llguidance",
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generates a grammar for constrained decoding of Kimi tool calls.

        With ``backend="xgrammar"`` this returns a serialized xgrammar
        StructuralTag (which constrains each call's arguments to that tool's
        JSON schema). With the default ``backend="llguidance"`` it returns a
        Lark grammar whose argument body is freeform.

        Kimi K2.5 performs "interleaved thinking": a single assistant turn
        can interleave multiple ``<think>...</think>`` reasoning blocks with
        multiple ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>``
        tool-call sections, and ends the turn with ``<|im_end|>``. The
        grammar admits up to ``_MAX_TOOL_CALL_SECTIONS`` sections, an
        optional reasoning block before each, and an optional trailing
        ``<|im_end|>`` so the model can stop before the cap.

        Structural markers, ``<think>``/``</think>``, and ``<|im_end|>`` are
        referenced as single-token symbols (``<[id]>``) resolved from
        ``tokenizer`` — they are atomic special tokens, so the freeform
        ``/[\\s\\S]*/`` argument and reasoning bodies terminate cleanly at
        the closing marker. Reasoning enforced this way is plain text; a
        mid-reasoning special token is not admitted under forced decoding.

        When ``response_format_schema`` is provided, the grammar also accepts
        a JSON response matching the schema (the model's first tokens select
        the branch).

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a JSON response matching the schema.
            tools: Optional list of OpenAI-style tool dicts. ``None`` accepts
                any length-capped identifier as the function name.
            tokenizer: Pipeline tokenizer used to resolve special-token IDs.
                Required.
            **kwargs: Ignored; accepts future kwargs.

        Returns:
            A grammar string compatible with the selected backend.
        """
        if backend == "xgrammar":
            if response_format_schema is not None:
                raise InputError(
                    "The xgrammar backend does not yet support combined "
                    "tool-calling and response_format json_schema. Use "
                    "--structured-output-backend=llguidance for that case."
                )
            normalized_choice = (
                tool_choice if tool_choice is not None else "auto"
            )
            # Forced tool calling (required / named) disables the reasoning
            # prefix so the envelope is forced from the first token; ``auto``
            # keeps reasoning optional.
            forced = normalized_choice == "required" or isinstance(
                normalized_choice, dict
            )
            return build_xgrammar_tool_grammar(
                KimiToolParser.XGRAMMAR_FORMAT,
                tools or [],
                normalized_choice,
                reasoning=not forced,
            )

        tool_names = names_from_tools(tools)
        refs = KimiToolParser._resolve_token_refs(tokenizer)
        envelope, shared_rules = KimiToolParser._build_envelope(
            tool_names, refs
        )

        if response_format_schema is None:
            start_rule = f"start: {envelope}"
            extra_rules: list[str] = []
        else:
            schema_str = json.dumps(response_format_schema)
            start_rule = "start: tool_calls | json_response"
            extra_rules = [
                f"tool_calls: {envelope}",
                f"json_response: %json {schema_str}",
            ]

        lark = (
            "\n".join(
                ["%llguidance {}", start_rule, *extra_rules, *shared_rules]
            )
            + "\n"
        )
        return LLMatcher.grammar_from_lark(lark)
