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

"""Tool call parser for GLM-4.5+ models (GLM-5.1 / GLM-5.2).

GLM emits tool calls as a flat sequence of ``<tool_call>`` blocks (no outer
section wrapper). Each block opens with the bare function name, followed by
alternating ``<arg_key>`` / ``<arg_value>`` pairs::

    <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>units</arg_key><arg_value>celsius</arg_value></tool_call>

All of ``<tool_call>``, ``</tool_call>``, ``<arg_key>``, ``</arg_key>``,
``<arg_value>``, ``</arg_value>`` are single special tokens in the vocab.

Value encoding mirrors the chat template
(``{{ v | tojson(ensure_ascii=False) if v is not string else v }}``): string
arguments are emitted *bare* (no surrounding quotes), while numbers, booleans,
arrays, and objects are emitted as JSON. Decoding therefore tries ``json.loads``
first and falls back to the raw string, exactly as the MiniMax M2 / Qwen 3.5
parsers do.

Constraint level for tool calls (see ``generate_tool_call_grammar``): the
generated grammar constrains the ``<tool_call>`` envelope and the
``<arg_key>``/``<arg_value>`` framing, restricts the function name to the
provided tools, and — when a tool supplies a ``parameters`` schema — constrains
the arguments to that schema (declared keys, ``required`` properties, and each
value's type). GLM emits string values *bare* and all other types as JSON, so
non-string values are constrained via ``%json`` over the sub-schema while
strings are handled with bare ``enum`` / ``pattern`` rules. ``response_format``
schemas are enforced via the dedicated ``json_response`` branch.

Reference: ``architectures/gemma4/tool_parser.py`` (flat-mode base + grammar),
``architectures/minimax_m2/tool_parser.py`` (structural grammar tier).
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    canonicalize_lark_rule_name,
    escape_for_lark_string,
    generate_call_id,
    get_token_id,
    maybe_name_from_tool,
    names_from_tools,
    register,
    resolve_lark_token_reference,
)
from max.pipelines.modeling.types import ParsedToolCall, PipelineTokenizer

# Special-token surface forms (all single tokens in the GLM vocab).
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
ARG_KEY_OPEN = "<arg_key>"
ARG_KEY_CLOSE = "</arg_key>"
ARG_VALUE_OPEN = "<arg_value>"
ARG_VALUE_CLOSE = "</arg_value>"

# Tokens the grammar references, in a stable order.
_GRAMMAR_TOKENS = (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    ARG_KEY_OPEN,
    ARG_KEY_CLOSE,
    ARG_VALUE_OPEN,
    ARG_VALUE_CLOSE,
)

# Turn-ender tokens the model emits to *end* a tool-calling turn — GLM's
# ``generation_config.eos_token_id`` set. ``<|observation|>`` is the token the
# chat template places immediately after ``</tool_call>`` (it opens the tool
# result), so it is what the model is trained to emit to hand off after a tool
# call; ``<|user|>`` / ``<|endoftext|>`` close a normal turn. The grammar must
# admit these after the calls — otherwise the only legal continuations are
# another ``<tool_call>`` or the bare tokenizer EOS, which denies the model its
# trained turn-ender and forces it to repeat tool calls until ``max_tokens``.
_TURN_END_TOKENS = ("<|observation|>", "<|user|>", "<|endoftext|>")

# Complete-parse regexes. Non-greedy so a value never swallows the next marker
# (markers are special tokens, so they never appear inside a value anyway).
_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
_ARG_PAIR_RE = re.compile(
    re.escape(ARG_KEY_OPEN)
    + r"(.*?)"
    + re.escape(ARG_KEY_CLOSE)
    + r"\s*"
    + re.escape(ARG_VALUE_OPEN)
    + r"(.*?)"
    + re.escape(ARG_VALUE_CLOSE),
    re.DOTALL,
)


def _decode_value(raw: str) -> object:
    """Decode a GLM ``<arg_value>`` payload.

    Strings arrive bare; numbers, booleans, arrays, and objects arrive as JSON.
    Try ``json.loads`` first (covers every non-string case plus quoted strings)
    and fall back to the raw text for bare strings. Empty payloads decode to the
    empty string.
    """
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return raw


def _parse_args(args_body: str) -> dict[str, object]:
    """Parse the ``<arg_key>``/``<arg_value>`` pairs in a call body to a dict."""
    args: dict[str, object] = {}
    for match in _ARG_PAIR_RE.finditer(args_body):
        key = match.group(1).strip()
        if not key:
            continue
        args[key] = _decode_value(match.group(2))
    return args


# ---------------------------------------------------------------------------
# Schema-aware grammar generation (constrains tool-call arguments to the
# declared JSON schema, not just the structure).
#
# GLM emits ``<arg_value>`` payloads with a type-dependent encoding (see
# ``_decode_value``): *strings are bare* (no quotes), every other type is
# JSON (``v | tojson``). So a non-string value is constrainable with
# llguidance's built-in ``%json`` over the property's sub-schema (which also
# covers nested objects/arrays, ``required``, numeric bounds, etc.), while
# string values need bare handling (free text, an ``enum`` alternation, or a
# ``pattern`` regex).
# ---------------------------------------------------------------------------

_JSON_VALUE_TYPES = frozenset(
    {"integer", "number", "boolean", "null", "object", "array"}
)


def _resolve_refs(
    node: Any, defs: dict[str, Any], _depth: int = 0, _max_depth: int = 10
) -> Any:
    """Inline ``$ref`` pointers using ``$defs``; cap recursion for cycles."""
    if isinstance(node, list):
        return [_resolve_refs(i, defs, _depth, _max_depth) for i in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if (
        isinstance(ref, str)
        and ref.startswith("#/$defs/")
        and _depth < _max_depth
    ):
        name = ref[len("#/$defs/") :]
        if name in defs:
            return _resolve_refs(defs[name], defs, _depth + 1, _max_depth)
    return {
        k: _resolve_refs(v, defs, _depth, _max_depth)
        for k, v in node.items()
        if k != "$defs"
    }


def _extract_tool_schemas(
    tools: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map function name -> ref-resolved ``parameters`` schema."""
    schemas: dict[str, dict[str, Any]] = {}
    for t in tools:
        name = maybe_name_from_tool(t)
        if not name:
            continue
        params = t.get("function", {}).get("parameters")
        if isinstance(params, dict) and params.get("properties"):
            defs = params.get("$defs", {})
            schemas[name] = _resolve_refs(params, defs) if defs else params
    return schemas


def _glm_value_symbol(
    prop_schema: dict[str, Any],
    prefix: str,
    rules_parts: list[str],
    terminals: list[str],
) -> str:
    """Return the Lark symbol that constrains one ``<arg_value>`` body.

    Strings are bare (``ARG_VALUE`` free text, an enum alternation, or a
    pattern regex); every other concrete scalar/container type delegates to
    ``%json``. Unknown/union types fall back to the permissive ``ARG_VALUE``.
    """
    enum = prop_schema.get("enum")
    if isinstance(enum, list) and enum:
        # Each value is matched in its on-the-wire form: strings bare, others
        # JSON-encoded (mirrors ``_decode_value`` / the chat template).
        alts = [
            '"'
            + escape_for_lark_string(
                v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            )
            + '"'
            for v in enum
        ]
        rule = f"{prefix}_enum"
        rules_parts.append(f"{rule}: " + " | ".join(alts))
        return rule

    jtype = prop_schema.get("type")
    if jtype == "string":
        pattern = prop_schema.get("pattern")
        if isinstance(pattern, str) and pattern:
            term = prefix.upper() + "_PAT"
            terminals.append(f"{term}: /{pattern}/")
            return term
        return (
            "ARG_VALUE"  # bare free string, bounded by the </arg_value> token
        )

    if isinstance(jtype, str) and jtype in _JSON_VALUE_TYPES:
        rule = f"{prefix}_json"
        rules_parts.append(
            f"{rule}: %json "
            + json.dumps(prop_schema, separators=(",", ":"), ensure_ascii=False)
        )
        return rule

    # No/union type: don't risk rejecting a valid bare string.
    return "ARG_VALUE"


def _glm_ordered_args(
    prefix: str,
    arg_rules: list[str],
    required: list[bool],
    rules_parts: list[str],
) -> str:
    """Build suffix rules enforcing schema order; required args can't be
    skipped, optional ones may be (no separator between GLM arg pairs)."""
    n = len(arg_rules)
    if n == 0:
        return ""
    has_req_after = [False] * n
    for i in range(n - 2, -1, -1):
        has_req_after[i] = required[i + 1] or has_req_after[i + 1]
    for i in range(n - 1, -1, -1):
        sfx = f"{prefix}_sfx_{i}"
        prop = arg_rules[i]
        if i == n - 1:
            rules_parts.append(f"{sfx}: {prop}")
        else:
            nxt = f"{prefix}_sfx_{i + 1}"
            if has_req_after[i]:
                if required[i]:
                    rules_parts.append(f"{sfx}: {prop} {nxt}")
                else:
                    rules_parts.append(f"{sfx}: {prop} {nxt} | {nxt}")
            else:
                if required[i]:
                    rules_parts.append(f"{sfx}: {prop} ({nxt})?")
                else:
                    rules_parts.append(f"{sfx}: {prop} ({nxt})? | {nxt}")
    top = f"{prefix}_sfx_0"
    return top if any(required) else f"{top}?"


def _glm_args_rule(
    name: str,
    schema: dict[str, Any],
    aks: str,
    ake: str,
    avs: str,
    ave: str,
    rules_parts: list[str],
    terminals: list[str],
) -> str:
    """Build and return the args-body rule name for one tool's schema."""
    canon = canonicalize_lark_rule_name(name)
    props: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))

    arg_rules: list[str] = []
    req_flags: list[bool] = []
    for pname, pschema in props.items():
        pcanon = canonicalize_lark_rule_name(pname)
        prefix = f"a_{canon}_{pcanon}"
        val = _glm_value_symbol(
            pschema if isinstance(pschema, dict) else {},
            prefix,
            rules_parts,
            terminals,
        )
        arg_rule = f"argp_{canon}_{pcanon}"
        rules_parts.append(
            f'{arg_rule}: {aks} "{escape_for_lark_string(pname)}" {ake} '
            f"{avs} {val} {ave}"
        )
        arg_rules.append(arg_rule)
        req_flags.append(pname in required)

    ordered_top = _glm_ordered_args(
        f"o_{canon}", arg_rules, req_flags, rules_parts
    )

    parts: list[str] = []
    if ordered_top:
        parts.append(ordered_top)
    # additionalProperties (default True) allows extra, unconstrained args.
    if schema.get("additionalProperties", True) is not False:
        parts.append("arg*")

    body = f"args_{canon}"
    rules_parts.append(f"{body}: " + (" ".join(parts) if parts else "arg*"))
    return body


@register("glm45")
class GlmToolParser(StructuralTagToolParser):
    """Parses GLM-4.5+ (GLM-5.1 / GLM-5.2) tool calls.

    Flat layout: only ``CALL_BEGIN``/``CALL_END`` are set, so the base class
    scans for ``<tool_call>`` … ``</tool_call>`` pairs directly. Within each
    call the function name precedes the first ``<arg_key>``; the remainder is
    parameter XML that we convert to growing JSON for streaming.
    """

    CALL_BEGIN: ClassVar[str] = TOOL_CALL_OPEN
    CALL_END: ClassVar[str] = TOOL_CALL_CLOSE

    # ----- Complete parsing --------------------------------------------

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for block in _TOOL_CALL_BLOCK_RE.finditer(tool_section):
            body = block.group(1)
            name, args_body = self._split_tool_call_body(body, is_complete=True)
            if not name:
                continue
            args = _parse_args(args_body or "")
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                )
            )
        return tool_calls

    # ----- Streaming hooks ---------------------------------------------

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits a ``<tool_call>`` body into (function-name, parameter-XML).

        The name is everything before the first ``<arg_key>``. Until either an
        ``<arg_key>`` marker or the closing ``</tool_call>`` has arrived the name
        boundary is unknown, so return ``(None, None)`` to defer.
        """
        key_idx = body.find(ARG_KEY_OPEN)
        if key_idx >= 0:
            name = body[:key_idx].strip()
            return (name or None), body[key_idx:]
        if is_complete:
            name = body.strip()
            return (name or None), ""
        return None, None

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Builds a growing JSON string from complete ``<arg_key>`` pairs.

        Omits the closing brace while the call is still streaming so that
        successive argument diffs concatenate into valid JSON once the final
        pair lands.
        """
        args = _parse_args(args_text)
        if not args:
            return "{}" if is_complete else ""
        inner = ", ".join(
            f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
            for k, v in args.items()
        )
        return "{" + inner + ("}" if is_complete else "")

    # ----- Constrained-decoding grammar --------------------------------

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: PipelineTokenizer[Any, Any, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generates a Lark grammar for GLM tool-call constrained decoding.

        Special tokens are referenced by ID (``<[N]>``) so multi-byte literal
        matches don't trip llguidance. The envelope and ``<arg_key>``/
        ``<arg_value>`` framing are always constrained, and the calls must end
        on a turn-ender token (``<|observation|>``/``<|user|>``/
        ``<|endoftext|>``) so the grammar closes instead of looping.

        When a tool supplies a ``parameters`` schema, arguments are constrained
        to it: ``<arg_key>`` is restricted to the declared property names,
        ``required`` properties must appear, and each ``<arg_value>`` is
        constrained to its property type — bare for strings (with ``enum`` /
        ``pattern`` support), and via ``%json`` over the sub-schema for every
        other type (numbers, booleans, nested objects/arrays, etc.). Tools with
        no properties schema fall back to permissive (valid-structure) args.
        When ``response_format_schema`` is provided an alternative JSON branch
        matching the schema is added.

        Not enforced for string values: ``maxLength`` / ``format`` (GLM strings
        are bare, so JSON-schema string facets beyond ``pattern`` aren't
        applied); numeric/object facets rely on ``%json`` coverage.

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a JSON response matching the schema.
            tools: Optional list of OpenAI-style tool dicts. ``None`` accepts
                any tool name.
            tokenizer: Tokenizer used to resolve GLM special-token IDs.
            **kwargs: Ignored (accepts ``backend``, ``tool_choice``, etc.).

        Returns:
            A Lark grammar string for the constrained-decoding backend.
        """
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required for GLM tool-call grammar generation"
            )

        token_ids: dict[str, int] = {}
        for token in _GRAMMAR_TOKENS:
            tid = get_token_id(tokenizer, token)
            if tid is None:
                raise ValueError(
                    f"GLM grammar generation could not resolve token {token!r}"
                )
            token_ids[token] = tid

        tcs = resolve_lark_token_reference(token_ids[TOOL_CALL_OPEN])
        tce = resolve_lark_token_reference(token_ids[TOOL_CALL_CLOSE])
        aks = resolve_lark_token_reference(token_ids[ARG_KEY_OPEN])
        ake = resolve_lark_token_reference(token_ids[ARG_KEY_CLOSE])
        avs = resolve_lark_token_reference(token_ids[ARG_VALUE_OPEN])
        ave = resolve_lark_token_reference(token_ids[ARG_VALUE_CLOSE])

        # Resolve the turn-ender tokens that may follow the calls. Include only
        # the ones present in this tokenizer's vocab.
        end_refs = [
            resolve_lark_token_reference(tid)
            for tok in _TURN_END_TOKENS
            if (tid := get_token_id(tokenizer, tok)) is not None
        ]

        tool_names = names_from_tools(tools)
        tool_schemas = _extract_tool_schemas(tools) if tools else {}

        schema_rules: list[str] = []
        terminals = [
            r"ARG_KEY: /[\s\S]+/",
            r"ARG_VALUE: /[\s\S]*/",
        ]
        func_rule = ""

        if tool_schemas:
            # Schema-aware: each tool gets its own alternative constraining keys
            # to the declared properties and each value to its property type.
            # Tools without a properties schema keep the permissive ``arg*``.
            alts: list[str] = []
            for name in dict.fromkeys(tool_names or []):
                esc = escape_for_lark_string(name)
                schema = tool_schemas.get(name)
                if schema is None:
                    alts.append(f'{tcs} "{esc}" arg* {tce}')
                    continue
                body = _glm_args_rule(
                    name, schema, aks, ake, avs, ave, schema_rules, terminals
                )
                alts.append(f'{tcs} "{esc}" {body} {tce}')
            tool_call_rule = "tool_call: " + " | ".join(f"({a})" for a in alts)
        elif tool_names:
            name_alts = " | ".join(
                f'"{escape_for_lark_string(n)}"' for n in tool_names
            )
            func_rule = f"func_name: {name_alts}"
            tool_call_rule = f"tool_call: {tcs} func_name arg* {tce}"
        else:
            func_rule = "func_name: FUNC_NAME"
            terminals.append(r"FUNC_NAME: /[^<\s][^<]*/")
            tool_call_rule = f"tool_call: {tcs} func_name arg* {tce}"

        # Require a turn-ender after the calls so the grammar closes
        # deterministically (mirrors gemma4's ``tool_call+ (TURN_END | ...)``).
        # Fall back to the bare ``tool_call+`` only if no turn-ender resolved.
        if end_refs:
            tool_calls_rule = "tool_calls: tool_call+ tool_calls_end"
            tool_calls_end_rule = "tool_calls_end: " + " | ".join(end_refs)
        else:
            tool_calls_rule = "tool_calls: tool_call+"
            tool_calls_end_rule = ""

        rules = [
            tool_calls_rule,
            tool_calls_end_rule,
            tool_call_rule,
            f"arg: {aks} ARG_KEY {ake} {avs} ARG_VALUE {ave}",
            func_rule,
            *schema_rules,
        ]
        rule_lines = "\n".join(line for line in rules if line)
        terminal_lines = "\n".join(line for line in terminals if line)
        tool_grammar = f"\n{rule_lines}\n\n{terminal_lines}\n"

        if response_format_schema is None:
            return f"\nstart: tool_calls\n{tool_grammar}"

        schema_with_opts = {
            **response_format_schema,
            "x-guidance": {"whitespace_pattern": ""},
        }
        schema_json = json.dumps(schema_with_opts, separators=(",", ":"))
        return (
            f"\nstart: tool_calls | json_response\n"
            f"json_response: %json {schema_json}\n{tool_grammar}"
        )
