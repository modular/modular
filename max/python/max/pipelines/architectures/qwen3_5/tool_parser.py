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

"""Tool call parser for Qwen 3.5 / 3.6 models.

Qwen 3.5/3.6 emits tool calls as XML-ish blocks::

    <tool_call>
    <function=function_name>
    <parameter=arg_name>
    arg_value
    </parameter>
    <parameter=arg2_name>
    {"key": "value"}
    </parameter>
    </function>
    </tool_call>

Strings are emitted bare; numbers, booleans, arrays, and objects are emitted
as JSON. Multiple ``<tool_call>`` blocks may appear back-to-back. Reasoning
text may precede the first tool call.

Streaming emits per-parameter argument fragments (``{"k":v``, ``,"k":v``,
``}``) — finer than per-call, coarser than per-token, and good enough for
OpenAI-style clients that just concatenate the fragments back into a JSON
string.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar

from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    build_xgrammar_tool_grammar,
)
from max.pipelines.lib.tool_parsing import register
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN_PREFIX = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN_PREFIX = "<parameter="
PARAMETER_CLOSE = "</parameter>"

_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
_FUNCTION_NAME_RE = re.compile(
    re.escape(FUNCTION_OPEN_PREFIX) + r"([^>]+)>",
)
_PARAMETER_RE = re.compile(
    re.escape(PARAMETER_OPEN_PREFIX)
    + r"([^>]+)>\n?(.*?)\n?"
    + re.escape(PARAMETER_CLOSE),
    re.DOTALL,
)


_MAX_REF_DEPTH = 8
_MAX_SCHEMA_NODES = 4096
_MAX_SCHEMA_DEPTH = 32


def _resolve_ref(
    schema: Mapping[str, Any], defs_root: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Follows local ``#/...`` ``$ref`` pointers (bounded) to their target."""
    for _ in range(_MAX_REF_DEPTH):
        ref = schema.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return schema
        target: Any = defs_root
        for part in ref[2:].split("/"):
            if not isinstance(target, Mapping):
                return schema
            target = target.get(part.replace("~1", "/").replace("~0", "~"))
        if not isinstance(target, Mapping):
            return schema
        schema = target
    return schema


def _json_type(value: object) -> str:
    """The JSON Schema type name of an already-decoded value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object" if isinstance(value, dict) else "array"


def _schema_types(
    schema: Mapping[str, Any] | None,
    defs_root: Mapping[str, Any],
    budget: list[int] | None = None,
    depth: int = 0,
) -> set[str] | None:
    """Returns the JSON types a parameter schema allows, or ``None`` if unknown.

    ``anyOf``/``oneOf`` and type arrays are unions; ``allOf`` intersects. An
    unconstrained schema yields ``None`` (schema-blind decoding), as does one
    over the node or depth limit: the schema is client-supplied, so the walk is
    capped against self-referencing ``$ref``s.
    """
    if budget is None:
        budget = [_MAX_SCHEMA_NODES]
    budget[0] -= 1
    if depth > _MAX_SCHEMA_DEPTH:
        budget[0] = -1
    if budget[0] < 0 or not isinstance(schema, Mapping):
        return None
    schema = _resolve_ref(schema, defs_root)
    types: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        types.add(declared)
    elif isinstance(declared, list):
        types.update(t for t in declared if isinstance(t, str))
    if not types:
        # ``const`` / ``enum`` pin the value, so they pin its type too.
        pinned = (
            [schema["const"]]
            if "const" in schema
            else schema["enum"]
            if isinstance(schema.get("enum"), list)
            else []
        )
        types.update(_json_type(value) for value in pinned)
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for branch in branches:
                branch_types = _schema_types(
                    branch, defs_root, budget, depth + 1
                )
                if branch_types is None:
                    return None  # an unconstrained branch allows anything
                types |= branch_types
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        constrained = [
            branch_types
            for branch in all_of
            if (
                branch_types := _schema_types(
                    branch, defs_root, budget, depth + 1
                )
            )
            is not None
        ]
        if constrained:
            narrowed = set.intersection(*constrained)
            types = types & narrowed if types else narrowed
    if budget[0] < 0:
        return None  # ran out of budget below: what we have may be partial
    return types or None


def _decode_value(
    raw: str,
    schema: Mapping[str, Any] | None = None,
    defs_root: Mapping[str, Any] | None = None,
) -> object:
    """Decode a Qwen parameter value, by the parameter's schema if known.

    The text is ambiguous: ``10001`` is a string for ``zip_code: str`` and an
    integer for ``count: int``, so a string-typed value is returned verbatim.
    Without a schema it falls back to ``json.loads``, then the Python-style
    ``True``/``False``/``None`` the chat template emits, then the raw string.
    """
    if schema:
        # ``enum`` / ``const`` pin the value, so padding the grammar allows
        # around it is not part of the value: match the member stripped.
        resolved = _resolve_ref(schema, defs_root or {})
        pinned = (
            [resolved["const"]] if "const" in resolved else resolved.get("enum")
        )
        if isinstance(pinned, list):
            for member in pinned:
                if isinstance(member, str) and member == raw.strip():
                    return member
    types = _schema_types(schema, defs_root or {}) if schema else None
    if types is not None and types <= {"string"}:
        return raw  # verbatim: whitespace is significant in a string value
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        value: object = json.loads(stripped)
    except json.JSONDecodeError:
        # The template renders bools/None with Python ``str()``, not JSON.
        value = {"True": True, "False": False, "None": None}.get(
            stripped, stripped
        )
    if types is not None and "string" in types:
        # A string among other types (``string | null``): keep a scalar only
        # if its type is allowed; strings stay verbatim.
        if isinstance(value, str):
            return raw
        decoded_type = _json_type(value)
        allowed = decoded_type in types or (
            decoded_type == "integer" and "number" in types
        )
        if not allowed:
            return raw
    return value


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


# Streaming state machine states.
_STATE_INIT = 0  # Outside any <tool_call>; emit content.
_STATE_AFTER_TOOL_CALL_OPEN = 1  # Inside <tool_call>, before <function=...>.
_STATE_IN_FUNCTION = 2  # After <function=NAME>, between parameters.
_STATE_IN_PARAMETER = 3  # After <parameter=KEY>, accumulating value.
_STATE_AFTER_FUNCTION_CLOSE = 4  # After </function>, before </tool_call>.


def _max_holdback(buffer: str, sentinel: str) -> int:
    """Return the length of the longest suffix of ``buffer`` that is a
    proper prefix of ``sentinel``. We hold back exactly that many
    characters so a sentinel split across deltas is still recognized."""
    max_keep = min(len(buffer), len(sentinel) - 1)
    for n in range(max_keep, 0, -1):
        if sentinel.startswith(buffer[-n:]):
            return n
    return 0


# xgrammar's ``qwen_3_5`` ``required`` section joins calls with a mandatory
# "\n", but the template ends every call with ``</tool_call>\n``: the model's
# trailing newline reads as "another call must follow" and it loops until
# ``max_tokens``. Fold the newline into each call's end tag instead.


def _without_mandatory_call_separator(node: Any) -> Any:
    """Rewrites ``"\n"``-separated tag sections to end each call in a newline."""
    if isinstance(node, list):
        return [_without_mandatory_call_separator(item) for item in node]
    if not isinstance(node, dict):
        return node
    if node.get("type") == "json_schema":
        return node  # a user schema: its own keys are data, not grammar
    if (
        node.get("type") == "tags_with_separator"
        and node.get("separator") == "\n"
        and isinstance(node.get("tags"), list)
    ):
        return {
            **node,
            "separator": "",
            "tags": [
                {**tag, "end": tag["end"] + "\n"}
                if isinstance(tag, dict) and isinstance(tag.get("end"), str)
                else tag
                for tag in node["tags"]
            ],
        }
    return {
        key: _without_mandatory_call_separator(value)
        for key, value in node.items()
    }


@register("qwen3_5")
class Qwen3_5ToolParser:
    """Parser for Qwen 3.5 / 3.6 tool calls."""

    # Structural start marker, exposed under the same name as
    # ``StructuralTagToolParser.CALL_BEGIN`` so the serving layer's
    # marker-leak guard covers this parser too.
    CALL_BEGIN: ClassVar[str] = TOOL_CALL_OPEN
    # xgrammar built-in for the Qwen 3.5 / Nemotron tool-call syntax.
    XGRAMMAR_FORMAT = "qwen_3_5"

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: Any | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        reject_unsupported: bool = False,
        **kwargs: Any,
    ) -> str:
        """Build a constrained-decoding grammar for Qwen 3.5 tool calls.

        Returns a serialized xgrammar ``StructuralTag`` that frames the
        ``<tool_call>`` envelope and constrains each call's arguments to that
        tool's JSON schema. When ``response_format_schema`` is provided, the
        grammar also accepts a JSON response matching the schema (the model's
        first tokens select the branch).

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a JSON response matching the schema.
            tools: Optional list of OpenAI-style tool dicts.
            tokenizer: Unused (the xgrammar tag references literal markers).
            backend: Structured-output backend; must be ``"xgrammar"``.
            tool_choice: ``"auto"``, ``"required"``, or a named choice.
            reject_unsupported: Whether to raise an error during grammar
                compilation when a tool's schema cannot be enforced. Otherwise,
                the schema is enforced best-effort.
            **kwargs: Ignored; accepts future kwargs.

        Returns:
            The StructuralTag serialized as a JSON string.
        """
        if backend != "xgrammar":
            raise InputError(
                "Qwen 3.5 constrained tool calling requires the xgrammar "
                "backend; run with --structured-output-backend=xgrammar."
            )
        normalized_choice = tool_choice if tool_choice is not None else "auto"
        grammar = build_xgrammar_tool_grammar(
            Qwen3_5ToolParser.XGRAMMAR_FORMAT,
            tools or [],
            normalized_choice,
            response_format_schema=response_format_schema,
            reject_unsupported=reject_unsupported,
        )
        return json.dumps(
            _without_mandatory_call_separator(json.loads(grammar))
        )

    def __init__(self) -> None:
        self._buffer: str = ""
        self._state: int = _STATE_INIT
        self._tool_index: int = -1
        self._param_count_in_call: int = 0
        self._current_param_key: str = ""
        self._current_function: str = ""
        self._tool_schemas: Mapping[str, dict[str, Any]] = {}

    def set_tool_schemas(self, schemas: Mapping[str, dict[str, Any]]) -> None:
        """Supplies each tool's ``parameters`` schema, keyed by function name.

        Optional hook, called by the router before parsing a request's output
        (streaming or not) so values are decoded by their declared type.
        """
        self._tool_schemas = schemas

    def _decode_param(self, function: str, key: str, raw: str) -> object:
        parameters = self._tool_schemas.get(function)
        properties = (
            parameters.get("properties") if parameters is not None else None
        )
        schema = (
            properties.get(key) if isinstance(properties, Mapping) else None
        )
        return _decode_value(raw, schema, parameters)

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parse a complete model response into tool calls."""
        if TOOL_CALL_OPEN not in response:
            return ParsedToolResponse(content=response, tool_calls=[])

        first_open = response.find(TOOL_CALL_OPEN)
        content_before = response[:first_open].strip() or None

        tool_calls: list[ParsedToolCall] = []
        for block in _TOOL_CALL_BLOCK_RE.finditer(response):
            body = block.group(1)
            name_match = _FUNCTION_NAME_RE.search(body)
            if not name_match:
                continue
            name = name_match.group(1).strip()

            args: dict[str, object] = {}
            for param in _PARAMETER_RE.finditer(body):
                key = param.group(1).strip()
                args[key] = self._decode_param(name, key, param.group(2))

            tool_calls.append(
                ParsedToolCall(
                    id=_new_call_id(),
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                )
            )

        if not tool_calls:
            raise ValueError(
                f"<tool_call> block(s) found but no valid calls parsed from: {response}"
            )

        return ParsedToolResponse(content=content_before, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Incrementally process one decoded-token delta.

        Returns content text to forward to the client and any tool-call
        increments to emit, in the order they were produced. Content
        deltas have ``content`` set; tool-call deltas have one or more of
        ``id`` / ``name`` / ``arguments`` set.
        """
        if delta:
            self._buffer += delta

        deltas: list[ParsedToolCallDelta] = []

        progressed = True
        while progressed:
            progressed = False

            if self._state == _STATE_INIT:
                idx = self._buffer.find(TOOL_CALL_OPEN)
                if idx >= 0:
                    # Only text *before the first* tool call is assistant
                    # content. Text between back-to-back calls (the template's
                    # "\n" separator) is structural and must be suppressed, not
                    # streamed as content.
                    if idx > 0 and self._tool_index == -1:
                        deltas.append(
                            ParsedToolCallDelta(
                                index=0,
                                content=self._buffer[:idx],
                            )
                        )
                    self._buffer = self._buffer[idx + len(TOOL_CALL_OPEN) :]
                    self._state = _STATE_AFTER_TOOL_CALL_OPEN
                    self._tool_index += 1
                    self._param_count_in_call = 0
                    progressed = True
                else:
                    # Hold back any tail that could be a partial sentinel.
                    keep = _max_holdback(self._buffer, TOOL_CALL_OPEN)
                    if keep < len(self._buffer):
                        emit = self._buffer[: len(self._buffer) - keep]
                        # Suppress trailing text once a tool call has started;
                        # the template forbids content after a function call.
                        if self._tool_index == -1:
                            deltas.append(
                                ParsedToolCallDelta(index=0, content=emit)
                            )
                        self._buffer = self._buffer[len(self._buffer) - keep :]

            elif self._state == _STATE_AFTER_TOOL_CALL_OPEN:
                # Looking for "<function=NAME>".
                start = self._buffer.find(FUNCTION_OPEN_PREFIX)
                if start < 0:
                    break  # need more
                end = self._buffer.find(">", start + len(FUNCTION_OPEN_PREFIX))
                if end < 0:
                    break  # need more
                name = self._buffer[
                    start + len(FUNCTION_OPEN_PREFIX) : end
                ].strip()
                self._buffer = self._buffer[end + 1 :]
                self._current_function = name
                deltas.append(
                    ParsedToolCallDelta(
                        index=self._tool_index,
                        id=_new_call_id(),
                        name=name,
                        arguments="{",
                    )
                )
                self._state = _STATE_IN_FUNCTION
                progressed = True

            elif self._state == _STATE_IN_FUNCTION:
                # Either "<parameter=KEY>" or "</function>".
                param_start = self._buffer.find(PARAMETER_OPEN_PREFIX)
                func_close = self._buffer.find(FUNCTION_CLOSE)

                next_param = param_start if param_start >= 0 else None
                next_close = func_close if func_close >= 0 else None
                if next_param is not None and (
                    next_close is None or next_param < next_close
                ):
                    end = self._buffer.find(
                        ">", next_param + len(PARAMETER_OPEN_PREFIX)
                    )
                    if end < 0:
                        break  # need more
                    self._current_param_key = self._buffer[
                        next_param + len(PARAMETER_OPEN_PREFIX) : end
                    ].strip()
                    self._buffer = self._buffer[end + 1 :]
                    self._state = _STATE_IN_PARAMETER
                    progressed = True
                elif next_close is not None:
                    self._buffer = self._buffer[
                        next_close + len(FUNCTION_CLOSE) :
                    ]
                    deltas.append(
                        ParsedToolCallDelta(
                            index=self._tool_index,
                            arguments="}",
                        )
                    )
                    self._state = _STATE_AFTER_FUNCTION_CLOSE
                    progressed = True
                else:
                    break  # need more

            elif self._state == _STATE_IN_PARAMETER:
                close = self._buffer.find(PARAMETER_CLOSE)
                if close < 0:
                    break  # need more
                # Strip the template's one newline on each side (as
                # ``_PARAMETER_RE`` does) once the whole value is buffered.
                raw_value = self._buffer[:close]
                raw_value = raw_value.removeprefix("\n").removesuffix("\n")
                self._buffer = self._buffer[close + len(PARAMETER_CLOSE) :]

                value = self._decode_param(
                    self._current_function, self._current_param_key, raw_value
                )
                fragment = (
                    ("," if self._param_count_in_call > 0 else "")
                    + json.dumps(self._current_param_key)
                    + ":"
                    + json.dumps(value, ensure_ascii=False)
                )
                deltas.append(
                    ParsedToolCallDelta(
                        index=self._tool_index,
                        arguments=fragment,
                    )
                )
                self._param_count_in_call += 1
                self._state = _STATE_IN_FUNCTION
                progressed = True

            elif self._state == _STATE_AFTER_FUNCTION_CLOSE:
                close = self._buffer.find(TOOL_CALL_CLOSE)
                if close < 0:
                    break  # need more
                self._buffer = self._buffer[close + len(TOOL_CALL_CLOSE) :]
                self._state = _STATE_INIT
                progressed = True

        if deltas:
            return deltas
        # No deltas emitted this chunk. If we're mid-tool-call (any state
        # other than INIT), we've consumed the chunk's structural tokens into
        # the buffer and will emit them later — return [] (not None) so the
        # router SUPPRESSES the raw tokens rather than leaking them as content
        # (e.g. a partial "<parameter=" / unterminated "</parameter" held while
        # awaiting the rest of the tag).
        if self._state != _STATE_INIT:
            return []
        # In INIT: once a tool call has started, any buffered text is an
        # inter-call separator or trailing token — return [] to suppress it
        # (the router passes raw tokens through as content on None). Only
        # before the first call do we defer to that raw-content passthrough —
        # BUT if we're holding back bytes that partially match a sentinel,
        # return [] to suppress the raw tokens so they don't leak as content.
        return [] if self._tool_index >= 0 or self._buffer else None

    def reset(self) -> None:
        """Reset internal state for a new streaming session."""
        self._buffer = ""
        self._state = _STATE_INIT
        self._tool_index = -1
        self._param_count_in_call = 0
        self._current_param_key = ""
        self._current_function = ""
        self._tool_schemas = {}

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """Records the schemas for typed argument decoding while streaming.

        See ``ToolParser.set_streaming_tool_schemas`` and
        :meth:`set_tool_schemas`.
        """
        self.set_tool_schemas(schemas)
