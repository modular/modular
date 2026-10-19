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

"""Tool call parser for Hy4.

Hy4 wraps its calls in one ``<tool_calls:opensource>`` section. Each call is
the bare function name followed by ``<arg_key>``/``<arg_value>`` pairs::

    <tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris</arg_value:opensource></tool_call:opensource></tool_calls:opensource>

The chat template writes string values bare and every other value as JSON,
so values are typed from the tool's parameter schema, following vLLM's
``hy_v4_tool_parser``: without a schema every value stays a string.

MAX diverges from vLLM 0.30.0 on purpose for a property with no ``type``:
vLLM types it as a string, so a ``$ref``, ``enum`` or ``const`` value comes
back as JSON text. MAX resolves local ``$ref`` pointers against the
``parameters`` schema and reads types from ``enum``/``const`` literals.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import suppress
from typing import Any, ClassVar

from max._xgrammar.builtin_structural_tag import (
    register_model_structural_tag,
)
from max._xgrammar.openai_tool_call_schema import FunctionToolParam
from max._xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    Format,
    OrFormat,
    SequenceFormat,
    StarFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    build_xgrammar_tool_grammar,
)
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    generate_call_id,
    partial_tag_overlap,
    register,
)
from max.pipelines.modeling.types import ParsedToolCall, ParsedToolResponse

from .reasoning import TOKEN_SUFFIX

_SECTION_OPEN = f"<tool_calls{TOKEN_SUFFIX}>"
_SECTION_CLOSE = f"</tool_calls{TOKEN_SUFFIX}>"
_CALL_OPEN = f"<tool_call{TOKEN_SUFFIX}>"
_CALL_CLOSE = f"</tool_call{TOKEN_SUFFIX}>"
_KEY_OPEN = f"<arg_key{TOKEN_SUFFIX}>"
_KEY_CLOSE = f"</arg_key{TOKEN_SUFFIX}>"
_VALUE_OPEN = f"<arg_value{TOKEN_SUFFIX}>"
_VALUE_CLOSE = f"</arg_value{TOKEN_SUFFIX}>"

_CALL_RE = re.compile(
    re.escape(_CALL_OPEN) + r"(.*?)" + re.escape(_CALL_CLOSE), re.DOTALL
)
_ARG_PAIR_RE = re.compile(
    re.escape(_KEY_OPEN)
    + r"(.*?)"
    + re.escape(_KEY_CLOSE)
    + re.escape(_VALUE_OPEN)
    + r"(.*?)"
    + re.escape(_VALUE_CLOSE),
    re.DOTALL,
)
_OPEN_VALUE_RE = re.compile(
    re.escape(_KEY_OPEN)
    + r"(.*?)"
    + re.escape(_KEY_CLOSE)
    + re.escape(_VALUE_OPEN)
    + r"(.*)",
    re.DOTALL,
)

_TYPE_ALIASES = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "bool": "boolean",
    "binary": "boolean",
    "int": "integer",
    "float": "number",
    "double": "number",
    "list": "array",
    "dict": "object",
    "map": "object",
}


def _normalize_type(raw: str) -> str:
    if raw in _TYPE_ALIASES:
        return _TYPE_ALIASES[raw]
    lower = raw.lower()
    if lower.startswith(("int", "uint", "long", "short", "unsigned")):
        return "integer"
    if lower.startswith(("num", "float")):
        return "number"
    return raw


_LITERAL_TYPES = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    type(None): "null",
    list: "array",
    dict: "object",
}

_Schema = Mapping[str, Any] | bool

_FREE_TEXT_KEYS = {"type", "title", "description", "default", "examples"}
# What the grammar enforces of a strict tool's schema: its keys, required or
# not; everything else must be an annotation.
_FRAMED_KEYS = _FREE_TEXT_KEYS | {
    "properties",
    "required",
    "additionalProperties",
    "$defs",
    "definitions",
    "$schema",
}


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> _Schema | None:
    """Follows a local ``#/...`` pointer such as ``#/$defs/Item``."""
    if not ref.startswith("#/"):
        return None
    node: object = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        node = node.get(part) if isinstance(node, Mapping) else None
    return node if isinstance(node, (Mapping, bool)) else None


def _schema_types(
    arg_schema: _Schema,
    root: Mapping[str, Any],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Returns the JSON types an argument schema admits.

    A declared ``null`` type is dropped, so a nullable string stays text; a
    literal ``null`` in ``enum``/``const`` is kept, since it is named exactly.
    A schema that types nothing, including a boolean schema and an
    unresolvable or cyclic ``$ref``, admits a string.
    """
    if not isinstance(arg_schema, Mapping):
        return {"string"}
    ref = arg_schema.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        if (target := _resolve_ref(ref, root)) is not None:
            return _schema_types(target, root, seen | {ref})
    if "type" in arg_schema:
        declared = arg_schema["type"]
        names = declared if isinstance(declared, list) else [declared]
        return {_normalize_type(t) for t in names} - {"null"}
    if "enum" in arg_schema or "const" in arg_schema:
        literals = arg_schema.get("enum", [arg_schema.get("const")])
        return {_LITERAL_TYPES.get(type(v), "string") for v in literals}
    if options := arg_schema.get("anyOf") or arg_schema.get("oneOf"):
        return set().union(*(_schema_types(o, root, seen) for o in options))
    return {"string"}


def _parse_number(raw: str) -> int | float | None:
    try:
        return float(raw) if any(c in raw for c in ".eE") else int(raw)
    except ValueError:
        return None


def _same(value: object, literal: object) -> bool:
    return value == literal and isinstance(value, bool) == isinstance(
        literal, bool
    )


def _admits(
    value: object,
    arg_schema: _Schema,
    root: Mapping[str, Any],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Whether ``arg_schema`` admits ``value``, checking its type and its
    ``enum``/``const`` literals, through ``$ref`` and each ``anyOf``/``oneOf``
    branch, in the order :func:`_schema_types` reads them."""
    if not isinstance(arg_schema, Mapping):
        return arg_schema
    ref = arg_schema.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        if (target := _resolve_ref(ref, root)) is not None:
            return _admits(value, target, root, seen | {ref})
    if "type" in arg_schema:
        declared = arg_schema["type"]
        names = declared if isinstance(declared, list) else [declared]
        types = {_normalize_type(t) for t in names}
        kind = _LITERAL_TYPES.get(type(value))
        if kind not in types and not (kind == "integer" and "number" in types):
            return False
    if "enum" in arg_schema or "const" in arg_schema:
        literals = arg_schema.get("enum", [arg_schema.get("const")])
        return any(_same(value, literal) for literal in literals)
    if "type" in arg_schema:
        return True
    if options := arg_schema.get("anyOf") or arg_schema.get("oneOf"):
        return any(_admits(value, o, root, seen) for o in options)
    return True


def _readings(raw: str, types: set[str]) -> Iterator[object]:
    """Yields the readings of a bare payload that ``types`` admit, best first."""
    if "boolean" in types and raw.lower() in ("true", "false"):
        yield raw.lower() == "true"
    if "integer" in types:
        with suppress(ValueError):
            yield int(raw)
    if "number" in types and (number := _parse_number(raw)) is not None:
        yield number
    if types - {"string", "boolean", "integer", "number"}:
        with suppress(ValueError):
            value = json.loads(raw)
            kind = _LITERAL_TYPES.get(type(value))
            if kind in types or (kind == "integer" and "number" in types):
                yield value
    if "string" in types:
        yield raw


def _parse_value(
    raw: str, arg_schema: _Schema, root: Mapping[str, Any]
) -> object:
    """Types one bare ``<arg_value>`` payload from its argument schema.

    The first reading the schema admits wins. A value no reading fits is kept
    as text: a wrong type is better than a lost argument.
    """
    types = _schema_types(arg_schema, root)
    for value in _readings(raw, types):
        if _admits(value, arg_schema, root):
            return value
    if "string" in types:
        return raw
    for decode in (json.loads, ast.literal_eval):
        try:
            return decode(raw)
        except Exception:
            pass
    return raw


def _property(root: Mapping[str, Any], key: str) -> _Schema:
    """The schema of argument ``key``: its property, else
    ``additionalProperties``."""
    properties = root.get("properties", {})
    if key in properties:
        return properties[key]
    extra = root.get("additionalProperties", {})
    return extra if isinstance(extra, (Mapping, bool)) else {}


def _coerce(
    args: Mapping[str, object], schema: Mapping[str, Any] | None
) -> dict[str, object]:
    root = schema or {}
    return {
        key: _parse_value(value, _property(root, key), root)
        if isinstance(value, str)
        else value
        for key, value in args.items()
    }


def _item(key: str, value: object) -> str:
    return f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"


def _split_call(body: str) -> tuple[str, str]:
    """Splits a call body into its function name and argument tags."""
    key_idx = body.find(_KEY_OPEN)
    if key_idx == -1:
        return body.strip(), ""
    return body[:key_idx].strip(), body[key_idx:]


def _strict_calls(response: str) -> list[tuple[str, dict[str, object]]]:
    """Parses the tool section, raising ``ValueError`` on any malformed tag."""
    start = response.find(_SECTION_OPEN)
    end = response.find(_SECTION_CLOSE)
    if start == -1 or end == -1 or end < start:
        raise ValueError("malformed tool section wrapper")
    section = response[start + len(_SECTION_OPEN) : end]
    bodies = _CALL_RE.findall(section)
    if not (
        section.count(_CALL_OPEN) == section.count(_CALL_CLOSE) == len(bodies)
    ):
        raise ValueError("malformed tool call wrapper")
    calls = []
    for body in bodies:
        name, args_text = _split_call(body)
        if not name:
            raise ValueError("empty function name")
        if args_text.count(_KEY_OPEN) != args_text.count(
            _KEY_CLOSE
        ) or args_text.count(_VALUE_OPEN) != args_text.count(_VALUE_CLOSE):
            raise ValueError("malformed argument tags")
        if _ARG_PAIR_RE.sub("", args_text).strip():
            raise ValueError("unparsed argument payload")
        calls.append((name, dict(_ARG_PAIR_RE.findall(args_text))))
    if not calls:
        raise ValueError("no tool calls")
    return calls


def _argument(key: str | None = None) -> SequenceFormat:
    """Frames one argument pair; without a ``key``, any key."""
    return SequenceFormat(
        elements=[
            ConstStringFormat(value=_KEY_OPEN),
            AnyTextFormat() if key is None else ConstStringFormat(value=key),
            ConstStringFormat(value=_KEY_CLOSE + _VALUE_OPEN),
            AnyTextFormat(),
            ConstStringFormat(value=_VALUE_CLOSE),
        ]
    )


def _is_free_text(arg_schema: _Schema) -> bool:
    if not isinstance(arg_schema, Mapping):
        return arg_schema
    return (
        set(arg_schema) <= _FREE_TEXT_KEYS
        and arg_schema.get("type", "string") == "string"
    )


def _unenforced(parameters: Mapping[str, Any]) -> str | None:
    """Names the first part of a strict tool's schema that the grammar, which
    frames keys and leaves values free text, cannot enforce."""
    if keywords := sorted(set(parameters) - _FRAMED_KEYS):
        return f"keyword '{keywords[0]}'"
    if parameters.get("type", "object") != "object":
        return "a non-object type"
    extra = parameters.get("additionalProperties", True)
    if not isinstance(extra, bool) and not _is_free_text(extra):
        return "additionalProperties"
    for key, arg_schema in (parameters.get("properties") or {}).items():
        if not _is_free_text(arg_schema):
            return f"argument '{key}'"
    return None


def _call(tool: FunctionToolParam, reject_unsupported: bool) -> TagFormat:
    name = tool.function.name
    parameters = tool.function.parameters
    if not isinstance(parameters, dict):
        parameters = {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    strict = tool.function.strict is not False
    if strict and reject_unsupported and (part := _unenforced(parameters)):
        raise InputError(
            f"Hy4 tool calls cannot enforce {part} of tool '{name}': argument"
            " values are free text."
        )
    keyed = bool(properties or required)
    if strict and not keyed and parameters.get("additionalProperties") is False:
        # The name is the whole body, so it cannot run on into more text.
        return TagFormat(
            begin=_CALL_OPEN,
            content=ConstStringFormat(value=name),
            end=_CALL_CLOSE,
        )
    if not (strict and keyed):
        return TagFormat(
            begin=_CALL_OPEN + name,
            content=StarFormat(content=_argument()),
            end=_CALL_CLOSE,
        )
    elements: list[Format] = [_argument(k) for k in properties if k in required]
    elements += [_argument(k) for k in required if k not in properties]
    optional: list[Format] = [
        _argument(k) for k in properties if k not in required
    ]
    if optional:
        # Optional keys may appear in any order, or not at all.
        elements.append(
            StarFormat(
                content=optional[0]
                if len(optional) == 1
                else OrFormat(elements=optional)
            )
        )
    return TagFormat(
        begin=_CALL_OPEN + name,
        content=SequenceFormat(elements=elements),
        end=_CALL_CLOSE,
    )


@register_model_structural_tag("hy_v4")
def _structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: object,
    tool_choice: str,
    reasoning: bool,
    reject_unsupported: bool = False,
    **kwargs: object,
) -> StructuralTag:
    """Frames calls, names and argument keys; values stay free text.

    Values are bare text that no JSON style spells, so only the skeleton is
    constrained. Adapted from vLLM's ``get_hy_v4_structural_tag``, which
    ignores ``strict`` and lets a tool without keys run its name on.
    """
    calls = TagsWithSeparatorFormat(
        tags=[_call(tool, reject_unsupported) for tool in tools],
        separator="",
        at_least_one=True,
        stop_after_first=tool_choice == "forced",
    )
    section: Format
    if tool_choice == "auto":
        section = TriggeredTagsFormat(
            triggers=[_SECTION_OPEN],
            tags=[
                TagFormat(
                    begin=_SECTION_OPEN, content=calls, end=_SECTION_CLOSE
                )
            ],
        )
    else:
        section = SequenceFormat(
            elements=[
                ConstStringFormat(value=_SECTION_OPEN),
                calls,
                ConstStringFormat(value=_SECTION_CLOSE),
            ]
        )
    if not reasoning:
        return StructuralTag(format=section)
    think = TagFormat(
        begin="", content=AnyTextFormat(), end=f"</think{TOKEN_SUFFIX}>"
    )
    return StructuralTag(format=SequenceFormat(elements=[think, section]))


@register("hy_v4")
class HYV4ToolParser(StructuralTagToolParser):
    """Parses Hy4 tool calls."""

    SECTION_BEGIN: ClassVar[str] = _SECTION_OPEN
    SECTION_END: ClassVar[str] = _SECTION_CLOSE
    CALL_BEGIN: ClassVar[str] = _CALL_OPEN
    CALL_END: ClassVar[str] = _CALL_CLOSE
    XGRAMMAR_FORMAT: ClassVar[str] = "hy_v4"

    def __init__(self) -> None:
        super().__init__()
        self._schemas: Mapping[str, dict[str, Any]] = {}
        self._active_schema: dict[str, Any] | None = None

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """Stores per-tool ``parameters`` schemas to type streamed values."""
        self._schemas = schemas

    def reset(self) -> None:
        super().reset()
        self._active_schema = None

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response; a malformed section stays content.

        Values are left as text here; the router types them through
        :meth:`coerce_arguments` when the request declared the tool.
        """
        try:
            calls = _strict_calls(response)
        except ValueError:
            return ParsedToolResponse(content=response, tool_calls=[])
        return ParsedToolResponse(
            content=response[: response.find(_SECTION_OPEN)] or None,
            tool_calls=[
                ParsedToolCall(
                    id=generate_call_id(),
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                )
                for name, args in calls
            ],
        )

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        reject_unsupported: bool = False,
        **kwargs: Any,
    ) -> str:
        """Returns a serialized xgrammar ``StructuralTag`` for Hy4 tool calls.

        Args:
            response_format_schema: Optional JSON schema; when set, the
                grammar also accepts a JSON response matching it.
            tools: OpenAI-style tool dicts.
            backend: Structured-output backend; must be ``"xgrammar"``.
            tool_choice: ``"auto"``, ``"required"``, or a named choice.
            reject_unsupported: Whether a strict tool whose argument values
                the grammar cannot enforce raises instead of going
                unconstrained.
            **kwargs: Ignored; accepts ``tokenizer`` and other kwargs.

        Returns:
            The StructuralTag serialized as a JSON string.

        Raises:
            InputError: If ``backend`` is not ``"xgrammar"``, or if
                ``reject_unsupported`` is set and a strict tool's schema
                constrains more than its keys.
        """
        if backend != "xgrammar":
            raise InputError(
                "Hy4 constrained tool calling requires the xgrammar backend; "
                "run with --structured-output-backend=xgrammar."
            )
        return build_xgrammar_tool_grammar(
            HYV4ToolParser.XGRAMMAR_FORMAT,
            tools or [],
            tool_choice if tool_choice is not None else "auto",
            response_format_schema=response_format_schema,
            reject_unsupported=reject_unsupported,
        )

    def coerce_arguments(
        self, args: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Types parsed text values from the tool's ``parameters`` schema."""
        return _coerce(args, schema)

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        raise NotImplementedError("parse_complete does not use sections")

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits a call body once its name boundary has arrived."""
        if not is_complete and _KEY_OPEN not in body:
            return None, None
        name, args_text = _split_call(body)
        self._active_schema = self._schemas.get(name)
        return name, args_text

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Builds a growing JSON object from the argument pairs seen so far.

        As in vLLM's ``_extract_streaming_incremental``, a string value
        streams while still open, and its closing quote and the closing
        brace are withheld until they arrive, so the streamed diffs
        concatenate into the final arguments.
        """
        root = self._active_schema or {}
        pairs = list(_ARG_PAIR_RE.finditer(args_text))
        # A repeated key stays a second pair: overwriting the first would
        # shrink text already streamed. JSON readers keep the last one, as
        # the complete parse does.
        items = [
            _item(key, _parse_value(value, _property(root, key), root))
            for key, value in (m.groups() for m in pairs)
        ]
        if is_complete:
            return "{" + ", ".join(items) + "}"
        tail = args_text[pairs[-1].end() if pairs else 0 :]
        if open_value := _OPEN_VALUE_RE.search(tail):
            key, value = open_value.groups()
            if _schema_types(_property(root, key), root) == {"string"}:
                value = value[
                    : len(value) - partial_tag_overlap(value, _VALUE_CLOSE)
                ]
                items.append(_item(key, value)[:-1])
        return "{" + ", ".join(items)
