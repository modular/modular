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
"""Shared helpers for normalizing OpenAI-style tool-call payloads."""

from __future__ import annotations

import json
from typing import Any


def normalize_tool_call_arguments(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns ``tool_calls`` with ``function.arguments`` coerced to a mapping.

    OpenAI's chat completion schema renders
    ``tool_calls[*].function.arguments`` as a JSON-encoded string, while
    most tool-use chat templates iterate the arguments as a mapping.

    - Non-string ``arguments`` (already a dict/list) are passed through.
    - Missing, ``None``, or empty-string ``arguments`` become ``{}``.
    - JSON strings are decoded; malformed JSON is passed through untouched
      so client-side encoding errors surface at the template layer instead
      of being silently swallowed here.

    The input list and its dicts are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        out = dict(tc)
        fn = out.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
            fn = dict(fn)
            if isinstance(args, (dict, list)):
                # Already a mapping/sequence; leave as-is.
                pass
            elif args is None or args == "":
                fn["arguments"] = {}
            elif isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    # Pass malformed JSON through so the template layer
                    # surfaces the original encoding error.
                    pass
            out["function"] = fn
        normalized.append(out)
    return normalized


def normalize_message_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Returns ``message`` with assistant ``tool_calls`` arguments coerced.

    Non-assistant messages and assistant messages without ``tool_calls``
    pass through unchanged. The input dict is not mutated.
    """
    if message.get("role") != "assistant":
        return message
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return message
    out = dict(message)
    out["tool_calls"] = normalize_tool_call_arguments(tool_calls)
    return out


_JSON_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)


def _validate_response_format_schema(
    schema: dict[str, Any] | None,
) -> None:
    """Validates the root ``type`` of a ``response_format.json_schema.schema``.

    OpenAI's structured outputs require an object root. A root pinned to a
    single *non-object scalar* type (e.g. ``{"type": "string"}``) is rejected:
    callers expect a JSON object back, and this matches OpenAI's contract. A
    type *union* that includes ``object``
    (``"type": ["object", "array", "string"]``) is accepted as an explicit
    caller choice.

    A *missing* root ``type`` is accepted here, but only because
    :func:`normalize_response_format_schema` defaults it to ``"object"``
    before the schema reaches the grammar backend. Compiling an untyped root
    directly against llguidance is unsafe: it permits a bare, unbounded
    top-level value (e.g. a string), and a model that degenerates into a
    repetition loop inside that value can never emit the only terminator (the
    closing quote), so generation runs to ``max_length``
    (see the runaway-output incident). Defaulting to ``"object"``
    forces the leading ``{`` and restores a terminating grammar.

    - ``None`` / empty dict are acceptable (normalized to an object root).
    - Missing ``type`` is acceptable (normalized to ``"object"``).
    - ``type`` may be a non-empty list of recognized types (a union).
    - A single ``type`` must be ``object``; any other single scalar is rejected.

    Raises:
        ValueError: If the root ``type`` is a single non-object type, a list
            containing an unrecognized type, or otherwise not a JSON Schema
            type.
    """
    if schema is None or schema == {}:
        return
    root_type = schema.get("type")
    if root_type is None:
        return
    if isinstance(root_type, list):
        invalid = [t for t in root_type if t not in _JSON_SCHEMA_TYPES]
        if not root_type or invalid:
            raise ValueError(
                "response_format.json_schema.schema: root 'type' list must "
                f"contain only JSON Schema types (got {root_type!r})"
            )
        return
    if root_type != "object":
        raise ValueError(
            "response_format.json_schema.schema: root 'type' must be "
            f"'object' or a type union including it (got {root_type!r})"
        )


# JSON Schema keywords that imply an object type when ``type`` is omitted.
# Mirrors xgrammar's ``json_schema_converter`` inference, which is the reason
# vLLM and SGLang (both on xgrammar) anchor such schemas to ``{`` and never
# hit the runaway.
_OBJECT_IMPLYING_KEYWORDS = frozenset(
    {"properties", "required", "additionalProperties", "patternProperties"}
)

# Keys whose values are themselves subschemas (recurse into these).
_SUBSCHEMA_KEYS = frozenset(
    {"items", "additionalItems", "contains", "not", "if", "then", "else"}
)

# Keys whose values map property/definition names to subschemas.
_SUBSCHEMA_MAP_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions"}
)

# Keys whose values are lists of subschemas.
_SUBSCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})


def normalize_response_format_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Returns ``schema`` with object-shaped untyped (sub)schemas given a type.

    A JSON Schema (sub)schema that omits ``type`` but carries an
    object-implying keyword (``properties``, ``required``,
    ``additionalProperties``, ``patternProperties``) is valid JSON Schema
    ("accept any type") but compiles, against the llguidance grammar backend,
    to a production that permits a bare, unbounded value -- including a JSON
    string. A model that loops inside that string can never emit its closing
    quote, so generation runs to ``max_length`` with ``finish_reason="length"``
    (the runaway-output incident).

    This pass mirrors xgrammar's type inference (the reason xgrammar-backed
    engines never hit this): where ``type`` is absent but an object-implying
    keyword is present, inject ``"type": "object"`` so the grammar anchors on
    ``{`` and stays terminating. The pass recurses through nested subschemas
    (property values, array items, ``$defs``, ``allOf``/``anyOf``/``oneOf``,
    etc.) so nested object-shaped schemas are anchored too.

    Deliberately left untouched:

    - A genuinely empty ``{}`` (no object-implying keyword): this is "any
      value" and is preserved for parity with xgrammar/SGLang and MAX's
      ``json_object`` semantics.
    - A ``type`` that is already present (a single type or a union list).

    The input is not mutated; a normalized copy is returned only where a
    change is needed.

    Args:
        schema: The ``response_format.json_schema.schema`` mapping (or any
            nested subschema).

    Returns:
        The schema with object types inferred where needed; the same object
        when no change applies.
    """
    return _normalize_subschema(schema)


def _normalize_subschema(node: Any) -> Any:
    """Recursively infer ``type: object`` for object-shaped untyped schemas."""
    if not isinstance(node, dict):
        return node

    changed = False
    out: dict[str, Any] = node

    # Infer an object type where it is omitted but implied.
    if "type" not in node and any(k in node for k in _OBJECT_IMPLYING_KEYWORDS):
        out = dict(node)
        out["type"] = "object"
        changed = True

    # Recurse into nested subschemas so inner object-shaped schemas are
    # anchored too (parity with xgrammar's recursive conversion).
    for key in _SUBSCHEMA_KEYS:
        if key in out and isinstance(out[key], dict):
            new_child = _normalize_subschema(out[key])
            if new_child is not out[key]:
                if not changed:
                    out = dict(out)
                    changed = True
                out[key] = new_child

    for key in _SUBSCHEMA_MAP_KEYS:
        mapping = out.get(key)
        if isinstance(mapping, dict):
            new_mapping: dict[str, Any] | None = None
            for name, child in mapping.items():
                new_child = _normalize_subschema(child)
                if new_child is not child:
                    if new_mapping is None:
                        new_mapping = dict(mapping)
                    new_mapping[name] = new_child
            if new_mapping is not None:
                if not changed:
                    out = dict(out)
                    changed = True
                out[key] = new_mapping

    for key in _SUBSCHEMA_LIST_KEYS:
        seq = out.get(key)
        if isinstance(seq, list):
            new_seq: list[Any] | None = None
            for i, child in enumerate(seq):
                new_child = _normalize_subschema(child)
                if new_child is not child:
                    if new_seq is None:
                        new_seq = list(seq)
                    new_seq[i] = new_child
            if new_seq is not None:
                if not changed:
                    out = dict(out)
                    changed = True
                out[key] = new_seq

    return out


def _normalize_tools_parameters(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns ``tools`` with ``function.parameters`` coerced to a dict.

    OpenAI's API normalizes ``tools[*].function.parameters: null`` to an
    empty parameter list (equivalent to omitting the field) and returns
    200. MAX should match.

    - Dict ``parameters`` are passed through unchanged.
    - ``None`` or missing ``parameters`` becomes ``{}``.
    - Other values pass through unchanged (downstream validation handles
      type errors).
    - Tool entries without a ``function`` dict pass through unchanged.

    The input list and its dicts are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        out = dict(tool)
        fn = out.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            params = fn.get("parameters")
            if params is None:
                fn["parameters"] = {}
            out["function"] = fn
        normalized.append(out)
    return normalized
