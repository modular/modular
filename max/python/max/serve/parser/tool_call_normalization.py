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

    OpenAI's structured outputs require an object root, but JSON Schema (and
    the grammar backend, llguidance) also accept a missing ``type`` ("any")
    and a type *union* that includes ``object``
    (``"type": ["object", "array", "string"]``); both compile to a
    constraining grammar, so we accept them. A root pinned to a single
    *non-object scalar* type (e.g. ``{"type": "string"}``) is still rejected:
    callers expect a JSON object back, and this matches OpenAI's contract.

    - ``None`` / empty dict are acceptable (no constraint).
    - Missing ``type`` is acceptable (means "any").
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
