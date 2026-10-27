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
"""Hy4 tool calls under the grammar serving builds for them.

The grammar comes from ``HYV4ToolParser.generate_tool_call_grammar`` and is
compiled over a byte-level vocabulary, so each completion is fed as its UTF-8
bytes. Calls are spelled as Hy4's chat template writes them: string values
bare, every other value as JSON.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any

import pytest
from max import _xgrammar as xgrammar
from max.pipelines.architectures.hy_v4.tool_parser import HYV4ToolParser
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    STRUCTURAL_MARKERS_BY_MODEL,
    special_token_ids_for_markers,
)

_INFO = xgrammar.TokenizerInfo(
    [bytes([i]) for i in range(256)] + [b"</s>"],
    vocab_type=xgrammar.VocabType.RAW,
    vocab_size=257,
    stop_token_ids=[256],
)
_S = ":opensource"


def _call(name: str, args: dict[str, Any]) -> str:
    pairs = "".join(
        f"<arg_key{_S}>{k}</arg_key{_S}><arg_value{_S}>"
        f"{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
        f"</arg_value{_S}>"
        for k, v in args.items()
    )
    return f"<tool_call{_S}>{name}{pairs}</tool_call{_S}>"


def _section(*calls: str) -> str:
    return f"<tool_calls{_S}>{''.join(calls)}</tool_calls{_S}>"


def _tool(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters, "strict": True},
    }


def _accepts(tools: list[dict[str, Any]], tool_choice: str, text: str) -> bool:
    grammar = HYV4ToolParser.generate_tool_call_grammar(
        tools=tools, tool_choice=tool_choice
    )
    compiled = xgrammar.GrammarCompiler(_INFO).compile_structural_tag(
        xgrammar.StructuralTag.model_validate_json(grammar)
    )
    matcher = xgrammar.GrammarMatcher(compiled)
    return all(matcher.accept_token(b) for b in text.encode()) and bool(
        matcher.is_completed()
    )


# Schemas and wrong values from llm-fuzz's tc_schema_enforcement cases; the
# wrong values are the ones Hy4 returned on nightly runs.
_Args = dict[str, Any]
_CASES: dict[str, tuple[_Args, _Args, _Args]] = {
    "ref_defs_boolean_type": (
        _tool(
            "save_settings",
            {
                "type": "object",
                "properties": {"settings": {"$ref": "#/$defs/UserSettings"}},
                "$defs": {
                    "UserSettings": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "notifications": {"type": "boolean"},
                            "dark_mode": {"type": "boolean"},
                            "auto_save": {"type": "boolean"},
                        },
                        "required": [
                            "username",
                            "notifications",
                            "dark_mode",
                            "auto_save",
                        ],
                    }
                },
                "required": ["settings"],
                "additionalProperties": False,
            },
        ),
        {
            "settings": {
                "username": "dana",
                "notifications": True,
                "dark_mode": True,
                "auto_save": False,
            }
        },
        {
            "settings": {
                "username": "dana",
                "notifications": "yes",
                "dark_mode": "enabled",
                "auto_save": "on",
            }
        },
    ),
    "ref_defs_enum_enforced": (
        _tool(
            "update_user",
            {
                "type": "object",
                "properties": {"user": {"$ref": "#/$defs/User"}},
                "$defs": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["active", "inactive", "suspended"],
                            },
                        },
                        "required": ["name", "status"],
                    }
                },
                "required": ["user"],
                "additionalProperties": False,
            },
        ),
        {"user": {"name": "Alice", "status": "inactive"}},
        {"user": {"name": "Alice", "status": "deleted"}},
    ),
    "ref_defs_anyof_nullable": (
        _tool(
            "register_user",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "backup_contact": {
                        "anyOf": [
                            {"$ref": "#/$defs/Contact"},
                            {"type": "null"},
                        ]
                    },
                },
                "$defs": {
                    "Contact": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"},
                        },
                        "required": ["email", "phone"],
                    }
                },
                "required": ["name", "backup_contact"],
                "additionalProperties": False,
            },
        ),
        {"name": "Bob", "backup_contact": None},
        {"name": "Bob", "backup_contact": "none"},
    ),
    "ref_defs_recursive_enum_leaf": (
        _tool(
            "create_fs_tree",
            {
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/FsNode"}},
                "$defs": {
                    "FsNode": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": ["file", "directory"],
                            },
                            "children": {
                                "type": ["array", "null"],
                                "items": {"$ref": "#/$defs/FsNode"},
                            },
                        },
                        "required": ["name", "kind", "children"],
                    }
                },
                "required": ["root"],
                "additionalProperties": False,
            },
        ),
        {
            "root": {
                "name": "src",
                "kind": "directory",
                "children": [
                    {"name": "a.txt", "kind": "file", "children": None}
                ],
            }
        },
        {
            "root": {
                "name": "src",
                "kind": "directory",
                "children": [
                    {"name": "link.txt", "kind": "symlink", "children": None}
                ],
            }
        },
    ),
    "type_list_object_with_properties": (
        _tool(
            "save_document",
            {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "metadata": {
                        "type": ["object", "null"],
                        "properties": {
                            "author": {"type": "string"},
                            "version": {"type": "integer"},
                        },
                        "required": ["author", "version"],
                    },
                },
                "required": ["label", "metadata"],
                "additionalProperties": False,
            },
        ),
        {"label": "report", "metadata": {"author": "42", "version": 3}},
        {"label": "report", "metadata": {"author": 42, "version": "three"}},
    ),
}

_CHOICES = ["required", "auto"]


@pytest.mark.parametrize("tool_choice", _CHOICES)
@pytest.mark.parametrize("case", _CASES)
def test_schema_valid_call_accepted(case: str, tool_choice: str) -> None:
    tool, good, _ = _CASES[case]
    name = tool["function"]["name"]
    assert _accepts([tool], tool_choice, _section(_call(name, good)))


@pytest.mark.parametrize("tool_choice", _CHOICES)
@pytest.mark.parametrize("case", _CASES)
def test_wrong_type_value_rejected(case: str, tool_choice: str) -> None:
    tool, _, bad = _CASES[case]
    name = tool["function"]["name"]
    assert not _accepts([tool], tool_choice, _section(_call(name, bad)))


@pytest.mark.parametrize("tool_choice", _CHOICES)
@pytest.mark.parametrize("case", _CASES)
def test_wrong_name_rejected(case: str, tool_choice: str) -> None:
    tool, good, _ = _CASES[case]
    assert not _accepts([tool], tool_choice, _section(_call("other", good)))


@pytest.mark.parametrize("tool_choice", _CHOICES)
@pytest.mark.parametrize("case", _CASES)
def test_missing_required_key_rejected(case: str, tool_choice: str) -> None:
    tool, good, _ = _CASES[case]
    name = tool["function"]["name"]
    partial = dict(list(good.items())[1:])
    assert not _accepts([tool], tool_choice, _section(_call(name, partial)))


_KINDS_TOOL = _tool(
    "kinds",
    {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "z": {"type": "null"},
            "e": {"type": "string", "enum": ["red", "green"]},
            "o": {
                "type": "object",
                "properties": {"k": {"type": "integer"}},
                "required": ["k"],
            },
            "a": {"type": "array", "items": {"type": "string"}},
            "r": {"$ref": "#/$defs/Point"},
        },
        "$defs": {
            "Point": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "tag": {"enum": [1, "a"]},
                },
                "required": ["x", "tag"],
            }
        },
        "required": ["s", "i", "n", "b", "z", "e", "o", "a", "r"],
        "additionalProperties": False,
    },
)
_KINDS = {
    "s": "two words, 42 and {braces}",
    "i": -7,
    "n": 2.5e-3,
    "b": False,
    "z": None,
    "e": "green",
    "o": {"k": 1},
    "a": ["x", "y z"],
    "r": {"x": 3, "tag": 1},
}


@pytest.mark.parametrize("tool_choice", _CHOICES)
def test_every_value_kind_accepted_and_typed(tool_choice: str) -> None:
    text = _section(_call("kinds", _KINDS))
    assert _accepts([_KINDS_TOOL], tool_choice, text)
    parser = HYV4ToolParser()
    (call,) = parser.parse_complete(text).tool_calls
    args = parser.coerce_arguments(
        json.loads(call.arguments), _KINDS_TOOL["function"]["parameters"]
    )
    assert args == _KINDS
    assert type(args["b"]) is bool and type(args["i"]) is int


@pytest.mark.parametrize(
    "text",
    [
        # Framing whitespace the template never writes; the parser fails to
        # pair it, or reads it into a value.
        _section(_call("kinds", _KINDS)).replace(
            f"<arg_value{_S}>-7", f"<arg_value{_S}> -7"
        ),
        _section(_call("kinds", _KINDS)).replace(
            f"</arg_key{_S}><arg_value{_S}>", f"</arg_key{_S}>\n<arg_value{_S}>"
        ),
        _section(_call("kinds", _KINDS)).replace(
            f"</arg_value{_S}><arg_key{_S}>", f"</arg_value{_S}>\n<arg_key{_S}>"
        ),
    ],
    ids=["before_json_value", "between_key_and_value", "between_pairs"],
)
def test_framing_whitespace_rejected(text: str) -> None:
    assert not _accepts([_KINDS_TOOL], "required", text)


# Hy4's tool-call markers at their ids in the checkpoint's vocabulary.
_MARKER_IDS = {
    f"<tool_call{_S}>": 120033,
    f"</tool_call{_S}>": 120034,
    f"<tool_calls{_S}>": 120035,
    f"</tool_calls{_S}>": 120036,
    f"<arg_key{_S}>": 120040,
    f"</arg_key{_S}>": 120041,
    f"<arg_value{_S}>": 120042,
    f"</arg_value{_S}>": 120043,
}
_MARKER_RE = re.compile("|".join(map(re.escape, _MARKER_IDS)))


def _hy4_markers() -> tuple[str, ...]:
    return STRUCTURAL_MARKERS_BY_MODEL.get(HYV4ToolParser.XGRAMMAR_FORMAT, ())


class _MarkerDelegate:
    unk_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return _MARKER_IDS.get(token)


@functools.cache
def _marker_compiler() -> xgrammar.GrammarCompiler:
    """A compiler over bytes plus Hy4's markers at their real ids, with the
    markers serving registers special marked special."""
    stop = max(_MARKER_IDS.values()) + 1
    vocab = [bytes([i]) for i in range(256)]
    vocab += [f"<unused_{i}>".encode() for i in range(256, stop)]
    for marker, token_id in _MARKER_IDS.items():
        vocab[token_id] = marker.encode()
    vocab.append(b"</s>")
    markers = _hy4_markers()
    info = xgrammar.TokenizerInfo(
        vocab,
        vocab_type=xgrammar.VocabType.RAW,
        vocab_size=len(vocab),
        stop_token_ids=[stop],
        special_token_ids=sorted(
            special_token_ids_for_markers(markers, _MarkerDelegate())
        ),
    )
    return xgrammar.GrammarCompiler(info)


def _marker_tokens(text: str) -> list[int]:
    """Tokenizes ``text`` as Hy4 does its markers: one token each."""
    tokens: list[int] = []
    pos = 0
    for m in _MARKER_RE.finditer(text):
        tokens += text[pos : m.start()].encode()
        tokens.append(_MARKER_IDS[m.group()])
        pos = m.end()
    return tokens + list(text[pos:].encode())


def _accepts_tokens(
    tools: list[dict[str, Any]], tool_choice: str, text: str
) -> bool:
    grammar = HYV4ToolParser.generate_tool_call_grammar(
        tools=tools, tool_choice=tool_choice
    )
    compiled = _marker_compiler().compile_structural_tag(
        xgrammar.StructuralTag.model_validate_json(grammar)
    )
    matcher = xgrammar.GrammarMatcher(compiled)
    return all(matcher.accept_token(t) for t in _marker_tokens(text)) and bool(
        matcher.is_completed()
    )


def test_argument_markers_registered_special() -> None:
    markers = _hy4_markers()
    ids = special_token_ids_for_markers(markers, _MarkerDelegate())
    assert ids == {120040, 120041, 120042, 120043}


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("tool_choice", _CHOICES)
def test_call_with_marker_tokens_accepted(
    strict: bool, tool_choice: str
) -> None:
    tool = _tool("kinds", _KINDS_TOOL["function"]["parameters"])
    tool["function"]["strict"] = strict
    text = _section(_call("kinds", _KINDS))
    assert _accepts_tokens([tool], tool_choice, text)


@pytest.mark.parametrize(
    "old,new",
    [
        (f"<arg_value{_S}>-7", f"<arg_value{_S}> -7"),
        (f"</arg_key{_S}><arg_value{_S}>", f"</arg_key{_S}>\n<arg_value{_S}>"),
        (f"</arg_value{_S}><arg_key{_S}>", f"</arg_value{_S}>\n<arg_key{_S}>"),
    ],
    ids=["before_json_value", "between_key_and_value", "between_pairs"],
)
def test_framing_whitespace_rejected_with_marker_tokens(
    old: str, new: str
) -> None:
    text = _section(_call("kinds", _KINDS)).replace(old, new)
    assert not _accepts_tokens([_KINDS_TOOL], "required", text)


def test_close_marker_token_inside_json_string_rejected() -> None:
    # The parser cuts a value at its first close marker, so a JSON string
    # holding that token would come back truncated, as text.
    args = {**_KINDS, "a": [f"x</arg_value{_S}>", "y z"]}
    assert not _accepts_tokens(
        [_KINDS_TOOL], "required", _section(_call("kinds", args))
    )


def test_angle_bracket_in_json_string_accepted() -> None:
    args = {**_KINDS, "a": ["<div>", "a < b"]}
    text = _section(_call("kinds", args))
    assert _accepts_tokens([_KINDS_TOOL], "required", text)
    parser = HYV4ToolParser()
    (call,) = parser.parse_complete(text).tool_calls
    parsed = parser.coerce_arguments(
        json.loads(call.arguments), _KINDS_TOOL["function"]["parameters"]
    )
    assert parsed["a"] == ["<div>", "a < b"]


def test_escaped_angle_bracket_in_json_string_accepted() -> None:
    text = _section(_call("kinds", _KINDS)).replace('"y z"', '"a \\u003c b"')
    assert _accepts([_KINDS_TOOL], "required", text)
    parser = HYV4ToolParser()
    (call,) = parser.parse_complete(text).tool_calls
    args = parser.coerce_arguments(
        json.loads(call.arguments), _KINDS_TOOL["function"]["parameters"]
    )
    assert args["a"] == ["x", "a < b"]


_OPEN_TOOL = _tool(
    "open_schema",
    {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "o": {"type": "object", "properties": {"k": {"type": "integer"}}},
            "free": {"type": "object"},
        },
        "required": ["a"],
    },
)


def test_undeclared_argument_follows_additional_properties() -> None:
    # As in JSON Schema, an unset additionalProperties admits undeclared
    # arguments; minProperties here needs two of them.
    extra = {"a": "x", "age": 30, "email": "a@b.c"}
    params = {**_OPEN_TOOL["function"]["parameters"], "minProperties": 3}
    tool = _tool("open_schema", params)
    assert _accepts([tool], "required", _section(_call("open_schema", extra)))
    closed = _tool(
        "open_schema",
        {**params, "additionalProperties": False, "minProperties": 1},
    )
    assert not _accepts(
        [closed], "required", _section(_call("open_schema", extra))
    )


@pytest.mark.parametrize(
    "args",
    [
        {"a": "x", "o": {"k": 1, "extra": True}},
        {"a": "x", "free": {"Content-Type": "application/json", "Accept": "*"}},
    ],
    ids=["declared_object", "free_object"],
)
def test_nested_object_takes_keys_its_schema_allows(args: _Args) -> None:
    assert _accepts(
        [_OPEN_TOOL], "required", _section(_call("open_schema", args))
    )
