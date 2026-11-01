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
# ruff: noqa: RUF001
"""DeepSeek-V4.1 tool calls under the grammar serving builds for them.

The grammar comes from ``DeepseekV41ToolParser.generate_tool_call_grammar``
and is compiled by the serving xgrammar backend over a byte-level vocabulary,
so each completion is fed as its UTF-8 bytes. Completions are rendered by the
vendored encoder, or written by hand where a test needs output the encoder
never produces.
"""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import dsml_fixtures
import pytest
from max.pipelines.architectures.deepseekV4.reasoning import (
    DeepseekV4ReasoningParser,
)
from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    thinking_end_token,
)
from max.pipelines.architectures.deepseekV41.tool_parser import (
    DeepseekV41ToolParser,
)
from max.pipelines.context import (
    GrammarEnforcementState,
    StructuredOutputRegionDelimiters,
)
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    GrammarBackend,
)
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper
from max.pipelines.lib.tool_parsing import get_parser_cls
from max.pipelines.modeling.types import PipelineTokenizer

_N_VOCAB = 256
_EOS_ID = 0


class _FakeTikTokenTokenizer:
    """TikToken-shaped delegate whose token ``i`` is the single byte ``i``."""

    eos_token_id: int = _EOS_ID
    bos_token_id: int | None = None
    all_special_ids: list[int] = []

    def __init__(self) -> None:
        self.byte_decoder = {chr(b): b for b in range(256)}

    def __len__(self) -> int:
        return _N_VOCAB

    def get_vocab(self) -> dict[str, int]:
        return {chr(i): i for i in range(256)}

    def convert_ids_to_tokens(self, idx: int) -> str:
        return chr(idx)

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        return list(text.encode("utf-8"))


def _helper() -> StructuredOutputHelper:
    """The structured-output helper serving builds for ``deepseekv41``."""
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = _FakeTikTokenTokenizer()
    pipeline_tokenizer.eos_token_ids = {_EOS_ID}
    return StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        tool_parser_name="deepseekv41",
        backend_name="xgrammar",
    )


def _backend() -> GrammarBackend:
    backend = _helper().backend
    assert backend is not None
    return backend


def _ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _grammar(
    tools: list[dict[str, Any]],
    tool_choice: Any = "required",
    reject_unsupported: bool = False,
) -> str:
    return DeepseekV41ToolParser.generate_tool_call_grammar(
        tools=tools,
        tool_choice=tool_choice,
        reject_unsupported=reject_unsupported,
    )


def _accepts_fully(grammar: str, text: str) -> bool:
    matcher = _backend().create_matcher(grammar)
    ids = _ids(text)
    consumed = matcher.try_consume_tokens(ids)
    return consumed == len(ids) and matcher.is_accepting()


_MULTI_CALL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "lang": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_MULTI_CALL_TURN = dsml_fixtures.multi_call_turn(
    [
        ("get_weather", {"location": "Beijing", "days": 3}),
        ("search", {"query": 'a "quoted" <b>', "filters": {"lang": ["en"]}}),
    ]
)

# (completion region after </think>, tools offered)
_TURNS = [
    pytest.param(
        dsml_fixtures.after_think(dsml_fixtures.tool_turn(name)),
        dsml_fixtures.tools(name),
        id=name,
    )
    for name in dsml_fixtures.CONVERSATIONS
] + [
    pytest.param(
        dsml_fixtures.after_think(_MULTI_CALL_TURN)[len("Checking both.") :],
        _MULTI_CALL_TOOLS,
        id="multi_call",
    )
]


def test_parser_routes_to_v41_tag() -> None:
    """Serving builds a grammar only for parsers with the hook."""
    parser_cls = get_parser_cls("deepseekv41")
    assert parser_cls is DeepseekV41ToolParser
    assert hasattr(parser_cls, "generate_tool_call_grammar")
    assert DeepseekV41ToolParser.XGRAMMAR_FORMAT == "deepseek_v41"
    tools = dsml_fixtures.tools("weather_thinking")
    assert '"deepseek_v41_xml"' in _grammar(tools)


def test_region_starts_at_the_section_and_runs_to_eos() -> None:
    """Enforcement keys on the section the reader frames, and stays on after
    it, which the reader requires to end the turn."""
    assert DeepseekV41ToolParser.SECTION_BEGIN == "<｜DSML｜ calls>\n"
    assert StructuredOutputHelper._get_tool_region_tags("deepseekv41") == (
        DeepseekV41ToolParser.SECTION_BEGIN,
        None,
    )


@pytest.mark.parametrize(("turn", "tools"), _TURNS)
def test_encoder_turns_start_with_the_section(
    turn: str, tools: list[dict[str, Any]]
) -> None:
    assert turn.startswith("\n\n" + DeepseekV41ToolParser.SECTION_BEGIN)
    assert turn.endswith(DeepseekV41ToolParser.SECTION_END)


@pytest.mark.parametrize("reject_unsupported", [False, True])
@pytest.mark.parametrize(("turn", "tools"), _TURNS)
def test_required_accepts_encoder_turns(
    turn: str, tools: list[dict[str, Any]], reject_unsupported: bool
) -> None:
    grammar = _grammar(tools, "required", reject_unsupported)
    assert _accepts_fully(grammar, turn)


@pytest.mark.parametrize(("turn", "tools"), _TURNS)
def test_auto_accepts_encoder_turns_from_the_section(
    turn: str, tools: list[dict[str, Any]]
) -> None:
    assert _accepts_fully(_grammar(tools, "auto"), turn)


@pytest.mark.parametrize(
    "value",
    [1.5, -3, True, False, None, [1, "a", None], {"k": [1, {"z": False}]}],
    ids=["number", "integer", "true", "false", "null", "array", "object"],
)
def test_json_values_round_trip_through_grammar_and_parser(
    value: Any,
) -> None:
    """Every JSON type the grammar admits as ``string="false"`` parses back."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "emit",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {}},
                    "required": ["value"],
                },
            },
        }
    ]
    output = (
        '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="emit">\n'
        f'<｜DSML｜ parameter name="value" string="false">{json.dumps(value)}'
        "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    assert _accepts_fully(_grammar(tools), output)
    (call,) = DeepseekV41ToolParser().parse_complete(output).tool_calls
    assert json.loads(call.arguments) == {"value": value}


@pytest.mark.parametrize(
    ("parameters", "params", "arguments"),
    [
        pytest.param(
            {
                "properties": {'foo"bar': {"$ref": "#/definitions/foo%22bar"}},
                "definitions": {'foo"bar': {"type": "number"}},
            },
            '<｜DSML｜ parameter name="foo"bar" string="false">1.5'
            "</｜DSML｜ parameter>\n",
            {'foo"bar': 1.5},
            id="key_needing_escapes",
        ),
    ],
)
def test_grammar_admitted_calls_parse_to_schema_values(
    parameters: dict[str, Any], params: str, arguments: dict[str, Any]
) -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "emit", "parameters": parameters},
        }
    ]
    output = (
        '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="emit">\n'
        f"{params}</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    assert _accepts_fully(_grammar(tools, reject_unsupported=True), output)
    (call,) = DeepseekV41ToolParser().parse_complete(output).tool_calls
    assert json.loads(call.arguments) == arguments


def test_thinking_then_forced_tool_call_is_enforced_after_think() -> None:
    """Reasoning bypasses the tool grammar; the call is then enforced.

    Mirrors ``test_reasoning_constrained_decoding.py`` for a forced tool
    grammar instead of a JSON schema.
    """
    full_turn = dsml_fixtures.tool_turn("weather_thinking")
    assert full_turn.startswith("<think>")
    reasoning = full_turn[len("<think>") : full_turn.index(thinking_end_token)]
    end_ids = _ids(DeepseekV4ReasoningParser.REASONING_END)

    state = GrammarEnforcementState(tools_forced=True, grammar_enforced=False)
    state.thinking_region_delimiters = StructuredOutputRegionDelimiters(
        start_token_ids=None, end_token_ids=end_ids
    )
    state._in_thinking_region = True
    matcher = _backend().create_matcher(
        _grammar(dsml_fixtures.tools("weather_thinking"))
    )

    for token in _ids(reasoning) + end_ids:
        assert state.update_enforcement_state(token) is False
    assert state.grammar_enforced is True

    for token in _ids(dsml_fixtures.after_think(full_turn)):
        assert state.update_enforcement_state(token) is True
        assert matcher.try_consume_tokens([token]) == 1
    assert matcher.is_accepting()


def test_auto_region_spans_the_section_to_the_end() -> None:
    """Auto mode arms on the section's opening tag and holds to the end.

    The region is the one serving derives for ``deepseekv41``. On the
    flip-on the whole start marker is fed to the matcher, as
    ``TextGenerationContext._tokens_for_consume`` does.
    """
    region = _helper().tool_call_region_delimiters
    assert region is not None and region.start_token_ids is not None
    state = GrammarEnforcementState(tool_region=region)
    matcher = _backend().create_matcher(_grammar(_MULTI_CALL_TOOLS, "auto"))

    enforced = 0
    for token in _ids(dsml_fixtures.after_think(_MULTI_CALL_TURN)):
        was_enforced = state.grammar_enforced
        if not state.update_enforcement_state(token):
            continue
        feed = list(region.start_token_ids) if not was_enforced else [token]
        assert matcher.try_consume_tokens(feed) == len(feed)
        enforced += 1
    assert enforced > 0
    assert state.grammar_enforced is True
    assert matcher.is_accepting()
    assert matcher.try_consume_tokens(_ids("\n")) == 0


def _tool(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters},
    }


def _param(name: str, is_string: bool, value: str) -> str:
    flag = "true" if is_string else "false"
    return (
        f'<｜DSML｜ parameter name="{name}" string="{flag}">{value}'
        "</｜DSML｜ parameter>\n"
    )


def _invoke(name: str, params: str) -> str:
    return f'<｜DSML｜ invoke name="{name}">\n{params}</｜DSML｜ invoke>\n'


def _calls(*invokes: str) -> str:
    return "\n\n<｜DSML｜ calls>\n" + "".join(invokes) + "</｜DSML｜ calls>"


_WEATHER = _tool(
    "get_weather",
    {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer"},
        },
        "required": ["location"],
    },
)
_TWO_PARAMS = _param("location", True, "Beijing") + _param(
    "unit", True, "celsius"
)
_WEATHER_CALL = _invoke("get_weather", _TWO_PARAMS)


def _emit_grammar(value_schema: dict[str, Any]) -> str:
    """The grammar for one ``emit(value)`` tool."""
    return _grammar(
        [
            _tool(
                "emit",
                {
                    "type": "object",
                    "properties": {"value": value_schema},
                    "required": ["value"],
                    "definitions": {"n": {"type": "integer"}},
                },
            )
        ],
        reject_unsupported=True,
    )


def _emit(is_string: bool, value: str) -> str:
    return _calls(_invoke("emit", _param("value", is_string, value)))


# (value schema, string attribute, wire value, accepted). The reader takes
# string="true" as raw text and string="false" as JSON, so the attribute must
# agree with the value's type.
_ATTRIBUTE_CASES = [
    ({"type": "integer"}, False, "2", True),
    ({"type": "integer"}, True, "2", False),
    ({"const": 2}, False, "2", True),
    ({"const": 2}, True, "2", False),
    ({"enum": [1, 2, 3]}, False, "1", True),
    ({"enum": [1, 2, 3]}, True, "1", False),
    ({"type": "string"}, True, "abc", True),
    ({"type": "string"}, True, "42", True),
    ({"type": "string"}, False, "abc", False),
    ({"type": "string"}, False, "42", False),
    ({"type": "string", "pattern": "^a+$"}, True, "aa", True),
    ({"type": "string", "pattern": "^a+$"}, False, "aa", False),
    ({"$ref": "#/definitions/n"}, False, "3", True),
    ({"$ref": "#/definitions/n"}, True, "3", False),
    ({"type": "array", "items": {}}, False, '["a"]', True),
    ({"type": "array", "items": {}}, False, "[a]", False),
    ({"type": "array", "items": {}}, True, '["a"]', False),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, True, "abc", True),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, False, "null", True),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, False, "abc", False),
    ({"type": ["integer", "null"]}, False, "7", True),
    ({"type": ["integer", "null"]}, True, "7", False),
    ({}, True, "hello", True),
    ({}, False, '{"a": 1}', True),
    ({}, False, "hello", False),
]


@pytest.mark.parametrize(
    ("value_schema", "is_string", "value", "accepted"), _ATTRIBUTE_CASES
)
def test_string_attribute_is_bound_to_the_value_type(
    value_schema: dict[str, Any], is_string: bool, value: str, accepted: bool
) -> None:
    assert (
        _accepts_fully(_emit_grammar(value_schema), _emit(is_string, value))
        is accepted
    )


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_TWO_PARAMS.replace(">\n<", "><", 1), id="no_newline"),
        pytest.param(_TWO_PARAMS.replace(">\n<", ">\n\n<", 1), id="blank_line"),
        pytest.param(" " + _TWO_PARAMS, id="space_before_first"),
        pytest.param("\n" + _TWO_PARAMS, id="newline_before_first"),
        pytest.param(_TWO_PARAMS + "\n", id="blank_line_before_close"),
        pytest.param(_TWO_PARAMS.rstrip("\n"), id="no_newline_before_close"),
    ],
)
def test_parameter_framing_is_exactly_what_the_reader_takes(body: str) -> None:
    """The reader needs exactly ``>\\n`` after the header and each close."""
    grammar = _grammar([_WEATHER])
    assert _accepts_fully(grammar, _calls(_WEATHER_CALL))
    assert not _accepts_fully(grammar, _calls(_invoke("get_weather", body)))


def test_auto_admits_nothing_after_the_section() -> None:
    """The reader rejects any text after the calls block."""
    grammar = _grammar([_WEATHER], "auto")
    output = _calls(_WEATHER_CALL)
    assert _accepts_fully(grammar, "Checking." + output)
    assert not _accepts_fully(grammar, output + "\nDone.")
    assert not _accepts_fully(grammar, output + _calls(_WEATHER_CALL))


def test_v4_spelling_is_rejected() -> None:
    v4_output = (
        _calls(_WEATHER_CALL)
        .replace("｜DSML｜ calls>", "｜DSML｜tool_calls>")
        .replace("｜DSML｜ invoke", "｜DSML｜invoke")
        .replace("｜DSML｜ parameter", "｜DSML｜parameter")
    )
    assert not _accepts_fully(_grammar([_WEATHER]), v4_output)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param('"get_weather"', '"get_forecast"', id="undeclared_tool"),
        pytest.param(">celsius<", ">kelvin<", id="enum_violation"),
    ],
)
def test_schema_violations_are_rejected(old: str, new: str) -> None:
    output = _calls(_WEATHER_CALL)
    assert old in output
    assert not _accepts_fully(_grammar([_WEATHER]), output.replace(old, new))


def test_root_key_needing_escapes_is_enforced() -> None:
    key = 'foo"bar'
    grammar = _grammar(
        [
            _tool(
                "emit",
                {
                    "type": "object",
                    "properties": {key: {"type": "number"}},
                    "required": [key],
                },
            )
        ]
    )
    assert _accepts_fully(
        grammar, _calls(_invoke("emit", _param(key, False, "1.5")))
    )
    assert not _accepts_fully(
        grammar, _calls(_invoke("emit", _param(key, True, "x")))
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "integer"},
        {"$ref": "#/definitions/c", "definitions": {"c": {"type": "integer"}}},
    ],
    ids=["integer", "ref_integer"],
)
def test_non_object_argument_root_does_not_compile(
    parameters: dict[str, Any],
) -> None:
    """DSML has no wire form for a non-object root, so the request must fail."""
    grammar = _grammar(
        [_tool("emit", parameters)],
        {"type": "function", "function": {"name": "emit"}},
        reject_unsupported=True,
    )
    with pytest.raises(Exception):
        _backend().create_matcher(grammar)


@pytest.mark.parametrize(
    ("additional", "extra_value", "accepted"),
    [
        pytest.param({}, "2", True, id="defaulted"),
        pytest.param({"additionalProperties": True}, "2", True, id="open"),
        pytest.param({"additionalProperties": False}, "2", False, id="closed"),
        pytest.param(
            {"additionalProperties": {"type": "integer"}}, "2", True, id="typed"
        ),
        pytest.param(
            {"additionalProperties": {"type": "integer"}},
            '"x"',
            False,
            id="typed_violation",
        ),
    ],
)
def test_root_extra_parameters_follow_the_schema(
    additional: dict[str, Any], extra_value: str, accepted: bool
) -> None:
    parameters = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        **additional,
    }
    grammar = _grammar([_tool("emit", parameters)])
    declared = _param("a", False, "1")
    assert _accepts_fully(grammar, _calls(_invoke("emit", declared)))
    extra = declared + _param("b", False, extra_value)
    assert _accepts_fully(grammar, _calls(_invoke("emit", extra))) is accepted
