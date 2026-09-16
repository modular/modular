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
"""Tests for the structured-output grammar backends."""

import json
import logging
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines.lib.pipeline_variants import structured_output_backend
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN,
    _compiled_shape,
    _log_if_slow,
)
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper
from max.pipelines.modeling.types import PipelineTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

_N_VOCAB = 256

# An ordinary printable byte ("~") the runtime additionally stops on. The
# declared EOS (byte 0) is deliberately distinct: the runtime EOS set layers
# extra terminators (chat turn-end tokens and the like) on top of the
# declared EOS, which is exactly the split the grammar backend's stop set
# has to respect. Unless the backend registers the extra terminator as a
# stop token, the grammar admits it as ordinary string content.
_EXTRA_TERMINATOR_ID = 126


class _FakeTikTokenTokenizer:
    """Prod-shaped TikToken delegate over a byte-level vocab."""

    eos_token_id: int = 0
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
        return [ord(c) for c in text]


def _fast_tokenizer_delegate() -> PreTrainedTokenizerFast:
    """``PreTrainedTokenizerFast`` delegate over the same byte-level vocab."""
    vocab = {chr(i): i for i in range(256)}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token=chr(1))),
        eos_token=chr(0),
        unk_token=chr(1),
    )


def _allowed_tokens(backend: Any, matcher: Any) -> np.ndarray:
    """Bool ``[vocab]`` mask of the tokens ``matcher`` currently allows."""
    packed = backend.allocate_token_bitmask(1, _N_VOCAB)
    backend.fill_next_token_bitmask(matcher, packed, 0)
    masks = np.int32(1) << np.arange(32, dtype=np.int32)
    bits = (packed[..., np.newaxis] & masks) != 0
    return bits.reshape(*packed.shape[:-1], -1)[0, :_N_VOCAB]


@pytest.mark.parametrize(
    "delegate_factory",
    [_FakeTikTokenTokenizer, _fast_tokenizer_delegate],
    ids=["tiktoken", "hf_fast"],
)
def test_xgrammar_stop_tokens_cover_runtime_eos_set(
    delegate_factory: Callable[[], Any],
) -> None:
    """The grammar's stop set must match the runtime EOS set, not just EOS."""
    delegate = delegate_factory()
    runtime_eos = {delegate.eos_token_id, _EXTRA_TERMINATOR_ID}
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = runtime_eos

    helper = StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name="xgrammar",
    )
    assert helper.backend is not None

    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(json.dumps(schema))
    )

    # Stop inside the string value: ordinary content bytes are allowed, but
    # no terminator may be sampled, or the runtime would end the request
    # mid-structure (silent truncation).
    for char in '{"a":"x':
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert not matcher.is_accepting()
    mid = _allowed_tokens(helper.backend, matcher)
    assert mid[ord("y")]
    assert not mid[sorted(runtime_eos)].any(), (
        "a terminator is sampleable mid-structure — the runtime would stop "
        "generation inside the constrained response and truncate it"
    )

    for char in '"}':
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert matcher.is_accepting()

    done = set(np.flatnonzero(_allowed_tokens(helper.backend, matcher)))
    assert done == runtime_eos, (
        "a completed grammar must permit exactly the runtime EOS set: a "
        "missing terminator forces an unnatural declared-EOS ending, an "
        "extra token would leak unconstrained output"
    )


def test_fill_slot_after_stop_token_accepted_forces_eos_without_crashing() -> (
    None
):
    """Once the matcher actually consumes its stop token (not merely reaching
    the "ready to stop" accepting state exercised above), xgrammar's native
    fill raises a fatal C++ check rather than returning a mask. This drives a
    real matcher through that exact transition and confirms
    ``fill_next_token_bitmask`` still forces the same stop-token-only mask
    instead of crashing -- built by hand from ``stop_token_ids`` rather than
    delegated to xgrammar's own (crashing) fill.
    """
    delegate = _FakeTikTokenTokenizer()
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = {delegate.eos_token_id}

    helper = StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name="xgrammar",
    )
    assert helper.backend is not None

    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(json.dumps(schema))
    )
    for char in "{}":
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert matcher.is_accepting()
    assert not matcher.is_stopped()

    assert matcher.try_consume_tokens([delegate.eos_token_id]) == 1
    assert matcher.is_stopped()
    assert matcher.is_accepting()

    allowed = set(np.flatnonzero(_allowed_tokens(helper.backend, matcher)))
    assert allowed == {delegate.eos_token_id}, (
        "a stopped-and-accepted xgrammar matcher must force exactly its "
        f"stop token set, not {allowed} -- xgrammar's own fill crashes here, "
        "so the mask is built by hand from stop_token_ids"
    )


def test_llguidance_completed_matcher_still_gets_eos_mask() -> None:
    """llguidance's ``is_stopped()`` becomes true the moment the grammar is
    satisfied -- before the stop token is even consumed -- so a completed
    grammar is llguidance's normal, frequent end-of-request state, not a
    rare corner case. It computes a real EOS-only mask here without
    crashing, matching xgrammar's hand-built one above.
    """
    delegate = _FakeTikTokenTokenizer()
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = {delegate.eos_token_id}

    helper = StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name="llguidance",
    )
    assert helper.backend is not None

    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(json.dumps(schema))
    )
    for char in "{}":
        assert matcher.try_consume_tokens([ord(char)]) == 1
    assert matcher.is_accepting()
    assert matcher.is_stopped()

    allowed = set(np.flatnonzero(_allowed_tokens(helper.backend, matcher)))
    assert allowed == {delegate.eos_token_id}, (
        f"a completed llguidance matcher must force exactly its stop token "
        f"set, not {allowed}"
    )


# Minimal schema for the whitespace-mode tests: one required string property.
_WS_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
)


# Structural tokens a framing's grammar references by token id rather than by
# byte literal. Its vocab has to hold them or the tag will not compile, so
# ``_make_helper`` gives that framing a word-level fake carrying them; the
# byte-level fake covers every other framing. The ids follow this order.
_TOKENS_BY_MODEL_FORMAT: dict[str, tuple[str, ...]] = {
    # MiniMax-M3's envelope is built from single tokens.
    "minimax_m3": ("<tool_call>", "</tool_call>", "]<]minimax[>["),
}


def _make_helper(
    backend_name: str, model_format: str | None = None, **kwargs: Any
) -> StructuredOutputHelper:
    """A helper for ``backend_name`` whose vocab ``model_format`` can compile.

    A framing listed in :data:`_TOKENS_BY_MODEL_FORMAT` gets a word-level fake
    holding its structural tokens. Every other framing -- and every caller that
    names none -- gets the TikToken-shaped fake, which exercises both backends:
    llguidance cannot infer a decoder from the WordLevel HF fake, but both
    backends accept the byte-level adapter path.
    """
    markers = _TOKENS_BY_MODEL_FORMAT.get(model_format or "", ())
    delegate: Any
    if markers:
        vocab = {chr(i): i for i in range(_N_VOCAB)}
        vocab.update({tok: _N_VOCAB + n for n, tok in enumerate(markers)})
        delegate = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel(vocab=vocab, unk_token=chr(1))
            ),
            eos_token=chr(0),
            unk_token=chr(1),
        )
    else:
        delegate = _FakeTikTokenTokenizer()
    pipeline_tokenizer = MagicMock()
    pipeline_tokenizer.delegate = delegate
    pipeline_tokenizer.eos_token_ids = {delegate.eos_token_id}
    return StructuredOutputHelper.from_tokenizer(
        cast("PipelineTokenizer[Any, Any, Any]", pipeline_tokenizer),
        enable_structured_output=True,
        backend_name=backend_name,
        **kwargs,
    )


@pytest.mark.parametrize("backend_name", ["xgrammar", "llguidance"])
def test_any_whitespace_grammar_admits_whitespace(backend_name: str) -> None:
    """``any_whitespace=True`` compiles a grammar that accepts whitespaceful JSON."""
    helper = _make_helper(backend_name, any_whitespace=True)
    assert helper.backend is not None
    matcher = helper.backend.create_matcher(
        helper.backend.compile_json_schema(_WS_SCHEMA)
    )
    payload = '{ "a": "x" }'
    tokens = [ord(c) for c in payload]
    assert matcher.try_consume_tokens(tokens) == len(tokens), (
        f"[{backend_name}] whitespace-tolerant grammar rejected {payload!r}"
    )
    assert matcher.is_accepting()


def test_any_whitespace_grammar_bounds_whitespace_runs() -> None:
    """The whitespace-tolerant xgrammar grammar caps each whitespace run.

    Guards the runaway-generation vector that motivated the compact default:
    a model looping on whitespace must be forced to converge instead of
    emitting whitespace forever (GLM 5.2 produced exactly that runaway
    inside tool calls). xgrammar-only: llguidance has no whitespace-run cap.
    """
    backend_name = "xgrammar"
    helper = _make_helper(backend_name, any_whitespace=True)
    assert helper.backend is not None
    grammar = helper.backend.compile_json_schema(_WS_SCHEMA)

    max_run = " " * STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN
    within = helper.backend.create_matcher(grammar)
    payload = "{" + max_run + '"a":"x"}'
    tokens = [ord(c) for c in payload]
    assert within.try_consume_tokens(tokens) == len(tokens), (
        f"[{backend_name}] bounded grammar rejected a "
        f"{STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN}-char whitespace run"
    )
    assert within.is_accepting()

    beyond = helper.backend.create_matcher(grammar)
    payload = "{" + max_run + ' "a":"x"}'
    consumed = beyond.try_consume_tokens([ord(c) for c in payload])
    assert consumed == 1 + STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN, (
        f"[{backend_name}] grammar consumed {consumed} tokens of "
        f"{payload!r}; expected rejection at whitespace char "
        f"{STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN + 1}"
    )


@pytest.mark.parametrize("backend_name", ["xgrammar", "llguidance"])
def test_default_grammar_stays_compact(backend_name: str) -> None:
    """The default (``any_whitespace`` unset) keeps the compact-JSON grammar.

    Guards the Gemma-4 runaway mitigation (0c57a6bd331): flipping the global
    default is a product decision, so an unset knob must reproduce today's
    whitespace-free grammar exactly.
    """
    helper = _make_helper(backend_name)
    assert helper.backend is not None
    grammar = helper.backend.compile_json_schema(_WS_SCHEMA)

    compact = helper.backend.create_matcher(grammar)
    tokens = [ord(c) for c in '{"a":"x"}']
    assert compact.try_consume_tokens(tokens) == len(tokens)
    assert compact.is_accepting()

    spaced = helper.backend.create_matcher(grammar)
    payload = '{"a": "x"}'
    consumed = spaced.try_consume_tokens([ord(c) for c in payload])
    assert consumed == payload.index(" "), (
        f"[{backend_name}] compact grammar consumed {consumed} tokens of "
        f"{payload!r}; expected rejection at the whitespace"
    )


# Every tool-call framing a model can be served under. The XML ones share one
# converter, so a defect in it reaches all of them; "kimi" frames arguments as
# plain JSON and reaches a different one, which makes it the control that tells
# an XML-framing bug apart from a JSON-schema-to-grammar bug.
_XML_TOOL_FORMATS = ("glm_4_7", "minimax", "minimax_m3")
# deepseek_v3_2 / deepseek_v4 also select an XML style, but no MAX tool parser
# declares either as its XGRAMMAR_FORMAT -- the DeepSeek parsers read a
# JSON-argument envelope instead -- so nothing routes to them today.


def _tool_matcher(
    helper: StructuredOutputHelper,
    model_format: str,
    name: str,
    schema: dict[str, Any],
) -> Any:
    """Compile ``schema`` as ``name``'s arguments in ``model_format``'s grammar."""
    assert helper.backend is not None
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "A tool.",
                "parameters": schema,
            },
        }
    ]
    grammar = structured_output_backend.build_xgrammar_tool_grammar(
        model_format, tools, "required"
    )
    return helper.backend.create_matcher(grammar)


def _xml_tool_wire(
    model_format: str, name: str, pairs: list[tuple[str, str]]
) -> str:
    """The bytes a model emits to call ``name`` with ``pairs`` under ``model_format``."""
    if model_format == "glm_4_7":
        body = "".join(
            f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"
            for k, v in pairs
        )
        return f"<tool_call>{name}{body}</tool_call>"
    if model_format == "minimax":
        body = "".join(
            f'<parameter name="{k}">{v}</parameter>' for k, v in pairs
        )
        return (
            "\n</think>\n\n\n\n\n<minimax:tool_call>\n"
            f'<invoke name="{name}">\n{body}</invoke>\n</minimax:tool_call>'
        )
    p = "]<]minimax[>["
    body = "".join(f"{p}<{k}>{v}{p}</{k}>" for k, v in pairs)
    return (
        f'{p}<tool_call>\n{p}<invoke name="{name}">{body}'
        f"{p}</invoke>\n{p}</tool_call>"
    )


def _consume_wire(matcher: Any, model_format: str, wire: str) -> int:
    """Bytes of ``wire`` the matcher accepts, feeding markers as tokens."""
    i = consumed = 0
    markers = _TOKENS_BY_MODEL_FORMAT.get(model_format, ())
    while i < len(wire):
        token_id, width = ord(wire[i]), 1
        for n, marker in enumerate(markers):
            if wire.startswith(marker, i):
                token_id, width = _N_VOCAB + n, len(marker)
                break
        if matcher.try_consume_tokens([token_id]) != 1:
            return consumed
        i += width
        consumed += width
    return consumed


_CONTAINER_ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "preset": {
            "enum": [
                {"mode": "fast", "threads": 4},
                {"mode": "safe", "threads": 1},
            ]
        },
    },
    "required": ["preset"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "model_format",
    ["kimi", "glm_4_7", "minimax", "deepseek_v3_2", "deepseek_v4"],
    ids=lambda f: f,
)
def test_container_enum_compiles_for_every_tool_format(
    model_format: str,
) -> None:
    """An object-valued ``enum`` must compile, whatever the tool-call framing.

    An XML-framed tool call emits a top-level string value bare, so the
    converter used to paste every literal's raw JSON text into an EBNF string
    literal. For a container that text carries its own quotes, which the EBNF
    parser reads as syntax: ``{"mode":...}`` closed the literal after ``{`` and
    left ``mode`` looking like a rule reference, so the compile died with
    ``Rule "mode" is not defined`` and the request 400'd.

    Every XML style shares the one converter, so every one of them was
    affected; ``kimi`` is the JSON-framed control that never was.
    """
    _tool_matcher(
        _make_helper("xgrammar", model_format),
        model_format,
        "apply_preset",
        _CONTAINER_ENUM_SCHEMA,
    )


def test_container_enum_is_refused_where_it_has_no_wire_form() -> None:
    """A tag-keyed style spells an object as nested tags at every depth.

    There is no JSON form to fall back on there, so the literal is refused with
    an explicit message rather than silently constrained to a spelling the
    model's own format never produces.
    """
    with pytest.raises(RuntimeError, match="no wire form"):
        _tool_matcher(
            _make_helper("xgrammar", "minimax_m3"),
            "minimax_m3",
            "apply_preset",
            _CONTAINER_ENUM_SCHEMA,
        )


@pytest.mark.parametrize(
    "model_format", ["glm_4_7", "minimax"], ids=lambda f: f
)
def test_container_enum_constrains_to_the_declared_literals(
    model_format: str,
) -> None:
    """Compiling is not enough: the literal must also be enforced exactly.

    A container literal is re-serialized as compact JSON inside the XML value
    markers, matching how a nested object-typed property is already emitted.
    """
    helper = _make_helper("xgrammar", model_format)

    declared = _xml_tool_wire(
        model_format,
        "apply_preset",
        [("preset", '{"mode":"fast","threads":4}')],
    )
    matcher = _tool_matcher(
        helper, model_format, "apply_preset", _CONTAINER_ENUM_SCHEMA
    )
    assert _consume_wire(matcher, model_format, declared) == len(declared), (
        f"[{model_format}] the grammar rejected a declared enum literal"
    )
    assert matcher.is_accepting()

    # Same shape, undeclared value: the enum must still pin the literal rather
    # than degrade into "any object".
    undeclared = _xml_tool_wire(
        model_format,
        "apply_preset",
        [("preset", '{"mode":"slow","threads":9}')],
    )
    matcher = _tool_matcher(
        helper, model_format, "apply_preset", _CONTAINER_ENUM_SCHEMA
    )
    assert _consume_wire(matcher, model_format, undeclared) < len(undeclared), (
        f"[{model_format}] the grammar admitted an object matching no literal"
    )


# A JSON Schema integer is a number with a zero fractional part, so the
# exponent forms below are integers and the fractional ones are not. The
# terminal is shared by every framing, XML or not, so the schema is exercised
# through a bare response_format schema, a JSON-framed tool call, and the XML
# ones.
_INTEGER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "quantity": {"type": "integer"},
    },
    "required": ["name", "quantity"],
    "additionalProperties": False,
}

_INTEGER_FRAMINGS = ("response_format", "kimi", *_XML_TOOL_FORMATS)


def _integer_wire(framing: str, literal: str) -> str:
    """The on-wire text a model emits for ``quantity = literal``."""
    args = f'{{"name":"x","quantity":{literal}}}'
    if framing == "response_format":
        return args
    if framing == "kimi":
        return (
            "<|tool_calls_section_begin|><|tool_call_begin|>"
            "functions.record_quantity:0<|tool_call_argument_begin|>"
            f"{args}<|tool_call_end|><|tool_calls_section_end|>"
        )
    return _xml_tool_wire(
        framing, "record_quantity", [("name", "x"), ("quantity", literal)]
    )


def _integer_matcher(helper: StructuredOutputHelper, framing: str) -> Any:
    assert helper.backend is not None
    if framing == "response_format":
        return helper.backend.create_matcher(
            helper.backend.compile_json_schema(json.dumps(_INTEGER_SCHEMA))
        )
    return _tool_matcher(helper, framing, "record_quantity", _INTEGER_SCHEMA)


@pytest.mark.parametrize("framing", _INTEGER_FRAMINGS, ids=lambda f: f)
@pytest.mark.parametrize(
    ("literal", "is_integer"),
    [
        ("180", True),
        ("100000000000000000000", True),
        ("1e80", True),
        ("1E80", True),
        ("1e+80", True),
        ("-1e80", True),
        # Legal JSON numbers that are not integers, or whose integer-ness a
        # terminal cannot decide. Each must stay rejected: the conformance
        # checker validates the decoded value under JSON Schema, so a grammar
        # that admitted these would hand the model a value its own schema
        # forbids.
        ("1e-5", False),
        ("1.5", False),
        ("1.0", False),
        ("1.5e3", False),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_integer_terminal_admits_an_exponent_but_no_fraction(
    framing: str, literal: str, is_integer: bool
) -> None:
    """An ``integer`` field must reach ``1e80``, and must not reach ``1e-5``.

    The terminal had no exponent suffix at all, so a model asked for a very
    large integer had the ``e`` masked out mid-number and was steered into the
    nearest legal continuation -- ``1e80`` became ``180``, silently and with no
    schema rejection anywhere to show for it.

    The exponent's sign is ``+`` or absent by construction: ``1e-5`` is a legal
    JSON number but not an integer. ``1.0`` and ``1.5e3`` are integers by value
    yet stay rejected, because deciding that needs the fraction length weighed
    against the exponent, which a regular terminal cannot do.

    This terminal is not XML-specific -- it is the one every framing shares --
    so ``response_format`` carried the same hole as tool calls.
    """
    helper = _make_helper("xgrammar", framing)
    wire = _integer_wire(framing, literal)
    matcher = _integer_matcher(helper, framing)
    consumed = _consume_wire(matcher, framing, wire)
    if is_integer:
        assert consumed == len(wire) and matcher.is_accepting(), (
            f"[{framing}] the grammar rejected the integer {literal!r} after "
            f"{consumed} of {len(wire)} bytes"
        )
    else:
        assert consumed < len(wire), (
            f"[{framing}] the grammar admitted {literal!r} for an integer field"
        )


class _SlowBackend:
    """Minimal stand-in carrying the ``name`` the decorator reads."""

    name = "fake"

    @_log_if_slow
    def compile_json_schema(self, schema: str) -> str:
        return schema


def test_slow_grammar_compile_log_carries_structured_fields(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The duration must be a queryable attribute, not just message text."""
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", -1.0
    )
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _SlowBackend().compile_json_schema("{}")

    record = next(
        r for r in caplog.records if r.msg.startswith("grammar %s took")
    )
    fields = vars(record)
    assert fields["event"] == "grammar_compile_slow"
    assert fields["grammar_compile_method"] == "compile_json_schema"
    assert fields["grammar_backend"] == "fake"
    assert fields["grammar_compile_time_ms"] > 0.0
    # ``name`` stays the logger name; the backend goes in its own key.
    assert record.name == "max.pipelines"


def test_fast_grammar_compile_logs_nothing(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", 1e9
    )
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _SlowBackend().compile_json_schema("{}")
    assert not [r for r in caplog.records if r.msg.startswith("grammar %s")]


class _ShapeBackend:
    """Stand-in whose compile entry point takes any of the wire forms."""

    name = "fake"

    @_log_if_slow
    def create_matcher(self, grammar: Any) -> Any:
        return grammar


def _slow_log(
    arg: Any, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Force the slow path for ``arg`` and return the emitted record's fields."""
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", -1.0
    )
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _ShapeBackend().create_matcher(arg)
    return vars(
        next(r for r in caplog.records if r.msg.startswith("grammar %s took"))
    )


def test_slow_compile_log_carries_schema_shape(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth and node count are attached to the slow-compile record."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "object", "properties": {"b": {"type": "string"}}}
        },
    }
    fields = _slow_log(json.dumps(schema), caplog, monkeypatch)
    # root -> a -> b
    assert fields["grammar_schema_depth"] == 3
    assert fields["grammar_schema_nodes"] == 3


def test_slow_compile_log_omits_shape_for_non_schema_arguments(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiled handle has no shape; the duration is still logged."""
    fields = _slow_log(object(), caplog, monkeypatch)
    assert fields["grammar_compile_time_ms"] > 0.0
    assert "grammar_schema_depth" not in fields
    assert "grammar_schema_nodes" not in fields


def _tool(name: str) -> dict[str, Any]:
    """An OpenAI tool whose arguments schema is 5 subschemas, 4 deep."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
                "required": ["query"],
            },
        },
    }


def test_compiled_shape_aggregates_schemas_inside_a_tool_grammar() -> None:
    """A tool grammar's shape is its embedded schemas, not its envelope.

    Depth is the deepest of them; node count is their sum.
    """
    one_tool = _compiled_shape(
        structured_output_backend.build_xgrammar_tool_grammar(
            "kimi", [_tool("a")], "auto"
        )
    )
    assert one_tool is not None
    depth, nodes = one_tool
    assert depth == 4, "root -> filters -> tags -> items"
    assert nodes == 5

    five_tools = _compiled_shape(
        structured_output_backend.build_xgrammar_tool_grammar(
            "kimi", [_tool(f"t{i}") for i in range(5)], "auto"
        )
    )
    assert five_tools is not None
    assert five_tools == (depth, nodes * 5), (
        "depth is the deepest tool's, node count the sum across tools"
    )


def test_slow_compile_log_carries_tool_grammar_shape(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The logged fields carry a tool grammar's aggregated shape."""
    grammar = structured_output_backend.build_xgrammar_tool_grammar(
        "kimi", [_tool(f"t{i}") for i in range(5)], "auto"
    )
    fields = _slow_log(grammar, caplog, monkeypatch)
    assert fields["grammar_schema_depth"] == 4
    assert fields["grammar_schema_nodes"] == 25, (
        "expected 5 tool schemas of 5 subschemas each"
    )


def test_slow_compile_log_carries_shape_for_a_keyword_call(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The payload is found whether it arrives positionally or by keyword."""
    monkeypatch.setattr(
        structured_output_backend, "_GRAMMAR_COMPILE_LOG_MS", -1.0
    )
    schema = _tool("a")["function"]["parameters"]
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _ShapeBackend().create_matcher(grammar=json.dumps(schema))
    fields = vars(
        next(r for r in caplog.records if r.msg.startswith("grammar %s took"))
    )
    assert fields["grammar_schema_depth"] == 4
    assert fields["grammar_schema_nodes"] == 5


def test_compiled_shape_reads_a_bare_schema_directly() -> None:
    """The response_format path hands over a schema, not a tag."""
    schema = _tool("a")["function"]["parameters"]
    assert _compiled_shape(json.dumps(schema)) == (4, 5)


def test_compiled_shape_ignores_a_lark_grammar() -> None:
    """A Lark grammar is not JSON, so no shape is reported."""
    assert _compiled_shape("start: /[a-z]+/\n") is None


def test_slow_compile_log_survives_malformed_json(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unparseable input omits the shape fields rather than raising."""
    fields = _slow_log("{not json", caplog, monkeypatch)
    assert fields["grammar_compile_time_ms"] > 0.0
    assert "grammar_schema_depth" not in fields
