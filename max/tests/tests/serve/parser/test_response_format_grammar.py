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
"""Deterministic, model-independent grammar/mask tests for response_format.

Regression for the runaway-output incident: a
``response_format.json_schema.schema`` that omits the root ``type`` could,
without object-type inference, compile to a grammar whose START state permits a
bare, unbounded top-level value -- including a JSON string. A model that
degenerates into a repetition loop inside that string can never emit the only
terminator (the closing quote), so EOS is never unmasked and generation runs
to ``max_length`` with ``finish_reason="length"``. xgrammar instead infers an
object type for an object-shaped untyped schema and anchors it on ``{``; these
tests assert that native behavior directly.

These tests assert the *token bitmask* directly with a tiny byte+special
tokenizer (the same pattern as the gemma4 ``test_tool_parser.py`` grammar
tests), so they need no GPU and no model: the broken mask is purely a
function of the schema and the grammar backend.
"""

from __future__ import annotations

from typing import Any

from max._xgrammar import GrammarCompiler, TokenizerInfo, VocabType
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    XgrammarBackend,
)

_N_VOCAB = 256
_EOS = 0

# Raw byte vocabulary: token IDs 0-255 map to single bytes.
# ID 0 doubles as EOS. Sufficient to exercise JSON-schema grammars,
# which operate over the raw UTF-8 byte stream.
_BYTE_VOCAB: list[bytes] = [bytes([i]) for i in range(_N_VOCAB)]


def _make_backend() -> XgrammarBackend:
    tokenizer_info = TokenizerInfo(
        _BYTE_VOCAB,
        vocab_type=VocabType.RAW,
        vocab_size=_N_VOCAB,
        stop_token_ids=[_EOS],
    )
    return XgrammarBackend(GrammarCompiler(tokenizer_info))


def _allowed_tokens(backend: XgrammarBackend, grammar: Any) -> set[int]:
    """Return the set of token IDs the matcher permits as the next token."""
    matcher = backend.create_matcher(grammar)
    bitmask = backend.allocate_token_bitmask(1, _N_VOCAB)
    backend.fill_next_token_bitmask(matcher, bitmask, index=0)
    return {
        t for t in range(_N_VOCAB) if (int(bitmask[0, t // 32]) >> (t % 32)) & 1
    }


def _matcher_for(backend: XgrammarBackend, schema: dict[str, Any]) -> Any:
    return backend.create_matcher(backend.compile_json_schema(schema))


_QUOTE = ord('"')
_OPEN_BRACE = ord("{")


def test_untyped_root_with_properties_anchored_by_backend() -> None:
    """xgrammar anchors a ``properties``-bearing untyped root to an object.

    The original runaway trigger was an untyped root schema whose START state
    permitted a bare top-level string, which a looping model could extend
    forever without ever unmasking EOS. xgrammar instead infers an object from
    ``properties`` even without an explicit root ``type``, so the START state
    requires ``{`` and a bare top-level string cannot open -- the vector is
    closed at the backend level.
    """
    backend = _make_backend()
    grammar = backend.compile_json_schema({"properties": {"x": {}}})
    start_allowed = _allowed_tokens(backend, grammar)
    # Object wrapper is inferred: only ``{`` may start, never a bare ``"``.
    assert start_allowed == {_OPEN_BRACE}
    assert _QUOTE not in start_allowed

    # Opening a top-level string is not a valid object start.
    matcher = backend.create_matcher(grammar)
    assert matcher.try_consume_tokens([_QUOTE]) == 0


def test_object_shaped_schema_terminates_and_unmasks_eos() -> None:
    """A completed object reaches an accepting state and unmasks EOS.

    xgrammar anchors the object-shaped untyped schema on ``{``; a minimal valid
    object then terminates, so EOS becomes available (the request can stop).
    """
    backend = _make_backend()
    grammar = backend.compile_json_schema({"properties": {"x": {}}})

    # A minimal valid object ``{"x":1}`` must terminate and unmask EOS.
    matcher = backend.create_matcher(grammar)
    assert matcher.try_consume_tokens(list(b'{"x":1}')) == 7
    assert matcher.is_accepting()
    bitmask = backend.allocate_token_bitmask(1, _N_VOCAB)
    backend.fill_next_token_bitmask(matcher, bitmask, index=0)
    assert _EOS in {
        t for t in range(_N_VOCAB) if (int(bitmask[0, t // 32]) >> (t % 32)) & 1
    }


def test_nested_untyped_object_subschema_anchored_by_backend() -> None:
    """xgrammar anchors a nested object-shaped untyped subschema on its own.

    A nested untyped-but-object-shaped value (``inner``) is object-anchored by
    the backend's recursive type inference -- at the ``inner`` value position
    only ``{`` is allowed, not a bare ``"`` -- with no schema normalization.
    """
    backend = _make_backend()
    schema: dict[str, Any] = {
        "properties": {"inner": {"properties": {"y": {}}}}
    }
    matcher = _matcher_for(backend, schema)

    # Walk to the inner value position: ``{"inner":``.
    assert matcher.try_consume_tokens(list(b'{"inner":')) == 9
    bitmask = backend.allocate_token_bitmask(1, _N_VOCAB)
    backend.fill_next_token_bitmask(matcher, bitmask, index=0)
    inner_allowed = {
        t for t in range(_N_VOCAB) if (int(bitmask[0, t // 32]) >> (t % 32)) & 1
    }
    assert _OPEN_BRACE in inner_allowed
    assert _QUOTE not in inner_allowed, (
        "nested inner value must be object-anchored, not a bare string"
    )
