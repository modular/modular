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

"""Parity/integration test for the ``max._xgrammar`` shim.

Validates that the shim restores the upstream xgrammar Python-layer surface MAX's
structured-output backend depends on -- over the nanobind binding --
including the structural-tag bridge (vendored pydantic models ->
``model_dump_json`` -> the C++ structural-tag compiler) used by the tool-call
path.
"""

import importlib.util
import sys

import numpy as np
import pytest
from max import _xgrammar as xgr
from max._xgrammar.structural_tag import (
    ConstStringFormat,
    JSONSchemaFormat,
    OrFormat,
)

_VOCAB = [
    "{",
    "}",
    "[",
    "]",
    '"',
    ":",
    ",",
    " ",
    "a",
    "b",
    "1",
    "true",
    "null",
    "<eos>",
]


def _compiler() -> xgr.GrammarCompiler:
    tokenizer_info = xgr.TokenizerInfo(
        _VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[len(_VOCAB) - 1],
    )
    return xgr.GrammarCompiler(tokenizer_info)


def test_shim_is_torch_free() -> None:
    assert "torch" not in sys.modules
    assert importlib.util.find_spec("torch") is None


def test_shim_surface_present() -> None:
    assert hasattr(xgr.TokenizerInfo, "from_huggingface")
    assert callable(xgr.get_builtin_structural_tag)
    for name in (
        "TokenizerInfo",
        "VocabType",
        "GrammarCompiler",
        "CompiledGrammar",
        "GrammarMatcher",
        "StructuralTag",
        "StructuralTagItem",
        "allocate_token_bitmask",
    ):
        assert hasattr(xgr, name)


def test_allocate_token_bitmask() -> None:
    bitmask = xgr.allocate_token_bitmask(2, 100)
    assert bitmask.shape == (2, (100 + 31) // 32)
    assert bitmask.dtype == np.int32
    assert (bitmask == -1).all()


def test_json_schema_path_through_shim() -> None:
    compiled = _compiler().compile_json_schema('{"type": "object"}')
    assert isinstance(compiled, xgr.CompiledGrammar)
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = np.full((xgr.get_bitmask_size(len(_VOCAB)),), -1, dtype=np.int32)
    assert matcher.fill_next_token_bitmask(bitmask) is True
    assert int(bitmask[0]) != -1


# Rejection of unenforceable keywords is opt-in: it happens only when the caller
# passes reject_unsupported=True. The default (exercised by the guard tests below)
# falls back to best-effort decoding instead.


def test_non_local_ref_rejected() -> None:
    # A non-local $ref (external URI) cannot be resolved into a grammar; it
    # would silently under-constrain, so reject_unsupported rejects it.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"$ref": "http://example.com"}', reject_unsupported=True
        )


def test_plain_name_anchor_rejected() -> None:
    # "#foo" is a plain-name anchor, not a JSON pointer; ResolveRef returns an
    # unconstrained AnySpec, so it is rejected rather than under-constrain.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"$ref": "#foo"}', reject_unsupported=True
        )


def test_multiple_of_rejected() -> None:
    # multipleOf has no faithful CFG encoding; reject rather than emit an
    # unconstrained number.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "number", "multipleOf": 5}', reject_unsupported=True
        )


def test_ambiguous_type_keywords_rejected() -> None:
    # minLength implies string, minItems implies array: JSON Schema keywords are
    # type-conditional and do not union, so an ambiguous keyword mix is rejected
    # rather than picking an arbitrary type.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"minLength": 5, "minItems": 2}', reject_unsupported=True
        )


def test_conflicting_type_keywords_rejected() -> None:
    # pattern implies string, minimum implies number: another ambiguous mix.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"pattern": "[a-z]+", "minimum": 0}', reject_unsupported=True
        )


def test_annotation_only_keyword_accepted() -> None:
    # An unknown annotation keyword (x-custom) alongside a concrete type does
    # not constrain the instance and must not block compilation.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "x-custom": "value"}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_array_without_items_compiles() -> None:
    # A single-type-family schema (only array keywords) is unambiguous and
    # compiles even without an items schema.
    compiled = _compiler().compile_json_schema(
        '{"type": "array", "minItems": 2}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_tool_call_with_multiple_of_in_arg_schema_rejected() -> None:
    # The tool-call path (StructuralTag -> compile_structural_tag) carries
    # reject_unsupported through the tag JSON via JSONSchemaFormat, so an
    # unenforceable keyword in a tool-argument schema is rejected there too.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"n": {"type": "number", "multipleOf": 5}},
            },
            reject_unsupported=True,
        )
    )
    with pytest.raises(Exception):
        _compiler().compile_structural_tag(tag)


def test_unsupported_keyword_compiles_by_default() -> None:
    # Without reject_unsupported, an unenforceable keyword (multipleOf) falls
    # back to best-effort decoding rather than raising -- the default is
    # permissive on the plain compile_json_schema path.
    compiled = _compiler().compile_json_schema(
        '{"type": "number", "multipleOf": 5}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_unsupported_keyword_in_tag_compiles_by_default() -> None:
    # Same permissive default on the structural-tag path: JSONSchemaFormat
    # defaults reject_unsupported to False.
    tag = xgr.StructuralTag(
        format=JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"n": {"type": "number", "multipleOf": 5}},
            }
        )
    )
    compiled = _compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_structural_tag_bridge() -> None:
    tag = xgr.StructuralTag.from_legacy_structural_tag(
        [xgr.StructuralTagItem(begin="a", schema={"type": "object"}, end="b")],
        triggers=["a"],
    )
    roundtripped = xgr.StructuralTag.model_validate_json(tag.model_dump_json())
    assert isinstance(roundtripped, xgr.StructuralTag)

    compiled = _compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_or_format_compiles() -> None:
    """``OrFormat`` must be usable end to end.

    ``OrFormat`` has a forward-referenced ``List["Format"]`` field, so it must
    appear in ``structural_tag``'s ``model_rebuild()`` block; otherwise
    constructing it raises ``PydanticUserError``. Guards the generic shim layer
    (construct, serialize, compile) independently of any model wiring.
    """
    tag = xgr.StructuralTag(
        format=OrFormat(
            elements=[
                JSONSchemaFormat(json_schema={"type": "object"}),
                ConstStringFormat(value="null"),
            ]
        )
    )
    roundtripped = xgr.StructuralTag.model_validate_json(tag.model_dump_json())
    assert roundtripped.format.type == "or"

    compiled = _compiler().compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_unique_items_false_compiles() -> None:
    # uniqueItems: false is a no-op (duplicates allowed = the default), so it
    # must compile even under reject_unsupported; only uniqueItems: true is
    # rejected -- xgrammar's grammar model has no uniqueness tracking, and
    # all-distinct isn't context-free for unbounded-domain arrays.
    compiled = _compiler().compile_json_schema(
        '{"type": "array", "items": {"type": "integer"}, "uniqueItems": false}',
        reject_unsupported=True,
    )
    assert compiled is not None


def test_unique_items_true_rejected() -> None:
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "array", "items": {"type": "integer"}, "uniqueItems": true}',
            reject_unsupported=True,
        )


def test_object_without_properties_compiles() -> None:
    # A bounds-only object {"minProperties": 1} (no declared properties) must
    # compile to an OPEN object (>=1 member of any type), mirroring
    # test_array_without_items_compiles -- not collapse to a closed empty object
    # and be rejected as unsatisfiable under strict mode.
    compiled = _compiler().compile_json_schema('{"minProperties": 1}')
    assert compiled is not None
