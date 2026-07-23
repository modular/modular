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


def _accepts(compiled: xgr.CompiledGrammar, s: str) -> bool:
    # Whether the grammar accepts s as a complete value.
    matcher = xgr.GrammarMatcher(compiled)
    return matcher.accept_string(s) and matcher.is_completed()


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


def test_local_ref_compiles() -> None:
    # A local JSON-pointer $ref ("#/...") is resolvable, so unlike a non-local or
    # plain-name ref it must NOT be rejected -- it compiles and enforces the
    # referenced schema.
    compiled = _compiler().compile_json_schema(
        '{"$ref": "#/$defs/S", "$defs": {"S": {"type": "string"}}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)
    assert _accepts(compiled, '"a"')


def test_dynamic_ref_rejected() -> None:
    # $dynamicRef is a reference (it resolves to a subschema the instance must
    # satisfy), not a no-op annotation; with no other type keyword it would fall
    # back to unconstrained, so it is rejected.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"$dynamicRef": "#node"}', reject_unsupported=True
        )


def test_unknown_keyword_without_type_rejected() -> None:
    # A type-less schema whose only keyword is unrecognized would fall back to
    # unconstrained decoding; reject rather than silently emit an any-value
    # grammar.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"x-custom": "value"}', reject_unsupported=True
        )


def test_annotation_only_without_type_compiles() -> None:
    # A type-less schema carrying only annotation keywords is genuinely
    # unconstrained; it compiles to the any-value grammar rather than being
    # rejected.
    compiled = _compiler().compile_json_schema('{"title": "foo"}')
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_multiple_of_rejected() -> None:
    # multipleOf has no faithful CFG encoding; reject rather than emit an
    # unconstrained number.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "number", "multipleOf": 5}', reject_unsupported=True
        )


def test_integer_multiple_of_rejected() -> None:
    # multipleOf is also rejected on the integer path (ParseInteger), distinct
    # from the number path above.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "integer", "multipleOf": 3}', reject_unsupported=True
        )


def test_unenforceable_top_level_keywords_rejected() -> None:
    # Each of these is categorically unenforceable and rejected on presence by
    # the top-level keyword check.
    for schema in (
        '{"not": {"type": "string"}}',
        '{"dependentRequired": {"a": ["b"]}}',
        '{"dependentSchemas": {"a": {"type": "string"}}}',
        '{"dependencies": {"a": ["b"]}}',
    ):
        with pytest.raises(Exception):
            _compiler().compile_json_schema(schema, reject_unsupported=True)


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


def test_typeless_single_family_keywords_compile() -> None:
    # With no explicit "type", validation keywords that all imply ONE type select
    # that arm of the type-inference dispatch and compile.
    for schema in (
        '{"properties": {"a": {"type": "string"}}}',  # object
        '{"minItems": 1}',  # array
        '{"minLength": 1}',  # string
        '{"minimum": 0}',  # number
    ):
        compiled = _compiler().compile_json_schema(schema)
        assert isinstance(compiled, xgr.CompiledGrammar)


def test_array_unique_items_rejected() -> None:
    # uniqueItems (like contains/minContains/maxContains) cannot be enforced by a
    # CFG; reject on the array path.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "array", "uniqueItems": true}', reject_unsupported=True
        )


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


def test_oneof_two_branches_rejected() -> None:
    # oneOf is only approximated as anyOf; reject under strict mode.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"oneOf": [{"type": "string"}, {"type": "number"}]}',
            reject_unsupported=True,
        )


def test_allof_two_nontrivial_branches_rejected() -> None:
    # Two enforceable allOf members cannot be merged into one grammar.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"allOf": [{"type": "string", "minLength": 1}, '
            '{"type": "string", "maxLength": 10}]}',
            reject_unsupported=True,
        )


def test_allof_with_only_empty_branch_compiles() -> None:
    # A single empty-object member is a no-op (0 enforceable branches):
    # no constraint to apply, so it compiles.
    compiled = _compiler().compile_json_schema('{"allOf": [{}]}')
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_with_true_member_compiles() -> None:
    # A boolean-true member accepts everything -- a no-op, dropped like {}; the
    # 0-enforceable allOf compiles.
    compiled = _compiler().compile_json_schema('{"allOf": [true]}')
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_allof_with_false_member_rejected() -> None:
    # A boolean-false member matches nothing (not a no-op), so the allOf is
    # unsatisfiable and is rejected.
    with pytest.raises(Exception):
        _compiler().compile_json_schema('{"allOf": [false]}')


def test_allof_with_additional_properties_rejected() -> None:
    # Any additionalProperties sibling of an enforceable allOf is dropped by the
    # exclusive dispatch and cannot be merged; reject it regardless of value
    # (bool or schema) rather than emit a grammar looser than the schema.
    for value in ("false", "true", '{"type": "string"}'):
        with pytest.raises(Exception):
            _compiler().compile_json_schema(
                '{"type": "object", "allOf": [{"properties": {"a": {"type": '
                '"string"}}}], "additionalProperties": ' + value + "}",
                reject_unsupported=True,
            )


def test_allof_with_sibling_object_keyword_rejected() -> None:
    # A non-additionalProperties object applicator (here required) sibling of an
    # enforceable allOf is silently dropped by dispatch; reject the combination.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "object", "allOf": [{"properties": {"a": {"type": '
            '"string"}}}], "required": ["a"]}',
            reject_unsupported=True,
        )


def test_if_then_rejected() -> None:
    # An ACTIVE if/then conditional cannot be enforced; reject it.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "string", "if": {"minLength": 5}, '
            '"then": {"pattern": "^hello"}}',
            reject_unsupported=True,
        )


def test_bare_if_no_then_compiles() -> None:
    # A lone if with no then/else is a draft-7 no-op and compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "if": {"minLength": 5}}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_anyof_compiles() -> None:
    # anyOf IS representable as a grammar union (GenerateAnyOf emits
    # alternatives), so unlike oneOf a multi-branch anyOf compiles. It is the
    # base-vs-branch key *conflict* that is rejected, not anyOf itself.
    compiled = _compiler().compile_json_schema(
        '{"anyOf": [{"type": "string"}, {"type": "number"}]}'
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


def test_anyof_base_branch_conflict_rejected() -> None:
    # A key set by BOTH the base schema and a branch is an unmergeable conflict;
    # reject it rather than silently letting the branch value win.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"minLength": 2, "anyOf": [{"minLength": 5}, {"type": "string"}]}',
            reject_unsupported=True,
        )


def test_anyof_boolean_branch_with_base_rejected() -> None:
    # A boolean anyOf branch is representable (true collapses the anyOf to the
    # base schema, false drops out), but the base-merge only folds base keys
    # into an object branch, so a boolean branch alongside a base schema is
    # rejected conservatively.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"minLength": 2, "anyOf": [{"type": "string"}, true]}',
            reject_unsupported=True,
        )


def test_anyof_base_merge_compiles() -> None:
    # The happy path of the base-merge: base keywords (type, required) are folded
    # into each object branch without conflict, so base AND (B1 OR B2) compiles.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "required": ["x"], "anyOf": ['
        '{"properties": {"x": {"type": "string"}}}, '
        '{"properties": {"x": {"type": "number"}}}]}'
    )
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_anyof_base_merge_accepts_union() -> None:
    # Semantic check (not just "compiles"): the merged grammar must accept the
    # UNION of both branches -- both "a" and "b" -- and nothing outside it.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "anyOf": [{"const": "a"}, {"const": "b"}]}'
    )
    assert _accepts(compiled, '"a"')  # branch 1
    assert _accepts(
        compiled, '"b"'
    )  # branch 2 -> proves the union, not one arm
    assert not _accepts(compiled, '"c"')  # outside the union is rejected


def test_anyof_base_merge_enforces_base() -> None:
    # The folded base constraint must actually bind: with an empty branch the
    # union is "any value", so only base type:string keeps a number out. If the
    # base were dropped, '1' would be accepted.
    compiled = _compiler().compile_json_schema(
        '{"type": "string", "anyOf": [{"const": "a"}, {}]}'
    )
    assert _accepts(compiled, '"a"')
    assert _accepts(
        compiled, '"b"'
    )  # {} branch (as string) accepts other strings
    assert not _accepts(compiled, "1")  # base type:string bars a bare number


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


def test_anyof_with_false_branch_compiles() -> None:
    # A false branch accepts nothing; it should be dropped from the union, not
    # abort the whole schema. The remaining branch is satisfiable.
    compiled = _compiler().compile_json_schema(
        '{"anyOf": [{"type": "string"}, false]}'
    )
    assert compiled is not None


def test_anyof_all_false_rejected() -> None:
    # If every branch is false, nothing is satisfiable -> reject.
    with pytest.raises(Exception):
        _compiler().compile_json_schema('{"anyOf": [false, false]}')


def test_object_with_false_property_compiles() -> None:
    # A property mapped to false forbids that key; skip it (never emit) rather
    # than aborting the object. Other declared properties still compile.
    compiled = _compiler().compile_json_schema(
        '{"type": "object", "properties": {"foo": {"type": "string"}, "bar": false}}'
    )
    assert compiled is not None


def test_required_false_property_rejected() -> None:
    # A required property forbidden by a false schema is genuinely unsatisfiable.
    with pytest.raises(Exception):
        _compiler().compile_json_schema(
            '{"type": "object", "properties": {"bar": false}, "required": ["bar"]}'
        )
