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
"""Cases for the JSON Schema admission scenario."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SERVE: Literal["serve"] = "serve"
REFUSE: Literal["refuse"] = "refuse"


DRAFT_7: Literal["7"] = "7"
DRAFT_2020_12: Literal["2020-12"] = "2020-12"


@dataclass(frozen=True)
class AdmissionCase:
    name: str
    expect: Literal["serve", "refuse"]
    schema: dict[str, object]
    prompt: str
    why: str
    # The draft served output is graded under. Cases whose meaning only a
    # later draft defines (``$ref`` siblings, ``prefixItems``) use 2020-12.
    draft: Literal["7", "2020-12"] = DRAFT_7
    # Also send the schema as the parameters of a named tool: the grammar
    # compiler refused these shapes for tool calls as well (AGENT-914).
    as_tool: bool = False


def _object(
    property_schema: dict[str, object],
    defs: dict[str, object] | None = None,
) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"v": property_schema},
        "required": ["v"],
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs
    return schema


CASES: tuple[AdmissionCase, ...] = (
    AdmissionCase(
        "utf8_enum_values",
        SERVE,
        _object({"enum": ["日本語", "Français", "Ελληνικά", "Ω≈ç√"]}),
        'Pick the language tag "Français".',
        "a multi-byte enum value must survive the lowering to EBNF",
    ),
    AdmissionCase(
        "utf8_const_value",
        SERVE,
        _object({"const": "café — naïve — 🎉"}),
        "Echo the only allowed value.",
        "a const is emitted as a literal, so its bytes reach the EBNF verbatim",
    ),
    AdmissionCase(
        "utf8_property_name",
        SERVE,
        {
            "type": "object",
            "properties": {"寿司": {"type": "string"}},
            "required": ["寿司"],
            "additionalProperties": False,
        },
        'Set 寿司 to "yes".',
        "a property name is emitted as a key literal in the same EBNF",
    ),
    AdmissionCase(
        "property_name_with_spaces",
        SERVE,
        {
            "type": "object",
            "properties": {"Autonomous SheetML showcase": {"type": "string"}},
            "required": ["Autonomous SheetML showcase"],
            "additionalProperties": False,
        },
        'Set the showcase field to "ready".',
        "a space in a key is legal JSON and the quoted form can spell it",
    ),
    AdmissionCase(
        "enum_beside_satisfiable_maxlength",
        SERVE,
        _object({"enum": ["ab", "cd"], "maxLength": 8}),
        'Choose "cd".',
        "both values are 2 characters, so maxLength 8 rules out nothing",
    ),
    AdmissionCase(
        "enum_beside_satisfiable_required",
        SERVE,
        _object({"enum": [{"a": 1, "b": 2}], "required": ["a"]}),
        "Return the only allowed object.",
        "the single enumerated object carries the required key",
    ),
    AdmissionCase(
        "enum_beside_enum_descriptions",
        SERVE,
        _object(
            {"enum": ["red", "blue"], "enumDescriptions": ["warm", "cool"]}
        ),
        'Choose "blue".',
        "enumDescriptions is an annotation and constrains no instance",
    ),
    AdmissionCase(
        "enum_beside_unsatisfiable_minlength",
        REFUSE,
        _object({"enum": ["a"], "minLength": 5}),
        "Return the allowed value.",
        "no enumerated value can be 5 characters, so nothing satisfies both",
    ),
    AdmissionCase(
        "enum_multibyte_beside_unsatisfiable_minlength",
        REFUSE,
        _object({"enum": ["é"], "minLength": 2}),
        "Return the allowed value.",
        "one codepoint, two bytes: minLength 2 still rules the value out",
    ),
    AdmissionCase(
        "unique_items_true",
        REFUSE,
        _object(
            {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
        ),
        "Return three integers.",
        "a regular grammar cannot compare an element against its predecessors",
    ),
    AdmissionCase(
        "format_and_pattern_together",
        REFUSE,
        _object({"type": "string", "format": "email", "pattern": "^a+$"}),
        "Return the allowed string.",
        "one rule cannot enforce both regex tiers without dropping one",
    ),
    # AGENT-914: valid shapes the grammar compiler refused with "invalid
    # grammar" until the converter learned to rewrite them into equivalent
    # schemas. Each is served, as response_format and as a named tool, and
    # graded under the draft that gives it its meaning.
    AdmissionCase(
        "ref_beside_same_object_sibling",
        SERVE,
        _object(
            {"$ref": "#/$defs/x", "maximum": 3},
            defs={"x": {"type": "integer"}},
        ),
        "Return a small integer.",
        "2019-09+ conjoins the sibling with the target: an integer at most 3",
        draft=DRAFT_2020_12,
        as_tool=True,
    ),
    AdmissionCase(
        "ref_beside_properties_and_required",
        SERVE,
        _object(
            {
                "$ref": "#/$defs/Base",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            defs={
                "Base": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                }
            },
        ),
        'Return v with id "a" and n 1.',
        "the object needs both the target's id and the sibling's n",
        draft=DRAFT_2020_12,
        as_tool=True,
    ),
    AdmissionCase(
        "all_of_with_two_any_ofs",
        SERVE,
        _object(
            {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                    "c": {"type": "integer"},
                },
                "allOf": [
                    {"anyOf": [{"required": ["a"]}, {"required": ["b"]}]},
                    {"anyOf": [{"required": ["b"]}, {"required": ["c"]}]},
                ],
            }
        ),
        "Return v with a set to 1 and c set to 2.",
        "the unions distribute into one: b, or a with c",
        as_tool=True,
    ),
    AdmissionCase(
        "prefix_items_past_max_items",
        SERVE,
        _object(
            {
                "type": "array",
                "prefixItems": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"type": "boolean"},
                ],
                "maxItems": 2,
            }
        ),
        'Return v as ["a", 1].',
        "no position past maxItems can exist, so the third prefix is moot",
        draft=DRAFT_2020_12,
        as_tool=True,
    ),
    AdmissionCase(
        "untyped_keywords_of_two_kinds",
        SERVE,
        _object({"minimum": 0, "pattern": "^[a-z]+$"}),
        'Set v to the word "hello".',
        "minimum binds numbers and pattern binds strings; other kinds pass",
        as_tool=True,
    ),
)
