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
"""Defines the `EnumLike` trait used by enum-style pattern matching."""

from std.builtin.variadics import TypeList, _MLIR


comptime KGENString = __mlir_type.`!kgen.string`
"""The MLIR string type used for enum case names."""


trait EnumLike:
    """A type with named, mutually exclusive alternatives.

    `__enum` declarations synthesize this conformance. Hand-written types
    (notably `Optional`) may also conform so pattern matching can treat their
    alternatives as cases, such as `case .Some(x):` / `case .None:`.

    By default, the declared alternatives cover every value. An open enum sets
    `_enum_is_exhaustive` to `False` when other values may exist. Payload type
    `NoneType` means the case has no associated value.
    """

    comptime _enum_case_names: _MLIR.KGENParamListType[KGENString]
    """Case names in discriminant order."""

    # TODO(MOCO-5080): `NoneType` as the marker cannot be told from a payload
    # of type `NoneType`; every payload becomes a `Tuple`, `Tuple[]` for none.
    comptime _enum_case_types: _MLIR.KGENParamListType[AnyType]
    """Payload type for each case (`NoneType` means no payload)."""

    comptime _enum_is_exhaustive: Bool = True
    """Whether the declared cases cover every value of the conforming type."""

    comptime _enum_elt_type_for_case[id: Int]: AnyType = TypeList[
        Trait=AnyType, Self._enum_case_types
    ]()[id]
    """Payload type for case `id`."""

    comptime _is_none_type[Type: AnyType] = Type == NoneType

    comptime _enum_has_no_payloads: Bool = TypeList[
        Trait=AnyType, Self._enum_case_types
    ].all[Self._is_none_type]()
    """Whether no case carries a payload."""

    def _get_enum_discriminant(self) -> Int:
        """Return the active case index.

        Different cases must return different discriminants. Two values are
        equal when their discriminants and payloads are equal, so for a case
        without a payload the discriminant alone defines the value. An open
        enum must give each value outside its declared cases a distinct
        discriminant that is not a case index.
        """
        ...

    # FIXME: Prefer an interior origin so payload refs preserve subject
    # mutability cleanly. Return type uses TypeList directly (not
    # `_enum_elt_type_for_case`) to avoid a recursive alias cycle.
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        """Return a reference to the payload of case `id`.

        Only valid when `_get_enum_discriminant()` equals `id`.
        """
        ...
