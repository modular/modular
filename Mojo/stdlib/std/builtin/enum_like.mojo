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
    """A type with a finite set of mutually exclusive alternatives.

    Syntactic `enum` declarations will synthesize this conformance. Hand-written
    types (notably `Optional`) may also conform so pattern matching can treat
    their alternatives as cases, e.g. `case .Some(x):` / `case .None:`.

    Payload type `NoneType` means the case has no associated value.
    """

    comptime _enum_case_length: Int
    """The number of cases."""

    comptime _enum_case_names: _MLIR.KGENParamListType[KGENString]
    """Case names in discriminant order."""

    comptime _enum_case_types: _MLIR.KGENParamListType[AnyType]
    """Payload type for each case (`NoneType` means no payload)."""

    comptime _enum_elt_type_for_case[id: Int]: AnyType = TypeList[
        Trait=AnyType, Self._enum_case_types
    ]()[id]
    """Payload type for case `id`."""

    def _get_enum_discriminant(self) -> Int:
        """Return the active case index."""
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
