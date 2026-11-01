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
"""Signatures of the `__enum` support in `std.builtin._enum`."""

from std.builtin.stubs import *
from std.builtin.stubs import _MLIR


struct __EnumStorage[case_types: _MLIR.KGENParamListType[AnyType]](
    Copyable where conforms_to(case_types, Copyable),
    Deinitable where conforms_to(case_types, Deinitable),
    ImplicitlyCopyable where conforms_to(case_types, ImplicitlyCopyable),
    Movable where conforms_to(case_types, Movable),
    RegisterPassable where conforms_to(case_types, RegisterPassable),
):
    var _impl: __mlir_type[
        `!kgen.variant<[rebind(:`,
        type_of(Self.case_types),
        ` `,
        Self.case_types,
        `)]>`,
    ]


trait __Enum(EnumLike):
    def __get_enum_storage(
        ref self,
    ) -> ref[self] __EnumStorage[Self._enum_case_types]:
        abort()

    @staticmethod
    def __enum_init[
        id: Int, Case: AnyType, *Ts: Movable
    ](var *payload: *Ts, out result: Self):
        abort()

    def _get_enum_discriminant(self) -> Int:
        abort()

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        abort()
