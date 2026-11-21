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
"""Signatures of the `__enum` and `__union` support in `std.builtin._enum`."""

from std.builtin.stubs import *
from std.builtin.stubs import _MLIR


@always_inline("builtin")
def __enum_discriminant_dtype(num_cases: Int) -> DType:
    return __mlir_op.`pop.select`(
        (num_cases <= 256).__mlir_bool__(),
        DType(mlir_value=__mlir_attr.`#kgen.dtype.constant<ui8> : !kgen.dtype`),
        __mlir_op.`pop.select`(
            (num_cases <= 65536).__mlir_bool__(),
            DType(
                mlir_value=__mlir_attr.`#kgen.dtype.constant<ui16> : !kgen.dtype`
            ),
            DType(
                mlir_value=__mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`
            ),
        ),
    )


comptime __EnumTag[num_cases: Int] = Scalar[
    __enum_discriminant_dtype(num_cases)
]


struct __UnionStorage[case_types: _MLIR.KGENParamListType[AnyType]](
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
    comptime _EnumCases = TypeList[Trait=AnyType, Self._enum_case_types]

    comptime _EnumStorage = __EnumTag[Self._EnumCases.length]

    def __get_enum_storage(
        ref self,
    ) -> ref[self] Scalar[__enum_discriminant_dtype(Self._EnumCases.length)]:
        abort()

    @staticmethod
    def __enum_init[id: Int](out result: Self):
        abort()

    def _get_enum_discriminant(self) -> Int:
        abort()

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] Self._enum_elt_type_for_case[id]:
        abort()


trait __Union(EnumLike):
    def __get_enum_storage(
        ref self,
    ) -> ref[self] __UnionStorage[Self._enum_case_types]:
        abort()

    @staticmethod
    def __enum_init[id: Int, *Ts: Movable](var *payload: *Ts, out result: Self):
        abort()

    def _get_enum_discriminant(self) -> Int:
        abort()

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        abort()
