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

# RUN: %parse-mojo-isolated %s -verify-diagnostics

# A hand-written `EnumLike` conformer must give every case a `Tuple` payload,
# and payload subpatterns cannot match a `Tuple[*Ts]` payload, whose element
# count is a compile-time parameter.


struct Bare(EnumLike):
    comptime _enum_case_names = ParameterList.of["number".value].values
    comptime _enum_case_types = TypeList.of[Int].values

    def _get_enum_discriminant(self) -> Int:
        return 0

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        while True:
            pass


def match_bare(value: Bare):
    match value:
        # expected-error @+1 {{payload type for case 'number' must be a 'Tuple', found 'Int'}}
        case .number(x):
            pass
        case _:
            pass


struct Pack[*Ts: Movable](EnumLike):
    comptime _enum_case_names = ParameterList.of["one".value].values
    comptime _enum_case_types = TypeList.of[Tuple[*Self.Ts]].values

    def _get_enum_discriminant(self) -> Int:
        return 0

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        while True:
            pass


def match_pack[*Ts: Movable](pack: Pack[*Ts]):
    match pack:
        # expected-error @+1 {{cannot match the payload of case 'one': its element count is a compile-time parameter}}
        case .one(x):
            pass
        # expected-error @+1 {{cannot match the payload of case 'one': its element count is a compile-time parameter}}
        case .one(a, b):
            pass
    match pack:
        case .one:
            pass
