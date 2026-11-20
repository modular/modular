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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# A failing `comptime assert` reached through a generic alias in a comptime
# ternary. Elaboration of the rest of the scope keeps running after the
# constraint fails and hands the node a second error; the first one is what
# explains the failure and has to survive to be reported.

from std.builtin.enum_like import EnumLike
from std.builtin.variadics import ParameterList, TypeList


struct Name(EnumLike, TrivialRegisterPassable):
    comptime _enum_case_names = ParameterList.of["A".value].values
    comptime _enum_case_types = TypeList.of[NoneType].values
    var tag: Int

    def __init__(out self, tag: Int):
        self.tag = tag

    @staticmethod
    def __getattr_param__[name: StaticString]() -> Self:
        # CHECK: constraint failed: unknown name: 'BAD'
        # A non-production build prints the stored error before aborting, so
        # the diagnostic alone does not prove the abort is gone.
        # CHECK-NOT: UNREACHABLE
        comptime assert name == "A", "unknown name: '" + name + "'"
        return Self(0)

    def _get_enum_discriminant(self) -> Int:
        return self.tag

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        comptime assert False, "no payload"


comptime IdOf[N: EnumLike, //, n: N]: Int = n._get_enum_discriminant()


def main():
    comptime x = 1 if IdOf[Name.BAD] >= 0 else 2
    print("r=", x)
