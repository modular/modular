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

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# A conformer without `_enum_case_names` reports the missing member once,
# before any case is looked at, however many named cases the match has. The
# failed reading must not crash the compiler or claim a parametric enum.


struct MissingCaseNames(EnumLike):
    comptime _enum_case_types = TypeList.of[Tuple[]].values

    def _get_enum_discriminant(self) -> Int:
        return 0

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        while True:
            pass


def match_missing_case_names(value: MissingCaseNames):
    match value:
    case .first:
        pass
    case .second:
        pass
    case .third:
        pass
    case _:
        pass


# CHECK: error: 'MissingCaseNames' value has no attribute '_enum_case_names'
# CHECK: error: 'MissingCaseNames' does not implement all requirements for 'EnumLike'
# CHECK-NOT: error:
# CHECK-NOT: cannot match on a parametric enum type
# CHECK-NOT: Assertion failed
