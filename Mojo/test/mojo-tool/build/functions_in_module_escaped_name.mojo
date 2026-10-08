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

# The quoted string parameter puts `"` in the tensor helper's name, and
# the mangler encodes each one as `~Q` to keep the symbol ELF-safe, so the
# instance name no longer matches the generator's own symbol. Reflection must
# find the generator anyway.

# RUN: %mojo-build %s -o %t
# RUN: %t | FileCheck %s

# CHECK: takes_tile_tensor{{.*}}~Qquoted~Q
# CHECK: PASS {{.*}} test_one

from std.reflection import reflect_fn
from std.testing import TestSuite
from layout import TileTensor
from layout.tile_layout import row_major

comptime layout = row_major[128, 1]()


@fieldwise_init
struct EscapedName[value: StaticString](TrivialRegisterPassable):
    pass


def takes_tile_tensor(
    x: TileTensor[DType.float32, type_of(layout), MutAnyOrigin],
    marker: EscapedName["quoted"],
):
    pass


def test_one() raises:
    pass


def main() raises:
    print(reflect_fn[takes_tile_tensor].linkage_name())
    TestSuite.discover_tests[__functions_in_module()]().run()
