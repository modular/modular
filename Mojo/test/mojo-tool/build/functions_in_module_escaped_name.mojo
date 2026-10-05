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

# A `LayoutTensor` argument puts `"` in the instantiated function's name, and
# the mangler encodes each one as `~Q` to keep the symbol ELF-safe, so the
# instance name no longer matches the generator's own symbol. Reflection must
# find the generator anyway.

# RUN: %mojo-build %s -o %t
# RUN: %t | FileCheck %s

# CHECK: PASS {{.*}} test_one

from std.testing import TestSuite
from layout import Layout, LayoutTensor

comptime layout = Layout.row_major(128)


def takes_layout_tensor(x: LayoutTensor[DType.float32, layout, MutAnyOrigin]):
    pass


def test_one() raises:
    pass


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
