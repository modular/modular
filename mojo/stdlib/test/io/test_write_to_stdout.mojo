# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import sys
from testing import TestSuite


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()


@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("Point(", self.x, ", ", self.y, ")")


# CHECK-LABEL: test_write_to_stdout
def test_write_to_stdout():
    print("== test_write_to_stdout")

    var stdout = sys.stdout

    # CHECK: Hello, World!
    stdout.write("Hello, World!")

    # CHECK: point = Point(1, 1)
    var point = Point(1, 1)
    stdout.write("point = ", point)
