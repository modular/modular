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

# Note: this code doesn't appear in the doc; it tests the assertions
# in the doc.

from std.testing import assert_equal


@fieldwise_init
struct RegPassableType(RegisterPassable):
    var a: Int
    var b: Int

    def say_hello(self) -> String:
        return "Hello from a register passable type!"


def test_register_passable_type() raises:
    var first: RegPassableType = RegPassableType(42, 24)
    # Ensure that the value is movable
    var second: RegPassableType = first^
    assert_equal(second.say_hello(), "Hello from a register passable type!")


def main() raises:
    test_register_passable_type()
