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

from std.testing import assert_equal


struct MemoryHolder(ImplicitlyCopyable):
    var p: UnsafePointer[Int, MutExternalOrigin]

    fn __init__(out self):
        self.p = alloc[Int](1)

    fn __is__(self, other: MemoryHolder) -> Bool:
        return self.p == other.p

    fn __isnot__(self, other: MemoryHolder) -> Bool:
        return not self.__is__(other)

    fn __del__(deinit self):
        self.p.destroy_pointee()
        self.p.free()


def main() raises:
    var list = [1, 2, 3]
    assert_equal(
        1 in list, True
    )  # `in` operator checks for membership in the list
    assert_equal(
        4 in list, False
    )  # `in` operator checks for membership in the list
    assert_equal(
        4 not in list, True
    )  # `not` and `in` operators checks for non-membership in the list

    var p1 = MemoryHolder()
    var p2 = MemoryHolder()
    assert_equal(p1 is p2, False)  # two pointer allocations
    assert_equal(p1 is not p2, True)  # two pointer allocations
    assert_equal(p1 is p1, True)  # same pointer allocation

    var s1 = "Hello"
    var s2 = ""  # Strings are Boolable, empty strings are Falsy
    assert_equal(Bool(not s1), False)  # non-empty string is Truthy
    assert_equal(Bool(not s2), True)  # empty string is Falsy
    assert_equal(Bool(s1 and s2), False)
    assert_equal(Bool(s1 and s1), True)
    assert_equal(Bool(s1 or s2), True)
    assert_equal(Bool(s2 or s2), False)

    # Change 0 or 1 to 0 to activate the constraint and error
    def test() -> String where 0 or 1:
        return "This should be compiled, the condition is true"

    assert_equal(test(), "This should be compiled, the condition is true")
