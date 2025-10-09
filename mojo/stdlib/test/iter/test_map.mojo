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

from test_utils import CopyCounter
from testing import TestSuite, assert_equal


def test_map():
    var l = [1, 2, 3]

    fn add_one(x: Int) -> Int:
        return x + 1

    var m = map[add_one](l)
    assert_equal(next(m), 2)
    assert_equal(next(m), 3)
    assert_equal(next(m), 4)
    assert_equal(m.__has_next__(), False)

    fn to_str(x: Int) -> String:
        return String(x)

    var m2 = map[to_str](l)
    assert_equal(next(m2), "1")
    assert_equal(next(m2), "2")
    assert_equal(next(m2), "3")
    assert_equal(m2.__has_next__(), False)


def test_map_function_can_take_owned_value():
    fn report_copies_owned(var counter: CopyCounter[NoneType]) -> Int:
        return counter.copy_count

    fn report_copies_ref(counter: CopyCounter[NoneType]) -> Int:
        return counter.copy_count

    var list = [CopyCounter(None)]

    # ensure the number of copies are equal between an "owned" and
    # "borrowed" mapping function.
    var m1 = map[report_copies_owned](list)
    var m2 = map[report_copies_ref](list)

    assert_equal(next(m1), next(m2))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
