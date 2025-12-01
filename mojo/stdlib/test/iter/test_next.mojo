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

from testing import TestSuite, assert_equal, assert_true


fn test_next() raises:
    var l = [1, 2, 3]

    var it = iter(l)
    assert_true(it.__has_next__())
    assert_equal(next(it), 1)
    assert_true(it.__has_next__())
    assert_equal(next(it), 2)
    assert_true(it.__has_next__())
    assert_equal(next(it), 3)
    assert_true(not it.__has_next__())
    var l2 = ["hi", "hey", "hello"]
    var it2 = iter(l2)
    assert_true(it2.__has_next__())
    assert_equal(next(it2), "hi")
    assert_true(it2.__has_next__())
    assert_equal(next(it2), "hey")
    assert_true(it2.__has_next__())
    assert_equal(next(it2), "hello")
    assert_true(not it2.__has_next__())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
