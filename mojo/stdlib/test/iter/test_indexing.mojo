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

from std.testing import TestSuite, assert_equal, assert_true, assert_raises


def test_idx() raises:
    var l = [1, 2, 3]
    var it = iter(l)
    assert_equal(it[0], 1)
    assert_equal(it[1], 2)
    assert_equal(it[2], 3)
    # _ = it[3] # TODO: test oob abort like the `range` iterators

    var l2 = ["hi", "hey", "hello"]
    var it2 = iter(l2)
    assert_equal(it2[0], "hi")
    assert_equal(it2[1], "hey")
    assert_equal(it2[2], "hello")
    # _ = it2[3] # TODO: test oob abort like the `range` iterators


def test_contiguous_slice() raises:
    var data = [0, 1, 2, 3, 4, 5]
    var it = iter(data)
    assert_equal(List(it[0:6]), [0, 1, 2, 3, 4, 5])
    assert_equal(List(it[:]), [0, 1, 2, 3, 4, 5])
    assert_equal(List(it[0:2]), [0, 1])
    assert_equal(List(it[1:3]), [1, 2])
    assert_equal(List(it[1:6]), [1, 2, 3, 4, 5])


def test_strided_slice() raises:
    var data = [0, 1, 2, 3, 4, 5]
    var it = iter(data)
    assert_equal(List(it[0:6:1]), [0, 1, 2, 3, 4, 5])
    assert_equal(List(it[::]), [0, 1, 2, 3, 4, 5])
    assert_equal(List(it[0:2:1]), [0, 1])
    assert_equal(List(it[1:3:1]), [1, 2])
    assert_equal(List(it[1:6:1]), [1, 2, 3, 4, 5])
    assert_equal(List(it[1:6:2]), [1, 3, 5])
    assert_equal(List(it[0:6:2]), [0, 2, 4])
    assert_equal(List(it[0:6:3]), [0, 3])
    assert_equal(List(it[1:6:3]), [1, 4])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
