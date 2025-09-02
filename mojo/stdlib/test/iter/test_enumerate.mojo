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

from testing import assert_equal, assert_true


fn test_enumerate() raises:
    var l = ["hey", "hi", "hello"]
    var it = enumerate(l)
    var elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], "hey")
    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], "hi")
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], "hello")
    assert_true(not it.__has_next__())


fn test_enumerate_with_start() raises:
    var l = ["hey", "hi", "hello"]
    var it = enumerate(l, start=1)
    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], "hey")
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], "hi")
    elem = next(it)
    assert_equal(elem[0], 3)
    assert_equal(elem[1], "hello")
    assert_true(not it.__has_next__())
    # Check negative start
    it = enumerate(l, start=-1)
    elem = next(it)
    assert_equal(elem[0], -1)
    assert_equal(elem[1], "hey")
    elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], "hi")


fn test_enumerate_destructure() raises:
    var l = ["hey", "hi", "hello"]
    var count = 0
    for i, elem in enumerate(l):
        assert_equal(i, count)
        assert_equal(elem, l[count])
        count += 1


fn main() raises:
    test_enumerate()
    test_enumerate_destructure()
