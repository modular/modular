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


fn test_zip2() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var it = zip(l, l2)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)
    assert_true(not it.__has_next__())


fn test_zip_destructure() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var count = 0
    for a, b in zip(l, l2):
        assert_equal(a, l[count])
        assert_equal(b, l2[count])
        count += 1


fn test_zip3() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var l3 = [100, 200, 300]
    var it = zip(l, l2, l3)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)
    assert_equal(elem[2], 300)
    assert_true(not it.__has_next__())


fn test_zip4() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var l3 = [100, 200, 300]
    var l4 = [1000, 2000, 3000]
    var it = zip(l, l2, l3, l4)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)
    assert_equal(elem[3], 1000)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)
    assert_equal(elem[3], 2000)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)
    assert_equal(elem[2], 300)
    assert_true(not it.__has_next__())


fn test_zip_unequal_lengths() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20]
    var it = zip(l, l2)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    assert_true(not it.__has_next__())


fn main() raises:
    test_zip2()
    test_zip3()
    test_zip4()
    test_zip_destructure()
    test_zip_unequal_lengths()
