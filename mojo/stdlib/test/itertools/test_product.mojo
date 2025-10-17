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

from itertools import product
from testing import TestSuite, assert_equal, assert_false, assert_true


def test_product2():
    var l1 = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]

    var it = product(l1, l2)

    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)

    assert_true(not it.__has_next__())


def test_product2_param():
    var trip_count = 0

    @parameter
    for i, j in product(range(2), range(2)):
        assert_true(i in (0, 1))
        assert_true(j in (0, 1))
        trip_count += 1

    assert_equal(trip_count, 4)


def test_product2_unequal():
    """Checks the product if the two input iterators are unequal."""
    var l1 = ["hey", "hi", "hello", "holla"]
    var l2 = [10, 20, 30]

    var it = product(l1, l2)

    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 30)

    assert_true(not it.__has_next__())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
