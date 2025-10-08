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

from testing import TestSuite, assert_equal, assert_false, assert_true


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


@fieldwise_init
struct TestIter(ImplicitlyCopyable, Iterable, Iterator, Movable):
    alias Element = Int
    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self

    var lower: Int
    var upper: Optional[Int]

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        return self.copy()

    fn __has_next__(self) -> Bool:
        return True

    fn __next__(mut self) -> Self.Element:
        return 42

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        return (self.lower, self.upper)


fn test_zip_bounds() raises:
    # same size bounds
    var zipA = zip(TestIter(2, {2}), TestIter(2, {2}))
    assert_equal(zipA.bounds()[0], 2)
    assert_equal(zipA.bounds()[1].value(), 2)

    # different size bounds
    var zipB = zip(TestIter(3, {3}), TestIter(2, {2}), TestIter(1, {1}))
    assert_equal(zipB.bounds()[0], 1)
    assert_equal(zipB.bounds()[1].value(), 1)

    # `None` upper get replaced with discrete upper bound
    var zipC = zip(TestIter(0, None), TestIter(2, {3}))
    assert_equal(zipC.bounds()[0], 0)
    assert_equal(zipC.bounds()[1].value(), 3)

    # Preserves `None` upper if all are none
    var zipD = zip(
        TestIter(1, {None}),
        TestIter(2, {None}),
        TestIter(3, {None}),
        TestIter(4, {None}),
    )
    assert_equal(zipD.bounds()[0], 1)
    assert_false(Bool(zipD.bounds()[1]))


def main():
    var suite = TestSuite()

    suite.test[test_zip2]()
    suite.test[test_zip3]()
    suite.test[test_zip4]()
    suite.test[test_zip_destructure]()
    suite.test[test_zip_unequal_lengths]()
    suite.test[test_zip_bounds]()

    suite^.run()
